import inspect
import warnings
import copy
from contextlib import contextmanager
from typing import Any, Callable, Generator, Iterable, Optional

import torch
from dask.distributed import (
    Client,
    LocalCluster,
    SSHCluster,
    WorkerPlugin,
    as_completed,
)
from dask_jobqueue import (
    HTCondorCluster,
    LSFCluster,
    MoabCluster,
    OARCluster,
    PBSCluster,
    SGECluster,
    SLURMCluster,
)
from omegaconf import DictConfig
from tqdm import tqdm
from typeguard import typechecked

parallel_config: Optional[DictConfig] = None

CLUSTER_MAP = {
    "htcondor": HTCondorCluster,
    "lsf": LSFCluster,
    "moab": MoabCluster,
    "oar": OARCluster,
    "pbs": PBSCluster,
    "sge": SGECluster,
    "slurm": SLURMCluster,
    "ssh": SSHCluster,
}


def make_local_gpu_cluster(n_workers: int, options: dict) -> Client:
    """
    Create a Dask LocalCUDACluster using available GPUs.

    This requires `dask_cuda` package.

    Args:
        n_workers (int): Number of Dask workers (must not exceed number of GPUs).
        options (dict): Additional options for the LocalCUDACluster.

    Returns:
        Client: Dask client connected to the GPU cluster.
    """
    try:
        from dask_cuda import LocalCUDACluster
    except ImportError:
        raise RuntimeError("Please install dask_cuda.")

    num_gpus = torch.cuda.device_count()
    if n_workers > num_gpus:
        raise ValueError(f"n_workers={n_workers} > num_gpus={num_gpus}")
    if n_workers < num_gpus:
        warnings.warn(
            f"n_workers={n_workers} < num_gpus={num_gpus}, some GPUs may be idle."
        )

    cluster = LocalCUDACluster(n_workers=n_workers, **options)
    return Client(cluster)


@typechecked
def set_parallel(config: DictConfig) -> None:
    """Set the global Dask cluster using the provided configuration.

    Args:
        config (DictConfig): Configuration object with 'env' and cluster options.

    Example:
        >>> from omegaconf import OmegaConf
        >>> config = OmegaConf.create({'env': 'local', 'n_workers': 2})
        >>> set_parallel(config)
    """
    global parallel_config
    options = dict(config.options) if hasattr(config, "options") else {}
    config.options = options
    parallel_config = copy.copy(config)


def get_parallel_config() -> Optional[DictConfig]:
    """Return the global Dask cluster configuration."""
    return parallel_config


def _make_client(config: DictConfig = None) -> Client:
    """Create a Dask client tied to the global singleton cluster."""
    set_parallel(config)
    if config.env == "local":
        return Client(LocalCluster(n_workers=config.n_workers, **config.options))

    elif config.env == "local_gpu":
        return make_local_gpu_cluster(config.n_workers, config.options)

    elif config.env == "kube":
        try:
            from dask_kubernetes import KubeCluster
        except ImportError:
            raise RuntimeError("Please install dask_kubernetes.")
        cluster = KubeCluster(**config.options)
        cluster.scale(config.n_workers)
        return Client(cluster)

    elif config.env in CLUSTER_MAP:
        cluster = CLUSTER_MAP[config.env](**config.options)
        cluster.scale(config.n_workers)
        return Client(cluster)

    else:
        raise ValueError(f"Unknown env: {config.env}")


def make_client(config: DictConfig = None) -> Client:
    """
    Create or retrieve a Dask client using the provided or global configuration.

    Args:
        config (DictConfig, optional): Cluster config. If None, uses global one.

    Returns:
        Client: Dask client instance.
    """
    if config is not None:
        return _make_client(config)

    if parallel_config is None:
        raise ValueError(
            "Parallel configuration not set. Use `set_parallel` to set it."
        )

    return _make_client(parallel_config)


class DictReturnWorkerPlugin(WorkerPlugin):
    """
    A WorkerPlugin that calls a user-defined setup function once per worker,
    and stores the returned dictionary in `worker.plugins["env"]`.
    """

    def __init__(self, setup_fn: Callable[[], dict]):
        self.setup_fn = setup_fn

    def setup(self, worker):
        env = self.setup_fn()
        if not isinstance(env, dict):
            raise ValueError("setup_fn must return a dict")
        worker.plugins["env"] = env


def wrap_func_with_worker_env(func: Callable) -> Callable:
    """
    Wrap a user-defined function so that it can transparently consume
    per-worker environment variables registered via a WorkerPlugin.

    This wrapper inspects the function's signature and automatically
    supplies keyword arguments from the worker's environment (`worker.plugins["env"]`)
    when they match parameter names of the function and are not explicitly
    provided by the caller.

    **Conflict detection**:
        If both the worker environment and the call's `kwargs` provide the same
        argument name, the wrapper raises a ``ValueError`` before calling the
        underlying function.

    Args:
        func (Callable):
            The original user-defined function to be executed on the worker.
            It may have positional parameters, keyword parameters, and/or
            a ``**kwargs`` catch-all.

    Returns:
        Callable:
            A wrapped function that:
              1. Runs on the worker.
              2. Retrieves the environment dict from ``worker.plugins["env"]``.
              3. Detects and errors on conflicts with explicit keyword arguments.
              4. Supplies any missing keyword arguments from the environment.

    Raises:
        ValueError:
            If there is at least one parameter name that is present both in the
            worker environment and in the keyword arguments provided to the call.

    Notes:
        - Only environment keys that match the function's parameter names
          (or any keys if the function accepts ``**kwargs``) will be considered
          for injection.

    Example:
        >>> def setup_fn():
        ...     return {"bias": 7}
        ...
        >>> def add_bias(x, bias):
        ...     return x + bias
        ...
        >>> with get_client(local_cfg, setup_fn=setup_fn) as client:
        ...     # 'bias' comes from worker env, no need to pass it explicitly
        ...     futs = client.map(add_bias, [1, 2])
        ...     print(client.gather(futs))
        [8, 9]

        >>> # Passing conflicting 'bias' both in env and kwargs will error:
        >>> with pytest.raises(ValueError):
        ...     client.map(add_bias, [1, 2], bias=5)
    """
    sig = inspect.signature(func)
    param_names = set(sig.parameters.keys())
    accepts_var_kw = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )

    def wrapped(*args, **kwargs):
        from distributed.worker import get_worker

        env = get_worker().plugins.get("env", {})

        kw_keys = set(kwargs.keys())
        considered = kw_keys if accepts_var_kw else (param_names & kw_keys)

        conflict = set(env.keys()) & considered
        if conflict:
            raise ValueError(
                f"Argument conflict: {conflict} passed via both kwargs and env"
            )

        filtered_env = {
            k: v for k, v in env.items() if (k in param_names) and (k not in kwargs)
        }
        return func(*args, **kwargs, **filtered_env)

    return wrapped


def _prime_client_env_keys_from_setup_fn(
    client: Client, setup_fn: Callable[[], dict]
) -> set:
    """Run setup_fn once on the client to cache its keys for pre-submit checks."""
    env = setup_fn()
    if not isinstance(env, dict):
        raise ValueError("setup_fn must return a dict")
    keys = set(env.keys())
    setattr(client, "_env_keys_hint", keys)
    return keys


@contextmanager
def get_client(
    config: DictConfig = None, setup_fn: Optional[Callable[[], dict]] = None
) -> Generator[Client, None, None]:
    """Context manager to yield a Dask client from the global singleton cluster.

    Args:
        config (DictConfig, optional): Cluster config.
        setup_fn (Callable[[], dict], optional): A setup function that runs
            on each worker and returns a dictionary of environment variables.

    Yields:
        Client: A Dask client instance tied to the global cluster.

    Example:
        >>> with get_client() as client:
        ...     results = client.map(lambda x: x**2, range(10))
    """
    client = make_client(config)
    if setup_fn is not None:
        plugin = DictReturnWorkerPlugin(setup_fn)
        reg = getattr(client, "register_worker_plugin", None)
        if reg is None:
            raise RuntimeError(
                "This Dask version lacks register_worker_plugin; please upgrade."
            )
        reg(plugin, name="env")
        _prime_client_env_keys_from_setup_fn(client, setup_fn)
    try:
        yield client
    finally:
        cluster = getattr(client, "cluster", None)
        if not isinstance(cluster, LocalCluster):
            client.shutdown()


def _check_conflict_client_side(
    func: Callable,
    kwargs: dict,
    client: Client,
):
    """
    Perform a pre-submit conflict check between explicit keyword arguments
    and worker environment variables cached on the client.

    This function inspects the signature of `func` and the keys of `kwargs`
    to detect whether any arguments are being passed both:
      - explicitly via `kwargs`, and
      - implicitly via the per-worker environment (as registered by a
        `DictReturnWorkerPlugin`).

    Args:
        func (Callable):
            The function that will be mapped over data.
        kwargs (dict):
            Explicit keyword arguments passed to `func` on submission.
        client (Client):
            A Dask client that may have cached environment keys in
            `_env_keys_hint`.

    Raises:
        ValueError:
            If one or more argument names appear both in the worker environment
            keys and in `kwargs`.

    Notes:
        - This is a **client-side** safeguard; the actual worker-side injection
          and conflict detection is handled separately by
          `wrap_func_with_worker_env`.
        - If `_env_keys_hint` is not present on the client (no environment
          setup function was registered), this function does nothing.

    """
    if not kwargs:
        return

    sig = inspect.signature(func)
    param_names = set(sig.parameters.keys())
    accepts_var_kw = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )

    env_keys = getattr(client, "_env_keys_hint", None)
    if not env_keys:
        return

    kw_keys = set(kwargs.keys())
    considered = kw_keys if accepts_var_kw else (param_names & kw_keys)
    conflict = env_keys & considered
    if conflict:
        raise ValueError(
            f"Argument conflict: {conflict} passed via both kwargs and env"
        )


@typechecked
def parallel_map(
    func: Callable[[Any], Any],
    data: Iterable[Any],
    client: Optional[Client] = None,
    setup_fn: Optional[Callable[[], dict]] = None,
    **kwargs: Any,
) -> list:
    """
    Apply a function to an iterable of inputs in parallel using Dask.

    This helper takes care of:
      - Creating (or reusing) a Dask client according to the global or
        provided configuration.
      - Optionally registering a per-worker environment via `setup_fn`,
        making its returned dictionary available to `func` automatically.
      - Detecting and preventing conflicts between explicit `kwargs` and
        environment-provided arguments on the **client side** before
        submitting tasks.
      - Wrapping `func` with `wrap_func_with_worker_env` so that missing
        keyword arguments can be injected from the worker environment.

    Args:
        func (Callable[[Any], Any]):
            The function to execute on each element of `data`. May take
            positional and/or keyword parameters.
        data (Iterable[Any]):
            Iterable of input elements to process.
        client (Optional[Client], default=None):
            An existing Dask client to use. If `None`, a temporary client
            will be created using `get_client` and shut down afterwards.
        setup_fn (Optional[Callable[[], dict]], default=None):
            A function run once per worker that returns a dictionary of
            environment variables. These variables are automatically
            injected into `func` if they match parameter names and are not
            explicitly provided.
        **kwargs:
            Additional keyword arguments to pass directly to `func` for all
            elements.

    Returns:
        list:
            The results of applying `func` to each element of `data`, in
            order. The list has the same length as `data`.

    Raises:
        ValueError:
            If any keyword argument name in `kwargs` is also present in the
            worker environment keys (conflict detected before submission).

    Example:
        >>> def setup_fn():
        ...     return {"bias": 10}
        >>> def add_bias(x, bias):
        ...     return x + bias
        >>> # Automatic injection of 'bias' from worker environment:
        >>> results = parallel_map(add_bias, [1, 2, 3], setup_fn=setup_fn)
        >>> results
        [11, 12, 13]
    """
    internal = client is None
    if internal:
        # respect global config and local/remote shutdown policy
        ctx = get_client(setup_fn=setup_fn)
        client_cm = ctx  # keep reference for exit
        client = ctx.__enter__()

    if setup_fn is not None:
        plugin = DictReturnWorkerPlugin(setup_fn)
        getattr(client, "register_worker_plugin")(plugin, name="env")
        _prime_client_env_keys_from_setup_fn(client, setup_fn)

    try:
        _check_conflict_client_side(func, kwargs, client)
        wrapped_func = wrap_func_with_worker_env(func)
        futures = client.map(wrapped_func, data, **kwargs)
        return list(tqdm(client.gather(futures), total=len(futures)))
    finally:
        if internal:
            client_cm.__exit__(None, None, None)


def parallel_for(
    func: Callable,
    args: Iterable,
    client: Optional[Client] = None,
    setup_fn: Optional[Callable[[], dict]] = None,
    **kwargs: Any,
) -> Generator:
    """
    Dispatch tasks to Dask and iterate over results as they complete.

    This helper:
      - Creates (or reuses) a Dask client based on the global/explicit config.
      - Optionally registers a per-worker environment via `setup_fn` and makes
        its returned dict available to `func` automatically.
      - Performs a **client-side** conflict check so that keyword arguments
        explicitly passed via `kwargs` don't collide with environment-provided
        arguments (pre-submission).
      - Wraps `func` with `wrap_func_with_worker_env` so any missing keyword
        parameters can be injected from the worker environment.

    Iteration order:
        Results are yielded in **completion order** (using `as_completed`),
        not in the original order of `args`.

    Args:
        func:
            The function to run on each element of `args`. It may accept
            positional and/or keyword parameters.
        args:
            Iterable of inputs to `func`.
        client (optional):
            Existing Dask `Client`. If `None`, a temporary client is created
            via `get_client` and shut down automatically when iteration ends.
        setup_fn (optional):
            A callable executed once per worker that returns a `dict` of
            environment variables. Keys that match parameter names of `func`
            (or any keys if `func` accepts `**kwargs`) are auto-injected
            unless explicitly provided in `kwargs`.
        **kwargs:
            Extra keyword arguments forwarded to every call of `func`.

    Yields:
        Each task's result as soon as it finishes.

    Raises:
        ValueError:
            If at least one key in `kwargs` conflicts with keys provided by
            the worker environment (pre-submit check).
        Exception:
            Any exception raised by `func` is re-raised at iteration time when
            accessing `future.result()` for the failing task.

    Notes:
        - This generator will close the internally created client once
          iteration finishes or the generator is exhausted.
        - Worker-side injection and an additional conflict check are also
          enforced by `wrap_func_with_worker_env`.

    Example:
        >>> def setup_fn():
        ...     return {"bias": 3}
        >>> def add_bias(x, bias):
        ...     return x + bias
        >>> # Stream results as tasks complete (completion order):
        >>> for y in parallel_for(add_bias, [1, 2, 3], setup_fn=setup_fn):
        ...     print(y)
        4
        5
        6
    """
    internal = client is None
    if internal:
        # respect global config and local/remote shutdown policy
        ctx = get_client(setup_fn=setup_fn)
        client_cm = ctx  # keep reference for exit
        client = ctx.__enter__()

    if setup_fn is not None:
        plugin = DictReturnWorkerPlugin(setup_fn)
        getattr(client, "register_worker_plugin")(plugin, name="env")
        _prime_client_env_keys_from_setup_fn(client, setup_fn)

    try:
        _check_conflict_client_side(func, kwargs, client)
        wrapped_func = wrap_func_with_worker_env(func)
        futures = client.map(wrapped_func, args, **kwargs)
        for future in as_completed(futures):
            yield future.result()
    finally:
        if internal:
            client_cm.__exit__(None, None, None)
