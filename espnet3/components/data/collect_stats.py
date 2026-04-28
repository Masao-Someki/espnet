"""Collect statistics over a dataset using a model's feature extraction."""

from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.fileio.npy_scp import NpyScpWriter
from espnet2.train.collate_fn import CommonCollateFn
from espnet3.parallel.base_runner import BaseRunner
from espnet3.parallel.env_provider import EnvironmentProvider
from espnet3.parallel.parallel import set_parallel
from espnet3.utils.task_utils import get_espnet_model

__all__ = [
    "CollectStatsInferenceProvider",
    "CollectStatsRunner",
    "collect_stats",
    "collect_stats_batch",
]


def collect_stats_batch(
    idxs: List[int],
    model=None,
    dataset=None,
    collate_fn=None,
    device: Optional[torch.device] = None,
    write_collected_feats: bool = False,
    collect_stats_kwargs: Optional[Dict[str, Any]] = None,
):
    """Process a batch of dataset indices and compute feature statistics."""
    structured_items: List[Tuple[str, Any]] = []
    for i in idxs:
        item = dataset[i]
        # We assume dataset should be DataOrganizer in espnet3.
        if (
            hasattr(dataset, "use_espnet_preprocessor")
            and dataset.use_espnet_preprocessor
        ):
            # Then it is a tuple with (uid, dict) and type(uid) is str.
            uid, sample = item
        else:
            uid, sample = str(i), item
        structured_items.append((uid, sample))

    batch = collate_fn(structured_items)
    if not isinstance(batch, Sequence) or len(batch) != 2:
        raise RuntimeError(
            "collect_stats expects the collate function to return (uids, batch_dict)."
        )

    uids, features = batch  # type: ignore[misc]
    if not isinstance(features, dict):
        raise RuntimeError(
            "collect_stats expects collate_fn to return a mapping for batch tensors."
        )

    tensors = {k: v.to(device) for k, v in features.items()}

    extra_kwargs = dict(collect_stats_kwargs or {})
    conflict = set(extra_kwargs).intersection(tensors)
    if conflict:
        raise ValueError(
            "collect_stats kwargs conflict with batch tensors: " + ", ".join(conflict)
        )

    with torch.no_grad():
        feats = model.collect_feats(**{**tensors, **extra_kwargs})

    feats = {
        k: (v.detach().cpu().numpy() if hasattr(v, "detach") else v)
        for k, v in feats.items()
    }

    stats = defaultdict(lambda: {"sum": 0, "sq": 0, "count": 0})
    shape_info = defaultdict(dict)

    for b_idx, uid in enumerate(list(uids)):
        for feat_key in list(feats.keys()):
            if f"{feat_key}_lengths" in feats:
                length = int(feats[f"{feat_key}_lengths"][b_idx])
                seq = feats[feat_key][b_idx][:length]
            else:
                seq = feats[feat_key][b_idx][None]

            stats[feat_key]["sum"] += seq.sum(0)
            stats[feat_key]["sq"] += (seq**2).sum(0)
            stats[feat_key]["count"] += len(seq)
            shape_info[feat_key][uid] = ",".join(map(str, seq.shape))

    if write_collected_feats:
        return stats, shape_info, feats
    else:
        return stats, shape_info


def _accumulate_and_persist_batch(
    stats: Dict,
    shape_info: Dict,
    feats: Optional[Dict],
    sum_dict: Dict,
    sq_dict: Dict,
    count_dict: Dict,
    datadir_writer: DatadirWriter,
    writers: Dict,
    mode: str,
    collected_feats_data_root: Path,
    collected_feats_scp_root: Path,
    write_collected_feats: bool,
    shape_key_suffix: str = "",
):
    """Merge per-batch stats and persist shapes/features."""
    for feat_key, agg in stats.items():
        sum_dict[feat_key] += agg["sum"]
        sq_dict[feat_key] += agg["sq"]
        count_dict[feat_key] += agg["count"]

    for feat_key, uid2shape in shape_info.items():
        shape_key = f"{feat_key}_shape{shape_key_suffix}"
        for uid, shape_str in uid2shape.items():
            datadir_writer[shape_key][uid] = shape_str

        if write_collected_feats and feats is not None and feat_key in feats:
            uids_in_order = list(uid2shape.keys())
            feat_batch = feats[feat_key]
            len_key = f"{feat_key}_lengths"
            len_batch = feats.get(len_key, None)

            writer_key = (feat_key, mode)
            if writer_key not in writers:
                data_dir = (
                    collected_feats_data_root
                    / mode
                    / "collect_feats"
                    / f"data_{feat_key}"
                )
                scp_dir = collected_feats_scp_root / mode / "collect_feats"
                writers[writer_key] = NpyScpWriter(
                    data_dir, scp_dir / f"{feat_key}.scp"
                )
            w = writers[writer_key]

            for b_idx, uid in enumerate(uids_in_order):
                seq = feat_batch[b_idx]
                if len_batch is not None:
                    L = int(len_batch[b_idx])
                    seq = seq[:L]
                else:
                    seq = seq[None]

                if not isinstance(seq, np.ndarray):
                    seq = np.asarray(seq)
                w[uid] = seq


def _build_collate_fn(dataloader_config):
    if not isinstance(dataloader_config, DictConfig):
        dataloader_config = (
            OmegaConf.create(dataloader_config)
            if dataloader_config is not None
            else OmegaConf.create({})
        )

    if (
        hasattr(dataloader_config, "collate_fn")
        and dataloader_config.collate_fn is not None
    ):
        return instantiate(dataloader_config.collate_fn)
    else:
        return CommonCollateFn(int_pad_value=-1)


def _build_dataset(config: DictConfig):
    dataset = _instantiate_dataset(config.dataset_config, config.mode)
    shard_idx = config.get("shard_idx")
    if shard_idx is not None:
        if not hasattr(dataset, "shard"):
            raise RuntimeError("Dataset does not support sharding")
        dataset = dataset.shard(shard_idx)

    if hasattr(dataset, "use_espnet_collator"):
        dataset.use_espnet_collator = True
    return dataset


def _build_model(config: DictConfig):
    model_config = config.model_config
    if not isinstance(model_config, DictConfig):
        model_config = OmegaConf.create(model_config)
    task = config.get("task")
    if task:
        model = get_espnet_model(task, model_config)
    else:
        model = instantiate(model_config)

    collect_fn = getattr(model, "collect_feats", None)
    if collect_fn is None or not callable(collect_fn):
        raise AttributeError(
            "Model is missing required callable 'collect_feats' method."
        )
    return model


def _chunk_indices(num_items: int, batch_size: int) -> List[List[int]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    batches = [
        list(range(i, min(i + batch_size, num_items)))
        for i in range(0, num_items, batch_size)
    ]
    return [b for b in batches if b]


def _chunk_values(values: Sequence[int], batch_size: int) -> List[List[int]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    return [
        list(values[i : i + batch_size]) for i in range(0, len(values), batch_size)
    ]


def _split_indices(num_items: int, num_shards: int) -> List[List[int]]:
    if num_shards <= 0:
        raise ValueError("num_shards must be a positive integer")
    if num_items <= 0:
        return []

    num_shards = min(num_shards, num_items)
    q, r = divmod(num_items, num_shards)
    shards = []
    start = 0
    for shard_idx in range(num_shards):
        shard_size = q + (1 if shard_idx < r else 0)
        stop = start + shard_size
        shards.append(list(range(start, stop)))
        start = stop
    return [shard for shard in shards if shard]


def _instantiate_dataset(dataset_config, mode: str):
    if not isinstance(dataset_config, DictConfig):
        dataset_config = OmegaConf.create(dataset_config)

    organizer = instantiate(dataset_config)
    dataset = getattr(organizer, mode, None)
    if dataset is None:
        raise ValueError(f"Dataset organizer does not provide split '{mode}'")
    return dataset


def _get_dataset_length(
    dataset_config, mode: str, shard_idx: Optional[int] = None
) -> int:
    dataset = _instantiate_dataset(dataset_config, mode)
    if shard_idx is not None:
        if not hasattr(dataset, "shard"):
            raise RuntimeError("Dataset does not support sharding")
        dataset = dataset.shard(shard_idx)
    return len(dataset)


class CollectStatsInferenceProvider(EnvironmentProvider):
    """EnvironmentProvider tailored for collect-stats jobs."""

    def __init__(
        self,
        model_config,
        dataset_config,
        dataloader_config,
        mode: str,
        task: Optional[str] = None,
        shard_idx: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        """Initialize CollectStatsInferenceProvider object."""
        config = OmegaConf.create({})
        config.model_config = model_config
        config.dataset_config = dataset_config
        config.dataloader_config = dataloader_config
        config.mode = mode
        config.task = task
        config.shard_idx = shard_idx
        config.update(**(params or {}))
        super().__init__(config)

    def build_env_local(self) -> Dict[str, Any]:
        """Build the environment once on the driver for local inference."""
        env = dict()
        collate_fn = _build_collate_fn(self.config.dataloader_config)
        env["collate_fn"] = collate_fn

        dataset = _build_dataset(self.config)
        if hasattr(dataset, "use_espnet_collator"):
            dataset.use_espnet_collator = isinstance(collate_fn, CommonCollateFn)

        env["dataset"] = dataset

        device = env.get("device")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        env["device"] = device

        env["model"] = _build_model(self.config).to(device).eval()
        env["write_collected_feats"] = self.config.write_collected_feats
        env["output_dir"] = self.config.output_dir
        env["batch_size"] = self.config.batch_size
        env["shape_key_suffix"] = self.config.shape_key_suffix
        env["collect_stats_kwargs"] = self.config.get("collect_stats_kwargs")
        env["mode"] = self.config.mode
        return env

    def build_worker_setup_fn(self):
        """Return a Dask worker setup function that builds dataset/model."""
        dataloader_config = self.config.dataloader_config
        config = self.config

        def setup():
            env = dict()
            collate_fn = _build_collate_fn(dataloader_config)
            env["collate_fn"] = collate_fn

            dataset = _build_dataset(config)
            if hasattr(dataset, "use_espnet_collator"):
                dataset.use_espnet_collator = isinstance(collate_fn, CommonCollateFn)
            env["dataset"] = dataset

            device = env.get("device")
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            env["device"] = device

            env["model"] = _build_model(config).to(device).eval()
            env["write_collected_feats"] = self.config.write_collected_feats
            env["output_dir"] = self.config.output_dir
            env["batch_size"] = self.config.batch_size
            env["shape_key_suffix"] = self.config.shape_key_suffix
            env["collect_stats_kwargs"] = self.config.get("collect_stats_kwargs")
            env["mode"] = self.config.mode
            return env

        return setup


class CollectStatsRunner(BaseRunner):
    """Runner that executes collect-stats over batches of indices."""

    @staticmethod
    def forward(
        batch_indices: Dict[str, Any] | Iterable[int] | int,
        dataset,
        model,
        collate_fn,
        device,
        output_dir,
        mode,
        batch_size,
        write_collected_feats: bool = False,
        shape_key_suffix: str = "",
        collect_stats_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Process a collect-stats task and persist shard outputs."""
        if isinstance(batch_indices, dict):
            indices = [int(i) for i in batch_indices["indices"]]
            part_dir = Path(batch_indices["part_dir"])
            part_mode_dir = part_dir / mode
            part_mode_dir.mkdir(parents=True, exist_ok=True)

            sum_dict = defaultdict(lambda: 0)
            sq_dict = defaultdict(lambda: 0)
            count_dict = defaultdict(lambda: 0)
            writers = {}

            try:
                with DatadirWriter(part_mode_dir) as datadir_writer:
                    for inner_batch_indices in _chunk_values(indices, batch_size):
                        result = collect_stats_batch(
                            inner_batch_indices,
                            model=model,
                            dataset=dataset,
                            collate_fn=collate_fn,
                            device=device,
                            write_collected_feats=write_collected_feats,
                            collect_stats_kwargs=collect_stats_kwargs,
                        )
                        if write_collected_feats:
                            stats, shape_info, feats = result
                        else:
                            stats, shape_info = result
                            feats = None

                        _accumulate_and_persist_batch(
                            stats=stats,
                            shape_info=shape_info,
                            feats=feats,
                            sum_dict=sum_dict,
                            sq_dict=sq_dict,
                            count_dict=count_dict,
                            datadir_writer=datadir_writer,
                            writers=writers,
                            mode=mode,
                            collected_feats_data_root=Path(output_dir),
                            collected_feats_scp_root=part_dir,
                            write_collected_feats=write_collected_feats,
                            shape_key_suffix=shape_key_suffix,
                        )
            finally:
                for writer in writers.values():
                    writer.close()

            keys = list(sum_dict.keys())
            _write_stats_npz_files(part_mode_dir, sum_dict, sq_dict, count_dict)
            _write_key_file(part_mode_dir / "batch_keys", keys)
            _write_key_file(part_mode_dir / "stats_keys", keys)
            return {"part_dir": str(part_dir), "keys": keys}

        if isinstance(batch_indices, Iterable) and not isinstance(
            batch_indices, (str, bytes)
        ):
            indices = [int(i) for i in batch_indices]
        else:
            indices = [int(batch_indices)]

        return collect_stats_batch(
            indices,
            model=model,
            dataset=dataset,
            collate_fn=collate_fn,
            device=device,
            write_collected_feats=write_collected_feats,
            collect_stats_kwargs=collect_stats_kwargs,
        )


def _write_key_file(path: Path, keys: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        if keys:
            f.write("\n".join(keys) + "\n")


def _read_key_file(path: Path) -> List[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _write_stats_npz_files(
    mode_dir: Path, sum_dict: Dict, sq_dict: Dict, count_dict: Dict
) -> None:
    mode_dir.mkdir(parents=True, exist_ok=True)
    for key in sum_dict:
        np.savez(
            mode_dir / f"{key}_stats.npz",
            count=count_dict[key],
            sum=sum_dict[key],
            sum_square=sq_dict[key],
        )


def _merge_sorted_text_files(input_paths: Sequence[Path], output_path: Path) -> None:
    lines = []
    for input_path in input_paths:
        if not input_path.exists():
            continue
        with input_path.open("r", encoding="utf-8") as f:
            lines.extend([line for line in f if line.strip()])

    with output_path.open("w", encoding="utf-8") as f:
        for line in sorted(lines, key=lambda x: x.split()[0]):
            f.write(line)


def _merge_shard_outputs(
    shard_dirs: Sequence[Path],
    output_dir: Path,
    mode: str,
    write_collected_feats: bool,
) -> None:
    mode_dir = output_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    batch_keys: List[str] = []
    stats_keys: List[str] = []
    seen_batch_keys = set()
    seen_stats_keys = set()

    for shard_dir in shard_dirs:
        shard_mode_dir = shard_dir / mode
        for key in _read_key_file(shard_mode_dir / "batch_keys"):
            if key not in seen_batch_keys:
                seen_batch_keys.add(key)
                batch_keys.append(key)
        for key in _read_key_file(shard_mode_dir / "stats_keys"):
            if key not in seen_stats_keys:
                seen_stats_keys.add(key)
                stats_keys.append(key)

    for key in batch_keys:
        _merge_sorted_text_files(
            [shard_dir / mode / f"{key}_shape" for shard_dir in shard_dirs],
            mode_dir / f"{key}_shape",
        )

    for key in stats_keys:
        merged_stats = None
        for shard_dir in shard_dirs:
            stats_path = shard_dir / mode / f"{key}_stats.npz"
            if not stats_path.exists():
                continue
            stats = np.load(stats_path)
            try:
                if merged_stats is None:
                    merged_stats = {name: np.array(stats[name]) for name in stats}
                else:
                    for name in stats:
                        merged_stats[name] += stats[name]
            finally:
                stats.close()

        if merged_stats is not None:
            np.savez(mode_dir / f"{key}_stats.npz", **merged_stats)

    if write_collected_feats:
        collect_feats_dir = mode_dir / "collect_feats"
        collect_feats_dir.mkdir(parents=True, exist_ok=True)
        for key in stats_keys:
            _merge_sorted_text_files(
                [
                    shard_dir / mode / "collect_feats" / f"{key}.scp"
                    for shard_dir in shard_dirs
                ],
                collect_feats_dir / f"{key}.scp",
            )

    _write_key_file(mode_dir / "batch_keys", batch_keys)
    _write_key_file(mode_dir / "stats_keys", stats_keys)


def _collect_stats_common(
    model_config,
    dataset_config,
    dataloader_config,
    mode: str,
    output_dir: Path,
    task: Optional[str],
    parallel_config: Optional[DictConfig],
    write_collected_feats: bool,
    batch_size: int,
    shard_idx: Optional[int] = None,
    shape_key_suffix: str = "",
):
    num_items = _get_dataset_length(dataset_config, mode, shard_idx)
    num_shards = 1
    if parallel_config is not None:
        num_shards = max(1, int(getattr(parallel_config, "n_workers", 1)))
    index_shards = _split_indices(num_items, num_shards)
    output_dir.mkdir(parents=True, exist_ok=True)
    parts_root = Path(mkdtemp(prefix=f".collect_stats_{mode}_", dir=output_dir))
    shard_tasks = [
        {
            "indices": shard_indices,
            "part_dir": str(parts_root / f"part-{shard_idx:05d}"),
        }
        for shard_idx, shard_indices in enumerate(index_shards)
    ]

    provider = CollectStatsInferenceProvider(
        model_config=model_config,
        dataset_config=dataset_config,
        dataloader_config=dataloader_config,
        mode=mode,
        task=task,
        shard_idx=shard_idx,
        params={
            "write_collected_feats": write_collected_feats,
            "output_dir": str(output_dir),
            "batch_size": batch_size,
            "shape_key_suffix": shape_key_suffix,
        },
    )
    runner = CollectStatsRunner(provider)

    try:
        if (
            parallel_config is not None
            and getattr(parallel_config, "env", "local") != "local"
        ):
            runner._run_parallel(shard_tasks)
        else:
            runner._run_local(shard_tasks)

        _merge_shard_outputs(
            [Path(task_info["part_dir"]) for task_info in shard_tasks],
            output_dir=output_dir,
            mode=mode,
            write_collected_feats=write_collected_feats,
        )
    finally:
        shutil.rmtree(parts_root, ignore_errors=True)


def collect_stats(
    model_config,
    dataset_config,
    dataloader_config,
    mode: str,
    output_dir: Path,
    task: Optional[str] = None,
    parallel_config: Optional[DictConfig] = None,
    write_collected_feats: bool = False,
    batch_size: int = 4,
):
    """Entry point for collecting dataset statistics used for feature normalization.

    Runs the runner-based collection once, optionally configuring parallel
    execution via :func:`espnet3.parallel.set_parallel` when ``parallel_config``
    is provided.

    Args:
        model_config: Configuration object used to instantiate the model that
            extracts features from the input examples.
        dataset_config: Configuration of the dataset organizer providing the
            split specified by ``mode``.
        dataloader_config: Dataloader configuration.
        mode: Name of the dataset split to process (``train`` or ``valid``).
        output_dir: Directory where aggregated statistics and optionally
            collected features are written.
        task: Name of the ESPnet task. If ``None``, ``model_config`` should be
            directly instantiable.
        parallel_config: Configuration for parallel execution.
        write_collected_feats: Whether to persist the raw collected features.
        batch_size: Number of dataset items processed per batch.

    Returns:
        None: Aggregated statistics are saved under ``output_dir / mode``.
    """
    mode_config = getattr(dataloader_config, mode, None)
    if mode_config is not None and hasattr(mode_config, "multiple_iterator"):
        raise RuntimeError(
            "ESPnet3 does not support multiple_iterator. "
            "If you need sharding, select a shard explicitly "
            "(e.g., point the dataset/shape files to split.*) "
            "and run collect_stats on that shard."
        )
    if parallel_config is not None:
        set_parallel(parallel_config)
    _collect_stats_common(
        model_config=model_config,
        dataset_config=dataset_config,
        dataloader_config=dataloader_config,
        mode=mode,
        output_dir=output_dir,
        task=task,
        parallel_config=parallel_config,
        write_collected_feats=write_collected_feats,
        batch_size=batch_size,
    )
