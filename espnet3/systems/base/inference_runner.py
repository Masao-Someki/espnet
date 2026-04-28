"""Inference runner with output validation."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp
from typing import Any, Dict, Iterable, List, Sequence

from omegaconf import ListConfig

from espnet3.parallel.base_runner import BaseRunner
from espnet3.utils.writer_utils import write_artifact
from espnet3.parallel.env_provider import EnvironmentProvider


class InferenceRunner(BaseRunner):
    """Inference runner with strict output-format validation.

    This runner implements ``forward`` to call a recipe-provided output
    function. The key names are configurable via ``idx_key`` and
    ``hyp_key``/``ref_key``. ``hyp_key`` and ``ref_key`` may be a single
    string or a list of strings to support multiple hypothesis/reference
    fields. ``idx_key`` is the key used to map each inference result to
    its source dataset index when writing SCP files.

    Output format requirements:
        - The result is a dict with the configured keys plus any extra fields.
        - A sample identifier key must exist under ``idx_key`` so SCP outputs
          can map each result back to the corresponding dataset sample.
        - The sample identifier must be a single value, not a list or tuple.
        - ``hyp_key`` and ``ref_key`` values may be scalars or lists/tuples.
          If lists are returned, each entry is written to its own SCP file
          (e.g., ``hyp0.scp``, ``hyp1.scp``).
    """

    def __init__(
        self,
        provider: EnvironmentProvider,
        idx_key: str = "utt_id",
        hyp_key: str | Sequence[str] = "hyp",
        ref_key: str | Sequence[str] = "ref",
        output_dir: str | Path | None = None,
        output_artifacts: Dict[str, Any] | None = None,
        persist_outputs: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the inference runner with output key settings.

        Args:
            provider: Environment provider that supplies dataset/model/env.
            idx_key: Output dict key used as the sample identifier written in
                the first column of each SCP line. This ties each inference
                result back to its dataset sample. Defaults to ``"utt_id"``.
            hyp_key: Hypothesis key or keys expected in the output dict.
            ref_key: Reference key or keys expected in the output dict.
            output_dir: Final output directory for SCPs and artifacts.
            output_artifacts: Optional artifact writer configuration per field.
            persist_outputs: If true, persist shard outputs instead of returning
                full inference payloads to the driver.
            **kwargs: Forwarded to ``BaseRunner``.
        """
        super().__init__(provider, **kwargs)
        self.idx_key = idx_key
        self.hyp_key = (
            list(hyp_key) if isinstance(hyp_key, (list, tuple, ListConfig)) else hyp_key
        )
        self.ref_key = (
            list(ref_key) if isinstance(ref_key, (list, tuple, ListConfig)) else ref_key
        )
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.output_artifacts = dict(output_artifacts or {})
        self.persist_outputs = persist_outputs
        self._parts_root: Path | None = None

    def resolve_idx_key(self, output: Dict[str, Any]) -> str:
        """Validate that the configured sample-identifier key exists in output."""
        if self.idx_key not in output:
            raise ValueError(
                "Inference output must include the configured sample identifier "
                "key used to map SCP results back to dataset samples. "
                f"idx_key={self.idx_key!r}"
            )
        return self.idx_key

    def _validate_output(self, output: Dict[str, Any]) -> None:
        if not isinstance(output, dict):
            raise TypeError(
                f"Expected dict output, got {type(output).__name__}: {output}"
            )

        hyp_keys = (
            list(self.hyp_key)
            if isinstance(self.hyp_key, (list, tuple))
            else [self.hyp_key]
        )
        ref_keys = (
            list(self.ref_key)
            if isinstance(self.ref_key, (list, tuple))
            else [self.ref_key]
        )
        idx_key = self.resolve_idx_key(output)
        expected = {idx_key, *hyp_keys, *ref_keys}
        actual = set(output.keys())
        missing = expected - actual
        if missing:
            raise ValueError(
                "Inference output keys must include all required keys. "
                f"missing={sorted(missing)}"
            )

        idx_value = output[idx_key]
        if isinstance(idx_value, (list, tuple)):
            raise TypeError(
                f"'{idx_key}' must be a single value, not {type(idx_value).__name__}"
            )

    def build_tasks(
        self, indices: Sequence[int], parallel_config: Dict | None = None
    ) -> List[Any]:
        if not self.persist_outputs:
            return super().build_tasks(indices, parallel_config)

        if self.output_dir is None:
            raise RuntimeError("output_dir must be set when persist_outputs=True.")

        n_workers = 1
        if parallel_config is not None:
            n_workers = max(1, int(getattr(parallel_config, "n_workers", 1)))

        chunks = _split_indices(indices, n_workers)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._parts_root = Path(mkdtemp(prefix=".infer_parts_", dir=self.output_dir))
        return [
            {
                "indices": chunk,
                "part_dir": str(self._parts_root / f"part-{rank:05d}"),
            }
            for rank, chunk in enumerate(chunks)
        ]

    @staticmethod
    def _is_scp_scalar(value) -> bool:
        return isinstance(value, (str, int, float, bool))

    @classmethod
    def _materialize_output_value(
        cls,
        *,
        idx_value,
        field_key: str,
        value,
        output_dir: Path,
        artifact_config: dict | None,
    ):
        if cls._is_scp_scalar(value):
            return value

        if isinstance(value, (list, tuple)):
            raise TypeError(
                f"Top-level list outputs are not supported for '{field_key}'. "
                "Return a single value per field, or wrap structured content in a "
                "dict so it can be saved as JSON."
            )

        artifact_dir = output_dir / field_key
        artifact_dir.mkdir(parents=True, exist_ok=True)
        target = artifact_dir / str(idx_value)
        artifact_path = write_artifact(value, target, field_config=artifact_config)
        return artifact_path.as_posix()

    @staticmethod
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

    def _finalize_persisted_outputs(self, outputs: List[Any]) -> Dict[str, Any]:
        if self.output_dir is None:
            raise RuntimeError("output_dir must be set when persist_outputs=True.")

        fields: List[str] = []
        seen_fields = set()
        part_dirs: List[Path] = []
        total_results = 0
        try:
            for output in outputs:
                if not isinstance(output, dict) or "part_dir" not in output:
                    raise TypeError(
                        "Persisted inference tasks must return shard metadata."
                    )
                part_dirs.append(Path(output["part_dir"]))
                total_results += int(output.get("num_results", 0))
                for field in output.get("fields", []):
                    if field not in seen_fields:
                        seen_fields.add(field)
                        fields.append(field)

            for field in fields:
                self._merge_sorted_text_files(
                    [part_dir / f"{field}.scp" for part_dir in part_dirs],
                    self.output_dir / f"{field}.scp",
                )
        finally:
            if self._parts_root is not None:
                rmtree(self._parts_root, ignore_errors=True)
                self._parts_root = None

        if total_results > 0 and not fields:
            raise RuntimeError("No output keys found in inference results.")

        return {
            "persisted": True,
            "output_keys": fields,
            "num_results": total_results,
            "idx_key": self.idx_key,
        }

    @staticmethod
    def forward(idx, dataset=None, model=None, **kwargs):
        """Run inference for one or more dataset items and return output dict(s).

        Args:
            idx: Integer index or an iterable of integer indices into the dataset.
            dataset: Dataset providing inference entries.
            model: Inference model callable on the configured input.
            **kwargs: Expects ``input_key`` and optionally ``output_fn_path``.

        Returns:
            Dict containing ``idx`` and output fields for a single item, or a list
            of dicts for batched inputs (as returned by ``output_fn``).

        Raises:
            RuntimeError: If required input settings are missing.
            KeyError: If required input keys are missing from the dataset item(s).
            RuntimeError: If batched inference fails; includes guidance to disable
                batching when unsupported.

        Notes:
            - ``input_key`` may be a string or a list/tuple of strings.
            - Batched inputs are passed to the model as lists per key; padding is
              the model's responsibility.

        Examples:
            >>> # Single-item inference
            >>> out = InferenceRunner.forward(
            ...     0, dataset=dataset, model=model,
            ...     input_key="speech", output_fn_path="m.mod.out_fn"
            ... )
            >>> # Batched inference
            >>> out = InferenceRunner.forward(
            ...     [0, 1], dataset=dataset, model=model,
            ...     input_key=["speech", "text"], output_fn_path="m.mod.out_fn"
            ... )
        """
        if isinstance(idx, dict):
            indices = [int(i) for i in idx["indices"]]
            part_dir = Path(idx["part_dir"])
            output_dir = Path(kwargs["output_dir"])
            idx_key = kwargs["idx_key"]
            configured_output_keys = kwargs.get("output_keys")
            artifact_configs = kwargs.get("output_artifacts", {}) or {}
            task_batch_size = kwargs.get("task_batch_size")
            part_dir.mkdir(parents=True, exist_ok=True)
            scp_lines: Dict[str, List[str]] = {}
            fields: List[str] = []
            seen_fields = set()
            num_results = 0
            recursive_kwargs = {k: v for k, v in kwargs.items() if k != "output_dir"}
            batches = (
                [[single_idx] for single_idx in indices]
                if task_batch_size is None
                else [
                    indices[i : i + int(task_batch_size)]
                    for i in range(0, len(indices), int(task_batch_size))
                ]
            )

            for batch_indices in batches:
                run_arg = batch_indices[0] if len(batch_indices) == 1 else batch_indices
                batch_result = InferenceRunner.forward(
                    run_arg,
                    dataset=dataset,
                    model=model,
                    **recursive_kwargs,
                )
                batch_outputs = batch_result if isinstance(batch_result, list) else [batch_result]

                for result in batch_outputs:
                    if not isinstance(result, dict):
                        raise TypeError(
                            f"Expected dict output, got {type(result).__name__}: {result}"
                        )

                    output_keys = (
                        list(configured_output_keys)
                        if configured_output_keys is not None
                        else [key for key in result.keys() if key != idx_key]
                    )
                    if not output_keys:
                        raise RuntimeError("No output keys found in inference results.")
                    if idx_key not in result:
                        raise ValueError(
                            "Inference output must include the configured sample "
                            "identifier key used to map SCP results back to dataset "
                            f"samples. idx_key={idx_key!r}"
                        )

                    idx_value = result[idx_key]
                    if isinstance(idx_value, (list, tuple)):
                        raise TypeError(
                            f"'{idx_key}' must be a single value, not "
                            f"{type(idx_value).__name__}"
                        )
                    num_results += 1

                    for field_key in output_keys:
                        if field_key not in result:
                            raise ValueError(
                                "Inference output keys must include all required keys. "
                                f"missing={[field_key]}"
                            )
                        if field_key not in seen_fields:
                            seen_fields.add(field_key)
                            fields.append(field_key)
                        value = InferenceRunner._materialize_output_value(
                            idx_value=idx_value,
                            field_key=field_key,
                            value=result[field_key],
                            output_dir=output_dir,
                            artifact_config=artifact_configs.get(field_key),
                        )
                        scp_lines.setdefault(field_key, []).append(f"{idx_value} {value}")

            for field_key, lines in scp_lines.items():
                with (part_dir / f"{field_key}.scp").open("w", encoding="utf-8") as f:
                    f.write("\n".join(lines))
                    if lines:
                        f.write("\n")

            return {
                "part_dir": str(part_dir),
                "fields": fields,
                "num_results": num_results,
            }

        if "input_key" not in kwargs:
            raise RuntimeError("input_key must be provided for inference.")
        input_key = kwargs["input_key"]
        output_fn_path = kwargs.get("output_fn_path")
        output_fn = _load_output_fn(output_fn_path) if output_fn_path else None

        keys = (
            list(input_key)
            if isinstance(input_key, (list, tuple, ListConfig))
            else [input_key]
        )

        is_batched = isinstance(idx, (list, tuple))
        if not is_batched:
            data = dataset[idx]
            inputs_dict = {}
            for key in keys:
                if key not in data:
                    raise KeyError(f"Input key '{key}' not found in dataset item.")
                inputs_dict[key] = data[key]
            model_output = model(**inputs_dict)
            if output_fn is None:
                return model_output
            return output_fn(data=data, model_output=model_output, idx=idx)

        indices = list(idx)
        data_batch = [dataset[i] for i in indices]
        inputs_dict = {}
        for key in keys:
            for data in data_batch:
                if key not in data:
                    raise KeyError(f"Input key '{key}' not found in dataset item.")
            inputs_dict[key] = [data[key] for data in data_batch]

        try:
            model_output = model(**inputs_dict)
            if output_fn is None:
                return model_output
            return output_fn(data=data_batch, model_output=model_output, idx=indices)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Batched inference failed. If your model/output_fn does not "
                "support batched inputs, set batch_size to None. "
            ) from exc

    def __call__(self, indices: Iterable[int]) -> List[Any] | None:
        """Run inference and validate output formats."""
        results = super().__call__(indices)
        if self.async_mode:
            return results
        if results is None:
            return None
        if self.persist_outputs:
            return self._finalize_persisted_outputs(results)

        flat_results: List[Any] = []
        for item in results:
            if isinstance(item, list):
                flat_results.extend(item)
            else:
                flat_results.append(item)

        for item in flat_results:
            self._validate_output(item)

        return flat_results


@lru_cache(maxsize=None)
def _load_output_fn(path: str):
    module_path, func_name = path.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, func_name)


def _split_indices(indices: Sequence[int], num_shards: int) -> List[List[int]]:
    if num_shards <= 0:
        raise ValueError("num_shards must be a positive integer")
    if not indices:
        return []
    num_shards = min(num_shards, len(indices))
    q, r = divmod(len(indices), num_shards)
    chunks = []
    start = 0
    for shard_idx in range(num_shards):
        shard_size = q + (1 if shard_idx < r else 0)
        stop = start + shard_size
        chunks.append(list(indices[start:stop]))
        start = stop
    return [chunk for chunk in chunks if chunk]
