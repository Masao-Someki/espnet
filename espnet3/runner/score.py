import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from espnet3.inference.abs_metrics import AbsMetrics

from .base import BaseRunner


def read_scp(scp_file: Path) -> Dict[str, str]:
    with open(scp_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    return {
        line.split(maxsplit=1)[0].strip(): line.split(maxsplit=1)[1].strip()
        for line in lines
    }


def validate_scp_files(
    decode_dir: Path,
    test_name: str,
    inputs: Union[List[str], Dict[str, str]],
    file_suffix: str = ".scp",
) -> Dict[str, Dict[str, str]]:
    task_dir = decode_dir / test_name
    assert task_dir.exists(), f"Missing decode output: {task_dir}"

    if isinstance(inputs, list):
        input_map = {k: k for k in inputs}
    else:
        input_map = dict(inputs)

    key_to_data: Dict[str, Dict[str, str]] = {}
    uid_sets: List[set] = []

    for alias, fname in input_map.items():
        file_path = task_dir / f"{fname}{file_suffix}"
        assert file_path.exists(), f"Missing SCP file: {file_path}"

        data = read_scp(file_path)
        assert len(data) > 0, f"No entries in {file_path}"

        key_to_data[alias] = data
        uid_sets.append(set(data.keys()))

    ref_uids = uid_sets[0]
    for uids in uid_sets[1:]:
        assert uids == ref_uids, f"Mismatch in UID sets across inputs, {uids, ref_uids}"

    return key_to_data


def load_scp_fields(
    decode_dir: Path,
    test_name: str,
    inputs: Union[List[str], Dict[str, str]],
    file_suffix: str = ".scp",
) -> Dict[str, List[str]]:
    task_dir = decode_dir / test_name
    assert task_dir.exists(), f"Missing decode output: {task_dir}"

    input_map = {k: k for k in inputs} if isinstance(inputs, list) else dict(inputs)

    data_dicts: Dict[str, Dict[str, str]] = {}
    uid_sets: List[set] = []

    for alias, fname in input_map.items():
        path = task_dir / f"{fname}{file_suffix}"
        assert path.exists(), f"Missing SCP file: {path}"
        d = read_scp(path)
        data_dicts[alias] = d
        uid_sets.append(set(d.keys()))

    ref_uids = uid_sets[0]
    for uids in uid_sets[1:]:
        assert uids == ref_uids, f"UID mismatch: {uids ^ ref_uids}"

    sorted_uids = sorted(ref_uids)
    result = {"utt_id": sorted_uids}
    for alias, d in data_dicts.items():
        result[alias] = [d[uid] for uid in sorted_uids]

    return result


def get_class_path(obj) -> str:
    return f"{obj.__module__}.{obj.__class__.__name__}"
class ScoreRunner(BaseRunner):
    """Evaluate metrics over decode directories."""

    def __init__(
        self,
        config: DictConfig,
        decode_dir: Path,
        parallel: Optional[DictConfig] = None,
    ) -> None:
        super().__init__(dataset_config=config.dataset, parallel=parallel)
        self.config = config
        self.decode_dir = Path(decode_dir)
        self.test_sets = config.dataset.test
        self.metric_cfgs = config.metrics
        self._metric_specs: List[SimpleNamespace] = []
        self._results: Dict[str, Dict[str, Any]] = {}

        for metric_cfg in self.metric_cfgs:
            self._metric_specs.extend(self._expand_metric(metric_cfg))

    def _expand_metric(self, metric_cfg: DictConfig) -> Sequence[SimpleNamespace]:
        if not hasattr(metric_cfg, "_target_"):
            raise KeyError(
                "Metric configuration must have '_target_' field to specify the metric class."
            )
        if not hasattr(metric_cfg, "inputs"):
            raise ValueError(
                f"Metric {metric_cfg._target_} must define 'inputs' field to specify required SCP files."
            )

        inputs = OmegaConf.to_container(metric_cfg.inputs, resolve=True)
        if not isinstance(inputs, (list, dict)):
            raise ValueError("'inputs' must be a list or dict of SCP keys")

        apply_to = metric_cfg.get("apply_to", None)
        applicable_tests = [conf.name for conf in self.test_sets]
        if apply_to is not None:
            applicable_tests = [name for name in applicable_tests if name in apply_to]

        specs: List[SimpleNamespace] = []
        for test_name in applicable_tests:
            validate_scp_files(
                decode_dir=self.decode_dir,
                test_name=test_name,
                inputs=inputs,
                file_suffix=".scp",
            )
            cfg_dict = OmegaConf.to_container(metric_cfg, resolve=False)
            cfg_dict.pop("apply_to", None)
            metric_cfg_copy = OmegaConf.create(cfg_dict)
            specs.append(
                SimpleNamespace(
                    metric_cfg=metric_cfg_copy,
                    inputs=inputs,
                    test_name=test_name,
                )
            )

        return specs

    def iter_tasks(self) -> Iterable[SimpleNamespace]:
        return [SimpleNamespace(name="score")]

    def iter_task_items(self, task: SimpleNamespace) -> Iterable[SimpleNamespace]:
        return self._metric_specs

    def task_description(self, task: SimpleNamespace) -> str:
        return "scoring"

    def on_task_start(self, task: SimpleNamespace) -> None:
        self._results = {}

    def process_item(self, task: SimpleNamespace, spec: SimpleNamespace) -> Any:
        metric = instantiate(spec.metric_cfg)
        if not isinstance(metric, AbsMetrics):
            raise TypeError(f"{type(metric)} is not a valid AbsMetrics instance")

        data = load_scp_fields(
            decode_dir=self.decode_dir,
            test_name=spec.test_name,
            inputs=spec.inputs,
            file_suffix=".scp",
        )
        metric_result = metric(data, spec.test_name, self.decode_dir)
        metric_name = get_class_path(metric)
        return metric_name, spec.test_name, metric_result

    def handle_result(
        self, task: SimpleNamespace, item: SimpleNamespace, result: Any
    ) -> None:
        metric_name, test_name, metric_result = result
        self._results.setdefault(metric_name, {})[test_name] = metric_result

    def on_task_end(self, task: SimpleNamespace) -> None:
        out_path = self.decode_dir / "scores.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self._results, f, indent=2, ensure_ascii=False)

    def run(self) -> Dict[str, Dict[str, Any]]:
        super().run()
        return self._results
