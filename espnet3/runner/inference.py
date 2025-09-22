import logging
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Union

import psutil
import torch

from .base import BaseRunner


@dataclass
class InferenceTask:
    dataset_key: str
    output_dir: Path
    num_items: int


class InferenceRunner(BaseRunner):
    """Runner for test-time inference with optional parallel execution."""

    def iter_tasks(self, decode_dir: Union[str, Path]) -> Iterable[InferenceTask]:
        if self.dataset_config is None or not hasattr(self.dataset_config, "test"):
            raise RuntimeError("dataset_config with test datasets is required")

        decode_dir = Path(decode_dir)
        for test_conf in self.dataset_config.test:
            dataset = self._initialize_dataset(test_conf.name)
            yield InferenceTask(
                dataset_key=test_conf.name,
                output_dir=decode_dir / test_conf.name,
                num_items=len(dataset),
            )

    def iter_task_items(self, task: InferenceTask) -> Iterable[int]:
        return range(task.num_items)

    def task_description(self, task: InferenceTask) -> str:
        return f"inference:{task.dataset_key}"

    def on_task_start(self, task: InferenceTask) -> None:
        task.output_dir.mkdir(parents=True, exist_ok=True)

    def setup_worker_env(self, task: InferenceTask) -> Dict[str, Any]:
        device = self._resolve_device(self.device)
        model = self.initialize_model(device)
        dataset = self.initialize_dataset(task.dataset_key)
        return {
            "model": model,
            "dataset": dataset,
            "proc": psutil.Process(),
        }

    def setup_local_env(self, task: InferenceTask) -> Dict[str, Any]:
        model = self._initialize_model(self.device)
        dataset = self._initialize_dataset(task.dataset_key)
        return {
            "model": model,
            "dataset": dataset,
            "proc": psutil.Process(),
        }

    def process_item(
        self,
        task: InferenceTask,
        index: int,
        *,
        model: Any,
        dataset: Any,
        proc: psutil.Process,
    ) -> Any:
        example = dataset[index]
        uid, sample = self._get_uid_sample(index, example)
        result = self.process_sample_core(
            sample,
            model,
            self.read,
            self.pre_inference,
            self.inference_body,
            self.post_inference,
            self.stream,
            proc,
        )
        return uid, result

    def handle_result(self, task: InferenceTask, item: int, result: Any) -> None:
        uid, payload = result
        if isinstance(payload, dict) and "error" in payload:
            logging.error(f"[{task.dataset_key}:{uid}] Error: {payload['error']}")
        else:
            self.write(uid, payload, task.output_dir)

    def handle_error(self, task: InferenceTask, item: int, exc: Exception) -> None:
        logging.error(f"[{task.dataset_key}:{item}] Error: {exc}")

    # ------------------------------------------------------------------
    # Public helpers retained from the previous implementation
    # ------------------------------------------------------------------
    def run_on_dataset(
        self, dataset_key: str, output_dir: Union[str, Path], async_decode: bool = False
    ) -> None:
        if async_decode:
            self.async_mode = True
        dataset = self._initialize_dataset(dataset_key)
        task = InferenceTask(dataset_key, Path(output_dir), len(dataset))
        self.run_task(task)
        self.async_mode = False

    def run_on_example(self, uid: str, sample: dict) -> dict:
        model = self._initialize_model(self.device)
        if self.stream and "audio_path" in sample:
            sample["stream"] = self.read("audio", sample["audio_path"], stream=True)
        model, sample = self.pre_inference(model, sample)
        with torch.no_grad():
            if self.stream and isinstance(sample.get("stream"), Iterator):
                outputs = [self.inference_body(model, chunk) for chunk in sample["stream"]]
                return self.post_inference(model, outputs)
            return self.inference_body(model, sample)

    def pre_inference(self, model, sample: dict):
        return model, sample

    @abstractmethod
    def inference_body(self, model, sample: Union[dict, Any]) -> dict:
        raise NotImplementedError

    def post_inference(self, model, outputs: list) -> dict:
        return outputs[0]
