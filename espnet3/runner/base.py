import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import psutil
import soundfile as sf
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from espnet2.train.preprocessor import AbsPreprocessor
from espnet3.data.data_organizer import do_nothing_transform
from espnet3.data.dataset import DatasetWithTransform
from espnet3.parallel import parallel_for, parallel_map, set_parallel

logging.basicConfig(level=logging.INFO)


def read_audio(path: str) -> np.ndarray:
    wav, _ = sf.read(path)
    return wav


def stream_audio(path: str, chunk_sec: float = 0.01) -> Iterator[np.ndarray]:
    with sf.SoundFile(path, "r") as f:
        sr = f.samplerate
        frames_per_chunk = int(sr * chunk_sec)
        while True:
            chunk = f.read(frames_per_chunk, dtype="float32")
            if len(chunk) == 0:
                break
            yield chunk


def read_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def stream_text(path: str, chunk_chars: int = 5) -> Iterator[str]:
    text = read_text(path)
    for i in range(0, len(text), chunk_chars):
        yield text[i : i + chunk_chars]


def measure_resources(proc: psutil.Process) -> Tuple[float, float]:
    mem = proc.memory_info().rss / 1024 / 1024
    gpu = (
        torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    )
    return mem, gpu


class BaseRunner(ABC):
    """Common orchestration for runner implementations."""

    def __init__(
        self,
        model_config: Optional[DictConfig] = None,
        dataset_config: Optional[DictConfig] = None,
        stream: bool = False,
        parallel: Optional[DictConfig] = None,
        async_mode: bool = False,
    ) -> None:
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.stream = stream
        self.parallel_config = parallel
        self.async_mode = async_mode

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.dataset = None
        self.current_dataset_key: Optional[str] = None

    # ------------------------------------------------------------------
    # Task orchestration
    # ------------------------------------------------------------------
    def run(self, *args: Any, **kwargs: Any) -> None:
        for task in self.iter_tasks(*args, **kwargs):
            self.run_task(task)

    def iter_tasks(self, *args: Any, **kwargs: Any) -> Iterable[Any]:
        raise NotImplementedError

    def run_task(self, task: Any) -> None:
        self.on_task_start(task)

        items = list(self.iter_task_items(task))
        if not items:
            self.on_task_end(task)
            return

        if self.parallel_config is not None:
            set_parallel(self.parallel_config)
            setup_fn = (
                (lambda task=task: self.setup_worker_env(task))
                if self.setup_worker_env is not None
                else None
            )

            def wrapper(item, **env):
                try:
                    return item, self.process_item(task, item, **env)
                except Exception as exc:  # pragma: no cover - safety net
                    return item, exc

            if self.async_mode:
                results = parallel_for(wrapper, items, setup_fn=setup_fn)
            else:
                results = parallel_map(wrapper, items, setup_fn=setup_fn)

            for item, result in results:
                self._handle_result(task, item, result)
        else:
            env = self.setup_local_env(task)
            for item in tqdm(items, desc=self.task_description(task)):
                try:
                    result = self.process_item(task, item, **env)
                except Exception as exc:  # pragma: no cover - safety net
                    self.handle_error(task, item, exc)
                else:
                    self.handle_result(task, item, result)

        self.on_task_end(task)

    def iter_task_items(self, task: Any) -> Iterable[Any]:
        raise NotImplementedError

    def setup_worker_env(self, task: Any) -> Dict[str, Any]:  # pragma: no cover - hook
        return {}

    def setup_local_env(self, task: Any) -> Dict[str, Any]:  # pragma: no cover - hook
        return self.setup_worker_env(task)

    @abstractmethod
    def process_item(self, task: Any, item: Any, **env: Any) -> Any:
        raise NotImplementedError

    def handle_result(self, task: Any, item: Any, result: Any) -> None:
        pass

    def handle_error(self, task: Any, item: Any, exc: Exception) -> None:
        raise exc

    def task_description(self, task: Any) -> str:
        return str(task)

    def on_task_start(self, task: Any) -> None:
        pass

    def on_task_end(self, task: Any) -> None:
        pass

    def _handle_result(self, task: Any, item: Any, result: Any) -> None:
        if isinstance(result, Exception):
            self.handle_error(task, item, result)
        else:
            self.handle_result(task, item, result)

    # ------------------------------------------------------------------
    # Dataset/model helpers shared across runners
    # ------------------------------------------------------------------
    def initialize_model(self, device: Optional[str] = None):
        return instantiate(self.model_config)

    def _initialize_model(self, device: Optional[str] = None):
        device = self._resolve_device(device)
        if self.model is None:
            self.model = self.initialize_model(device)
        return self.model

    def _resolve_device(self, device: Optional[str]) -> Optional[str]:
        if device == "cuda" and self.parallel_config is not None:
            if getattr(self.parallel_config, "env", None) == "local_gpu":
                cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                if cuda_visible:
                    first = cuda_visible.split(",")[0]
                    return f"cuda:{first}"
        return device

    def initialize_dataset(
        self, dataset_key: str, dataset_config: Optional[DictConfig] = None
    ) -> DatasetWithTransform:
        if dataset_config is None:
            dataset_config = self.dataset_config

        if dataset_config is None or not hasattr(dataset_config, "test"):
            raise RuntimeError("Dataset configuration with 'test' field is required")

        test_ds_conf = None
        for ds_conf in dataset_config.test:
            if ds_conf.name == dataset_key:
                test_ds_conf = ds_conf
                break

        if test_ds_conf is None:
            raise RuntimeError(f"{dataset_key} not found in inference config.")

        if hasattr(dataset_config, "preprocessor"):
            preprocessor = instantiate(dataset_config.preprocessor)
        else:
            preprocessor = do_nothing_transform

        is_espnet_preprocessor = isinstance(preprocessor, AbsPreprocessor)

        if hasattr(test_ds_conf, "transform"):
            transform = instantiate(test_ds_conf.transform)
        else:
            transform = do_nothing_transform

        return DatasetWithTransform(
            instantiate(test_ds_conf.dataset),
            transform,
            preprocessor,
            use_espnet_preprocessor=is_espnet_preprocessor,
        )

    def _initialize_dataset(
        self, dataset_key: str, dataset_config: Optional[DictConfig] = None
    ):
        if self.current_dataset_key != dataset_key:
            self.dataset = self.initialize_dataset(dataset_key, dataset_config)
            self.current_dataset_key = dataset_key
        return self.dataset

    # ------------------------------------------------------------------
    # Sample utilities shared with inference runners
    # ------------------------------------------------------------------
    def read(
        self,
        input_type: str,
        path: str,
        stream: bool = False,
        chunk_sec: float = 0.01,
        chunk_chars: int = 5,
    ) -> Any:
        if input_type == "audio":
            if stream:
                return stream_audio(path, chunk_sec)
            return read_audio(path)
        if input_type == "text":
            if stream:
                return stream_text(path, chunk_chars)
            return read_text(path)
        raise ValueError(f"Unsupported input type: {input_type}")

    def write(self, uid: str, output: Dict[str, Any], output_dir: Path) -> None:
        output_dir = Path(output_dir)
        for key, val in output.items():
            line_parts = [uid]
            if isinstance(val, str):
                line_parts.append(val)
            elif isinstance(val, dict):
                val_type = val.get("type")
                val_value = val.get("value")
                if val_type == "text":
                    line_parts.append(val_value)
                elif val_type in ("audio", "image"):
                    ext = "flac" if val_type == "audio" else "png"
                    data_dir = output_dir / "data" / key
                    data_dir.mkdir(parents=True, exist_ok=True)
                    file_path = data_dir / f"{uid}.{ext}"
                    if val_type == "audio":
                        sf.write(file_path, val_value, 16000, format="FLAC")
                    elif val_type == "image":
                        if np.issubdtype(val_value.dtype, np.floating):
                            val_value = np.clip(val_value, 0, 1)
                            val_value = (val_value * 255).astype(np.uint8)
                        import matplotlib.pyplot as plt

                        plt.imsave(file_path, val_value, format="png")
                    line_parts.append(str(file_path))
                else:
                    raise ValueError(f"Unsupported output type: {val_type}")
            else:
                raise ValueError(f"Unsupported output value type: {type(val)}")

            with open(output_dir / f"{key}.scp", "a") as f:
                f.write(" ".join(line_parts) + "\n")

    def _get_uid_sample(self, idx: int, example: Any) -> Tuple[str, Dict[str, Any]]:
        if isinstance(example, tuple):
            uid, sample = example
        elif isinstance(example, dict):
            uid = str(idx)
            sample = example
        else:
            raise RuntimeError(f"Not supported type {type(example)}")
        return uid, sample

    def process_sample_core(
        self,
        sample: Dict[str, Any],
        model: Any,
        read_fn,
        pre_fn,
        infer_fn,
        post_fn,
        stream: bool,
        proc: Optional[psutil.Process] = None,
    ) -> Dict[str, Any]:
        if stream and "audio_path" in sample:
            sample["stream"] = read_fn("audio", sample["audio_path"], stream=True)

        mem_before = gpu_before = 0
        if proc is not None:
            mem_before, gpu_before = measure_resources(proc)

        t_start = time.time()
        t0 = time.time()
        model, sample = pre_fn(model, sample)
        pre_time = time.time() - t0

        outputs: List[Any] = []
        infer_times: List[float] = []
        stream_data = sample.get("stream")

        if stream and isinstance(stream_data, Iterator):
            for chunk in stream_data:
                t1 = time.time()
                outputs.append(infer_fn(model, chunk))
                infer_times.append(time.time() - t1)
        else:
            t1 = time.time()
            outputs = [infer_fn(model, sample)]
            infer_times.append(time.time() - t1)

        t2 = time.time()
        result = post_fn(model, outputs)
        post_time = time.time() - t2
        total_time = time.time() - t_start

        if proc is not None:
            mem_after, gpu_after = measure_resources(proc)
            result.update(
                {
                    "cpu_mem_MB": str(round(mem_after - mem_before, 2)),
                    "gpu_mem_MiB": str(round(gpu_after - gpu_before, 2)),
                }
            )

        result.update(
            {
                "pre_time": str(round(pre_time, 4)),
                "infer_time": " ".join(str(round(t, 4)) for t in infer_times),
                "post_time": str(round(post_time, 4)),
                "total_time": str(round(total_time, 4)),
            }
        )
        return result

    def merge_scp_files(
        self, output_dir: Path, final_output_dir: Path, suffix: str = ".scp"
    ) -> None:
        final_output_dir.mkdir(parents=True, exist_ok=True)
        all_scp_files = list(output_dir.glob(f"*{suffix}.*"))
        scp_groups: Dict[str, List[Path]] = {}
        for file in all_scp_files:
            key = file.name.split(".")[0] + suffix
            scp_groups.setdefault(key, []).append(file)

        for key, files in scp_groups.items():
            lines: List[str] = []
            for file in sorted(files):
                lines.extend(file.read_text().splitlines())
            with open(final_output_dir / key, "w") as fout:
                fout.write("\n".join(lines) + "\n")
