"""Metric JSON serialization supports NumPy scalars and preserves prior results."""

import json
from test.espnet3.systems.base.test_metric import DummyMetric

import numpy as np
import pytest
from omegaconf import OmegaConf

from espnet3.systems.base import metric as module


@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.longdouble])
@pytest.mark.parametrize("failure", [False, True])
def test_metric_json_atomicity(tmp_path, monkeypatch, failure, dtype):
    split = tmp_path / "test"
    split.mkdir()
    for key in ["ref", "hyp"]:
        (split / f"{key}.scp").write_text("utt hello\n")
    destination = tmp_path / "metrics.json"
    destination.write_text('{"previous": true}')
    config = OmegaConf.create(
        {
            "inference_dir": str(tmp_path),
            "dataset": {"test": [{"name": "test"}]},
            "metrics": [
                {
                    "metric": {
                        "_target_": "test.espnet3.systems.base.test_metric.DummyMetric"
                    },
                    "inputs": {"ref": "ref", "hyp": "hyp"},
                }
            ],
        }
    )
    monkeypatch.setattr(DummyMetric, "compute", lambda self: {"count": dtype(1.25)})
    if failure:

        def fail_dump(value, handle, **kwargs):
            handle.write("partial")
            raise OSError("disk failure")

        monkeypatch.setattr(module.json, "dump", fail_dump)
        with pytest.raises(OSError, match="disk failure"):
            module.measure(config, OmegaConf.create({}))
        assert json.loads(destination.read_text()) == {"previous": True}
    else:
        module.measure(config, OmegaConf.create({}))
        results = json.loads(destination.read_text())
        assert next(iter(results.values()))["test"]["count"] == float(dtype(1.25))
    assert sorted(path.name for path in tmp_path.iterdir()) == ["metrics.json", "test"]
