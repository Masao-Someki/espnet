"""Standalone stages retain the configs they consume from earlier stages."""

import pytest
from omegaconf import OmegaConf

from espnet3.contract.stages import StageSpec
from espnet3.systems.base.system import BaseSystem
from espnet3.systems.launch import launch


class _SavedConfigSystem(BaseSystem):
    """Read prerequisite config without executing prerequisite stages."""

    stages = (
        StageSpec(name="prepare", config="training", log_dir="exp_dir"),
        StageSpec(name="train", config="training", log_dir="exp_dir"),
        StageSpec(name="infer", config="inference", log_dir="exp_dir"),
        StageSpec(name="measure", config="metrics", log_dir="exp_dir"),
        StageSpec(name="pack_model", config="publication", log_dir="exp_dir"),
    )
    instances = []

    def __init__(self, **kwargs):
        """Keep the system created by the real launcher for assertions."""
        super().__init__(**kwargs)
        self.executed = []
        self.prerequisites = {}
        self.instances.append(self)

    def _record(self, stage):
        """Require earlier configs, as actual stage implementations do."""
        self.executed.append(stage)
        for spec in self.stages:
            if spec.name == stage:
                break
            self.prerequisites[spec.name] = self.stage_configs[spec.name].marker

    def prepare(self):
        """Fail if the launcher reexecutes setup for a standalone stage."""
        raise AssertionError("Prerequisite stage bodies must not run")

    def train(self):
        """Consume the saved preparation config."""
        self._record("train")

    def infer(self):
        """Consume the saved training config."""
        self._record("infer")

    def measure(self):
        """Consume the saved inference config."""
        self._record("measure")

    def pack_model(self):
        """Consume the saved training and inference configs."""
        self._record("pack_model")


@pytest.mark.parametrize("stage", ["train", "infer", "measure", "pack_model"])
def test_standalone_stage_retains_prerequisite_configs(tmp_path, stage):
    """Resolve saved prerequisite configs and execute only the requested body."""
    config = OmegaConf.create({"exp_dir": str(tmp_path), "marker": "saved"})
    for spec in _SavedConfigSystem.stages:
        OmegaConf.save(config, tmp_path / f"{spec.name}.yaml")
    launch(_SavedConfigSystem, argv=["--stages", stage, "--exp_dir", str(tmp_path)])

    system = _SavedConfigSystem.instances[-1]
    requested_index = [spec.name for spec in system.stages].index(stage)
    prefix = system.stages[: requested_index + 1]
    assert set(system.stage_configs) == {spec.name for spec in prefix}
    assert system.executed == [stage]
    assert system.prerequisites == {spec.name: "saved" for spec in prefix[:-1]}
