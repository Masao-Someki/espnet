"""Dotted directory flags reach the effective stage config."""

import pytest
from omegaconf import OmegaConf

from espnet3.contract.stages import StageSpec
from espnet3.systems.base.system import BaseSystem
from espnet3.systems.launch import launch


@pytest.mark.parametrize(
    "key", ["tokenizer.save_path", "pack.out_dir", "remove_long_short.save_path"]
)
def test_dotted_directory_override(tmp_path, key):
    """Use the real parser, resolver and stage invocation."""
    observed = []

    class System(BaseSystem):
        stages = (StageSpec(name="record", config="training", log_dir=key),)

        def record(self):
            observed.append(OmegaConf.select(self.stage_configs["record"], key))

    config = OmegaConf.create({"exp_dir": str(tmp_path / "exp")})
    OmegaConf.update(config, key, str(tmp_path / "old"))
    source = tmp_path / "training.yaml"
    OmegaConf.save(config, source)
    launch(
        System,
        conf_package="egs3.TEMPLATE.esp2_asr",
        argv=[
            "--stages",
            "record",
            "--training_config",
            str(source),
            "--" + key.replace(".", "-").replace("_", "-"),
            str(tmp_path / "new"),
        ],
    )
    assert observed == [str(tmp_path / "new")]
