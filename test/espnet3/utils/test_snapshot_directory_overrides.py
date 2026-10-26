"""Directory overrides remain portable YAML values in saved stage configs."""

import pytest
from omegaconf import OmegaConf

from espnet3.contract.stages import StageSpec
from espnet3.utils.stage_configs import (
    bake,
    merge_stage_config_from_snapshots,
    snapshot_config,
)


@pytest.mark.parametrize("key", ["exp_dir", "inference_dir", "pack.out_dir"])
def test_path_directory_override_roundtrip(tmp_path, key):
    """Path CLI values serialize without Python-specific YAML constructors."""

    class System:
        stages = (StageSpec(name="infer", config="inference", log_dir=key),)

    config = merge_stage_config_from_snapshots(
        System, "infer", {"inference": OmegaConf.create({})}, {key: tmp_path}
    )
    assert OmegaConf.select(config, key) == str(tmp_path)
    baked = bake(tmp_path, "infer", config, own_keys=config.keys())
    snapshot_config(tmp_path, "infer", config)
    for destination in [baked, tmp_path / "infer.yaml"]:
        assert "!!python" not in destination.read_text()
        assert OmegaConf.select(OmegaConf.load(destination), key) == str(tmp_path)
    reloaded = merge_stage_config_from_snapshots(
        System, "infer", {"inference": None}, {key: tmp_path}
    )
    assert OmegaConf.select(reloaded, key) == str(tmp_path)
