import pytest
from omegaconf import OmegaConf

from espnet3.utils.run_utils import (
    ConfigError,
    _is_missing_or_empty,
    resolve_loaded_configs,
)


def test_config_error_is_a_value_error():
    assert issubclass(ConfigError, ValueError)


def test_is_missing_or_empty_treats_none_and_blank_as_missing():
    assert _is_missing_or_empty(None) is True
    assert _is_missing_or_empty("") is True
    assert _is_missing_or_empty("   ") is True


def test_is_missing_or_empty_keeps_other_values():
    assert _is_missing_or_empty("train_debug") is False
    assert _is_missing_or_empty(0) is False


def test_resolve_loaded_configs_resolves_each_entry_in_place():
    train = OmegaConf.create({"exp_tag": "train_debug", "exp_dir": "./exp/${exp_tag}"})
    infer = OmegaConf.create({"inference_dir": "${exp_dir}/inference"})
    infer["exp_dir"] = "./exp/train_debug"

    resolve_loaded_configs({"train": train, "infer": infer})

    assert train.exp_dir == "./exp/train_debug"
    assert infer.inference_dir == "./exp/train_debug/inference"


def test_resolve_loaded_configs_accepts_an_empty_mapping():
    resolve_loaded_configs({})


def test_resolve_loaded_configs_raises_config_error_naming_the_key():
    infer = OmegaConf.create({"inference_dir": "${exp_dir}/inference"})

    with pytest.raises(ConfigError, match="infer config cannot resolve"):
        resolve_loaded_configs({"infer": infer})


def test_resolve_loaded_configs_names_the_failing_key_not_others():
    train = OmegaConf.create({"exp_dir": "./exp/train_debug"})
    measure = OmegaConf.create({"inference_dir": "${exp_dir}/measure"})

    try:
        resolve_loaded_configs({"train": train, "measure": measure})
    except ConfigError as exc:
        assert "measure" in str(exc)
        assert "train" not in str(exc)
    else:
        pytest.fail("expected ConfigError")
