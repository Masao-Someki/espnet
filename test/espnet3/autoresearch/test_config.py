"""Tests for espnet3.autoresearch.config."""

import pytest
from omegaconf import OmegaConf

from espnet3.autoresearch.config import (
    AutoResearchConfig,
    ConfigError,
    load_config,
    render,
    render_list,
)

_MINIMAL = {
    "autoresearch": {
        "study_name": "demo",
        "study_dir": "exp/autoresearch/demo",
        "recipe": {
            "training_config": "conf/training.yaml",
            "inference_config": "conf/inference.yaml",
            "metrics_config": "conf/metrics.yaml",
        },
        "trial": {
            "commands": [["python", "run.py", "--stages", "train"]],
            "timeout_sec": 100,
        },
        "metric": {
            "name": "dev_clean_wer",
            "mode": "min",
            "source": [{"path": "{trial_dir}/result.json", "key": "wer"}],
        },
        "agent": {
            "type": "command",
            "command": ["echo", "hi"],
        },
    }
}


def test_render_substitutes_known_placeholder():
    assert render("{trial_dir}/exp", {"trial_dir": "/x"}) == "/x/exp"


def test_render_raises_on_unknown_placeholder():
    with pytest.raises(ValueError, match="Unknown placeholder"):
        render("{nope}", {"trial_dir": "/x"})


def test_render_list():
    out = render_list(["{a}", "fixed", "{b}"], {"a": "1", "b": "2"})
    assert out == ["1", "fixed", "2"]


def test_load_config_from_dict_minimal(tmp_path):
    path = tmp_path / "autoresearch.yaml"
    OmegaConf.save(OmegaConf.create(_MINIMAL), path)
    config = load_config(path)
    assert isinstance(config, AutoResearchConfig)
    assert config.study_name == "demo"
    assert config.trial.commands == [["python", "run.py", "--stages", "train"]]
    assert config.metric.mode == "min"
    assert config.agent.type == "command"
    # Defaults
    assert config.budget.max_trials == 50
    assert config.edit.mode == "direct"


def test_load_config_without_autoresearch_wrapper(tmp_path):
    path = tmp_path / "autoresearch.yaml"
    OmegaConf.save(OmegaConf.create(_MINIMAL["autoresearch"]), path)
    config = load_config(path)
    assert config.study_name == "demo"


def test_load_config_rejects_bad_metric_mode(tmp_path):
    data = {"autoresearch": {**_MINIMAL["autoresearch"]}}
    data["autoresearch"]["metric"] = {
        **data["autoresearch"]["metric"],
        "mode": "sideways",
    }
    path = tmp_path / "autoresearch.yaml"
    OmegaConf.save(OmegaConf.create(data), path)
    with pytest.raises(ConfigError, match="mode"):
        load_config(path)


def test_load_config_rejects_missing_commands(tmp_path):
    data = {"autoresearch": {**_MINIMAL["autoresearch"]}}
    data["autoresearch"]["trial"] = {"timeout_sec": 10}
    path = tmp_path / "autoresearch.yaml"
    OmegaConf.save(OmegaConf.create(data), path)
    with pytest.raises(ConfigError, match="commands"):
        load_config(path)


def test_load_config_rejects_file_agent_type_without_command_required():
    # agent.type == "file" should not require agent.command.
    data = {**_MINIMAL["autoresearch"], "agent": {"type": "file"}}
    config = AutoResearchConfig.from_dict(data)
    assert config.agent.type == "file"
    assert config.agent.command == []


def test_load_config_rejects_command_type_without_command():
    data = {**_MINIMAL["autoresearch"], "agent": {"type": "command"}}
    with pytest.raises(ConfigError, match="agent.command"):
        AutoResearchConfig.from_dict(data)


def test_load_config_edit_allowlist_requires_patterns():
    data = {**_MINIMAL["autoresearch"], "edit": {"mode": "allowlist"}}
    with pytest.raises(ConfigError, match="edit.allowlist"):
        AutoResearchConfig.from_dict(data)


def test_load_config_edit_allowlist_accepted_with_patterns():
    data = {
        **_MINIMAL["autoresearch"],
        "edit": {"mode": "allowlist", "allowlist": ["conf/tuning/*.yaml"]},
    }
    config = AutoResearchConfig.from_dict(data)
    assert config.edit.mode == "allowlist"
    assert config.edit.allowlist == ["conf/tuning/*.yaml"]
