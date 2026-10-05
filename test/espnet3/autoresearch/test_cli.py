"""Tests for espnet3.autoresearch.cli."""

import json
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from espnet3.autoresearch import cli
from espnet3.autoresearch.agent import CommandAgent, FileAgent
from espnet3.autoresearch.config import AgentConfig, AutoResearchConfig

_RECIPE_DIR = Path(__file__).resolve().parents[3] / "egs3" / "mini_an4" / "esp2_asr"


def _agent_config(agent_type: str = "command") -> AgentConfig:
    if agent_type == "command":
        return AgentConfig.from_dict({"type": "command", "command": ["echo", "hi"]})
    return AgentConfig.from_dict({"type": "file"})


def test_build_agent_command_returns_command_agent():
    config = AutoResearchConfig.from_dict(
        {
            "study_name": "x",
            "study_dir": "exp",
            "recipe": {
                "training_config": "a",
                "inference_config": "b",
                "metrics_config": "c",
            },
            "trial": {"commands": [["echo"]], "timeout_sec": 1},
            "metric": {
                "name": "x",
                "mode": "min",
                "source": [{"path": "p", "key": "k"}],
            },
            "agent": {"type": "command", "command": ["echo", "hi"]},
        }
    )
    agent = cli.build_agent(config)
    assert isinstance(agent, CommandAgent)


def test_build_agent_file_returns_file_agent():
    config = AutoResearchConfig.from_dict(
        {
            "study_name": "x",
            "study_dir": "exp",
            "recipe": {
                "training_config": "a",
                "inference_config": "b",
                "metrics_config": "c",
            },
            "trial": {"commands": [["echo"]], "timeout_sec": 1},
            "metric": {
                "name": "x",
                "mode": "min",
                "source": [{"path": "p", "key": "k"}],
            },
            "agent": {"type": "file"},
        }
    )
    agent = cli.build_agent(config)
    assert isinstance(agent, FileAgent)


def _autoresearch_yaml(tmp_path, study_dir: Path, score: float) -> Path:
    write_metric = (
        "import json, pathlib; "
        f"d = pathlib.Path('{{inference_dir}}'); d.mkdir(parents=True, exist_ok=True); "
        f"json.dump({{'wer': {score}}}, open(str(d / 'result.json'), 'w'))"
    )
    fake_agent_response = (
        "import json; "
        "json.dump({'rationale': 'r', 'config_patch': {'optimizer.lr': 0.02}}, "
        "open('{response_file}', 'w'))"
    )
    data = {
        "autoresearch": {
            "study_name": "demo",
            "study_dir": str(study_dir),
            # Absolute, so tests never write program.md into the real recipe dir.
            "objective_file": str(tmp_path / "program.md"),
            "recipe": {
                "training_config": "conf/training.yaml",
                "inference_config": "conf/inference.yaml",
                "metrics_config": "conf/metrics.yaml",
            },
            "trial": {
                "commands": [[sys.executable, "-c", write_metric]],
                "timeout_sec": 60,
            },
            "metric": {
                "name": "wer",
                "mode": "min",
                "source": [{"path": "{inference_dir}/result.json", "key": "wer"}],
            },
            "budget": {"max_trials": 1, "max_failures": 5, "no_improve_stop": 10},
            "search_space": {"allowed_keys": ["optimizer.*"], "denied_keys": []},
            "agent": {
                "type": "command",
                "command": [sys.executable, "-c", fake_agent_response],
            },
        }
    }
    path = tmp_path / "autoresearch.yaml"
    OmegaConf.save(OmegaConf.create(data), path)
    return path


def test_run_command_completes_one_trial_end_to_end(tmp_path):
    study_dir = tmp_path / "exp" / "autoresearch" / "demo"
    config_path = _autoresearch_yaml(tmp_path, study_dir, score=3.5)

    cli.main(["run", "--config", str(config_path), "--recipe_dir", str(_RECIPE_DIR)])

    trials_csv = (study_dir / "trials.csv").read_text()
    assert "accepted" in trials_csv
    best = json.loads((study_dir / "best.json").read_text())
    assert best["score"] == 3.5


def test_status_command_prints_summary(tmp_path, capsys):
    study_dir = tmp_path / "exp" / "autoresearch" / "demo"
    config_path = _autoresearch_yaml(tmp_path, study_dir, score=3.5)
    cli.main(["run", "--config", str(config_path), "--recipe_dir", str(_RECIPE_DIR)])

    cli.main(["status", "--config", str(config_path), "--recipe_dir", str(_RECIPE_DIR)])
    out = capsys.readouterr().out
    assert "trials: 1" in out
    assert "best:" in out
    assert "score=3.5" in out


def test_init_command_creates_study_without_running_trials(tmp_path):
    study_dir = tmp_path / "exp" / "autoresearch" / "demo"
    config_path = _autoresearch_yaml(tmp_path, study_dir, score=3.5)

    cli.main(["init", "--config", str(config_path), "--recipe_dir", str(_RECIPE_DIR)])

    assert (study_dir / "autoresearch.yaml").exists()
    assert (study_dir / "program.md").exists()
    assert not (study_dir / "trials").exists() or not list(
        (study_dir / "trials").iterdir()
    )


def test_build_agent_rejects_unknown_type():
    config = AutoResearchConfig.from_dict(
        {
            "study_name": "x",
            "study_dir": "exp",
            "recipe": {
                "training_config": "a",
                "inference_config": "b",
                "metrics_config": "c",
            },
            "trial": {"commands": [["echo"]], "timeout_sec": 1},
            "metric": {
                "name": "x",
                "mode": "min",
                "source": [{"path": "p", "key": "k"}],
            },
            "agent": {"type": "file"},
        }
    )
    config.agent.type = "bogus"
    with pytest.raises(ValueError, match="Unsupported agent.type"):
        cli.build_agent(config)
