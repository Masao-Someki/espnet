"""Tests for espnet3.autoresearch.runner."""

import json
import sys

import pytest

from espnet3.autoresearch.runner import run_trial


def _py(code: str) -> list:
    return [sys.executable, "-c", code]


def test_run_trial_success_writes_commands_json_and_logs(tmp_path):
    trial_dir = tmp_path / "trial_000001"
    trial_dir.mkdir()
    commands = [_py("print('hello {trial_id}')"), _py("print('world')")]
    result = run_trial(
        commands=commands,
        workdir="{recipe_dir}",
        env={},
        timeout_sec=30,
        placeholders={"trial_id": "t1", "recipe_dir": str(tmp_path)},
        trial_dir=trial_dir,
    )
    assert result.status == "success"
    assert (trial_dir / "cmd_00.log").read_text() == "hello t1\n"
    assert (trial_dir / "cmd_01.log").read_text() == "world\n"
    rendered = json.loads((trial_dir / "commands.json").read_text())
    assert rendered[0][-1] == "print('hello t1')"


def test_run_trial_stops_at_first_nonzero_exit(tmp_path):
    trial_dir = tmp_path / "trial_000002"
    trial_dir.mkdir()
    commands = [_py("import sys; sys.exit(3)"), _py("print('should not run')")]
    result = run_trial(
        commands=commands,
        workdir="{recipe_dir}",
        env={},
        timeout_sec=30,
        placeholders={"recipe_dir": str(tmp_path)},
        trial_dir=trial_dir,
    )
    assert result.status == "failed"
    assert result.returncode == 3
    assert result.failed_command_index == 0
    assert not (trial_dir / "cmd_01.log").exists()


def test_run_trial_times_out(tmp_path):
    trial_dir = tmp_path / "trial_000003"
    trial_dir.mkdir()
    commands = [_py("import time; time.sleep(5)")]
    result = run_trial(
        commands=commands,
        workdir="{recipe_dir}",
        env={},
        timeout_sec=1,
        placeholders={"recipe_dir": str(tmp_path)},
        trial_dir=trial_dir,
    )
    assert result.status == "timeout"
    assert result.failed_command_index == 0


def test_run_trial_passes_rendered_env(tmp_path):
    trial_dir = tmp_path / "trial_000004"
    trial_dir.mkdir()
    commands = [_py("import os; print(os.environ['ESPNET_AR_TRIAL_DIR'])")]
    result = run_trial(
        commands=commands,
        workdir="{recipe_dir}",
        env={"ESPNET_AR_TRIAL_DIR": "{trial_dir}"},
        timeout_sec=30,
        placeholders={"recipe_dir": str(tmp_path), "trial_dir": str(trial_dir)},
        trial_dir=trial_dir,
    )
    assert result.status == "success"
    assert (trial_dir / "cmd_00.log").read_text().strip() == str(trial_dir)


def test_run_trial_raises_on_unknown_placeholder(tmp_path):
    trial_dir = tmp_path / "trial_000005"
    trial_dir.mkdir()
    commands = [[sys.executable, "-c", "print('{nope}')"]]
    with pytest.raises(ValueError, match="Unknown placeholder"):
        run_trial(
            commands=commands,
            workdir="{recipe_dir}",
            env={},
            timeout_sec=30,
            placeholders={"recipe_dir": str(tmp_path)},
            trial_dir=trial_dir,
        )
