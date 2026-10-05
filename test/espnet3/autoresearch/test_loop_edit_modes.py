"""test_loop's edit-mode-inclusive cycle: begin/end/finalize wired into loop.py."""

import logging
import shutil
import subprocess
import sys
from pathlib import Path

from espnet3.autoresearch import loop, study
from espnet3.autoresearch.agent import AgentResponse
from espnet3.autoresearch.config import (
    AgentConfig,
    AutoResearchConfig,
    BudgetConfig,
    EditConfig,
    MetricConfig,
    MetricSourceEntry,
    RecipeConfig,
    SearchSpaceConfig,
    TrialConfig,
)

_REAL_RECIPE_DIR = (
    Path(__file__).resolve().parents[3] / "egs3" / "mini_an4" / "esp2_asr"
)
_LOG = logging.getLogger("test_loop_edit_modes")


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True
    )


def _git_recipe(tmp_path: Path) -> Path:
    """A git-tracked copy of mini_an4/esp2_asr (so TEMPLATE merging still works)."""
    recipe_dir = tmp_path / "egs3" / "faketest" / "esp2_asr"
    shutil.copytree(_REAL_RECIPE_DIR / "conf", recipe_dir / "conf")
    (recipe_dir / "src").mkdir(parents=True)
    (recipe_dir / "src" / "extra.py").write_text("VALUE = 1\n")
    _git(recipe_dir, "init", "-q", "-b", "main")
    _git(recipe_dir, "config", "user.email", "test@example.com")
    _git(recipe_dir, "config", "user.name", "Test")
    _git(recipe_dir, "add", "-A")
    _git(recipe_dir, "commit", "-q", "-m", "initial commit")
    return recipe_dir


def _write_metric_command(score: float) -> list:
    code = (
        "import json, pathlib; "
        f"d = pathlib.Path('{{inference_dir}}'); d.mkdir(parents=True, exist_ok=True); "
        f"json.dump({{'wer': {score}}}, open(str(d / 'result.json'), 'w'))"
    )
    return [sys.executable, "-c", code]


def _make_allowlist_config(
    tmp_path: Path, *, commands: list, allowlist: list
) -> AutoResearchConfig:
    return AutoResearchConfig(
        study_name="demo",
        study_dir=str(tmp_path / "exp" / "autoresearch" / "demo"),
        objective_file=str(tmp_path / "program.md"),
        recipe=RecipeConfig.from_dict(
            {
                "training_config": "conf/training.yaml",
                "inference_config": "conf/inference.yaml",
                "metrics_config": "conf/metrics.yaml",
            }
        ),
        trial=TrialConfig(workdir="{recipe_dir}", commands=commands, timeout_sec=60),
        metric=MetricConfig(
            name="wer",
            mode="min",
            source=[MetricSourceEntry(path="{inference_dir}/result.json", key="wer")],
        ),
        budget=BudgetConfig(max_trials=50, max_failures=5, no_improve_stop=10),
        search_space=SearchSpaceConfig(allowed_keys=["optimizer.*"], denied_keys=[]),
        agent=AgentConfig.from_dict({"type": "command", "command": ["unused"]}),
        edit=EditConfig(mode="allowlist", allowlist=allowlist),
    )


class _FakeAgent:
    def __init__(self, patch=None):
        self.patch = patch or {"optimizer.lr": 0.02}

    def propose(self, request, *, artifact_dir, placeholders=None, cwd=None):
        return AgentResponse(rationale="r", config_patch=self.patch)


def test_allowlist_mode_edit_inside_allowlist_is_accepted(tmp_path):
    recipe_dir = _git_recipe(tmp_path)
    edit_script = (
        "import pathlib; "
        "pathlib.Path('{recipe_dir}', 'src', 'extra.py').write_text('VALUE = 2\\n')"
    )
    commands = [[sys.executable, "-c", edit_script], _write_metric_command(5.0)]
    config = _make_allowlist_config(
        tmp_path, commands=commands, allowlist=["src/**/*.py"]
    )
    study_dir = loop.resolve_study_dir(config, recipe_dir)
    study.init_study(study_dir, config_yaml_text="x", objective_text="goal")

    record = loop.run_trial_once(
        config, recipe_dir=recipe_dir, study_dir=study_dir, agent=_FakeAgent(), log=_LOG
    )

    assert record.status == "accepted"
    changes_diff = study.trial_dir(study_dir, record.trial_id) / "changes.diff"
    assert changes_diff.exists()
    assert "VALUE = 2" in changes_diff.read_text()
    # The accepted edit is kept in the actual recipe_dir (finalize no-ops on accept).
    assert "VALUE = 2" in (recipe_dir / "src" / "extra.py").read_text()


def test_allowlist_mode_edit_outside_allowlist_fails_and_restores(tmp_path):
    recipe_dir = _git_recipe(tmp_path)
    edit_script = (
        "import pathlib; "
        "pathlib.Path('{recipe_dir}', 'conf', 'metrics.yaml')"
        ".write_text('tampered: true\\n')"
    )
    commands = [[sys.executable, "-c", edit_script], _write_metric_command(5.0)]
    config = _make_allowlist_config(
        tmp_path, commands=commands, allowlist=["src/**/*.py"]
    )
    study_dir = loop.resolve_study_dir(config, recipe_dir)
    study.init_study(study_dir, config_yaml_text="x", objective_text="goal")

    record = loop.run_trial_once(
        config, recipe_dir=recipe_dir, study_dir=study_dir, agent=_FakeAgent(), log=_LOG
    )

    assert record.status == "failed"
    assert record.reason == "edit_violation"
    tdir = study.trial_dir(study_dir, record.trial_id)
    assert (tdir / "violation.txt").exists()
    assert (tdir / "violation.diff").exists()
    # Restored: the tampered file is back to its committed state.
    assert "tampered" not in (recipe_dir / "conf" / "metrics.yaml").read_text()


def test_allowlist_mode_rejected_trial_reverts_in_allowlist_edit(tmp_path):
    recipe_dir = _git_recipe(tmp_path)
    edit_script = (
        "import pathlib; "
        "pathlib.Path('{recipe_dir}', 'src', 'extra.py').write_text('VALUE = 3\\n')"
    )
    # First trial: accepted baseline (score=1.0). Second trial edits + worse score
    # (rejected) -- its in-allowlist edit should be reverted by finalize().
    commands_good = [_write_metric_command(1.0)]
    commands_bad = [[sys.executable, "-c", edit_script], _write_metric_command(9.0)]
    config = _make_allowlist_config(
        tmp_path, commands=commands_good, allowlist=["src/**/*.py"]
    )
    study_dir = loop.resolve_study_dir(config, recipe_dir)
    study.init_study(study_dir, config_yaml_text="x", objective_text="goal")

    first = loop.run_trial_once(
        config, recipe_dir=recipe_dir, study_dir=study_dir, agent=_FakeAgent(), log=_LOG
    )
    assert first.status == "accepted"

    config.trial.commands = commands_bad
    second = loop.run_trial_once(
        config, recipe_dir=recipe_dir, study_dir=study_dir, agent=_FakeAgent(), log=_LOG
    )

    assert second.status == "rejected"
    assert "VALUE = 3" not in (recipe_dir / "src" / "extra.py").read_text()
