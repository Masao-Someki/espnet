"""Tests for espnet3.autoresearch.loop (fake agent, fake commands, one full cycle)."""

import sys
from pathlib import Path

from espnet3.autoresearch import loop, study
from espnet3.autoresearch.agent import AgentResponse, AgentResponseError
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

_RECIPE_DIR = Path(__file__).resolve().parents[3] / "egs3" / "mini_an4" / "esp2_asr"


def _write_metric_command(score: float) -> list:
    code = (
        "import json, pathlib; "
        f"d = pathlib.Path('{{inference_dir}}'); d.mkdir(parents=True, exist_ok=True); "
        f"json.dump({{'wer': {score}}}, open(str(d / 'result.json'), 'w'))"
    )
    return [sys.executable, "-c", code]


def _make_config(
    tmp_path, *, score: float, budget: BudgetConfig = None
) -> AutoResearchConfig:
    return AutoResearchConfig(
        study_name="demo",
        study_dir=str(tmp_path / "exp" / "autoresearch" / "demo"),
        # Absolute, so tests never write program.md into the real recipe dir.
        objective_file=str(tmp_path / "program.md"),
        recipe=RecipeConfig.from_dict(
            {
                "training_config": "conf/training.yaml",
                "inference_config": "conf/inference.yaml",
                "metrics_config": "conf/metrics.yaml",
            }
        ),
        trial=TrialConfig(
            workdir="{recipe_dir}",
            commands=[_write_metric_command(score)],
            timeout_sec=60,
        ),
        metric=MetricConfig(
            name="wer",
            mode="min",
            source=[MetricSourceEntry(path="{inference_dir}/result.json", key="wer")],
        ),
        budget=budget
        or BudgetConfig(max_trials=50, max_failures=5, no_improve_stop=10),
        search_space=SearchSpaceConfig(allowed_keys=["optimizer.*"], denied_keys=[]),
        agent=AgentConfig.from_dict({"type": "command", "command": ["unused"]}),
        edit=EditConfig(),
    )


class _FakeAgent:
    """A fake agent returning a fixed proposal (or raising), for test use."""

    def __init__(self, response=None, error=None):
        self.response = response or AgentResponse(
            rationale="try a different lr", config_patch={"optimizer.lr": 0.01}
        )
        self.error = error
        self.calls = 0

    def propose(self, request, *, artifact_dir, placeholders=None, cwd=None):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.response


def test_run_trial_once_accepts_first_trial(tmp_path):
    config = _make_config(tmp_path, score=5.0)
    study_dir = loop.resolve_study_dir(config, _RECIPE_DIR)
    study.init_study(study_dir, config_yaml_text="x", objective_text="goal")
    agent = _FakeAgent()

    record = loop.run_trial_once(
        config,
        recipe_dir=_RECIPE_DIR,
        study_dir=study_dir,
        agent=agent,
        log=__import__("logging").getLogger("t"),
    )

    assert record.status == "accepted"
    assert record.score == 5.0
    assert record.patch == {"optimizer.lr": 0.01}
    best = study.read_best(study_dir)
    assert best["trial_id"] == record.trial_id
    assert best["score"] == 5.0
    # training.yaml actually got the patch applied.
    training_text = (
        study.trial_dir(study_dir, record.trial_id) / "training.yaml"
    ).read_text()
    assert "lr: 0.01" in training_text


def test_run_trial_once_rejects_worse_score(tmp_path):
    config = _make_config(tmp_path, score=5.0)
    study_dir = loop.resolve_study_dir(config, _RECIPE_DIR)
    study.init_study(study_dir, config_yaml_text="x", objective_text="goal")
    log = __import__("logging").getLogger("t")

    first = loop.run_trial_once(
        config, recipe_dir=_RECIPE_DIR, study_dir=study_dir, agent=_FakeAgent(), log=log
    )
    second = loop.run_trial_once(
        config, recipe_dir=_RECIPE_DIR, study_dir=study_dir, agent=_FakeAgent(), log=log
    )

    assert first.status == "accepted"
    assert second.status == "rejected"  # same score, not strictly better
    assert study.read_best(study_dir)["trial_id"] == first.trial_id


def test_run_trial_once_fails_on_agent_response_error(tmp_path):
    config = _make_config(tmp_path, score=5.0)
    study_dir = loop.resolve_study_dir(config, _RECIPE_DIR)
    study.init_study(study_dir, config_yaml_text="x", objective_text="goal")
    agent = _FakeAgent(error=AgentResponseError("no response file"))

    record = loop.run_trial_once(
        config,
        recipe_dir=_RECIPE_DIR,
        study_dir=study_dir,
        agent=agent,
        log=__import__("logging").getLogger("t"),
    )
    assert record.status == "failed"
    assert record.reason == "agent_response"


def test_run_trial_once_fails_on_empty_patch_after_sanitize(tmp_path):
    config = _make_config(tmp_path, score=5.0)
    config.search_space.allowed_keys = ["nothing_matches.*"]
    study_dir = loop.resolve_study_dir(config, _RECIPE_DIR)
    study.init_study(study_dir, config_yaml_text="x", objective_text="goal")

    record = loop.run_trial_once(
        config,
        recipe_dir=_RECIPE_DIR,
        study_dir=study_dir,
        agent=_FakeAgent(),
        log=__import__("logging").getLogger("t"),
    )
    assert record.status == "failed"
    assert record.reason == "empty_patch"


def test_run_trial_once_fails_on_nonzero_command(tmp_path):
    config = _make_config(tmp_path, score=5.0)
    config.trial.commands = [[sys.executable, "-c", "import sys; sys.exit(1)"]]
    study_dir = loop.resolve_study_dir(config, _RECIPE_DIR)
    study.init_study(study_dir, config_yaml_text="x", objective_text="goal")

    record = loop.run_trial_once(
        config,
        recipe_dir=_RECIPE_DIR,
        study_dir=study_dir,
        agent=_FakeAgent(),
        log=__import__("logging").getLogger("t"),
    )
    assert record.status == "failed"
    assert record.reason == "failed"


def test_should_stop_max_trials(tmp_path):
    config = _make_config(
        tmp_path,
        score=5.0,
        budget=BudgetConfig(max_trials=1, max_failures=5, no_improve_stop=10),
    )
    study_dir = loop.resolve_study_dir(config, _RECIPE_DIR)
    study.init_study(study_dir, config_yaml_text="x", objective_text="goal")
    loop.run_trial_once(
        config,
        recipe_dir=_RECIPE_DIR,
        study_dir=study_dir,
        agent=_FakeAgent(),
        log=__import__("logging").getLogger("t"),
    )
    stop, reason = loop.should_stop(config, study_dir)
    assert stop is True
    assert reason == "max_trials"


def test_should_stop_max_failures(tmp_path):
    config = _make_config(
        tmp_path,
        score=5.0,
        budget=BudgetConfig(max_trials=50, max_failures=1, no_improve_stop=10),
    )
    study_dir = loop.resolve_study_dir(config, _RECIPE_DIR)
    study.init_study(study_dir, config_yaml_text="x", objective_text="goal")
    agent = _FakeAgent(error=AgentResponseError("boom"))
    log = __import__("logging").getLogger("t")
    loop.run_trial_once(
        config, recipe_dir=_RECIPE_DIR, study_dir=study_dir, agent=agent, log=log
    )
    stop, reason = loop.should_stop(config, study_dir)
    assert stop is True
    assert reason == "max_failures"


def test_should_stop_no_improve(tmp_path):
    config = _make_config(
        tmp_path,
        score=5.0,
        budget=BudgetConfig(max_trials=50, max_failures=50, no_improve_stop=2),
    )
    study_dir = loop.resolve_study_dir(config, _RECIPE_DIR)
    study.init_study(study_dir, config_yaml_text="x", objective_text="goal")
    log = __import__("logging").getLogger("t")
    for _ in range(3):
        loop.run_trial_once(
            config,
            recipe_dir=_RECIPE_DIR,
            study_dir=study_dir,
            agent=_FakeAgent(),
            log=log,
        )
    stop, reason = loop.should_stop(config, study_dir)
    assert stop is True
    assert reason == "no_improve_stop"


def test_should_stop_stop_file(tmp_path):
    config = _make_config(tmp_path, score=5.0)
    study_dir = loop.resolve_study_dir(config, _RECIPE_DIR)
    study.init_study(study_dir, config_yaml_text="x", objective_text="goal")
    (study_dir / "STOP").write_text("")
    stop, reason = loop.should_stop(config, study_dir)
    assert stop is True
    assert reason == "stop_file"


def test_run_study_resumes_interrupted_running_trial(tmp_path):
    config = _make_config(
        tmp_path,
        score=5.0,
        budget=BudgetConfig(max_trials=0, max_failures=5, no_improve_stop=10),
    )
    study_dir = loop.resolve_study_dir(config, _RECIPE_DIR)
    study.init_study(study_dir, config_yaml_text="x", objective_text="goal")
    study.write_trial_record(
        study_dir, study.TrialRecord(trial_id="trial_000001", status="running")
    )

    loop.run_study(config, recipe_dir=_RECIPE_DIR, agent=_FakeAgent())

    reloaded = study.read_trial_record(study_dir, "trial_000001")
    assert reloaded.status == "failed"
    assert reloaded.reason == "interrupted"


def test_run_study_creates_objective_stub_when_missing(tmp_path):
    config = _make_config(
        tmp_path,
        score=5.0,
        budget=BudgetConfig(max_trials=0, max_failures=5, no_improve_stop=10),
    )
    config.objective_file = str(tmp_path / "program.md")
    loop.run_study(config, recipe_dir=_RECIPE_DIR, agent=_FakeAgent())
    assert Path(config.objective_file).exists()
