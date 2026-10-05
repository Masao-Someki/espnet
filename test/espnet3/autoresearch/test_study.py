"""Tests for espnet3.autoresearch.study."""

from espnet3.autoresearch import study
from espnet3.autoresearch.study import TrialRecord


def test_init_study_creates_layout(tmp_path):
    study_dir = tmp_path / "exp" / "autoresearch" / "demo"
    study.init_study(
        study_dir, config_yaml_text="autoresearch: {}", objective_text="goal"
    )

    assert (study_dir / "autoresearch.yaml").read_text() == "autoresearch: {}"
    assert (study_dir / "program.md").read_text() == "goal"
    assert (study_dir / "trials.csv").exists()
    assert (study_dir / "trials").is_dir()


def test_init_study_does_not_overwrite_existing_program_md(tmp_path):
    study_dir = tmp_path
    study.init_study(study_dir, config_yaml_text="a", objective_text="first")
    study.init_study(study_dir, config_yaml_text="b", objective_text="second")
    assert (study_dir / "program.md").read_text() == "first"
    # autoresearch.yaml is always refreshed with the latest resolved config.
    assert (study_dir / "autoresearch.yaml").read_text() == "b"


def test_next_trial_id_sequential(tmp_path):
    study.init_study(tmp_path, config_yaml_text="a", objective_text="x")
    assert study.next_trial_id(tmp_path) == "trial_000001"
    (study.trials_dir(tmp_path) / "trial_000001").mkdir()
    assert study.next_trial_id(tmp_path) == "trial_000002"
    (study.trials_dir(tmp_path) / "trial_000007").mkdir()
    assert study.next_trial_id(tmp_path) == "trial_000008"


def test_write_and_read_trial_record_roundtrip(tmp_path):
    record = TrialRecord(
        trial_id="trial_000001",
        status="accepted",
        patch={"trainer.lr": 0.1},
        score=12.3,
        rationale="try lower lr",
        created_at=study.utc_now(),
    )
    study.write_trial_record(tmp_path, record)
    loaded = study.read_trial_record(tmp_path, "trial_000001")
    assert loaded == record


def test_load_all_trial_records_sorted(tmp_path):
    for i, trial_id in enumerate(["trial_000002", "trial_000001"]):
        study.write_trial_record(
            tmp_path,
            TrialRecord(trial_id=trial_id, status="running", created_at=str(i)),
        )
    records = study.load_all_trial_records(tmp_path)
    assert [r.trial_id for r in records] == ["trial_000001", "trial_000002"]


def test_mark_interrupted_running_trials(tmp_path):
    study.write_trial_record(
        tmp_path, TrialRecord(trial_id="trial_000001", status="running")
    )
    study.write_trial_record(
        tmp_path, TrialRecord(trial_id="trial_000002", status="accepted", score=1.0)
    )
    changed = study.mark_interrupted_running_trials(tmp_path)
    assert changed == ["trial_000001"]
    reloaded = study.read_trial_record(tmp_path, "trial_000001")
    assert reloaded.status == "failed"
    assert reloaded.reason == "interrupted"
    # untouched
    assert study.read_trial_record(tmp_path, "trial_000002").status == "accepted"


def test_append_trials_csv_and_read_text(tmp_path):
    study.init_study(tmp_path, config_yaml_text="a", objective_text="x")
    record = TrialRecord(
        trial_id="trial_000001",
        status="accepted",
        patch={"a.b": 1},
        score=1.5,
        rationale="r",
        created_at="t",
    )
    study.append_trials_csv(tmp_path, record)
    text = study.read_trials_csv_text(tmp_path)
    assert "trial_000001" in text
    assert "accepted" in text


def test_update_best_tracks_min_mode(tmp_path):
    worse = TrialRecord(
        trial_id="trial_000001", status="accepted", score=10.0, patch={}
    )
    better = TrialRecord(
        trial_id="trial_000002", status="accepted", score=5.0, patch={}
    )
    assert study.update_best(tmp_path, worse, mode="min") is True
    assert study.read_best(tmp_path)["trial_id"] == "trial_000001"
    assert study.update_best(tmp_path, better, mode="min") is True
    assert study.read_best(tmp_path)["trial_id"] == "trial_000002"
    assert study.update_best(tmp_path, worse, mode="min") is False
    assert study.read_best(tmp_path)["trial_id"] == "trial_000002"


def test_update_best_tracks_max_mode(tmp_path):
    lower = TrialRecord(trial_id="trial_000001", status="accepted", score=1.0, patch={})
    higher = TrialRecord(
        trial_id="trial_000002", status="accepted", score=9.0, patch={}
    )
    study.update_best(tmp_path, lower, mode="max")
    assert study.update_best(tmp_path, higher, mode="max") is True
    assert study.read_best(tmp_path)["trial_id"] == "trial_000002"


def test_write_leaderboard_sorts_by_score(tmp_path):
    for trial_id, score in [
        ("trial_000001", 5.0),
        ("trial_000002", 1.0),
        ("trial_000003", 3.0),
    ]:
        study.write_trial_record(
            tmp_path,
            TrialRecord(trial_id=trial_id, status="accepted", score=score, patch={}),
        )
    study.write_leaderboard(tmp_path, mode="min")
    text = (tmp_path / "leaderboard.csv").read_text()
    lines = [line.split(",")[0] for line in text.strip().splitlines()[1:]]
    assert lines == ["trial_000002", "trial_000003", "trial_000001"]


def test_write_leaderboard_excludes_non_accepted(tmp_path):
    study.write_trial_record(
        tmp_path, TrialRecord(trial_id="trial_000001", status="rejected", score=1.0)
    )
    study.write_leaderboard(tmp_path, mode="min")
    text = (tmp_path / "leaderboard.csv").read_text()
    assert "trial_000001" not in text
