"""Tests for the direct/allowlist edit-mode guard in espnet3.autoresearch.workspace."""

import subprocess
from pathlib import Path

import pytest

from espnet3.autoresearch import workspace


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True
    )


def _init_recipe(tmp_path: Path) -> Path:
    """A tiny git repo with one commit, standing in for a real recipe_dir."""
    recipe_dir = tmp_path / "recipe"
    (recipe_dir / "conf" / "tuning").mkdir(parents=True)
    (recipe_dir / "src" / "sub").mkdir(parents=True)
    (recipe_dir / "conf" / "tuning" / "base.yaml").write_text("lr: 0.001\n")
    (recipe_dir / "src" / "model.py").write_text("VALUE = 1\n")
    (recipe_dir / "src" / "sub" / "deep.py").write_text("DEEP = 1\n")
    (recipe_dir / "data.txt").write_text("not editable\n")
    _git(recipe_dir, "init", "-q", "-b", "main")
    _git(recipe_dir, "config", "user.email", "test@example.com")
    _git(recipe_dir, "config", "user.name", "Test")
    _git(recipe_dir, "add", "-A")
    _git(recipe_dir, "commit", "-q", "-m", "initial commit")
    return recipe_dir


@pytest.fixture
def recipe_dir(tmp_path: Path) -> Path:
    return _init_recipe(tmp_path)


@pytest.fixture
def trial_dir(tmp_path: Path) -> Path:
    d = tmp_path / "trial_000001"
    d.mkdir()
    return d


# --- (i): allowed-only edits -------------------------------------------------


def test_allowlist_edit_inside_allowlist_writes_changes_diff(recipe_dir, trial_dir):
    allowlist = ["conf/tuning/*.yaml", "src/**/*.py"]
    state = workspace.begin("allowlist", allowlist, recipe_dir, trial_dir)

    (recipe_dir / "conf" / "tuning" / "base.yaml").write_text("lr: 0.01\n")

    result = workspace.end(state)

    assert result.status == "ok"
    assert result.reason is None
    assert result.violation_paths == []
    assert result.changed_paths == ["conf/tuning/base.yaml"]
    assert result.diff_path is not None and result.diff_path.is_file()
    assert "lr: 0.01" in result.diff_path.read_text(encoding="utf-8")
    # The allowed edit is NOT reverted by end() itself.
    assert (recipe_dir / "conf" / "tuning" / "base.yaml").read_text() == "lr: 0.01\n"


# --- (ii): forbidden edit -> failed + violation.diff + restored, in one test -


def test_allowlist_edit_outside_allowlist_fails_and_restores(recipe_dir, trial_dir):
    allowlist = ["conf/tuning/*.yaml"]
    state = workspace.begin("allowlist", allowlist, recipe_dir, trial_dir)

    (recipe_dir / "src" / "model.py").write_text("VALUE = 999\n")

    result = workspace.end(state)

    assert result.status == "failed"
    assert result.reason == "edit_violation"
    assert result.violation_paths == ["src/model.py"]
    assert result.violation_txt_path.read_text(encoding="utf-8") == "src/model.py\n"
    assert "VALUE = 999" in result.violation_diff_path.read_text(encoding="utf-8")
    # The violating file is restored to its pre-trial (HEAD) content.
    assert (recipe_dir / "src" / "model.py").read_text() == "VALUE = 1\n"


# --- (iii): forbidden new untracked file -> deleted --------------------------


def test_allowlist_new_untracked_file_outside_allowlist_is_deleted(
    recipe_dir, trial_dir
):
    allowlist = ["conf/tuning/*.yaml"]
    state = workspace.begin("allowlist", allowlist, recipe_dir, trial_dir)

    new_file = recipe_dir / "junk.bin"
    new_file.write_text("not allowed\n")

    result = workspace.end(state)

    assert result.status == "failed"
    assert result.violation_paths == ["junk.bin"]
    assert not new_file.exists()


# --- (iv): dirty-outside-allowlist at start -> refuse to begin ---------------


def test_allowlist_begin_refuses_when_outside_allowlist_is_already_dirty(
    recipe_dir, trial_dir
):
    (recipe_dir / "src" / "model.py").write_text("VALUE = 2\n")

    with pytest.raises(workspace.WorkspaceError, match="already dirty"):
        workspace.begin("allowlist", ["conf/tuning/*.yaml"], recipe_dir, trial_dir)

    # Untouched: begin() must not have modified anything.
    assert (recipe_dir / "src" / "model.py").read_text() == "VALUE = 2\n"


# --- (v): ** (recursive) vs * (one segment) ----------------------------------


def test_glob_star_matches_only_one_path_segment(recipe_dir, trial_dir):
    allowlist = ["conf/tuning/*.yaml"]
    state = workspace.begin("allowlist", allowlist, recipe_dir, trial_dir)

    nested = recipe_dir / "conf" / "tuning" / "nested"
    nested.mkdir()
    (nested / "deep.yaml").write_text("x: 1\n")

    result = workspace.end(state)

    assert result.status == "failed"
    assert result.violation_paths == ["conf/tuning/nested/deep.yaml"]


def test_glob_double_star_matches_recursively(recipe_dir, trial_dir):
    allowlist = ["src/**/*.py"]
    state = workspace.begin("allowlist", allowlist, recipe_dir, trial_dir)

    (recipe_dir / "src" / "model.py").write_text("VALUE = 2\n")
    (recipe_dir / "src" / "sub" / "deep.py").write_text("DEEP = 2\n")

    result = workspace.end(state)

    assert result.status == "ok"
    assert result.violation_paths == []
    assert set(result.changed_paths) == {"src/model.py", "src/sub/deep.py"}


# --- (vi): direct mode never enforces, only records --------------------------


def test_direct_mode_records_but_never_reverts(recipe_dir, trial_dir):
    state = workspace.begin("direct", None, recipe_dir, trial_dir)

    (recipe_dir / "src" / "model.py").write_text("VALUE = 42\n")
    (recipe_dir / "data.txt").write_text("edited anyway\n")

    result = workspace.end(state)

    assert result.status == "ok"
    assert result.reason is None
    assert result.violation_paths == []
    assert set(result.changed_paths) == {"src/model.py", "data.txt"}
    assert result.diff_path is not None
    diff_text = result.diff_path.read_text(encoding="utf-8")
    assert "VALUE = 42" in diff_text and "edited anyway" in diff_text
    # direct mode never reverts, even after end().
    assert (recipe_dir / "src" / "model.py").read_text() == "VALUE = 42\n"

    # finalize() with accepted=False is also a no-op in direct mode.
    workspace.finalize(state, accepted=False)
    assert (recipe_dir / "src" / "model.py").read_text() == "VALUE = 42\n"


# --- (vii): rejected trial's in-allowlist edit is reverted from snapshot -----


def test_finalize_reverts_allowlist_edit_when_not_accepted(recipe_dir, trial_dir):
    allowlist = ["conf/tuning/*.yaml"]
    state = workspace.begin("allowlist", allowlist, recipe_dir, trial_dir)

    (recipe_dir / "conf" / "tuning" / "base.yaml").write_text("lr: 0.05\n")
    result = workspace.end(state)
    assert result.status == "ok"

    workspace.finalize(state, accepted=False)

    assert (recipe_dir / "conf" / "tuning" / "base.yaml").read_text() == "lr: 0.001\n"


def test_finalize_keeps_allowlist_edit_when_accepted(recipe_dir, trial_dir):
    allowlist = ["conf/tuning/*.yaml"]
    state = workspace.begin("allowlist", allowlist, recipe_dir, trial_dir)

    (recipe_dir / "conf" / "tuning" / "base.yaml").write_text("lr: 0.05\n")
    result = workspace.end(state)
    assert result.status == "ok"

    workspace.finalize(state, accepted=True)

    assert (recipe_dir / "conf" / "tuning" / "base.yaml").read_text() == "lr: 0.05\n"


def test_finalize_reverts_already_dirty_allowlist_path_from_snapshot(
    recipe_dir, trial_dir
):
    # The file is already dirty (uncommitted) BEFORE the trial starts -- a
    # legitimate carry-over edit from a previously accepted trial.
    (recipe_dir / "conf" / "tuning" / "base.yaml").write_text("lr: 0.02\n")
    allowlist = ["conf/tuning/*.yaml"]
    state = workspace.begin("allowlist", allowlist, recipe_dir, trial_dir)

    # This trial edits it further.
    (recipe_dir / "conf" / "tuning" / "base.yaml").write_text("lr: 0.09\n")
    result = workspace.end(state)
    assert result.status == "ok"

    workspace.finalize(state, accepted=False)

    # Reverted to the begin()-time (already-dirty) content, not to HEAD.
    assert (recipe_dir / "conf" / "tuning" / "base.yaml").read_text() == "lr: 0.02\n"


# --- extra: recipe_dir outside edits count as violations in allowlist mode --


def test_allowlist_counts_edits_outside_recipe_dir_as_violations(tmp_path):
    outer = tmp_path / "repo"
    (outer / "recipe" / "conf" / "tuning").mkdir(parents=True)
    (outer / "recipe" / "conf" / "tuning" / "base.yaml").write_text("lr: 0.001\n")
    (outer / "other.py").write_text("OUTSIDE = 1\n")
    _git(outer, "init", "-q", "-b", "main")
    _git(outer, "config", "user.email", "test@example.com")
    _git(outer, "config", "user.name", "Test")
    _git(outer, "add", "-A")
    _git(outer, "commit", "-q", "-m", "initial commit")

    recipe_dir = outer / "recipe"
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()

    state = workspace.begin("allowlist", ["conf/tuning/*.yaml"], recipe_dir, trial_dir)
    (outer / "other.py").write_text("OUTSIDE = 2\n")

    result = workspace.end(state)

    assert result.status == "failed"
    assert result.violation_paths == ["../other.py"]
    assert (outer / "other.py").read_text() == "OUTSIDE = 1\n"


# --- extra: git missing / not a repo -> allowlist refuses to begin ----------


def test_allowlist_begin_refuses_when_not_a_git_repo(tmp_path):
    recipe_dir = tmp_path / "plain_dir"
    (recipe_dir / "conf" / "tuning").mkdir(parents=True)
    (recipe_dir / "conf" / "tuning" / "base.yaml").write_text("lr: 0.001\n")
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()

    with pytest.raises(workspace.WorkspaceError, match="git work tree"):
        workspace.begin("allowlist", ["conf/tuning/*.yaml"], recipe_dir, trial_dir)


def test_direct_mode_does_not_require_a_git_repo(tmp_path):
    recipe_dir = tmp_path / "plain_dir"
    (recipe_dir / "conf").mkdir(parents=True)
    (recipe_dir / "conf" / "x.yaml").write_text("a: 1\n")
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()

    state = workspace.begin("direct", None, recipe_dir, trial_dir)
    (recipe_dir / "conf" / "x.yaml").write_text("a: 2\n")
    result = workspace.end(state)

    assert result.status == "ok"
    assert result.changed_paths == []
    assert result.diff_path is None
