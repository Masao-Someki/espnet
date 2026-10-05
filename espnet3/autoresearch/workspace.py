"""Edit-mode guard for autoresearch trials: direct vs. allowlist.

Two edit modes for a trial that may let an agent modify files under a recipe
directory:

- ``"direct"``: the agent may edit anything. Nothing is enforced; the trial's
  edits (if any) are simply recorded as ``changes.diff``.
- ``"allowlist"``: the agent may only touch paths matching one of a list of
  glob patterns (recipe-relative; ``**`` matches zero or more whole path
  segments, ``*`` matches within one segment and never crosses ``/``). A path
  outside the allowlist -- including a path outside ``recipe_dir`` entirely
  -- is a violation: the trial is marked failed, ``violation.txt``/
  ``violation.diff`` are written, and only the violating paths are restored
  (tracked: ``git checkout -- <path>``; untracked: deleted).

The guard is implemented entirely via ``git status``/``git diff`` against
``recipe_dir``; it does not use a worktree or a copy (see the design doc for
why: ``data/``/``exp/``/``dump/`` are typically untracked or gitignored and
would not follow into a worktree).

Public API, used by the trial loop as::

    state = begin(mode, allowlist, recipe_dir, trial_dir)
    ... run the trial's commands, which may edit files under recipe_dir ...
    result = end(state)
    ... score the trial, decide accepted/rejected ...
    finalize(state, accepted)

``begin`` and ``end`` take and return plain values (no config dataclass), per
the caller's own requirement to wire this up without depending on
``AutoResearchConfig``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


class WorkspaceError(RuntimeError):
    """A workspace precondition failed.

    Raised when: git is unavailable or ``recipe_dir`` is not inside a git
    work tree (allowlist mode only); a path outside the allowlist is already
    dirty when a trial is about to begin (allowlist mode only); or restoring
    a violating path did not actually clean up the working tree.
    """


@dataclass
class WorkspaceState:
    """Opaque state from :func:`begin`, passed to :func:`end` and :func:`finalize`."""

    mode: str
    allowlist: list[str]
    recipe_dir: Path
    trial_dir: Path
    baseline: dict[str, str] = field(default_factory=dict)
    snapshot_dir: Path | None = None


@dataclass
class WorkspaceResult:
    """Outcome of :func:`end`."""

    status: str  # "ok" | "failed"
    reason: str | None  # "edit_violation" when status == "failed", else None
    changed_paths: list[str]
    violation_paths: list[str]
    diff_path: Path | None
    violation_txt_path: Path | None
    violation_diff_path: Path | None


_MODES = ("direct", "allowlist")


def _glob_to_regex(pattern: str) -> re.Pattern:
    """Translate an allowlist glob into an anchored regex.

    ``*`` matches any run of characters within one path segment (never
    crosses ``/``). ``**`` matches zero or more whole path segments,
    including the separating ``/`` on either side (so ``a/**/b`` matches
    both ``a/b`` and ``a/x/y/b``). Every other character is matched
    literally; there is no ``?``/``[...]``/brace support, since the
    allowlist syntax used here never needs it.
    """
    i, n = 0, len(pattern)
    out: list[str] = []
    while i < n:
        if pattern[i : i + 2] == "**":
            i += 2
            if i < n and pattern[i] == "/":
                i += 1
                out.append("(?:.*/)?")
            else:
                out.append(".*")
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def _matches_allowlist(path: str, allowlist_regexes: list[re.Pattern]) -> bool:
    if _is_outside_recipe_dir(path):
        return False
    return any(regex.match(path) for regex in allowlist_regexes)


def _is_outside_recipe_dir(path: str) -> bool:
    return path.startswith("../") or path.startswith("..\\") or Path(path).is_absolute()


def _run_git(recipe_dir: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(recipe_dir), *args],
        capture_output=True,
        text=True,
    )


def _git_available(recipe_dir: Path) -> bool:
    if shutil.which("git") is None:
        return False
    try:
        proc = _run_git(recipe_dir, "rev-parse", "--is-inside-work-tree")
    except OSError:
        return False
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def _repo_root(recipe_dir: Path) -> Path:
    proc = _run_git(recipe_dir, "rev-parse", "--show-toplevel")
    if proc.returncode != 0:
        raise WorkspaceError(
            f"git rev-parse --show-toplevel failed in {recipe_dir}: "
            f"{proc.stderr.strip()}"
        )
    return Path(proc.stdout.strip())


def _git_status(recipe_dir: Path) -> dict[str, str]:
    """Return ``{path: 2-char status code}`` for every non-clean path.

    ``git status`` always reports paths relative to the repo root, no matter
    which directory you run it from (``-C recipe_dir`` does not change this).
    This converts each one to a path relative to ``recipe_dir`` instead, so a
    path outside ``recipe_dir`` comes back with a leading ``../`` and the
    rest of this module can treat "is this recipe-relative path inside the
    allowlist" uniformly. ``--no-renames`` is passed so a rename always shows
    up as a plain delete + add pair instead of a single rename record --
    simpler to reason about, and matches how this module treats renames
    elsewhere.
    """
    proc = _run_git(
        recipe_dir,
        "status",
        "--porcelain",
        "-z",
        "--untracked-files=all",
        "--no-renames",
    )
    if proc.returncode != 0:
        raise WorkspaceError(
            f"git status failed in {recipe_dir}: {proc.stderr.strip()}"
        )
    repo_root = _repo_root(recipe_dir)
    status: dict[str, str] = {}
    for entry in proc.stdout.split("\0"):
        if not entry:
            continue
        code, root_relative_path = entry[:2], entry[3:]
        recipe_relative_path = os.path.relpath(
            repo_root / root_relative_path, recipe_dir
        )
        status[recipe_relative_path] = code
    return status


def _content_differs(path_a: Path, path_b: Path) -> bool:
    a_exists, b_exists = path_a.is_file(), path_b.is_file()
    if a_exists != b_exists:
        return True
    if not a_exists:
        return False
    return path_a.read_bytes() != path_b.read_bytes()


def _compute_changed_paths(state: WorkspaceState, current: dict[str, str]) -> list[str]:
    """Paths that differ from the ``begin()``-time baseline.

    A path not dirty at ``begin()`` but dirty now is "changed" simply by
    virtue of appearing in ``current``. A path that was *already* dirty at
    ``begin()`` needs a real content comparison against its snapshot: two
    git status codes being equal (e.g. both " M") only means "still
    modified from HEAD", not "unchanged since begin()" -- a file can be
    further edited during a trial while keeping the same status code.
    """
    changed = {p for p in current if p not in state.baseline}
    if state.snapshot_dir is not None:
        for path in state.baseline:
            if _content_differs(
                state.recipe_dir / path, _snapshot_path(state.snapshot_dir, path)
            ):
                changed.add(path)
    else:
        for path in state.baseline:
            if current.get(path) != state.baseline.get(path):
                changed.add(path)
    return sorted(changed)


def _snapshot_path(snapshot_dir: Path, path: str) -> Path:
    # path may contain "../" segments (edits outside recipe_dir); keep the
    # snapshot tree flat-safe by using the same relative structure, which
    # Path handles fine as long as we never escape snapshot_dir ourselves.
    return snapshot_dir / path


def _write_snapshot(recipe_dir: Path, snapshot_dir: Path, paths: list[str]) -> None:
    for path in paths:
        src = recipe_dir / path
        if not src.is_file():
            continue
        dst = _snapshot_path(snapshot_dir, path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _restore_path(
    recipe_dir: Path,
    snapshot_dir: Path | None,
    path: str,
    baseline: dict[str, str],
    current: dict[str, str],
) -> None:
    """Restore one path to its pre-trial state.

    If the path was already dirty at ``begin()`` time (present in
    ``baseline``), its content is copied back from the snapshot taken then.
    Otherwise it was clean (matching HEAD) at ``begin()`` time: a currently
    untracked path is simply deleted, and a currently tracked path is
    restored via ``git checkout -- <path>``.
    """
    if path in baseline:
        if snapshot_dir is None:
            raise WorkspaceError(
                f"no snapshot available to restore dirty-at-start path: {path}"
            )
        snap = _snapshot_path(snapshot_dir, path)
        target = recipe_dir / path
        if snap.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(snap, target)
        elif target.exists():
            target.unlink()
        return

    current_status = current.get(path, "")
    if current_status.startswith("?"):
        target = recipe_dir / path
        if target.exists():
            target.unlink()
        return

    proc = _run_git(recipe_dir, "checkout", "--", path)
    if proc.returncode != 0:
        raise WorkspaceError(
            f"git checkout -- {path} failed in {recipe_dir}: {proc.stderr.strip()}"
        )


def _diff_for_paths(recipe_dir: Path, paths: list[str], status: dict[str, str]) -> str:
    """Build a unified diff for ``paths`` (git diff; untracked via --no-index)."""
    tracked = [p for p in paths if not status.get(p, "").startswith("?")]
    untracked = [p for p in paths if status.get(p, "").startswith("?")]
    chunks: list[str] = []
    if tracked:
        proc = _run_git(recipe_dir, "diff", "--no-color", "--", *tracked)
        if proc.stdout:
            chunks.append(proc.stdout)
    for path in untracked:
        proc = _run_git(
            recipe_dir, "diff", "--no-color", "--no-index", "--", "/dev/null", path
        )
        # `git diff --no-index` exits 1 when the compared paths differ (the
        # expected case here, since /dev/null is always "empty"); only
        # treat >=2 as a real failure.
        if proc.returncode >= 2:
            raise WorkspaceError(
                f"git diff --no-index failed for new file {path}: {proc.stderr.strip()}"
            )
        if proc.stdout:
            chunks.append(proc.stdout)
    return "".join(chunks)


def begin(
    mode: str,
    allowlist: list[str] | None,
    recipe_dir: Path,
    trial_dir: Path,
) -> WorkspaceState:
    """Take the pre-trial baseline and (allowlist mode) snapshot in-allowlist dirt.

    Args:
        mode: ``"direct"`` or ``"allowlist"``.
        allowlist: glob patterns (recipe-dir-relative), required (non-empty)
            when ``mode == "allowlist"``; ignored for ``"direct"``.
        recipe_dir: the recipe directory a trial's commands will run in and
            may edit.
        trial_dir: this trial's own directory; ``trial_dir/snapshot/`` is
            used (allowlist mode) to hold copies of paths that were already
            dirty at the start.

    Returns:
        A :class:`WorkspaceState` to pass to :func:`end` and :func:`finalize`.

    Raises:
        WorkspaceError: ``mode`` is invalid; or (allowlist mode) git is
            unavailable, ``recipe_dir`` is not inside a git work tree, or a
            path outside the allowlist is already dirty.
    """
    if mode not in _MODES:
        raise WorkspaceError(f"mode must be one of {_MODES}, got {mode!r}")

    if mode == "direct":
        if _git_available(recipe_dir):
            full_status = _git_status(recipe_dir)
            baseline = {
                p: c for p, c in full_status.items() if not _is_outside_recipe_dir(p)
            }
        else:
            baseline = {}
        return WorkspaceState(
            mode=mode,
            allowlist=list(allowlist or []),
            recipe_dir=recipe_dir,
            trial_dir=trial_dir,
            baseline=baseline,
            snapshot_dir=None,
        )

    if not allowlist:
        raise WorkspaceError("allowlist mode requires a non-empty allowlist")
    if not _git_available(recipe_dir):
        raise WorkspaceError(
            "allowlist mode requires recipe_dir to be inside a git work tree: "
            f"{recipe_dir}"
        )

    full_status = _git_status(recipe_dir)
    baseline = {p: c for p, c in full_status.items() if not _is_outside_recipe_dir(p)}
    regexes = [_glob_to_regex(p) for p in allowlist]
    violating = sorted(p for p in baseline if not _matches_allowlist(p, regexes))
    if violating:
        raise WorkspaceError(
            "cannot start an allowlist-mode trial: path(s) outside the allowlist "
            f"are already dirty: {violating}"
        )

    snapshot_dir = trial_dir / "snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    _write_snapshot(recipe_dir, snapshot_dir, sorted(baseline))

    return WorkspaceState(
        mode=mode,
        allowlist=list(allowlist),
        recipe_dir=recipe_dir,
        trial_dir=trial_dir,
        baseline=baseline,
        snapshot_dir=snapshot_dir,
    )


def end(state: WorkspaceState) -> WorkspaceResult:
    """After a trial's commands ran, diff against the baseline and enforce the mode.

    ``direct`` mode never fails here: every changed path is written to
    ``trial_dir/changes.diff`` and the result status is always ``"ok"``.

    ``allowlist`` mode: every currently-dirty path is checked against the
    allowlist (a path outside ``recipe_dir`` is always a violation, by
    construction, since :func:`begin` only ever accepted an already-dirty
    path when it matched the allowlist). If any violation is found,
    ``violation.txt``/``violation.diff`` are written, the violating paths
    (only) are restored, and the result status is ``"failed"`` with reason
    ``"edit_violation"``. Otherwise every change is written to
    ``changes.diff`` and the status is ``"ok"``.

    Raises:
        WorkspaceError: a violating path could not be fully restored (a
            difference remains after the restore attempt).
    """
    current = _git_status(state.recipe_dir) if _git_available(state.recipe_dir) else {}
    changed_paths = _compute_changed_paths(state, current)

    if state.mode == "direct":
        diff_path = None
        if changed_paths:
            diff_path = state.trial_dir / "changes.diff"
            diff_path.write_text(
                _diff_for_paths(state.recipe_dir, changed_paths, current),
                encoding="utf-8",
            )
        return WorkspaceResult(
            status="ok",
            reason=None,
            changed_paths=changed_paths,
            violation_paths=[],
            diff_path=diff_path,
            violation_txt_path=None,
            violation_diff_path=None,
        )

    regexes = [_glob_to_regex(p) for p in state.allowlist]
    violation_paths = sorted(
        p for p in changed_paths if not _matches_allowlist(p, regexes)
    )

    if violation_paths:
        violation_txt_path = state.trial_dir / "violation.txt"
        violation_txt_path.write_text(
            "\n".join(violation_paths) + "\n", encoding="utf-8"
        )
        violation_diff_path = state.trial_dir / "violation.diff"
        violation_diff_path.write_text(
            _diff_for_paths(state.recipe_dir, violation_paths, current),
            encoding="utf-8",
        )
        for path in violation_paths:
            _restore_path(
                state.recipe_dir, state.snapshot_dir, path, state.baseline, current
            )

        remaining = _git_status(state.recipe_dir)
        still_different = [p for p in violation_paths if p in remaining]
        if still_different:
            raise WorkspaceError(
                "failed to fully restore violating path(s) after an allowlist "
                f"violation: {still_different}"
            )

        return WorkspaceResult(
            status="failed",
            reason="edit_violation",
            changed_paths=changed_paths,
            violation_paths=violation_paths,
            diff_path=None,
            violation_txt_path=violation_txt_path,
            violation_diff_path=violation_diff_path,
        )

    diff_path = None
    if changed_paths:
        diff_path = state.trial_dir / "changes.diff"
        diff_path.write_text(
            _diff_for_paths(state.recipe_dir, changed_paths, current), encoding="utf-8"
        )
    return WorkspaceResult(
        status="ok",
        reason=None,
        changed_paths=changed_paths,
        violation_paths=[],
        diff_path=diff_path,
        violation_txt_path=None,
        violation_diff_path=None,
    )


def finalize(state: WorkspaceState, accepted: bool) -> None:
    """Revert a rejected/failed trial's in-allowlist edits; keep an accepted trial's.

    No-op in ``direct`` mode (its edits are never reverted, only recorded by
    :func:`end`) and when ``accepted`` is true. In ``allowlist`` mode with
    ``accepted=False``, every path still dirty at this point (any allowlist
    violation was already restored by :func:`end`, so this only ever touches
    legitimate, in-allowlist edits) is restored the same way :func:`end`
    restores a violation: from the ``begin()``-time snapshot if it was
    already dirty then, otherwise via ``git checkout --`` / delete.
    """
    if state.mode != "allowlist" or accepted:
        return
    if not _git_available(state.recipe_dir):
        return

    current = _git_status(state.recipe_dir)
    changed_paths = _compute_changed_paths(state, current)
    for path in changed_paths:
        _restore_path(
            state.recipe_dir, state.snapshot_dir, path, state.baseline, current
        )
