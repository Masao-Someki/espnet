"""Let the agent implement code changes needed for a trial."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from espnet3.autoresearch.agents.interface import AgentRequest
from espnet3.autoresearch.core.result import StageResult
from espnet3.autoresearch.core.stage import AutoResearchStage


def _list_allowed_files(recipe_dir: Path, allowed_dirs: list[str]) -> str:
    lines: list[str] = []
    for rel in allowed_dirs:
        d = recipe_dir / rel
        if not d.exists():
            lines.append(f"{rel}: (does not exist yet — you may create it)")
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                lines.append(str(p.relative_to(recipe_dir)))
    return "\n".join(lines) if lines else "(none)"


def _build_prompt(
    trial,
    recipe_dir: Path,
    allowed_dirs: list[str],
    allow_pixi: bool,
) -> str:
    patch_str = json.dumps(
        {k: v for k, v in (trial.config_patch or {}).items()
         if k not in {"recipe_dir", "data_dir", "exp_tag", "exp_dir", "stats_dir"}},
        ensure_ascii=False,
        indent=2,
    )
    allowed_str = "\n".join(f"  - {d}" for d in allowed_dirs)
    existing_files = _list_allowed_files(recipe_dir, allowed_dirs)
    pixi_section = (
        "\nTo add a Python package dependency, run: `pixi add <package>`"
        " from the recipe directory.\n"
        if allow_pixi
        else ""
    )
    return "\n".join([
        "## Task: implement_code",
        "",
        "You previously proposed the following configuration change:",
        f"```json\n{patch_str}\n```",
        "",
        f"Rationale: {trial.rationale.splitlines()[0] if trial.rationale else '(none)'}",
        "",
        "Now implement any source code changes required to make this trial work.",
        "You MUST write code only to the directories listed below.",
        "",
        "Allowed directories (relative to recipe root):",
        allowed_str,
        "",
        "Existing files in those directories:",
        existing_files,
        pixi_section,
        "Guidelines:",
        "- Follow existing code style and module patterns.",
        "- If no code change is needed (config patch alone suffices), say so.",
        "- Do NOT touch files outside the allowed directories.",
        "- When loading a checkpoint with load_state_dict, always use strict=True. Never use strict=False.",
        "- After you finish, output EXACTLY one JSON object (no markdown fences):",
        '  {"files_created": [...], "files_modified": [...], '
        '"packages_added": [...], "summary": "one-line description"}',
    ])


class ImplementCodeStage(AutoResearchStage):
    """Invoke the agent to write source code for a trial."""

    study_lock_name = "study_controller"

    def run(self, context) -> StageResult:
        trial = context.current_trial
        assert trial is not None

        code_edit_cfg = getattr(context.config.autoresearch, "code_edit", None)
        allowed_dirs: list[str] = list(
            getattr(code_edit_cfg, "allowed_dirs", []) or []
        )
        allow_pixi: bool = bool(getattr(code_edit_cfg, "allow_pixi", False))

        if not allowed_dirs:
            return StageResult(
                status="success",
                message="code_edit.allowed_dirs not configured — skipping",
                payload={"trial_id": trial.trial_id},
            )

        # Ensure allowed directories exist
        for rel in allowed_dirs:
            (context.recipe_dir / rel).mkdir(parents=True, exist_ok=True)

        prompt = _build_prompt(trial, context.recipe_dir, allowed_dirs, allow_pixi)

        artifact_dir = context.trial_dir() / "implement_code"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        request = AgentRequest(
            task="implement_code",
            objective=(context.study_dir / "program.md").read_text(encoding="utf-8"),
            knowledge=(context.study_dir / "knowledge" / "knowledge_pack.md").read_text(
                encoding="utf-8"
            ) if (context.study_dir / "knowledge" / "knowledge_pack.md").exists() else "",
            repo_context=(context.study_dir / "knowledge" / "repo_context.md").read_text(
                encoding="utf-8"
            ) if (context.study_dir / "knowledge" / "repo_context.md").exists() else "",
            trial_history_csv="",
            latest_metrics={},
            latest_logs={"implement_prompt": prompt},
            allowed_actions=["file_write", "pixi_add"] if allow_pixi else ["file_write"],
            output_schema={
                "files_created": ["list of relative paths created"],
                "files_modified": ["list of relative paths modified"],
                "packages_added": ["list of pixi packages added"],
                "summary": "one-line description of what was implemented",
            },
            resume_thread_id=trial.codex_thread_id or None,
        )

        response = context.agent.run(request, artifact_dir=artifact_dir)

        if response.status != "success":
            return StageResult(
                status="failure",
                message=f"implement_code agent failed: {response.message}",
                payload={"trial_id": trial.trial_id},
            )

        structured = dict(response.structured or {})
        summary = str(structured.get("summary", ""))
        files_created = list(structured.get("files_created", []) or [])
        files_modified = list(structured.get("files_modified", []) or [])
        packages_added = list(structured.get("packages_added", []) or [])

        # Run pixi add for any declared packages
        if allow_pixi and packages_added:
            pixi_log = artifact_dir / "pixi_add.log"
            with open(pixi_log, "w", encoding="utf-8") as fh:
                for pkg in packages_added:
                    proc = subprocess.run(
                        ["pixi", "add", pkg],
                        cwd=str(context.recipe_dir),
                        stdout=fh,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=False,
                    )
                    if proc.returncode != 0:
                        return StageResult(
                            status="failure",
                            message=f"pixi add {pkg} failed — see {pixi_log}",
                            payload={"trial_id": trial.trial_id},
                        )

        (artifact_dir / "implement_summary.md").write_text(
            "\n".join([
                f"# Code Implementation: {trial.trial_id}",
                "",
                f"**Summary:** {summary}",
                "",
                f"**Files created:** {files_created}",
                f"**Files modified:** {files_modified}",
                f"**Packages added:** {packages_added}",
            ]),
            encoding="utf-8",
        )

        return StageResult(
            status="success",
            message=summary or "Code implemented",
            payload={"trial_id": trial.trial_id},
            artifacts={"summary": str(artifact_dir / "implement_summary.md")},
        )
