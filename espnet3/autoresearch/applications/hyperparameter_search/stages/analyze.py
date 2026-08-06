"""Generate a comprehensive analysis report for the completed study."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from espnet3.autoresearch.agents.interface import AgentRequest
from espnet3.autoresearch.core.result import StageResult
from espnet3.autoresearch.core.stage import AutoResearchStage


def _build_structured_data(context) -> dict:
    """Collect trial scores and patches for analysis."""
    trials = context.state.list_trials(context.scheduler.study_id)
    mode = str(context.config.autoresearch.metric.mode)
    metric_name = str(context.config.autoresearch.metric.name)

    _runtime_keys = {"recipe_dir", "data_dir", "exp_tag", "exp_dir", "stats_dir"}

    scored = [t for t in trials if t.score is not None]
    reverse = mode == "max"
    scored.sort(key=lambda t: float(t.score), reverse=reverse)

    # leaderboard rows
    leaderboard = []
    for rank, t in enumerate(scored, start=1):
        patch = {k: v for k, v in (t.config_patch or {}).items() if k not in _runtime_keys}
        leaderboard.append({
            "rank": rank,
            "trial_id": t.trial_id,
            "score": t.score,
            "metric": metric_name,
            "status": t.status,
            "config_patch": patch,
        })

    # per-axis effect: group by each patched key
    axis_effects: dict[str, list[dict]] = defaultdict(list)
    for t in scored:
        patch = {k: v for k, v in (t.config_patch or {}).items() if k not in _runtime_keys}
        for key in patch:
            axis_effects[key].append({"trial_id": t.trial_id, "value": patch[key], "score": t.score})

    # early-stopped summary
    early_stopped = [
        {"trial_id": t.trial_id, "rationale": (t.rationale or "")[:200]}
        for t in trials if t.status == "early_stopped"
    ]

    # best trial
    best = scored[0] if scored else None

    return {
        "metric_name": metric_name,
        "mode": mode,
        "total_trials": len(trials),
        "scored_trials": len(scored),
        "early_stopped_count": len(early_stopped),
        "leaderboard": leaderboard,
        "axis_effects": dict(axis_effects),
        "early_stopped": early_stopped,
        "best": {
            "trial_id": best.trial_id,
            "score": best.score,
            "config_patch": {k: v for k, v in (best.config_patch or {}).items() if k not in _runtime_keys},
        } if best else None,
    }


def _fallback_analysis(data: dict, objective: str) -> str:
    """Generate a plain-text analysis.md without an agent."""
    lines = [
        "# Study Analysis",
        "",
        "## Objective",
        objective.strip(),
        "",
        "## Summary",
        f"- Total trials run: {data['total_trials']}",
        f"- Trials with scores: {data['scored_trials']}",
        f"- Early-stopped trials: {data['early_stopped_count']}",
        f"- Metric: {data['metric_name']} ({data['mode']})",
        "",
    ]

    if data["best"]:
        b = data["best"]
        lines += [
            "## Best Trial",
            f"- Trial: {b['trial_id']}",
            f"- Score ({data['metric_name']}): {b['score']}",
            f"- Config patch: `{json.dumps(b['config_patch'], ensure_ascii=False)}`",
            "",
        ]

    lines += ["## Leaderboard", ""]
    lines.append("| Rank | Trial | Score | Config patch |")
    lines.append("|------|-------|-------|--------------|")
    for row in data["leaderboard"][:20]:
        patch_str = json.dumps(row["config_patch"], ensure_ascii=False)
        if len(patch_str) > 80:
            patch_str = patch_str[:77] + "..."
        lines.append(f"| {row['rank']} | {row['trial_id']} | {row['score']} | `{patch_str}` |")
    lines.append("")

    lines += ["## Per-axis Effect", ""]
    for axis, entries in data["axis_effects"].items():
        lines.append(f"### `{axis}`")
        entries_sorted = sorted(entries, key=lambda e: float(e["score"]))
        for e in entries_sorted:
            lines.append(f"- value={e['value']} → {data['metric_name']}={e['score']} ({e['trial_id']})")
        lines.append("")

    return "\n".join(lines)


class AnalyzeStage(AutoResearchStage):
    """Generate analysis.md summarising the completed study."""

    study_lock_name = "study_controller"

    def run(self, context) -> StageResult:
        study_dir: Path = context.study_dir
        objective = (study_dir / "program.md").read_text(encoding="utf-8")
        knowledge = ""
        kp = study_dir / "knowledge" / "knowledge_pack.md"
        if kp.exists():
            knowledge = kp.read_text(encoding="utf-8")
        trials_csv = study_dir / "trials.csv"
        trial_history_csv = trials_csv.read_text(encoding="utf-8") if trials_csv.exists() else ""

        structured_data = _build_structured_data(context)
        structured_json = json.dumps(structured_data, ensure_ascii=False, indent=2)

        request = AgentRequest(
            task="analyze",
            objective=objective,
            knowledge=knowledge,
            repo_context=(study_dir / "knowledge" / "repo_context.md").read_text(encoding="utf-8")
            if (study_dir / "knowledge" / "repo_context.md").exists()
            else "",
            trial_history_csv=trial_history_csv,
            latest_metrics={},
            latest_logs={"structured_data": structured_json},
            allowed_actions=["analysis_only"],
            output_schema={"analysis_markdown": "string"},
        )

        response = context.agent.run(request, artifact_dir=study_dir / "analysis_agent")
        if response.status == "success":
            structured = dict(response.structured or {})
            content = str(structured.get("analysis_markdown", response.content or "")).strip()
        else:
            content = _fallback_analysis(structured_data, objective)

        analysis_path = study_dir / "analysis.md"
        analysis_path.write_text(content, encoding="utf-8")

        return StageResult(
            status="terminal",
            message="analysis.md generated",
            artifacts={"analysis": str(analysis_path)},
        )
