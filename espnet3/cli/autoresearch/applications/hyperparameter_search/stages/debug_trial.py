"""Failure-analysis and debug stage for failed trials."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from espnet3.cli.autoresearch.agents.interface import AgentRequest
from espnet3.cli.autoresearch.applications.hyperparameter_search.config_schema import (
    load_recipe_stage_config,
    write_resolved_config,
)
from espnet3.cli.autoresearch.core.models import StageResult
from espnet3.cli.autoresearch.core.serialization import (
    apply_dotted_patch,
    save_yaml,
    utc_now,
)
from espnet3.cli.autoresearch.core.stage import AutoResearchStage
from espnet3.cli.autoresearch.core.models import Trial

def _glob_match(patterns, key: str) -> bool:
    import fnmatch

    return any(fnmatch.fnmatch(key, pattern) for pattern in patterns)


class DebugTrialStage(AutoResearchStage):
    """Inspect failure logs and ask an agent for a patch-only retry fix."""

    def _sanitize_patch(self, context, patch: dict) -> tuple[dict, list[str]]:
        search_space = context.config.autoresearch.search_space
        allowed = list(getattr(search_space, "allowed_keys", []) or [])
        denied = list(getattr(search_space, "denied_keys", []) or [])
        sanitized = {}
        removed = []
        for key, value in patch.items():
            key = str(key)
            if denied and _glob_match(denied, key):
                removed.append(key)
                continue
            if allowed and not _glob_match(allowed, key):
                removed.append(key)
                continue
            sanitized[key] = value
        return sanitized, removed

    def _build_request(self, context, trial) -> AgentRequest:
        trial_dir = context.trial_dir()
        logs = {
            "trial_dir": str(trial_dir),
            "training_log": str(trial_dir / "train.log"),
            "evaluation_log": str(trial_dir / "eval.log"),
        }
        if trial.attempt_count > 0:
            latest_attempt = trial_dir / "attempts" / f"attempt_{trial.attempt_count:06d}"
            logs["attempt_stdout"] = str(latest_attempt / "stdout.log")
            logs["attempt_stderr"] = str(latest_attempt / "stderr.log")

        resolved_config_text = (
            (trial_dir / "resolved_training_config.yaml").read_text(encoding="utf-8")
            if (trial_dir / "resolved_training_config.yaml").exists()
            else "(missing)"
        )

        def _load_base(config_path_key: str, config_name: str) -> str:
            try:
                from espnet3.cli.autoresearch.applications.hyperparameter_search.config_schema import (
                    load_recipe_stage_config,
                )
                cfg = load_recipe_stage_config(
                    context.recipe_dir,
                    Path(getattr(context.config.autoresearch.recipe, config_path_key)),
                    config_name,
                    resolve=False,
                )
                return OmegaConf.to_yaml(cfg)
            except Exception:
                return "(unavailable)"

        base_configs_context = "\n\n".join([
            "## Base configs\n"
            "Runtime keys are injected automatically — do NOT include them.\n",
            "### training config\n```yaml\n" + _load_base("training_config", "training.yaml") + "```",
            "### Fixed inference config (immutable; do not propose patches)\n```yaml\n" + _load_base("inference_config", "inference.yaml") + "```",
            "### Fixed metrics config (immutable; do not propose patches)\n```yaml\n" + _load_base("metrics_config", "metrics.yaml") + "```",
        ])

        repo_context_parts = [
            f"Trial ID: {trial.trial_id}",
            f"Trial status: {trial.status}",
            f"Current patch: {trial.config_patch}",
            "",
            "## Resolved training config (what actually ran)",
            resolved_config_text,
            "",
            base_configs_context,
        ]

        trial_history_csv = ""
        trials_csv = context.study_dir / "trials.csv"
        if trials_csv.exists():
            trial_history_csv = trials_csv.read_text(encoding="utf-8")

        knowledge = ""
        knowledge_path = context.study_dir / "knowledge" / "knowledge_pack.md"
        if knowledge_path.exists():
            knowledge = knowledge_path.read_text(encoding="utf-8")

        code_edit_cfg = getattr(context.config.autoresearch, "code_edit", None)
        allowed_dirs = list(getattr(code_edit_cfg, "allowed_dirs", ["src"]) or ["src"])
        allow_pixi = bool(getattr(code_edit_cfg, "allow_pixi", True))

        allowed_actions = ["file_write"]
        if allow_pixi:
            allowed_actions.append("pixi_add")

        return AgentRequest(
            task="debug",
            objective=(context.study_dir / "program.md").read_text(encoding="utf-8"),
            knowledge=knowledge,
            repo_context="\n".join(repo_context_parts),
            trial_history_csv=trial_history_csv,
            latest_metrics=trial.metrics,
            context_files=logs,
            allowed_actions=allowed_actions,
            output_schema={
                "root_cause": "string",
                "fix_strategy": "string",
                "files_modified": ["list of src/ files edited (empty if none)"],
                "config_patches": {
                    "training": {"dotted.key": "value (or empty dict)"},
                },
                "should_retry": True,
            },
        )

    def run(self, context) -> StageResult:
        trial = context.current_trial
        assert trial is not None

        request = self._build_request(context, trial)
        artifact_dir = context.trial_dir()
        response = context.agent.run(request, artifact_dir=artifact_dir)

        if response.status != "success":
            return StageResult(
                status="continue",
                message="No debug fix available from agent.",
                payload={"trial_id": trial.trial_id},
            )

        structured = dict(response.structured or {})
        config_patches = dict(structured.get("config_patches", {}) or {})
        patch = dict(config_patches.get("training", {}) or structured.get("config_patch", {}) or {})
        patch, removed = self._sanitize_patch(context, patch)
        files_modified = list(structured.get("files_modified", []) or [])
        should_retry = bool(structured.get("should_retry", bool(patch or files_modified)))
        root_cause = str(structured.get("root_cause", ""))
        fix_strategy = str(structured.get("fix_strategy", response.content or ""))

        debug_note = "\n".join(
            [
                "# Debug Result",
                "",
                f"- root_cause: {root_cause}",
                f"- fix_strategy: {fix_strategy}",
                f"- files_modified: {files_modified}",
                f"- removed_keys: {removed}",
                f"- should_retry: {should_retry}",
                "",
            ]
        )
        (artifact_dir / "debug.md").write_text(debug_note, encoding="utf-8")

        if not should_retry or (not patch and not files_modified):
            trial.reflection = (trial.reflection + "\n\n" + debug_note).strip()
            context.state.update_trial(trial)
            return StageResult(
                status="continue",
                message="Debug finished without retry patch.",
                payload={"trial_id": trial.trial_id},
                artifacts={"debug": str(artifact_dir / "debug.md")},
            )

        # Both code-only and config-patch fixes retry the same trial as the next attempt.
        _runtime_keys = frozenset({"exp_dir", "stats_dir", "recipe_dir", "data_dir", "exp_tag"})

        trial.reflection = (trial.reflection + "\n\n" + debug_note).strip()

        if patch:
            # Re-resolve configs in-place with the combined (parent + debug) patch.
            parent_hp_patch = {k: v for k, v in trial.config_patch.items() if k not in _runtime_keys}
            combined_hp_patch = {**parent_hp_patch, **patch}
            runtime_patch = {
                "recipe_dir": str(context.recipe_dir),
                "data_dir": str(context.recipe_dir / "data"),
                "exp_tag": trial.trial_id,
                "exp_dir": str(artifact_dir / "exp"),
            }
            combined_patch = {**combined_hp_patch, **runtime_patch}

            train_cfg = load_recipe_stage_config(
                context.recipe_dir,
                Path(context.config.autoresearch.recipe.training_config),
                "training.yaml",
                resolve=False,
            )
            train_unresolved = OmegaConf.to_container(train_cfg, resolve=False)
            train_unresolved = apply_dotted_patch(train_unresolved, combined_patch)
            merged_training = OmegaConf.to_container(
                OmegaConf.create(train_unresolved), resolve=True
            )
            training_out = artifact_dir / "resolved_training_config.yaml"
            write_resolved_config(training_out, OmegaConf.create(merged_training))

            # Evaluation is immutable across attempts.  Debug patches may repair
            # training only; inference/metrics receive trial-local paths only.
            runtime_only_infer = {**runtime_patch, "inference_dir": str(artifact_dir / "inference")}
            for config_key, config_name, out_name in [
                ("inference_config", "inference.yaml", "resolved_inference_config.yaml"),
                ("metrics_config", "metrics.yaml", "resolved_metrics_config.yaml"),
            ]:
                cfg = load_recipe_stage_config(
                    context.recipe_dir,
                    Path(getattr(context.config.autoresearch.recipe, config_key)),
                    config_name,
                    resolve=False,
                )
                unresolved = OmegaConf.to_container(cfg, resolve=False)
                unresolved = apply_dotted_patch(unresolved, runtime_only_infer)
                plain = OmegaConf.to_container(OmegaConf.create(unresolved), resolve=True)
                write_resolved_config(artifact_dir / out_name, OmegaConf.create(plain))

            # Update trial config_patch and persist
            trial.config_patch = combined_hp_patch
            save_yaml(artifact_dir / "config_patch.yaml", combined_hp_patch, resolve=False)

        context.state.update_trial(trial)
        return StageResult(
            status="success",
            message=f"Debug fix applied to {trial.trial_id}; retrying as next attempt.",
            payload={"trial_id": trial.trial_id},
            artifacts={"debug": str(artifact_dir / "debug.md")},
        )
