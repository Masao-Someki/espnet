"""Propose the next hyperparameter trial."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from espnet3.autoresearch.agents.interface import AgentRequest
from espnet3.autoresearch.core.result import StageResult
from espnet3.autoresearch.core.serialization import (
    apply_dotted_patch,
    save_yaml,
    utc_now,
)
from espnet3.autoresearch.core.stage import AutoResearchStage
from espnet3.autoresearch.core.trial import Trial
from espnet3.autoresearch.observers.logs import (
    read_csv_tail,
    read_log_tail,
    summarize_recent_trial_curves,
)
from espnet3.autoresearch.applications.hyperparameter_search.config_schema import (
    load_recipe_stage_config,
    write_resolved_config,
)


def _collect_live_metrics_csv_paths(study_dir: Path) -> dict[str, str]:
    """Return paths to live metrics CSVs for all trials that have one."""
    result: dict[str, str] = {}
    trials_dir = study_dir / "trials"
    if not trials_dir.is_dir():
        return result
    for trial_dir in sorted(trials_dir.iterdir(), key=lambda p: p.name):
        if not trial_dir.is_dir():
            continue
        candidates = sorted(
            trial_dir.glob("exp/csv_logs/**/metrics.csv"),
            key=lambda p: p.stat().st_mtime,
        )
        if candidates:
            result[trial_dir.name] = str(candidates[-1])
    return result


def _glob_match(patterns, key: str) -> bool:
    import fnmatch

    return any(fnmatch.fnmatch(key, pattern) for pattern in patterns)


class ProposeTrialStage(AutoResearchStage):
    """Create a new Trial from agent output or fallback candidates."""

    study_lock_name = "study_controller"

    def _next_trial_id(self, trials) -> str:
        return f"trial_{len(trials) + 1:06d}"

    def _ensure_provisional_trial(self, context, latest_trial, trials) -> Trial:
        trial_id = self._next_trial_id(trials)
        trial_dir = context.study_dir / "trials" / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        trial = Trial(
            trial_id=trial_id,
            study_id=str(context.scheduler.study_id),
            status="proposing",
            config_patch={},
            resolved_config_path=None,
            rationale="",
            expected_effect="",
            risk="",
            score=None,
            score_name=str(context.config.autoresearch.metric.name),
            decision=None,
            parent_trial_id=latest_trial.trial_id if latest_trial else None,
            attempt_count=0,
            created_at=utc_now(),
            updated_at=utc_now(),
        )
        context.state.create_trial(trial)
        save_yaml(
            trial_dir / "trial.yaml",
            {"trial_id": trial_id, "status": trial.status, "pending_agent": None},
        )
        return trial

    def _sanitize_patch(self, context, patch: dict) -> tuple[dict, list[str]]:
        search_space = context.config.autoresearch.search_space
        allowed = list(getattr(search_space, "allowed_keys", []) or [])
        denied = list(getattr(search_space, "denied_keys", []) or [])
        sanitized: dict = {}
        removed: list[str] = []
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

    def _fallback_candidate(self, context, used_patches: list[dict]) -> dict | None:
        candidates = list(
            getattr(
                getattr(context.config.autoresearch, "search_space", None),
                "candidates",
                [],
            )
            or []
        )
        for candidate in candidates:
            plain = dict(candidate)
            if plain not in used_patches:
                return plain
        return None

    def _should_reuse_current_trial(self, context, current_trial: Trial | None) -> bool:
        if current_trial is None:
            return False
        if current_trial.status != "proposing":
            return False
        if current_trial.resolved_config_path:
            return False
        return True

    def _build_runtime_path_patch(
        self, context, trial_dir: Path, trial_id: str
    ) -> dict:
        return {
            "recipe_dir": str(context.recipe_dir),
            "data_dir": str(context.recipe_dir / "data"),
            "exp_tag": trial_id,
            "exp_dir": str(trial_dir / "exp"),
        }

    def _rewrite_logger_save_dirs(
        self, train_plain: dict, trial_dir: Path
    ) -> None:
        trainer = train_plain.get("trainer")
        if not isinstance(trainer, dict):
            return
        logger_cfg = trainer.get("logger")
        if isinstance(logger_cfg, dict):
            if "save_dir" in logger_cfg:
                logger_cfg["save_dir"] = str(trial_dir / "exp" / "csv_logs")
        elif isinstance(logger_cfg, list):
            for entry in logger_cfg:
                if isinstance(entry, dict) and "save_dir" in entry:
                    entry["save_dir"] = str(trial_dir / "exp" / "csv_logs")

    def run(self, context) -> StageResult:
        if context.scheduler.should_stop():
            return StageResult(status="no_budget", message="Budget exhausted")

        trials = context.state.list_trials(context.scheduler.study_id)
        used_patches = [t.config_patch for t in trials]
        latest_trial = trials[-1] if trials else None
        current_trial = context.current_trial

        _runtime_keys = frozenset({"exp_dir", "stats_dir", "recipe_dir", "data_dir", "exp_tag"})

        if self._should_reuse_current_trial(context, current_trial):
            trial = current_trial
        else:
            trial = self._ensure_provisional_trial(context, latest_trial, trials)

        trial_dir = context.study_dir / "trials" / trial.trial_id
        trial_id = trial.trial_id

        # Build inflight patches from trials currently in flight
        inflight_patches = []
        for t in trials:
            if t.status in frozenset({"running", "proposing", "proposed"}) and t.trial_id != trial_id:
                patch = {k: v for k, v in t.config_patch.items() if k not in _runtime_keys}
                inflight_patches.append(patch)

        # Knowledge and history
        knowledge_path = context.study_dir / "knowledge" / "knowledge_pack.md"
        knowledge = knowledge_path.read_text(encoding="utf-8") if knowledge_path.exists() else ""
        trials_csv = context.study_dir / "trials.csv"
        trial_history_csv = trials_csv.read_text(encoding="utf-8") if trials_csv.exists() else ""

        # Latest logs
        latest_logs = {
            "train": read_log_tail(trial_dir / "train.log"),
            "eval": read_log_tail(trial_dir / "eval.log"),
            "train_curve_csv_tail": read_csv_tail(trial_dir / "training_metrics.csv"),
            "recent_trial_curves": summarize_recent_trial_curves(
                context.study_dir, exclude_trial_id=trial_id
            ),
            "live_metrics_csv_paths": _collect_live_metrics_csv_paths(context.study_dir),
        }

        # Load all base configs so the agent can see what is patchable
        def _load_base_config_text(config_path_key: str, config_name: str) -> str:
            try:
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
            "## Base configs (read-only reference — propose patches against these)\n"
            "Runtime keys (exp_dir, stats_dir, recipe_dir, data_dir, exp_tag, inference_dir) "
            "are injected automatically — do NOT include them in your patches.\n",
            "### training config\n```yaml\n"
            + _load_base_config_text("training_config", "training.yaml")
            + "```",
            "### inference config\n```yaml\n"
            + _load_base_config_text("inference_config", "inference.yaml")
            + "```",
            "### metrics config\n```yaml\n"
            + _load_base_config_text("metrics_config", "metrics.yaml")
            + "```",
        ])

        repo_context_parts = []
        repo_context_path = context.study_dir / "repo_context.md"
        if repo_context_path.exists():
            repo_context_parts.append(repo_context_path.read_text(encoding="utf-8"))
        repo_context_parts.append(base_configs_context)
        repo_context = "\n\n".join(repo_context_parts)

        _ALLOWED_ACTIONS_BY_APPLICATION = {
            "hyperparameter_search": ["config_patch_only"],
            "architecture_search": ["config_patch", "implement_code", "pixi_add"],
        }
        application = str(
            getattr(context.config.autoresearch, "application", "hyperparameter_search")
        )
        allowed_actions = _ALLOWED_ACTIONS_BY_APPLICATION.get(
            application, ["config_patch_only"]
        )

        if "implement_code" in allowed_actions:
            extra_stages_hint = (
                "list of stage names to run before train — "
                "include 'implement_code' to write/modify src/ files for a new architecture, "
                "or 'train_tokenizer' if a new tokenizer is needed"
            )
        else:
            extra_stages_hint = "optional list of stage names to run before train, e.g. train_tokenizer"

        request = AgentRequest(
            task="propose",
            objective=(context.study_dir / "program.md").read_text(encoding="utf-8"),
            knowledge=knowledge,
            repo_context=repo_context,
            trial_history_csv=trial_history_csv,
            latest_metrics={},
            latest_logs=latest_logs,
            inflight_patches=inflight_patches,
            allowed_actions=allowed_actions,
            output_schema={
                "rationale": "string",
                "config_patches": {
                    "training": {"dotted.key": "value (required — HP changes go here)"},
                    "inference": {"dotted.key": "value (optional — only if inference.yaml must change). IMPORTANT: use only concrete resolved values here (strings, numbers, booleans). Do NOT use OmegaConf interpolations like ${key} — they will not resolve in inference.yaml context."},
                    "metrics": {"dotted.key": "value (optional — only if metrics.yaml must change)"},
                },
                "expected_effect": "string",
                "risk": "string",
                "extra_stages": [extra_stages_hint],
            },
        )
        artifact_dir = trial_dir
        response = context.agent.run(request, artifact_dir=artifact_dir)

        if response.status != "success":
            agent_type = str(context.config.autoresearch.agent.type)
            if agent_type == "file":
                return StageResult(
                    status="waiting_agent",
                    message=response.message,
                    payload={"trial_id": trial_id},
                    artifacts={"agent_request": str(artifact_dir / "agent_request.md")},
                )
            # Try fallback candidate
            fallback = self._fallback_candidate(context, used_patches)
            if fallback is None:
                trial.status = "cancelled"
                context.state.update_trial(trial)
                return StageResult(
                    status="cancelled",
                    message="No remaining candidates",
                    payload={"trial_id": trial_id},
                )
            patch = fallback
            rationale = "Fallback candidate from configured search_space."
            expected_effect = "Explore configured candidate."
            risk = "May not improve the current best."
            extra_stages: list[str] = []
            removed: list[str] = []
            extra_config_patches: dict = {}
        else:
            structured = dict(response.structured or {})
            # Support both new multi-config format and legacy single config_patch
            config_patches = dict(structured.get("config_patches", {}) or {})
            patch = dict(config_patches.get("training", {}) or structured.get("config_patch", {}) or {})
            extra_config_patches = {
                k: dict(v or {})
                for k, v in config_patches.items()
                if k != "training" and v
            }
            rationale = str(structured.get("rationale", response.content or ""))
            expected_effect = str(structured.get("expected_effect", ""))
            risk = str(structured.get("risk", ""))
            _known_stages = {"create_dataset", "train_tokenizer", "collect_stats", "implement_code", "write_files"}
            extra_stages = [
                str(s) for s in (structured.get("extra_stages") or [])
                if str(s) in _known_stages
            ]

            patch, removed = self._sanitize_patch(context, patch)
            if not patch:
                trial.status = "cancelled"
                context.state.update_trial(trial)
                return StageResult(
                    status="failure",
                    message=f"Patch became empty after validation. removed={removed}",
                    payload={"trial_id": trial_id},
                )

            # Deduplicate
            hp_patch = {k: v for k, v in patch.items() if k not in _runtime_keys}
            for existing in trials:
                existing_hp = {k: v for k, v in existing.config_patch.items() if k not in _runtime_keys}
                if existing_hp == hp_patch and existing.trial_id != trial_id:
                    dup_id = existing.trial_id
                    trial.status = "cancelled"
                    context.state.update_trial(trial)
                    return StageResult(
                        status="failure",
                        message=f"Duplicate patch: same as {dup_id}",
                        payload={"trial_id": trial_id},
                    )

        # Build runtime path patch and merge
        runtime_patch = self._build_runtime_path_patch(context, trial_dir, trial_id)
        effective_patch = {**patch, **runtime_patch}

        # Load training config without resolving so that OmegaConf interpolations
        # (e.g. model.vocab_size: ${tokenizer.vocab_size}) remain as strings.
        # Applying the patch before resolution ensures dependent fields propagate
        # correctly when tokenizer settings change.
        train_cfg = load_recipe_stage_config(
            context.recipe_dir,
            Path(context.config.autoresearch.recipe.training_config),
            "training.yaml",
            resolve=False,
        )
        train_unresolved = OmegaConf.to_container(train_cfg, resolve=False)
        train_unresolved = apply_dotted_patch(train_unresolved, effective_patch)
        train_plain = OmegaConf.to_container(
            OmegaConf.create(train_unresolved), resolve=True
        )
        self._rewrite_logger_save_dirs(train_plain, trial_dir)
        training_out = trial_dir / "resolved_training_config.yaml"
        write_resolved_config(training_out, OmegaConf.create(train_plain))

        # Resolve inference and metrics configs: runtime keys + agent patches + resolve.
        # Agent may supply extra patches via config_patches.inference / config_patches.metrics.
        # Pre-resolve any OmegaConf interpolations in effective_patch using train_plain so that
        # keys like ${tokenizer.save_path} resolve correctly even in configs that lack those keys.
        def _pre_resolve_patch(patch: dict, resolved_train: dict) -> dict:
            def _get_dotted(d: dict, dotted_key: str):
                keys = dotted_key.split(".")
                cur = d
                for k in keys:
                    if not isinstance(cur, dict) or k not in cur:
                        return None
                    cur = cur[k]
                return cur

            result = {}
            for k, v in patch.items():
                if isinstance(v, str) and "${" in v:
                    resolved_v = _get_dotted(resolved_train, k)
                    result[k] = resolved_v if resolved_v is not None else v
                else:
                    result[k] = v
            return result

        runtime_with_infer_dir = _pre_resolve_patch(
            {**effective_patch, "inference_dir": str(trial_dir / "inference")},
            train_plain,
        )

        def _key_exists_in_config(cfg: dict, dotted_key: str) -> bool:
            keys = dotted_key.split(".")
            cur = cfg
            for k in keys:
                if not isinstance(cur, dict) or k not in cur:
                    return False
                cur = cur[k]
            return True

        for config_key, config_name, out_name in [
            ("inference_config", "inference.yaml", "resolved_inference_config.yaml"),
            ("metrics_config", "metrics.yaml", "resolved_metrics_config.yaml"),
        ]:
            cfg_name_stem = config_name.split(".")[0]  # "inference" or "metrics"
            agent_patch = extra_config_patches.get(cfg_name_stem, {})
            cfg = load_recipe_stage_config(
                context.recipe_dir,
                Path(getattr(context.config.autoresearch.recipe, config_key)),
                config_name,
                resolve=False,
            )
            unresolved = OmegaConf.to_container(cfg, resolve=False)
            # Filter out training-only HP keys that don't exist in this config's base
            # structure (e.g. model.specaug_conf.* or dataset.preprocessor.* from a
            # training patch must not bleed into inference/metrics configs).
            safe_patch = {
                k: v for k, v in runtime_with_infer_dir.items()
                if k in _runtime_keys or k == "inference_dir"
                or _key_exists_in_config(unresolved, k)
            }
            unresolved = apply_dotted_patch(unresolved, {**safe_patch, **agent_patch})
            plain = OmegaConf.to_container(OmegaConf.create(unresolved), resolve=True)
            write_resolved_config(trial_dir / out_name, OmegaConf.create(plain))

        inference_out = trial_dir / "resolved_inference_config.yaml"
        metrics_out = trial_dir / "resolved_metrics_config.yaml"

        # Save metadata
        hp_patch_only = {k: v for k, v in effective_patch.items() if k not in _runtime_keys}
        save_yaml(trial_dir / "config_patch.yaml", hp_patch_only, resolve=False)
        (trial_dir / "rationale.md").write_text(rationale, encoding="utf-8")

        # Update trial
        trial.status = "proposed"
        trial.config_patch = effective_patch
        trial.resolved_config_path = training_out
        trial.rationale = rationale
        trial.expected_effect = expected_effect
        trial.risk = risk
        trial.updated_at = utc_now()
        if response.status == "success" and getattr(response, "thread_id", None):
            trial.codex_thread_id = response.thread_id
        trial.extra_stages = extra_stages
        context.state.update_trial(trial)
        save_yaml(
            trial_dir / "trial.yaml",
            {
                "trial_id": trial_id,
                "status": trial.status,
                "config_patch": hp_patch_only,
                "rationale": rationale,
                "expected_effect": expected_effect,
                "risk": risk,
                "removed_keys": removed,
                "extra_stages": extra_stages,
                "codex_thread_id": trial.codex_thread_id,
            },
            resolve=False,
        )

        if "write_files" in extra_stages:
            next_node = "write_files"
        elif "implement_code" in extra_stages:
            next_node = "implement_code"
        else:
            next_node = None
        return StageResult(
            status=next_node if next_node else "success",
            next_node=next_node,
            message=f"Proposed {trial_id}",
            payload={"trial_id": trial_id},
            artifacts={
                "trial_dir": str(trial_dir),
                "training_config": str(training_out),
                "inference_config": str(inference_out),
                "metrics_config": str(metrics_out),
            },
        )
