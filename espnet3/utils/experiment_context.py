"""Persisted experiment context for ESPnet3 runner entry points.

An `ExperimentContext` captures the experiment identity (where the recipe
lives, which experiment tag/directory a run uses, and the output locations
derived from it) that `launch()` needs to pass between stages, and that a
standalone invocation (e.g. requesting only `measure` with no training
config) needs to recover without re-deriving it from scratch.

`build_experiment_context` derives an `ExperimentContext` from the loaded,
resolved runner configs (preferring `training_config` when present, falling
back to a previously saved context for `exp_dir`, and finally to whichever
standalone config carries its own identity). `save_experiment_context`
persists that context under `<exp_dir>/config/context.yaml` so a later,
separate process can recover it with `load_experiment_context`.
`save_stage_config` separately snapshots one stage's own config role, as it
stood right before that stage ran, under `<exp_dir>/config/<stage>.yaml`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from omegaconf import DictConfig, OmegaConf

from espnet3.components.contract.stages import stage_spec
from espnet3.utils.run_utils import (
    ExperimentContextError,
    _copy_config_context,
    apply_training_experiment_context,
)

_CONFIG_DIRNAME = "config"
_CONTEXT_FILENAME = "context.yaml"
_HISTORY_DIRNAME = "history"
_SCHEMA_VERSION = 1
_IDENTITY_FIELDS = ("recipe_dir", "exp_tag", "exp_dir", "stats_dir")

#: Config roles consulted, in this fixed order, when neither `training_config`
#: nor a saved `exp_dir` context is available to supply identity.
_FALLBACK_ROLE_ORDER = ("inference", "metrics", "publication", "demo")


@dataclass(frozen=True)
class ExperimentContext:
    """Immutable summary of one experiment's identity and output locations.

    All path-like fields are kept exactly as they appear in the resolved
    config (typically relative to `recipe_dir`, per the TEMPLATE convention).
    Nothing here is absolutized: a saved context must stay valid if the whole
    recipe directory is copied elsewhere.

    Attributes:
        recipe_dir: Resolved `recipe_dir` string (TEMPLATE default `"."`).
        exp_tag: Resolved experiment tag.
        exp_dir: Resolved experiment directory.
        stats_dir: `collect_stats` output directory, when the experiment has
            a training config. `None` for standalone inference/metrics runs.
        data_dir: Dataset directory, when known.
        inference_dir: Inference output directory, when known (populated by
            whichever of inference/metrics/publication/demo declares it).
        tokenizer_dir: `training_config.tokenizer.save_path`, when the
            training config declares a tokenizer (ASR only; `None` for TTS).
        sources: Reserved for a future record of which file backed each
            config role; always empty in this release.

    Examples:
        >>> ctx = ExperimentContext(
        ...     recipe_dir=".", exp_tag="train_asr", exp_dir="./exp/train_asr"
        ... )
        >>> ctx.exp_tag, ctx.stats_dir
        ('train_asr', None)
    """

    recipe_dir: str
    exp_tag: str
    exp_dir: str
    stats_dir: Optional[str] = None
    data_dir: Optional[str] = None
    inference_dir: Optional[str] = None
    tokenizer_dir: Optional[str] = None
    sources: Dict[str, str] = field(default_factory=dict)


def _is_missing_or_empty(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _get_str(config: Optional[DictConfig], key: str) -> Optional[str]:
    """Return `config[key]` as a non-empty string, or `None`."""
    if config is None:
        return None
    value = config.get(key)
    return None if _is_missing_or_empty(value) else value


def _tokenizer_dir(training_config: Optional[DictConfig]) -> Optional[str]:
    if training_config is None:
        return None
    tokenizer_cfg = training_config.get("tokenizer")
    if tokenizer_cfg is None or not hasattr(tokenizer_cfg, "get"):
        return None
    return _get_str(tokenizer_cfg, "save_path")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _history_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _canonical_json(obj: Any) -> str:
    """Serialize `obj` the same way every time, for stable fingerprinting."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


def compute_fingerprint(obj: Any) -> str:
    """Return the sha256 hex digest of `obj`'s canonical JSON form.

    Used for the experiment identity fingerprint, so re-running with an
    unchanged identity is detectable without a byte-for-byte file comparison.

    Examples:
        >>> compute_fingerprint({"a": 1}) == compute_fingerprint({"a": 1})
        True
        >>> compute_fingerprint({"a": 1}) == compute_fingerprint({"a": 2})
        False
    """
    return hashlib.sha256(_canonical_json(obj).encode("utf-8")).hexdigest()


def _identity_payload(context: ExperimentContext) -> Dict[str, Optional[str]]:
    return {name: getattr(context, name) for name in _IDENTITY_FIELDS}


def _git_sha() -> Optional[str]:
    """Best-effort `git rev-parse HEAD` for this checkout, or `None`."""
    repo_dir = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except Exception:
        return None
    sha = result.stdout.strip()
    return sha or None


_SAVED_CONTEXT_KEYS = ("exp_tag", "exp_dir", "inference_dir")


def _fill_role_config_from_saved(
    role_config: Optional[DictConfig],
    role_name: str,
    saved: "ExperimentContext",
    log: logging.Logger,
) -> None:
    """Fill `role_config`'s empty identity keys from a saved context.

    Only `exp_tag`/`exp_dir`/`inference_dir` keys that are currently empty
    on `role_config` are filled; a key that already carries a value is left
    untouched. This is what lets a standalone `--exp_dir` run (e.g.
    `measure` alone, no `training_config`) recover `inference_dir` from a
    context a previous `train`/`infer` run saved, instead of only reaching
    `ExperimentContext.inference_dir` (which nothing reads back into the
    role config itself).

    Args:
        role_config: One run's inference/metrics/publication/demo config, or
            `None` when that role was not given.
        role_name: The role's name (e.g. `"metrics"`), used in log/error
            messages.
        saved: The context loaded from `<exp_dir>/config/context.yaml`.
        log: Logger used for the `inference_dir`-conflict warning.

    Raises:
        ExperimentContextError: `role_config` already has its own `exp_dir`
            and it differs from `saved.exp_dir`.
    """
    if role_config is None:
        return

    saved_values = {
        "exp_tag": saved.exp_tag,
        "exp_dir": saved.exp_dir,
        "inference_dir": saved.inference_dir,
    }
    empty_keys = [
        key
        for key in _SAVED_CONTEXT_KEYS
        if _is_missing_or_empty(role_config.get(key) if key in role_config else None)
    ]
    if empty_keys:
        _copy_config_context(
            source=OmegaConf.create(saved_values),
            target=role_config,
            keys=empty_keys,
            source_name="saved_context",
            target_name=f"{role_name}_config",
            log=log,
        )

    if "exp_dir" not in empty_keys:
        current = _get_str(role_config, "exp_dir")
        if current and saved.exp_dir and current != saved.exp_dir:
            raise ExperimentContextError(
                f"{role_name}_config.exp_dir {current!r} does not match the "
                f"context already saved at {saved.exp_dir!r}. Use a "
                "different exp_dir, or drop the explicit exp_dir to reuse "
                "the saved experiment."
            )

    if "inference_dir" not in empty_keys:
        current = _get_str(role_config, "inference_dir")
        if current and saved.inference_dir and current != saved.inference_dir:
            log.warning(
                "%s_config.inference_dir %r differs from the saved "
                "context's inference_dir %r; keeping %s_config's own value.",
                role_name,
                current,
                saved.inference_dir,
                role_name,
            )


def build_experiment_context(
    *,
    training_config: Optional[DictConfig] = None,
    inference_config: Optional[DictConfig] = None,
    metrics_config: Optional[DictConfig] = None,
    publication_config: Optional[DictConfig] = None,
    demo_config: Optional[DictConfig] = None,
    exp_dir: Optional[str] = None,
    roles: Sequence[str] = (),
    log: Optional[logging.Logger] = None,
) -> ExperimentContext:
    """Derive the `ExperimentContext` for the current runner invocation.

    Propagation across the provided configs is delegated to the existing
    `apply_training_experiment_context` (so `exp_tag`/`exp_dir`/`inference_dir`
    end up on the configs that need them), then the identity fields are read
    back out in this priority order:

    1. `training_config`, when present, is authoritative.
    2. Otherwise, a previously saved context at `<exp_dir>/config/context.yaml`
       (via `load_experiment_context`), when `exp_dir` is given.
    3. Otherwise, the first config named by `roles` that was actually given
       (e.g. `roles=("metrics",)` prefers `metrics_config`); if `roles` is
       empty or none of the named configs were given, falls back to the
       fixed order inference, metrics, publication, demo.

    Args:
        training_config: Training config selected for the current run.
        inference_config: Inference config for the `infer` stage.
        metrics_config: Metrics config for the `measure` stage.
        publication_config: Publication config for `pack_model`/`upload_model`.
        demo_config: Demo config for `pack_demo`/`upload_demo`.
        exp_dir: `--exp_dir` value, used both as a fallback identity source
            (case 2 above) and, when none of the configs declare a concrete
            `exp_dir`, as the resulting context's `exp_dir`.
        roles: The config roles the current run's requested stages actually
            need, in priority order (e.g. `("metrics",)` for a `measure`-only
            run). Affects only the case-3 fallback above; it has no bearing
            on `inference_dir` or `tokenizer_dir` derivation. Stage names
            never appear here - a caller that knows stage names maps them to
            roles itself (one role per stage is
            `espnet3.components.contract.stages.stage_spec(...).config`).
        log: Logger used by the underlying propagation helper.

    Returns:
        ExperimentContext: The derived context. Identity fields may be empty
        strings/`None` if none of the three sources above had them; callers
        are expected to validate sufficiency separately (this function only
        derives, it does not validate).

    Examples:
        Training-backed run, identity comes from `training_config`:

        ```python
        training = OmegaConf.create(
            {"exp_tag": "train_asr", "exp_dir": "./exp/train_asr",
             "stats_dir": "./exp/stats"}
        )
        ctx = build_experiment_context(training_config=training)
        # ctx.exp_tag == "train_asr", ctx.exp_dir == "./exp/train_asr"
        ```

        Standalone `measure`, recovering identity from a saved context:

        ```python
        ctx = build_experiment_context(
            metrics_config=OmegaConf.create({}),
            exp_dir="./exp/train_asr",
            roles=("metrics",),
        )
        # ctx is read from ./exp/train_asr/config/context.yaml
        ```
    """
    log = log or logging.getLogger(__name__)

    role_configs = {
        "inference": inference_config,
        "metrics": metrics_config,
        "publication": publication_config,
        "demo": demo_config,
    }

    # Snapshot each candidate's *own* identity before propagation below can
    # overwrite it (e.g. inference's exp_tag/exp_dir always wins over
    # metrics's when both are given) - `roles` must rank each config's own
    # declared identity, not whatever another role's propagation left behind.
    identity_source: Optional[DictConfig] = None
    if training_config is not None:
        identity_source = training_config
    else:
        for role in tuple(roles) + _FALLBACK_ROLE_ORDER:
            candidate = role_configs.get(role)
            if candidate is not None and (
                _get_str(candidate, "exp_tag") or _get_str(candidate, "exp_dir")
            ):
                identity_source = candidate
                break

    own_identity = None
    if identity_source is not None:
        own_identity = {
            "recipe_dir": _get_str(identity_source, "recipe_dir") or ".",
            "exp_tag": _get_str(identity_source, "exp_tag") or "",
            "exp_dir": _get_str(identity_source, "exp_dir") or "",
            "stats_dir": _get_str(identity_source, "stats_dir"),
            "data_dir": _get_str(identity_source, "data_dir"),
        }

    saved: Optional[ExperimentContext] = None
    if exp_dir is not None:
        try:
            saved = load_experiment_context(exp_dir)
        except ExperimentContextError:
            saved = None  # Nothing saved yet at exp_dir; nothing to reconcile.

    if saved is not None:
        # Recover identity/inference_dir from a previous run's saved context
        # into this run's own role configs, before validation sees them -
        # e.g. a standalone `measure` run (no training_config) needs
        # metrics_config.inference_dir filled in from here.
        for role_name, role_config in role_configs.items():
            _fill_role_config_from_saved(role_config, role_name, saved, log)

    # Propagate identity/inference_dir among this run's own configs
    # (training_config -> inference/metrics/publication/demo, and
    # inference_config -> metrics/publication) regardless of whether
    # training_config is given, so a standalone --inference_config +
    # --metrics_config run still has inference_dir propagated between them.
    apply_training_experiment_context(
        training_config=training_config,
        inference_config=inference_config,
        metrics_config=metrics_config,
        publication_config=publication_config,
        demo_config=demo_config,
        log=log,
    )

    if own_identity is not None:
        if saved is not None:
            saved_identity = (
                saved.recipe_dir,
                saved.exp_tag,
                saved.exp_dir,
                saved.stats_dir,
            )
            this_identity = (
                own_identity["recipe_dir"],
                own_identity["exp_tag"],
                own_identity["exp_dir"],
                own_identity["stats_dir"],
            )
            if this_identity != saved_identity:
                raise ExperimentContextError(
                    f"This run's experiment identity {this_identity} does "
                    f"not match the context already saved at {exp_dir} "
                    f"{saved_identity}. Use a different exp_tag/exp_dir, or "
                    "pass --exp_dir only (no --training_config/explicit "
                    "exp_dir) to reuse the saved experiment."
                )
        ctx_recipe_dir = own_identity["recipe_dir"]
        exp_tag = own_identity["exp_tag"]
        resolved_exp_dir = own_identity["exp_dir"] or (exp_dir or "")
        stats_dir = own_identity["stats_dir"]
        data_dir = own_identity["data_dir"]
        tokenizer_dir = _tokenizer_dir(training_config)
    elif saved is not None:
        ctx_recipe_dir = saved.recipe_dir
        exp_tag = saved.exp_tag
        resolved_exp_dir = exp_dir or saved.exp_dir
        stats_dir = saved.stats_dir
        data_dir = saved.data_dir
        tokenizer_dir = saved.tokenizer_dir
    else:
        ctx_recipe_dir = "."
        exp_tag = ""
        resolved_exp_dir = exp_dir or ""
        stats_dir = None
        data_dir = None
        tokenizer_dir = None

    inference_dir = (
        _get_str(inference_config, "inference_dir")
        or _get_str(metrics_config, "inference_dir")
        or _get_str(publication_config, "inference_dir")
        or _get_str(demo_config, "inference_dir")
    )
    if saved is not None and saved.inference_dir is not None:
        if inference_dir is None:
            inference_dir = saved.inference_dir
        elif inference_dir != saved.inference_dir:
            log.warning(
                "inference_dir %r from this run's configs differs from the "
                "saved context's inference_dir %r; using this run's value.",
                inference_dir,
                saved.inference_dir,
            )

    return ExperimentContext(
        recipe_dir=ctx_recipe_dir,
        exp_tag=exp_tag,
        exp_dir=resolved_exp_dir,
        stats_dir=stats_dir,
        data_dir=data_dir,
        inference_dir=inference_dir,
        tokenizer_dir=tokenizer_dir,
    )


def save_experiment_context(
    exp_dir: str,
    context: ExperimentContext,
    *,
    overwrite_context: bool = False,
    stages: Sequence[str] = (),
    argv: Sequence[str] = (),
    log: Optional[logging.Logger] = None,
) -> Path:
    """Persist `context` to `<exp_dir>/config/context.yaml`.

    Re-running against an existing `context.yaml`: if the saved identity
    (`recipe_dir`/`exp_tag`/`exp_dir`/`stats_dir`) differs from `context`'s,
    this is a different experiment reusing the same `exp_dir` - raises
    `ExperimentContextError` unless `overwrite_context=True`, in which case
    the old `context.yaml` is moved to `config/history/` (with a `WARNING`)
    and a fresh one is written. Neither `recipe_dir` nor any other path on
    `context` is absolutized before writing.

    Args:
        exp_dir: Experiment directory; `<exp_dir>/config/` is created if
            missing.
        context: The context to persist (see `build_experiment_context`).
        overwrite_context: Allow replacing a saved context for a different
            experiment identity (normally an error).
        stages: Stage names requested on this run, recorded under
            `last_run.stages`.
        argv: The invoking command line, recorded under `last_run.argv`.
        log: Logger used for the overwrite warning.

    Returns:
        Path: The written `context.yaml` path.

    Raises:
        ExperimentContextError: `exp_dir` already holds a saved context for a
            different experiment identity and `overwrite_context` is `False`.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as exp_dir:
        ...     ctx = ExperimentContext(
        ...         recipe_dir=".", exp_tag="t", exp_dir=exp_dir
        ...     )
        ...     path = save_experiment_context(exp_dir, ctx, stages=["train"])
        ...     path.name
        'context.yaml'
    """
    log = log or logging.getLogger(__name__)

    config_dir = Path(exp_dir) / _CONFIG_DIRNAME
    config_dir.mkdir(parents=True, exist_ok=True)
    history_dir = config_dir / _HISTORY_DIRNAME

    identity_fingerprint = compute_fingerprint(_identity_payload(context))

    context_path = config_dir / _CONTEXT_FILENAME
    if context_path.exists():
        old_payload = OmegaConf.to_container(OmegaConf.load(context_path))
        old_identity_fp = (old_payload.get("fingerprints") or {}).get("identity")
        if old_identity_fp is not None and old_identity_fp != identity_fingerprint:
            if not overwrite_context:
                raise ExperimentContextError(
                    f"{context_path} already holds a saved experiment "
                    "context for a different experiment (recipe_dir/exp_tag/"
                    "exp_dir/stats_dir do not match this run). Pass "
                    "overwrite_context=True to replace it, or use a "
                    "different exp_tag/exp_dir."
                )
            log.warning(
                "Overwriting saved experiment context at %s for a different "
                "experiment identity (overwrite_context=True).",
                context_path,
            )
            history_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(
                str(context_path),
                str(history_dir / f"context.{_history_timestamp()}.yaml"),
            )

    payload = {
        "schema_version": _SCHEMA_VERSION,
        "written_at": _now_iso(),
        "espnet_git_sha": _git_sha(),
        "recipe_dir": context.recipe_dir,
        "recipe_dir_abs": str(Path(context.recipe_dir).resolve()),
        "exp_tag": context.exp_tag,
        "exp_dir": context.exp_dir,
        "stats_dir": context.stats_dir,
        "data_dir": context.data_dir,
        "inference_dir": context.inference_dir,
        "tokenizer_dir": context.tokenizer_dir,
        "sources": dict(context.sources),
        "fingerprints": {"identity": identity_fingerprint},
        "last_run": {
            "stages": list(stages),
            "argv": list(argv),
        },
    }
    OmegaConf.save(OmegaConf.create(payload), context_path)
    return context_path


def save_stage_config(
    system: Any,
    stage: str,
    *,
    dry_run: bool = False,
    rank: int = 0,
    log: Optional[logging.Logger] = None,
) -> Optional[Path]:
    """Snapshot one stage's config role to `<exp_dir>/config/<stage>.yaml`.

    Writes the stage's declared config role (`stage_spec(type(system),
    stage).config`, read off `system` as `getattr(system, f"{role}_config")`)
    exactly as it stands right before the stage runs - the readonly original
    held by `system`, not the writable per-call copy a stage mutates, so this
    is always the input, before any in-stage pop or edit. The destination is
    `system.exp_dir` (the system's own resolved experiment directory, set
    once at construction), not any path read out of the role config itself.

    Args:
        system: The system instance about to run `stage`.
        stage: The stage name about to run, e.g. `"train"`.
        dry_run: Skip writing (a dry run makes no on-disk changes).
        rank: Process rank; only rank 0 writes, to avoid concurrent writers
            in a multi-process launch.
        log: Logger used to note a skipped or written snapshot.

    Returns:
        Path | None: The written `<stage>.yaml` path, or `None` if skipped
        (`dry_run`, `rank != 0`, the role's config is `None`, or
        `system.exp_dir` is `None`).

    Examples:
        >>> from omegaconf import OmegaConf
        >>> from espnet3.components.contract.stages import StageSpec
        >>> class ExampleSystem:
        ...     stages = (StageSpec("train", "training", "exp_dir"),)
        ...     def __init__(self, exp_dir):
        ...         self.exp_dir = exp_dir
        ...         self.training_config = OmegaConf.create(
        ...             {"exp_dir": exp_dir}
        ...         )
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as exp_dir:
        ...     path = save_stage_config(ExampleSystem(exp_dir), "train")
        ...     path.name
        'train.yaml'
    """
    log = log or logging.getLogger(__name__)
    if dry_run or rank != 0:
        return None

    role = stage_spec(type(system), stage).config
    config = getattr(system, f"{role}_config", None)
    if config is None:
        return None

    exp_dir = getattr(system, "exp_dir", None)
    if exp_dir is None:
        log.warning(
            "Skipped saving config/%s.yaml: system has no exp_dir.",
            stage,
        )
        return None

    config_dir = Path(exp_dir) / _CONFIG_DIRNAME
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{stage}.yaml"
    new_text = OmegaConf.to_yaml(config, resolve=True)

    if path.exists():
        if path.read_text() == new_text:
            return path  # Unchanged since the last run; leave the file as-is.
        history_dir = config_dir / _HISTORY_DIRNAME
        history_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(
            str(path), str(history_dir / f"{stage}.{_history_timestamp()}.yaml")
        )
        log.warning(
            "%s config changed since the last saved run; moved the "
            "previous %s to %s.",
            role,
            path.name,
            history_dir,
        )

    path.write_text(new_text)
    log.info("Saved %s config for stage %r to %s.", role, stage, path)
    return path


def load_experiment_context(exp_dir: str) -> ExperimentContext:
    """Load the `ExperimentContext` last saved for `exp_dir`.

    Args:
        exp_dir: Experiment directory whose `config/context.yaml` should be
            read.

    Returns:
        ExperimentContext: Reconstructed from the saved `context.yaml`.

    Raises:
        ExperimentContextError: `<exp_dir>/config/context.yaml` does not
            exist.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as exp_dir:
        ...     ctx = ExperimentContext(
        ...         recipe_dir=".", exp_tag="t", exp_dir=exp_dir
        ...     )
        ...     _ = save_experiment_context(exp_dir, ctx)
        ...     load_experiment_context(exp_dir).exp_tag
        't'
    """
    context_path = Path(exp_dir) / _CONFIG_DIRNAME / _CONTEXT_FILENAME
    if not context_path.exists():
        raise ExperimentContextError(
            f"{context_path} not found; run training stages first or pass "
            "the configs explicitly"
        )
    payload = OmegaConf.to_container(OmegaConf.load(context_path))
    return ExperimentContext(
        recipe_dir=payload.get("recipe_dir", "."),
        exp_tag=payload.get("exp_tag", ""),
        exp_dir=payload.get("exp_dir", str(exp_dir)),
        stats_dir=payload.get("stats_dir"),
        data_dir=payload.get("data_dir"),
        inference_dir=payload.get("inference_dir"),
        tokenizer_dir=payload.get("tokenizer_dir"),
        sources=dict(payload.get("sources") or {}),
    )
