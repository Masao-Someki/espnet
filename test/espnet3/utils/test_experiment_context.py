"""Tests for `espnet3.utils.experiment_context`."""

import logging
from typing import ClassVar

import pytest
from omegaconf import OmegaConf

from espnet3.components.contract.stages import StageSpec
from espnet3.utils.experiment_context import (
    ExperimentContext,
    build_experiment_context,
    compute_fingerprint,
    load_experiment_context,
    save_experiment_context,
    save_stage_config,
)
from espnet3.utils.run_utils import (
    ExperimentContextError,
    validate_experiment_context,
)

_LOG = logging.getLogger("test_experiment_context")


class _FakeSystem:
    """Minimal stand-in for `BaseSystem`: just what `save_stage_config` reads.

    The stage -> config role mapping comes from the stages contract
    (`StageSpec.config`), the same source `stage_log_dir` uses.
    """

    stages: ClassVar[tuple[StageSpec, ...]] = (
        StageSpec("collect_stats", "training"),
        StageSpec("train", "training"),
        StageSpec("measure", "metrics"),
    )

    def __init__(self, exp_dir=None, **role_configs):
        self.exp_dir = exp_dir
        for role, cfg in role_configs.items():
            setattr(self, f"{role}_config", cfg)


def _training_config(tmp_path, exp_tag="train_asr_transformer"):
    return OmegaConf.create(
        {
            "exp_tag": exp_tag,
            "exp_dir": f"./exp/{exp_tag}",
            "stats_dir": "./exp/stats",
            "data_dir": "./data",
            "tokenizer": {"save_path": "./data/bpe_5000"},
        }
    )


def test_build_experiment_context_from_training_config():
    training = _training_config(None)
    ctx = build_experiment_context(training_config=training, log=_LOG)

    assert ctx.recipe_dir == "."
    assert ctx.exp_tag == "train_asr_transformer"
    assert ctx.exp_dir == "./exp/train_asr_transformer"
    assert ctx.stats_dir == "./exp/stats"
    assert ctx.data_dir == "./data"
    assert ctx.tokenizer_dir == "./data/bpe_5000"
    assert ctx.inference_dir is None


def test_build_experiment_context_propagates_inference_dir_from_inference_config():
    training = _training_config(None)
    inference = OmegaConf.create(
        {"inference_dir": "./exp/train_asr_transformer/inference"}
    )
    metrics = OmegaConf.create({})

    ctx = build_experiment_context(
        training_config=training,
        inference_config=inference,
        metrics_config=metrics,
        roles=("training", "inference", "metrics"),
        log=_LOG,
    )

    assert ctx.inference_dir == "./exp/train_asr_transformer/inference"
    # apply_training_experiment_context propagated identity onto metrics too.
    assert metrics.exp_tag == "train_asr_transformer"


def test_build_experiment_context_standalone_from_inference_config():
    inference = OmegaConf.create(
        {"exp_tag": "eval_debug", "exp_dir": "./exp/eval_debug"}
    )

    ctx = build_experiment_context(
        inference_config=inference, roles=("inference",), log=_LOG
    )

    assert ctx.exp_tag == "eval_debug"
    assert ctx.exp_dir == "./exp/eval_debug"
    assert ctx.stats_dir is None


def _standalone_inference_and_metrics_configs():
    """Fresh inference/metrics configs, each with its own identity.

    A new pair per call: `build_experiment_context` propagates identity
    across the configs it is given, so reusing one pair across two calls
    would let the first call's propagation leak into the second.
    """
    inference = OmegaConf.create(
        {"exp_tag": "from_inference", "exp_dir": "./exp/from_inference"}
    )
    metrics = OmegaConf.create(
        {"exp_tag": "from_metrics", "exp_dir": "./exp/from_metrics"}
    )
    return inference, metrics


def test_build_experiment_context_roles_order_picks_fallback_identity():
    # Neither config has training_config to anchor to; two standalone configs
    # each carry their own identity. `roles` says which one wins.
    inference, metrics = _standalone_inference_and_metrics_configs()
    by_inference = build_experiment_context(
        inference_config=inference,
        metrics_config=metrics,
        roles=("inference", "metrics"),
        log=_LOG,
    )

    inference, metrics = _standalone_inference_and_metrics_configs()
    by_metrics = build_experiment_context(
        inference_config=inference,
        metrics_config=metrics,
        roles=("metrics", "inference"),
        log=_LOG,
    )

    assert by_inference.exp_tag == "from_inference"
    assert by_metrics.exp_tag == "from_metrics"


def test_build_experiment_context_falls_back_to_saved_context(tmp_path):
    training = _training_config(tmp_path)
    exp_dir = tmp_path / "exp" / "train_asr_transformer"
    ctx = build_experiment_context(
        training_config=training, roles=("training",), log=_LOG
    )
    save_experiment_context(str(exp_dir), ctx)

    metrics = OmegaConf.create({})
    recovered = build_experiment_context(
        metrics_config=metrics,
        roles=("metrics",),
        exp_dir=str(exp_dir),
        log=_LOG,
    )

    assert recovered.exp_tag == "train_asr_transformer"
    # The `--exp_dir` CLI value takes priority over the saved context's
    # (relative) exp_dir, since it names the actual directory read from.
    assert recovered.exp_dir == str(exp_dir)
    assert recovered.stats_dir == "./exp/stats"


def test_experiment_context_is_frozen():
    ctx = ExperimentContext(recipe_dir=".", exp_tag="t", exp_dir="./exp/t")
    with pytest.raises(Exception):
        ctx.exp_tag = "other"


def test_compute_fingerprint_is_stable_and_order_independent():
    fp1 = compute_fingerprint({"a": 1, "b": 2})
    fp2 = compute_fingerprint({"b": 2, "a": 1})
    fp3 = compute_fingerprint({"a": 1, "b": 3})

    assert fp1 == fp2
    assert fp1 != fp3


def test_save_and_load_experiment_context_round_trip(tmp_path):
    training = _training_config(tmp_path)
    exp_dir = tmp_path / "exp" / "train_asr_transformer"
    ctx = build_experiment_context(
        training_config=training, roles=("training",), log=_LOG
    )

    context_path = save_experiment_context(str(exp_dir), ctx)
    assert context_path.exists()

    loaded = load_experiment_context(str(exp_dir))
    assert loaded.recipe_dir == ctx.recipe_dir
    assert loaded.exp_tag == ctx.exp_tag
    assert loaded.exp_dir == ctx.exp_dir
    assert loaded.stats_dir == ctx.stats_dir
    assert loaded.data_dir == ctx.data_dir
    assert loaded.tokenizer_dir == ctx.tokenizer_dir


def test_save_experiment_context_keeps_relative_paths(tmp_path):
    training = _training_config(tmp_path)
    exp_dir = tmp_path / "exp" / "train_asr_transformer"
    ctx = build_experiment_context(
        training_config=training, roles=("training",), log=_LOG
    )

    context_path = save_experiment_context(str(exp_dir), ctx)
    payload = OmegaConf.to_container(OmegaConf.load(context_path))

    assert payload["recipe_dir"] == "."
    assert payload["exp_dir"] == "./exp/train_asr_transformer"
    assert payload["stats_dir"] == "./exp/stats"
    # recipe_dir_abs is recorded for diagnostics only, not used for identity.
    assert payload["recipe_dir_abs"]


def test_save_experiment_context_records_stages_and_argv(tmp_path):
    training = _training_config(tmp_path)
    exp_dir = tmp_path / "exp" / "train_asr_transformer"
    ctx = build_experiment_context(
        training_config=training, roles=("training",), log=_LOG
    )

    context_path = save_experiment_context(
        str(exp_dir),
        ctx,
        stages=("train", "infer"),
        argv=("run.py", "--stages", "train", "infer"),
    )
    payload = OmegaConf.to_container(OmegaConf.load(context_path))

    assert payload["last_run"]["stages"] == ["train", "infer"]
    assert payload["last_run"]["argv"] == ["run.py", "--stages", "train", "infer"]


def test_load_experiment_context_missing_raises():
    with pytest.raises(ExperimentContextError, match="context.yaml"):
        load_experiment_context("/nonexistent/exp_dir")


def test_save_experiment_context_rejects_identity_mismatch_without_overwrite(tmp_path):
    exp_dir = tmp_path / "exp" / "shared_dir"

    first = build_experiment_context(
        training_config=_training_config(tmp_path), roles=("training",), log=_LOG
    )
    save_experiment_context(str(exp_dir), first)

    other_training = _training_config(tmp_path, exp_tag="different_tag")
    second = build_experiment_context(
        training_config=other_training, roles=("training",), log=_LOG
    )

    with pytest.raises(ExperimentContextError, match="overwrite_context"):
        save_experiment_context(str(exp_dir), second)


def test_save_experiment_context_overwrite_context_replaces_and_archives(tmp_path):
    exp_dir = tmp_path / "exp" / "shared_dir"

    first = build_experiment_context(
        training_config=_training_config(tmp_path), roles=("training",), log=_LOG
    )
    save_experiment_context(str(exp_dir), first)

    other_training = _training_config(tmp_path, exp_tag="different_tag")
    second = build_experiment_context(
        training_config=other_training, roles=("training",), log=_LOG
    )
    save_experiment_context(str(exp_dir), second, overwrite_context=True)

    loaded = load_experiment_context(str(exp_dir))
    assert loaded.exp_tag == "different_tag"

    history_dir = exp_dir / "config" / "history"
    archived = list(history_dir.glob("context.*.yaml"))
    assert len(archived) == 1


def test_save_stage_config_writes_file_named_after_the_stage(tmp_path):
    exp_dir = tmp_path / "exp" / "train_asr_transformer"
    training = _training_config(tmp_path)
    system = _FakeSystem(exp_dir=str(exp_dir), training=training)

    written = save_stage_config(system, "train")

    assert written == exp_dir / "config" / "train.yaml"
    assert written.exists()


def test_save_stage_config_same_role_different_stages_are_identical(tmp_path):
    exp_dir = tmp_path / "exp" / "train_asr_transformer"
    training = _training_config(tmp_path)
    system = _FakeSystem(exp_dir=str(exp_dir), training=training)

    save_stage_config(system, "collect_stats")
    save_stage_config(system, "train")

    config_dir = exp_dir / "config"
    assert (config_dir / "collect_stats.yaml").read_text() == (
        config_dir / "train.yaml"
    ).read_text()


def test_save_stage_config_content_change_warns_and_archives(tmp_path, caplog):
    exp_dir = tmp_path / "exp" / "train_asr_transformer"
    training = _training_config(tmp_path)
    system = _FakeSystem(exp_dir=str(exp_dir), training=training)
    save_stage_config(system, "train")

    changed_training = _training_config(tmp_path)
    changed_training["stats_dir"] = "./exp/stats_v2"
    system.training_config = changed_training
    with caplog.at_level(logging.WARNING):
        save_stage_config(system, "train")

    assert any("train" in record.message for record in caplog.records)
    history_dir = exp_dir / "config" / "history"
    assert len(list(history_dir.glob("train.*.yaml"))) == 1
    assert "stats_v2" in (exp_dir / "config" / "train.yaml").read_text()


def test_save_stage_config_unchanged_is_not_rewritten(tmp_path):
    exp_dir = tmp_path / "exp" / "train_asr_transformer"
    training = _training_config(tmp_path)
    system = _FakeSystem(exp_dir=str(exp_dir), training=training)
    save_stage_config(system, "train")
    stage_path = exp_dir / "config" / "train.yaml"
    before = stage_path.stat().st_mtime_ns

    save_stage_config(system, "train")

    assert stage_path.stat().st_mtime_ns == before
    history_dir = exp_dir / "config" / "history"
    assert not history_dir.exists() or not list(history_dir.glob("train.*.yaml"))


def test_save_stage_config_noop_without_exp_dir():
    system = _FakeSystem(exp_dir=None)
    assert save_stage_config(system, "train") is None


def test_save_stage_config_noop_when_role_config_is_none(tmp_path):
    exp_dir = tmp_path / "exp" / "train_asr_transformer"
    system = _FakeSystem(exp_dir=str(exp_dir))
    assert save_stage_config(system, "train") is None
    assert not (exp_dir / "config").exists()


def test_save_stage_config_noop_for_dry_run(tmp_path):
    exp_dir = tmp_path / "exp" / "train_asr_transformer"
    training = _training_config(tmp_path)
    system = _FakeSystem(exp_dir=str(exp_dir), training=training)

    assert save_stage_config(system, "train", dry_run=True) is None
    assert not (exp_dir / "config").exists()


def test_save_stage_config_noop_for_non_rank_zero(tmp_path):
    exp_dir = tmp_path / "exp" / "train_asr_transformer"
    training = _training_config(tmp_path)
    system = _FakeSystem(exp_dir=str(exp_dir), training=training)

    assert save_stage_config(system, "train", rank=1) is None
    assert not (exp_dir / "config").exists()


# Identity reconciliation tests for build_experiment_context


def _saved_training_context(tmp_path, inference_dir=None):
    """Save a training-derived context at exp/train_asr_transformer, return exp_dir."""
    training = _training_config(tmp_path)
    inference = (
        OmegaConf.create({"inference_dir": inference_dir}) if inference_dir else None
    )
    exp_dir = tmp_path / "exp" / "train_asr_transformer"
    ctx = build_experiment_context(
        training_config=training,
        inference_config=inference,
        roles=("training", "inference") if inference else ("training",),
        log=_LOG,
    )
    save_experiment_context(str(exp_dir), ctx)
    return training, str(exp_dir)


def test_build_experiment_context_training_and_exp_dir_matching_identity(tmp_path):
    training, exp_dir = _saved_training_context(tmp_path)

    # Same training_config, same exp_dir: identity matches, no error.
    reconciled = build_experiment_context(
        training_config=training, roles=("training",), exp_dir=exp_dir, log=_LOG
    )
    assert reconciled.exp_tag == "train_asr_transformer"


def test_build_experiment_context_training_and_exp_dir_mismatched_identity(tmp_path):
    _, exp_dir = _saved_training_context(tmp_path)

    other_training = _training_config(tmp_path, exp_tag="train_asr_other")
    with pytest.raises(ExperimentContextError, match="does not match"):
        build_experiment_context(
            training_config=other_training,
            roles=("training",),
            exp_dir=exp_dir,
            log=_LOG,
        )


def test_build_experiment_context_standalone_explicit_exp_dir_mismatch(tmp_path):
    _, exp_dir = _saved_training_context(tmp_path)

    metrics = OmegaConf.create({"exp_dir": "./exp/some_other_run"})
    with pytest.raises(ExperimentContextError, match="does not match"):
        build_experiment_context(
            metrics_config=metrics, roles=("metrics",), exp_dir=exp_dir, log=_LOG
        )


def test_build_experiment_context_standalone_inference_dir_mismatch_warns(
    tmp_path, caplog
):
    _, exp_dir = _saved_training_context(
        tmp_path, inference_dir="./exp/train_asr_transformer/inference"
    )

    metrics = OmegaConf.create({"inference_dir": "./exp/train_asr_transformer/alt"})
    with caplog.at_level(logging.WARNING):
        recovered = build_experiment_context(
            metrics_config=metrics, roles=("metrics",), exp_dir=exp_dir, log=_LOG
        )

    # Explicit config wins; saved value is only used for the warning.
    assert recovered.inference_dir == "./exp/train_asr_transformer/alt"
    assert "inference_dir" in caplog.text


# Regression tests for standalone measure/metrics:
# build_experiment_context must fill empty identity/inference_dir keys on the
# role config it derived them from (a saved context or a sibling config), not
# just on the ExperimentContext it returns - validate_experiment_context and
# measure() both read the role config directly, never the ExperimentContext.


def test_standalone_measure_fills_metrics_config_from_saved_context(tmp_path):
    exp_dir = tmp_path / "exp" / "train_asr_rnn"
    training = _training_config(tmp_path, exp_tag="train_asr_rnn")
    inference = OmegaConf.create({"inference_dir": "./exp/train_asr_rnn/inference"})
    first_run_ctx = build_experiment_context(
        training_config=training,
        inference_config=inference,
        roles=("training", "inference"),
        log=_LOG,
    )
    save_experiment_context(str(exp_dir), first_run_ctx)

    # A later, standalone `measure` run: only --exp_dir and an otherwise
    # empty metrics_config, recovering identity and inference_dir from the
    # context `train`+`infer` saved earlier.
    metrics_config = OmegaConf.create({})
    build_experiment_context(
        metrics_config=metrics_config,
        exp_dir=str(exp_dir),
        roles=("metrics",),
        log=_LOG,
    )

    assert metrics_config.get("exp_tag") == "train_asr_rnn"
    assert metrics_config.get("inference_dir") == "./exp/train_asr_rnn/inference"

    # With the role config filled in, validate_experiment_context (which
    # reads metrics_config directly, not the ExperimentContext) must accept
    # the standalone measure request.
    validate_experiment_context(
        training_config=None,
        inference_config=None,
        metrics_config=metrics_config,
        publication_config=None,
        demo_config=None,
        stages_to_run=["measure"],
    )


def test_standalone_measure_without_a_saved_inference_dir_is_rejected(tmp_path):
    # metrics_config carries its own standalone identity (exp_tag/exp_dir),
    # so the identity check alone would not catch a missing inference_dir -
    # this isolates the "measure always needs inference_dir" rule from the
    # identity rule, which the two checks must enforce independently of
    # training_config.
    exp_dir = tmp_path / "exp" / "standalone_eval"
    metrics_config = OmegaConf.create(
        {
            "exp_tag": "standalone_eval",
            "exp_dir": str(exp_dir),
        }
    )

    # build_experiment_context derives a context from metrics_config's own
    # identity; since nothing anywhere names an inference_dir, the context's
    # inference_dir stays unset too.
    ctx = build_experiment_context(
        metrics_config=metrics_config, roles=("metrics",), log=_LOG
    )
    assert ctx.inference_dir is None

    with pytest.raises(ExperimentContextError, match="measure stage"):
        validate_experiment_context(
            training_config=None,
            inference_config=None,
            metrics_config=metrics_config,
            publication_config=None,
            demo_config=None,
            stages_to_run=["measure"],
        )


def test_standalone_inference_and_metrics_propagates_inference_dir_without_training():
    # No training_config at all: a bare `--inference_config` + `--metrics_config`
    # run must still have metrics inherit inference_dir from inference, the
    # same as the training-backed case does.
    inference = OmegaConf.create(
        {
            "exp_tag": "standalone_eval",
            "exp_dir": "./exp/standalone_eval",
            "inference_dir": "./exp/standalone_eval/inference",
        }
    )
    metrics = OmegaConf.create({})

    build_experiment_context(
        inference_config=inference,
        metrics_config=metrics,
        roles=("inference", "metrics"),
        log=_LOG,
    )

    assert metrics.get("inference_dir") == "./exp/standalone_eval/inference"
