import logging

import pytest
from omegaconf import OmegaConf

from espnet3.utils.run_utils import (
    ExperimentContextError,
    apply_training_experiment_context,
    resolve_loaded_configs,
    validate_experiment_context,
)


def test_experiment_context_error_is_a_value_error():
    assert issubclass(ExperimentContextError, ValueError)


def test_apply_training_experiment_context_inserts_missing_values(caplog) -> None:
    training = OmegaConf.create(
        {
            "exp_tag": "train_debug",
            "exp_dir": "./exp/train_debug",
        }
    )
    inference = OmegaConf.create({"exp_tag": None})

    with caplog.at_level(logging.INFO):
        apply_training_experiment_context(
            training_config=training,
            inference_config=inference,
            metrics_config=None,
            publication_config=None,
            log=logging.getLogger("test.run_utils"),
        )

    assert inference.exp_tag == "train_debug"
    assert inference.exp_dir == "./exp/train_debug"
    assert "Inserted inference_config.exp_tag from training_config" in caplog.text
    assert "Inserted inference_config.exp_dir from training_config" in caplog.text


def test_apply_training_experiment_context_warns_on_overwrite(caplog) -> None:
    training = OmegaConf.create(
        {
            "exp_tag": "train_debug",
            "exp_dir": "./exp/train_debug",
        }
    )
    inference = OmegaConf.create(
        {
            "exp_tag": "other_tag",
            "exp_dir": "./exp/other_tag",
        }
    )

    with caplog.at_level(logging.WARNING):
        apply_training_experiment_context(
            training_config=training,
            inference_config=inference,
            metrics_config=None,
            publication_config=None,
            log=logging.getLogger("test.run_utils"),
        )

    assert inference.exp_tag == "train_debug"
    assert inference.exp_dir == "./exp/train_debug"
    assert "Overriding inference_config.exp_tag" in caplog.text
    assert "Overriding inference_config.exp_dir" in caplog.text


def test_apply_training_experiment_context_noop_without_training() -> None:
    inference = OmegaConf.create(
        {
            "exp_tag": "standalone_eval",
            "exp_dir": "./exp/standalone_eval",
        }
    )

    apply_training_experiment_context(
        training_config=None,
        inference_config=inference,
        metrics_config=None,
        publication_config=None,
        log=logging.getLogger("test.run_utils"),
    )

    assert inference.exp_tag == "standalone_eval"
    assert inference.exp_dir == "./exp/standalone_eval"


def test_apply_training_experiment_context_syncs_metrics_from_inference(
    caplog,
) -> None:
    inference = OmegaConf.create(
        {
            "exp_tag": "standalone_eval",
            "exp_dir": "./exp/standalone_eval",
            "inference_dir": "./custom/infer",
        }
    )
    metrics = OmegaConf.create({"inference_dir": None})

    with caplog.at_level(logging.INFO):
        apply_training_experiment_context(
            training_config=None,
            inference_config=inference,
            metrics_config=metrics,
            publication_config=None,
            log=logging.getLogger("test.run_utils"),
        )

    assert metrics.exp_tag == "standalone_eval"
    assert metrics.exp_dir == "./exp/standalone_eval"
    assert metrics.inference_dir == "./custom/infer"
    assert "Inserted metrics_config.exp_tag from inference_config" in caplog.text
    assert "Inserted metrics_config.exp_dir from inference_config" in caplog.text
    assert "Inserted metrics_config.inference_dir from inference_config" in caplog.text


def test_apply_training_experiment_context_metrics_inference_dir_follows_inference(
    caplog,
) -> None:
    training = OmegaConf.create(
        {
            "exp_tag": "train_debug",
            "exp_dir": "./exp/train_debug",
        }
    )
    inference = OmegaConf.create(
        {
            "inference_dir": "./exp/train_debug/custom_inference",
        }
    )
    metrics = OmegaConf.create(
        {
            "inference_dir": "./exp/train_debug/old_inference",
        }
    )

    with caplog.at_level(logging.WARNING):
        apply_training_experiment_context(
            training_config=training,
            inference_config=inference,
            metrics_config=metrics,
            publication_config=None,
            log=logging.getLogger("test.run_utils"),
        )

    assert metrics.exp_tag == "train_debug"
    assert metrics.exp_dir == "./exp/train_debug"
    assert metrics.inference_dir == "./exp/train_debug/custom_inference"
    assert "Overriding metrics_config.inference_dir" in caplog.text
    assert "inference_config value" in caplog.text


def test_validate_experiment_context_accepts_standalone_inference() -> None:
    validate_experiment_context(
        training_config=None,
        inference_config=OmegaConf.create(
            {
                "exp_tag": "standalone_eval",
                "exp_dir": "./exp/standalone_eval",
            }
        ),
        metrics_config=None,
        publication_config=None,
        demo_config=None,
        stages_to_run=["infer"],
    )


def test_validate_experiment_context_requires_identity() -> None:
    with pytest.raises(ExperimentContextError, match="infer stage requires"):
        validate_experiment_context(
            training_config=None,
            inference_config=OmegaConf.create({"exp_tag": None}),
            metrics_config=None,
            publication_config=None,
            demo_config=None,
            stages_to_run=["infer"],
        )


def test_validate_experiment_context_accepts_training_backed_inference() -> None:
    validate_experiment_context(
        training_config=OmegaConf.create(
            {
                "exp_tag": "train_asr_rnn",
                "exp_dir": "./exp/train_asr_rnn",
            }
        ),
        inference_config=OmegaConf.create({"inference_dir": "${exp_dir}/inference"}),
        metrics_config=None,
        publication_config=None,
        demo_config=None,
        stages_to_run=["infer"],
    )


def test_validate_experiment_context_accepts_standalone_metrics_with_inference_dir():
    # measure always needs inference_dir (regardless of training_config), so
    # a standalone metrics_config must supply its own to pass - exp_dir alone
    # is not enough.
    validate_experiment_context(
        training_config=None,
        inference_config=None,
        metrics_config=OmegaConf.create(
            {
                "exp_dir": "./exp/standalone_eval",
                "inference_dir": "./exp/standalone_eval/inference",
            }
        ),
        publication_config=None,
        demo_config=None,
        stages_to_run=["measure"],
    )


def test_validate_experiment_context_rejects_non_standalone_metrics() -> None:
    with pytest.raises(ExperimentContextError, match="measure stage requires"):
        validate_experiment_context(
            training_config=None,
            inference_config=None,
            metrics_config=OmegaConf.create({"exp_dir": "./exp/None/metrics"}),
            publication_config=None,
            demo_config=None,
            stages_to_run=["measure"],
        )


def test_validate_experiment_context_requires_inference_dir_for_measure() -> None:
    # measure always needs inference_dir, even when training_config is given:
    # a training-backed run that requests measure without first running infer
    # (or pointing inference_dir at an existing run) has nothing to score.
    with pytest.raises(ExperimentContextError, match="inference_dir"):
        validate_experiment_context(
            training_config=OmegaConf.create(
                {"exp_tag": "train_asr_rnn", "exp_dir": "./exp/train_asr_rnn"}
            ),
            inference_config=None,
            metrics_config=OmegaConf.create({"inference_dir": None}),
            publication_config=None,
            demo_config=None,
            stages_to_run=["measure"],
        )


def test_validate_experiment_context_rejects_unresolved_exp_tag_in_pack_demo() -> None:
    with pytest.raises(ExperimentContextError, match=r"\$\{exp_tag\}"):
        validate_experiment_context(
            training_config=None,
            inference_config=None,
            metrics_config=None,
            publication_config=None,
            demo_config=OmegaConf.create({"pack": {"out_dir": "./demo/${exp_tag}"}}),
            stages_to_run=["pack_demo"],
        )


def test_validate_experiment_context_rejects_task_without_model() -> None:
    with pytest.raises(ExperimentContextError, match="model"):
        validate_experiment_context(
            training_config=OmegaConf.create(
                {
                    "exp_tag": "train_debug",
                    "exp_dir": "./exp/train_debug",
                    "task": "asr",
                }
            ),
            inference_config=None,
            metrics_config=None,
            publication_config=None,
            demo_config=None,
            stages_to_run=["train"],
        )


def test_resolve_loaded_configs_resolves_interpolations() -> None:
    training = OmegaConf.create(
        {
            "exp_tag": "train_debug",
            "exp_dir": "./exp/${exp_tag}",
        }
    )
    inference = OmegaConf.create({"inference_dir": "${exp_dir}/inference"})

    apply_training_experiment_context(
        training_config=training,
        inference_config=inference,
        metrics_config=None,
        publication_config=None,
        log=logging.getLogger("test.run_utils"),
    )
    resolve_loaded_configs(training=training, inference=inference)

    assert training.exp_dir == "./exp/train_debug"
    assert inference.inference_dir == "./exp/train_debug/inference"


def test_validate_experiment_context_accepts_metrics_synced_from_inference() -> None:
    inference = OmegaConf.create(
        {
            "exp_tag": "standalone_eval",
            "exp_dir": "./exp/standalone_eval",
            "inference_dir": "./exp/standalone_eval/inference",
        }
    )
    metrics = OmegaConf.create({"inference_dir": None})

    apply_training_experiment_context(
        training_config=None,
        inference_config=inference,
        metrics_config=metrics,
        publication_config=None,
        log=logging.getLogger("test.run_utils"),
    )

    validate_experiment_context(
        training_config=None,
        inference_config=inference,
        metrics_config=metrics,
        publication_config=None,
        demo_config=None,
        stages_to_run=["infer", "measure"],
    )


def test_resolve_loaded_configs_ignores_none_entries() -> None:
    inference = OmegaConf.create({"inference_dir": "./exp/standalone_eval/inference"})

    resolve_loaded_configs(training=None, inference=inference)

    assert inference.inference_dir == "./exp/standalone_eval/inference"


def test_resolve_loaded_configs_raises_on_missing_interpolation() -> None:
    inference = OmegaConf.create({"inference_dir": "${exp_dir}/inference"})

    with pytest.raises(ExperimentContextError, match="inference"):
        resolve_loaded_configs(inference=inference)


def test_resolve_loaded_configs_wraps_omegaconf_error_with_role_name() -> None:
    metrics = OmegaConf.create({"inference_dir": "${exp_dir}/metrics"})

    try:
        resolve_loaded_configs(metrics=metrics)
    except ExperimentContextError as exc:
        assert "metrics" in str(exc)
    else:
        pytest.fail("expected ExperimentContextError")


def test_resolve_loaded_configs_is_keyword_only() -> None:
    inference = OmegaConf.create({"inference_dir": "./exp/standalone_eval/inference"})

    with pytest.raises(TypeError):
        resolve_loaded_configs(inference)


def test_apply_training_context_syncs_publication_from_training_and_inference(
    caplog,
) -> None:
    training = OmegaConf.create(
        {
            "exp_tag": "train_debug",
            "exp_dir": "./exp/train_debug",
        }
    )
    inference = OmegaConf.create(
        {
            "inference_dir": "./exp/train_debug/inference/custom_eval",
        }
    )
    publication = OmegaConf.create(
        {
            "pack_model": {
                "out_dir": "${exp_dir}/model_pack",
                "inference_dir": "${inference_dir}",
            }
        }
    )

    with caplog.at_level(logging.INFO):
        apply_training_experiment_context(
            training_config=training,
            inference_config=inference,
            metrics_config=None,
            publication_config=publication,
            log=logging.getLogger("test.run_utils"),
        )

    assert publication.exp_tag == "train_debug"
    assert publication.exp_dir == "./exp/train_debug"
    assert publication.inference_dir == "./exp/train_debug/inference/custom_eval"
    assert "Inserted publication_config.exp_tag from training_config" in caplog.text
    assert "Inserted publication_config.exp_dir from training_config" in caplog.text
    assert (
        "Inserted publication_config.inference_dir from inference_config" in caplog.text
    )


def test_apply_training_context_propagates_pack_model_out_dir_to_demo(
    caplog,
) -> None:
    publication = OmegaConf.create({"pack_model": {"out_dir": "./model_pack"}})
    demo = OmegaConf.create({"model": {"dir_or_tag": None}})

    with caplog.at_level(logging.INFO):
        apply_training_experiment_context(
            training_config=None,
            inference_config=None,
            metrics_config=None,
            publication_config=publication,
            demo_config=demo,
            log=logging.getLogger("test.run_utils"),
        )

    assert demo.model.dir_or_tag == "./model_pack"
    assert "Inserted demo_config.model.dir_or_tag" in caplog.text


def test_apply_training_context_does_not_override_existing_demo_dir_or_tag() -> None:
    publication = OmegaConf.create({"pack_model": {"out_dir": "./model_pack"}})
    demo = OmegaConf.create({"model": {"dir_or_tag": "existing/model"}})

    apply_training_experiment_context(
        training_config=None,
        inference_config=None,
        metrics_config=None,
        publication_config=publication,
        demo_config=demo,
        log=logging.getLogger("test.run_utils"),
    )

    assert demo.model.dir_or_tag == "existing/model"


def test_apply_training_context_prefers_upload_model_hf_repo_over_pack_model_out_dir(
    caplog,
) -> None:
    publication = OmegaConf.create(
        {
            "pack_model": {"out_dir": "./model_pack"},
            "upload_model": {"hf_repo": "myorg/my-model"},
        }
    )
    demo = OmegaConf.create({"model": {"dir_or_tag": None}})

    with caplog.at_level(logging.INFO):
        apply_training_experiment_context(
            training_config=None,
            inference_config=None,
            metrics_config=None,
            publication_config=publication,
            demo_config=demo,
            log=logging.getLogger("test.run_utils"),
        )

    assert demo.model.dir_or_tag == "myorg/my-model"
    assert "upload_model.hf_repo" in caplog.text


def test_apply_training_context_falls_back_to_pack_model_out_dir_when_no_hf_repo(
    caplog,
) -> None:
    publication = OmegaConf.create({"pack_model": {"out_dir": "./model_pack"}})
    demo = OmegaConf.create({"model": {"dir_or_tag": None}})

    with caplog.at_level(logging.INFO):
        apply_training_experiment_context(
            training_config=None,
            inference_config=None,
            metrics_config=None,
            publication_config=publication,
            demo_config=demo,
            log=logging.getLogger("test.run_utils"),
        )

    assert demo.model.dir_or_tag == "./model_pack"
    assert "pack_model.out_dir" in caplog.text


def test_resolve_loaded_configs_resolves_publication_interpolations() -> None:
    training = OmegaConf.create(
        {
            "exp_tag": "train_debug",
            "exp_dir": "./exp/${exp_tag}",
        }
    )
    inference = OmegaConf.create({"inference_dir": "${exp_dir}/inference"})
    publication = OmegaConf.create(
        {
            "pack_model": {
                "out_dir": "${exp_dir}/model_pack",
                "inference_dir": "${inference_dir}",
            }
        }
    )

    apply_training_experiment_context(
        training_config=training,
        inference_config=inference,
        metrics_config=None,
        publication_config=publication,
        log=logging.getLogger("test.run_utils"),
    )
    resolve_loaded_configs(
        training=training, inference=inference, publication=publication
    )

    assert publication.pack_model.out_dir == "./exp/train_debug/model_pack"
    assert publication.pack_model.inference_dir == "./exp/train_debug/inference"
