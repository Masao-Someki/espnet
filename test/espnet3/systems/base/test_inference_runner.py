import pytest

from espnet3.systems.base.inference_runner import InferenceRunner, _load_output_fn


def test_write_record_does_not_validate_the_id(tmp_path):
    """write_record no longer validates the id.

    id validation now happens once, inside item_uid; write_record just
    writes whatever id the record already carries.
    """
    writers = InferenceRunner.open_writers(tmp_path)
    InferenceRunner.write_record(writers, {"utt_id": "a/b", "hyp": "h"}, {})
    InferenceRunner.close_writers(writers, {})
    assert (tmp_path / "hyp.scp").read_text() == "a/b h\n"


def test_load_output_fn_rejects_missing_module():
    with pytest.raises(ModuleNotFoundError):
        _load_output_fn("no.such.module.output_fn")


def test_forward_raises_without_input_key_kwarg():
    with pytest.raises(RuntimeError, match="input_key must be provided"):
        InferenceRunner.forward(0, dataset=[], model=lambda: None)


def test_forward_single_raises_key_error_for_missing_dataset_key():
    dataset = [{"speech": 1.0}]
    with pytest.raises(KeyError, match="Input key"):
        InferenceRunner.forward(
            0, dataset=dataset, model=lambda **kw: None, input_key="text"
        )


def test_forward_batched_raises_key_error_for_missing_dataset_key():
    dataset = [{"speech": 1.0}, {"speech": 2.0}]
    with pytest.raises(KeyError, match="Input key"):
        InferenceRunner.forward(
            [0, 1], dataset=dataset, model=lambda **kw: None, input_key="text"
        )


def test_forward_batched_returns_model_output_without_output_fn():
    dataset = [{"speech": 1.0}, {"speech": 2.0}]

    def model(speech):
        return {"result": speech}

    result = InferenceRunner.forward(
        [0, 1], dataset=dataset, model=model, input_key="speech"
    )
    assert result == {"result": [1.0, 2.0]}


def test_forward_single_returns_model_output_without_output_fn():
    dataset = [{"speech": 1.0}]

    def model(speech):
        return {"result": speech}

    result = InferenceRunner.forward(
        0, dataset=dataset, model=model, input_key="speech"
    )
    assert result == {"result": 1.0}


def test_forward_single_passes_model_kwargs_to_model():
    dataset = [{"speech": 1.0}]

    def model(speech, beam_size):
        return {"result": f"{speech}:{beam_size}"}

    result = InferenceRunner.forward(
        0,
        dataset=dataset,
        model=model,
        input_key="speech",
        model_kwargs={"beam_size": 2},
    )
    assert result == {"result": "1.0:2"}


def test_forward_batched_passes_model_kwargs_to_model():
    dataset = [{"speech": 1.0}, {"speech": 2.0}]

    def model(speech, beam_size):
        return {"result": f"{speech}:{beam_size}"}

    result = InferenceRunner.forward(
        [0, 1],
        dataset=dataset,
        model=model,
        input_key="speech",
        model_kwargs={"beam_size": 4},
    )
    assert result == {"result": "[1.0, 2.0]:4"}


def test_forward_batched_wraps_model_exception_in_runtime_error():
    dataset = [{"speech": 1.0}]

    def failing_model(speech):
        raise ValueError("model broken")

    with pytest.raises(RuntimeError, match="one at a time") as info:
        InferenceRunner.forward(
            [0], dataset=dataset, model=failing_model, input_key="speech"
        )
    assert isinstance(info.value.__cause__, ValueError)


def test_forward_batched_reports_out_of_memory_with_lengths_and_batch_size():
    """An OOM is not "your model does not support batches": name the items."""
    import numpy as np
    import torch

    dataset = [{"speech": np.zeros(16000)}, {"speech": np.zeros(8000)}]

    def model(speech):
        raise torch.OutOfMemoryError("CUDA out of memory (simulated)")

    with pytest.raises(RuntimeError, match="ran out of memory") as info:
        InferenceRunner.forward(
            [0, 1], dataset=dataset, model=model, input_key="speech"
        )
    message = str(info.value)
    assert "[16000, 8000]" in message
    assert "batch_size" in message and "2 items" in message
    assert "set batch_size to None" not in message
    assert isinstance(info.value.__cause__, torch.OutOfMemoryError)
