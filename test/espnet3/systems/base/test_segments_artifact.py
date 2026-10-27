"""Contract-checked segments persist as one JSON artifact per utterance."""

import json
from pathlib import Path

import pytest

from espnet3.contract.inference import load_inference_contract
from espnet3.systems.base.inference_runner import InferenceRunner


@pytest.mark.parametrize("segments", [[], [{"text": "hi", "start": 0, "end": 1}]])
def test_segments_forward_and_writer(tmp_path, segments):
    contract = load_inference_contract(
        {
            "inputs": {"text": {"kind": "text"}},
            "outputs": {"segments": {"kind": "segments"}},
        }
    )
    env = {"inference_contract": contract, "input_key": "text"}
    result = InferenceRunner.forward(
        0, dataset=[{"text": "hi"}], model=lambda text: {"segments": segments}, **env
    )
    writers = InferenceRunner.open_writers(tmp_path, **env)
    try:
        InferenceRunner.write_record(
            writers, result, {}, output_keys=["segments"], **env
        )
    finally:
        InferenceRunner.close_writers(writers, {})
    uid, artifact = (tmp_path / "segments.scp").read_text().strip().split(maxsplit=1)
    assert uid == "0"
    assert json.loads(Path(artifact).read_text()) == segments
