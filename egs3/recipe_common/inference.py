"""Inference output formatting for Kaldi data-dir recipes."""


def build_output(data, model_output, idx):
    """Return hypothesis, reference, and utterance ID fields."""
    return {
        "utt_id": data.get("utt_id", str(idx)),
        "hyp": model_output[0][0],
        "ref": data.get("text", ""),
    }
