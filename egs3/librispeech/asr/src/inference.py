"""Inference output helpers for the full LibriSpeech ASR recipe."""


def build_output(data, model_output, idx):
    """Convert one Speech2Text result into fields written by the runner."""
    return {
        "utt_id": data.get("utt_id", str(idx)),
        "hyp": model_output[0][0],
        "ref": data.get("text", ""),
    }
