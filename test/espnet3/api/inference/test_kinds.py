"""Tests for the ``number`` and ``path`` kinds, and ``Kind.accepts``."""

import numpy as np
import pytest

from espnet3.api.inference import KINDS, AudioKind, Field, NumberKind

# The path-kind and AudioKind.accepts tests below use KINDS["path"] and
# Field("wav", "path") rather than importing PathKind directly, so this file
# stays importable when copied alone onto a tree that predates the path kind
# (there, Field("wav", "path") itself raises ValueError: known kinds are ...).


def test_number_kind_is_registered():
    assert KINDS["number"] is not None
    assert isinstance(KINDS["number"], NumberKind)


def test_number_kind_accepts_int_and_float():
    field = Field("WER", "number")
    assert NumberKind().check(4, field, model=None, output=True) == 4
    assert NumberKind().check(4.3, field, model=None, output=True) == 4.3


def test_number_kind_rejects_bool():
    field = Field("WER", "number")
    with pytest.raises(TypeError, match="must be int or float"):
        NumberKind().check(True, field, model=None, output=True)


def test_number_kind_rejects_str():
    field = Field("WER", "number")
    with pytest.raises(TypeError, match="must be int or float"):
        NumberKind().check("4.3", field, model=None, output=False)


def test_path_kind_registered():
    assert "path" in KINDS


def test_path_kind_accepts_str_and_path_like():
    import os

    field = Field("wav", "path")
    assert KINDS["path"].check("a.wav", field, model=None, output=False) == "a.wav"
    assert (
        KINDS["path"].check(os.fspath("a.wav"), field, model=None, output=False)
        == "a.wav"
    )


def test_path_kind_rejects_non_path():
    field = Field("wav", "path")
    with pytest.raises(TypeError, match="must be a str or os.PathLike"):
        KINDS["path"].check(7, field, model=None, output=False)


@pytest.mark.parametrize(
    "value",
    [
        np.zeros(16000, dtype=np.float32),
        "a.wav",
        (16000, np.zeros(16000, dtype=np.float32)),
    ],
)
def test_audio_kind_accepts_rate_less_forms(value):
    field = Field("speech", "audio")
    assert AudioKind().accepts(value, field) is True


def test_audio_kind_accepts_rejects_unrelated_value():
    field = Field("speech", "audio")
    assert AudioKind().accepts(42, field) is False
