"""Tests for the ``number`` and ``path`` kinds, and ``Kind.accepts``."""

import numpy as np
import pytest

from espnet3.components.contract import KINDS, AudioKind, Field, NumberKind, PathKind


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
    assert isinstance(KINDS["path"], PathKind)


def test_path_kind_accepts_str_and_path_like():
    import os

    field = Field("wav", "path")
    assert PathKind().check("a.wav", field, model=None, output=False) == "a.wav"
    assert PathKind().check(
        os.fspath("a.wav"), field, model=None, output=False
    ) == "a.wav"


def test_path_kind_rejects_non_path():
    field = Field("wav", "path")
    with pytest.raises(TypeError, match="must be a str or os.PathLike"):
        PathKind().check(7, field, model=None, output=False)


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
