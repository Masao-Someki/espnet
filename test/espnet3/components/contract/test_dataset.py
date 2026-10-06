"""Tests for espnet3.components.contract.dataset."""

from pathlib import Path

import numpy as np
import pytest

from espnet3.api.inference import Field
from espnet3.components.contract.dataset import (
    DatasetContractError,
    check_fields,
    check_item,
    check_manifests,
    fields_from_config,
    reconcile_fields,
    require_declared,
)

# ---------------------------------------------------------------------------
# check_fields
# ---------------------------------------------------------------------------


def test_check_fields_accepts_well_formed_tuple():
    class Good:
        fields = (Field("speech", "audio"), Field("text", "text"))

    check_fields(Good, "fields")


def test_check_fields_rejects_non_tuple():
    class Bad:
        fields = [Field("speech", "audio")]

    with pytest.raises(TypeError, match="must be a tuple of Field"):
        check_fields(Bad, "fields")


def test_check_fields_rejects_empty_tuple():
    class Bad:
        fields = ()

    with pytest.raises(TypeError, match="must name at least one field"):
        check_fields(Bad, "fields")


def test_check_fields_rejects_duplicate_names():
    class Bad:
        fields = (Field("speech", "audio"), Field("speech", "text"))

    with pytest.raises(TypeError, match="repeats a name"):
        check_fields(Bad, "fields")


# ---------------------------------------------------------------------------
# fields_from_config / reconcile_fields
# ---------------------------------------------------------------------------


def test_fields_from_config_builds_field_tuple():
    fields = fields_from_config({"speech": "audio", "text": "text"})
    assert fields == (Field("speech", "audio"), Field("text", "text"))


def test_reconcile_fields_prefers_class_when_config_absent():
    class_fields = (Field("speech", "audio"),)
    assert reconcile_fields(class_fields, None, class_name="X") is class_fields


def test_reconcile_fields_falls_back_to_config_when_class_absent():
    config_fields = (Field("speech", "audio"),)
    assert reconcile_fields(None, config_fields, class_name="X") is config_fields


def test_reconcile_fields_returns_none_when_neither_given():
    assert reconcile_fields(None, None, class_name="X") is None


def test_reconcile_fields_accepts_agreeing_declarations():
    class_fields = (Field("speech", "audio"), Field("text", "text"))
    config_fields = (Field("text", "text"), Field("speech", "audio"))
    assert reconcile_fields(class_fields, config_fields, class_name="X") == class_fields


def test_reconcile_fields_rejects_disagreement():
    class_fields = (
        Field("speech", "audio"),
        Field("text", "text"),
        Field("speaker", "text"),
    )
    config_fields = (Field("speech", "audio"), Field("text", "text"))
    with pytest.raises(DatasetContractError, match="make them agree or drop one"):
        reconcile_fields(class_fields, config_fields, class_name="MiniAn4Dataset")


# ---------------------------------------------------------------------------
# check_item
# ---------------------------------------------------------------------------

_FIELDS = (Field("speech", "audio"), Field("text", "text"))


def test_check_item_accepts_matching_item():
    check_item(
        _FIELDS, {"speech": np.zeros(16000, dtype=np.float32), "text": "hi"}, "x"
    )


def test_check_item_allows_undeclared_extra_keys():
    check_item(
        _FIELDS,
        {"speech": np.zeros(16000, dtype=np.float32), "text": "hi", "utt_id": "u1"},
        "x",
    )


def test_check_item_rejects_non_mapping():
    with pytest.raises(DatasetContractError, match="item must be a dict"):
        check_item(_FIELDS, ["not", "a", "dict"], "x")


def test_check_item_rejects_missing_field():
    with pytest.raises(DatasetContractError, match="lacks declared field 'text'"):
        check_item(_FIELDS, {"speech": np.zeros(16000, dtype=np.float32)}, "x")


def test_check_item_rejects_wrong_kind():
    with pytest.raises(DatasetContractError, match="declared audio but the item holds"):
        check_item(_FIELDS, {"speech": 42, "text": "hi"}, "x")


def test_check_item_allows_missing_optional_field():
    fields = (Field("speech", "audio"), Field("prompt", "text", optional=True))
    check_item(fields, {"speech": np.zeros(16000, dtype=np.float32)}, "x")


# ---------------------------------------------------------------------------
# require_declared
# ---------------------------------------------------------------------------


def test_require_declared_always_raises():
    class Undeclared:
        pass

    with pytest.raises(DatasetContractError, match="does not declare fields"):
        require_declared(Undeclared, "fields")


# ---------------------------------------------------------------------------
# check_manifests
# ---------------------------------------------------------------------------


class _FakeBuilder:
    manifest_columns = (
        Field("utt_id", "text"),
        Field("wav", "path"),
        Field("text", "text"),
    )
    manifest_header = False

    def __init__(self, manifests):
        self._manifests = manifests

    def built_manifests(self, **kwargs):
        return self._manifests


def test_check_manifests_accepts_matching_row(tmp_path: Path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    manifest = tmp_path / "train.tsv"
    manifest.write_text(f"utt1\t{wav}\thello\n", encoding="utf-8")

    check_manifests(_FakeBuilder({"train": manifest}))


def test_check_manifests_rejects_wrong_column_count(tmp_path: Path):
    manifest = tmp_path / "train.tsv"
    manifest.write_text("utt1\thello\n", encoding="utf-8")

    with pytest.raises(DatasetContractError, match="row 1 has 2 columns"):
        check_manifests(_FakeBuilder({"train": manifest}))


def test_check_manifests_rejects_missing_path_file(tmp_path: Path):
    manifest = tmp_path / "train.tsv"
    manifest.write_text("utt1\t/no/such/file.wav\thello\n", encoding="utf-8")

    with pytest.raises(DatasetContractError, match="points to a missing file"):
        check_manifests(_FakeBuilder({"train": manifest}))


def test_check_manifests_skips_header_row(tmp_path: Path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    manifest = tmp_path / "train.tsv"
    manifest.write_text(f"id\twav\ttext\nutt1\t{wav}\thello\n", encoding="utf-8")

    class HeaderedBuilder(_FakeBuilder):
        manifest_header = True

    check_manifests(HeaderedBuilder({"train": manifest}))


def test_check_manifests_noop_when_builder_writes_no_manifest():
    class NoManifest:
        def built_manifests(self, **kwargs):
            return {}

    check_manifests(NoManifest())


def test_check_manifests_raises_when_manifests_but_undeclared(tmp_path: Path):
    manifest = tmp_path / "train.tsv"
    manifest.write_text("utt1\thello\n", encoding="utf-8")

    class Undeclared:
        def built_manifests(self, **kwargs):
            return {"train": manifest}

    with pytest.raises(DatasetContractError, match="does not declare manifest_columns"):
        check_manifests(Undeclared())
