import json
from pathlib import Path

import pytest

from espnet3.components.data import dataset_uid as du
from espnet3.components.data.dataset_uid import (
    UID_TABLE_FILENAME,
    UID_TABLE_FORMAT,
    DatasetUidEntry,
    canonicalize_entry,
    check_entry_hashes,
    compute_entry_hash,
    format_uid,
    load_uid_table,
    parse_uid,
    strip_dir_keys,
    validate_against_uid_table,
    write_uid_table,
)

# ===============================================================
# strip_dir_keys / canonicalize_entry / compute_entry_hash
# ===============================================================


def test_strip_dir_keys_removes_nested_dir_suffixed_keys():
    entry = {
        "data_src_args": {
            "split": "train",
            "data_dir": "/a",
            "cache": {"enabled": True, "cache_dir": "/x"},
        },
        "recipe_dir": "/r",
    }
    assert strip_dir_keys(entry) == {
        "data_src_args": {
            "split": "train",
            "cache": {"enabled": True},
        },
    }


def test_strip_dir_keys_walks_lists():
    entry = {"items": [{"path_dir": "/a", "keep": 1}, {"keep": 2}]}
    assert strip_dir_keys(entry) == {"items": [{"keep": 1}, {"keep": 2}]}


def test_strip_dir_keys_leaves_scalars_and_non_dir_keys_alone():
    assert strip_dir_keys("plain") == "plain"
    assert strip_dir_keys(3) == 3
    assert strip_dir_keys({"data_directory": "/a"}) == {"data_directory": "/a"}


def test_entry_hash_ignores_dir_keys():
    """[b] Entries differing only in *_dir values (including nested) hash the same."""
    entry_a = {
        "name": "train",
        "data_src_args": {
            "split": "train",
            "data_dir": "/a",
            "cache": {"enabled": True, "cache_dir": "/x"},
        },
        "recipe_dir": "/recipe/a",
        "source_dir": "/src/a",
    }
    entry_b = {
        "name": "train",
        "data_src_args": {
            "split": "train",
            "data_dir": "/b",
            "cache": {"enabled": True, "cache_dir": "/y"},
        },
        "recipe_dir": "/recipe/b",
        "source_dir": "/src/b",
    }
    assert compute_entry_hash(entry_a) == compute_entry_hash(entry_b)
    assert canonicalize_entry(entry_a) == canonicalize_entry(entry_b)


def test_entry_hash_changes_with_data_src_args_and_transform():
    """[c] Any non-*_dir difference changes the hash; key order does not."""
    base = {
        "name": "train",
        "data_src": "mini_an4/asr",
        "data_src_args": {"split": "train"},
        "transform": {"_target_": "my.module.Transform"},
    }
    base_hash = compute_entry_hash(base)

    split_diff = {**base, "data_src_args": {"split": "valid"}}
    transform_diff = {
        **base,
        "transform": {"_target_": "my.module.OtherTransform"},
    }
    cache_diff = {
        **base,
        "data_src_args": {"split": "train", "cache": {"enabled": True}},
    }
    name_diff = {**base, "name": "train2"}
    data_src_diff = {**base, "data_src": "mini_an4/asr2"}

    for variant in (split_diff, transform_diff, cache_diff, name_diff, data_src_diff):
        assert compute_entry_hash(variant) != base_hash

    # key order alone must not change the hash
    reordered = {
        "transform": base["transform"],
        "data_src_args": base["data_src_args"],
        "data_src": base["data_src"],
        "name": base["name"],
    }
    assert compute_entry_hash(reordered) == base_hash


def test_compute_entry_hash_is_eight_lowercase_hex_chars():
    h = compute_entry_hash({"name": "x"})
    assert len(h) == du.UID_HASH_LENGTH == 8
    assert h == h.lower()
    int(h, 16)  # does not raise


def test_json_default_stringifies_path_and_rejects_arbitrary_objects():
    entry = {"data_src_args": {"template": Path("/a/b.yaml")}}
    text = canonicalize_entry(entry)
    assert json.loads(text)["data_src_args"]["template"] == "/a/b.yaml"

    class Unhashable:
        pass

    with pytest.raises(TypeError, match="cannot be hashed"):
        canonicalize_entry({"obj": Unhashable()})


# ===============================================================
# format_uid / parse_uid
# ===============================================================


def test_format_and_parse_uid_round_trip():
    uid = format_uid("a1b2c3d4", 123)
    assert uid == "a1b2c3d4:123"
    assert parse_uid(uid) == ("a1b2c3d4", 123)


@pytest.mark.parametrize(
    "uid",
    [
        "abc:1",  # prefix too short
        "a1b2c3d4:-1",  # negative position
        "a1b2c3d4:1x",  # non-numeric position
        "A1B2C3D4:1",  # uppercase hex not accepted
        "a1b2c3d4",  # missing position
        "a1b2c3d4@s1:5",  # shard-local label, not a resolvable UID
        "",
    ],
)
def test_parse_uid_rejects_malformed_strings(uid):
    assert parse_uid(uid) is None


# ===============================================================
# write_uid_table / load_uid_table
# ===============================================================


def _entry(prefix="a1b2c3d4", label="train_dummy", num_items=3, config=None):
    return DatasetUidEntry(
        uid_prefix=prefix,
        label=label,
        num_items=num_items,
        config=config if config is not None else {"data_src": "dummy"},
    )


def test_write_and_load_uid_table_round_trip(tmp_path):
    entries = [_entry(), _entry(prefix="deadbeef", label="valid_dummy", num_items=5)]
    target = write_uid_table(tmp_path, entries)

    assert target == tmp_path / UID_TABLE_FILENAME
    assert target.is_file()

    loaded = load_uid_table(tmp_path)
    assert loaded == entries


def test_write_uid_table_is_atomic_no_leftover_tmp_file(tmp_path):
    write_uid_table(tmp_path, [_entry()])
    remaining = list(tmp_path.iterdir())
    assert remaining == [tmp_path / UID_TABLE_FILENAME]


def test_write_uid_table_content_matches_format(tmp_path):
    write_uid_table(tmp_path, [_entry()])
    payload = json.loads((tmp_path / UID_TABLE_FILENAME).read_text(encoding="utf-8"))
    assert payload["format"] == UID_TABLE_FORMAT
    assert payload["datasets"] == [
        {
            "uid_prefix": "a1b2c3d4",
            "label": "train_dummy",
            "num_items": 3,
            "config": {"data_src": "dummy"},
        }
    ]


def test_load_uid_table_returns_none_when_missing(tmp_path):
    assert load_uid_table(tmp_path) is None


def test_load_uid_table_raises_on_corrupt_json(tmp_path):
    (tmp_path / UID_TABLE_FILENAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Failed to read"):
        load_uid_table(tmp_path)


def test_load_uid_table_raises_on_unsupported_format(tmp_path):
    (tmp_path / UID_TABLE_FILENAME).write_text(
        json.dumps({"format": 999, "datasets": []}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="unsupported format"):
        load_uid_table(tmp_path)


def test_load_uid_table_raises_on_missing_fields(tmp_path):
    (tmp_path / UID_TABLE_FILENAME).write_text(
        json.dumps({"format": UID_TABLE_FORMAT, "datasets": [{"uid_prefix": "x"}]}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="malformed"):
        load_uid_table(tmp_path)


# ===============================================================
# check_entry_hashes
# ===============================================================


def test_duplicate_entries_raise():
    """[e] Same *_dir-stripped config listed twice raises ValueError."""
    canon = canonicalize_entry({"data_src": "dummy"})
    entries = [("aaaa1111", canon, "train_a"), ("aaaa1111", canon, "train_b")]
    with pytest.raises(ValueError, match="identical configuration"):
        check_entry_hashes(entries)


def test_hash_collision_raises():
    """[e] Same prefix, different canonical config -> collision error."""
    entries = [
        ("aaaa1111", canonicalize_entry({"data_src": "a"}), "train_a"),
        ("aaaa1111", canonicalize_entry({"data_src": "b"}), "train_b"),
    ]
    with pytest.raises(ValueError, match="32-bit collision"):
        check_entry_hashes(entries)


def test_check_entry_hashes_allows_distinct_entries():
    entries = [
        ("aaaa1111", canonicalize_entry({"data_src": "a"}), "train_a"),
        ("bbbb2222", canonicalize_entry({"data_src": "b"}), "train_b"),
    ]
    check_entry_hashes(entries)  # does not raise


# ===============================================================
# validate_against_uid_table
# ===============================================================


def test_validate_against_uid_table_skips_when_current_is_none(tmp_path):
    # No table on disk at all; current=None must still be a no-op.
    validate_against_uid_table(tmp_path, None)


def test_validate_against_uid_table_passes_for_matching_config(tmp_path):
    entries = [_entry()]
    write_uid_table(tmp_path, entries)
    validate_against_uid_table(tmp_path, entries)  # does not raise


def test_validate_against_uid_table_raises_when_table_missing(tmp_path):
    with pytest.raises(RuntimeError, match="not found"):
        validate_against_uid_table(tmp_path, [_entry()])


def test_validate_against_uid_table_raises_on_changed_dataset(tmp_path):
    old = _entry(config={"data_src_args": {"split": "train"}})
    write_uid_table(tmp_path, [old])
    new = _entry(
        prefix="cafebabe",
        config={"data_src_args": {"split": "train_960"}},
    )
    with pytest.raises(RuntimeError) as exc_info:
        validate_against_uid_table(tmp_path, [new])
    message = str(exc_info.value)
    assert "train_dummy" in message
    assert "cafebabe" in message
    assert "collect_stats" in message


def test_validate_against_uid_table_raises_on_stale_dataset(tmp_path):
    kept = _entry()
    removed = _entry(prefix="cafebabe", label="removed_split")
    write_uid_table(tmp_path, [kept, removed])
    with pytest.raises(RuntimeError, match="not in the current config"):
        validate_against_uid_table(tmp_path, [kept])


def test_validate_against_uid_table_raises_on_changed_num_items(tmp_path):
    old = _entry(num_items=3)
    write_uid_table(tmp_path, [old])
    new = _entry(num_items=5)
    with pytest.raises(RuntimeError, match="has 5 items now but 3"):
        validate_against_uid_table(tmp_path, [new])
