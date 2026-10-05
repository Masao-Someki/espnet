"""Tests for espnet3.autoresearch.patch."""

from espnet3.autoresearch.patch import (
    apply_dotted_patch,
    flatten_dict,
    pattern_to_regex,
    sanitize_patch,
)


def test_flatten_dict_nested():
    assert flatten_dict({"a": {"b": 1, "c": {"d": 2}}, "e": 3}) == {
        "a.b": 1,
        "a.c.d": 2,
        "e": 3,
    }


def test_apply_dotted_patch_sets_nested_key():
    base = {"trainer": {"lr": 0.1}, "model": {"specaug_conf": {"freq_mask": 2}}}
    patched = apply_dotted_patch(
        base, {"trainer.lr": 0.2, "model.specaug_conf.freq_mask": 3}
    )
    assert patched["trainer"]["lr"] == 0.2
    assert patched["model"]["specaug_conf"]["freq_mask"] == 3
    # original is untouched (apply_dotted_patch deep-copies via json round-trip)
    assert base["trainer"]["lr"] == 0.1


def test_apply_dotted_patch_creates_missing_parents():
    patched = apply_dotted_patch({}, {"a.b.c": 1})
    assert patched == {"a": {"b": {"c": 1}}}


def test_apply_dotted_patch_indexes_into_lists():
    base = {
        "dataset": {"train": [{"transform": {"transforms": [{"apply_prob": 0.5}]}}]}
    }
    patched = apply_dotted_patch(
        base, {"dataset.train.0.transform.transforms.1.apply_prob": 0.9}
    )
    transforms = patched["dataset"]["train"][0]["transform"]["transforms"]
    assert transforms[0]["apply_prob"] == 0.5  # untouched sibling
    assert transforms[1]["apply_prob"] == 0.9  # extended and set


def test_pattern_to_regex_extracts_value():
    regex = pattern_to_regex("WER: {value}")
    match = regex.search("some prefix\nWER: 12.34\nmore text")
    assert match is not None
    assert float(match.group("value")) == 12.34


def test_sanitize_patch_keeps_allowed_drops_denied_and_unmatched():
    patch = {
        "trainer.lr": 0.1,
        "exp_dir": "/somewhere/else",
        "optimizer.weight_decay": 0.01,
        "unrelated.key": 1,
    }
    sanitized = sanitize_patch(
        patch,
        allowed_keys=["trainer.*", "optimizer.*"],
        denied_keys=["exp_dir", "stats_dir"],
    )
    assert sanitized == {"trainer.lr": 0.1, "optimizer.weight_decay": 0.01}


def test_sanitize_patch_deny_wins_over_allow():
    patch = {"dataset.train": ["x"]}
    sanitized = sanitize_patch(
        patch, allowed_keys=["dataset.*"], denied_keys=["dataset.*"]
    )
    assert sanitized == {}


def test_sanitize_patch_empty_when_nothing_matches():
    assert sanitize_patch({"a.b": 1}, allowed_keys=["c.*"], denied_keys=[]) == {}
