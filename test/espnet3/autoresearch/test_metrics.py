"""Tests for espnet3.autoresearch.metrics."""

import json

import pytest

from espnet3.autoresearch.metrics import MetricNotFoundError, read_metric


def test_read_metric_json_key(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(
        json.dumps(
            {"espnet3.systems.esp2_asr.metrics.wer.WER": {"dev-clean": {"WER": 12.3}}}
        )
    )
    score = read_metric(
        [
            {
                "path": str(path),
                "key": "espnet3.systems.esp2_asr.metrics.wer.WER.dev-clean.WER",
            }
        ]
    )
    assert score == 12.3


def test_read_metric_pattern(tmp_path):
    path = tmp_path / "result.txt"
    path.write_text("some header\nWER: 9.87\n")
    score = read_metric([{"path": str(path), "pattern": "WER: {value}"}])
    assert score == 9.87


def test_read_metric_tries_sources_in_order_first_readable_wins(tmp_path):
    missing = tmp_path / "missing.json"
    present = tmp_path / "present.json"
    present.write_text(json.dumps({"score": 1.5}))
    score = read_metric(
        [
            {"path": str(missing), "key": "score"},
            {"path": str(present), "key": "score"},
        ]
    )
    assert score == 1.5


def test_read_metric_raises_when_nothing_found(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(MetricNotFoundError):
        read_metric([{"path": str(missing), "key": "score"}])


def test_read_metric_raises_when_key_missing(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"other": 1}))
    with pytest.raises(MetricNotFoundError):
        read_metric([{"path": str(path), "key": "score"}])
