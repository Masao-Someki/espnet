"""Tests for `espnet3.autoresearch.agent`."""

from __future__ import annotations

import sys
import threading
import time

import pytest

from espnet3.autoresearch.agent import (
    AgentRequest,
    AgentResponseError,
    AgentWaitingError,
    CommandAgent,
    FileAgent,
    render_placeholders,
    render_prompt,
)

PY = sys.executable


def _request(**context) -> AgentRequest:
    return AgentRequest(
        task="propose_patch",
        objective="Lower dev_clean WER.",
        context=context,
        allowed_actions=["edit_config"],
        output_schema={"rationale": "str", "config_patch": "dict"},
    )


# ---------------------------------------------------------------------------
# render_placeholders
# ---------------------------------------------------------------------------


def test_render_placeholders_substitutes_known_names():
    out = render_placeholders(
        ["--dir", "{recipe_dir}/conf", "--out={response_file}"],
        {"recipe_dir": "/r", "response_file": "/r/resp.json"},
    )
    assert out == ["--dir", "/r/conf", "--out=/r/resp.json"]


def test_render_placeholders_leaves_non_placeholder_braces_alone():
    out = render_placeholders(['{"a": 1}'], {})
    assert out == ['{"a": 1}']


def test_render_placeholders_raises_on_unknown_name():
    with pytest.raises(ValueError, match="trial_id"):
        render_placeholders(["{trial_id}"], {"recipe_dir": "/r"})


# ---------------------------------------------------------------------------
# render_prompt
# ---------------------------------------------------------------------------


def test_render_prompt_includes_fixed_sections():
    request = _request(
        edit_mode="allowlist",
        edit_allowlist=["conf/tuning/*.yaml"],
        base_config="trainer:\n  max_epoch: 10\n",
        search_space={"allowed_keys": ["trainer.*"]},
        trial_history_csv="id,status\n1,accepted\n",
        best_patch={"trainer.lr": 0.001},
        best_score=5.2,
        recent_failure_log="RuntimeError: CUDA OOM",
    )
    text = render_prompt(request)

    for heading in (
        "## Objective",
        "## Edit Rules",
        "## Base Training Config",
        "## Search Space",
        "## Trial History",
        "## Best Patch & Score",
        "## Recent Failure Log",
        "## Allowed Actions",
        "## Output Schema",
    ):
        assert heading in text
    assert "Lower dev_clean WER." in text
    assert "mode: allowlist" in text
    assert "conf/tuning/*.yaml" in text
    assert "max_epoch: 10" in text
    assert "trainer.lr" in text
    assert "5.2" in text
    assert "CUDA OOM" in text
    assert "edit_config" in text


def test_render_prompt_renders_missing_context_as_none():
    request = _request()
    text = render_prompt(request)
    assert "## Base Training Config\n```yaml\n(none)" in text
    assert "## Best Patch & Score\n(none)" in text
    assert "## Recent Failure Log\n```\n(none)" in text


# ---------------------------------------------------------------------------
# CommandAgent
# ---------------------------------------------------------------------------

_FAKE_AGENT_STDIN = (
    "import json, sys\n"
    "prompt = sys.stdin.read()\n"
    "assert 'Lower dev_clean WER.' in prompt, prompt\n"
    "with open(sys.argv[1], 'w') as f:\n"
    "    json.dump({'rationale': 'try lower lr', "
    "'config_patch': {'trainer.lr': 0.0005}}, f)\n"
)

_FAKE_AGENT_PROMPT_FILE = (
    "import json, sys\n"
    "with open(sys.argv[1]) as f:\n"
    "    prompt = f.read()\n"
    "assert 'Lower dev_clean WER.' in prompt, prompt\n"
    "with open(sys.argv[2], 'w') as f:\n"
    "    json.dump({'rationale': 'ok', 'config_patch': {'a': 1}}, f)\n"
)

_FAKE_AGENT_NO_RESPONSE = "pass\n"

_FAKE_AGENT_BAD_JSON = "open(sys.argv[1], 'w').write('not json')\nimport sys\n"

_FAKE_AGENT_MISSING_PATCH = (
    "import json, sys\n"
    "with open(sys.argv[1], 'w') as f:\n"
    "    json.dump({'rationale': 'no patch here'}, f)\n"
)

_FAKE_AGENT_SLEEPS = "import time\ntime.sleep(5)\n"


def test_command_agent_propose_via_stdin(tmp_path):
    agent = CommandAgent(
        command=[PY, "-c", _FAKE_AGENT_STDIN, "{response_file}"],
        prompt_via="stdin",
        response_format="json",
    )
    response = agent.propose(_request(), artifact_dir=tmp_path)

    assert response.rationale == "try lower lr"
    assert response.config_patch == {"trainer.lr": 0.0005}
    assert (tmp_path / "prompt.md").exists()
    assert (tmp_path / "stdout.log").exists()
    assert (tmp_path / "stderr.log").exists()


def test_command_agent_propose_via_prompt_file(tmp_path):
    agent = CommandAgent(
        command=[PY, "-c", _FAKE_AGENT_PROMPT_FILE, "{prompt_file}", "{response_file}"],
        prompt_via="file",
        response_format="json",
    )
    response = agent.propose(_request(), artifact_dir=tmp_path)

    assert response.config_patch == {"a": 1}


def test_command_agent_substitutes_caller_placeholders(tmp_path):
    script = (
        "import json, sys\n"
        "assert sys.argv[1] == 'r1', sys.argv\n"
        "with open(sys.argv[2], 'w') as f:\n"
        "    json.dump({'rationale': 'ok', 'config_patch': {}}, f)\n"
    )
    agent = CommandAgent(command=[PY, "-c", script, "{trial_id}", "{response_file}"])
    response = agent.propose(
        _request(), artifact_dir=tmp_path, placeholders={"trial_id": "r1"}
    )
    assert response.config_patch == {}


def test_command_agent_unknown_placeholder_raises_before_running(tmp_path):
    agent = CommandAgent(command=[PY, "-c", "raise SystemExit(1)", "{nope}"])
    with pytest.raises(ValueError, match="nope"):
        agent.propose(_request(), artifact_dir=tmp_path)
    # Nothing should have run (and so no stdout.log should exist).
    assert not (tmp_path / "stdout.log").exists()


def test_command_agent_raises_when_no_response_file_written(tmp_path):
    agent = CommandAgent(command=[PY, "-c", _FAKE_AGENT_NO_RESPONSE, "{response_file}"])
    with pytest.raises(AgentResponseError) as excinfo:
        agent.propose(_request(), artifact_dir=tmp_path)
    assert excinfo.value.reason == "agent_response"


def test_command_agent_raises_on_malformed_response(tmp_path):
    script = "import sys\nopen(sys.argv[1], 'w').write('not json')\n"
    agent = CommandAgent(command=[PY, "-c", script, "{response_file}"])
    with pytest.raises(AgentResponseError):
        agent.propose(_request(), artifact_dir=tmp_path)


def test_command_agent_raises_when_response_missing_config_patch(tmp_path):
    agent = CommandAgent(
        command=[PY, "-c", _FAKE_AGENT_MISSING_PATCH, "{response_file}"]
    )
    with pytest.raises(AgentResponseError, match="config_patch"):
        agent.propose(_request(), artifact_dir=tmp_path)


def test_command_agent_timeout_raises_agent_response_error(tmp_path):
    agent = CommandAgent(command=[PY, "-c", _FAKE_AGENT_SLEEPS], timeout_sec=0.2)
    with pytest.raises(AgentResponseError):
        agent.propose(_request(), artifact_dir=tmp_path)


def test_command_agent_reads_yaml_response(tmp_path):
    script = (
        "import sys\n"
        "with open(sys.argv[1], 'w') as f:\n"
        "    f.write('rationale: ok\\nconfig_patch:\\n  trainer.lr: 0.1\\n')\n"
    )
    agent = CommandAgent(
        command=[PY, "-c", script, "{response_file}"], response_format="yaml"
    )
    response = agent.propose(_request(), artifact_dir=tmp_path)
    assert response.config_patch == {"trainer.lr": 0.1}


def test_command_agent_rejects_empty_command():
    with pytest.raises(ValueError):
        CommandAgent(command=[])


def test_command_agent_rejects_bad_prompt_via():
    with pytest.raises(ValueError):
        CommandAgent(command=["true"], prompt_via="carrier-pigeon")


# ---------------------------------------------------------------------------
# FileAgent
# ---------------------------------------------------------------------------


def test_file_agent_writes_request_file(tmp_path):
    agent = FileAgent(wait=False)
    with pytest.raises(AgentWaitingError):
        agent.propose(_request(), artifact_dir=tmp_path)
    text = (tmp_path / "agent_request.md").read_text()
    assert "Lower dev_clean WER." in text


def test_file_agent_returns_existing_response(tmp_path):
    (tmp_path / "agent_response.yaml").write_text(
        "rationale: human says so\nconfig_patch:\n  trainer.lr: 0.2\n"
    )
    agent = FileAgent(wait=False)
    response = agent.propose(_request(), artifact_dir=tmp_path)
    assert response.rationale == "human says so"
    assert response.config_patch == {"trainer.lr": 0.2}


def test_file_agent_waits_for_response(tmp_path):
    def write_soon():
        time.sleep(0.1)
        (tmp_path / "agent_response.yaml").write_text(
            "rationale: later\nconfig_patch: {}\n"
        )

    threading.Thread(target=write_soon).start()
    agent = FileAgent(wait=True, poll_interval_sec=0.05, timeout_sec=5)
    response = agent.propose(_request(), artifact_dir=tmp_path)
    assert response.rationale == "later"


def test_file_agent_wait_times_out(tmp_path):
    agent = FileAgent(wait=True, poll_interval_sec=0.05, timeout_sec=0.2)
    with pytest.raises(AgentResponseError):
        agent.propose(_request(), artifact_dir=tmp_path)


def test_file_agent_raises_on_malformed_response(tmp_path):
    (tmp_path / "agent_response.yaml").write_text("not: [valid: yaml:")
    agent = FileAgent(wait=False)
    with pytest.raises(AgentResponseError):
        agent.propose(_request(), artifact_dir=tmp_path)
