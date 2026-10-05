"""Confirms the TEMPLATE autoresearch.yaml itself stays loadable and valid."""

from pathlib import Path

from espnet3.autoresearch.config import load_config

_TEMPLATE_PATH = (
    Path(__file__).resolve().parents[3]
    / "egs3"
    / "TEMPLATE"
    / "esp2_asr"
    / "conf"
    / "autoresearch.yaml"
)


def test_template_autoresearch_yaml_loads_and_validates():
    config = load_config(_TEMPLATE_PATH)
    assert config.study_dir == "exp/autoresearch/ls100_ebf"
    assert config.metric.mode == "min"
    assert config.agent.type == "command"
    assert config.edit.mode == "direct"


def test_template_autoresearch_yaml_placeholders_are_recognized():
    # Every {name} used in commands/metric.source/agent.command must be one
    # config.render() (and CommandAgent/runner) actually knows how to fill.
    from espnet3.autoresearch.config import KNOWN_PLACEHOLDERS

    config = load_config(_TEMPLATE_PATH)
    names = set()
    import re

    pattern = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
    for cmd in config.trial.commands:
        for arg in cmd:
            names.update(pattern.findall(arg))
    for entry in config.metric.source:
        names.update(pattern.findall(entry.path))
        if entry.pattern:
            names.discard("value")  # {value} belongs to pattern_to_regex, not render()
    for arg in config.agent.command:
        names.update(pattern.findall(arg))
    assert names <= KNOWN_PLACEHOLDERS | {"prompt_file", "response_file"}
