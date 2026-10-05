"""CLI entry point: `python -m espnet3.autoresearch {init,run,status}`."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from espnet3.autoresearch.agent import CommandAgent, FileAgent
from espnet3.autoresearch.config import AutoResearchConfig, load_config
from espnet3.autoresearch.loop import (
    ensure_objective_file,
    render_config_yaml,
    resolve_study_dir,
    run_study,
)
from espnet3.autoresearch.study import init_study, load_all_trial_records, read_best


def build_agent(config: AutoResearchConfig) -> Any:
    """Construct the agent `config.agent` describes (`CommandAgent` or `FileAgent`)."""
    if config.agent.type == "command":
        return CommandAgent(
            config.agent.command,
            prompt_via=config.agent.prompt_via,
            response_format=config.agent.response_format,
            timeout_sec=config.agent.timeout_sec,
            env=config.agent.env,
        )
    if config.agent.type == "file":
        return FileAgent()
    raise ValueError(f"Unsupported agent.type: {config.agent.type!r}")


def _default_recipe_dir(config_path: Path) -> Path:
    # autoresearch.yaml conventionally lives at <recipe_dir>/conf/autoresearch.yaml.
    return config_path.resolve().parent.parent


def _cmd_init(args: argparse.Namespace, log: logging.Logger) -> None:
    config = load_config(args.config)
    recipe_dir = args.recipe_dir or _default_recipe_dir(args.config)
    study_dir = resolve_study_dir(config, recipe_dir)
    objective_text = ensure_objective_file(recipe_dir, config.objective_file)
    init_study(
        study_dir,
        config_yaml_text=render_config_yaml(config),
        objective_text=objective_text,
    )
    log.info("Initialized study at %s", study_dir)


def _cmd_run(args: argparse.Namespace, log: logging.Logger) -> None:
    config = load_config(args.config)
    recipe_dir = args.recipe_dir or _default_recipe_dir(args.config)
    agent = build_agent(config)
    run_study(config, recipe_dir=recipe_dir, agent=agent, log=log)


def _cmd_status(args: argparse.Namespace, log: logging.Logger) -> None:
    config = load_config(args.config)
    recipe_dir = args.recipe_dir or _default_recipe_dir(args.config)
    study_dir = resolve_study_dir(config, recipe_dir)
    records = load_all_trial_records(study_dir)
    counts: dict = {}
    for record in records:
        counts[record.status] = counts.get(record.status, 0) + 1
    best = read_best(study_dir)
    print(f"study_dir: {study_dir}")
    print(
        f"trials: {len(records)}"
        + (
            f" ({', '.join(f'{k}={v}' for k, v in sorted(counts.items()))})"
            if counts
            else ""
        )
    )
    if best:
        print(f"best: {best['trial_id']} score={best['score']}")
    else:
        print("best: (none yet)")


def build_parser() -> argparse.ArgumentParser:
    """Build the `python -m espnet3.autoresearch` argument parser."""
    parser = argparse.ArgumentParser(prog="python -m espnet3.autoresearch")
    parser.add_argument("command", choices=["init", "run", "status"])
    parser.add_argument(
        "--config", required=True, type=Path, help="Path to autoresearch.yaml"
    )
    parser.add_argument(
        "--recipe_dir",
        type=Path,
        default=None,
        help="Recipe directory (defaults to --config's grandparent directory)",
    )
    return parser


_COMMANDS = {"init": _cmd_init, "run": _cmd_run, "status": _cmd_status}


def main(argv=None) -> None:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    log = logging.getLogger("espnet3.autoresearch")
    _COMMANDS[args.command](args, log)


if __name__ == "__main__":
    main()
