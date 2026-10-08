"""Generic entry point: parse CLI args and run a system's declared stages."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from espnet3.components.contract.stages import (
    check_requested_stages,
    roles,
    stage_names,
    stage_spec,
)
from espnet3.utils.config_utils import load_and_merge_config
from espnet3.utils.experiment_context import (
    build_experiment_context,
    save_experiment_context,
)
from espnet3.utils.logging_utils import configure_logging
from espnet3.utils.run_utils import resolve_loaded_configs, validate_experiment_context
from espnet3.utils.stages_utils import _get_process_rank, resolve_stages, run_stages


def build_parser(system_cls: type) -> argparse.ArgumentParser:
    """Build the CLI parser for ``system_cls``: stages and per-role configs.

    Args:
        system_cls: The system class whose declared stages set the
            ``--stages`` choices.

    Returns:
        A parser with ``--stages``, one ``--<role>_config`` per
        :func:`~espnet3.components.contract.stages.roles` of
        ``system_cls``, ``--exp_dir``, ``--overwrite_context``,
        ``--dry_run``, and ``--write_requirements``.

    Examples:
        >>> from espnet3.systems.esp2_asr.system import ASRSystem
        >>> parser = build_parser(ASRSystem)
        >>> args = parser.parse_args(["--stages", "train"])
        >>> args.stages
        ['train']
    """
    names = stage_names(system_cls)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stages",
        choices=names + ["all"],
        nargs="+",
        default=names,
        help="Which stages to run. Multiple values allowed.",
    )
    for role in roles(system_cls):
        parser.add_argument(
            f"--{role}_config",
            default=None,
            type=Path,
            help=f"Hydra config for the {role} role's stages.",
        )
    parser.add_argument(
        "--exp_dir",
        default=None,
        type=str,
        help="Experiment directory for a standalone run with no training_config.",
    )
    parser.add_argument(
        "--overwrite_context",
        action="store_true",
        help="Allow this run's identity to replace a saved context.yaml.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be executed without actually running stages.",
    )
    parser.add_argument(
        "--write_requirements",
        action="store_true",
        help="Write requirements.txt alongside each stage log.",
    )
    return parser


def default_conf_package(system_cls: type) -> str:
    """Return the ``egs3.TEMPLATE`` package that holds ``system_cls``'s default configs.

    Derived from ``system_cls``'s own module, ``espnet3.systems.<x>.system``,
    by the same ``<x>`` naming :data:`~espnet3.api.inference.loading.SYSTEM_ALIASES`
    uses elsewhere.

    Examples:
        >>> from espnet3.systems.esp2_asr.system import ASRSystem
        >>> default_conf_package(ASRSystem)
        'egs3.TEMPLATE.esp2_asr'
    """
    system_name = system_cls.__module__.split(".")[-2]
    return f"egs3.TEMPLATE.{system_name}"


def launch(
    system_cls: type,
    *,
    conf_package: str | None = None,
    argv: Sequence[str] | None = None,
    parser: argparse.ArgumentParser | None = None,
) -> None:
    """Parse CLI args, build ``system_cls``, and run its requested stages.

    Loads each config role, builds and validates the experiment context
    (identity such as ``exp_tag``/``exp_dir`` propagated across roles, or
    recovered from a saved ``context.yaml`` for a standalone run), resolves
    interpolations, builds ``system_cls``, saves the run's context once (rank
    0, unless ``--dry_run``), and runs the requested stages.

    Args:
        system_cls: The system class to instantiate and run.
        conf_package: Package holding the default configs; defaults to
            :func:`default_conf_package`.
        argv: Arguments to parse instead of ``sys.argv``.
        parser: A parser to use instead of ``build_parser(system_cls)``,
            for a recipe that needs its own extra arguments.

    Raises:
        espnet3.components.contract.stages.StageContractError: A requested
            stage is not declared, or its config role was not given.
        espnet3.utils.run_utils.ExperimentContextError: The configs lack
            enough experiment identity for the requested stages.

    Examples:
        ```python
        # egs3/mini_an4/esp2_asr/run.py
        from espnet3.systems.esp2_asr.system import ASRSystem
        from espnet3.systems.launch import launch

        launch(ASRSystem)
        ```
    """
    parser = parser or build_parser(system_cls)
    args = parser.parse_args(argv)
    names = stage_names(system_cls)
    stages_to_run = resolve_stages(args.stages, names)
    system_roles = roles(system_cls)
    check_requested_stages(
        system_cls,
        stages_to_run,
        provided={role: getattr(args, f"{role}_config") for role in system_roles},
    )

    package = conf_package or default_conf_package(system_cls)
    configs = {
        role: load_and_merge_config(
            getattr(args, f"{role}_config"),
            config_name=f"{role}.yaml",
            default_package=package,
            resolve=False,
        )
        for role in system_roles
    }

    logger = configure_logging()
    requested_roles = tuple(
        dict.fromkeys(stage_spec(system_cls, s).config for s in stages_to_run)
    )
    context = build_experiment_context(
        training_config=configs.get("training"),
        inference_config=configs.get("inference"),
        metrics_config=configs.get("metrics"),
        publication_config=configs.get("publication"),
        demo_config=configs.get("demo"),
        exp_dir=args.exp_dir,
        roles=requested_roles,
        log=logger,
    )
    validate_experiment_context(
        training_config=configs.get("training"),
        inference_config=configs.get("inference"),
        metrics_config=configs.get("metrics"),
        publication_config=configs.get("publication"),
        demo_config=configs.get("demo"),
        stages_to_run=stages_to_run,
    )
    resolve_loaded_configs(
        training=configs.get("training"),
        inference=configs.get("inference"),
        metrics=configs.get("metrics"),
        publication=configs.get("publication"),
        demo=configs.get("demo"),
    )

    system = system_cls(
        training_config=configs.get("training"),
        inference_config=configs.get("inference"),
        metrics_config=configs.get("metrics"),
        publication_config=configs.get("publication"),
        demo_config=configs.get("demo"),
    )

    logger.info("System: %s", system_cls.__name__)
    logger.info("Requested stages: %s", args.stages)
    logger.info("Resolved stages: %s", stages_to_run)

    if not args.dry_run and _get_process_rank() == 0:
        save_experiment_context(
            context.exp_dir,
            context,
            overwrite_context=args.overwrite_context,
            stages=stages_to_run,
            argv=list(argv) if argv is not None else sys.argv,
            log=logger,
        )

    run_stages(system=system, stages_to_run=stages_to_run, args=args, log=logger)
