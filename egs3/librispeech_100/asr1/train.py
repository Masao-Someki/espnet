import argparse
import logging
from pathlib import Path

import lightning as L
import torch
from hydra.utils import instantiate
from tqdm import tqdm

from espnet3 import get_espnet_model, save_espnet_config
from espnet3.preprocess import train_sentencepiece
from espnet3.trainer import ESPnetEZLightningTrainer, LitESPnetModel
from espnet3.utils.config import load_config_with_defaults
from espnet3.utils.logging import (
    log_configuration,
    log_dataset_snapshot,
    log_environment_snapshot,
    log_experiment_directories,
    log_model_summary,
    log_trainer_summary,
    setup_logging_from_config,
)


def train_tokenizer(config):
    config.dataset.preprocessor = None
    organizer = instantiate(config.dataset)

    dataset_size = len(organizer.train)
    with open("train_text.txt", "w", encoding="utf-8") as f:
        for idx in tqdm(range(dataset_size)):
            f.write(organizer.train.get_text(idx) + "\n")
            f.flush()

    train_sentencepiece(
        dump_text_path="train_text.txt",
        output_path="sentencepiece_model",
        vocab_size=config.vocab_size,
        character_coverage=1.0,
        model_type="bpe",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="train.yaml")
    parser.add_argument(
        "--train_tokenizer", action="store_true", help="Train tokenizer before training"
    )
    parser.add_argument(
        "--collect_stats", action="store_true", help="Run collect_stats before training"
    )
    args = parser.parse_args()

    # Load config
    config = load_config_with_defaults(args.config)

    _, logging_options = setup_logging_from_config(getattr(config, "logging", None))
    logger = logging.getLogger("espnet3.train")

    if logging_options.get("path"):
        logger.info("Logging to file: %s", logging_options["path"])
    else:
        logger.info("Logging to stdout (no log file path configured)")

    logger.info("Configuration file: %s", Path(args.config).resolve())
    logger.info("Script arguments: %s", vars(args))

    log_environment_snapshot(
        logger,
        env_keys=logging_options.get("env_keys"),
        log_all_env_vars=logging_options.get("log_all_env_vars", False),
    )

    log_experiment_directories(
        logger,
        expdir=getattr(config, "expdir", None),
        statsdir=getattr(config, "statsdir", None),
    )

    log_configuration(logger, config)

    if args.train_tokenizer:
        logger.info("Training tokenizer before starting model training.")
        train_tokenizer(config)
        logger.info("Tokenizer training finished.")

    # Set seed
    if getattr(config, "seed", None) is not None:
        assert isinstance(config.seed, int), "seed should be an integer"
        L.seed_everything(config.seed)

    # Prepare for collect_stats
    normalize = None
    normalize_conf = None
    if args.collect_stats:
        logger.info("Collecting normalization statistics before training.")
        if "normalize" in config.model:
            normalize = config.model.pop("normalize")
            logger.info("Temporarily removed 'normalize' module for stats collection.")
        if "normalize_conf" in config.model:
            normalize_conf = config.model.pop("normalize_conf")
            logger.info(
                "Temporarily removed 'normalize_conf' for stats collection.")

    task = getattr(config, "task", None)
    model = get_espnet_model(task, config.model) if task else instantiate(config.model)
    lit_model = LitESPnetModel(model, config)

    # Float32 precision
    torch.set_float32_matmul_precision("high")

    log_model_summary(logger, lit_model)
    log_dataset_snapshot(
        logger,
        train_dataset=lit_model.train_dataset,
        valid_dataset=lit_model.valid_dataset,
    )

    # Setup trainer
    trainer = ESPnetEZLightningTrainer(
        model=lit_model,
        expdir=config.expdir,
        config=config.trainer,
        best_model_criterion=config.best_model_criterion,
    )

    log_trainer_summary(logger, trainer)

    if args.collect_stats:
        logger.info("Running collect_stats using current trainer configuration.")
        trainer.collect_stats()
        logger.info("collect_stats finished.")

        if normalize is not None:
            config.model["normalize"] = normalize
            logger.info("Restored 'normalize' module after stats collection.")
        if normalize_conf is not None:
            config.model["normalize_conf"] = normalize_conf
            logger.info("Restored 'normalize_conf' after stats collection.")

        model = (
            get_espnet_model(task, config.model) if task else instantiate(config.model)
        )
        lit_model = LitESPnetModel(model, config)

        log_model_summary(logger, lit_model)
        log_dataset_snapshot(
            logger,
            train_dataset=lit_model.train_dataset,
            valid_dataset=lit_model.valid_dataset,
        )

        trainer = ESPnetEZLightningTrainer(
            model=lit_model,
            expdir=config.expdir,
            config=config.trainer,
            best_model_criterion=config.best_model_criterion,
        )

        log_trainer_summary(logger, trainer)

    # save espnet-like config for inference
    if task:
        save_espnet_config(task, config, config.expdir)
        logger.info("Saved ESPnet-compatible config to %s", Path(config.expdir) / "config.yaml")

    fit_params = {} if not hasattr(config, "fit") else config.fit
    if fit_params:
        logger.info("Starting training with fit parameters: %s", dict(fit_params))
    else:
        logger.info("Starting training with default fit parameters.")
    trainer.fit(**fit_params)


if __name__ == "__main__":
    main()
