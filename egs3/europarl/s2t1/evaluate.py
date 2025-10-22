import argparse
import logging
import time
from pathlib import Path

import torch.nn as nn
from hydra.utils import instantiate

# from espnet2.bin.asr_inference_ctc import Speech2Text
from espnet3.inference.inference_runner import InferenceRunner
from espnet3.inference.score_runner import ScoreRunner
from espnet3.utils.config import load_config_with_defaults
from espnet3.utils.logging import (
    log_configuration,
    log_environment_snapshot,
    log_experiment_directories,
    setup_logging_from_config,
)


class ASRInferenceRunner(InferenceRunner, nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        InferenceRunner.__init__(self, **kwargs)

    def initialize_model(self, device=None):
        if device is None:
            device = self.device
        return instantiate(self.model_config, device=device)

    def inference_body(self, model, sample: dict) -> dict:
        assert "speech" in sample, "Missing 'speech' key in sample"
        speech = sample["speech"]

        start = time.time()
        results = model(speech)
        end = time.time()

        hyp_text = results[0][0] if results else ""
        duration = len(speech) / 16000
        elapsed = end - start
        rtf = elapsed / duration if duration > 0 else 0.0
        output = {
            "hypothesis": {"type": "text", "value": hyp_text},
            "rtf": {"type": "text", "value": str(round(rtf, 4))},
        }
        if "text" in sample:
            text = model.tokenizer.tokens2text(
                model.converter.ids2tokens(sample["text"])
            )
            output["ref"] = {"type": "text", "value": text}

        return output


def run_single_sample_inference(config, logger):
    runner = ASRInferenceRunner(
        model_config=config.model,
        dataset_config=config.dataset,
    )

    logger.info("Running debug inference on the first sample of each test set.")
    for test_sets in config.dataset.test:
        test_key = test_sets.name
        dataset = runner.initialize_dataset(test_key)
        sample = dataset[0]

        result = runner.run_on_example(test_key, sample)
        for key, val in result.items():
            logger.info("%s - %s: %s", test_key, key, val["value"])
            print(f"{key}: {val['value']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str,
                        choices=["decode", "score", "all"],
                        default="all")
    parser.add_argument(
        "--config",
        type=str,
        default="evaluate.yaml",
        help="Path to evaluation config (e.g., evaluate.yaml)",
    )
    parser.add_argument("--debug_sample", action="store_true",
                        help="Run debug inference on one sample")

    args = parser.parse_args()

    base_config = load_config_with_defaults(args.config)

    _, logging_options = setup_logging_from_config(getattr(base_config, "logging", None))
    logger = logging.getLogger("espnet3.evaluate")

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
        expdir=getattr(base_config, "expdir", None),
        decode_dir=getattr(base_config, "decode_dir", None),
    )

    log_configuration(logger, base_config)
    if getattr(base_config, "parallel", None) is not None:
        log_configuration(logger, base_config.parallel, header="Parallel Configuration")

    if getattr(base_config, "dataset", None) and getattr(base_config.dataset, "test", None):
        test_names = [ds_conf.name for ds_conf in base_config.dataset.test]
        logger.info("Configured test sets: %s", test_names)

    if getattr(base_config, "metrics", None):
        logger.info("Configured metrics: %s", [m._target_ for m in base_config.metrics])

    if args.stage in ["decode", "all"]:
        if args.debug_sample:
            run_single_sample_inference(base_config, logger)
        else:
            config = load_config_with_defaults(args.config)

            runner = ASRInferenceRunner(
                model_config=config.model,
                dataset_config=config.dataset,
                parallel=config.parallel,
            )
            test_keys = [ds_conf.name for ds_conf in config.dataset.test]

            logger.info("Starting batched decoding for test sets: %s", test_keys)

            for test_key in test_keys:
                output_dir = Path(config.decode_dir) / test_key
                logger.info("Decoding dataset '%s' into %s", test_key, output_dir)
                runner.run_on_dataset(test_key, output_dir=str(output_dir))

    if args.stage in ["score", "all"]:
        config = load_config_with_defaults(args.config)
        runner = ScoreRunner(config, config.decode_dir)
        results = runner.run()

        # Print results summary
        logger.info("===== Score Summary =====")
        print("\n===== Score Summary =====")
        for metric_name, test_results in results.items():
            logger.info("Metric: %s", metric_name)
            print(f"Metric: {metric_name}")
            for test_name, scores in test_results.items():
                logger.info("  [%s]", test_name)
                print(f"  [{test_name}]")
                for k, v in scores.items():
                    logger.info("    %s: %s", k, v)
                    print(f"    {k}: {v}")
        logger.info("=========================")
        print("=========================")


if __name__ == "__main__":
    main()
