import argparse
import time
import os
from pathlib import Path

import torch.nn as nn
from omegaconf import OmegaConf
from hydra.utils import instantiate

# from espnet2.bin.asr_inference_ctc import Speech2Text
from espnet3.inference.inference_runner import InferenceRunner
from espnet3.utils.config import load_config_with_defaults
from espnet3.inference.score_runner import ScoreRunner



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
        results = model(
            speech,
        )
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
            output["ref"] = {"type": "text", "value": sample["text"]}

        return output


def run_single_sample_inference(config_path):
    config = load_config_with_defaults(config_path)

    runner = ASRInferenceRunner(
        model_config=config.model,
        dataset_config=config.dataset,
    )

    for test_sets in config.dataset.test:
        test_key = test_sets.name
        dataset = runner.initialize_dataset(test_key)
        sample = dataset[0]

        result = runner.run_on_example(test_key, sample)
        for key, val in result.items():
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

    if args.stage in ["decode", "all"]:
        if args.debug_sample:
            run_single_sample_inference(args.config)
        else:
            config = load_config_with_defaults(args.config)

            runner = ASRInferenceRunner(
                model_config=config.model,
                dataset_config=config.dataset,
                parallel=config.parallel,
            )
            test_keys = [ds_conf.name for ds_conf in config.dataset.test]

            for test_key in test_keys:
                runner.run_on_dataset(test_key, output_dir=f"{config.decode_dir}/{test_key}")

    if args.stage in ["score", "all"]:
        config = load_config_with_defaults(args.config)
        runner = ScoreRunner(config, config.decode_dir)
        results = runner.run()

        # Print results summary
        print("\n===== Score Summary =====")
        for metric_name, test_results in results.items():
            print(f"Metric: {metric_name}")
            for test_name, scores in test_results.items():
                print(f"  [{test_name}]")
                for k, v in scores.items():
                    print(f"    {k}: {v}")
        print("=========================")


if __name__ == "__main__":
    main()
