import os
from argparse import Namespace

import lightning as L
import numpy as np
import torch
from datasets import concatenate_datasets, load_from_disk
from hydra.utils import instantiate
from lhotse.audio.backend import set_current_audio_backend
from omegaconf import OmegaConf

import espnet3 as ez
from espnet3.data import HuggingfaceDatasetsBackend, cutset_from_huggingface
from espnet3.parallel import set_parallel
from espnet3.trainer import ESPnetEZLightningTrainer, LitESPnetModel

print("IMPORTED", flush=True)

def load_line(path):
    with open(path, "r") as f:
        data = f.readlines()
    return [t.strip() for t in data]


def get_dataset(config):
    # load dataset
    train_dataset = load_from_disk(
        config.dataset.id,
    )["train-clean-100"]
    dev_dataset = concatenate_datasets([
        load_from_disk(
            config.dataset.id,
        )["dev-clean"],
    ])

    # Convert to Cut format
    data_info = {
        "text": lambda x: x["text"],
        "language": lambda x: "English",
        "speaker": lambda x: x['id'].split("-")[0],
        "channel": lambda x: 0,
    }

    # cutset_from_huggingface is parallelized with config.parallel
    train_cuts = cutset_from_huggingface(
        data_info, len(train_dataset), config.dataset.id, "dev-other"
    )
    dev_cuts = cutset_from_huggingface(
        data_info, len(dev_dataset), config.dataset.id, "dev-clean"
    )
    return train_cuts, dev_cuts


def get_dataset_ez(config, tokenize):
    # load dataset
    train_dataset = load_from_disk(
        config.dataset.id,
    )["train-clean-100"]
    dev_dataset = concatenate_datasets(
        [
            load_from_disk(
                config.dataset.id,
            )["dev-clean"],
        ]
    )
    # Convert to Cut format
    data_info = {
        "speech": lambda d: d["audio"]["array"].astype(np.float32),
        "text": lambda d: tokenize(d["text"]).astype(np.int64),
        # "text_prev": lambda d: tokenize("<na>"),
        # "text_ctc": lambda d: tokenize(d["text_ctc"]),
    }
    # Convert into ESPnet-EZ dataset format
    train_dataset = ez.data.ESPnetEZDataset(train_dataset, data_info=data_info)
    valid_dataset = ez.data.ESPnetEZDataset(dev_dataset, data_info=data_info)
    return train_dataset, valid_dataset


if __name__ == "__main__":
    # load config
    print("Loading config", flush=True)
    config = OmegaConf.load("egs3/config.yaml")
    OmegaConf.register_new_resolver("load_line", load_line)

    print("Config loaded", flush=True)

    # save the configuration for inference
    ez.save_espnet_config("asr", config, os.path.join(config.expdir, "config.yaml"))

    # Set random seed if required
    if getattr(config, "seed", None) is not None:
        assert isinstance(config.seed, int), "seed should be an integer"
        L.seed_everything(config.seed)

    # Set parallelism config
    set_parallel(config.parallel)
    print("Parallelism config set", flush=True)

    # Required if using Huggingface + Lhotse
    # set_current_audio_backend(HuggingfaceDatasetsBackend(
    #     config.dataset.id,
    # ))
    # print("Set audio backend", flush=True)

    # Get datase
    tokenizer = instantiate(config.tokenizer)
    converter = instantiate(config.converter)

    def tokenize(text):
        return np.array(converter.tokens2ids(tokenizer.text2tokens(text)))

    train_cuts, dev_cuts = get_dataset_ez(config, tokenize)
    print("Dataset loaded", flush=True)

    set_current_audio_backend(HuggingfaceDatasetsBackend(
        config.dataset.id,
    ))
    print("Set audio backend", flush=True)

    # Define model
    # model = LitESPnetModel(
    #     config.training,
    #     instantiate(config.model),
    #     train_dataset=train_cuts,
    #     valid_dataset=dev_cuts,
    # )
    from espnetez.task import get_ez_task
    model_config = OmegaConf.load("egs3/espnet_config.yaml")
    task = get_ez_task("asr")
    default_config = task.get_default_config()
    default_config.update(model_config.model)
    espnet_model = task.build_model(Namespace(**default_config))
    model = LitESPnetModel(
        config.training,
        espnet_model,
        train_dataset=train_cuts,
        valid_dataset=dev_cuts,
    )
    print(model)
    print("Model defined", flush=True)
    lightning_config = {} if not hasattr(config, "lightning") else config.lightning

    trainer = ESPnetEZLightningTrainer(
        config.trainer,
        model=model,
        **lightning_config,
    )
    print("Trainer defined", flush=True)

    fit_params = {} if not hasattr(config, "fit") else config.fit
    trainer.fit(**fit_params)
