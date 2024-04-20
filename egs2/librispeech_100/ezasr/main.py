import torch
import numpy as np
import librosa
from pathlib import Path

import espnetez as ez

BASE_DIR = Path("/jet/home/someki/workspace/espnet/egs2/librispeech_100/ezasr")


CONFIG = "owsm_finetune_base"
FINETUNE_MODEL = "espnet/owsm_v3.1_ebf_base"

DATA_PATH = f"{BASE_DIR}/data"
DUMP_DIR = f"{BASE_DIR}/dump/raw"
EXP_DIR = f"{BASE_DIR}/ezexp/train_{CONFIG}"
STATS_DIR = f"{BASE_DIR}/ezexp/stats_{CONFIG}"

training_config = ez.config.from_yaml(
    "asr",
    f"conf/{CONFIG}.yaml"
)
from espnet2.bin.s2t_inference import Speech2Text
pretrained_model = Speech2Text.from_pretrained(
    FINETUNE_MODEL,
    category_sym="<en>",
    beam_size=10,
    device="cpu"
)
tokenizer = pretrained_model.tokenizer
converter = pretrained_model.converter
del pretrained_model

def tokenize(text):
    return np.array(converter.tokens2ids(tokenizer.text2tokens(text)))

training_config['token_list'] = converter.token_list
data_info = {
    "speech": ["wav.scp", "sound"],
    "text": ["text", "text"],
    "text_prev": lambda d: tokenize("<na>"),
    "text_ctc": ["text", "text"],
}

trainer = ez.Trainer(
      task="asr",
      train_config=training_config,
      train_dump_dir=f"{DUMP_DIR}/train_clean_100_sp",
      valid_dump_dir=f"{DUMP_DIR}/dev",
      data_info=data_info,
      output_dir=EXP_DIR,
      stats_dir=STATS_DIR,
      ngpu=1,
  )
trainer.collect_stats()
trainer.train()

