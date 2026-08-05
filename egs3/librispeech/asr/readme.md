# LibriSpeech 960h ASR recipe

The recipe reads the original LibriSpeech directory layout. Place the corpus
under `download/LibriSpeech`, or set `LIBRISPEECH` to its parent directory.

## Quick start

```bash
source path.sh
python run.py --stages create_dataset \
    --training_config conf/tuning/training_e_branchformer.yaml
python run.py --stages train \
    --training_config conf/tuning/training_e_branchformer.yaml
python run.py --stages infer \
    --training_config conf/tuning/training_e_branchformer.yaml \
    --inference_config conf/inference.yaml
python run.py --stages measure \
    --training_config conf/tuning/training_e_branchformer.yaml \
    --metrics_config conf/metrics.yaml
```

Use `training_transformer.yaml` or `training_conformer.yaml` to select the
other encoder configurations.
