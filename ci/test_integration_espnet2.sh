#!/usr/bin/env bash

set -euo pipefail

source tools/activate_python.sh
PYTHONPATH="${PYTHONPATH:-}:$(pwd)/tools/s3prl"
export PYTHONPATH
python="coverage run --append"
cwd=$(pwd)

if [ $# -gt 2 ]; then
    echo "Usage: $0 [task]"
    exit 1;
elif [ $# -eq 1 ]; then
    task="$1"
elif [ $# -eq 0 ]; then
    task="asr"
fi

gen_dummy_coverage(){
    # To avoid a problem when parallel running for `coverage run`.
    # Please put this command after cd ./egs2/foo/bar
    touch empty.py; ${python} empty.py
}

#### Make sure chainer-independent ####
python3 -m pip uninstall -y chainer

# First uninstall all espnet-related dependencies including all extras.
# I use toml and load the pyproject.toml
python3 -m pip install toml
python3 test_utils/uninstall_extra.py

# [ESPnet2] test asr recipe
# Install ASR dependency
python3 -m pip install -e '.[task-asr]'

# Run tests
cd ./egs2/mini_an4/asr1
./run.sh --stage 1 --stop-stage 1

if [ "${task}" == "asr" ] || [ "${task}" == "all" ]; then
    gen_dummy_coverage
    echo "==== [ESPnet2] ASR ==="
    ./run.sh --stage 1 --stop-stage 1
    feats_types="raw fbank_pitch"
    token_types="bpe char"
    for t in ${feats_types}; do
        ./run.sh --stage 2 --stop-stage 4 --feats-type "${t}" --python "${python}"
    done
    cp -r dump/raw data/
    ./run.sh --stage 2 --stop-stage 4 --feats-type "raw_copy" \
        --train_set raw/train_nodev --valid_set raw/train_dev --test_sets raw/test --python "${python}"
    for t in ${token_types}; do
        ./run.sh --stage 5 --stop-stage 5 --token-type "${t}" --python "${python}"
    done
    use_lm=true
    for t in ${feats_types}; do
        for t2 in ${token_types}; do
            echo "==== feats_type=${t}, token_types=${t2} ==="
            ./run.sh --use_lm ${use_lm} --ngpu 0 --stage 6 --stop-stage 13 --skip-packing false --feats-type "${t}" --token-type "${t2}" --python "${python}" --asr-args "--num_workers 0"
        done
        use_lm=false
        echo "==== feats_type=raw_copy, token_types=bpe ==="
        cp -r dump/raw data/
        ./run.sh --use_lm ${use_lm} --ngpu 0 --stage 4 --stop-stage 13 --skip-packing false --feats-type "raw_copy" --token-type "${t2}" \
            --train_set raw/train_nodev --valid_set raw/train_dev --test_sets raw/test --python "${python}" --asr-args "--num_workers 0"
    done
    echo "==== feats_type=raw, token_types=bpe, model_conf.extract_feats_in_collect_stats=False, normalize=utt_mvn ==="
    ./run.sh --ngpu 0 --stage 10 --stop-stage 13 --skip-packing false --feats-type "raw" --token-type "bpe" \
        --feats_normalize "utterance_mvn" --python "${python}" \
        --asr-args "--model_conf extract_feats_in_collect_stats=false --num_workers 0"

    echo "==== feats_type=raw, token_types=bpe, model_conf.extract_feats_in_collect_stats=False, normalize=utt_mvn, with data augmentation ==="
    ./run.sh --ngpu 0 --stage 10 --stop-stage 13 --skip-packing false --feats-type "raw" --token-type "bpe" \
        --asr_config "conf/train_asr_rnn_data_aug_debug.yaml" \
        --feats_normalize "utterance_mvn" --python "${python}" \
        --asr-args "--model_conf extract_feats_in_collect_stats=false --num_workers 0"

    echo "==== use_streaming, feats_type=raw, token_types=bpe, model_conf.extract_feats_in_collect_stats=False, normalize=utt_mvn ==="
    ./run.sh --use_streaming true --ngpu 0 --stage 10 --stop-stage 13 --skip-packing false --feats-type "raw" --token-type "bpe" \
        --feats_normalize "utterance_mvn"  --python "${python}" \
        --asr_config "" --asr-tag "train_raw_bpe_streaming" \
        --asr-args "--model_conf extract_feats_in_collect_stats=false --encoder=contextual_block_transformer
                    --encoder_conf='{'block_size': 40, 'hop_size': 16, 'look_ahead': 16, 'output_size': 2, 'attention_heads': 2, 'linear_units': 2, 'num_blocks': 1}'
                    --decoder=transformer --decoder_conf='{'attention_heads': 2, 'linear_units': 2, 'num_blocks': 1}'
                    --max_epoch 1 --num_iters_per_epoch 1 --batch_size 2 --batch_type folded --num_workers 0"

    if python3 -c "from warprnnt_pytorch import RNNTLoss" &> /dev/null; then
        echo "==== Transducer, feats_type=raw, token_types=bpe ==="
        ./run.sh --asr-tag "espnet_model_transducer" --ngpu 0 --stage 10 --stop-stage 13 --skip-packing false \
            --feats-type "raw" --token-type "bpe" --python "${python}" \
            --asr-args "--decoder transducer --decoder_conf hidden_size=2 --model_conf ctc_weight=0.0 --joint_net_conf joint_space_size=2 --num_workers 0 \
            --best_model_criterion '(valid, loss, min)'" --inference_asr_model "valid.loss.best.pth"

        if [ "$(python3 -c "import torch; print(torch.cuda.is_available())")" == "True" ]; then
            echo "==== Multi-Blank Transducer, feats_type=raw, token_types=bpe ==="
            ./run.sh --asr-tag "espnet_model_multi_blank_transducer" --ngpu 1 --stage 10 --stop-stage 13 --skip-packing false \
                --feats-type "raw" --token-type "bpe" --python "${python}" \
                --asr-tag "train_multi_black_transducer" \
                --asr_args "--decoder transducer --decoder_conf hidden_size=2 --model_conf ctc_weight=0.0 --joint_net_conf joint_space_size=2 \
                            --best_model_criterion '(valid, loss, min)' --model_conf transducer_multi_blank_durations=[2] \
                            --max_epoch 1 --num_iters_per_epoch 1 --batch_size 2 --batch_type folded --num_workers 0" \
                --inference_asr_model "valid.loss.best.pth" --inference_config "conf/decode_multi_blank_transducer_debug.yaml"
        fi
    fi

    if python3 -c "import k2" &> /dev/null; then
        echo "==== use_k2, num_paths > nll_batch_size, feats_type=raw, token_types=bpe, model_conf.extract_feats_in_collect_stats=False, normalize=utt_mvn ==="
        ./run.sh --num_paths 4 --nll_batch_size 2 --use_k2 true --ngpu 0 --stage 12 --stop-stage 13 --skip-packing false --feats-type "raw" --token-type "bpe" \
            --feats_normalize "utterance_mvn" --python "${python}" --asr-args "--model_conf extract_feats_in_collect_stats=false --num_workers 0"

        echo "==== use_k2, num_paths == nll_batch_size, feats_type=raw, token_types=bpe, model_conf.extract_feats_in_collect_stats=False, normalize=utt_mvn ==="
        ./run.sh --num_paths 2 --nll_batch_size 2 --use_k2 true --ngpu 0 --stage 12 --stop-stage 13 --skip-packing false --feats-type "raw" --token-type "bpe" \
        --feats_normalize "utterance_mvn" --python "${python}" --asr-args "--model_conf extract_feats_in_collect_stats=false --num_workers 0"
    fi

    if python3 -c "from warprnnt_pytorch import RNNTLoss" &> /dev/null; then
        echo "==== [ESPnet2] ASR Transducer (standalone) ==="

        for t in ${token_types}; do
            asr_tag="transducer_${t}"

            echo "==== [Conformer-RNN-T] feats_type=raw, token_types=${t}, model_conf.extract_feats_in_collect_stats=False, normalize=utt_mvn ==="
            ./run.sh --asr_config "" --asr_task "asr_transducer" --ngpu 0 --stage 10 --stop-stage 13 --skip-packing false --feats-type "raw" --token-type ${t} \
                --feats_normalize "utterance_mvn" --python "${python}" --inference_asr_model "valid.loss.best.pth" \
                --asr-tag "${asr_tag}_conformer" \
                --asr-args "--model_conf extract_feats_in_collect_stats=false \
                            --encoder_conf body_conf='[{'block_type': 'conformer', 'hidden_size': 2, 'linear_size': 4, 'heads': 2, 'conv_mod_kernel_size': 3}]' \
                            --decoder_conf='{'embed_size': 4, 'hidden_size': 4}' --joint_network_conf joint_space_size=4 \
                            --max_epoch 1 --num_iters_per_epoch 1 --batch_size 2 --batch_type folded --num_workers 0"

            echo "==== [Streaming Conformer-RNN-T] feats_type=raw, token_types=${t}, model_conf.extract_feats_in_collect_stats=False, normalize=utt_mvn ==="
            ./run.sh --asr_config "" --asr_task "asr_transducer" --ngpu 0 --stage 10 --stop-stage 13 --skip-packing false --feats-type "raw" --token-type ${t} \
                --feats_normalize "utterance_mvn" --python "${python}" --inference_asr_model "valid.loss.best.pth" \
                --asr-tag "${asr_tag}_conformer_streaming" \
                --asr-args "--model_conf extract_feats_in_collect_stats=false \
                            --encoder_conf main_conf='{'dynamic_chunk_training': True}' \
                            --encoder_conf body_conf='[{'block_type': 'conformer', 'hidden_size': 2, 'linear_size': 4, 'heads': 2, 'conv_mod_kernel_size': 3}]' \
                            --decoder_conf='{'embed_size': 4, 'hidden_size': 4}' --joint_network_conf joint_space_size=4 \
                            --max_epoch 1 --num_iters_per_epoch 1 --batch_size 2 --batch_type folded --num_workers 0" \
                --inference-args "--streaming true --decoding_window 160 --left_context 2"
        done
    fi

    echo "==== [PIT_ASR] feats_type=raw, token_types=bpe, model_conf.extract_feats_in_collect_stats=False, normalize=utt_mvn ==="
    for i in $(seq 2); do
        for rr in raw raw/org; do
            cp dump/${rr}/train_nodev/text dump/${rr}/train_nodev/text_spk${i}
            cp dump/${rr}/train_dev/text dump/${rr}/train_dev/text_spk${i}
        done
        cp dump/raw/test/text dump/raw/test/text_spk${i}
        cp dump/raw/test_seg/text dump/raw/test_seg/text_spk${i}
    done
    ./run_multispkr.sh --ngpu 0 --stage 10 --stop-stage 13 --skip-packing false --feats-type "raw" --token-type "bpe" \
        --feats_normalize "utterance_mvn" --python "${python}" \
        --asr_config "" \
        --asr_tag "train_multispkr_raw_en_bpe30" \
        --asr-args "--model_conf extract_feats_in_collect_stats=false \
                    --ctc_conf reduce=False --encoder transformer_multispkr \
                    --encoder_conf '{'num_blocks': 2, 'num_blocks_sd': 2, 'num_inf': 2, 'output_size': 2, 'attention_heads': 2, 'linear_units': 2}' \
                    --decoder rnn \
                    --model pit_espnet --model_conf '{'num_inf': 2, 'num_ref': 2}' \
                    --preprocessor multi --preprocessor_conf text_name='['text', 'text_spk2']' \
                    --max_epoch 1 --num_iters_per_epoch 1 --batch_size 2 --batch_type folded --num_workers 0" \
        --inference-args "--multi_asr true"

    # Remove generated files in order to reduce the disk usage
    rm -rf exp dump data
fi
cd "${cwd}"

# Uninstall task-dependency
python3 test_utils/uninstall_extra.py

# [ESPnet2] test tts recipe
# Install TTS dependency
python3 -m pip install -e '.[task-tts]'

# Run tests
cd ./egs2/mini_an4/tts1

if [ "${task}" == "tts" ] || [ "${task}" == "all" ]; then
    gen_dummy_coverage
    echo "==== [ESPnet2] TTS ==="
    ./run.sh --ngpu 0 --stage 1 --stop-stage 7 --skip-packing false --python "${python}" --train-args "--num_workers 0"
    # Remove generated files in order to reduce the disk usage
    rm -rf exp dump data

    # [ESPnet2] test gan-tts recipe
    # NOTE(kan-bayashi): pytorch 1.4 - 1.6 works but 1.6 has a problem with CPU,
    #   so we test this recipe using only pytorch > 1.6 here.
    #   See also: https://github.com/pytorch/pytorch/issues/42446
    if python3 -c 'import torch as t; from packaging.version import parse as L; assert L(t.__version__) > L("1.6")' &> /dev/null; then
        ./run.sh --fs 22050 --tts_task gan_tts --feats_extract linear_spectrogram --feats_normalize none --inference_model latest.pth \
            --ngpu 0 --stop-stage 7 --skip-packing false --python "${python}" \
            --train-config "" --train-args "--max_epoch 1 --num_iters_per_epoch 1 --batch_size 1 --batch_type folded --num_workers 0"
        rm -rf exp dump data
    fi
fi
cd "${cwd}"
# Uninstall task-dependency
python3 test_utils/uninstall_extra.py


# [ESPnet2] test asr2 recipe
# Install ASR2 dependency
python3 -m pip install -e '.[task-asr2]'
cd ./egs2/mini_an4/asr2
gen_dummy_coverage
echo "==== [ESPnet2] ASR2 ==="
./run.sh --ngpu 0 --stage 1 --stop-stage 15 --skip-packing false --use-lm false --python "${python}" --asr-args "--num_workers 0"
# Remove generated files in order to reduce the disk usage
rm -rf exp dump data
cd "${cwd}"
# Uninstall task-dependency
python3 test_utils/uninstall_extra.py


# [ESPnet2] test enh recipe
# Install ENH dependency
# ENH + Speech2Text requires s2t dependency
python3 -m pip install -e '.[task-enh]'
python3 -m pip install -e '.[task-st]'

# Run tests
if python -c 'import torch as t; from packaging.version import parse as L; assert L(t.__version__) >= L("1.2.0")' &> /dev/null; then
    cd ./egs2/mini_an4/enh1
    if [ "${task}" == "enh" ] || [ "${task}" == "all" ]; then
        gen_dummy_coverage
        echo "==== [ESPnet2] ENH ==="
        ./run.sh --stage 1 --stop-stage 1 --python "${python}"
        feats_types="raw"
        for t in ${feats_types}; do
            echo "==== feats_type=${t} with preprocessor ==="
            ./run.sh --ngpu 0 --stage 2 --stop-stage 10 --skip-packing false --feats-type "${t}" --ref-num 1 --python "${python}" \
                --extra_wav_list "rirs.scp noises.scp" --enh_config ./conf/train_with_preprocessor_debug.yaml --enh-args "--num_workers 0"
            ./run.sh --ngpu 0 --stage 5 --stop-stage 10 --skip-packing false --feats-type "${t}" --ref-num 1 --python "${python}" \
                --enh_config conf/train_with_data_aug_debug.yaml --enh-args "--num_workers 0"
            ./run.sh --ngpu 0 --stage 2 --stop-stage 10 --skip-packing false --feats-type "${t}" --ref-num 2 --python "${python}" \
                --enh_config conf/train_with_dynamic_mixing_debug.yaml --enh-args "--num_workers 0"
        done
        rm data/**/utt2category 2>/dev/null || true
        rm -r dump
        for t in ${feats_types}; do
            echo "==== feats_type=${t} without preprocessor ==="
            ./run.sh --ngpu 0 --stage 2 --stop-stage 10 --skip-packing false --feats-type "${t}" --ref-num 1 --python "${python}" --enh-args "--num_workers 0"
            ./run.sh --ngpu 0 --stage 6 --stop-stage 10 --skip-packing false --feats-type "${t}" --ref-num 1 --python "${python}" \
                --enh_config conf/train_with_chunk_iterator_debug.yaml --enh-args "--num_workers 0"
        done
        # Remove generated files in order to reduce the disk usage
        rm -rf exp dump data
    fi
    cd "${cwd}"
fi

# [ESPnet2] test enh_tse recipe
if python -c 'import torch as t; from packaging.version import parse as L; assert L(t.__version__) >= L("1.2.0")' &> /dev/null; then
    cd ./egs2/mini_an4/tse1
    if [ "${task}" == "tse" ] || [ "${task}" == "all" ]; then
        gen_dummy_coverage
        echo "==== [ESPnet2] ENH_TSE ==="
        feats_types="raw"
        for t in ${feats_types}; do
            echo "==== feats_type=${t} ==="
            ./run.sh --ngpu 0 --stage 1 --stop-stage 10 --skip-packing false --feats-type "${t}" --ref-num 1 --python "${python}" --enh-args "--num_workers 0"
            ./run.sh --ngpu 0 --stage 3 --stop-stage 6 --skip-packing false --feats-type "${t}" --ref-num 1 --python "${python}" \
                --train_set train_nodev_unk_nspk --valid_set test_unk_nspk --test_sets "train_dev_unk_nspk" \
                --enh_config ./conf/train_variable_nspk_debug.yaml --enh-args "--num_workers 0" --variable_num_refs true
            ./run.sh --ngpu 0 --stage 1 --stop-stage 10 --skip-packing false --feats-type "${t}" --ref-num 1 --python "${python}" \
                --local_data_opts "--random-enrollment true" --enh_config ./conf/train_random_enrollment_debug.yaml --enh-args "--num_workers 0"
            ./run.sh --ngpu 0 --stage 3 --stop-stage 6 --skip-packing false --feats-type "${t}" --ref-num 1 --python "${python}" \
                --train_set train_nodev_unk_nspk --valid_set test_unk_nspk --test_sets "train_dev_unk_nspk" \
                --enh_config ./conf/train_variable_nspk_random_enrollment_debug.yaml --enh-args "--num_workers 0" --variable_num_refs true
        done
        # Remove generated files in order to reduce the disk usage
        rm -rf exp dump data
    fi
    cd "${cwd}"
fi

# [ESPnet2] test enh_asr1 recipe
python3 -m pip install -e '.[task-asr]'
if python -c 'import torch as t; from packaging.version import parse as L; assert L(t.__version__) >= L("1.2.0")' &> /dev/null; then
    cd ./egs2/mini_an4/enh_asr1
    gen_dummy_coverage
    echo "==== [ESPnet2] ENH_ASR ==="
    ./run.sh --ngpu 0 --stage 0 --stop-stage 15 --skip-packing false --skip-upload_hf false --feats-type "raw" --spk-num 1 --enh_asr_args "--enh_separator_conf num_spk=1 --num_workers 0" --python "${python}"
    # Remove generated files in order to reduce the disk usage
    rm -rf exp dump data
    cd "${cwd}"
fi
# Uninstall task-dependency
python3 test_utils/uninstall_extra.py

# [ESPnet2] test ssl1 recipe
if python3 -c 'import torch as t; from packaging.version import parse as L; assert L(t.__version__) >= L("1.12.0")' &> /dev/null; then
    cd ./egs2/mini_an4/ssl1
    if [ "${task}" == "ssl" ] || [ "${task}" == "all" ]; then
        gen_dummy_coverage
        echo "==== [ESPnet2] SSL1/HUBERT ==="
        ./run.sh --ngpu 0 --stage 1 --stop-stage 7 --feats-type "raw" --token_type "word" --skip_upload_hf false --python "${python}" --hubert-args "--num_workers 0"
        # Remove generated files in order to reduce the disk usage
        rm -rf exp dump data
    fi
    cd "${cwd}"
fi

# [ESPnet2] test enh_asr1 recipe
if python -c 'import torch as t; from packaging.version import parse as L; assert L(t.__version__) >= L("1.2.0")' &> /dev/null; then
    cd ./egs2/mini_an4/enh_asr1
    if [ "${task}" == "enh_asr" ] || [ "${task}" == "all" ]; then
        gen_dummy_coverage
        echo "==== [ESPnet2] ENH_ASR ==="
        ./run.sh --ngpu 0 --stage 0 --stop-stage 15 --skip-packing false --skip-upload_hf false --feats-type "raw" --spk-num 1 --enh_asr_args "--enh_separator_conf num_spk=1 --num_workers 0" --python "${python}"
        # Remove generated files in order to reduce the disk usage
        rm -rf exp dump data
    fi
    cd "${cwd}"
fi

# [ESPnet2] test st recipe
# Install ST dependency
python3 -m pip install -e '.[task-st]'

# Run tests
cd ./egs2/mini_an4/st1

if [ "${task}" == "st" ] || [ "${task}" == "all" ]; then
    echo "==== [ESPnet2] ST ==="
    ./run.sh --stage 1 --stop-stage 1
    feats_types="raw fbank_pitch"
    token_types="bpe char"
    for t in ${feats_types}; do
        ./run.sh --stage 2 --stop-stage 4 --feats-type "${t}" --python "${python}"
    done
    for t in ${token_types}; do
        ./run.sh --stage 5 --stop-stage 5 --tgt_token_type "${t}" --src_token_type "${t}" --python "${python}"
    done
    use_lm=true
    for t in ${feats_types}; do
        for t2 in ${token_types}; do
            echo "==== feats_type=${t}, token_types=${t2} ==="
            ./run.sh --use_lm ${use_lm} --ngpu 0 --stage 6 --stop-stage 13 --skip-packing false --feats-type "${t}" --tgt_token_type "${t2}" --src_token_type "${t2}" --python "${python}" --st-args "--num_workers 0"
        done
        use_lm=false
    done
    echo "==== feats_type=raw, token_types=bpe, model_conf.extract_feats_in_collect_stats=False, normalize=utt_mvn ==="
    ./run.sh --ngpu 0 --stage 10 --stop-stage 13 --skip-packing false --feats-type "raw" --tgt_token_type "bpe" --src_token_type "bpe" \
        --feats_normalize "utterance_mvn" --python "${python}" --st-args "--model_conf extract_feats_in_collect_stats=false --num_workers 0"

    echo "==== use_streaming, feats_type=raw, token_types=bpe, model_conf.extract_feats_in_collect_stats=False, normalize=utt_mvn ==="
    ./run.sh --use_streaming true --ngpu 0 --stage 10 --stop-stage 13 --skip-packing false --feats-type "raw" --tgt_token_type "bpe" --src_token_type "bpe" \
        --feats_normalize "utterance_mvn" --python "${python}" \
        --st-config conf/train_st_streaming_debug.yaml --st-args "--model_conf extract_feats_in_collect_stats=false --num_workers 0"

    # Remove generated files in order to reduce the disk usage
    rm -rf exp dump data
fi
cd "${cwd}"

# [ESPnet2] test asr2 recipe
cd ./egs2/mini_an4/asr2
if [ "${task}" == "asr2" ] || [ "${task}" == "all" ]; then
    gen_dummy_coverage
    echo "==== [ESPnet2] ASR2 ==="
    ./run.sh --ngpu 0 --stage 1 --stop-stage 15 --skip-packing false --use-lm false --python "${python}" --asr-args "--num_workers 0"
    # Remove generated files in order to reduce the disk usage
    rm -rf exp dump data
fi
cd "${cwd}"

# [ESPnet2] test spk1 recipe
# Install SPK dependency
python3 -m pip install -e '.[task-spk]'

# Run tests
cd ./egs2/mini_an4/spk1
if [ "${task}" == "spk" ] || [ "${task}" == "all" ]; then
    gen_dummy_coverage
    echo "==== [ESPnet2] SPK ==="
    ./run.sh --ngpu 0 --stage 0 --stop-stage 4 --feats-type "raw" --python "${python}" --spk-args "--num_workers 0"
    ./run.sh --ngpu 0 --stage 5 --stop-stage 5 --feats-type "raw" --python "${python}" --spk_config conf/train_rawnet3_dataaug_debug.yaml --spk-args "--num_workers 0"
    ./run.sh --ngpu 0 --stage 5 --stop-stage 5 --feats-type "raw" --python "${python}" --spk_config conf/train_rawnet3_sampler.yaml --spk-args "--num_workers 0"
    ./run.sh --ngpu 0 --stage 5 --stop-stage 5 --feats-type "raw" --python "${python}" --spk_config conf/train_ecapa.yaml --spk-args "--num_workers 0"
    ./run.sh --ngpu 0 --stage 5 --stop-stage 5 --feats-type "raw" --python "${python}" --spk_config conf/train_xvector.yaml --spk-args "--num_workers 0"
    ./run.sh --ngpu 0 --stage 5 --stop-stage 5 --feats-type "raw" --python "${python}" --spk_config conf/train_ska.yaml --spk-args "--num_workers 0"
    ./run.sh --ngpu 0 --stage 5 --stop-stage 5 --feats-type "raw" --python "${python}" --spk_config conf/train_identity.yaml --spk-args "--num_workers 0"
    ./run.sh --ngpu 0 --stage 5 --stop-stage 5 --feats-type "raw" --python "${python}" --spk_config conf/train_conformer.yaml --spk-args "--num_workers 0"
    ./run.sh --ngpu 0 --stage 6 --stop-stage 7 --feats-type "raw" --python "${python}" --spk_config conf/train_rawnet3_sampler.yaml --spk-args "--num_workers 0" --inference_model "valid.eer.ave.pth"
    # Remove generated files in order to reduce the disk usage
    rm -rf exp dump data
fi
cd "${cwd}"
# Uninstall task-dependency
python3 test_utils/uninstall_extra.py

# [ESPnet2] test s2t1 recipe
# # Install s2t dependency
python3 -m pip install -e '.[task-s2t]'

cd ./egs2/mini_an4/s2t1
if [ "${task}" == "s2t" ] || [ "${task}" == "all" ]; then
    gen_dummy_coverage
    echo "==== [ESPnet2] S2T ==="
    ./run.sh --ngpu 0 --stage 1 --stop_stage 13 --use_lm false --feats_type raw --audio_format flac.ark --token_type bpe --python "${python}"
    # Remove generated files in order to reduce the disk usage
    rm -rf exp dump data
fi
cd "${cwd}"
# Uninstall task-dependency
python3 test_utils/uninstall_extra.py

# [ESPnet2] test s2st1 recipe
# # Install s2st dependency
python3 -m pip install -e '.[task-s2st]'

# [ESPnet2] test s2st1 recipe
cd ./egs2/mini_an4/s2st1
if [ "${task}" == "s2st" ] || [ "${task}" == "all" ]; then
    gen_dummy_coverage
    echo "==== [ESPnet2] S2ST ==="
    ./run.sh --ngpu 0 --stage 1 --stop_stage 8 --use_discrete_unit false --s2st_config conf/s2st_spec_debug.yaml --python "${python}"
    if python3 -c "import s3prl" &> /dev/null; then
        ./run.sh --ngpu 0 --stage 1 --stop_stage 8 --python "${python}" --use_discrete_unit true --s2st_config conf/train_s2st_discrete_unit_debug.yaml --clustering_num_threads 2 --feature_num_clusters 5
    fi
    # Remove generated files in order to reduce the disk usage
    rm -rf exp dump data ckpt .cache
fi
cd "${cwd}"
# Uninstall task-dependency
python3 test_utils/uninstall_extra.py

# [ESPnet2] test lm1 recipe
cd ./egs2/mini_an4/lm1
if [ "${task}" == "lm" ] || [ "${task}" == "all" ]; then
    gen_dummy_coverage
    echo "==== [ESPnet2] LM ==="
    ./run.sh --ngpu 0 --stage 1 --stop-stage 12 --python "${python}"
    # Remove generated files in order to reduce the disk usage
    rm -rf exp dump data
fi
cd "${cwd}"

# [ESPnet2] test codec1 recipe
cd ./egs2/mini_an4/codec1
if [ "${task}" == "codec" ] || [ "${task}" == "all" ]; then
    gen_dummy_coverage
    echo "==== [ESPnet2] Codec ==="
    ./run.sh --ngpu 0 --stage 1 --stop_stage 6 --python "${python}"
    # Remove generated files in order to reduce the disk usage
    rm -rf exp dump data
fi
cd "${cwd}"

echo "=== report ==="
coverage combine egs2/*/*/.coverage
coverage report -i
coverage xml -i
