#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

stage=1
stop_stage=100
skip_stages=
cpu_cmd="run.pl"
num_threads=64      # number of cpu threads in learn_kmeans
cuda_cmd="run.pl"
nj=16               # number of parallel jobs
python=python3      # Specify python to execute espnet commands.
train_set=          # Name of training set
dev_set=            # Name of valid set
other_sets=         # Name of other sets
datadir=dump/raw    # Directory for the source speech data used to dump feature and label.
featdir=dump/hubert_feats   # Directory for the dumped features and labels.
alignment_phoneme_dir="data/mfa_phoneme_alignment"  # Directory for alignment labels
phn_sets="dev-other dev-clean"      # Datasets of alignment used to measure the pseudo-label quality
upsample=           # Upsampling rate of pseudo-labels to measure the pseudo-lable quality
use_gpu=false       # Whether to use gpu in feature extraction
suffix=             # A suffix to distinguish the feature dump directory. Empty in usual cases.
audio_format="wav"  # The audio format of the source speech (flac, wav, *_ark, etc)

lsh_algorithm="simple"  # lsh_algorithm for random projection based LSH.
                    # 'simple' for simple LSH in https://arxiv.org/pdf/1410.5518.pdf and 'e2lsh' for E2LSH
n_bits=10       # Number of hyperplanes. The number of clusters will be 2^n_bits
seed=2024           # Random seed for random projection
w=1                 # w used to define bucket width in E2LSH. Details in https://www.mit.edu/~andoni/LSH/
storage_save_mode=false     # Save storage on SSL feature extraction
                            # If true, feature extraction and kmeans clustering on the fly

feature_conf=       # feature configuration in json string format
feature_type=mfcc   # mfcc / fairseq_hubert / espnet_hubert
layer=              # The layer index of SSL models to extract features from.
batch_bins=         # batch size when extracting features and labels.

# Legacy Fairseq HuBERT model and ESPnet-trained HuBERT model related for feature extraction.
# Example of legacy Fairseq HuBERT model
hubert_url="https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt"
hubert_dir_path="./downloads/hubert_pretrained_models/hubert_base_ls960.pt"
# Example of espnet-trained model
# hubert_url="espnet"
# hubert_dir_path="" # Pretrained Hubert model dir contains 'valid.acc.best.pth' and 'config.yaml'

log "$0 $*"
. utils/parse_options.sh

. ./path.sh

if [ $# -ne 0 ]; then
    echo "Usage: $0 <--nclusters:100> <--feature_type:mfcc>"
    exit 0
fi

if [ ${feature_type} = "mfcc" ]; then  # MFCC has no layer
    use_gpu=false
elif [ -z "${suffix}" ]; then
    suffix="layer${layer}/"
fi
if [ -z "${feature_conf}" ]; then
    feature_conf="{type=${feature_type}"
    if [ ${feature_type} = "espnet_hubert" ]; then
        feature_conf+=",conf={\
sample_rate=16000,hubert_model_path=${hubert_dir_path},\
layer=${layer}\
}"
    elif [ ${feature_type} = "fairseq_hubert" ]; then
        feature_conf+=",conf={\
sample_rate=16000,hubert_url=${hubert_url},\
hubert_dir_path=${hubert_dir_path},layer=${layer}\
}"
    elif [ ${feature_type} != "mfcc" ]; then
        log "Error: unsupported feature type ${feature_type}" && exit 2
    fi
    feature_conf+="}"
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && ! [[ " ${skip_stages} " =~ [[:space:]]1[[:space:]] ]]; then
    log "stage 1: Generate pseudo-labels with random projection"

    if ${use_gpu}; then
        _cmd="${cuda_cmd} --gpu 1"
    else
        _cmd="${cpu_cmd}"
    fi

    # for dset in "${train_set}" "${dev_set}" ${other_sets}; do
    for dset in "${dev_set}" ${other_sets}; do
        log "Extract labels to ${featdir}/${feature_type}/${suffix}${dset}"

        _dump_dir="${featdir}/${feature_type}/${suffix}${dset}"

        _opts=
        if ${storage_save_mode}; then
            utils/copy_data_dir.sh --validate_opts --non-print "${datadir}/${dset}" "${_dump_dir}"
            key="wav.scp"
            if [[ "${audio_format}" == *ark* ]]; then
                _opts+="--in_filetype kaldi_ark "
            else
                # "sound" supports "wav", "flac", etc.
                _opts+="--in_filetype sound "
            fi
            _opts+="--online_feature_extract ${storage_save_mode} "
            _opts+="--feature_conf \"${feature_conf}\" "
            if [ -n "${batch_bins}" ]; then
                _opts+="--batch_bins ${batch_bins} "
            fi
        else
            key="feats.scp"
            _opts+="--in_filetype mat "
        fi
        mkdir -p "${_dump_dir}"/logdir

        nutt=$(<"${_dump_dir}"/${key} wc -l)
        _nj=$((nj<nutt?nj:nutt))

        key_file="${_dump_dir}"/${key}
        split_scps=""
        for n in $(seq ${_nj}); do
            split_scps+=" ${_dump_dir}/logdir/inference_random_projection.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        for n in $(seq ${_nj}); do
            awk '(FILENAME==ARGV[1]){utt2num[$1]=$2} (FILENAME==ARGV[2]){print($1, utt2num[$1])}' \
                ${datadir}/${dset}/utt2num_samples ${_dump_dir}/logdir/inference_random_projection.${n}.scp \
                > ${_dump_dir}/logdir/utt2num_samples.${n}
        done

        ${_cmd} JOB=1:${_nj} "${_dump_dir}"/logdir/inference_pseudo_labels_rp_${lsh_algorithm}_${n_bits}_${w}.JOB.log \
            ${python} pyscripts/feats/dump_rp_label.py \
                ${_opts} \
                --out_filetype "mat" \
                --seed ${seed} \
                --n_bits ${n_bits} \
                --w ${w} \
                --lsh_algorithm ${lsh_algorithm} \
                --use_gpu ${use_gpu} \
                --utt2num_samples "${_dump_dir}/logdir/utt2num_samples.JOB" \
                "scp:${_dump_dir}/logdir/inference_random_projection.JOB.scp" \
                "ark,t:${_dump_dir}/logdir/pseudo_labels_rp_${lsh_algorithm}_${n_bits}_${w}.JOB.txt" || exit 1;

        # concatenate scp files
        for n in $(seq ${_nj}); do
            cat "${_dump_dir}"/logdir/pseudo_labels_rp_${lsh_algorithm}_${n_bits}_${w}.${n}.txt || exit 1;
        done | sed 's/ \[ \| \]//g' | sort -u > "${_dump_dir}"/pseudo_labels_rp_${lsh_algorithm}_${n_bits}_${w}.txt || exit 1;
    done
fi


rp_tag=$(basename ${km_dir})

if [ -n "${alignment_phoneme_dir}" ]; then
    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && ! [[ " ${skip_stages} " =~ [[:space:]]2[[:space:]] ]]; then
        log "Stage 2: Measure qualities of pseudo labels"

        if [ -z "${upsample}" ]; then
            # upsample the pseudo labels to match the length of alignment
            if [ "${feature_type}" = "mfcc" ]; then
                upsample=1
            else
                upsample=2
            fi
        fi

        if [ -d "${alignment_phoneme_dir}" ]; then
            # TODO(simpleoier): This script and arguments design are specific to LibriSpeech dataset.
            ${python} local/measure_teacher_quality.py \
                --lab_dir "${featdir}/${feature_type}/${suffix}" \
                --lab_name "pseudo_labels_km${nclusters}.txt" \
                --lab_sets "${dev_set}" \
                --phn_dir "${alignment_phoneme_dir}" \
                --phn_sets ${phn_sets} \
                --pad_len 0 \
                --upsample ${upsample} \
                --ref_lab_dir "" \
                --ref_lab_name "" | tee ${km_dir}/phoneme_pseudo_label_quality.txt
        else
            log "Skipping quality measurement because no ${alignment_phoneme_dir} exists. You can specify the \
alignment by \"--alignment_phoneme_dir\". The alignment is in tsv file with format: \"utt_id1 a1,a2,a3,...\""
        fi
    fi
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ] && ! [[ " ${skip_stages} " =~ [[:space:]]3[[:space:]] ]]; then
    log "stage 3: Prepare pseudo-labels for training and dictionary: <token> <count>"

    for dset in "${train_set}" "${dev_set}" ${other_sets}; do
        label_dir="${featdir}/${feature_type}/${suffix}${dset}"
        if [ -f "${label_dir}"/pseudo_labels_rp_${lsh_algorithm}_${n_bits}_${w}.txt ]; then
            cp "${label_dir}"/pseudo_labels_rp_${lsh_algorithm}_${n_bits}_${w}.txt ${datadir}/${dset}/text.rp.${rp_tag}
        fi
        utils/fix_data_dir.sh --utt_extra_files "text.rp.${rp_tag}" ${datadir}/${dset}
    done

    # generate dictionaries
    if [ -n "${dictdir}" ]; then
        mkdir -p ${dictdir}

        oov="<unk>"         # Out of vocabulary symbol.
        blank="<blank>"     # CTC blank symbol
        pad="<pad>"
        sos_eos="<sos/eos>" # sos and eos symbole

        <${datadir}/${train_set}/text.rp.${rp_tag} cut -d" " -f2- | \
            awk '{for (i=1; i<=NF; i++) {count[$i]+=1}} END{for (k in count) {print(k, count[k])}}' | \
                sort -n -r -k 2  | \
                awk -v oov=${oov} -v blank=${blank} -v sos_eos=${sos_eos} -v pad=${pad} \
                    '{print($1)} END{print(oov); print(sos_eos)}' \
                > ${dictdir}/tokens.txt

        log "Successfully generate the ${dictdir}/{dict,tokens}.txt"
    fi

fi
