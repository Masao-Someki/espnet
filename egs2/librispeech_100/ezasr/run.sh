#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=4

set -e
set -u
set -o pipefail

. ../asr1/cmd.sh 

BASE_DIR=/jet/home/someki/workspace/espnet

. ${BASE_DIR}/tools/activate_python.sh

python main.py

