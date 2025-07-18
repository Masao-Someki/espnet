#!/usr/bin/env bash

if [ ! -e tools/kaldi ]; then
    git clone https://github.com/kaldi-asr/kaldi --depth 1 tools/kaldi
fi

PATH=$(pwd)/test_utils/bats-core/bin:$(pwd)/shellcheck-stable:$PATH
if ! [ -x "$(command -v bats)" ]; then
    echo "=== install bats ==="
    git clone https://github.com/bats-core/bats-core.git "$(pwd)"/test_utils/bats-core
    git clone https://github.com/bats-core/bats-support.git "$(pwd)"/test_utils/bats-support
    git clone https://github.com/bats-core/bats-assert.git "$(pwd)"/test_utils/bats-assert
fi
if ! [ -x "$(command -v shellcheck)" ]; then
    echo "=== install shellcheck ==="
    wget https://github.com/koalaman/shellcheck/releases/download/stable/shellcheck-stable.linux.x86_64.tar.xz
    tar -xvf shellcheck-stable.linux.x86_64.tar.xz
fi
. tools/activate_python.sh
. tools/extra_path.sh

set -euo pipefail

echo "=== run shellcheck ==="
rm -fv tools/Miniconda*.sh  # exclude from schellcheck
rm -fv tools/Miniforge*.sh  # exclude from schellcheck
find ci doc egs3/ -name "*.sh" -printf "=> %p\n" -execdir shellcheck -Calways -x -e SC2001 -e SC1091 -e SC2086 {} \; | tee check_shellcheck

if grep -q "SC[0-9]\{4\}" check_shellcheck; then
    echo "[ERROR] shellcheck failed"
    exit 1
fi
