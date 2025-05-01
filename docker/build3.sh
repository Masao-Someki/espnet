#!/usr/bin/env bash

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
build_cores=24
docker_ver=$(docker version -f '{{.Server.Version}}')

log "Using Docker Ver.${docker_ver}"

this_tag=espnet/espnet:dev3-cpu
docker build --build-arg FROM_IMAGE=ubuntu:24.04 \
    --build-arg DOCKER_BUILT_VER=${docker_ver} \
    --build-arg NUM_BUILD_CORES=${build_cores} \
    -f prebuilt/devel3.dockerfile -t ${this_tag} . 

this_tag=espnet/espnet:dev3-gpu
docker build --build-arg FROM_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu24.04 \
    --build-arg DOCKER_BUILT_VER=${docker_ver} \
    --build-arg NUM_BUILD_CORES=${build_cores} \
    -f prebuilt/devel3.dockerfile -t ${this_tag} . 
