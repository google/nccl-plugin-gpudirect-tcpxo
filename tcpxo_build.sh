#!/bin/bash
# Copyright 2025 Google LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE.md file or at
# https://developers.google.com/open-source/licenses/bsd

set -xe

# This script currently operates under the assumption that rxdm_build has already been run and we have built webrtc.

bazel build --compilation_mode=opt //tcpdirect_plugin/fastrak_offload:libnccl-net.so

mkdir -p out

cp -f bazel-bin/tcpdirect_plugin/fastrak_offload/libnccl-net.so out/

./prepare_source.sh

docker build -f tcpxo.dockerfile .
