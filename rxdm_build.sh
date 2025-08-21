#!/bin/bash
# Copyright 2025 Google LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE.md file or at
# https://developers.google.com/open-source/licenses/bsd

set -xe

bazel run webrtc:build_sctp -- $(realpath webrtc)

bazel build --compilation_mode=opt //buffer_mgmt_daemon:fastrak_gpumem_manager

mkdir -p out/

cp -f bazel-bin/buffer_mgmt_daemon/fastrak_gpumem_manager out/

docker build -f rxdm.dockerfile .
