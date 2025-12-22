#!/bin/bash
# Copyright 2025 Google LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE.md file or at
# https://developers.google.com/open-source/licenses/bsd

set -xe

bazel run webrtc:build_sctp -- $(realpath webrtc)

bazel build --compilation_mode=opt --@rules_cuda//cuda:runtime=@local_cuda//:cuda_runtime_static //tcpxo_prober:prober

mkdir -p out/

cp -f bazel-bin/tcpxo_prober/prober out/
