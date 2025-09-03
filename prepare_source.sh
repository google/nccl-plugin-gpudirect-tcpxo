#!/bin/bash
# Copyright 2025 Google LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE.md file or at
# https://developers.google.com/open-source/licenses/bsd

set -xe

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

pushd "$SCRIPT_DIR"
# Assume failures are due to the repo already being cloned, and continue.
git clone https://github.com/NVIDIA/nccl.git nccl-netsupport || true
git -C nccl-netsupport fetch --all --tags
git -C nccl-netsupport checkout v2.28.3-1
popd
