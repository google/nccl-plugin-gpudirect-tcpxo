#!/bin/bash
# Copyright 2025 Google LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE.md file or at
# https://developers.google.com/open-source/licenses/bsd

set -xe

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd "$SCRIPT_DIR"
git clone https://github.com/NVIDIA/nccl.git nccl-netsupport \
  || git -C nccl-netsupport fetch --all --tags
cd nccl-netsupport
git checkout -B github_nccl_2_26_5 3000e3c797b4b236221188c07aa09c1f3a0170d4
