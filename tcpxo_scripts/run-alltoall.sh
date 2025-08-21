#!/bin/bash
# Copyright 2025 Google LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE.md file or at
# https://developers.google.com/open-source/licenses/bsd


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

NCCL_NCHANNELS_PER_NET_PEER=1 \
NCCL_NSOCKS_PERTHREAD=2 \
  "${SCRIPT_DIR}"/run-nccl.sh alltoall_perf "${LD_LIBRARY_PATH}:/third_party/nccl-netsupport/build/lib:/nccl-plugin-gpudirecttcpx/build" "$@"
