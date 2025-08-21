#!/bin/bash
# Copyright 2025 Google LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE.md file or at
# https://developers.google.com/open-source/licenses/bsd


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

source "${SCRIPT_DIR}"/unix_client_prefix_selection.sh

# for example, use 2 for alltoall
export NCCL_NSOCKS_PERTHREAD=${NCCL_NSOCKS_PERTHREAD:-4}
export NCCL_NCHANNELS_PER_NET_PEER=${NCCL_NCHANNELS_PER_NET_PEER:-1}
export NCCL_GPUDIRECTTCPX_REPORT_NETWORK_LATENCY=${NCCL_GPUDIRECTTCPX_REPORT_NETWORK_LATENCY:-80}
export NCCL_ALGO=${NCCL_ALGO:-Ring} \

run_nccl() {
  local -r benchmark=$1
  local -r ld_library_path_override=$2
  local -r gpu_per_node=$3
  local -r socket_ifnames=$4
  local -r data_b=$5
  local -r data_e=$6
  local nhosts=2
  if ! [[ -z "$7" ]]; then
    nhosts=$7
  fi
  local iters=20
  if ! [[ -z "$8" ]]; then
    iters=$8
  fi

  for i in $(seq 1 1); do

  LD_LIBRARY_PATH=${ld_library_path_override} \
  mpirun --mca btl tcp,self --mca btl_tcp_if_include eth0 --allow-run-as-root \
    -np $(( gpu_per_node * "${nhosts}" )) \
    --hostfile "${SCRIPT_DIR}/hostfiles${nhosts}/hostfile${gpu_per_node}" \
    -x NCCL_SOCKET_IFNAME=eth0 \
    -x LD_LIBRARY_PATH -x PATH \
    -x NCCL_CROSS_NIC=0 \
    -x NCCL_ALGO \
    -x NCCL_PROTO=Simple \
    -x NCCL_NSOCKS_PERTHREAD \
    -x NCCL_SOCKET_NTHREADS=1 \
    -x NCCL_DYNAMIC_CHUNK_SIZE=524288 \
    -x NCCL_BUFFSIZE=4194304 \
    -x CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -x NCCL_GPUDIRECTTCPX_SOCKET_IFNAME="${socket_ifnames}" \
    -x NCCL_GPUDIRECTTCPX_CTRL_DEV=eth0 \
    -x NCCL_NET_GDR_LEVEL=PIX \
    -x NCCL_P2P_PXN_LEVEL=2 \
    -x NCCL_NCHANNELS_PER_NET_PEER \
    -x NCCL_P2P_NET_CHUNKSIZE=524288 \
    -x NCCL_P2P_PCI_CHUNKSIZE=524288 \
    -x NCCL_P2P_NVL_CHUNKSIZE=1048576 \
    -x NCCL_DEBUG=INFO -x NCCL_DEBUG_SUBSYS=ENV \
    -x NCCL_GPUDIRECTTCPX_UNIX_CLIENT_PREFIX="${UNIX_CLIENT_PREFIX}" \
    -x NCCL_GPUDIRECTTCPX_PROGRAM_FLOW_STEERING_WAIT_MICROS=1000000 \
    -x NCCL_GPUDIRECTTCPX_FORCE_ACK \
    -x NCCL_GPUDIRECTTCPX_REPORT_NETWORK_LATENCY \
    -x NCCL_TESTS_SPLIT_MASK \
    /third_party/nccl-tests-mpi/build/"${benchmark}" \
      -b "${data_b}" -e "${data_e}" -f 2 -g 1 -w 5 --iters "${iters}" 2>&1 | \
    tee "a_${nhosts}_${gpu_per_node}_${socket_ifnames}_iter${i}.txt"
  done
}

run_nccl "$@"
