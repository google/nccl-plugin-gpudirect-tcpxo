#!/bin/bash
# Copyright 2025 Google LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE.md file or at
# https://developers.google.com/open-source/licenses/bsd


set -x

# Sample command to run the workload for customer demo
#   $ BENCHMARK=all_gather_perf NHOSTS=2 NCCL_LIB_DIR=/var/lib/tcpxo/lib64 LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" /scripts/run-nccl-test-tcpxo-via-mpi.sh

# Specify the NCCL library directory mounted inside the workload container.
: "${NCCL_LIB_DIR:?Must set NCCL_LIB_DIR}"
# Specify the benchmark name e.g. `all_gather_perf`,`all_reduce_perf`
: "${BENCHMARK:?Must set BENCHMARK}"
# Specify the load directory path.
: "${LD_LIBRARY_PATH:?Must set LD_LIBRARY_PATH}"
# Specify the number of hosts.
: "${NHOSTS:?Must set NHOSTS}"


DATA_MIN="${DATA_MIN:-8}"
DATA_MAX="${DATA_MAX:-8G}"
GPU_PER_NODE="${GPU_PER_NODE:-8}"
RUN_ITERS="${RUN_ITERS:-20}"
WARMUP_ITERS="${WARMUP_ITERS:-5}"


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

run_nccl() {

  mpirun --mca btl tcp,self --mca btl_tcp_if_include eth0 --allow-run-as-root \
    -np $(( GPU_PER_NODE * "${NHOSTS}" )) \
    --hostfile "${SCRIPT_DIR}/hostfiles${NHOSTS}/hostfile${GPU_PER_NODE}" \
    -x NCCL_DEBUG_FILE="/tmp/${BENCHMARK}"-%h-%p.log \
    -x NCCL_TOPO_DUMP_FILE="/tmp/${BENCHMARK}"_topo.txt \
    -x NCCL_GRAPH_DUMP_FILE="/tmp/${BENCHMARK}"_graph.txt \
    -x LD_LIBRARY_PATH -x PATH \
    -x NCCL_DEBUG=INFO -x NCCL_DEBUG_SUBSYS=INIT,NET \
    -x NCCL_TESTS_SPLIT_MASK="${NCCL_TESTS_SPLIT_MASK:-0x0}" \
    -x NCCL_FASTRAK_LLCM_DEVICE_DIRECTORY="${NCCL_FASTRAK_LLCM_DEVICE_DIRECTORY}" \
    -x NCCL_LIB_DIR="${NCCL_LIB_DIR}" \
    taskset -c 32-63 /scripts/demo_mpi_entry_with_config_profile.sh "${BENCHMARK}" \
      -b "${DATA_MIN}" -e "${DATA_MAX}" -f 2 -g 1 -w "${WARMUP_ITERS}" --iters "${RUN_ITERS}" 2>&1 | \
    tee "/tmp/${BENCHMARK}_nh${NHOSTS}_ng${GPU_PER_NODE}_i${RUN_ITERS}.txt"
}

run_nccl "$@"
