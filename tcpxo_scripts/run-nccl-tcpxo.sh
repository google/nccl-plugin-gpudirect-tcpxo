#!/bin/bash
# Copyright 2025 Google LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE.md file or at
# https://developers.google.com/open-source/licenses/bsd


set -x

# Sample command to run the workload:
#  /scripts/run-nccl-tcpxo.sh ${collective_name}_perf "${LD_LIBRARY_PATH}" ${gpu_per_node} ${if_name_list} ${msg_min} ${msg_max} ${channel_per_gpu} ${num_node} ${num_iteration} ${num_flows} ${flows_per_group} ${warmup_iter}
#  e.g. /scripts/run-nccl-tcpxo.sh all_gather_perf "${LD_LIBRARY_PATH}" 8 eth1,eth2,eth3,eth4,eth5,eth6,eth7,eth8 1M 512M 3 2 10 8 2 10

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

run_nccl() {
  local -r benchmark=$1
  local -r ld_library_path_override=$2
  local -r gpu_per_node=$3
  local -r socket_ifnames=$4
  local -r data_b=$5
  local -r data_e=$6
  local -r channels_per_gpu=$7
  local -r nhosts=${8}
  local -r iter=${9}
  local -r num_flows=${10:-8}
  local -r flows_per_group=${11:-2}
  local -r warmup_iter=${12:-5}
  local -r num_channel=$((gpu_per_node*channels_per_gpu))

  local nccl_algo=${NCCL_ALGO:-Ring}

  for i in $(seq 1 1); do

  LD_LIBRARY_PATH=${ld_library_path_override} \
  mpirun --mca btl tcp,self --mca btl_tcp_if_include eth0 --allow-run-as-root \
    -np $(( gpu_per_node * "${nhosts}" )) \
    --hostfile "${SCRIPT_DIR}/hostfiles${nhosts}/hostfile${gpu_per_node}" \
    -x NCCL_FASTRAK_CTRL_DEV=eth0 \
    -x NCCL_FASTRAK_IFNAME=${socket_ifnames} \
    -x NCCL_DEBUG_FILE="${OUTPUT_DIR}/%h/${benchmark}"-%p.log \
    -x NCCL_TOPO_DUMP_FILE="${OUTPUT_DIR}/${VM_NAME}/${benchmark}"_topo.txt \
    -x NCCL_GRAPH_DUMP_FILE="${OUTPUT_DIR}/${VM_NAME}/${benchmark}"_graph.txt \
    -x NCCL_SOCKET_IFNAME=eth0 \
    -x LD_LIBRARY_PATH -x PATH \
    -x NCCL_CROSS_NIC=0 \
    -x NCCL_ALGO=${nccl_algo} \
    -x NCCL_PROTO=${NCCL_PROTO:-Simple} \
    -x NCCL_TUNER_PLUGIN="${NCCL_TUNER_PLUGIN:-UNUSED}" \
    -x NCCL_TUNER_CONFIG_PATH="${NCCL_TUNER_CONFIG_PATH:-}" \
    -x NCCL_DYNAMIC_CHUNK_SIZE=524288 \
    -x NCCL_DYNAMIC_CHUNK_SIZE=524288 -x NCCL_P2P_NET_CHUNKSIZE=524288 \
    -x NCCL_P2P_PCI_CHUNKSIZE=524288 -x NCCL_P2P_NVL_CHUNKSIZE=1048576 \
    -x NCCL_FASTRAK_NUM_FLOWS=${num_flows} -x NCCL_FASTRAK_FLOWS_PER_GROUP=${flows_per_group} \
    -x NCCL_BUFFSIZE=8388608 \
    -x CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -x NCCL_NET_GDR_LEVEL=PIX \
    -x NCCL_DEBUG="${NCCL_DEBUG:-INFO}" \
    -x NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}" \
    -x NCCL_FASTRAK_ENABLE_HOTPATH_LOGGING=0 \
    -x NCCL_SHIMNET_SHIM_LAYERS=UNUSED \
    -x NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}" \
    -x NCCL_TESTS_SPLIT_MASK="${NCCL_TESTS_SPLIT_MASK:-0x0}" \
    -x NCCL_NET_PLUGIN_TELEMETRY_MODE="${NCCL_NET_PLUGIN_TELEMETRY_MODE:-0}" \
    -x NCCL_GPUVIZ_READ_INTERVAL_IN_MICROSECONDS="${NCCL_GPUVIZ_READ_INTERVAL_IN_MICROSECONDS:-1000000}" \
    -x NCCL_GPUVIZ_GET_MAX_BUCKETS_LATENCY_HISTOGRAM_IN_NANOSECONDS="${NCCL_GPUVIZ_GET_MAX_BUCKETS_LATENCY_HISTOGRAM_IN_NANOSECONDS:-10000000}" \
    -x NCCL_GPUVIZ_GET_SCALE_LATENCY_HISTOGRAM_IN_NANOSECONDS="${NCCL_GPUVIZ_GET_SCALE_LATENCY_HISTOGRAM_IN_NANOSECONDS:-1}" \
    -x NCCL_GPUVIZ_GET_MAX_BUCKETS_SIZE_HISTOGRAM_IN_BYTES="${NCCL_GPUVIZ_GET_MAX_BUCKETS_SIZE_HISTOGRAM_IN_BYTES:-10000000}" \
    -x NCCL_GPUVIZ_GET_SCALE_SIZE_HISTOGRAM_IN_BYTES="${NCCL_GPUVIZ_GET_SCALE_SIZE_HISTOGRAM_IN_BYTES:-1}" \
    -x NCCL_GPUVIZ_BANDWIDTH_HISTOGRAM_COLLECTION_ENABLE="${NCCL_GPUVIZ_BANDWIDTH_HISTOGRAM_COLLECTION_ENABLE:-1}" \
    -x NCCL_GPUVIZ_FILE_ROTATION_INTERVAL_IN_SECONDS="${NCCL_GPUVIZ_FILE_ROTATION_INTERVAL_IN_SECONDS:-10}" \
    taskset -c 32-63 /scripts/mpi_entry.sh "${benchmark}" \
      -b "${data_b}" -e "${data_e}" -f 2 -g 1 -w "${warmup_iter}" --iters "${iter}" 2>&1 | \
    tee "${OUTPUT_DIR}/${VM_NAME}/${benchmark}_${nhosts}_${gpu_per_node}_${socket_ifnames}_w${warmup_iter}_i${iter}.txt"
  done
}



run_nccl "$@"

