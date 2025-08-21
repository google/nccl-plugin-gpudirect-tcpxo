# Copyright 2025 Google LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE.md file or at
# https://developers.google.com/open-source/licenses/bsd

# Set the recommended environment variables for tcpo stack.
# User must set the ${NCCL_LIB_DIR} used
# inside the workload container.
# Sample usage before running workload:
# NCCL_LIB_DIR="/usr/local/tcpxo/lib64" source scripts/nccl-env-profile.sh

: "${NCCL_LIB_DIR:?Must set NCCL_LIB_DIR}"

# Default values
host_nic="eth0"
gpu_nics="eth1,eth2,eth3,eth4,eth5,eth6,eth7,eth8"
find_nics() {
    local nics_discovered=0
    local host_nic_discovered=""
    local gpu_nics_discovered=""
    for netdev in $(ls /sys/class/net | awk '{print $1}'); do
        pci_bdf=$(cat "/sys/class/net/${netdev}/device/uevent" 2>/dev/null | grep "PCI_SLOT_NAME" | awk -F '=' '{print $2}' || echo "")
        if [[ -z "${pci_bdf}" ]]; then
            continue
        fi
        if [[ "${pci_bdf}" =~ 00\:[0-9a-f]{2}\.0 ]]; then
            host_nic_discovered="${netdev}"
        elif [[ "${pci_bdf}" =~ [0|8][6-7d-e]\:00\.0 ]]; then
            if [[ -z "${gpu_nics_discovered}" ]]; then
                gpu_nics_discovered="${netdev}"
            else
                gpu_nics_discovered="${gpu_nics_discovered},${netdev}"
            fi
        fi
    done
    if [[ -n "${host_nic_discovered}" ]]; then
        host_nic="${host_nic_discovered}"
    fi
    if [[ -n "${gpu_nics_discovered}" ]]; then
        gpu_nics="${gpu_nics_discovered}"
    fi
}

find_nics
export LD_LIBRARY_PATH="${NCCL_LIB_DIR}:${LD_LIBRARY_PATH}"
export NCCL_FASTRAK_IFNAME="${gpu_nics}"
export NCCL_FASTRAK_CTRL_DEV="${host_nic}"
export NCCL_SOCKET_IFNAME="${host_nic}"
export NCCL_CROSS_NIC=0
export NCCL_PROTO=Simple,LL128
export NCCL_MIN_NCHANNELS=4
export NCCL_NVLSTREE_MAX_CHUNKSIZE=131072
export NCCL_P2P_NET_CHUNKSIZE=524288
export NCCL_P2P_PCI_CHUNKSIZE=524288
export NCCL_P2P_NVL_CHUNKSIZE=1048576
export NCCL_FASTRAK_NUM_FLOWS=2
export NCCL_FASTRAK_ENABLE_CONTROL_CHANNEL=0
export NCCL_BUFFSIZE=8388608
export NCCL_FASTRAK_USE_SNAP=1
export NCCL_FASTRAK_USE_LLCM=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_NET_GDR_LEVEL=PIX
export NCCL_FASTRAK_ENABLE_HOTPATH_LOGGING=0
export NCCL_TUNER_PLUGIN=libnccl-tuner.so
export NCCL_TUNER_CONFIG_PATH=${NCCL_LIB_DIR}/a3plus_tuner_config.textproto
export NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE=${NCCL_LIB_DIR}/a3plus_guest_config.textproto
export NCCL_FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS=600000
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=INIT,NET,ENV,COLL,GRAPH
