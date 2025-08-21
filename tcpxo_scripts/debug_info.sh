#!/bin/bash
# Copyright 2025 Google LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE.md file or at
# https://developers.google.com/open-source/licenses/bsd


date
hostname

#
# Setup
#

NVIDIA_SMI_BIN=""

case ${SCRIPT_ENV:-""} in
  "gke")
    NVIDIA_SMI_BIN="/home/kubernetes/bin/nvidia/bin/nvidia-smi"
    ;;
  *)
    NVIDIA_SMI_BIN="/var/lib/nvidia/bin/nvidia-smi"
    ;;
esac

#
# Versions
#

version_COS() {
  cat /etc/os-release
}

version_CUDA() {
  ${NVIDIA_SMI_BIN}
}

for dep in "COS" "CUDA"; do
  echo "${dep}: "; version_${dep}
done

#
# Versions
#

settings_mtu() {
  ip link list | egrep "eth[0-9]+"
}

settings_ip_route() {
  ip route show
}

settings_sys_kernel_core_pattern() {
  cat /proc/sys/kernel/core_pattern
}

for setting in "mtu"  "sys_kernel_core_pattern" "ip_route"; do
  echo "${setting}: "; settings_${setting} || echo
done

#
# Tunings
#

vm_net_tunings=(
  "/proc/sys/net/ipv4/tcp_mtu_probing"
  "/proc/sys/net/ipv4/tcp_slow_start_after_idle"
  "/proc/sys/net/ipv4/tcp_no_metrics_save"
  "/proc/sys/net/ipv4/tcp_rmem"
  "/proc/sys/net/ipv4/tcp_wmem"
  "/proc/sys/net/core/optmem_max"
  "/proc/sys/net/core/somaxconn"
  "/proc/sys/net/ipv4/tcp_max_syn_backlog"
)

for vnt in "${vm_net_tunings[@]}"; do
  echo -n "${vnt}: "; cat "${vnt}" || echo
done

#
# Longer Logs
#

log_dmesg() {
  sudo dmesg
}

log_ecc() {
  ${NVIDIA_SMI_BIN} --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv
  ${NVIDIA_SMI_BIN} --query-gpu=ecc.errors.uncorrected.aggregate.total --format=csv
}

for target in "dmesg"  "ecc"; do
  echo "${target}: "; log_${target}
done

