#!/bin/bash
# Copyright 2025 Google LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE.md file or at
# https://developers.google.com/open-source/licenses/bsd


#  /scripts/check_log.sh 2 '0xa0e0000d\|Binding to interface:' /tmp/log/

set -x

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

nhosts=$1
query="${2:-WARN}"
log_dir="${3:-/tmp/log/}"

mpirun --mca btl tcp,self --mca btl_tcp_if_include eth0 --allow-run-as-root \
   -np "${nhosts}" \
   --hostfile "${SCRIPT_DIR}/hostfiles${nhosts}/hostfile1" grep -nr "${query}" "${log_dir}"
