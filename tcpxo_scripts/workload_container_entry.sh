#!/bin/bash
# Copyright 2025 Google LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE.md file or at
# https://developers.google.com/open-source/licenses/bsd


set -u

: "${OUTPUT_DIR:=/tmp/log}"
: "${GCS_BUCKET:=fastrak_bct_tests}"
: "${GCS_MOUNT_POINT:=/tmp/log}"
: "${VM_NAME:=test_vm}"

mkdir -p ${GCS_MOUNT_POINT}
gcsfuse --implicit-dirs ${GCS_BUCKET} ${GCS_MOUNT_POINT}
mkdir -p ${OUTPUT_DIR}/${VM_NAME}

service ssh restart
sleep inf
