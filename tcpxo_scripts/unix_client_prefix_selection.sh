#!/bin/bash
# Copyright 2025 Google LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE.md file or at
# https://developers.google.com/open-source/licenses/bsd


# preference order
PREFERENCE_ORDER=(
  "/run/tcpx"
  "/tmp"
)

UNIX_CLIENT_PREFIX="/tmp" # backwards compatibility default

for possible_dir in "${PREFERENCE_ORDER[@]}"; do
  if [[ -e "${possible_dir}/rx_rule_manager" ]]; then
    UNIX_CLIENT_PREFIX="${possible_dir}"
  fi
done

echo UNIX_CLIENT_PREFIX "${UNIX_CLIENT_PREFIX}"
