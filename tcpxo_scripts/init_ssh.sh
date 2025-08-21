#!/bin/bash
# Copyright 2025 Google LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE.md file or at
# https://developers.google.com/open-source/licenses/bsd


PORT=${PORT:-222}

while true; do
  host=$1
  if [[ -z $host ]]; then
    break
  fi
  ssh -o StrictHostKeyChecking=no -p "${PORT}" "$host" \
    echo "Hello from ${host}"
  shift
done
