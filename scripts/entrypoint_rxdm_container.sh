#!/bin/bash
# Copyright 2025 Google LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE.md file or at
# https://developers.google.com/open-source/licenses/bsd

set -x

func_trap() {
  # Forward the received signal
  kill -"$1" "$pid"
}

trap_with_arg() {
    func="$1" ; shift
    for sig ; do
        trap "$func $sig" "$sig"
    done
}

chmod 755 /fts/kernel_tuning.sh
chmod 755 /fts/cleanup_tuning.sh
chmod 755 /fts/tuning_helper.sh
/fts/kernel_tuning.sh
chmod +x /fts/fastrak_gpumem_manager
/fts/fastrak_gpumem_manager "$@" &
pid=$!
trap_with_arg func_trap INT TERM

# Wait until a signal is triggered to stop the manager
wait $pid

# Because the manager uses custom signal handler, the process still runs its
# cleanup logic after the `wait` come back. So we have a polling loop to check
# if the process finish.
while kill -0 $pid 2>/dev/null; do
  sleep 1
done

/fts/cleanup_tuning.sh
