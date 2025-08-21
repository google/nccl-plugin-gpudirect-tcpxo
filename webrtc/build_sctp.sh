#!/bin/bash
# Copyright 2025 Google LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE.md file or at
# https://developers.google.com/open-source/licenses/bsd

# Webrtc commit to use for building sctp.
WEBRTC_COMMIT=${2:-"7f614d2a7569e6e01ae4d752967657bc19556c8e"}

# --- begin runfiles.bash initialization v3 ---
# Copy-pasted from the Bazel Bash runfiles library v3.
set -uo pipefail; set +e; f=bazel_tools/tools/bash/runfiles/runfiles.bash
# shellcheck disable=SC1090
source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null || \
source "$(grep -sm1 "^$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f2- -d' ')" 2>/dev/null || \
source "$0.runfiles/$f" 2>/dev/null || \
source "$(grep -sm1 "^$f " "$0.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
source "$(grep -sm1 "^$f " "$0.exe.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
{ echo>&2 "ERROR: cannot find $f"; exit 1; }; f=; set -e
# --- end runfiles.bash initialization v3 ---

dir=${1}

pushd ${dir}
rm -f libwebrtc.a libwebrtc_debug.a
rm -rf include
popd

script_dir=$(dirname $(rlocation webrtc_build/build.sh))
pushd ${script_dir}
./build.sh -r ${WEBRTC_COMMIT}
cp -r out/webrtc*/include ${dir}/
cp out/webrtc*/lib/Release/libwebrtc_full.a ${dir}/libwebrtc.a
cp out/webrtc*/lib/Debug/libwebrtc_full.a ${dir}/libwebrtc_debug.a
rm -rf out
popd
