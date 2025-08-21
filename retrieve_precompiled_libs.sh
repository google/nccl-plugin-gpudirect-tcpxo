#!/bin/bash
# Copyright 2025 Google LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE.md file or at
# https://developers.google.com/open-source/licenses/bsd


set -e

DEFAULT_IMAGE="us-docker.pkg.dev/gce-ai-infra/gpudirect-tcpxo/nccl-plugin-gpudirect-tcpxo-precompiled-libs:latest"
DEFAULT_OUTPUT_DIR="out/"

IMAGE="${DEFAULT_IMAGE}"
OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"
CONTAINER_PATH="/plugins"

usage() {
  cat <<-EOF
Usage: $0 [-i IMAGE_NAME] [-o OUTPUT_DIRECTORY]

Retrieves the precompiled libraries from the TCPXO plugin Docker image.

OPTIONS:
  -i    The name of the Docker image to use.
        (Default: $DEFAULT_IMAGE)
  -o    The directory where files will be copied.
        (Default: $DEFAULT_OUTPUT_DIR)
  -h    Show this help message.
EOF
}

while getopts ":i:o:h" opt; do
  case ${opt} in
    i )
      IMAGE="${OPTARG}"
      ;;
    o )
      OUTPUT_DIR="${OPTARG}"
      ;;
    h )
      usage
      exit 0
      ;;
    \? )
      echo "Invalid option: -${OPTARG}" 1>&2
      usage
      ;;
    : )
      echo "Invalid option: -${OPTARG} requires an argument" 1>&2
      usage
      ;;
  esac
done
shift $((OPTIND -1))

if ! command -v docker &> /dev/null
then
    echo "Docker could not be found, please install it."
    exit 1
fi

if [[ ! -d "${OUTPUT_DIR}" ]]; then
  echo "Creating output directory: ${OUTPUT_DIR}"
  mkdir -p "${OUTPUT_DIR}"
fi

echo "Docker image: ${IMAGE}"
echo "Output directory: ${OUTPUT_DIR}"

CID=$(docker create "${IMAGE}")
# Ensure the container is removed on exit, whether successful or not
trap 'docker rm -f "${CID}" > /dev/null' EXIT

if ! docker cp "${CID}:${CONTAINER_PATH}/." "${OUTPUT_DIR}"; then
  echo "Failed to copy files." >&2
  exit 1
fi
