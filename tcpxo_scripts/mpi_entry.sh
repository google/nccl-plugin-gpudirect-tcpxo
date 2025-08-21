#!/bin/bash
# Copyright 2025 Google LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE.md file or at
# https://developers.google.com/open-source/licenses/bsd


# OpenMPI parameters.
: "${OMPI_COMM_WORLD_RANK:?Must set OMPI_COMM_WORLD_RANK}"
: "${OMPI_COMM_WORLD_LOCAL_SIZE:?Must set OMPI_COMM_WORLD_LOCAL_SIZE}"
: "${OMPI_COMM_WORLD_LOCAL_RANK:?Must set OMPI_COMM_WORLD_LOCAL_RANK}"

# Set the user limit for number of open files allowed per process.
ulimit -n 1048576

# The command argument doesn't support the string format.
# shellcheck disable=SC2086
/third_party/nccl-tests-mpi/build/"$@"

