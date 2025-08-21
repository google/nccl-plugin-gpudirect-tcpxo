#!/bin/bash
# Copyright 2025 Google LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE.md file or at
# https://developers.google.com/open-source/licenses/bsd


help() {
  echo "Usage: $ProgName <subcommand> [options]\n"
  echo "Subcommands:"
  echo "    install    Install NCCL plugin"
  echo "               --install-nccl installs NCCL main branch"
  echo "               --tune_net applies a tune network script"
  echo "               --no-install-nccl-shim-net do not install the guest config checker"
  echo "    shell      Run interactive shell"
  echo "    daemon     Start sshd and sleep"
  echo "    debug-info Install a script grabbing debug info that can invoked"
  echo "               afterwards by"
  echo "               sudo /var/lib/tcpxo/debug_info.sh"
}

install_subcommand() {

  mkdir -p /var/lib/tcpxo/lib64
  mkdir -p /var/lib/fastrak/lib64

  local -r ARGUMENT_LIST=(
    "install-nccl"
    "no-install-nccl-shim-net"
    "tune_net"
  )
  OPTS=$(getopt \
    --longoptions "$(printf "%s," "${ARGUMENT_LIST[@]}")" \
    --name "$(basename "$0")" \
    --options "" \
    -- "$@"
  )

  eval set -- "${OPTS}"

  local install_nccl_specified=false
  local skip_nccl_shim_net=false
  while (( $# )); do
    local flag="$1"; shift;
    case "${flag}" in
      --install-nccl)
        install_nccl_specified=true
      ;;
      --tune_net)
        chmod 755 /scripts/tune_net.sh
        sudo mount -o remount,exec /home
        /scripts/tune_net.sh
      ;;
      --no-install-nccl-shim-net)
        skip_nccl_shim_net=true
      ;;
    esac
  done
  if $install_nccl_specified; then
    install_nccl "${nccl_buildtype}"
  fi
  install_nccl_plugin "${nccl_plugin_buildtype}"
  install_nccl_env_profile
  install_nccl_tuner_plugin

  if ! $skip_nccl_shim_net; then
    install_nccl_shim
  fi
}

install_nccl_env_profile() {
 cp /scripts/nccl-env-profile*.sh /var/lib/fastrak/lib64/.
 ln -sf /var/lib/fastrak/lib64/nccl-env-profile.sh /var/lib/fastrak/lib64/nccl-env-profile-ll128.sh
 ln -sf /var/lib/fastrak/lib64/nccl-env-profile.sh /var/lib/fastrak/lib64/nccl-env-profile-nvlstree.sh
 cp /scripts/nccl-env-profile*.sh /var/lib/tcpxo/lib64/.
 ln -sf /var/lib/tcpxo/lib64/nccl-env-profile.sh /var/lib/tcpxo/lib64/nccl-env-profile-ll128.sh
 ln -sf /var/lib/tcpxo/lib64/nccl-env-profile.sh /var/lib/tcpxo/lib64/nccl-env-profile-nvlstree.sh
}

install_nccl() {
  cp -P /third_party/nccl-netsupport/build/lib/libnccl.so* /var/lib/tcpxo/lib64/
  cp -P /third_party/nccl-netsupport/build/lib/libnccl.so* /var/lib/fastrak/lib64/
}

install_nccl_plugin() {
  echo -n "Installing NCCL plugin."
  cp /plugins/lib* /var/lib/tcpxo/lib64/
  cp /plugins/lib* /var/lib/fastrak/lib64/
}

install_nccl_shim() {
  for loc in tcpxo fastrak; do
    mv /var/lib/${loc}/lib64/libnccl-net.so /var/lib/${loc}/lib64/libnccl-net_internal.so
    cp /plugins/libnccl-net-shim.so /var/lib/${loc}/lib64/libnccl-net.so
    cp /plugins/a3plus_guest_config*.textproto /var/lib/${loc}/lib64/
    ln -sf /var/lib/${loc}/lib64/a3plus_guest_config.textproto /var/lib/${loc}/lib64/a3plus_guest_config_ll128.textproto
    ln -sf /var/lib/${loc}/lib64/a3plus_guest_config.textproto /var/lib/${loc}/lib64/a3plus_guest_config_nvlstree.textproto
  done
  echo -n "Installed NCCL net plugin shim as libnccl-net.so; the original libnccl-net.so has been renamed to libnccl-net_internal.so"
}

install_nccl_tuner_plugin() {
  for loc in tcpxo fastrak; do
    cp -P /plugins/libnccl-tuner* /var/lib/${loc}/lib64/
    cp -P /plugins/a3plus_tuner_config*.textproto /var/lib/${loc}/lib64/
    ln -sf /var/lib/${loc}/lib64/a3plus_tuner_config.textproto /var/lib/${loc}/lib64/a3plus_tuner_config_ll128.textproto
    ln -sf /var/lib/${loc}/lib64/a3plus_tuner_config.textproto /var/lib/${loc}/lib64/a3plus_tuner_config_nvlstree.textproto
  done
}

gather_debug_info() {
  mkdir -p /var/lib/tcpxo/
  mkdir -p /var/lib/fastrak/
  cp /scripts/debug_info.sh /var/lib/tcpxo/
  cp /scripts/debug_info.sh /var/lib/fastrak/
}

SUBCOMMAND=$1
shift;

case ${SUBCOMMAND} in
  "install")
    echo "install"
    install_subcommand "$@"
    ;;
  "shell")
    echo "shell"
    service ssh restart
    /bin/bash
    ;;
  "daemon")
    echo "daemon"
    service ssh restart
    sleep inf
    ;;
  "debug-info")
    echo "debug-info"
    gather_debug_info
    ;;
  *)
    help
esac
