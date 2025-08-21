FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND='noninteractive'

RUN apt update \
  && apt-get install -y --no-install-recommends \
        git openssh-server wget iproute2 vim build-essential cmake gdb net-tools iptables \
        protobuf-compiler libprotobuf-dev libprotoc-dev rsync libssl-dev \
        pkg-config libmnl-dev python3 xserver-xorg-core less tcpdump \
  && rm -rf /var/lib/apt/lists/*

# build ethtool
WORKDIR /third_party
RUN wget https://mirrors.edge.kernel.org/pub/software/network/ethtool/ethtool-6.3.tar.gz
RUN tar -xvf ethtool-6.3.tar.gz
WORKDIR ethtool-6.3
RUN ./configure && make && make install

# copy the rxdm.
WORKDIR /fts
COPY ./out/fastrak_gpumem_manager /fts
COPY scripts/entrypoint_rxdm_container.sh /fts
COPY scripts/kernel_tuning.sh /fts
COPY scripts/cleanup_tuning.sh /fts
COPY scripts/tuning_helper.sh /fts

ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda-12:/usr/local/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

ENTRYPOINT ["bash", "/fts/entrypoint_rxdm_container.sh"]
