# TCPXO Prober

## Overview

The TCPXO Prober is a tool designed to run on Google Cloud VMs equipped with
NVIDIA H100 GPUs using the GPUDirect-TCPXO offloaded networking stack. Its
primary purpose is to measure the network availability and performance of the
Network Interface Cards (NICs) used for high-performance GPU-to-GPU
communication.

This prober runs as an agent within each guest VM and can be orchestrated by an
external controller system. It sends active probe traffic between GPU NICs
across different VMs to provide insights into network latency and connectivity.

## Building the prober

### Build environment

Please follow the setup outlined in the `README.md` file at the root directory.

### Build script

Please use the `prober_build.sh` found at the root directory.

## Running the prober with load_test.py

We have included a `load_test.py` program to demonstrate the functionality of
the TCPXO Prober and how to interact with it.

This Python program performs the following steps:

1.  Sends a StartPings() RPC to each node involved in the test.
2.  Waits a configurable amount of time.
3.  Sends a StopPings() RPC to each node involved in the test.
4.  Collects the results.
5.  Repeats the process, but calls StartPings() with more targets.

### Start the prober agent on each VM

```
export TCPXO_PROBER_GPU_NIC_IPS=xxx,xxx,xxx,xxx,xxx,xxx,xxx,xxx
export TCPXO_PROBER_SERVER_PORT=8080
./prober
```

The `TCPXO_PROBER_GPU_NIC_IPS` are the GPU NIC IPs on the local VM.

Running `ifconfig` or `ip addr show` will list all 9 NIC IPs on an A3Mega VM.
Typically, the first IP address belongs to the primary NIC while the remaining 8
are dedicated to the GPU NICs.

### Start `load_test.py` on one of the VMs

```
pip3 install grpcio-tools

python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. src/prober.proto

python3 load_test.py --nic_ips="$ALL_GPU_NIC_IPS"
```

`$ALL_GPU_NIC_IPS` is a list of all primary NICs and GPU NICs from all the VMs
to be involved in the probing. The list must be ordered per VM: list the primary
NIC first, followed by its 8 GPU NICs in rail order.

### Collect results

As `load_test.py` progresses, probing results for each agent are written to a
CSV file in the `/tmp` folder, prefixed with `tcpxo_probe_results_`.

Each agent reports results for connections initiated by the `StartPings` RPC.

```
<probe_start_timestamp>,<local_nic_ip>,<remote_nic_ip>,<probe_result>
```

If a probe is successful, `<probe_result>` will be the round-trip time (RTT) in
nanoseconds. If a probe is unsuccessful, `<probe_result>` will be a string
detailing the error message.

At the end of the run, `load_test.py` creates a summary of probing statuses.
