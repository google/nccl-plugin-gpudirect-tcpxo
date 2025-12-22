"""load_test sets up a probe job and records the RTT results."""

import csv
import glob
import os
import statistics
import time
from typing import Dict, Sequence
from absl import app
from absl import flags
from absl import logging
from google.rpc import status_pb2
import grpc
import prober_pb2
import prober_pb2_grpc


_NICS = flags.DEFINE_list(
    "nic_ips",
    "",
    "List of GPU NICs that should be involved in the probing. Must be 9 per"
    " Node. The 1st must be the primary NIC and then the 8 GPU NICs in rail"
    " order.",
)

_ITERATIONS = flags.DEFINE_integer(
    "iterations",
    5,
    "Number of iterations to run the load test.",
)

_ITERATION_LENGTH_IN_SECONDS = flags.DEFINE_integer(
    "iteration_length_in_seconds",
    60,
    "Number of seconds to run each iteration.",
)

_ITERATION_INCREASE_CONNECTIONS_PER_NIC = flags.DEFINE_integer(
    "iteration_increase_connections_per_nic",
    8,
    "How many connections to increase for each GPU NIC per iteration.",
)

_CONNECTIONS_PER_NIC_AT_START = flags.DEFINE_integer(
    "connections_per_nic_at_start",
    1,
    "Number of connections to set up per GPU NIC at start of the test.",
)

_VERIFY_PAYLOAD = flags.DEFINE_bool(
    "verify_payload",
    False,
    "Whether to verify the payload of the probe.",
)

_PROBE_QPS_PER_CONNECTION = flags.DEFINE_integer(
    "probe_qps_per_connection",
    1,
    "How many probes to send per connection per second.",
)


# GetTargets returns a list of prober_pb2.Target objects, each representing a
# pair of NICs involved in the probing.
def GetTargets(
    src_primary_nic: str,
    src_gpu_nics: Sequence[str],
    all_nics: Dict[str, Sequence[str]],
    num_nodes_to_target: int,
) -> Sequence[prober_pb2.Target]:
  num_nodes = 0
  targets = []
  while True:
    for primary_nic, gpu_nics in all_nics.items():
      if num_nodes >= num_nodes_to_target:
        return targets

      if primary_nic == src_primary_nic:
        continue

      num_nodes += 1
      for n1, n2 in zip(src_gpu_nics, gpu_nics):
        targets.append(
            prober_pb2.Target(
                local_nic_ip_address=n1,
                peer_nic_ip_address=n2,
            )
        )


def GetRPCStatus(e: grpc.RpcError) -> status_pb2.Status:
  serialized_status = None
  for key, value in e.trailing_metadata():
    if key == "grpc-status-details-bin":
      serialized_status = value
      break
  if not serialized_status:
    return None
  rpc_status = status_pb2.Status()
  rpc_status.ParseFromString(serialized_status)
  return rpc_status


# StartPings sends a StartPings RPC request to the primary_nic_ip and that
# contains 8 * num_nodes_to_probe targets.
def StartPings(
    primary_nic_ip: str,
    port: int,
    gpu_nics: Sequence[str],
    all_nics: Dict[str, Sequence[str]],
    num_nodes_to_probe: int,
    qps: int,
    verify_payload: bool,
):
  request = prober_pb2.StartPingsRequest(
      probe_rate_qps=qps,
      verify_payload=verify_payload,
      targets=GetTargets(
          primary_nic_ip, gpu_nics, all_nics, num_nodes_to_probe
      ),
  )
  address = primary_nic_ip + ":" + str(port)
  with grpc.insecure_channel(address) as channel:
    stub = prober_pb2_grpc.AgentServiceStub(channel)
    try:
      stub.StartPings(request)
    except grpc.RpcError as e:
      logging.error("StartPings failed: %s", GetRPCStatus(e))


# StopPings sends a StopPings RPC request to each of the a3mega prober agents
# involved in the test.
def StopPings(ip: str, port: int):
  request = prober_pb2.StopPingsRequest()
  address = ip + ":" + str(port)
  with grpc.insecure_channel(address) as channel:
    stub = prober_pb2_grpc.AgentServiceStub(channel)
    try:
      stub.StopPings(request)
    except grpc.RpcError as e:
      logging.error("StopPings failed: %s", e)


def GetResultsFile() -> str:
  results_files = glob.glob("/tmp/tcpxo_probe_results_0_*.csv")
  if len(results_files) != 1:
    logging.fatal("Expected 1 results file, got %s", len(results_files))
  return results_files[0]


# GetRTTResults returns the average and median RTT results from the local
# probe results file.
def GetRTTResults():
  # Wait for the results file to be written.
  time.sleep(1)
  num_success_ops = 0
  num_fail_ops = 0
  rtts = []
  file_name = GetResultsFile()
  with open(file_name, "r") as file:
    reader = csv.reader(file)
    for row in reader:
      if not row:
        continue
      rtt = row[-1]
      if str.isdigit(rtt):
        num_success_ops += 1
        rtts.append(int(rtt))
      else:
        num_fail_ops += 1

  os.remove(file_name)
  if num_success_ops == 0:
    logging.fatal("No successful ops found")

  avg_rtt = sum(rtts) // len(rtts)
  median_rtt = statistics.median(rtts)

  logging.info("Average RTT (ns): %s", avg_rtt)
  logging.info("Median RTT (ns): %s", median_rtt)
  logging.info("Num success ops: %s", num_success_ops)
  logging.info("Num failed ops: %s", num_fail_ops)
  if num_success_ops == 0:
    raise ValueError("No results found")


def main(argv: Sequence[str]):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  try:
    port = os.environ["TCPXO_PROBER_SERVER_PORT"]
  except KeyError:
    port = 8080

  try:
    os.environ["TCPXO_PROBER_OUTPUT_INTERVAL_IN_SECONDS"]
  except KeyError:
    logging.warning(
        "To get the most accurate results set `export"
        " TCPXO_PROBER_OUTPUT_INTERVAL_IN_SECONDS=1` and restart the prober"
    )

  conn_per_nic_at_start = _CONNECTIONS_PER_NIC_AT_START.value
  conn_increase_per_iter = _ITERATION_INCREASE_CONNECTIONS_PER_NIC.value
  verify_payload = _VERIFY_PAYLOAD.value
  qps = _PROBE_QPS_PER_CONNECTION.value

  if len(_NICS.value) % 9 != 0:
    raise app.UsageError(
        "Number of NICs must be a multiple of 9 (1 primary_nic and 8 GPU NICs)."
    )

  # Create a dictionary of primary NICs and their corresponding GPU NICs.
  all_nics = dict()
  for i in range(0, len(_NICS.value), 9):
    primary_nic = _NICS.value[i]
    gpu_nics = _NICS.value[i + 1 : i + 9]
    all_nics[primary_nic] = gpu_nics
  logging.info("Nics: %s", all_nics)

  # Run the load test.
  for i in range(_ITERATIONS.value):
    connections_per_nic = conn_per_nic_at_start + (i * conn_increase_per_iter)
    logging.info(
        "Starting iteration=%s, connections_per_nic=%s",
        i,
        connections_per_nic,
    )
    for primary_nic, gpu_nics in all_nics.items():
      StartPings(
          primary_nic,
          port,
          gpu_nics,
          all_nics,
          connections_per_nic,
          qps,
          verify_payload,
      )

    time.sleep(_ITERATION_LENGTH_IN_SECONDS.value)

    for primary_nic in all_nics:
      StopPings(primary_nic, port)

    GetRTTResults()


if __name__ == "__main__":
  app.run(main)
