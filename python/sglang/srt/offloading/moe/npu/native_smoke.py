"""Explicit target-NPU smoke for the ACL shim; never runs during package import.

Run with --help. CPU fake-SDK tests cannot replace this hardware check. This
checks eager copy/event ordering and bytes only, not the full MoE decode graph.
"""

import argparse
import ctypes
import json

from ...transfer import HostCopyPlan, HostCopyRegion
from ..storage import HostMemoryCapability
from .native_acl import get_native_acl, make_acl_host_ops
from .transfer import DeviceMemoryRegion


def run_smoke(
    build_directory, *, device_index=0, nbytes=4093, repeats=4, queued_ops=32
):
    import torch
    import torch_npu

    if nbytes < 3 or repeats <= 0 or queued_ops < 0:
        raise ValueError("Require bytes >= 3, repeats > 0 and queued_ops >= 0")
    if not torch.npu.is_available():
        raise RuntimeError("This smoke requires a real NPU and matching CANN/torch_npu")
    torch.npu.set_device(device_index)
    device = torch.device("npu", device_index)
    native = get_native_acl(build_directory)
    addresses, buffers, retained = [], [], []
    streams = [torch.npu.Stream(device=device) for _ in range(3)]
    copy_streams, producer = streams[:2], streams[2]
    completed = 0
    try:
        for _ in range(2):
            address = native.malloc_host(nbytes, device_index)
            addresses.append(address)
            backing = (ctypes.c_uint8 * nbytes).from_address(address)
            buffers.append((backing, torch.frombuffer(backing, dtype=torch.uint8)))
        source, readback = buffers[0][1], buffers[1][1]
        host = HostMemoryCapability(addresses[0], nbytes, True, None, buffers[0])
        target = torch.empty(nbytes, dtype=torch.uint8, device=device)
        destination = DeviceMemoryRegion.from_tensor(target)
        boundaries = (0, nbytes // 3, 2 * nbytes // 3, nbytes)
        plan = HostCopyPlan(
            tuple(
                HostCopyRegion(
                    host, destination, start, start, end - start, nbytes, nbytes
                )
                for start, end in zip(boundaries, boundaries[1:])
            )
        )
        ops = make_acl_host_ops(
            module=native,
            retain=retained.append,
            source_location="host",
            max_batch_entries=2,
        )
        for iteration in range(repeats):
            for copy_stream in copy_streams:
                pattern = (
                    (torch.arange(nbytes, dtype=torch.int64) + completed) % 251
                ).to(torch.uint8)
                source.copy_(pattern)  # CPU initialization of the sole host source.
                with torch.npu.stream(producer):
                    target.fill_(17)
                    # Leave producer work in flight. The copy stream must honor
                    # its queued wait before native ACL can overwrite the target.
                    for _ in range(queued_ops):
                        target.add_(1)
                    prior = torch.npu.Event()
                    prior.record(producer)
                copy_stream.wait_event(prior)
                ticket = ops.materialize_into(plan, destination, copy_stream)
                ticket.completion.synchronize()  # Explicit diagnostic boundary.
                native.memcpy_sync(
                    addresses[1],
                    nbytes,
                    target.data_ptr(),
                    nbytes,
                    "device_to_host",
                    device_index,
                )
                if not torch.equal(readback, pattern):
                    raise AssertionError(
                        f"ACL byte/queue-order mismatch at round {completed}"
                    )
                completed += 1
        return {
            "status": "passed",
            "copies": completed,
            "bytes_per_copy": nbytes,
            "copy_streams": 2,
            "queued_producer_ops": queued_ops,
            "batch_entries": 2,
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "device": str(device),
            "native": native.api_info(),
            "full_decode_graph_tested": False,
        }
    finally:
        # On a failed drain deliberately keep allocations live. Freeing a host
        # allocation while DMA may still read it would invalidate the smoke.
        for stream in streams:
            stream.synchronize()
        for address in reversed(addresses):
            native.free_host(address, device_index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-directory", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--bytes", type=int, default=4093)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--queued-ops", type=int, default=32)
    args = parser.parse_args()
    print(
        json.dumps(
            run_smoke(
                args.build_directory,
                device_index=args.device,
                nbytes=args.bytes,
                repeats=args.repeats,
                queued_ops=args.queued_ops,
            ),
            indent=2,
        )
    )
