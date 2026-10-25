# Native V4 expert storage and bounded loading

`NpuV4FP4Format` uses the existing resident path's
`prepare_w4a8_mxfp_weight`. It does not implement an alternative quantizer or
dequantize a layer. The native GMM's `weight_dtype=float4_e2m1fn_x2` determines
numeric interpretation; the tensor carrier dtype does not establish FP8 numeric
storage. Native shape, stride, carrier dtype and physical byte extent are probed
on the installed backend and checked again for every conversion and bank.

## Ownership and budgets

`MemFabricHostPool` owns one LOCAL MemFabric pool and one page-aligned slab.
`physical_bytes` must be an explicit whole GiB, matching MemFabric's physical
reservation granularity. The allocator reports this entire reservation once,
including unused reservation space. The slab size is rounded to the host page
size and must fit that budget. Individual component tensors are zero-copy views
of this slab. There is no parallel pinned copy of the complete expert master.

MemFabric's `get_dva` must certify a linear mapping over the slab; ACL's real
`aclrtHostRegisterV2(..., ACL_HOST_REG_PINNED)` must succeed before asynchronous
host-read capability is published. The existing MemFabric mapping is reused;
ACL is not asked to make a second device mapping. The production backend submits ACL batches with `source_location="host"`;
registration independently certifies asynchronous access. The location attribute
and registration compatibility still require qualification on the target SDK.

The reference MemFabric `initialize` silently returns success if somebody else
already initialized its process singleton. This owner therefore requires
exclusive ownership of that LOCAL offload pool. Concurrent MoE owners and the
repository's MemFabric HiCache backend are rejected. An unrelated external
caller that initializes the same singleton is unsupported; the reference API
provides no query that would certify its configuration or ownership.

The Linux-version guard helper remains in `storage.py`, but its call is currently
commented out; startup does not reject older kernels through that guard. Deployment
must qualify its kernel/vendor support for external registration. See the recorded [aclrtHostRegisterV2 contract](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_2128.html).

Runtime teardown must drain host submitters, streams and captured-graph leases,
then call each `ExpertLayerStorage.close()` before `MemFabricHostPool.close()`.
Layer close retires module-visible parameter views. Pool close unregisters ACL,
frees the slab, then uninitializes MemFabric. Unregister failure preserves the
allocation and owner for a safe retry. No GC destructor guesses reader lifetime.

## Checkpoint and fixed bank layout

FusedMoE weight creation invokes `NpuMoEBackend.prepare_expert_weights` before
the first expert allocation. The selected format supplies `CheckpointComponent`
recipes to the common probe; the resulting `BundleComponent` is the single
authority for checkpoint and runtime shapes, strides and byte extents. The four registered CPU parameters retain the existing
checkpoint names and packed byte shapes:

| Component | Checkpoint shape per expert | Runtime adaptation |
| --- | --- | --- |
| `w13_weight` | `[2I, H/2]` uint8 | Native FP4 NZ conversion, then transpose |
| `w2_weight` | `[H, I/2]` uint8 | Native FP4 NZ conversion, then transpose |
| `w13_weight_scale_inv` | `[2I, H/32]` E8M0 bytes | `[H/64, 2I, 2]` view |
| `w2_weight_scale_inv` | `[H, I/32]` E8M0 bytes | `[I/64, H, 2]` view |

`H` and padded `I` must be multiples of 64. Expert dimension stride in the host
master is the probed physical native component size, so the same allocation can
first hold checkpoint bytes and subsequently the complete padded native image.
All storage starts at zero; gate/up/down intermediate padding stays zero.
The loader accepts CPU uint8 weights and byte views of E8M0 scales, validates
contiguous EP ownership, rejects duplicate/inconsistent pieces, and requires
all six weight/scale shards per local expert. Its checks and writes are locked
because V4 can dispatch checkpoint loads through a thread pool.

The generic `device_loading_context` bypasses whole-module staging for this
owner. Finalization sends one expert's weight/scale pair to the NPU, invokes
the existing native conversion, records and synchronizes its stream event,
then copies physical bytes back into that expert's original host region using
`aclrtMemcpy`. Ordinary `.cpu()` is not used for NZ serialization because it
may convert layout. Failed partial repacks poison the writable master; retrying
would otherwise mistake already converted bytes for a checkpoint tensor.
Only complete finalization seals `ExpertLayerStorage`; it then clears shard
bookkeeping and raw-component views and replaces parameter loaders with a
read-only rejection function. There is no separate host-store sealing state.

The loader checks both known input/output bytes and torch_npu's measured
incremental allocation peak against `workspace_limit_bytes`. The native probe
is necessary to discover backend padding and scratch, so an unknown native
peak can only be rejected after that bounded probe. The measurement resets
torch_npu peak statistics during serial model loading. Allocations outside the
torch_npu allocator require separate backend accounting and hardware validation.

Banks allocate the native pre-transpose shape with `empty_with_format`, then
apply the resident path's transpose/scale views. The [official torch_npu
operator schema](https://raw.githubusercontent.com/Ascend/pytorch/master/torch_npu/csrc/aten/npu_native_functions.yaml)
exposes both `empty_with_format` and `get_storage_size`. Byte carrier descriptor
size must agree with the allocation extent. Every bank must have exactly
capacity times the single-expert physical extent. Nonzero-offset views must
retain storage and the exact physical expert offset; unsupported slicing fails
closed. Views report no new allocation, and a zero-capacity hot bank reports no
storage. No pointer or bank tensor is rebound during execution.

## Target hardware checks

Production uses the deployed torch_npu adapter and installed MemFabric host
allocator. Existing CPU tests patch only native/device boundaries to verify byte
layout, loading, workspace rejection and retirement; they do not certify NZ
operators, registration or graph behavior on hardware.

On the target Ascend/CANN/torch_npu/MemFabric combination, qualify registration
of the MemFabric allocation, compare a resident expert with host-serialized and
reloaded fixed-bank weights/scales including padding, and run the same GMM inputs
through both paths. Measure loader peak bytes and exercise registration/copy
failure plus teardown. Validate full graph replay and stream ordering separately.
