# Temporary Qwen3.8-Flash-Next NPU Triton kernels

This directory is a temporary staging area for pure-Python Triton kernels used
to validate and optimize Qwen3.8-Flash-Next on Ascend. The intended long-term
owner is [`sgl-kernel-npu`](https://github.com/sgl-project/sgl-kernel-npu).

Code in this directory is temporary, but it is production-executed while it is
present. It must therefore meet the same correctness, fallback, review, and
testing standards as permanent runtime code.

## What belongs here

Only code that can later move into `sgl-kernel-npu` with a small, reviewable
integration change belongs here:

- Triton JIT kernel implementations;
- thin Python launch wrappers;
- launch configuration and shape/dtype/stride validation;
- small utilities used exclusively by those kernels.

Keep kernels split by function rather than collecting them in one large model
file. For example, QSA indexer kernels belong in `qsa.py`; sparse-attention,
normalization/RoPE, and hyperconnection kernels should use separate modules.

The following remain in their normal SGLang layers or NPU runtime components:

- Torch correctness references and fallbacks;
- model classes and forward-mode decisions;
- QSA/indexer orchestration and metadata objects;
- KV-pool ownership and graph-runner lifecycle;
- CUDA, ROCm, or TileLang implementations;
- server arguments, logging, and user-facing policy.

## How SGLang calls these kernels

The owning layer performs platform/configuration dispatch and imports the
temporary wrapper lazily inside the NPU branch:

```python
if q.device.type == "npu" and q.shape[-1] == 128:
    from sglang.srt.hardware_backend.npu.kernels.qwen3_8_flash_next.qsa import (
        triton_qsa_mqa_decode,
    )

    return triton_qsa_mqa_decode(...)
return torch_reference(...)
```

Lazy imports keep CPU/CUDA module loading independent of this temporary NPU
implementation. Unsupported configurations must use an existing correct
fallback; they must not be silently forced through a partially validated
kernel.

Do not emit a runtime warning merely because a kernel is awaiting migration.
Migration is a maintainer concern, not a user-visible correctness problem.
Warnings are appropriate only for actionable runtime conditions, such as a
materially slow fallback that users can avoid.

## Single Triton source rule

Each operation must have exactly one Triton implementation source at any point
in time:

- before migration, SGLang imports the implementation from this directory;
- after migration, SGLang imports the implementation from `sgl-kernel-npu`.

Do not implement priority probing between a local Triton kernel and an external
Triton kernel. In particular, code like this is prohibited:

```python
try:
    from sgl_kernel_npu.indexer.qsa import qsa_mqa_decode
except ImportError:
    from .qsa import qsa_mqa_decode
```

Do not keep copied implementations in both repositories, even temporarily in
a submitted SGLang change. Dual Triton sources inevitably drift in correctness,
launch policy, supported shapes, and performance tuning.

A Torch reference/fallback is not a second Triton source. It should remain
available for correctness testing and unsupported configurations. An optional
external import may fall back to Torch when the required API or supported shape
is absent, but it must never search for another Triton implementation.

Only capability detection may trigger fallback. Once a supported Triton kernel
has been selected, compilation or execution failures must remain visible. Do
not catch a broad exception around kernel execution and silently run Torch,
because that can make tests pass without exercising the intended kernel.

## Shape handling and capability ownership

Classify a shape difference before deciding how to handle it:

1. Normal dynamic shapes, such as batch size, context length, ragged lengths,
   partial tiles, graph padding rows, and an incomplete final page, should be
   handled by the kernel with runtime arguments and masks. They should not
   normally trigger fallback or warnings.
2. A valid model/kernel configuration that is not optimized yet, such as an
   otherwise supported dtype, head count, head dimension, page size, layout,
   or device generation, may use the Torch fallback. This is capability
   fallback, not an input error.
3. A shape that violates the model or runtime contract, such as a tensor head
   dimension disagreeing with model configuration, an invalid KV-head ratio,
   or a page table incompatible with the cache layout, must raise a clear
   error. Do not hide invariant violations behind a Torch fallback.

Padding is appropriate only when the mathematical result and physical layout
are preserved. Examples include zero-padding a head/vector tile to a required
alignment and masking an incomplete token tile. Do not use padding to disguise
incompatible page sizes, cache layouts, head relationships, or metadata.

Each kernel module should own and export its capability predicate alongside the
launch wrapper, for example:

```python
def can_run_qsa_mqa_decode(q, k_cache, page_table, context_lens) -> bool:
    return (
        q.device.type == "npu"
        and q.dtype in (torch.bfloat16, torch.float16)
        and k_cache.dtype == q.dtype
        and q.shape[-1] == 128
        and k_cache.shape[1] == 16
        and k_cache.shape[2] == 1
    )
```

The owning SGLang layer queries that predicate to choose the optimized path or
the Torch fallback. The kernel wrapper must validate the same contract again
and raise if it is called directly with unsupported inputs. This mirrors the
GPU pattern of a public support predicate plus defensive wrapper validation.

SGLang owns model and runtime invariants; the kernel module owns optimization
coverage. Do not permanently duplicate kernel-specific dtype, shape, layout,
or architecture gates in SGLang. When a kernel moves to `sgl-kernel-npu`, its
capability predicate moves with it and SGLang imports both the predicate and
the wrapper from that single source.

## Correctness and performance requirements

Before a temporary kernel replaces a Torch path, it must be checked against the
Torch reference on the target NPU. Coverage should include all applicable:

- production dtypes and model shapes;
- decode, prefill, and speculative/graph modes;
- batch-size and ragged-length boundaries;
- page sizes, padding, empty inputs, and invalid/sentinel indices;
- non-contiguous layouts when the public wrapper claims to support them;
- eager execution and graph capture/replay;
- numerical tolerances justified by the accumulation order and dtype.

Benchmarks must synchronize the NPU around timed regions, include JIT warmup,
and report any shape where the Triton path regresses. Dispatch should remain on
Torch for unsupported or slower configurations.

## Provisional API policy

The wrappers here are migration-friendly boundaries, not a promise that the
future `sgl-kernel-npu` API will have the same name, signature, output layout,
validation behavior, or launch policy. The destination repository may prefer a
more general API, fuse adjacent operations, return top-k indices directly, or
reuse another indexer primitive.

When that API differs, adapt SGLang at the owning layer. Do not preserve a poor
temporary interface merely to make migration textually identical. Validate the
external API against the same Torch reference and benchmarks in an isolated
kernel-development environment, then switch SGLang to the external source and
remove the local source as one atomic SGLang change.

## Migration procedure

1. Stabilize correctness, graph behavior, and performance in this directory.
2. Design the destination API according to `sgl-kernel-npu` conventions.
3. Move the kernel and its focused tests/benchmarks to the appropriate package,
   such as `sgl_kernel_npu.indexer`, `attention`, or `norm`.
4. Test the local `sgl-kernel-npu` build in an isolated environment so its
   Python package and compiled shared library cannot come from mixed versions.
5. Land or publish the kernel dependency.
6. In one atomic SGLang change, remove the local kernel, replace its import with
   the external import, and adapt arguments/results if the final API changed.
7. Keep only a Torch fallback for a missing API or explicitly unsupported
   configuration; never retain the local Triton implementation as fallback.
8. Repeat Torch-reference, graph, and performance validation through the
   SGLang call site, with an assertion or test proving the external kernel ran.
9. Update this README's inventory and verify that no copy of the migrated
   Triton implementation remains in SGLang.

## Migration completion checklist

A kernel migration is complete only when all of the following are true:

- the local Triton implementation has been deleted from this directory;
- SGLang contains exactly one Triton import path for the operation, pointing to
  `sgl-kernel-npu`;
- repository search confirms there is no copied implementation left behind;
- the remaining fallback is pure Torch rather than another Triton kernel;
- missing external capability and unsupported shapes take the Torch fallback;
- supported shapes execute the external kernel, and kernel failures are not
  swallowed by fallback logic;
- correctness, graph capture/replay, and performance tests pass through the
  real SGLang call site;
- CI or an equivalent recorded test proves that the external kernel path was
  exercised instead of merely passing through the Torch fallback.

## Inventory

| Kernel | Temporary module | Likely destination | Status |
| --- | --- | --- | --- |
| QSA paged MQA decode score | `qsa.py` | `sgl_kernel_npu.indexer.qsa` | Eager NPU correctness and microbenchmark passed; graph validation pending |
