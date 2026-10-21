"""Build/load the header-matched ACL shim and bind it to eager prefill.

Building is explicit and requires the installed CANN SDK and torch_npu. Importing
this module on CPU does not import either library or start a build. A successful
host-queue flush provides enqueue ordering, not device completion or graph proof.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from threading import RLock

from .npu_transfer import (
    AclBatchHostTransferOps,
    DeviceMemoryRegion,
    _npu_completion,
    _npu_stream_context,
)

_MODULE_NAME = "_sglang_moe_acl"
_NATIVE = Path(__file__).with_name("native")
_LOAD_LOCK = RLock()
_LOADED = {}


def probe_batch_abi(include_directory, *, compiler=None, work_directory=None):
    """Compile exact function-pointer casts; resolve SDK overloads without guessing.

    Prefer the C form with failIndex. The nine-argument C++ form is accepted only
    if the installed header itself declares it with the expected full signature.
    Linking is left to the real extension build; compile probes cannot certify a
    shared library or hardware implementation.
    """
    command = shlex.split(compiler or os.environ.get("CXX", "c++"))
    if not command:
        raise ValueError("A C++ compiler command is required")
    errors = []
    with tempfile.TemporaryDirectory(dir=work_directory, prefix="acl-abi-") as tmp:
        tmp = Path(tmp)
        for arity in (10, 9):
            source = tmp / f"probe_{arity}.cpp"
            failure_parameter = "size_t*, " if arity == 10 else ""
            source.write_text(
                "#include <acl/acl.h>\n#include <cstddef>\n"
                "using Fn = aclError (*)(void**, size_t*, void**, size_t*, size_t, "
                "aclrtMemcpyBatchAttr*, size_t*, size_t, "
                + failure_parameter
                + "aclrtStream);\n"
                "Fn checked = static_cast<Fn>(&aclrtMemcpyBatchAsync);\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                command
                + [
                    "-std=c++17",
                    "-I",
                    str(include_directory),
                    "-c",
                    str(source),
                    "-o",
                    str(tmp / f"probe_{arity}.o"),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                return arity
            errors.append(f"{arity} arguments: {result.stderr[-3000:]}")
    raise RuntimeError(
        "Installed CANN headers expose no supported aclrtMemcpyBatchAsync ABI. "
        "No ctypes/dynamic-symbol fallback is allowed.\n" + "\n".join(errors)
    )


def build_native_acl(build_directory, *, cann_home=None, verbose=False):
    """Explicitly compile/link against this process's torch/torch_npu and CANN."""
    import torch_npu
    from torch.utils.cpp_extension import load

    home = Path(
        cann_home
        or os.environ.get("ASCEND_HOME_PATH")
        or os.environ.get("ASCEND_TOOLKIT_HOME")
        or "/usr/local/Ascend/ascend-toolkit/latest"
    ).resolve()
    include = home / "include"
    library = home / "lib64"
    if not (include / "acl" / "acl.h").is_file():
        raise RuntimeError(f"CANN SDK header missing: {include / 'acl/acl.h'}")
    if not (library / "libascendcl.so").is_file():
        raise RuntimeError(
            f"CANN runtime library missing: {library / 'libascendcl.so'}"
        )
    build_directory = Path(build_directory).resolve()
    build_directory.mkdir(parents=True, exist_ok=True)
    abi = probe_batch_abi(include, work_directory=build_directory)
    npu = Path(torch_npu.__file__).resolve().parent
    module = load(
        name=_MODULE_NAME,
        sources=[str(_NATIVE / "acl_batch.cpp")],
        build_directory=str(build_directory),
        extra_include_paths=[
            str(include),
            str(npu / "include"),
            str(npu / "include" / "third_party" / "hccl" / "inc"),
        ],
        extra_cflags=[
            "-std=c++17",
            "-fvisibility=hidden",
            f"-DSGLANG_ACL_BATCH_ABI={abi}",
        ],
        extra_ldflags=[
            f"-L{library}",
            "-lascendcl",
            f"-L{npu / 'lib'}",
            "-ltorch_npu",
            f"-Wl,-rpath,{library}",
            f"-Wl,-rpath,{npu / 'lib'}",
        ],
        verbose=verbose,
    )
    with _LOAD_LOCK:
        _LOADED[str(build_directory)] = module
    return module


def get_native_acl(build_directory=None):
    """Load a built extension; absence never triggers compilation or eager fallback."""
    # torch_npu loads its matching shared libraries and device registration first.
    import torch_npu  # noqa: F401

    key = str(Path(build_directory).resolve()) if build_directory is not None else None
    with _LOAD_LOCK:
        if key in _LOADED:
            return _LOADED[key]
        module = _load_native_acl(key)
        _LOADED[key] = module
        return module


def _load_native_acl(directory):
    if directory is None:
        try:
            return importlib.import_module(_MODULE_NAME)
        except ImportError as error:
            raise RuntimeError(
                "ACL extension is unbuilt. Run python -m "
                "sglang.srt.offloading.moe.native_acl --build-directory PATH "
                "on the target CANN/torch_npu host and pass that directory."
            ) from error
    directory = Path(directory)
    candidates = [
        directory / (_MODULE_NAME + suffix)
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
        if (directory / (_MODULE_NAME + suffix)).is_file()
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one built ACL extension in " + str(directory)
        )
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, candidates[0])
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load ACL extension: " + str(candidates[0]))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NativeAclBatchSubmitter:
    """Bridge AclCopyBatch to owning native descriptors before any async writes.

    source_location describes the original allocator: ``host`` for ACL host
    allocations, ``unregistered`` for externally allocated memory (which still
    requires a successful pinned-memory registration), or ``host_numa`` for the
    documented host NUMA allocation type. It is not a pinning inference.
    Runtime/pipeline grants this stream one submitting thread at a time.
    """

    def __init__(self, module, retain, *, source_location="host", source_location_id=0):
        if not callable(retain):
            raise TypeError("A durable runtime retention callback is required")
        if source_location not in ("host", "host_numa", "unregistered"):
            raise ValueError("Unsupported ACL source allocation location")
        if type(source_location_id) is not int or not 0 <= source_location_id < 2**32:
            raise ValueError("ACL source location ID must fit uint32")
        info = module.api_info()
        if info.get("queue_ordering") != "torch_npu_stream_flush" or info.get(
            "device_synchronize"
        ):
            raise ValueError(
                "Loaded ACL shim lacks the expected host-queue ordering contract"
            )
        self.module, self.retain = module, retain
        self.source_location, self.source_location_id = (
            source_location,
            source_location_id,
        )

    def __call__(self, batch, stream):
        device = getattr(stream, "device", None)
        device_index = getattr(device, "index", None)
        stream_id = getattr(stream, "stream_id", None)
        if getattr(device, "type", None) != "npu" or type(device_index) is not int:
            raise ValueError(
                "Native ACL requires a managed NPU stream with an explicit device"
            )
        if type(stream_id) is not int or not 0 <= device_index <= 127:
            raise ValueError("Invalid torch_npu stream ID/device")
        if batch.direction != "host_to_device":
            raise ValueError("The native prefill shim accepts H2D only")
        if len(batch.keepalive) != len(batch.sizes):
            raise ValueError("ACL batch must retain each source/destination region")
        for owners in batch.keepalive:
            if not isinstance(owners, tuple) or len(owners) != 2:
                raise ValueError("ACL batch must retain source/destination pairs")
            destination = owners[1]
            if (
                not isinstance(destination, DeviceMemoryRegion)
                or getattr(destination.device, "type", None) != "npu"
                or getattr(destination.device, "index", None) != device_index
            ):
                raise ValueError(
                    "ACL destination must belong to the submitting stream device"
                )
        # No raw aclrtStream is obtained through Python. C++ reads and validates
        # the current managed stream after entering torch.npu.stream(stream).
        native = self.module.Batch(
            batch.source_addresses,
            batch.destination_addresses,
            batch.sizes,
            batch.destination_capacities,
            device_index,
            self.source_location,
            self.source_location_id,
        )
        self.retain((native, batch, stream, self))
        self.module.submit(native, stream_id)
        return 0


def make_acl_host_ops(
    *,
    retain,
    module=None,
    build_directory=None,
    source_location="host",
    source_location_id=0,
    max_batch_entries=256,
):
    """Construct the existing ACL backend with real SDK calls and NPU events."""
    module = get_native_acl(build_directory) if module is None else module
    submitter = NativeAclBatchSubmitter(
        module,
        retain,
        source_location=source_location,
        source_location_id=source_location_id,
    )
    return AclBatchHostTransferOps(
        native_submit=submitter,
        record_completion=_npu_completion,
        retain=retain,
        queue_ordering_verified=True,
        stream_context=_npu_stream_context,
        max_batch_entries=max_batch_entries,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build the CANN/torch_npu ACL batch extension"
    )
    parser.add_argument("--build-directory", required=True)
    parser.add_argument("--cann-home")
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args()
    result = build_native_acl(
        arguments.build_directory,
        cann_home=arguments.cann_home,
        verbose=arguments.verbose,
    )
    print(result.api_info())
