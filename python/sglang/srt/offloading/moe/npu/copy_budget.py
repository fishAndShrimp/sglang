"""Explicit ABI for the optional, isolated A3 copy-budget experiment library."""

import ctypes
from pathlib import Path


class CopyBudgetKernel:
    def __init__(self, library):
        self.library = ctypes.CDLL(str(Path(library).resolve()))
        self.function = self.library.AccOffloadGroupPackCopyBudget
        self.function.argtypes = [ctypes.c_void_p] * 6 + [
            ctypes.c_uint8,
            ctypes.c_uint32,
        ]
        self.function.restype = None

    def __call__(self, source, dest, lengths, scan, groups, packed, device, cores):
        self.function(
            source.data_ptr(),
            dest.data_ptr(),
            lengths.data_ptr(),
            scan.data_ptr(),
            groups.data_ptr(),
            packed.data_ptr(),
            device.index,
            cores,
        )
        return 0
