"""Single-writer 32-bit publication/snapshot using explicit NPU DCache bypass."""
import ctypes
from pathlib import Path
import torch


class ReplicaFlag:
    def __init__(self, library):
        self.library = ctypes.CDLL(str(Path(library).resolve()))
        self.function = self.library.AccReplicaFlag
        self.function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint8]
        self.function.restype = None

    def __call__(self, source, destination):
        if (source.dtype != torch.int32 or destination.dtype != torch.int32
                or source.device != destination.device or source.device.type != 'npu'
                or source.numel()!=1 or destination.numel()!=1):
            raise ValueError('Replica flags must be int32 scalars on the same NPU')
        self.function(source.data_ptr(), destination.data_ptr(), source.device.index)
