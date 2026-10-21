"""Ascend-specific MoE offload storage, execution and native transport adapters.

The package itself does not import torch_npu, load a native extension or allocate
resources. Device dependencies are loaded explicitly by the backend or CLI.
"""
