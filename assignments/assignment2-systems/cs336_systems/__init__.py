import importlib.metadata

from .ddp import DistributedDataParallel
from .flash_attention import FlashAttentionPyTorch, FlashAttentionTriton
from .fsdp import FullyShardedDataParallel
from .sharded_optimizer import ShardedOptimizer

try:
    __version__ = importlib.metadata.version("cs336-systems")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "DistributedDataParallel",
    "FlashAttentionPyTorch",
    "FlashAttentionTriton",
    "FullyShardedDataParallel",
    "ShardedOptimizer",
    "__version__",
]
