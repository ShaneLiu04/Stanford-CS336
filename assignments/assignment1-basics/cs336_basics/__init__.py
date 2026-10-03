import importlib.metadata

try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    pass
