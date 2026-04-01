from importlib.metadata import PackageNotFoundError, version

try:
    dist_name = "cellarr-se"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

from .cellarr_se import CellArrSE

__all__ = ["CellArrSE"]
