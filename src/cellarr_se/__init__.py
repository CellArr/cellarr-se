from .cellarr_se import CellArrSE
from importlib.metadata import PackageNotFoundError, version

__all__ = ["CellArrSE"]

try:
    dist_name = "cellarr-se"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError
