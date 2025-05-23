"""Library that supports wavelength calibration."""
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

from .atlas.atlas import *
from .fitter.parameters import *
from .fitter.wavelength_fitter import *

try:
    __version__ = version(distribution_name=__name__)
except PackageNotFoundError:  # pragma: no cover
    # package is not installed
    __version__ = "unknown"  # pragma: no cover
