"""Library that supports wavelength calibration."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

from .atlas.atlas import Atlas
from .atlas.atlas import DownloadConfig
from .atlas.atlas import LocalAtlas
from .fitter.parameters import AngleBoundRange
from .fitter.parameters import BoundsModel
from .fitter.parameters import DispersionBoundRange
from .fitter.parameters import FitFlagsModel
from .fitter.parameters import LengthBoundRange
from .fitter.parameters import UnitlessBoundRange
from .fitter.parameters import WavelengthCalibrationParameters
from .fitter.wavelength_fitter import WavelengthCalibrationFitter

try:
    __version__ = version(distribution_name=__name__)
except PackageNotFoundError:  # pragma: no cover
    # package is not installed
    __version__ = "unknown"  # pragma: no cover
