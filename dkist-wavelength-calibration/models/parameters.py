"""Wavelength calibration parameters."""
from functools import cached_property

import astropy.units as u
import numpy as np

from dkist_processing_common.models.parameters import ParameterArmIdMixin
from dkist_processing_common.models.parameters import ParameterBase
from dkist_processing_common.models.parameters import ParameterWavelengthMixin

class WaveCalParameters(ParameterBase, ParameterWavelengthMixin, ParameterArmIdMixin):
    """Put all wavelength calibration parameters parsed from the input dataset document in a single property."""

    property

    def camera_mirror_focal_length_mm(self) -> u.Quantity:
        """Return the CryoNIRSP camera mirror focal length."""
        return (
                self._find_most_recent_past_value("cryonirsp_camera_mirror_focal_length_mm")
                * u.millimeter
        )

    @property
    def pixel_pitch_micron(self) -> u.Quantity:
        """Return the CryoNIRSP pixel pitch."""
        return self._find_most_recent_past_value("cryonirsp_pixel_pitch_micron") * u.micron

    @cached_property
    def solar_atlas(self) -> np.ndarray:
        """Solar reference atlas.

        Contains two arrays:
        - wavelength in nanometers
        - transmission at given wavelength
        """
        param_dict = self._find_most_recent_past_value("cryonirsp_solar_atlas")
        return self._load_param_value_from_numpy_save(param_dict)

    @cached_property
    def telluric_atlas(self) -> np.ndarray:
        """Telluric reference atlas.

        Contains two arrays:
        - wavelength in nanometers
        - transmission at given wavelength
        """
        param_dict = self._find_most_recent_past_value("cryonirsp_telluric_atlas")
        return self._load_param_value_from_numpy_save(param_dict)