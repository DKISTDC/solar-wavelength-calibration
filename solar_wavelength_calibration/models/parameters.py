"""Wavelength calibration parameters."""
from functools import cached_property

import numpy as np
from dkist_processing_common.models.parameters import ParameterArmIdMixin
from dkist_processing_common.models.parameters import ParameterBase
from dkist_processing_common.models.parameters import ParameterWavelengthMixin


class WavecalParameters(ParameterBase, ParameterWavelengthMixin, ParameterArmIdMixin):
    """Put all wavelength calibration parameters parsed from the input dataset document in a single property."""

    @cached_property
    def solar_atlas(self) -> np.ndarray:
        """Solar reference atlas.

        Contains two arrays:
        - wavelength in nanometers
        - transmission at given wavelength
        """
        param_dict = self._find_most_recent_past_value("solar_atlas")
        return self._load_param_value_from_numpy_save(param_dict)

    @cached_property
    def telluric_atlas(self) -> np.ndarray:
        """Telluric reference atlas.

        Contains two arrays:
        - wavelength in nanometers
        - transmission at given wavelength
        """
        param_dict = self._find_most_recent_past_value("telluric_atlas")
        return self._load_param_value_from_numpy_save(param_dict)
