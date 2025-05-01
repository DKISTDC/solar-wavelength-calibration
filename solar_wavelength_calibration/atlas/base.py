"""Base class for solar atlas data."""
from abc import ABC
from abc import abstractmethod

import numpy as np
from astropy.units import Quantity


class AtlasBase(ABC):
    """Define the atlas interface for atlas data used in wavelength calibration."""

    @property
    @abstractmethod
    def telluric_atlas_wavelength(self) -> Quantity:
        """Return the wavelength array of the telluric atlas."""

    @property
    @abstractmethod
    def telluric_atlas_transmission(self) -> np.ndarray:
        """Return the transmission array of the telluric atlas."""

    @property
    @abstractmethod
    def solar_atlas_wavelength(self) -> Quantity:
        """Return the wavelength array of the solar atlas."""

    @property
    @abstractmethod
    def solar_atlas_transmission(self) -> np.ndarray:
        """Return the transmission array of the solar atlas."""

    @property
    def telluric_atlas_wavelength_is_monotonic(self):
        """Check if the telluric atlas wavelength array is strictly increasing."""
        return self._is_monotonic(self.telluric_atlas_wavelength)

    @property
    def solar_atlas_wavelength_is_monotonic(self):
        """Check if the solar atlas wavelength array is strictly increasing."""
        return self._is_monotonic(self.solar_atlas_wavelength)

    @staticmethod
    def _is_monotonic(array: np.ndarray) -> bool:
        """Check that the wavelength array is strictly increasing."""
        return np.all(np.diff(array) > 0)  # Strictly increasing check
