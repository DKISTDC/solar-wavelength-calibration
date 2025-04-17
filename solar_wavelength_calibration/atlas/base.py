"""Base class for solar atlas data."""
from abc import ABC
from abc import abstractmethod

import numpy as np


class AtlasBase(ABC):
    """Define the atlas interface for atlas data used in wavelength calibration."""

    @property
    @abstractmethod
    def telluric_atlas_wavelength(self) -> np.ndarray:
        """Return the wavelength array of the telluric atlas."""

    @property
    @abstractmethod
    def telluric_atlas_transmission(self) -> np.ndarray:
        """Return the transmission array of the telluric atlas."""

    @property
    @abstractmethod
    def solar_atlas_wavelength(self) -> np.ndarray:
        """Return the wavelength array of the solar atlas."""

    @property
    @abstractmethod
    def solar_atlas_transmission(self) -> np.ndarray:
        """Return the transmission array of the solar atlas."""


class CroppedAtlasBase(AtlasBase):
    """Cropped version of the atlas."""

    def __init__(
        self,
        cropped_solar_atlas_wavelength: np.ndarray,
        cropped_telluric_atlas_wavelength: np.ndarray,
        cropped_solar_atlas_transmission: np.ndarray,
        cropped_telluric_atlas_transmission: np.ndarray,
    ):
        self._solar_atlas_wavelength = cropped_solar_atlas_wavelength
        self._telluric_atlas_wavelength = cropped_telluric_atlas_wavelength
        self._solar_atlas_transmission = cropped_solar_atlas_transmission
        self._telluric_atlas_transmission = cropped_telluric_atlas_transmission

    @property
    def telluric_atlas_wavelength(self) -> np.ndarray:
        """Return the wavelength array of the telluric atlas."""
        return self._telluric_atlas_wavelength

    @property
    def telluric_atlas_transmission(self) -> np.ndarray:
        """Return the transmission array of the telluric atlas."""
        return self._telluric_atlas_transmission

    @property
    def solar_atlas_wavelength(self) -> np.ndarray:
        """Return the wavelength array of the solar atlas."""
        return self._solar_atlas_wavelength

    @property
    def solar_atlas_transmission(self) -> np.ndarray:
        """Return the transmission array of the solar atlas."""
        return self._solar_atlas_transmission
