"""Task to load the reference atlas files."""
import logging
from functools import cached_property

import numpy as np
import pooch
from pydantic import BaseModel

from solar_wavelength_calibration.atlas.base import AtlasBase


logger = logging.getLogger(__name__)


class DownloadConfig(BaseModel):
    """Configuration for downloading the reference atlas files."""

    base_url: str
    telluric_reference_atlas_file_name: str
    telluric_reference_atlas_hash_id: str
    solar_reference_atlas_file_name: str
    solar_reference_atlas_hash_id: str

    @property
    def registry(self) -> dict[str, str]:
        """Registry of the reference atlas files."""
        return {
            self.telluric_reference_atlas_file_name: self.telluric_reference_atlas_hash_id,
            self.solar_reference_atlas_file_name: self.solar_reference_atlas_hash_id,
        }


default_config = DownloadConfig(
    base_url="doi:10.5281/zenodo.14646787/",
    telluric_reference_atlas_file_name="telluric_reference_atlas.npy",
    telluric_reference_atlas_hash_id="md5:8db5e12508b293bca3495d81a0747447",
    solar_reference_atlas_file_name="solar_reference_atlas.npy",
    solar_reference_atlas_hash_id="md5:84ab4c50689ef235fe5ed4f7ee905ca0",
)

CACHE_DIRECTORY_OVERRIDE_ENVIRONMENT_VAR = "FTS_ATLAS_DATA_DIR"


class Atlas(AtlasBase):
    """Reference atlas file class."""

    def __init__(self, config: DownloadConfig | None = None):
        if config is None:
            logger.info(f"Using default configuration for FTS atlas: {default_config}")
            config = default_config
        self.config = config
        self.pooch = pooch.create(
            path=pooch.os_cache(__package__),
            base_url=self.config.base_url,
            registry=self.config.registry,
            env=CACHE_DIRECTORY_OVERRIDE_ENVIRONMENT_VAR,
        )
        self.local_telluric_atlas_path = self.pooch.fetch(
            self.config.telluric_reference_atlas_file_name
        )
        self._telluric_atlas = np.load(self.local_telluric_atlas_path)
        self.local_solar_atlas_path = self.pooch.fetch(self.config.solar_reference_atlas_file_name)
        self._solar_atlas = np.load(self.local_solar_atlas_path)

    @property
    def telluric_atlas_wavelength(self) -> np.ndarray:
        """
        Wavelength array of the high-resolution telluric atlas.

        The atlas provides detailed spectral measurements of Earth's atmosphere,
        capturing the absorption features caused by atmospheric molecules such as water vapor,
        oxygen, and carbon dioxide.
        """
        return self._telluric_atlas[0]

    @property
    def telluric_atlas_transmission(self) -> np.ndarray:
        """
        Transmission array of the high-resolution telluric atlas.

        The atlas provides detailed spectral measurements of Earth's atmosphere,
        capturing the absorption features caused by atmospheric molecules such as water vapor,
        oxygen, and carbon dioxide.
        """
        return self._telluric_atlas[1]

    @property
    def solar_atlas_wavelength(self) -> np.ndarray:
        """
        Wavelength array of the high-resolution solar atlas.

        It provides detailed measurements across a wide range of wavelengths,
        capturing the spectral features of the Sun with exceptional precision.
        """
        return self._solar_atlas[0]

    @property
    def solar_atlas_transmission(self) -> np.ndarray:
        """
        Transmission array of the high-resolution solar atlas.

        It provides detailed measurements across a wide range of wavelengths,
        capturing the spectral features of the Sun with exceptional precision.
        """
        return self._solar_atlas[1]

    def __repr__(self):
        return f"Atlas(config={self.config!r})"
