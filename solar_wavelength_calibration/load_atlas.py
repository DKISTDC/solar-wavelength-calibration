"""Task to load the reference atlas files."""
import logging

import numpy as np
import pooch
from pydantic import BaseModel

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


class Atlas:
    """Class to load the reference atlas files."""

    def __init__(self, config: DownloadConfig | None = None):
        if config is None:
            logger.info(f"Using default configureation for FTS atlas: {default_config}")
            config = default_config
        self.config = config
        self.telluric_atlas_file_name = self.config.telluric_reference_atlas_file_name
        self.solar_atlas_file_name = self.config.solar_reference_atlas_file_name
        self.cache_directory_override_environment_var = "FTS_ATLAS_DATA_DIR"
        self.pooch = pooch.create(
            path=pooch.os_cache("pooch"),
            base_url=self.config.base_url,
            registry=self.config.registry,
            env=self.cache_directory_override_environment_var,
        )

    @property
    def telluric_atlas(self):
        """Load telluric reference atlas."""
        telluric_atlas = self.pooch.fetch(self.telluric_atlas_file_name)
        return np.load(
            telluric_atlas
        )  # returns telluric_atlas_wavelength and telluric_atlas_transmission in that order

    @property
    def solar_atlas(self):
        """Load solar reference atlas."""
        solar_atlas = self.pooch.fetch(self.solar_atlas_file_name)
        return np.load(
            solar_atlas
        )  # returns solar_atlas_wavelength and solar_atlas_transmission in that order

    def __repr__(self):
        return f"Atlas(config={self.config!r}) # cache_directory_override_environment_var={self.cache_directory_override_environment_var}"
