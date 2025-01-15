import os

import numpy as np
import pooch


def test_load_atlas():
    """Test that passes."""
    solar_atlas_filepath = pooch.retrieve(
        url="doi:10.5281/zenodo.14641640/solar_reference_atlas.npy",
        known_hash="md5:84ab4c50689ef235fe5ed4f7ee905ca0",
    )
    solar_atlas_wavelength, solar_atlas_transmission = np.load(solar_atlas_filepath)

    telluric_atlas_filepath = pooch.retrieve(
        url="doi:10.5281/zenodo.14646787/telluric_reference_atlas.npy",
        known_hash="md5:8db5e12508b293bca3495d81a0747447",
    )
    telluric_atlas_wavelength, telluric_atlas_transmission = np.load(telluric_atlas_filepath)

    assert os.path.exists(solar_atlas_filepath)
    assert os.path.exists(telluric_atlas_filepath)
    assert isinstance(solar_atlas_wavelength, np.ndarray)
    assert isinstance(solar_atlas_transmission, np.ndarray)
    assert isinstance(telluric_atlas_wavelength, np.ndarray)
    assert isinstance(telluric_atlas_transmission, np.ndarray)
