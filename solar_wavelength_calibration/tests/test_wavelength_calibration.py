import os
import warnings

import astropy.units as u
import numpy as np
import pooch
import pytest
from astropy.wcs import WCS

from solar_wavelength_calibration.wavelength_calibration import load_and_resample_atlas


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


def test_resample_atlas():
    """Test that passes."""
    spectrum = np.ones(916)  # 916 is the number of pixels in a typical cryo-nirsp spectrum
    number_of_wave_pix = np.size(spectrum)
    header["CRPIX1"] = number_of_wave_pix // 2 + 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  ## TO ELIMINATE datafix warnings
        wcs = WCS(header)
        # get wavelength based on header info (not fully accurate)
        expected_wavelength_vector = wcs.spectral.pixel_to_world(np.arange(number_of_wave_pix)).to(
            u.nm
        )

    # expected_wavelength_vector: u.Quantity =

    """2. Test that atlas was resampled to correct size
    assert len(telluric_atlas_wavelength)

    3. Test that resampled atlas is smaller than the original atlas"""


def test_initial_alignment():
    """Test that passes."""
    assert True


def test_fit_to_atlas():
    """Test that passes."""
    assert True
