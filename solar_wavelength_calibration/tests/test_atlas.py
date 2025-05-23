"""Tests for the atlas package."""
import os
from pathlib import Path
from unittest import mock

import numpy as np
import pooch
import pydantic
import pytest

from solar_wavelength_calibration import Atlas
from solar_wavelength_calibration import WavelengthCalibrationFitter
from solar_wavelength_calibration.atlas.atlas import CACHE_DIRECTORY_OVERRIDE_ENVIRONMENT_VAR
from solar_wavelength_calibration.atlas.atlas import DEFAULT_POOCH_CACHE_DIR
from solar_wavelength_calibration.atlas.atlas import DownloadConfig


@pytest.fixture(
    params=[pytest.param(True, id="override_cache"), pytest.param(False, id="default_cache")],
)
def cache_dir(request, tmp_path) -> Path:
    override_cache_dir = request.param
    match override_cache_dir:
        case True:
            cache_path = tmp_path
            environ_dict = {CACHE_DIRECTORY_OVERRIDE_ENVIRONMENT_VAR: str(cache_path)}
        case False:
            cache_path = pooch.os_cache(DEFAULT_POOCH_CACHE_DIR)
            environ_dict = dict()

    with mock.patch.dict(os.environ, os.environ | environ_dict):
        yield cache_path


test_config = DownloadConfig(
    base_url="doi:10.5281/zenodo.14728809",
    telluric_reference_atlas_file_name="test_telluric_reference_atlas.npy",
    telluric_reference_atlas_hash_id="md5:a06c6923b794479f2b0ac483733402a7",
    solar_reference_atlas_file_name="test_solar_reference_atlas.npy",
    solar_reference_atlas_hash_id="md5:d692bff029923e833f900ebc59c4435a",
)


@pytest.fixture(
    params=[
        pytest.param(None, id="default_config"),
        pytest.param(test_config, id="test_config"),
    ],
)
def function_config(request) -> DownloadConfig | None:
    return request.param


@pytest.fixture()
def function_atlas(cache_dir: Path, function_config: DownloadConfig | None) -> Atlas:
    return Atlas(config=function_config)


def test_atlas(function_atlas: Atlas, cache_dir: Path):
    """Given: an Atlas object
    When: the object is created
    Then: the object should have the correct attributes."""
    assert isinstance(function_atlas.telluric_atlas_wavelength, np.ndarray)
    assert isinstance(function_atlas.telluric_atlas_transmission, np.ndarray)
    assert isinstance(function_atlas.solar_atlas_wavelength, np.ndarray)
    assert isinstance(function_atlas.solar_atlas_transmission, np.ndarray)
    assert function_atlas.pooch.path == cache_dir


def test_atlas_repr():
    """Given: an Atlas object
    When: the object is printed
    Then: the object should have the correct representation."""
    atlas = Atlas()
    assert isinstance(eval(repr(atlas)), Atlas)


def test_solar_wavelength_non_monotonic(parameters):
    """Given: an Atlas object with non-monotonically increasing solar wavelength
    When: instantiating the WavelengthCalibrationFitter
    Then: a Pydantic ValidationError should be raised."""
    atlas = Atlas()
    # Inject non-monotonic solar wavelength data into the Atlas object
    atlas._solar_atlas = (
        np.array([400, 500, 480, 700]),  # Non-monotonic wavelength
        np.array([1.0, 0.9, 0.8, 0.7]),  # Corresponding transmission
    )

    with pytest.raises(
        pydantic.ValidationError, match="Solar atlas wavelength array is not monotonic."
    ):
        WavelengthCalibrationFitter(input_parameters=parameters, atlas=atlas)


def test_telluric_wavelength_non_monotonic(parameters):
    """Given: an Atlas object with non-monotonically increasing telluric wavelength
    When: instantiating the WavelengthCalibrationFitter
    Then: a Pydantic ValidationError should be raised."""
    atlas = Atlas()
    # Inject non-monotonic telluric wavelength data into the Atlas object
    atlas._telluric_atlas = (
        np.array([400, 500, 480, 700]),  # Non-monotonic wavelength
        np.array([1.0, 0.9, 0.8, 0.7]),  # Corresponding transmission
    )

    with pytest.raises(
        pydantic.ValidationError, match="Telluric atlas wavelength array is not monotonic."
    ):
        WavelengthCalibrationFitter(input_parameters=parameters, atlas=atlas)
