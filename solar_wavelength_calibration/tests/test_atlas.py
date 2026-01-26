"""Tests for the atlas package."""

import os
from pathlib import Path
from typing import Generator
from unittest import mock

import numpy as np
import pooch
import pydantic
import pytest
from filelock import FileLock

from solar_wavelength_calibration import Atlas
from solar_wavelength_calibration import WavelengthCalibrationFitter
from solar_wavelength_calibration.atlas.atlas import CACHE_DIRECTORY_OVERRIDE_ENVIRONMENT_VAR
from solar_wavelength_calibration.atlas.atlas import DEFAULT_POOCH_CACHE_DIR
from solar_wavelength_calibration.atlas.atlas import DownloadConfig
from solar_wavelength_calibration.tests.conftest import test_full_atlas_config


@pytest.fixture(scope="session")
def override_cache_path(tmp_path_factory) -> Path:
    root_tmp_dir = tmp_path_factory.getbasetemp().parent  # shared by all xdist workers
    cache_dir = root_tmp_dir / "pooch_cache_override"
    return cache_dir


@pytest.fixture(
    params=[pytest.param(True, id="override_cache"), pytest.param(False, id="default_cache")],
)
def cache_dir(request, override_cache_path) -> Generator[Path, None, None]:
    override_cache_dir = request.param
    match override_cache_dir:
        case True:
            cache_path = override_cache_path
            environ_dict = {CACHE_DIRECTORY_OVERRIDE_ENVIRONMENT_VAR: str(cache_path)}
        case False:
            cache_path = pooch.os_cache(DEFAULT_POOCH_CACHE_DIR)
            environ_dict = dict()

    with mock.patch.dict(os.environ, os.environ | environ_dict):
        yield cache_path


# Version of small atlas hosted at NSO to avoid hammering Dataverse
# Do we need to test the small atlas anymore?
test_small_atlas_config = DownloadConfig(
    base_url="https://share.nso.edu/shared/dkist/DC/solar_wavelength_calibration_atlas/",
    telluric_reference_atlas_file_name="test_telluric_reference_atlas.npy",
    telluric_reference_atlas_hash_id="md5:a06c6923b794479f2b0ac483733402a7",
    solar_reference_atlas_file_name="test_solar_reference_atlas.npy",
    solar_reference_atlas_hash_id="md5:d692bff029923e833f900ebc59c4435a",
)


@pytest.fixture(
    params=[
        pytest.param(test_full_atlas_config, id="full_atlas_config"),
        pytest.param(test_small_atlas_config, id="small_atlas_config"),
    ],
)
def function_config(request) -> DownloadConfig | None:
    return request.param


@pytest.fixture()
def function_atlas(
    cache_dir: Path, function_config: DownloadConfig | None, atlas_lock_file_path
) -> Atlas:
    # See session scoped `atlas` in conftest for more info
    with FileLock(atlas_lock_file_path):
        atlas = Atlas(config=function_config)

        telluric = atlas._telluric_atlas
        solar = atlas._solar_atlas

    return atlas


def test_atlas(function_atlas: Atlas, cache_dir: Path):
    """
    Given: an Atlas object
    When: the object is created
    Then: the object should have the correct attributes.
    """
    assert isinstance(function_atlas.telluric_atlas_wavelength, np.ndarray)
    assert isinstance(function_atlas.telluric_atlas_transmission, np.ndarray)
    assert isinstance(function_atlas.solar_atlas_wavelength, np.ndarray)
    assert isinstance(function_atlas.solar_atlas_transmission, np.ndarray)
    assert function_atlas.pooch.path == cache_dir


def test_atlas_repr(atlas):
    """
    Given: an Atlas object
    When: the object is printed
    Then: the object should have the correct representation.
    """
    assert isinstance(eval(repr(atlas)), Atlas)


def test_solar_wavelength_non_monotonic(function_atlas, parameters):
    """
    Given: an Atlas object with non-monotonically increasing solar wavelength
    When: instantiating the WavelengthCalibrationFitter
    Then: a Pydantic ValidationError should be raised.
    """
    # Inject non-monotonic solar wavelength data into the Atlas object
    function_atlas._solar_atlas = (
        np.array([400, 500, 480, 700]),  # Non-monotonic wavelength
        np.array([1.0, 0.9, 0.8, 0.7]),  # Corresponding transmission
    )

    with pytest.raises(
        pydantic.ValidationError, match="Solar atlas wavelength array is not monotonic."
    ):
        WavelengthCalibrationFitter(input_parameters=parameters, atlas=function_atlas)


def test_telluric_wavelength_non_monotonic(function_atlas, parameters):
    """
    Given: an Atlas object with non-monotonically increasing telluric wavelength
    When: instantiating the WavelengthCalibrationFitter
    Then: a Pydantic ValidationError should be raised.
    """
    # Inject non-monotonic telluric wavelength data into the Atlas object
    function_atlas._telluric_atlas = (
        np.array([400, 500, 480, 700]),  # Non-monotonic wavelength
        np.array([1.0, 0.9, 0.8, 0.7]),  # Corresponding transmission
    )

    with pytest.raises(
        pydantic.ValidationError, match="Telluric atlas wavelength array is not monotonic."
    ):
        WavelengthCalibrationFitter(input_parameters=parameters, atlas=function_atlas)


def test_public_default_accessible(tmp_path):
    """
    Given: The non-testing default Atlas config
    When: Trying to fetch the files from the service defined in that config
    Then: The files can be downloaded
    """
    # AKA This test will fail if Dataverse or whoever start blocking downloads
    # This test is more a canary in the coal mine than anything else
    unique_override_cache_dir = tmp_path
    environ_dict = {CACHE_DIRECTORY_OVERRIDE_ENVIRONMENT_VAR: str(unique_override_cache_dir)}
    with mock.patch.dict(os.environ, os.environ | environ_dict):
        atlas = Atlas(config=None)
        assert atlas.telluric_atlas_wavelength.size
        assert atlas.solar_atlas_wavelength.size
