"""Tests for the atlas package."""
import os
from pathlib import Path
from unittest import mock

import numpy as np
import pooch
import pytest

from solar_wavelength_calibration.atlas.atlas import Atlas
from solar_wavelength_calibration.atlas.atlas import CACHE_DIRECTORY_OVERRIDE_ENVIRONMENT_VAR
from solar_wavelength_calibration.atlas.atlas import default_config
from solar_wavelength_calibration.atlas.atlas import DEFAULT_POOCH_CACHE_DIR
from solar_wavelength_calibration.atlas.atlas import DownloadConfig


@pytest.fixture(
    params=[pytest.param(True, id="override_cache"), pytest.param(False, id="default_cache")]
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
    ]
)
def config(request) -> DownloadConfig | None:
    return request.param


@pytest.fixture()
def atlas(cache_dir: Path, config: DownloadConfig | None) -> Atlas:
    return Atlas(config=config)


def test_atlas(atlas: Atlas, cache_dir: Path):
    """Given: an Atlas object
    When: the object is created
    Then: the object should have the correct attributes."""
    assert isinstance(atlas.telluric_atlas_wavelength, np.ndarray)
    assert isinstance(atlas.telluric_atlas_transmission, np.ndarray)
    assert isinstance(atlas.solar_atlas_wavelength, np.ndarray)
    assert isinstance(atlas.solar_atlas_transmission, np.ndarray)
    assert atlas.pooch.path == cache_dir


def test_atlas_repr():
    """Given: an Atlas object
    When: the object is printed
    Then: the object should have the correct representation."""
    atlas = Atlas()
    assert isinstance(eval(repr(atlas)), Atlas)
