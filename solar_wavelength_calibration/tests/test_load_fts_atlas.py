import os
from unittest.mock import MagicMock
from unittest.mock import patch

import pooch
import pytest

from solar_wavelength_calibration.load_atlas import Atlas
from solar_wavelength_calibration.load_atlas import default_config
from solar_wavelength_calibration.load_atlas import DownloadConfig


@pytest.fixture
def custom_config():
    return DownloadConfig(
        base_url="custom_base_url/",
        telluric_reference_atlas_file_name="custom_telluric_file.npy",
        telluric_reference_atlas_hash_id="md5:custom_hash_1",
        solar_reference_atlas_file_name="custom_solar_file.npy",
        solar_reference_atlas_hash_id="md5:custom_hash_2",
    )


def test_default_config_usage():
    atlas = Atlas()
    assert atlas.config == default_config
    assert atlas.pooch.base_url == default_config.base_url
    assert atlas.pooch.registry == default_config.registry


def test_custom_config_usage(custom_config):
    atlas = Atlas(config=custom_config)
    assert atlas.config == custom_config
    assert atlas.pooch.base_url == custom_config.base_url
    assert atlas.pooch.registry == custom_config.registry


def test_environment_variable_override():
    with patch("pooch.create") as mock_pooch_create:
        mock_pooch_create.return_value = MagicMock()
        with patch.dict(os.environ, {"FTS_ATLAS_DATA_DIR": "/custom/path"}):
            Atlas()
            mock_pooch_create.assert_called_once_with(
                path=pooch.os_cache("pooch"),
                base_url=default_config.base_url,
                registry=default_config.registry,
                env="FTS_ATLAS_DATA_DIR",
            )


def test_no_environment_variable():
    with patch("pooch.create") as mock_pooch_create:
        mock_pooch_create.return_value = MagicMock()
        if "FTS_ATLAS_DATA_DIR" in os.environ:
            del os.environ["FTS_ATLAS_DATA_DIR"]
        Atlas()
        mock_pooch_create.assert_called_once_with(
            path=pooch.os_cache("pooch"),
            base_url=default_config.base_url,
            registry=default_config.registry,
            env="FTS_ATLAS_DATA_DIR",
        )


def test_fetch_telluric_atlas(custom_config):
    with patch("numpy.load", return_value=("mock_wavelength", "mock_transmission")) as mock_load:
        atlas = Atlas(config=custom_config)
        atlas.pooch.fetch = MagicMock(return_value="mock_file_path")

        telluric_atlas = atlas.telluric_atlas

        atlas.pooch.fetch.assert_called_once_with(custom_config.telluric_reference_atlas_file_name)
        assert mock_load.called_once()
        assert telluric_atlas == ("mock_wavelength", "mock_transmission")


def test_fetch_solar_atlas(custom_config):
    with patch("numpy.load", return_value=("mock_wavelength", "mock_transmission")) as mock_load:
        atlas = Atlas(config=custom_config)
        atlas.pooch.fetch = MagicMock(return_value="mock_file_path")

        solar_atlas = atlas.solar_atlas

        atlas.pooch.fetch.assert_called_once_with(custom_config.solar_reference_atlas_file_name)
        assert mock_load.called_once()
        assert solar_atlas == ("mock_wavelength", "mock_transmission")
