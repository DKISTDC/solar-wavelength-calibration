from functools import partial
from pathlib import Path
from typing import Callable

import astropy.units as u
import numpy as np
import pytest
from astropy.units import Quantity
from astropy.wcs import WCS
from filelock import FileLock
from lmfit import Parameters

from solar_wavelength_calibration.atlas.atlas import Atlas
from solar_wavelength_calibration.atlas.atlas import DownloadConfig
from solar_wavelength_calibration.atlas.base import AtlasBase
from solar_wavelength_calibration.fitter.parameters import WavelengthCalibrationParameters

# Version of full atlas hosted at NSO to avoid hammering Dataverse
test_full_atlas_config = DownloadConfig(
    base_url="https://share.nso.edu/shared/dkist/DC/solar_wavelength_calibration_atlas",
    telluric_reference_atlas_file_name="telluric_reference_atlas.npy",
    telluric_reference_atlas_hash_id="md5:8db5e12508b293bca3495d81a0747447",
    solar_reference_atlas_file_name="solar_reference_atlas.npy",
    solar_reference_atlas_hash_id="md5:84ab4c50689ef235fe5ed4f7ee905ca0",
)


@pytest.fixture(
    params=[
        pytest.param(test_full_atlas_config, id="full_atlas_config"),
    ],
    scope="session",
)
def config(request) -> DownloadConfig | None:
    return request.param


@pytest.fixture(scope="session")
def atlas_lock_file_path(tmp_path_factory) -> Path:
    root_tmp_dir = tmp_path_factory.getbasetemp().parent  # shared by all xdist workers
    lock_path = root_tmp_dir / "atlas_download.lock"
    return lock_path


@pytest.fixture(scope="session")
def atlas(config: DownloadConfig | None, atlas_lock_file_path) -> Atlas:
    """Fixture that guarantees the Atlas is only downloaded once, regardless of the number of xdist workers."""
    # Taking inspiration from https://pytest-xdist.readthedocs.io/en/latest/how-to.html#making-session-scoped-fixtures-execute-only-once
    with FileLock(atlas_lock_file_path):
        atlas = Atlas(config=config)

        # Actually trigger the download by accessing these properties
        telluric = atlas._telluric_atlas
        solar = atlas._solar_atlas

    return atlas


@pytest.fixture(scope="session")
def num_wave_pix() -> int:
    return 2000


@pytest.fixture(scope="session")
def test_crval() -> Quantity:
    return 1072 * u.nm


@pytest.fixture(scope="session")
def test_dispersion():
    """Dummy dispersion value."""
    return 0.004401829598401599 * u.nm / u.pix


@pytest.fixture(scope="session")
def test_incident_light_angle():
    """Dummy incident_light_angle value."""
    return 57.00577487796545 * u.deg


@pytest.fixture(scope="session")
def test_grating_constant():
    """Dummy grating constant."""
    return 31600.0 / u.m


@pytest.fixture(scope="session")
def test_doppler_velocity():
    """Dummy doppler velocity value."""
    return -0.4281437757265212 * u.km / u.s


@pytest.fixture(scope="session")
def test_order():
    """Dummy order."""
    return 52


@pytest.fixture(scope="session")
def test_WCS_header(
    test_crval,
    test_dispersion,
    test_incident_light_angle,
    test_grating_constant,
    test_order,
    num_wave_pix,
) -> dict:
    return {
        "CTYPE1": "AWAV-GRA",
        "CUNIT1": "nm",
        "CRPIX1": num_wave_pix // 2 + 1,
        "CRVAL1": test_crval.to_value("nm"),
        "CDELT1": test_dispersion.to_value("nm / pix"),
        "PV1_0": test_grating_constant.to_value("1 / m"),
        "PV1_1": test_order,
        "PV1_2": test_incident_light_angle.to_value("deg"),
    }


@pytest.fixture(scope="session")
def observed_wavelength_vector(test_WCS_header, num_wave_pix) -> Quantity:
    wcs = WCS(test_WCS_header)

    obs_wave = wcs.spectral.pixel_to_world(np.arange(num_wave_pix)).to("nm")

    return obs_wave


@pytest.fixture(scope="session")
def observed_spectrum(atlas, observed_wavelength_vector, observed_continuum_level) -> np.ndarray:
    obs_spec = np.interp(
        observed_wavelength_vector.to_value("nm"),
        atlas.solar_atlas_wavelength.to_value("nm"),
        atlas.solar_atlas_transmission,
    )

    return obs_spec * observed_continuum_level


@pytest.fixture(scope="session")
def observed_continuum_level() -> float:
    return 6.28


@pytest.fixture(scope="session")
def test_parameters_dict(
    atlas: AtlasBase,
    observed_spectrum,
    test_crval,
    test_dispersion,
    test_incident_light_angle,
    test_grating_constant,
    test_doppler_velocity,
    test_order,
):
    return {
        "crval": test_crval,
        "dispersion": test_dispersion,
        "incident_light_angle": test_incident_light_angle,
        "resolving_power": 72500,
        "opacity_factor": 5.0,
        "straylight_fraction": 0.25,
        "grating_constant": test_grating_constant,
        "doppler_velocity": test_doppler_velocity,
        "order": test_order,
        "continuum_level": 1.0,
    }


@pytest.fixture(scope="session")
def test_only_required_parameters_dict(
    atlas: AtlasBase,
    observed_spectrum,
    test_crval,
    test_dispersion,
    test_incident_light_angle,
    test_grating_constant,
    test_doppler_velocity,
    test_order,
):
    return {
        "crval": test_crval,
        "dispersion": test_dispersion,
        "incident_light_angle": test_incident_light_angle,
        "grating_constant": test_grating_constant,
        "doppler_velocity": test_doppler_velocity,
        "order": test_order,
    }


@pytest.fixture(scope="function")
def parameters(
    observed_spectrum,
    test_crval,
    test_dispersion,
    test_incident_light_angle,
    test_grating_constant,
    test_doppler_velocity,
    test_order,
    atlas: AtlasBase,
) -> WavelengthCalibrationParameters:
    params = WavelengthCalibrationParameters(
        crval=test_crval,
        dispersion=test_dispersion,
        incident_light_angle=test_incident_light_angle,
        resolving_power=72500,
        opacity_factor=5.0,
        straylight_fraction=0.25,
        grating_constant=test_grating_constant,
        doppler_velocity=test_doppler_velocity,
        order=test_order,
    )
    return params


def user_defined_continuum_function(
    input_wave: np.ndarray, params: Parameters, not_used: bool
) -> np.ndarray:
    return (np.ones_like(input_wave) * 3) ** params["continuum_exponent"]


@pytest.fixture
def wavecal_params_with_extra_continuum_param():
    # AKA an example of how a user might add a custom continuum function
    class NewWaveCalParams(WavelengthCalibrationParameters):

        continuum_exponent: float
        not_used: bool

        @property
        def continuum_function(self) -> Callable[[np.ndarray, Parameters, ...], np.ndarray]:
            return partial(user_defined_continuum_function, not_used=self.not_used)

        @property
        def lmfit_parameters(self):
            params = super().lmfit_parameters
            del params["continuum_level"]

            params.add("continuum_exponent", value=self.continuum_exponent, vary=True)
            return params

    return NewWaveCalParams
