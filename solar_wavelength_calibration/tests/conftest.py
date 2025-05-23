import astropy.units as u
import numpy as np
import pytest
from astropy.units import Quantity
from astropy.wcs import WCS
from lmfit import Parameters

from solar_wavelength_calibration.atlas.atlas import Atlas
from solar_wavelength_calibration.atlas.atlas import DownloadConfig
from solar_wavelength_calibration.atlas.base import AtlasBase
from solar_wavelength_calibration.fitter.parameters import WavelengthCalibrationParameters


@pytest.fixture(
    params=[
        pytest.param(None, id="default_config"),
    ],
    scope="session",
)
def config(request) -> DownloadConfig | None:
    return request.param


@pytest.fixture(scope="session")
def atlas(config: DownloadConfig | None) -> Atlas:
    return Atlas(config=config)


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
def observed_spectrum(atlas, observed_wavelength_vector) -> np.ndarray:
    obs_spec = np.interp(
        observed_wavelength_vector.to_value("nm"),
        atlas.solar_atlas_wavelength.to_value("nm"),
        atlas.solar_atlas_transmission,
    )

    return obs_spec


@pytest.fixture(scope="session")
def observed_continuum_level(observed_spectrum) -> float:
    return np.nanpercentile(observed_spectrum, 75)


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
        "input_spectrum": observed_spectrum,
        "grating_constant": test_grating_constant,
        "doppler_velocity": test_doppler_velocity,
        "order": test_order,
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
) -> Parameters:
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
