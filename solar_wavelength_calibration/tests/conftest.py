import astropy.units as u
import numpy as np
import pytest
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
def test_unfit_1d_spectrum():
    return np.linspace(0, 1, 1000)


@pytest.fixture(scope="session")
def test_dispersion():
    """Dummy dispersion value."""
    return 0.004401829598401599 * u.nm


@pytest.fixture(scope="session")
def test_incident_light_angle():
    """Dummy incident_light_angle value."""
    return 0.9949384642711658 * u.rad


@pytest.fixture(scope="session")
def test_grating_constant():
    """Dummy grating constant."""
    return 31600.0 * u.mm ** (-1)


@pytest.fixture(scope="session")
def test_doppler_velocity():
    """Dummy doppler velocity value."""
    return -0.4281437757265212 * u.km / u.s


@pytest.fixture(scope="session")
def test_order():
    """Dummy order."""
    return 52


@pytest.fixture(scope="session")
def test_parameters_dict(
    atlas: AtlasBase,
    test_unfit_1d_spectrum,
    test_dispersion,
    test_incident_light_angle,
    test_grating_constant,
    test_doppler_velocity,
    test_order,
):
    return {
        "crval1": atlas.telluric_atlas_wavelength[np.size(test_unfit_1d_spectrum) // 2 + 1],
        "dispersion": test_dispersion,
        "incident_light_angle": test_incident_light_angle,
        "resolving_power": 72500,
        "opacity_factor": 5.0,
        "straylight_fraction": 0.25,
        "unfit_1d_spectrum": test_unfit_1d_spectrum,
        "grating_constant": test_grating_constant,
        "doppler_velocity": test_doppler_velocity,
        "order": test_order,
    }


@pytest.fixture(scope="session")
def test_only_required_parameters_dict(
    atlas: AtlasBase,
    test_unfit_1d_spectrum,
    test_dispersion,
    test_incident_light_angle,
    test_grating_constant,
    test_doppler_velocity,
    test_order,
):
    return {
        "crval1": atlas.telluric_atlas_wavelength[np.size(test_unfit_1d_spectrum) // 2 + 1],
        "dispersion": test_dispersion,
        "incident_light_angle": test_incident_light_angle,
        "unfit_1d_spectrum": test_unfit_1d_spectrum,
        "grating_constant": test_grating_constant,
        "doppler_velocity": test_doppler_velocity,
        "order": test_order,
    }


@pytest.fixture(scope="session")
def parameters(
    test_unfit_1d_spectrum,
    test_dispersion,
    test_incident_light_angle,
    test_grating_constant,
    test_doppler_velocity,
    test_order,
    atlas: AtlasBase,
) -> Parameters:
    params = WavelengthCalibrationParameters(
        crval1=atlas.telluric_atlas_wavelength[np.size(test_unfit_1d_spectrum) // 2 + 1],
        dispersion=test_dispersion,
        incident_light_angle=test_incident_light_angle,
        resolving_power=72500,
        opacity_factor=5.0,
        straylight_fraction=0.25,
        unfit_1d_spectrum=test_unfit_1d_spectrum,
        grating_constant=test_grating_constant,
        doppler_velocity=test_doppler_velocity,
        order=test_order,
    )
    return params
