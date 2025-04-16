import astropy.units as u
import numpy as np
import pytest
from astropy.wcs import WCS
from lmfit import Parameters
from scipy.interpolate import interp1d

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
def test_expected_wavelength_vector(atlas):
    """Use a chunk of the telluric atlas to simulate an expected wavelength vector."""
    return atlas.telluric_atlas_wavelength[500:2000]


@pytest.fixture(scope="session")
def test_unfit_1d_spectrum(
    atlas, test_grating_constant, test_order, test_dispersion, test_incident_light_angle
):
    """Use a chunk of the telluric atlas and shift and scale by random numbers and resample onto a known WCS-based wavelength grid to simulate an unfit 1d spectra."""
    telluric_atlas = atlas.telluric_atlas_transmission
    telluric_atlas_shifted = telluric_atlas + 30.0  # shift by a random amount
    telluric_atlas_scaled = telluric_atlas_shifted * 2.3  # scale by a random amount

    # Create a dummy WCS object for the known wavelength grid
    non_linear_header = {
        "CTYPE1": "AWAV-GRA",  # Grating dispersion function with air wavelengths
        "CUNIT1": "nm",  # Dispersion units
        "CRPIX1": np.size(telluric_atlas) // 2 + 1,  # Reference pixel [pix]
        "PV1_0": test_grating_constant.value,  # Grating density
        "PV1_1": test_order,  # Diffraction order
        "CRVAL1": atlas.telluric_atlas_wavelength[
            np.size(telluric_atlas) // 2 + 1
        ].value,  # Reference value [nm] (<<<< TO BE OPTIMIZED <<<<<<)
        "CDELT1": test_dispersion.value,  # Linear dispersion [nm/pix] (<<<< TO BE OPTIMIZED <<<<<<)
        "PV1_2": test_incident_light_angle.value,  # Incident angle [deg] (<<<< TO BE OPTIMIZED <<<<<<)
    }

    non_linear_wcs = WCS(non_linear_header)
    # Generate the wavelength grid from the WCS object
    linear_wave = (  # linear_wavelength_vector
        non_linear_wcs.spectral.pixel_to_world(np.arange(np.size(telluric_atlas)))
    ).to_value(u.nm)
    print(linear_wave)

    # Interpolate the unfit spectrum onto the new wavelength grid
    interpolator = interp1d(
        atlas.telluric_atlas_wavelength,
        telluric_atlas_scaled,
        bounds_error=False,
        fill_value="extrapolate",
    )
    resampled_spectrum = interpolator(linear_wave)

    return resampled_spectrum[500:2000]


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
