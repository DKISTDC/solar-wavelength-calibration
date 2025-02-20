import astropy.units as u
import numpy as np
import pytest
from lmfit import Parameters

from solar_wavelength_calibration.atlas.atlas import Atlas
from solar_wavelength_calibration.atlas.atlas import DownloadConfig
from solar_wavelength_calibration.atlas.base import AtlasBase
from solar_wavelength_calibration.fitter.parameters import WavelengthCalibrationParameters
from solar_wavelength_calibration.fitter.wavelength_fitter import WaveCalFitter


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
    return 31600.0


@pytest.fixture(scope="session")
def test_doppler_shift():
    """Dummy doppler shift value."""
    return -0.4281437757265212


@pytest.fixture(scope="session")
def test_order():
    """Dummy order."""
    return 52


@pytest.fixture(scope="session")
def parameters(
    test_unfit_1d_spectrum,
    test_dispersion,
    test_incident_light_angle,
    test_grating_constant,
    test_doppler_shift,
    test_order,
    atlas: AtlasBase,
) -> Parameters:
    params = WavelengthCalibrationParameters(
        crval1=atlas.telluric_atlas_wavelength[np.size(test_unfit_1d_spectrum) // 2 + 1],
        dispersion=test_dispersion.value,
        incident_light_angle=test_incident_light_angle.value,
        resolving_power=72500,
        opacity_factor=5.0,
        straylight_fraction=0.25,
        unfit_1d_spectrum=test_unfit_1d_spectrum,
        grating_constant=test_grating_constant,
        doppler_shift=test_doppler_shift,
        order=test_order,
    )
    return params


def test_wavecal_fitter(
    parameters: WavelengthCalibrationParameters,
    test_unfit_1d_spectrum,
    atlas: AtlasBase,
    test_dispersion,
):
    """
    Given:
    When: Initializing a WaveCalFitter object with these inputs
    Then: Everything needed for the fit is correctly set up
    """
    test_expected_wavelength_vector = np.linspace(1500, 2000, 1000) * u.nm
    fitter = WaveCalFitter(
        input_parameters=parameters,
        atlas=atlas,
        expected_wavelength_vector=test_expected_wavelength_vector,
        unfit_1d_spectrum=test_unfit_1d_spectrum,
    )

    # test that values have been fit (i.e. changed)
    assert fitter.fit_result_params.get("dispersion") != test_dispersion
    assert (
        fitter.fit_result_params.get("crval1")
        != atlas.telluric_atlas_wavelength[np.size(test_unfit_1d_spectrum) // 2 + 1]
    )

    # test that the same amount of parameters we get out of the fit is the same number that we put in
    assert len(fitter.fit_result_params) == len(parameters.lmfit_parameters)

    # TODO: test that wavecal outputs are what we expected


def test_incorrect_atlas():
    pass


def test_dispersion_not_fit():
    pass


def test_expected_wavelength_and_unfit_spectrum_different_lengths():
    pass


def test_fit_is_correct(atlas: AtlasBase):
    # shift atlas by known amount
    shift = 1.70856 * u.nm
    shifted_atlas = atlas.solar_atlas_wavelength + shift
    # add noise
    # do a fit
    WaveCalFitter
    # see if you get your shift back


def test_prepare_atlas(
    parameters: WavelengthCalibrationParameters, test_unfit_1d_spectrum, atlas: AtlasBase
):
    test_expected_wavelength_vector = np.linspace(1500, 2000, 1000) * u.nm
    fitter = WaveCalFitter(
        input_parameters=parameters,
        atlas=atlas,
        expected_wavelength_vector=test_expected_wavelength_vector,
        unfit_1d_spectrum=test_unfit_1d_spectrum,
    )
    fts_wavelength, fts_solar, fts_telluric = fitter.prepare_atlas()

    # test that we took out a chunk of the atlas
    assert len(fts_solar) < len(atlas.solar_atlas_transmission)
    assert len(fts_telluric) < len(atlas.telluric_atlas_transmission)

    # test that we took out the correct chunk (i.e. mean of expected wavelength vector is contained in fts_wavelength)
    assert (
        fts_wavelength[0]
        < np.mean(test_expected_wavelength_vector).value
        < fts_wavelength[len(fts_wavelength) - 1]
    )
