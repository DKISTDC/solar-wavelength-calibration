import astropy
import astropy.units as u
import numpy as np
import pytest
from astropy.units import Quantity
from astropy.wcs import WCS
from pydantic import ValidationError

from solar_wavelength_calibration.atlas.base import AtlasBase
from solar_wavelength_calibration.fitter.parameters import BoundRange
from solar_wavelength_calibration.fitter.parameters import BoundsModel
from solar_wavelength_calibration.fitter.parameters import FitFlagsModel
from solar_wavelength_calibration.fitter.parameters import WavelengthCalibrationParameters
from solar_wavelength_calibration.fitter.wavelength_fitter import WaveCalFitter


def test_wavecal_fitter(
    parameters: WavelengthCalibrationParameters,
    test_unfit_1d_spectrum,
    atlas: AtlasBase,
    test_dispersion,
    test_parameters_dict,
):
    """
    Given: The correct WaveCalFitter inputs (data, models, and fit parameters)
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
    assert fitter.fit_result.wcs.cdelt[0] != test_dispersion
    assert (
        fitter.fit_result.wcs.crval[0]
        != atlas.telluric_atlas_wavelength[np.size(test_unfit_1d_spectrum) // 2 + 1]
    )


def test_incorrect_atlas(parameters: WavelengthCalibrationParameters, test_unfit_1d_spectrum):
    """
    Given: An incorrect atlas (i.e. a random array of numbers)
    When: Initializing a WaveCalFitter object
    Then: Raise a ValidationError
    """
    test_expected_wavelength_vector = np.linspace(1500, 2000, 1000) * u.nm
    atlas = np.linspace(1000, 3000, 1000) * u.nm
    with pytest.raises(ValidationError, match="Input should be an instance of AtlasBase"):
        fitter = WaveCalFitter(
            input_parameters=parameters,
            atlas=atlas,
            expected_wavelength_vector=test_expected_wavelength_vector,
            unfit_1d_spectrum=test_unfit_1d_spectrum,
        )
        fitter.fit_result


def test_dispersion_not_fit(test_parameters_dict, test_unfit_1d_spectrum, atlas: AtlasBase):
    """
    Given: Correct input parameters, but we don't want to fit the dispersion
    When: Initializing a WaveCalFitter object
    Then: The dispersion value has not changed
    """
    initialized_parameters = WavelengthCalibrationParameters(
        crval1=test_parameters_dict["crval1"],
        dispersion=test_parameters_dict["dispersion"],
        incident_light_angle=test_parameters_dict["incident_light_angle"],
        resolving_power=test_parameters_dict["resolving_power"],
        opacity_factor=test_parameters_dict["opacity_factor"],
        straylight_fraction=test_parameters_dict["straylight_fraction"],
        unfit_1d_spectrum=test_parameters_dict["unfit_1d_spectrum"],
        grating_constant=test_parameters_dict["grating_constant"],
        doppler_velocity=test_parameters_dict["doppler_velocity"],
        order=test_parameters_dict["order"],
        fit_flags=FitFlagsModel(dispersion=False),
    )

    test_expected_wavelength_vector = np.linspace(1500, 2000, 1000) * u.nm
    fitter = WaveCalFitter(
        input_parameters=initialized_parameters,
        atlas=atlas,
        expected_wavelength_vector=test_expected_wavelength_vector,
        unfit_1d_spectrum=test_unfit_1d_spectrum,
        full=True,
    )

    # make sure that the dispersion was not fit (wasn't changed)
    wcs_fit_result_header, fit_result_params = fitter.fit_result
    assert test_parameters_dict["dispersion"].value == fit_result_params["dispersion"]


@pytest.fixture
def num_wave_pix() -> int:
    return 987


@pytest.fixture
def test_crval1() -> Quantity:
    return 1072 * u.nm


@pytest.fixture
def test_WCS_header(
    test_crval1, test_dispersion, test_incident_light_angle, test_grating_constant, test_order
) -> dict:
    return {
        "CTYPE1": "AWAV-GRA",
        "CUNIT1": "nm",
        "CRPIX1": 1,
        "CRVAL1": test_crval1.to_value("nm"),
        "CDELT1": test_dispersion.to_value("nm / pix"),
        "PV1_0": test_grating_constant.to_value("1 / mm"),
        "PV1_1": test_order,
        "PV1_2": test_incident_light_angle.to_value("deg"),
    }


@pytest.fixture
def observed_wave(test_WCS_header, num_wave_pix) -> np.ndarray:
    wcs = WCS(test_WCS_header)

    obs_wave = wcs.spectral.pixel_to_world(np.arange(num_wave_pix)).to_value("nm")

    return obs_wave


@pytest.fixture
def observed_spec(atlas, observed_wave) -> np.ndarray:
    obs_spec = np.interp(
        observed_wave, atlas.solar_atlas_wavelength.to_value("nm"), atlas.solar_atlas_transmission
    )

    return obs_spec


def test_fit_is_correct(
    test_crval1,
    test_dispersion,
    test_incident_light_angle,
    test_grating_constant,
    test_order,
    observed_spec,
    observed_wave,
    atlas: AtlasBase,
    test_parameters_dict,
):
    """
    Given: A simulated unfit 1D spectrum with a known shift and dispersion scale.
    When: The WaveCalFitter is used to fit the spectrum.
    Then: The fitted dispersion (CDELT1) is approximately equal to the expected dispersion.
    """

    parameters = WavelengthCalibrationParameters(
        crval1=test_crval1 + 1.0 * u.nm,
        dispersion=test_dispersion * 1.01,
        incident_light_angle=test_incident_light_angle * 1.01,
        resolving_power=72500,
        opacity_factor=1,
        straylight_fraction=0.0,
        unfit_1d_spectrum=observed_spec,
        grating_constant=test_grating_constant,
        doppler_velocity=0 * u.km / u.s,
        order=test_order,
        bounds=BoundsModel(
            crval1=BoundRange(min=1070 * u.nm, max=1080 * u.nm),
            dispersion=BoundRange(min=0 * u.nm / u.pix, max=np.inf * u.nm / u.pix),
            incident_light_angle=BoundRange(min=-360 * u.deg, max=360 * u.deg),
            opacity_factor=BoundRange(min=0),
            straylight_fraction=BoundRange(min=0),
            resolving_power=BoundRange(min=0),
        ),
    )

    fitter = WaveCalFitter(
        input_parameters=parameters,
        atlas=atlas,
        expected_wavelength_vector=observed_wave * u.nm,
        unfit_1d_spectrum=observed_spec,
    )
    assert np.isclose(fitter.fit_result.wcs.crval[0], test_crval1.to_value("m"))
    # TODO: test other fit parameters


def test_prepare_atlas(
    parameters: WavelengthCalibrationParameters, test_unfit_1d_spectrum, atlas: AtlasBase
):
    """
    Given: The correct WaveCalFitter inputs (data, models, and fit parameters)
    When: Preparing the atlas for the fit
    Then: Make sure that a sample of the atlas is taken, and make sure that it is the correct sample
    """
    test_expected_wavelength_vector = np.linspace(1500, 2000, 1000) * u.nm
    fitter = WaveCalFitter(
        input_parameters=parameters,
        atlas=atlas,
        expected_wavelength_vector=test_expected_wavelength_vector,
        unfit_1d_spectrum=test_unfit_1d_spectrum,
    )
    fts_solar_wavelength, fts_telluric_wavelength, fts_solar, fts_telluric = fitter.prepare_atlas()

    # test that we took out a chunk of the atlas
    assert len(fts_solar) < len(atlas.solar_atlas_transmission)
    assert len(fts_telluric) < len(atlas.telluric_atlas_transmission)

    # test that we took out the correct chunk for each atlas (i.e. mean of expected wavelength vector is contained in fts_[solar/telluric]wavelength)
    assert (
        fts_solar_wavelength[0]
        < np.mean(test_expected_wavelength_vector).value
        < fts_solar_wavelength[len(fts_solar_wavelength) - 1]
    )

    assert (
        fts_telluric_wavelength[0]
        < np.mean(test_expected_wavelength_vector).value
        < fts_telluric_wavelength[len(fts_telluric_wavelength) - 1]
    )


def test_full_flag_true(
    parameters: WavelengthCalibrationParameters, test_unfit_1d_spectrum, atlas: AtlasBase
):
    """
    Given: WaveCal inputs with the full flag set to true
    When: Initializing a WaveCalFitter object
    Then: We get two fit results back; one WCS object, and one dictionary
    """
    test_expected_wavelength_vector = np.linspace(1500, 2000, 1000) * u.nm
    fitter = WaveCalFitter(
        input_parameters=parameters,
        atlas=atlas,
        expected_wavelength_vector=test_expected_wavelength_vector,
        unfit_1d_spectrum=test_unfit_1d_spectrum,
        full=True,
    )

    # make sure we get 2 outputs
    assert len(fitter.fit_result) == 2
    wcs_fit_result_header, fit_result_params = fitter.fit_result
    # and make sure that the first output is the WCS object...
    assert isinstance(wcs_fit_result_header, astropy.wcs.wcs.WCS)
    # and the second is a dictionary containing the rest of the fitted parameters
    assert isinstance(fit_result_params, dict)


def test_full_flag_false(
    parameters: WavelengthCalibrationParameters, test_unfit_1d_spectrum, atlas: AtlasBase
):
    """
    Given: WaveCal inputs with the full flag set to false (default)
    When: Initializing a WaveCalFitter object
    Then: We get one fit result back, and it is a WCS object
    """
    test_expected_wavelength_vector = np.linspace(1500, 2000, 1000) * u.nm
    fitter = WaveCalFitter(
        input_parameters=parameters,
        atlas=atlas,
        expected_wavelength_vector=test_expected_wavelength_vector,
        unfit_1d_spectrum=test_unfit_1d_spectrum,
        full=False,
    )

    # make sure we get one fit result back, and it is a WCS object
    assert isinstance(fitter.fit_result, astropy.wcs.wcs.WCS)
