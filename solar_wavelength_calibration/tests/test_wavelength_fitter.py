import astropy
import astropy.units as u
import numpy as np
import pytest
from astropy.wcs import WCS
from pydantic import ValidationError

from solar_wavelength_calibration.atlas.base import AtlasBase
from solar_wavelength_calibration.fitter.parameters import FitFlagsModel
from solar_wavelength_calibration.fitter.parameters import WavelengthCalibrationParameters
from solar_wavelength_calibration.fitter.wavelength_fitter import WaveCalFitter


def test_wavecal_fitter(
    parameters: WavelengthCalibrationParameters,
    observed_spectrum,
    atlas: AtlasBase,
    test_dispersion,
    test_parameters_dict,
    observed_wavelength_vector,
):
    """
    Given: The correct WaveCalFitter inputs (data, models, and fit parameters)
    When: Initializing a WaveCalFitter object with these inputs
    Then: Everything needed for the fit is correctly set up
    """
    fitter = WaveCalFitter(
        input_parameters=parameters,
        atlas=atlas,
        expected_wavelength_vector=observed_wavelength_vector,
        unfit_1d_spectrum=observed_spectrum,
    )

    print(fitter.fit_result)
    # test that values have been fit (i.e. changed)
    """assert fitter.fit_result.wcs.cdelt[0] != test_dispersion
    assert (
        fitter.fit_result.wcs.crval[0]
        != atlas.telluric_atlas_wavelength[np.size(test_unfit_1d_spectrum) // 2 + 1]
    )"""


def test_incorrect_atlas(
    parameters: WavelengthCalibrationParameters,
    observed_spectrum,
    observed_wavelength_vector,
):
    """
    Given: An incorrect atlas (i.e. a random array of numbers)
    When: Initializing a WaveCalFitter object
    Then: Raise a ValidationError
    """
    atlas = np.linspace(1000, 3000, 1000) * u.nm
    with pytest.raises(ValidationError, match="Input should be an instance of AtlasBase"):
        fitter = WaveCalFitter(
            input_parameters=parameters,
            atlas=atlas,
            expected_wavelength_vector=observed_wavelength_vector,
            unfit_1d_spectrum=observed_spectrum,
        )
        fitter.fit_result


def test_dispersion_not_fit(
    test_parameters_dict, observed_spectrum, atlas: AtlasBase, observed_wavelength_vector
):
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

    fitter = WaveCalFitter(
        input_parameters=initialized_parameters,
        atlas=atlas,
        expected_wavelength_vector=observed_wavelength_vector,
        unfit_1d_spectrum=observed_spectrum,
        full=True,
    )

    # make sure that the dispersion was not fit (wasn't changed)
    wcs_fit_result_header, fit_result_params = fitter.fit_result
    assert test_parameters_dict["dispersion"].value == fit_result_params["dispersion"]


def test_fit_is_correct(
    test_crval1,
    test_dispersion,
    test_incident_light_angle,
    test_grating_constant,
    test_order,
    observed_spectrum,
    observed_wavelength_vector,
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
        unfit_1d_spectrum=observed_spectrum,
        grating_constant=test_grating_constant,
        doppler_velocity=100 * u.km / u.s,
        order=test_order,
    )

    fitter = WaveCalFitter(
        input_parameters=parameters,
        atlas=atlas,
        expected_wavelength_vector=observed_wavelength_vector,
        unfit_1d_spectrum=observed_spectrum,
    )
    assert np.isclose(fitter.fit_result.wcs.crval[0], test_crval1.to_value("m"))
    assert np.isclose(fitter.fit_result.wcs.cdelt[0], test_dispersion.to_value(u.m / u.pix))

    fit_incident_light_angle = [
        pvi[2] for pvi in fitter.fit_result.wcs.get_pv() if pvi[0] == 1 and pvi[1] == 2
    ][0]
    # Incident light angle is pretty weakly constrained for our test setup so let's say we pass if we get within a
    # degree.
    assert abs(fit_incident_light_angle - test_incident_light_angle.to_value(u.deg)) < 1


def test_prepare_atlas(
    parameters: WavelengthCalibrationParameters,
    observed_spectrum,
    atlas: AtlasBase,
    observed_wavelength_vector,
):
    """
    Given: The correct WaveCalFitter inputs (data, models, and fit parameters)
    When: Preparing the atlas for the fit
    Then: Make sure that a sample of the atlas is taken, and make sure that it is the correct sample
    """
    fitter = WaveCalFitter(
        input_parameters=parameters,
        atlas=atlas,
        expected_wavelength_vector=observed_wavelength_vector,
        unfit_1d_spectrum=observed_spectrum,
    )
    fts_solar_wavelength, fts_telluric_wavelength, fts_solar, fts_telluric = fitter.prepare_atlas()

    # test that we took out a chunk of the atlas
    assert len(fts_solar) < len(atlas.solar_atlas_transmission)
    assert len(fts_telluric) < len(atlas.telluric_atlas_transmission)

    # test that we took out the correct chunk for each atlas (i.e. mean of expected wavelength vector is contained in fts_[solar/telluric]wavelength)
    assert (
        fts_solar_wavelength[0]
        < np.mean(observed_wavelength_vector).value
        < fts_solar_wavelength[len(fts_solar_wavelength) - 1]
    )

    assert (
        fts_telluric_wavelength[0]
        < np.mean(observed_wavelength_vector).value
        < fts_telluric_wavelength[len(fts_telluric_wavelength) - 1]
    )


def test_full_flag_true(
    parameters: WavelengthCalibrationParameters,
    observed_spectrum,
    atlas: AtlasBase,
    observed_wavelength_vector,
):
    """
    Given: WaveCal inputs with the full flag set to true
    When: Initializing a WaveCalFitter object
    Then: We get two fit results back; one WCS object, and one dictionary
    """
    fitter = WaveCalFitter(
        input_parameters=parameters,
        atlas=atlas,
        expected_wavelength_vector=observed_wavelength_vector,
        unfit_1d_spectrum=observed_spectrum,
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
    parameters: WavelengthCalibrationParameters,
    observed_spectrum,
    atlas: AtlasBase,
    observed_wavelength_vector,
):
    """
    Given: WaveCal inputs with the full flag set to false (default)
    When: Initializing a WaveCalFitter object
    Then: We get one fit result back, and it is a WCS object
    """
    fitter = WaveCalFitter(
        input_parameters=parameters,
        atlas=atlas,
        expected_wavelength_vector=observed_wavelength_vector,
        unfit_1d_spectrum=observed_spectrum,
        full=False,
    )

    # make sure we get one fit result back, and it is a WCS object
    assert isinstance(fitter.fit_result, astropy.wcs.wcs.WCS)
