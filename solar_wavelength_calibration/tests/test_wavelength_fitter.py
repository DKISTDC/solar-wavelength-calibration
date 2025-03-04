import astropy.units as u
import numpy as np
import pytest

from solar_wavelength_calibration.atlas.base import AtlasBase
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
    assert fitter.fit_result.get("dispersion") != test_dispersion
    assert (
        fitter.fit_result.get("crval1")
        != atlas.telluric_atlas_wavelength[np.size(test_unfit_1d_spectrum) // 2 + 1]
    )

    # test that the same amount of parameters we get out of the fit is the same number that we put in
    assert len(fitter.fit_result) == len(parameters.lmfit_parameters)


def test_incorrect_atlas(parameters: WavelengthCalibrationParameters, test_unfit_1d_spectrum):
    """
    Given: An incorrect atlas (i.e. a random array of numbers)
    When: Initializing a WaveCalFitter object
    Then: Raise a ValidationError
    """
    test_expected_wavelength_vector = np.linspace(1500, 2000, 1000) * u.nm
    atlas = np.linspace(1000, 3000, 1000) * u.nm
    with pytest.raises(ValueError, match="Input should be an instance of AtlasBase"):
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
        doppler_shift=test_parameters_dict["doppler_shift"],
        order=test_parameters_dict["order"],
        fit_flags={"dispersion": False},
    )

    test_expected_wavelength_vector = np.linspace(1500, 2000, 1000) * u.nm
    fitter = WaveCalFitter(
        input_parameters=initialized_parameters,
        atlas=atlas,
        expected_wavelength_vector=test_expected_wavelength_vector,
        unfit_1d_spectrum=test_unfit_1d_spectrum,
    )

    # make sure that the dispersion was not fit (wasn't changed)
    assert test_parameters_dict["dispersion"].value == fitter.fit_result["dispersion"]


def test_expected_wavelength_and_unfit_spectrum_different_lengths(
    parameters: WavelengthCalibrationParameters, test_unfit_1d_spectrum, atlas: AtlasBase
):
    """
    Given: An expected_wavelength_vector and unfit_1d_spectrum vector that are not of the same size
    When: Initializing a WaveCalFitter object
    Then: Raise an error
    """
    test_expected_wavelength_vector_small = np.linspace(1500, 2000, 100) * u.nm
    with pytest.raises(ValueError, match="operands could not be broadcast together"):
        fitter = WaveCalFitter(
            input_parameters=parameters,
            atlas=atlas,
            expected_wavelength_vector=test_expected_wavelength_vector_small,
            unfit_1d_spectrum=test_unfit_1d_spectrum,
        )
        fitter.fit_result


def test_fit_is_correct(
    parameters: WavelengthCalibrationParameters, test_unfit_1d_spectrum, atlas: AtlasBase
):
    """
    Given:
    When:
    Then:
    """
    test_expected_wavelength_vector = np.linspace(1500, 2000, 1000) * u.nm
    fitter = WaveCalFitter(
        input_parameters=parameters,
        atlas=atlas,
        expected_wavelength_vector=test_expected_wavelength_vector,
        unfit_1d_spectrum=test_unfit_1d_spectrum,
    )
    fitter.fit_result


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
