import numpy as np
import pytest
from lmfit import Parameters
from pydantic import ValidationError

from solar_wavelength_calibration.atlas.base import AtlasBase
from solar_wavelength_calibration.fitter.parameters import WavelengthCalibrationParameters


def test_incorrect_parameter_types():
    """
    Given: Incorrect input parameter value types
    When: Initializing a WavelengthCalibrationParameters object with these incorrect inputs
    Then: Raise a ValidationError
    """
    with pytest.raises(ValidationError) as excinfo:
        WavelengthCalibrationParameters(
            crval1=3,
            dispersion="shouldbeafloat",
            incident_light_angle=154,
            resolving_power="shouldbeanint",
            opacity_factor=5,
            straylight_fraction=2,
            unfit_1d_spectrum="shouldbeanarray",
            grating_constant=5,
            doppler_shift=12,
            order=1.5,
        )
    assert "Input should be a valid" in str(excinfo.value)


def test_missing_parameters(
    atlas: AtlasBase, test_unfit_1d_spectrum, parameters, test_only_required_parameters_dict
):
    """
    Given: Only the calculated crval1 input parameter value
    When: Initializing a WavelengthCalibrationParameters object with these inputs
    Then: Raise a ValidationError
    """
    with pytest.raises(ValidationError) as excinfo:
        WavelengthCalibrationParameters(
            crval1=atlas.telluric_atlas_wavelength[np.size(test_unfit_1d_spectrum) // 2 + 1]
        )

    del test_only_required_parameters_dict["crval1"]  # removing this because we gave it as an input
    # check that all of the required inputs are reported to be missing
    for key in test_only_required_parameters_dict.keys():
        assert key in str(excinfo.value)


def test_lmfit_parameters_was_set_up_correctly(test_parameters_dict):
    """
    Given: A dictionary of input parameter values
    When: Initializing a WavelengthCalibrationParameters object
    Then: The lmfit parameters were set up correctly
    """
    input_params = WavelengthCalibrationParameters(
        crval1=test_parameters_dict.get("crval1"),
        dispersion=test_parameters_dict.get("dispersion"),
        incident_light_angle=test_parameters_dict.get("incident_light_angle"),
        resolving_power=test_parameters_dict.get("resolving_power"),
        opacity_factor=test_parameters_dict.get("opacity_factor"),
        straylight_fraction=test_parameters_dict.get("straylight_fraction"),
        unfit_1d_spectrum=test_parameters_dict.get("unfit_1d_spectrum"),
        grating_constant=test_parameters_dict.get("grating_constant"),
        doppler_shift=test_parameters_dict.get("doppler_shift"),
        order=test_parameters_dict.get("order"),
    )

    assert isinstance(input_params.lmfit_parameters, Parameters)
    # make sure than only 7 parameters were added into lmfit_parameters [crval1, dispersion, incident_light_angle, resolving_power, opacity_factor, straylight_fraction, unfit_1d_spectrum]
    assert len(input_params.lmfit_parameters.valuesdict()) == 7


def test_default_parameters(atlas: AtlasBase, test_parameters_dict):
    """
    Given: The required input parameter values
    When: Resolving power, opacity factor, and straylight fraction are not given and dafault values are used
    Then: The initialized parameters are returned and the non-required parameter values are set to their default values
    """
    initialized_parameters = WavelengthCalibrationParameters(
        crval1=test_parameters_dict.get("crval1"),
        dispersion=test_parameters_dict.get("dispersion"),
        incident_light_angle=test_parameters_dict.get("incident_light_angle"),
        unfit_1d_spectrum=test_parameters_dict.get("unfit_1d_spectrum"),
        grating_constant=test_parameters_dict.get("grating_constant"),
        doppler_shift=test_parameters_dict.get("doppler_shift"),
        order=test_parameters_dict.get("order"),
    )

    # make sure that the non-required values are set to their default values
    initialized_params_dict = initialized_parameters.lmfit_parameters.valuesdict()
    assert initialized_params_dict["resolving_power"] == 1
    assert initialized_params_dict["opacity_factor"] == 1
    assert initialized_params_dict["straylight_fraction"] == 0
    # make sure than 7 parameters were added into lmfit_parameters [crval1, dispersion, incident_light_angle, resolving_power, opacity_factor, straylight_fraction, unfit_1d_spectrum]
    assert len(initialized_parameters.lmfit_parameters.valuesdict()) == 7
