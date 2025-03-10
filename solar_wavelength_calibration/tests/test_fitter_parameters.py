import astropy.units as u
import numpy as np
import pytest
from lmfit import Parameters
from pydantic import ValidationError

from solar_wavelength_calibration.atlas.base import AtlasBase
from solar_wavelength_calibration.fitter.parameters import WavelengthCalibrationParameters


def test_incorrect_parameter_types(test_unfit_1d_spectrum):
    """
    Given: Incorrect input parameter value types
    When: Initializing a WavelengthCalibrationParameters object with these incorrect inputs
    Then: Raise a ValidationError
    """
    with pytest.raises(TypeError) as excinfo:
        WavelengthCalibrationParameters(
            crval1=3,
            dispersion="shouldbeafloat",
            incident_light_angle=154 * u.nm,
            resolving_power="shouldbeanint",
            opacity_factor=5,
            straylight_fraction=2,
            unfit_1d_spectrum=test_unfit_1d_spectrum,
            grating_constant=5,
            doppler_velocity=12,
            order=1.5,
        )
    assert "must be an astropy Quantity" in str(excinfo.value)


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
    )

    assert isinstance(input_params.lmfit_parameters, Parameters)
    # make sure than only 7 parameters were added into lmfit_parameters [crval1, dispersion, incident_light_angle, resolving_power, opacity_factor, straylight_fraction, unfit_1d_spectrum]
    assert len(input_params.lmfit_parameters.valuesdict()) == 7


def test_default_parameters(atlas: AtlasBase, test_parameters_dict):
    """
    Given: The required input parameter values
    When: Resolving power, opacity factor, and straylight fraction are not given and default values are used
    Then: The initialized parameters are returned and the non-required parameter values are set to their default values
    """
    initialized_parameters = WavelengthCalibrationParameters(
        crval1=test_parameters_dict.get("crval1"),
        dispersion=test_parameters_dict.get("dispersion"),
        incident_light_angle=test_parameters_dict.get("incident_light_angle"),
        unfit_1d_spectrum=test_parameters_dict.get("unfit_1d_spectrum"),
        grating_constant=test_parameters_dict.get("grating_constant"),
        doppler_velocity=test_parameters_dict.get("doppler_velocity"),
        order=test_parameters_dict.get("order"),
    )

    # make sure that the non-required values are set to their default values
    initialized_params_dict = initialized_parameters.lmfit_parameters.valuesdict()
    assert initialized_params_dict["resolving_power"] == 1
    assert initialized_params_dict["opacity_factor"] == 1
    assert initialized_params_dict["straylight_fraction"] == 0
    # make sure than 7 parameters were added into lmfit_parameters [crval1, dispersion, incident_light_angle, resolving_power, opacity_factor, straylight_fraction, unfit_1d_spectrum]
    assert len(initialized_parameters.lmfit_parameters.valuesdict()) == 7


def test_default_bounds(test_parameters_dict):
    """
    Given: No custom input bounds
    When: Initializing the fit parameters
    Then: Each parameter's bound is the default value
    """

    input_params = WavelengthCalibrationParameters(
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
    )

    assert input_params._get_bounds("crval1") == [-0.5 * u.nm, 0.5 * u.nm]
    assert input_params._get_bounds("dispersion") == [-0.05 * u.nm, 0.05 * u.nm]
    assert input_params._get_bounds("incident_light_angle") == [-5 * u.rad, 5 * u.rad]
    assert input_params._get_bounds("resolving_power") == [-50000, 50000]
    assert input_params._get_bounds("opacity_factor") == [-5, 5]
    assert input_params._get_bounds("straylight_fraction") == [-0.25, 0.25]
    assert input_params._get_bounds("continuum_intensity_correction") == [0.8, 1.2]


def test_custom_bounds(test_parameters_dict):
    """
    Given: Some custom input bounds
    When: Initializing the fit parameters
    Then: The customized input bounds are set to the custom values and others are set to the default values
    """
    bounds = {
        "crval1": [-0.3, 0.3] * u.nm,
        "dispersion": [-55, 55] * u.nm,
        "incident_light_angle": [-5, 103] * u.rad,
        "resolving_power": [-50, 0.0001],
        "opacity_factor": [-1, 1],
    }

    input_params = WavelengthCalibrationParameters(
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
        bounds=bounds,
    )

    assert input_params._get_bounds("crval1") == [-0.3 * u.nm, 0.3 * u.nm]
    assert input_params._get_bounds("dispersion") == [-55 * u.nm, 55 * u.nm]
    assert input_params._get_bounds("incident_light_angle") == [-5 * u.rad, 103 * u.rad]
    assert input_params._get_bounds("resolving_power") == [-50, 0.0001]
    assert input_params._get_bounds("opacity_factor") == [-1, 1]
    assert input_params._get_bounds("straylight_fraction") == [-0.25, 0.25]  # default bounds
    assert input_params._get_bounds("continuum_intensity_correction") == [
        0.8,
        1.2,
    ]  # default bounds


def test_bounds_have_no_units(test_parameters_dict):
    """
    Given: Some custom input bounds with no units
    When: Initializing the fit parameters
    Then: A ValidationError is raised
    """
    bounds = {
        "crval1": [-0.3, 0.3],
        "dispersion": [-55, 55],
        "incident_light_angle": [-5, 103],
        "resolving_power": [-50, 0.0001],
        "opacity_factor": [-1, 1],
    }

    with pytest.raises(ValidationError) as excinfo:
        input_params = WavelengthCalibrationParameters(
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
            bounds=bounds,
        )
    assert "must be an astropy Quantity with appropriate units" in str(excinfo.value)


def test_params_have_no_bounds(test_parameters_dict):
    """
    Given: Input parameters with no units
    When: Initializing the fit parameters
    Then: Each parameter's bound is the default value
    """

    with pytest.raises(TypeError) as excinfo:
        input_params = WavelengthCalibrationParameters(
            crval1=test_parameters_dict["crval1"].to_value(u.nm),
            dispersion=test_parameters_dict["dispersion"].to_value(u.nm),
            incident_light_angle=test_parameters_dict["incident_light_angle"].to_value(u.rad),
            resolving_power=test_parameters_dict["resolving_power"],
            opacity_factor=test_parameters_dict["opacity_factor"],
            straylight_fraction=test_parameters_dict["straylight_fraction"],
            unfit_1d_spectrum=test_parameters_dict["unfit_1d_spectrum"],
            grating_constant=test_parameters_dict["grating_constant"],
            doppler_velocity=test_parameters_dict["doppler_velocity"],
            order=test_parameters_dict["order"],
        )
    assert "must be an astropy Quantity" in str(excinfo.value)


def test_default_fit_flags(test_parameters_dict):
    """
    Given: No custom input fit flags
    When: Initializing the fit parameters
    Then: Each parameter's fit flag is the default value (True)
    """
    input_params = WavelengthCalibrationParameters(
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
    )

    assert input_params.fit_flags["crval1"] == True
    assert input_params.fit_flags["dispersion"] == True
    assert input_params.fit_flags["incident_light_angle"] == True
    assert input_params.fit_flags["resolving_power"] == True
    assert input_params.fit_flags["opacity_factor"] == True
    assert input_params.fit_flags["straylight_fraction"] == True
    assert input_params.fit_flags["continuum_intensity_correction"] == True


def test_custom_fit_flags(test_parameters_dict):
    """
    Given: Some custom fit flags
    When: Initializing the fit parameters
    Then: The customized input fit flags are set to the custom values and others are set to the default values
    """
    fit_flags = {
        "dispersion": False,
        "resolving_power": False,
        "opacity_factor": False,
        "continuum_intensity_correction": False,
    }

    input_params = WavelengthCalibrationParameters(
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
        fit_flags=fit_flags,
    )

    assert input_params.fit_flags["crval1"] == True
    assert input_params.fit_flags["dispersion"] == False
    assert input_params.fit_flags["incident_light_angle"] == True
    assert input_params.fit_flags["resolving_power"] == False
    assert input_params.fit_flags["opacity_factor"] == False
    assert input_params.fit_flags["straylight_fraction"] == True
    assert input_params.fit_flags["continuum_intensity_correction"] == False
