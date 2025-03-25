import astropy.units as u
import numpy as np
import pytest
from astropy.units import Quantity
from lmfit import Parameters
from pydantic import ValidationError

from solar_wavelength_calibration.atlas.base import AtlasBase
from solar_wavelength_calibration.fitter.parameters import BoundRange
from solar_wavelength_calibration.fitter.parameters import BoundsModel
from solar_wavelength_calibration.fitter.parameters import WavelengthCalibrationParameters


def test_incorrect_parameter_types(test_unfit_1d_spectrum):
    """
    Given: Incorrect input parameter value types
    When: Initializing a WavelengthCalibrationParameters object with these incorrect inputs
    Then: Raise a ValidationError
    """
    with pytest.raises(ValidationError):
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

    assert input_params.bounds.crval1.min == -np.inf
    assert input_params.bounds.crval1.max == np.inf
    assert input_params.bounds.dispersion.min == -np.inf
    assert input_params.bounds.dispersion.max == np.inf
    assert input_params.bounds.incident_light_angle.min == -np.inf
    assert input_params.bounds.incident_light_angle.max == np.inf
    assert input_params.bounds.resolving_power.min == -np.inf
    assert input_params.bounds.resolving_power.max == np.inf
    assert input_params.bounds.opacity_factor.min == -np.inf
    assert input_params.bounds.opacity_factor.max == np.inf
    assert input_params.bounds.straylight_fraction.min == -np.inf
    assert input_params.bounds.straylight_fraction.max == np.inf


def test_custom_bounds(test_parameters_dict):
    """
    Given: Some custom input bounds
    When: Initializing the fit parameters
    Then: The customized input bounds are set to the custom values and others are set to the default values
    """
    bounds = BoundsModel(
        crval1=BoundRange(
            min=test_parameters_dict["crval1"] - (0.3 * u.nm),
            max=test_parameters_dict["crval1"] + (0.3 * u.nm),
        ),
        dispersion=BoundRange(
            min=test_parameters_dict["dispersion"] - (55 * u.nm / u.pix),
            max=test_parameters_dict["dispersion"] + (55 * u.nm / u.pix),
        ),
        incident_light_angle=BoundRange(
            min=test_parameters_dict["incident_light_angle"] - (5 * u.deg),
            max=test_parameters_dict["incident_light_angle"] + (103 * u.deg),
        ),
        resolving_power=BoundRange(
            min=test_parameters_dict["resolving_power"] - 50,
            max=test_parameters_dict["resolving_power"] + 0.0001,
        ),
        opacity_factor=BoundRange(
            min=test_parameters_dict["opacity_factor"] - 1,
            max=test_parameters_dict["opacity_factor"] + 1,
        ),
    )

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

    assert np.isclose(
        input_params.bounds.crval1.min.value, test_parameters_dict["crval1"].to_value(u.nm) - 0.3
    )
    assert np.isclose(
        input_params.bounds.crval1.max.value, test_parameters_dict["crval1"].to_value(u.nm) + 0.3
    )
    assert np.isclose(
        input_params.bounds.dispersion.min.value,
        test_parameters_dict["dispersion"].to_value(u.nm / u.pix) - 55,
    )
    assert np.isclose(
        input_params.bounds.dispersion.max.value,
        test_parameters_dict["dispersion"].to_value(u.nm / u.pix) + 55,
    )
    assert np.isclose(
        input_params.bounds.incident_light_angle.min.value,
        test_parameters_dict["incident_light_angle"].to_value(u.deg) - 5,
    )
    assert np.isclose(
        input_params.bounds.incident_light_angle.max.value,
        test_parameters_dict["incident_light_angle"].to_value(u.deg) + 103,
    )
    assert np.isclose(
        input_params.bounds.resolving_power.min, test_parameters_dict["resolving_power"] - 50
    )
    assert np.isclose(
        input_params.bounds.resolving_power.max, test_parameters_dict["resolving_power"] + 0.0001
    )
    assert np.isclose(
        input_params.bounds.opacity_factor.min, test_parameters_dict["opacity_factor"] - 1
    )
    assert np.isclose(
        input_params.bounds.opacity_factor.max, test_parameters_dict["opacity_factor"] + 1
    )
    assert input_params.bounds.straylight_fraction.min == -np.inf  # default bounds
    assert input_params.bounds.straylight_fraction.max == np.inf  # default bounds


def test_custom_bounds_incorrect_units(test_parameters_dict):
    """
    Given: Some custom input bounds with incorrect input units
    When: Initializing the fit parameters
    Then: The customized input bounds are converted to the correct units and others are set to the default values
    """
    bounds = BoundsModel(
        crval1=BoundRange(
            min=test_parameters_dict["crval1"] - (3 * u.Angstrom),
            max=test_parameters_dict["crval1"] + (3 * u.Angstrom),
        )
    )

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

    # bound value was converted to correct units
    assert np.isclose(
        input_params.bounds.crval1.min.value, test_parameters_dict["crval1"].to_value(u.nm) - 0.3
    )
    assert np.isclose(
        input_params.bounds.crval1.max.value, test_parameters_dict["crval1"].to_value(u.nm) + 0.3
    )


def test_bounds_have_no_units(test_parameters_dict):
    """
    Given: Some custom input bounds with no units
    When: Initializing the fit parameters
    Then: Bounds are assigned the correct units
    """
    bounds = BoundsModel(
        crval1=BoundRange(
            min=test_parameters_dict["crval1"].to_value(u.nm) - 0.3,
            max=test_parameters_dict["crval1"].to_value(u.nm) + 0.3,
        ),
        dispersion=BoundRange(
            min=test_parameters_dict["dispersion"].to_value(u.nm / u.pix) - 55,
            max=test_parameters_dict["dispersion"].to_value(u.nm / u.pix) + 55,
        ),
        incident_light_angle=BoundRange(
            min=test_parameters_dict["incident_light_angle"].to_value(u.deg) - 5,
            max=test_parameters_dict["incident_light_angle"].to_value(u.deg) + 103,
        ),
        resolving_power=BoundRange(
            min=test_parameters_dict["resolving_power"] - 50,
            max=test_parameters_dict["resolving_power"] + 0.0001,
        ),
        opacity_factor=BoundRange(
            min=test_parameters_dict["opacity_factor"] - 1,
            max=test_parameters_dict["opacity_factor"] + 1,
        ),
    )

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

    assert isinstance(input_params.bounds.crval1.min, Quantity)
    assert isinstance(input_params.bounds.crval1.max, Quantity)
    assert isinstance(input_params.bounds.dispersion.min, Quantity)
    assert isinstance(input_params.bounds.dispersion.max, Quantity)
    assert isinstance(input_params.bounds.incident_light_angle.min, Quantity)
    assert isinstance(input_params.bounds.incident_light_angle.max, Quantity)


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

    assert input_params.fit_flags.crval1 == True
    assert input_params.fit_flags.dispersion == True
    assert input_params.fit_flags.incident_light_angle == True
    assert input_params.fit_flags.resolving_power == True
    assert input_params.fit_flags.opacity_factor == True
    assert input_params.fit_flags.straylight_fraction == True
    assert input_params.fit_flags.continuum_intensity_correction == True


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

    assert input_params.fit_flags.crval1 == True
    assert input_params.fit_flags.dispersion == False
    assert input_params.fit_flags.incident_light_angle == True
    assert input_params.fit_flags.resolving_power == False
    assert input_params.fit_flags.opacity_factor == False
    assert input_params.fit_flags.straylight_fraction == True
    assert input_params.fit_flags.continuum_intensity_correction == False
