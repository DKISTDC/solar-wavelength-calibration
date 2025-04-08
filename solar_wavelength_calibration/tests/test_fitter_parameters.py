import re

import astropy.units as u
import numpy as np
import pytest
from lmfit import Parameters
from pydantic import ValidationError

from solar_wavelength_calibration.atlas.base import AtlasBase
from solar_wavelength_calibration.fitter.parameters import AngleBoundRange
from solar_wavelength_calibration.fitter.parameters import BoundsModel
from solar_wavelength_calibration.fitter.parameters import DispersionBoundRange
from solar_wavelength_calibration.fitter.parameters import FitFlagsModel
from solar_wavelength_calibration.fitter.parameters import LengthBoundRange
from solar_wavelength_calibration.fitter.parameters import UnitlessBoundRange
from solar_wavelength_calibration.fitter.parameters import WavelengthCalibrationParameters


def test_unitless_bound_range():
    """
    Given: The `UnitlessBoundRange` class
    When: Instantiating with valid and invalid inputs
    Then: Correct inputs are assigned to the min/max properties and invlaid inputs rais the correct error
    """
    bounds = UnitlessBoundRange(min=-10, max=1e35)
    assert bounds.min == -10
    assert bounds.max == 1e35
    with pytest.raises(
        ValidationError, match="self.min = 10 is greater than or equal to self.max = 0"
    ):
        UnitlessBoundRange(min=10, max=0)

    with pytest.raises(ValidationError, match="input_type=Quantity"):
        UnitlessBoundRange(max=6.28 * u.m)


@pytest.mark.parametrize(
    "bound_range_class, min, max, expected_unit, expected_min, expected_max",
    [
        pytest.param(
            LengthBoundRange, "default", 3 * u.m, u.nm, -np.inf, 3e9, id="LengthBoundRange"
        ),
        pytest.param(
            AngleBoundRange, np.pi * u.rad, "default", u.deg, 180.0, np.inf, id="AngleBoundRange"
        ),
        pytest.param(
            DispersionBoundRange,
            "default",
            6.28 * u.um / u.pix,
            u.nm / u.pix,
            -np.inf,
            6.28e3,
            id="DispersionBoundRange",
        ),
    ],
)
def test_typed_bound_ranges(bound_range_class, min, max, expected_unit, expected_min, expected_max):
    """
    Given: A typed subclass of `UnitlessBoundRange` and inputs that have the same type, but different units, as expected
      by the subclass
    When: Instantiating the class with valid and invalid inputs
    Then: Units are converted correctly and the correct error is raised with inputs with either the wrong type of units
      or no units at all.
    """
    args = {"min": m for m in [min] if m != "default"} | {"max": m for m in [max] if m != "default"}
    bounds = bound_range_class(**args)
    assert bounds.min.unit == bounds.max.unit == expected_unit
    np.testing.assert_allclose(bounds.min.value, expected_min)
    np.testing.assert_allclose(bounds.max.value, expected_max)

    # None of the units are, or will ever be, mass so lets use "kg" as a wrong unit
    with pytest.raises(ValidationError, match=re.escape("'kg' (mass) and")):
        bound_range_class(min=314 * u.kg)

    with pytest.raises(ValidationError, match="Input should be an instance of Quantity"):
        bound_range_class(max=6.28)


def test_incorrect_parameter_types(test_unfit_1d_spectrum):
    """
    Given: Incorrect input parameter value types
    When: Initializing a WavelengthCalibrationParameters object with these incorrect inputs
    Then: Raise a ValidationError
    """
    with pytest.raises(ValidationError) as excinfo:
        WavelengthCalibrationParameters(
            crval1=3,
            dispersion="shouldbeafloat",
            incident_light_angle=154 * u.nm,
            resolving_power="shouldbeanint",
            opacity_factor=5,
            straylight_fraction=2,
            unfit_1d_spectrum=test_unfit_1d_spectrum,
            grating_constant=5 / u.m,
            doppler_velocity=12 * u.km / u.s,
            order=1.5,
        )

    assert excinfo.match("crval1\s+Input should be an instance of Quantity")
    assert excinfo.match("dispersion\s+Input should be an instance of Quantity")
    assert excinfo.match(
        "incident_light_angle\s+Value error, 'nm' \(length\) and 'deg' \(angle\) are not convertible"
    )
    assert excinfo.match("order\s+Input should be a valid integer")


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
    assert input_params.bounds.dispersion.min == 0 * u.nm / u.pix
    assert input_params.bounds.dispersion.max == np.inf
    assert input_params.bounds.incident_light_angle.min == -360 * u.deg
    assert input_params.bounds.incident_light_angle.max == 360 * u.deg
    assert input_params.bounds.resolving_power.min == 0
    assert input_params.bounds.resolving_power.max == np.inf
    assert input_params.bounds.opacity_factor.min == 0
    assert input_params.bounds.opacity_factor.max == np.inf
    assert input_params.bounds.straylight_fraction.min == 0
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
    assert input_params.bounds.straylight_fraction.min == 0  # default bounds
    assert input_params.bounds.straylight_fraction.max == np.inf  # default bounds


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


def test_custom_fit_flags(test_parameters_dict):
    """
    Given: Some custom fit flags
    When: Initializing the fit parameters
    Then: The customized input fit flags are set to the custom values and others are set to the default values
    """
    fit_flags = FitFlagsModel(dispersion=False, resolving_power=False, opacity_factor=False)

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


def test_bounds_min_max_have_different_units(test_parameters_dict):
    """
    Given: Some bounds that have different min/max units.
    When: Initializing the fit parameters.
    Then: The bounds should have been converted to the same units.
    """
    bounds = BoundsModel(
        crval1=BoundRange(
            min=(test_parameters_dict["crval1"].to_value(u.Angstrom)) / 10.0 - 0.3,
            max=test_parameters_dict["crval1"].to_value(u.nm) + 0.3,
        ),
        dispersion=BoundRange(
            min=test_parameters_dict["dispersion"].to_value(u.um / u.pix) - 55,
            max=test_parameters_dict["dispersion"].to_value(u.nm / u.pix) + 55,
        ),
        incident_light_angle=BoundRange(
            min=test_parameters_dict["incident_light_angle"].to_value(u.rad) - 5,
            max=test_parameters_dict["incident_light_angle"].to_value(u.deg) + 103,
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

    # make sure min units were converted
    assert input_params.bounds.crval1.min.unit.is_equivalent(u.nm)
    assert input_params.bounds.dispersion.min.unit.is_equivalent(u.nm / u.pix)
    assert input_params.bounds.incident_light_angle.min.unit.is_equivalent(u.deg)


def test_bounds_given_as_dict(test_parameters_dict):
    """
    Given: Custom bounds given as a dictionary instead of a BoundsModel
    When:
    Then:
    """
    bounds = {
        "crval1": {
            "min": test_parameters_dict["crval1"] - (0.3 * u.nm),
            "max": test_parameters_dict["crval1"] + (0.3 * u.nm),
        },
        "dispersion": {
            "min": test_parameters_dict["dispersion"] - (55 * u.nm / u.pix),
            "max": test_parameters_dict["dispersion"] + (55 * u.nm / u.pix),
        },
        "incident_light_angle": {
            "min": test_parameters_dict["incident_light_angle"] - (5 * u.deg),
            "max": test_parameters_dict["incident_light_angle"] + (103 * u.deg),
        },
        "resolving_power": {
            "min": test_parameters_dict["resolving_power"] - 50,
            "max": test_parameters_dict["resolving_power"] + 0.0001,
        },
        "opacity_factor": {
            "min": test_parameters_dict["opacity_factor"] - 1,
            "max": test_parameters_dict["opacity_factor"] + 1,
        },
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
