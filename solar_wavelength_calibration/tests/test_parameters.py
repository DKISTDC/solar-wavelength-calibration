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


def test_bounds_model():
    """
    Given: A collection of *BoundRange objects and a dict representation of a BoundRange object
    When: Instantiating a `BoundsModel` object with these objects
    Then: Correctly assignment results in the expected `BoundsModel` object. Incorrect assignment results in failure.
    """
    # Not much to test here, tbh, because `BoundsModel` is so light
    length_bound = LengthBoundRange()
    dispersion_bound = DispersionBoundRange(max=6.28 * u.angstrom / u.pix)
    unitless_bound = UnitlessBoundRange(min=-1e8)

    # Test dict as input. Please NEVER do this.
    bounds = BoundsModel(
        crval1={"min": -1 * u.m, "max": 3 * u.nm},
        dispersion=dispersion_bound,
        resolving_power=unitless_bound,
        opacity_factor=unitless_bound,
        straylight_fraction=unitless_bound,
    )
    np.testing.assert_allclose(bounds.crval1.min, -1e9 * u.nm)
    np.testing.assert_allclose(bounds.crval1.max, 3 * u.nm)
    assert bounds.dispersion == dispersion_bound
    assert bounds.resolving_power == unitless_bound
    assert bounds.opacity_factor == unitless_bound
    assert bounds.straylight_fraction == unitless_bound

    # Test a default
    assert bounds.incident_light_angle == AngleBoundRange(min=0 * u.deg, max=180 * u.deg)

    # Test a bad type
    with pytest.raises(ValidationError, match="instance of AngleBoundRange"):
        BoundsModel(incident_light_angle=length_bound)


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


def test_parameter_outside_bounds_range():
    """
    Given: A WavelengthCalibrationParameters class where the input value of a parameter is outside of its bounds
    When: Instantiating the class
    Then: The correct error is raised
    """
    bounds = BoundsModel(crval1=LengthBoundRange(min=1000 * u.nm, max=1001 * u.nm))

    with pytest.raises(
        ValidationError,
        match="crval1 with value 999.0 nm is outside the bounds range of \(1000.0 nm, 1001.0 nm\)",
    ):
        WavelengthCalibrationParameters(
            crval1=999 * u.nm,
            dispersion=1 * u.nm / u.pix,
            incident_light_angle=45 * u.deg,
            grating_constant=1 / u.m,
            doppler_velocity=3 * u.km / u.s,
            unfit_1d_spectrum=np.arange(3),
            opacity_factor=0,
            order=51,
            bounds=bounds,
        )


def test_lmfit_parameters_was_set_up_correctly(test_parameters_dict):
    """
    Given: A dictionary of input parameter values
    When: Initializing a WavelengthCalibrationParameters object
    Then: The lmfit parameters were set up correctly and default parameters are used where necessary
    """
    input_params = WavelengthCalibrationParameters(
        crval1=test_parameters_dict["crval1"],
        dispersion=test_parameters_dict["dispersion"],
        incident_light_angle=test_parameters_dict["incident_light_angle"],
        opacity_factor=test_parameters_dict["opacity_factor"],
        unfit_1d_spectrum=test_parameters_dict["unfit_1d_spectrum"],
        grating_constant=test_parameters_dict["grating_constant"],
        doppler_velocity=test_parameters_dict["doppler_velocity"],
        order=test_parameters_dict["order"],
    )

    assert isinstance(input_params.lmfit_parameters, Parameters)
    # make sure than only 7 parameters were added into lmfit_parameters [crval1, dispersion, incident_light_angle, resolving_power, opacity_factor, straylight_fraction, unfit_1d_spectrum]
    assert len(input_params.lmfit_parameters.valuesdict()) == 7

    # Test default parameter values
    initialized_params_dict = input_params.lmfit_parameters.valuesdict()
    assert initialized_params_dict["resolving_power"] == 1
    assert initialized_params_dict["straylight_fraction"] == 0


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

    np.testing.assert_allclose(
        input_params.bounds.crval1.min, test_parameters_dict["crval1"] - (0.3 * u.nm)
    )
