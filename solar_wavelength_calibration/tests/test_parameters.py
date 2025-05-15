import re

import astropy.units as u
import numpy as np
import pytest
from hypothesis import example
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from lmfit import Parameters
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import ValidationError

from solar_wavelength_calibration import AngleBoundRange
from solar_wavelength_calibration import BoundsModel
from solar_wavelength_calibration import DispersionBoundRange
from solar_wavelength_calibration import FitFlagsModel
from solar_wavelength_calibration import LengthBoundRange
from solar_wavelength_calibration import UnitlessBoundRange
from solar_wavelength_calibration import WavelengthCalibrationParameters
from solar_wavelength_calibration.fitter.parameters import angle_quantity
from solar_wavelength_calibration.fitter.parameters import inverse_length_quantity
from solar_wavelength_calibration.fitter.parameters import length_quantity


class WCSValidityParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    crval: length_quantity
    incident_light_angle: angle_quantity
    grating_constant: inverse_length_quantity
    order: int
    delta: float


@st.composite
def wcs_validity_params(draw, valid: bool) -> WCSValidityParams:
    """
    Use hypothesis to generate a set of parameters that affect WCS validity.

    If `valid is True` the parameters are guaranteed to represent a valid WCS. The converse is true is `valid is False`.
    """
    # Cast this as nm for now to avoid potential overflows in unit conversion on `crval`
    grating_constant = draw(st.integers().filter(lambda n: n != 0)) / u.nm
    incident_light_angle = draw(st.floats(allow_nan=False, allow_infinity=False)) * u.deg
    order = draw(st.integers().filter(lambda n: n != 0))
    if valid:
        # Need to subtract eps-ish from the limits or else `Quantity` floating point stuff will result in borked inputs.
        machine_precision = np.finfo(float).eps * 10
        delta = draw(
            st.floats(min_value=-1.0 + machine_precision, max_value=1.0 - machine_precision)
        )
    else:
        delta = draw(st.floats(allow_nan=False, allow_infinity=False).filter(lambda f: abs(f) > 1))
    crval = ((delta + np.sin(incident_light_angle)) / (grating_constant * order)).to(u.nm)

    return WCSValidityParams(
        crval=crval,
        incident_light_angle=incident_light_angle,
        grating_constant=grating_constant.to(1 / u.m),
        order=order,
        delta=delta,
    )


class SaneBoundsAndWCSParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    crval: length_quantity
    incident_light_angle: angle_quantity
    grating_constant: inverse_length_quantity
    order: int
    crval_bounds: LengthBoundRange
    incident_light_angle_bounds: AngleBoundRange


@st.composite
def sane_wcs_params_and_bounds(
    draw, completely_sane_angles: bool = False
) -> SaneBoundsAndWCSParams:
    """
    Use hypothesis to generate a set of parameters and then make bounds on `crval` and `incident_light_angle.

    Very similar to `wcs_validity_params`, but here we're limiting the outer edges of numbers we're considering to be
    values well short of ±inf. So "sane" is still relative here.

    If `completely_sane_angles` is True then only generate angles in the range [0, 360]. This is to mimic inputs after
    they have passed through `WavelengthCalibrationParameters.ensure_sane_ligth_angle_values_if_crval_fixed`.
    """
    sanity = 1e10
    # Grow the limits on bounds so we don't end up in a case where min=max=value
    bound_sanity = sanity * 100

    if completely_sane_angles:
        min_angle = 0
        max_angle = 360
        min_angle_bound = 0
        max_angle_bound = 360
    else:
        min_angle = -sanity
        max_angle = sanity
        min_angle_bound = -bound_sanity
        max_angle_bound = bound_sanity

    grating_constant = (
        draw(st.integers(min_value=int(-sanity), max_value=int(sanity)).filter(lambda n: n != 0))
        / u.m
    )
    incident_light_angle = (
        draw(
            st.floats(
                allow_nan=False, allow_infinity=False, min_value=min_angle, max_value=max_angle
            )
        )
        * u.deg
    )
    order = draw(
        st.integers(min_value=int(-sanity), max_value=int(sanity)).filter(lambda n: n != 0)
    )

    # Need to subtract eps-ish from the limits or else `Quantity` floating point stuff will result in borked inputs.
    if completely_sane_angles:
        machine_precision = 0.01
    else:
        machine_precision = np.finfo(float).eps * 10
    delta = draw(st.floats(min_value=-1.0 + machine_precision, max_value=1.0 - machine_precision))
    crval = ((delta + np.sin(incident_light_angle)) / (grating_constant * order)).to(u.nm)

    crval_min = draw(
        st.floats(
            allow_nan=False,
            allow_infinity=False,
            min_value=-bound_sanity,
            max_value=crval.value,
            exclude_max=True,
        )
    )
    crval_max = draw(
        st.floats(
            allow_nan=False,
            allow_infinity=False,
            min_value=crval.value,
            max_value=bound_sanity,
            exclude_min=True,
        )
    )

    if completely_sane_angles and incident_light_angle.value == 0:
        angle_min = 0
    else:
        angle_min = draw(
            st.floats(
                allow_nan=False,
                allow_infinity=False,
                min_value=min_angle_bound,
                max_value=incident_light_angle.value,
                exclude_max=True,
            )
        )

    if completely_sane_angles and incident_light_angle.value == 360:
        angle_max = 360
    else:
        angle_max = draw(
            st.floats(
                allow_nan=False,
                allow_infinity=False,
                min_value=incident_light_angle.value,
                max_value=max_angle_bound,
                exclude_min=True,
            )
        )

    # Because we had "sane" limits on the bounds we couldn't get ±inf, which is a useful case of "unbounded"
    # Let's add that case here by flipping a coin for each limit
    # (st.integers() % 2 preferable to st.boolean() because the latter shrinks towards False)
    if draw(st.integers()) % 2:
        crval_min = -np.inf
    if draw(st.integers()) % 2:
        crval_max = np.inf
    if draw(st.integers()) % 2:
        angle_min = -np.inf
    if draw(st.integers()) % 2:
        angle_max = np.inf

    crval_bounds = LengthBoundRange(min=crval_min * u.nm, max=crval_max * u.nm)
    angle_bounds = AngleBoundRange(min=angle_min * u.deg, max=angle_max * u.deg)

    return SaneBoundsAndWCSParams(
        crval=crval,
        incident_light_angle=incident_light_angle,
        grating_constant=grating_constant,
        order=order,
        crval_bounds=crval_bounds,
        incident_light_angle_bounds=angle_bounds,
    )


sane_bounds_with_360angle_and_0_min = SaneBoundsAndWCSParams(
    crval=1070 * u.nm,
    incident_light_angle=360 * u.deg,
    grating_constant=31600 / u.m,
    order=1,
    crval_bounds=LengthBoundRange(min=1060 * u.nm, max=1080 * u.nm),
    incident_light_angle_bounds=AngleBoundRange(min=0 * u.deg, max=360 * u.deg),
)


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
        crval={"min": -1 * u.m, "max": 3 * u.nm},
        dispersion=dispersion_bound,
        resolving_power=unitless_bound,
        opacity_factor=unitless_bound,
        straylight_fraction=unitless_bound,
    )
    np.testing.assert_allclose(bounds.crval.min, -1e9 * u.nm)
    np.testing.assert_allclose(bounds.crval.max, 3 * u.nm)
    assert bounds.dispersion == dispersion_bound
    assert bounds.resolving_power == unitless_bound
    assert bounds.opacity_factor == unitless_bound
    assert bounds.straylight_fraction == unitless_bound

    # Test a default
    assert bounds.incident_light_angle == AngleBoundRange(min=0 * u.deg, max=180 * u.deg)

    # Test a bad type
    with pytest.raises(ValidationError, match="instance of AngleBoundRange"):
        BoundsModel(incident_light_angle=length_bound)


def test_incorrect_parameter_types(observed_spectrum):
    """
    Given: Incorrect input parameter value types
    When: Initializing a WavelengthCalibrationParameters object with these incorrect inputs
    Then: Raise a ValidationError
    """
    with pytest.raises(ValidationError) as excinfo:
        WavelengthCalibrationParameters(
            crval=3,
            dispersion="shouldbeafloat",
            incident_light_angle=154 * u.nm,
            resolving_power="shouldbeanint",
            opacity_factor=5,
            straylight_fraction=2,
            grating_constant=5 / u.m,
            doppler_velocity=12 * u.km / u.s,
            order=1.5,
        )

    assert excinfo.match(r"crval\s+Input should be an instance of Quantity")
    assert excinfo.match(r"dispersion\s+Input should be an instance of Quantity")
    assert excinfo.match(
        r"incident_light_angle\s+Value error, 'nm' \(length\) and 'deg' \(angle\) are not convertible"
    )
    assert excinfo.match(r"order\s+Input should be a valid integer")


def test_non_finite_inputs():
    """
    Given: A set of `WavelengthCalibrationParameters` inputs that are not finite
    When: Instantiating a `WavelengthCalibrationParameters` object with those parameters
    Then: The correct errors are raised
    """
    with pytest.raises(ValidationError) as excinfo:
        WavelengthCalibrationParameters(
            crval=np.inf * u.m,  # Note that some units are the right type, but incorrect units.
            dispersion=np.nan * u.nm / u.pix,
            incident_light_angle=-np.inf * u.deg,
            resolving_power=3,
            opacity_factor=np.inf,
            straylight_fraction=np.nan,
            grating_constant=-np.inf / u.nm,
            doppler_velocity=np.nan * u.km / u.s,
            order=np.nan,
        )

    assert excinfo.match("8 validation errors")
    assert excinfo.match(r"crval\s+Value error, parameters must be finite")
    assert excinfo.match(r"dispersion\s+Value error, parameters must be finite")
    assert excinfo.match(r"incident_light_angle\s+Value error, parameters must be finite")
    assert excinfo.match(r"opacity_factor\s+Input should be a finite number")
    assert excinfo.match(r"straylight_fraction\s+Input should be a finite number")
    assert excinfo.match(r"grating_constant\s+Value error, parameters must be finite")
    assert excinfo.match(r"doppler_velocity\s+Value error, parameters must be finite")
    assert excinfo.match(r"order\s+Input should be a finite number")


def test_missing_parameters(test_only_required_parameters_dict):
    """
    Given: Only the calculated crval input parameter value
    When: Initializing a WavelengthCalibrationParameters object with these inputs
    Then: Raise a ValidationError
    """
    with pytest.raises(ValidationError) as excinfo:
        WavelengthCalibrationParameters(crval=9 * u.nm)

    del test_only_required_parameters_dict["crval"]  # removing this because we gave it as an input
    # check that all of the required inputs are reported to be missing
    for key in test_only_required_parameters_dict.keys():
        assert key in str(excinfo.value)


def test_parameter_outside_bounds_range():
    """
    Given: A WavelengthCalibrationParameters class where the input value of a parameter is outside of its bounds
    When: Instantiating the class
    Then: The correct error is raised
    """
    bounds = BoundsModel(crval=LengthBoundRange(min=1000 * u.nm, max=1001 * u.nm))

    with pytest.raises(
        ValidationError,
        match=r"crval with value 999.0 nm is outside the bounds range of \(1000.0 nm, 1001.0 nm\)",
    ):
        WavelengthCalibrationParameters(
            crval=999 * u.nm,
            dispersion=1 * u.nm / u.pix,
            incident_light_angle=45 * u.deg,
            grating_constant=1 / u.m,
            doppler_velocity=3 * u.km / u.s,
            opacity_factor=0,
            order=51,
            bounds=bounds,
        )


def test_annotated_types_unit_conversion():
    """
    Given: A WavelengthCalibrationParameters object initialized with values that are the same unit type, but not
      identical to units in the internal representation of that value
    When: Accessing the parameter values
    Then: The value with the correct, internal units is returned
    """
    params = WavelengthCalibrationParameters(
        crval=999 * u.m,
        dispersion=1 * u.um / u.pix,
        incident_light_angle=np.pi * u.rad,
        grating_constant=1 / u.km,
        doppler_velocity=3 * u.m / u.s,
        opacity_factor=0,
        order=1,
    )

    # Use `.value` here directly because we want to NOT rely on automatic unit conversion provided by `Quantity`
    np.testing.assert_allclose(params.crval.value, 999e9)
    np.testing.assert_allclose(params.dispersion.value, 1e3)
    np.testing.assert_allclose(params.incident_light_angle.value, 180.0)
    np.testing.assert_allclose(params.grating_constant.value, 1e-3)
    np.testing.assert_allclose(params.doppler_velocity.value, 3e-3)


@given(wcs_params=wcs_validity_params(valid=True))
@settings(max_examples=10000)  # L E T ' S  G O O O O O O O O
def test_valid_grating_equation(wcs_params: WCSValidityParams, test_parameters_dict):
    """
    Given: A set of inputs values that should represent a valid WCS
    When: Instantiating a `WavelengthCalibrationParameters` object with these parameters
    Then: No error is raised
    """
    WavelengthCalibrationParameters(
        crval=wcs_params.crval,
        dispersion=test_parameters_dict["dispersion"],
        incident_light_angle=wcs_params.incident_light_angle,
        opacity_factor=test_parameters_dict["opacity_factor"],
        grating_constant=wcs_params.grating_constant,
        doppler_velocity=test_parameters_dict["doppler_velocity"],
        order=wcs_params.order,
        # Make bounds [-inf, inf] so we can test WCS validity on full range of wacko inputs
        bounds=BoundsModel(
            crval=LengthBoundRange(),
            incident_light_angle=AngleBoundRange(),
        ),
    )


@given(wcs_params=wcs_validity_params(valid=False))
def test_invalid_grating_equation(wcs_params: WCSValidityParams, test_parameters_dict):
    """
    Given: A set of input values such that `abs(crval * grating_constant * order - sin(incident_light_angle) > 1`
    When: Instantiating a `WavelengthCalibrationParameters` object with these parameters
    Then: The correct ValidationError is raised
    """
    with pytest.raises(
        ValidationError, match="Input parameter values represent an un-invertible grating equation."
    ):
        WavelengthCalibrationParameters(
            crval=wcs_params.crval,
            dispersion=test_parameters_dict["dispersion"],
            incident_light_angle=wcs_params.incident_light_angle,
            opacity_factor=test_parameters_dict["opacity_factor"],
            grating_constant=wcs_params.grating_constant,
            doppler_velocity=test_parameters_dict["doppler_velocity"],
            order=wcs_params.order,
            # Make bounds [-inf, inf] so we can test WCS validity on full range of wacko inputs
            bounds=BoundsModel(
                crval=LengthBoundRange(),
                incident_light_angle=AngleBoundRange(),
            ),
        )


@pytest.mark.parametrize(
    "fit_crval", [pytest.param(True, id="crval_free"), pytest.param(False, id="crval_fixed")]
)
def test_lmfit_parameters_was_set_up_correctly(test_parameters_dict, fit_crval):
    """
    Given: A dictionary of input parameter values
    When: Initializing a WavelengthCalibrationParameters object
    Then: The lmfit parameters were set up correctly and default parameters are used where necessary
    """
    input_params = WavelengthCalibrationParameters(
        crval=test_parameters_dict["crval"],
        dispersion=test_parameters_dict["dispersion"],
        incident_light_angle=test_parameters_dict["incident_light_angle"],
        opacity_factor=test_parameters_dict["opacity_factor"],
        grating_constant=test_parameters_dict["grating_constant"],
        doppler_velocity=test_parameters_dict["doppler_velocity"],
        order=test_parameters_dict["order"],
        fit_flags=FitFlagsModel(crval=fit_crval),
    )

    assert isinstance(input_params.lmfit_parameters, Parameters)
    # make sure than only 7 parameters were added into lmfit_parameters [crval, dispersion, incident_light_angle, resolving_power, opacity_factor, straylight_fraction, input_spectrum]
    assert len(input_params.lmfit_parameters.valuesdict()) == 8 if fit_crval else 7

    # Test default parameter values
    initialized_params_dict = input_params.lmfit_parameters.valuesdict()
    assert initialized_params_dict["resolving_power"] == 1
    assert initialized_params_dict["straylight_fraction"] == 0


@pytest.mark.parametrize(
    "angle_free", [pytest.param(True, id="angle_free"), pytest.param(False, id="angle_fixed")]
)
@given(valid_bounds_and_params=sane_wcs_params_and_bounds())
@settings(max_examples=1000)
def test_internal_delta_bounds(
    valid_bounds_and_params: SaneBoundsAndWCSParams, angle_free, test_parameters_dict
):
    """
    Given: A `WavelengthCalibrationParameters` object with bounds on `crval` and/or `incident_light_angle`
    When: Computing bounds on the internal `_delta` fit parameter to match
    Then: The input bounds on `crval` are at two corners of the `crval(_delta_bounds, angle_bounds)` square
      and the new bounds on `_delta` are still in the range [-1, 1].
    """
    input_crval_bounds = valid_bounds_and_params.crval_bounds
    input_angle_bounds = valid_bounds_and_params.incident_light_angle_bounds
    params = WavelengthCalibrationParameters(
        crval=valid_bounds_and_params.crval,
        dispersion=test_parameters_dict["dispersion"],
        incident_light_angle=valid_bounds_and_params.incident_light_angle,
        opacity_factor=test_parameters_dict["opacity_factor"],
        grating_constant=valid_bounds_and_params.grating_constant,
        doppler_velocity=test_parameters_dict["doppler_velocity"],
        order=valid_bounds_and_params.order,
        bounds=BoundsModel(crval=input_crval_bounds, incident_light_angle=input_angle_bounds),
        fit_flags=FitFlagsModel(incident_light_angle=angle_free),
    )

    delta_bounds = params.delta_bounds_given_crval_bounds

    min_angle = (
        params.bounds.incident_light_angle.min
        if np.isfinite(params.bounds.incident_light_angle.min)
        else 0 * u.deg
    )
    max_angle = (
        params.bounds.incident_light_angle.max
        if np.isfinite(params.bounds.incident_light_angle.max)
        else 180 * u.deg
    )

    if angle_free:
        angle_lims = u.Quantity([min_angle, max_angle])
    else:
        angle_lims = u.Quantity([params.incident_light_angle])
    delta_lims = np.array([delta_bounds.min, delta_bounds.max])

    crval_at_corners = (delta_lims[:, None] + np.sin(angle_lims[None, :])) / (
        params.grating_constant * params.order
    )

    delta_min = params.delta_bounds_given_crval_bounds.min
    delta_max = params.delta_bounds_given_crval_bounds.max
    if not (np.isfinite(input_crval_bounds.min) and np.isfinite(input_crval_bounds.max)):
        assert delta_min == -1
        assert delta_max == 1
    else:
        assert -1.0 <= delta_min <= 1.0
        assert -1.0 <= delta_max <= 1.0
        if not (np.allclose(delta_min, -1.0) or np.allclose(delta_max, 1.0)):
            # This is a weird way to ask `input_crval_bounds.[min,max] in `crval_at_corners`
            # We need to do this because floating point precision is crazy wack with `Quantities`
            # print(f"{crval_at_corners.to(u.nm) = }\n{input_crval_bounds.min = }")
            np.testing.assert_allclose(
                np.min(np.abs(crval_at_corners - input_crval_bounds.min)).to_value("nm"),
                0.0,
                atol=1e-5,
            )
            np.testing.assert_allclose(
                np.min(np.abs(crval_at_corners - input_crval_bounds.max)).to_value("nm"),
                0.0,
                atol=1e-5,
            )


@pytest.mark.parametrize(
    "crval_free", [pytest.param(True, id="crval_free"), pytest.param(False, id="crval_fixed")]
)
@given(valid_bounds_and_params=sane_wcs_params_and_bounds(completely_sane_angles=True))
@example(valid_bounds_and_params=sane_bounds_with_360angle_and_0_min)
@settings(max_examples=1000)
def test_angle_bounds(
    valid_bounds_and_params: SaneBoundsAndWCSParams, crval_free, test_parameters_dict
):
    """
    Given: A `WavelengthCalibrationParameters` object
    When: Updating the bounds on `incident_light_angle` if `crval` is fixed in the fit
    Then: The updated bounds are only within the original bounds, and contain no invalid WCS values
    """
    input_angle_bounds = valid_bounds_and_params.incident_light_angle_bounds

    params = WavelengthCalibrationParameters(
        crval=valid_bounds_and_params.crval,
        dispersion=test_parameters_dict["dispersion"],
        incident_light_angle=valid_bounds_and_params.incident_light_angle,
        opacity_factor=test_parameters_dict["opacity_factor"],
        grating_constant=valid_bounds_and_params.grating_constant,
        doppler_velocity=test_parameters_dict["doppler_velocity"],
        order=valid_bounds_and_params.order,
        bounds=BoundsModel(crval=LengthBoundRange(), incident_light_angle=input_angle_bounds),
        fit_flags=FitFlagsModel(crval=crval_free),
    )

    new_angle_bounds = params.bounds.incident_light_angle
    if crval_free:
        assert new_angle_bounds == input_angle_bounds
    else:
        # Make sure we didn't grow the range
        old_range = input_angle_bounds.max - input_angle_bounds.min
        new_range = new_angle_bounds.max - new_angle_bounds.min
        assert new_range <= old_range or np.allclose(new_range, old_range)

        # Make sure the value is still in the range
        assert new_angle_bounds.min <= params.incident_light_angle <= new_angle_bounds.max

        # Make sure the range doesn't include any values that generate an invalid WCS
        delta = float(params.crval * params.grating_constant * params.order)
        angle_vector = np.linspace(new_angle_bounds.min, new_angle_bounds.max, 1000)
        delta_range = delta - np.sin(angle_vector)
        invalid_idx = (delta_range > 1.0) | (delta_range < -1.0)
        assert np.sum(invalid_idx) == 0


@pytest.mark.parametrize(
    "angle_bounds",
    [
        pytest.param(AngleBoundRange(min=-10 * u.deg, max=360 * u.deg), id="negative"),
        pytest.param(AngleBoundRange(min=-12141 * u.deg, max=246243 * u.deg), id="outside_range"),
    ],
)
def test_insane_angle_values_when_crval_fixed(test_parameters_dict, angle_bounds):
    """
    Given: A set of parameters where the `incident_light_angle` or its bounds are outside the range [0, 360]
    When: Instantiatin a `WavelengthCalibrationParameters` object with `crval` fixed in the fit
    Then: The correct error is raised
    """
    err_str = "With `crval` fixed in the fit the `incident_light_angle"
    with pytest.raises(ValidationError, match=err_str):
        _ = WavelengthCalibrationParameters(
            bounds=BoundsModel(incident_light_angle=angle_bounds),
            fit_flags=FitFlagsModel(crval=False),
            **test_parameters_dict
        )

    params_with_out_of_range_angle = test_parameters_dict | {
        "incident_light_angle": 360 * u.deg + test_parameters_dict["incident_light_angle"]
    }
    with pytest.raises(ValidationError, match=err_str):
        _ = WavelengthCalibrationParameters(
            bounds=BoundsModel(incident_light_angle=AngleBoundRange()),
            fit_flags=FitFlagsModel(crval=False),
            **params_with_out_of_range_angle
        )


def test_too_close_valid_WCS(test_parameters_dict):
    """
    Given: A set of parameters that represents a valid WCS, but is so close to invalidity that there is no room
      to adjust the range on `incident_light_angle`
    When: Instantiating `WavelengthCalibrationParameters` with these parameters and fixing crval
    Then: The correct error is raised
    """
    close_delta = 1 - np.finfo(float).eps
    close_angle = (90 - np.finfo(float).eps) * u.deg
    crval = (close_delta + np.sin(close_angle)) / (
        test_parameters_dict["grating_constant"] * test_parameters_dict["order"]
    )
    input_params = test_parameters_dict | {"crval": crval, "incident_light_angle": close_angle}
    with pytest.raises(ValidationError, match="Input values are np.allclose to an invalid WCS"):
        _ = WavelengthCalibrationParameters(fit_flags=FitFlagsModel(crval=False), **input_params)


def test_default_fit_flags(test_parameters_dict):
    """
    Given: No custom input fit flags
    When: Initializing the fit parameters
    Then: Each parameter's fit flag is the default value (True)
    """
    input_params = WavelengthCalibrationParameters(
        crval=test_parameters_dict["crval"],
        dispersion=test_parameters_dict["dispersion"],
        incident_light_angle=test_parameters_dict["incident_light_angle"],
        resolving_power=test_parameters_dict["resolving_power"],
        opacity_factor=test_parameters_dict["opacity_factor"],
        straylight_fraction=test_parameters_dict["straylight_fraction"],
        grating_constant=test_parameters_dict["grating_constant"],
        doppler_velocity=test_parameters_dict["doppler_velocity"],
        order=test_parameters_dict["order"],
    )

    assert input_params.fit_flags.crval == True
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
        crval=test_parameters_dict["crval"],
        dispersion=test_parameters_dict["dispersion"],
        incident_light_angle=test_parameters_dict["incident_light_angle"],
        resolving_power=test_parameters_dict["resolving_power"],
        opacity_factor=test_parameters_dict["opacity_factor"],
        straylight_fraction=test_parameters_dict["straylight_fraction"],
        grating_constant=test_parameters_dict["grating_constant"],
        doppler_velocity=test_parameters_dict["doppler_velocity"],
        order=test_parameters_dict["order"],
        fit_flags=fit_flags,
    )

    assert input_params.fit_flags.crval == True
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
        "crval": {
            "min": test_parameters_dict["crval"] - (0.3 * u.nm),
            "max": test_parameters_dict["crval"] + (0.3 * u.nm),
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
        crval=test_parameters_dict["crval"],
        dispersion=test_parameters_dict["dispersion"],
        incident_light_angle=test_parameters_dict["incident_light_angle"],
        resolving_power=test_parameters_dict["resolving_power"],
        opacity_factor=test_parameters_dict["opacity_factor"],
        straylight_fraction=test_parameters_dict["straylight_fraction"],
        grating_constant=test_parameters_dict["grating_constant"],
        doppler_velocity=test_parameters_dict["doppler_velocity"],
        order=test_parameters_dict["order"],
        bounds=bounds,
    )

    np.testing.assert_allclose(
        input_params.bounds.crval.min, test_parameters_dict["crval"] - (0.3 * u.nm)
    )
