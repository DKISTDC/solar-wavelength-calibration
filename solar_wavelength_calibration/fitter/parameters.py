"""Objects for storing fit parameters."""
from __future__ import annotations

import logging
from functools import cached_property
from typing import Annotated
from typing import Callable

import astropy.units as u
import numpy as np
from astropy.units import Quantity
from lmfit import Parameters
from pydantic import AfterValidator
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import field_validator
from pydantic import FiniteFloat
from pydantic import model_validator
from pydantic_core.core_schema import ValidationInfo

logger = logging.getLogger(__name__)

__all__ = [
    "WavelengthCalibrationParameters",
    "BoundsModel",
    "FitFlagsModel",
    "UnitlessBoundRange",
    "LengthBoundRange",
    "AngleBoundRange",
    "DispersionBoundRange",
]


def to_nm(value: Quantity) -> Quantity:
    """Convert a length unit to nm."""
    return value.to(u.nm)


def to_deg(value: Quantity) -> Quantity:
    """Convert an angle unit to degrees."""
    return value.to(u.deg)


def to_nm_per_px(value: Quantity) -> Quantity:
    """Convert a dispersion unit to nm / px."""
    return value.to(u.nm / u.pix)


def to_over_m(value: Quantity) -> Quantity:
    """Convert an inverse length unit to 1 / m."""
    return value.to(1 / u.m)


def to_km_per_s(value: Quantity) -> Quantity:
    """Convert a speed unit to km / s."""
    return value.to(u.km / u.s)


def is_finite(value: Quantity) -> Quantity:
    """Raise a `ValueError` if the input is not finite."""
    if not np.isfinite(value):
        raise ValueError("parameters must be finite")
    return value


length_quantity = Annotated[Quantity, AfterValidator(to_nm)]
angle_quantity = Annotated[Quantity, AfterValidator(to_deg)]
dispersion_quantity = Annotated[Quantity, AfterValidator(to_nm_per_px)]
inverse_length_quantity = Annotated[Quantity, AfterValidator(to_over_m)]
speed_quantity = Annotated[Quantity, AfterValidator(to_km_per_s)]

finite_validator = AfterValidator(is_finite)
finite_length_quantity = Annotated[length_quantity, finite_validator]
finite_angle_quantity = Annotated[angle_quantity, finite_validator]
finite_dispersion_quantity = Annotated[dispersion_quantity, finite_validator]
finite_inverse_length_quantity = Annotated[inverse_length_quantity, finite_validator]
finite_speed_quantity = Annotated[speed_quantity, finite_validator]


class UnitlessBoundRange(BaseModel):
    """Represents a min and max bound for a parameter."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    min: int | float = -np.inf
    max: int | float = np.inf

    @model_validator(mode="after")
    def ensure_min_less_than_max(self) -> UnitlessBoundRange:
        """Ensure that the min and max properties are ordered correctly."""
        if self.min >= self.max:
            raise ValueError(f"{self.min = } is greater than or equal to {self.max = }")
        return self


class LengthBoundRange(UnitlessBoundRange):
    """BoundRange subclass that enforces inputs to have 'length' units."""

    min: length_quantity = -np.inf * u.nm
    max: length_quantity = np.inf * u.nm


class AngleBoundRange(UnitlessBoundRange):
    """BoundRange subclass that enforces inputs to be angles (rad or deg)."""

    min: angle_quantity = -np.inf * u.deg
    max: angle_quantity = np.inf * u.deg


class DispersionBoundRange(UnitlessBoundRange):
    """BoundRange subclass that enforces inputs to have 'length / pix' units."""

    min: dispersion_quantity = -np.inf * u.nm / u.pix
    max: dispersion_quantity = np.inf * u.nm / u.pix


class BoundsModel(BaseModel):
    r"""
    Container for \*BoundRanges that match the fit parameters in `WavelengthCalibrationParameters`.

    This container also defines sensible defaults.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    crval: LengthBoundRange = LengthBoundRange(min=0 * u.nm)
    dispersion: DispersionBoundRange = DispersionBoundRange(min=0 * u.nm / u.pix)
    incident_light_angle: AngleBoundRange = AngleBoundRange(min=0 * u.deg, max=180 * u.deg)
    resolving_power: UnitlessBoundRange = UnitlessBoundRange(min=0)
    opacity_factor: UnitlessBoundRange = UnitlessBoundRange(min=0)
    straylight_fraction: UnitlessBoundRange = UnitlessBoundRange(min=0)
    continuum_level: UnitlessBoundRange = UnitlessBoundRange(min=0)


class FitFlagsModel(BaseModel):
    """
    Base model for fit flags with default values set to True.

    True indicates that the parameter should be fit.
    """

    crval: bool = True
    dispersion: bool = True
    incident_light_angle: bool = True
    resolving_power: bool = True
    opacity_factor: bool = True
    straylight_fraction: bool = True
    continuum_level: bool = True


class WavelengthCalibrationParameters(BaseModel):
    """Object for initializing the fit parameters prior to a fit."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    crval: finite_length_quantity
    dispersion: finite_dispersion_quantity
    incident_light_angle: finite_angle_quantity
    grating_constant: finite_inverse_length_quantity
    doppler_velocity: finite_speed_quantity
    order: int
    continuum_level: float = 1.0
    resolving_power: int = 1
    opacity_factor: FiniteFloat = 1.0
    straylight_fraction: FiniteFloat = 0.0
    fit_flags: FitFlagsModel = FitFlagsModel()
    bounds: BoundsModel = BoundsModel()

    @field_validator("bounds")
    @classmethod
    def ensure_value_in_bounds_range(cls, bounds: BoundsModel, info: ValidationInfo) -> BoundsModel:
        """
        Validate that each parameter value falls within its corresponding bounds range.

        This method compares the current model values (from the `info` context) to their associated
        min and max bounds and raises an error if any value falls outside its allowed range.
        """
        for param in bounds.model_fields.keys():
            try:
                param_value = info.data[param]
            except KeyError:
                # if param not in info.data, it is probably an invalid type and Pydantic will handle this later in the code.
                continue
            bounds_range = getattr(bounds, param)
            if not (bounds_range.min <= param_value <= bounds_range.max):
                raise ValueError(
                    f"{param} with value {param_value} is outside the bounds range of ({bounds_range.min}, {bounds_range.max})"
                )

        return bounds

    @model_validator(mode="after")
    def ensure_invertible_WCS(self) -> WavelengthCalibrationParameters:
        r"""
        Make sure the input parameters values represent a physically valid grating.

        Specifically, we need parameters that result in an invertible grating equation. Simplifying Eq. (84) from
        `Greisen et al (2006) <https://ui.adsabs.harvard.edu/abs/2006A%26A...446..747G/abstract>`_ for the case of a
        diffraction grating in air, the refracted angle, :math:`\gamma`, is given by

        .. math::
            \gamma = \sin^{-1}(\lambda G m - \sin\alpha),

        where :math:`\lambda` is `crval`, :math:`G` is `grating_constant`, :math:`m` is `order`, and :math:`\alpha` is
        `incident_light_angle`. The arcsine is only valid for inputs in the range [-1, 1] and so a combination of
        `crval`, `grating_constant`, `order`, and `incident_light_angle` that fall outside of this range are considered
        invalid and will raise an Error.
        """
        if abs(self.delta) > 1:
            raise ValueError(
                "Input parameter values represent an un-invertible grating equation. "
                "We need `crval * grating_constant * order - sin(incident_light_angle)` to be in the range "
                f"[-1, 1] but the values provide result in {self.delta}"
            )
        return self

    @property
    def delta(self) -> float:
        r"""
        Compute initial guess for internal lmfit `_delta` parameter.

        To ensure an invertible grating equation (see `ensure_invertible_WCS`) we need

        .. math::
            -1 \leq \lambda G m - \sin\alpha \leq 1

        where :math:`G` (`grating_constant`) and :math:`m` (`order`) are fixed parameters while
        :math:`\lambda` (`crval`) and :math:`\alpha` (`incident_light_angle`) are free in the fit.

        As suggested by `lmfit <https://lmfit.github.io/lmfit-py/constraints.html#using-inequality-constraints>`_ we
        ensure this constraint by forcing CRVAL to follow the equation

        .. math::
            \lambda = \frac{\delta + \sin\alpha}{G m}

        where :math:`\delta` (`_delta`) is a free parameter whose bounds are [-1, 1].

        This property is the initial guess of :math:`\delta_0 = \lambda_0 G m - \sin\alpha_0`.

        NOTE: `_delta` is only used if `crval` is set to vary during the fit (see `FitFlagsModel`).
        """
        delta = self.crval * self.grating_constant * self.order - np.sin(self.incident_light_angle)

        # `float` fully simplifies all units to be just a number
        return float(delta)

    @property
    def crval_constraint_function(self) -> Callable[[float, float], float]:
        """
        Return a function that computes `crval` given values of `_delta` and `incident_light_angle`.

        See `delta` for more information.
        """
        # Note that inside the minimization call all parameters do NOT have units; they are just bare `floats`
        # That's why we need to use `np.deg2rad` in the call to `np.sin` and use `to_value` in the hardcoded denominator
        # (u.nm) as the unit for `crval` derives from `length_quantity`
        def constrain_crval(_delta: float, incident_light_angle_deg: float) -> float:
            numerator = _delta + np.sin(np.deg2rad(incident_light_angle_deg))
            denominator = (self.grating_constant * self.order).to_value(1 / u.nm)
            return numerator / denominator

        return constrain_crval

    @cached_property
    def delta_bounds_given_crval_bounds(self) -> UnitlessBoundRange:
        """
        Adjust the internal bounds on `_delta` to approximate the desired range on `crval`.

        Because `crval` is, internally, a dependent variable that is a function of `incident_light_angle` and `_delta`
        (see `crval_constraint_function` and `delta`) the user-supplied bounds on `crval` can't be strictly honored.
        This method computes modified internal bounds on `_delta` such that the search space of `crval` is as close to
        the input bounds as possible.

        Note that, because the bounds on `_delta` must be defined independent of the bounds on `incident_light_angle`,
        the true range of `crval` values considered will likely fall outside the input bounds placed on `crval`. The
        amount of "bleed" outside the user-supplied bounds is correlated with the width of the bounds on
        `incident_light_angle`; as the angle is given tighter bounds the *true* range of `crval` values gets closer to
        the input bounds on `crval`. When `incident_light_angle` is fixed the true range of `crval` values is equal to
        the input bounds on `crval`.
        """
        if not (np.isfinite(self.bounds.crval.min) and np.isfinite(self.bounds.crval.max)):
            return UnitlessBoundRange(min=-1.0, max=1.0)

        logger.info(
            "Updating internal bounds on `_delta` to approximate the desired bounds on `crval` "
            f"([{self.bounds.crval.min}, {self.bounds.crval.max}])"
        )
        # We need pretty high resolution to get a good range in _delta. 1000 seems to work well enough
        # (this is all an approximation anyway).
        num_points = 1000

        # If the angle is free then consider the full range given by its input bounds
        if self.fit_flags.incident_light_angle:
            input_angle_min = self.bounds.incident_light_angle.min
            input_angle_max = self.bounds.incident_light_angle.max

            # (Or the default bounds, if the input bounds are ±`np.inf`)
            default_angle_bounds = BoundsModel().incident_light_angle
            if not np.isfinite(input_angle_min):
                input_angle_min = default_angle_bounds.min
                logger.info(
                    f"WARNING: Given `incident_light_angle` minimum bound is not finite. Using default min of {input_angle_min} instead (only to compute bounds on `_delta`)."
                )

            if not np.isfinite(input_angle_max):
                input_angle_max = default_angle_bounds.max
                logger.info(
                    f"WARNING: Given `incident_light_angle` maximum bound is not finite. Using default max of {input_angle_max} instead (only to compute bounds on `_delta`)."
                )

            angles = np.linspace(input_angle_min, input_angle_max, num_points)
        # Otherwise only consider the actual value
        else:
            angles = u.Quantity([self.incident_light_angle])

        delta_at_min_crval = self.bounds.crval.min * self.grating_constant * self.order - np.sin(
            angles
        )
        delta_at_max_crval = self.bounds.crval.max * self.grating_constant * self.order - np.sin(
            angles
        )

        min_delta = np.min(np.r_[delta_at_min_crval, delta_at_max_crval])
        max_delta = np.max(np.r_[delta_at_min_crval, delta_at_max_crval])

        # Have to make sure we're still in the range of valid WCS
        min_delta = max(min_delta, -1.0)
        max_delta = min(max_delta, 1.0)

        new_delta_bounds = UnitlessBoundRange(min=min_delta, max=max_delta)

        logger.info(f"New bounds on _delta are = [{new_delta_bounds.min}, {new_delta_bounds.max}]")
        return new_delta_bounds

    @model_validator(mode="after")
    def ensure_sane_light_angle_values_if_crval_fixed(self) -> WavelengthCalibrationParameters:
        """
        Check that the `input_light_angle` and its bounds are in the range [0, 360] iff `crval` is fixed in the fit.

        We do this to make sure the inputs to `update_incident_light_angle_if_crval_fixed` are sane, which removes
        the need to check lots of edge cases in that method.

        If the value for `incident_light_angle` is 360 deg and the minimum bound is 0 deg then the value of
        `incident_light_angle` is changed to 0 deg.
        """
        if self.fit_flags.crval:
            return self

        if not np.isfinite(self.bounds.incident_light_angle.min):
            self.bounds.incident_light_angle.min = 0 * u.deg
        if not np.isfinite(self.bounds.incident_light_angle.max):
            self.bounds.incident_light_angle.max = 360 * u.deg
        if (
            self.incident_light_angle == 360 * u.deg
            and self.bounds.incident_light_angle.min == 0 * u.deg
        ):
            self.incident_light_angle = 0 * u.deg

        values_to_check = Quantity(
            [
                self.bounds.incident_light_angle.min,
                self.incident_light_angle,
                self.bounds.incident_light_angle.max,
            ]
        )
        if any(values_to_check > 360.0 * u.deg) or any(values_to_check < 0.0 * u.deg):
            raise ValueError(
                "With `crval` fixed in the fit the `incident_light_angle` and its bounds must be in the "
                "range [0, 360]."
            )

        return self

    @model_validator(mode="after")
    def update_incident_light_angle_if_crval_fixed(self) -> WavelengthCalibrationParameters:
        """
        Potentially update the bounds on `incident_light_angle` to ensure we stay in a valid region of the fit.

        We only need to update the `incident_light_angle` bounds if `crval` is fixed in the fit (see `fit_flags`). In
        this case, the validity assurance provided by `delta` does not apply, and so we need to ensure validity by
        constraining `incident_light_angle` to only span a valid parameter space.

        If either of the given `bounds` on `incident_light_angle` is already within the valid region it will be
        unchanged. In other words, this method will only ever shrink bounds, not expand them.
        """
        if self.fit_flags.crval:
            return self

        logger.info(
            "Updating bounds on `incident_light_angle` to maintain valid WCS (this is needed because `crval` is fixed"
        )

        input_min = self.bounds.incident_light_angle.min
        input_max = self.bounds.incident_light_angle.max

        if np.allclose(
            [input_min, self.incident_light_angle, input_max], self.incident_light_angle
        ):
            logger.info(
                "`incident_light_angle` and its bounds are np.allclose to each other. "
                "Is this really what you want?"
            )
            return self

        num_points = 100000
        # Make two separate vectors to ensure we get the actual value included in the array of angles we're considering
        # Thus, if the min/max valid angle is the actual input value then we'll see that here
        min_to_value_vector = np.linspace(input_min, self.incident_light_angle, num_points // 2)
        value_to_max_vector = np.linspace(self.incident_light_angle, input_max, num_points // 2)
        angle_vector = np.r_[min_to_value_vector, value_to_max_vector]

        # Float reduces and cancels out all units
        length_expression = float(self.crval * self.grating_constant * self.order)

        delta = length_expression - np.sin(angle_vector)
        valid_idx = (delta >= -1.0) & (delta <= 1.0)

        # Find contiguous regions of validity
        edges = np.diff(valid_idx.astype(int))
        starts = np.where(edges == 1)[0] + 1
        ends = np.where(edges == -1)[0]
        if valid_idx[0]:
            starts = np.r_[0, starts]
        if valid_idx[-1]:
            ends = np.r_[ends, valid_idx.size - 1]

        # The `filter` here ignores any regions that only have a single value in them
        valid_border_idx = list(filter(lambda b: b[0] != b[1], zip(starts, ends)))

        # Use the region of validity that contains the input value
        valid_ranges = [angle_vector[b[0] : b[1] + 1] for b in valid_border_idx]
        valid_angles = [
            r for r in valid_ranges if np.min(r) <= self.incident_light_angle <= np.max(r)
        ][0]

        min_angle = np.min(valid_angles)
        max_angle = np.max(valid_angles)

        if np.allclose(Quantity([min_angle, max_angle]), self.incident_light_angle):
            raise ValueError(
                "Input values are np.allclose to an invalid WCS and therefore no valid range of "
                "`incident_light_angle` exists. You should either fix `incident_light_angle` in the fit "
                "or examine your input parameters."
            )

        logger.info(f"New bounds on `incident_light_angle` are [{min_angle}, {max_angle}]")
        new_bounds = AngleBoundRange(min=min_angle, max=max_angle)
        self.bounds.incident_light_angle = new_bounds

        return self

    @property
    def lmfit_parameters(self) -> Parameters:
        """Create a single lmfit `Parameters` object with custom bounds and free/fixed status."""
        param = Parameters()

        # Give lmfit knowledge of the constraint function
        # Fear not! This accessing of the private API is the official method recommended by lmfit:
        #  https://lmfit.github.io/lmfit-py/constraints.html#advanced-usage-of-expressions-in-lmfit
        param._asteval.symtable["constrain_crval"] = self.crval_constraint_function

        param.add(
            "dispersion",
            vary=self.fit_flags.dispersion,
            value=self.dispersion.value,
            min=self.bounds.dispersion.min.value,
            max=self.bounds.dispersion.max.value,
        )
        param.add(
            "incident_light_angle",
            vary=self.fit_flags.incident_light_angle,
            value=self.incident_light_angle.value,
            min=self.bounds.incident_light_angle.min.value,
            max=self.bounds.incident_light_angle.max.value,
        )

        if self.fit_flags.crval:
            param.add(
                "_delta",
                vary=self.fit_flags.crval,
                value=self.delta,
                min=self.delta_bounds_given_crval_bounds.min,
                max=self.delta_bounds_given_crval_bounds.max,
            )
            param.add(
                "crval",
                expr="constrain_crval(_delta, incident_light_angle)",
            )
        else:
            param.add("crval", vary=False, value=self.crval.value)

        param.add(
            "resolving_power",
            vary=self.fit_flags.resolving_power,
            value=self.resolving_power,
            min=self.bounds.resolving_power.min,
            max=self.bounds.resolving_power.max,
        )
        param.add(
            "opacity_factor",
            vary=self.fit_flags.opacity_factor,
            value=self.opacity_factor,
            min=self.bounds.opacity_factor.min,
            max=self.bounds.opacity_factor.max,
        )
        param.add(
            "straylight_fraction",
            vary=self.fit_flags.straylight_fraction,
            value=self.straylight_fraction,
            min=self.bounds.straylight_fraction.min,
            max=self.bounds.straylight_fraction.max,
        )

        param.add(
            "continuum_level",
            vary=self.fit_flags.continuum_level,
            value=self.continuum_level,
            min=self.bounds.continuum_level.min,
            max=self.bounds.continuum_level.max,
        )
        return param

    @property
    def constant_parameters(self) -> dict:
        """Create a single dictionary with correct starting values for the constant instrument specific parameters that will not vary."""
        # Initialize a dict to hold the constant parameters and their values
        constant_params = {}
        constant_params["grating_constant"] = self.grating_constant.value
        constant_params["doppler_velocity"] = self.doppler_velocity
        constant_params["order"] = self.order
        return constant_params
