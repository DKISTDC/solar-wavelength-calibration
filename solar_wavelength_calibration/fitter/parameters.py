"""Objects for storing fit parameters."""
from __future__ import annotations

import logging
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
from pydantic import model_validator
from pydantic_core.core_schema import ValidationInfo

logger = logging.getLogger(__name__)


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


length_quantity = Annotated[Quantity, AfterValidator(to_nm)]
angle_quantity = Annotated[Quantity, AfterValidator(to_deg)]
dispersion_quantity = Annotated[Quantity, AfterValidator(to_nm_per_px)]
inverse_length_quantity = Annotated[Quantity, AfterValidator(to_over_m)]
speed_quantity = Annotated[Quantity, AfterValidator(to_km_per_s)]


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

    crval1: LengthBoundRange = LengthBoundRange(min=0 * u.nm)
    dispersion: DispersionBoundRange = DispersionBoundRange(min=0 * u.nm / u.pix)
    incident_light_angle: AngleBoundRange = AngleBoundRange(min=0 * u.deg, max=180 * u.deg)
    resolving_power: UnitlessBoundRange = UnitlessBoundRange(min=0)
    opacity_factor: UnitlessBoundRange = UnitlessBoundRange(min=0)
    straylight_fraction: UnitlessBoundRange = UnitlessBoundRange(min=0)


class FitFlagsModel(BaseModel):
    """
    Base model for fit flags with default values set to True.

    True indicates that the parameter should be fit.
    """

    crval1: bool = True
    dispersion: bool = True
    incident_light_angle: bool = True
    resolving_power: bool = True
    opacity_factor: bool = True
    straylight_fraction: bool = True
    continuum_intensity_correction: bool = True


class WavelengthCalibrationParameters(BaseModel):
    """Object for initializing the fit parameters prior to a fit."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    crval1: length_quantity
    dispersion: dispersion_quantity
    incident_light_angle: angle_quantity
    unfit_1d_spectrum: np.ndarray
    grating_constant: inverse_length_quantity
    doppler_velocity: speed_quantity
    order: int
    resolving_power: int = 1
    opacity_factor: float = 1
    straylight_fraction: float = 0
    quartile: int = 75
    fit_flags: FitFlagsModel = FitFlagsModel()
    bounds: BoundsModel = BoundsModel()

    @field_validator("quartile", mode="after")
    @classmethod
    def validate_quartile(cls, value):
        """Ensure the quartile value is between 0 and 100."""
        if not (0 <= value <= 100):
            raise ValueError("Quartile value must be between 0 and 100.")
        return value

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
                # if param not in info.data, it is probably an invalid type and Pydantic will handle this soon.
                continue
            bounds_range = getattr(bounds, param)
            if not (bounds_range.min <= param_value <= bounds_range.max):
                raise ValueError(
                    f"{param} with value {param_value} is outside the bounds range of ({bounds_range.min}, {bounds_range.max})"
                )

        return bounds

    @property
    def delta(self) -> float:
        r"""
        Compute initial guess for internal lmfit `_delta` parameter.

        To ensure an invertible grating equation we need

        .. math::
            -1 \leq G m \lambda - \sin\alpha \leq 1

        where :math:`G` (grating constant) and :math:`m` (diffraction order) are fixed parameters while
        :math:`\lambda` (CRVAL1) and :math:`\alpha` (incident light angle) are free in the fit.

        As suggested by `lmfit <https://lmfit.github.io/lmfit-py/constraints.html#using-inequality-constraints>`_ we
        ensure this constraint by forcing CRVAL1 to follow the equation

        .. math::
            \lambda = \frac{\delta + \sin(\alpha)}{G m}

        where :math:`\delta` (`_delta`) is a free parameter whose bounds are [-1, 1].
        """
        unit_value = (self.grating_constant * self.order * self.crval1).to_value(
            u.dimensionless_unscaled
        )
        sin_value = np.sin(self.incident_light_angle.to_value("rad"))
        return unit_value - sin_value

    @property
    def crval1_constraint_function(self) -> Callable[[float, float], float]:
        """
        Return a function that computes `crval1` given values of `_delta` and `incident_light_angle`.

        See `delta` for more information.
        """

        def constrain_crval1(_delta: float, incident_light_angle_deg: float) -> float:
            numerator = _delta + np.sin(np.deg2rad(incident_light_angle_deg))
            denominator = (self.grating_constant * self.order).to_value(1 / u.nm)
            return numerator / denominator

        return constrain_crval1

    @property
    def lmfit_parameters(self) -> Parameters:
        """Create a single lmfit `Parameters` object with custom bounds and free/fixed status."""
        param = Parameters()

        # Give lmfit knowledge of the constraint function
        # Fear not! This accessing of the private API is the official method recommended by lmfit:
        #  https://lmfit.github.io/lmfit-py/constraints.html#advanced-usage-of-expressions-in-lmfit
        param._asteval.symtable["constrain_crval1"] = self.crval1_constraint_function

        param.add(
            "dispersion",
            vary=self.fit_flags.dispersion,
            value=self.dispersion.value,
            min=self.bounds.dispersion.min.value,
            max=self.bounds.dispersion.max.value,
        )
        param.add(
            "_delta", vary=self.fit_flags.incident_light_angle, value=self.delta, min=-1, max=1.0
        )
        param.add(
            "incident_light_angle",
            vary=self.fit_flags.incident_light_angle,
            value=self.incident_light_angle.value,
            min=self.bounds.incident_light_angle.min.value,
            max=self.bounds.incident_light_angle.max.value,
        )
        param.add(
            "crval1",
            expr="constrain_crval1(_delta, incident_light_angle)",
        )
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

        spectrum_percentile = np.nanpercentile(self.unfit_1d_spectrum, self.quartile)
        param.add(
            "continuum_intensity_correction",
            vary=self.fit_flags.continuum_intensity_correction,
            value=spectrum_percentile,
            min=-np.inf,
            max=np.inf,
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
