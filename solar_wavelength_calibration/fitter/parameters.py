"""Objects for storing fit parameters."""
import logging

import astropy.units as u
import numpy as np
from astropy.units import Quantity
from lmfit import Parameters
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import field_validator
from pydantic import model_validator
from pydantic_core.core_schema import ValidationInfo

logger = logging.getLogger(__name__)

QUANTITY_OR_NUMBER = Quantity | float | int


class BoundRange(BaseModel):
    """Represents a min and max bound for a parameter."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    min: QUANTITY_OR_NUMBER = -np.inf
    max: QUANTITY_OR_NUMBER = np.inf

    @model_validator(mode="after")
    def validate_units_match(self) -> "BoundRange":
        """Ensure that if both min and max are Quantities, they have the same physical units."""
        if isinstance(self.min, Quantity) or isinstance(self.max, Quantity):
            if not self.min.unit.is_equivalent(self.max.unit):
                raise ValueError(
                    f"Incompatible units: min has unit {self.min.unit}, "
                    f"but max has unit {self.max.unit}"
                )
        return self


class BoundsModel(BaseModel):
    """Base model for bounds with default values set to None."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    crval1: BoundRange = BoundRange()
    dispersion: BoundRange = BoundRange(min=0 * u.nm / u.pix, max=np.inf * u.nm / u.pix)
    incident_light_angle: BoundRange = BoundRange(min=0 * u.deg, max=360 * u.deg)
    resolving_power: BoundRange = BoundRange(min=0)
    opacity_factor: BoundRange = BoundRange(min=0)
    straylight_fraction: BoundRange = BoundRange(min=0)


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

    crval1: Quantity
    dispersion: Quantity
    incident_light_angle: Quantity
    unfit_1d_spectrum: np.ndarray
    grating_constant: Quantity
    doppler_velocity: Quantity
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

    @field_validator("crval1")
    @classmethod
    def validate_crval1_units(cls, value):
        """Ensure crval1 has correct length units."""
        return value.to(u.nm)

    @field_validator("incident_light_angle")
    @classmethod
    def validate_incident_light_angle_units(cls, value):
        """Ensure incident_light_angle has correct length units."""
        return value.to(u.deg)

    @field_validator("dispersion")
    @classmethod
    def validate_dispersion_units(cls, value):
        """Ensure dispersion has correct length units."""
        return value.to(u.nm / u.pix)

    @field_validator("grating_constant")
    @classmethod
    def validate_grating_constant_units(cls, value):
        """Ensure grating_constant has correct length units."""
        return value.to(1 / u.mm)

    @field_validator("doppler_velocity")
    @classmethod
    def validate_doppler_velocity_units(cls, value):
        """Ensure doppler_velocity has correct length units."""
        return value.to(u.km / u.s)

    @field_validator("bounds")
    @classmethod
    def ensure_correct_bounds_units(cls, bounds: BoundsModel, info: ValidationInfo) -> BoundsModel:
        """
        Ensure that all bounds in the BoundsModel have units compatible with their corresponding parameter values.

        For parameters that are expected to have units (e.g., crval1, dispersion, incident_light_angle),
        this method converts the bound values to match the units of the corresponding parameter in the model input.

        For unitless parameters, it raises an error if bounds are incorrectly specified with units.
        """
        params_with_units = ["crval1", "dispersion", "incident_light_angle"]
        for param in bounds.model_fields.keys():
            try:
                param_value = info.data[param]
            except KeyError:
                continue
            bounds_range = getattr(bounds, param)
            if param in params_with_units:
                param_unit = param_value.unit
                if isinstance(bounds_range.min, Quantity):
                    new_min = bounds_range.min.to(param_unit)
                    new_max = bounds_range.max.to(param_unit)
                else:
                    new_min = bounds_range.min * param_unit
                    new_max = bounds_range.max * param_unit
                new_range = BoundRange(min=new_min, max=new_max)
                setattr(bounds, param, new_range)

            else:
                if isinstance(bounds_range.min, Quantity):
                    raise TypeError(
                        f"{param} has units ({bounds_range.min.unit}), but it shouldn't."
                    )

        return bounds

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
                raise TypeError(
                    f"{param} with value {param_value} is outside the bounds range of ({bounds_range.min}, {bounds_range.max})"
                )

        return bounds

    @property
    def lmfit_parameters(self) -> Parameters:
        """Create a single lmfit `Parameters` object with custom bounds and free/fixed status."""
        param = Parameters()

        param.add(
            "crval1",
            vary=self.fit_flags.crval1,
            value=self.crval1.value,
            min=self.bounds.crval1.min.value,
            max=self.bounds.crval1.max.value,
        )
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
