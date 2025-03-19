"""Objects for storing fit parameters."""
import logging
from typing import Optional
from typing import Union

import astropy.units as u
import numpy as np
from astropy.units import Quantity
from lmfit import Parameters
from numpydantic import NDArray
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import field_validator

logger = logging.getLogger(__name__)


class BoundsModel(BaseModel):
    """Base model for bounds with default values set to None."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    crval1: Optional[list[Union[Quantity, float, int]]] = None
    dispersion: Optional[list[Union[Quantity, float, int]]] = None
    incident_light_angle: Optional[list[Union[Quantity, float, int]]] = None
    resolving_power: Optional[list[Union[Quantity, float, int]]] = None
    opacity_factor: Optional[list[Union[Quantity, float, int]]] = None
    straylight_fraction: Optional[list[Union[Quantity, float, int]]] = None
    continuum_intensity_correction: Optional[list[Union[Quantity, float, int]]] = None

    @field_validator(
        "crval1",
        "dispersion",
        "incident_light_angle",
        "resolving_power",
        "opacity_factor",
        "straylight_fraction",
        "continuum_intensity_correction",
        mode="after",
    )
    @classmethod
    def validate_and_convert_bounds(cls, value, info):
        """Ensure bounds have the correct units or convert if necessary."""
        # Define expected units for each parameter
        _expected_units = {
            "crval1": u.nm,
            "dispersion": u.nm / u.pix,
            "incident_light_angle": u.deg,
            "resolving_power": None,  # Unitless
            "opacity_factor": None,  # Unitless
            "straylight_fraction": None,  # Unitless
            "continuum_intensity_correction": None,  # Unitless
        }

        param_name = info.field_name
        expected_unit = _expected_units[param_name]

        if value is None:
            return value  # Allow None values

        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"Bounds for '{param_name}' must be a list of exactly two values.")

        validated_bounds = []
        for v in value:
            if expected_unit is not None:  # If the parameter requires a unit
                if isinstance(v, Quantity):
                    # Convert to the expected unit if possible
                    v = v.to(expected_unit)
                elif isinstance(v, (int, float)):
                    raise ValueError(
                        f"Bounds for '{param_name}' must have units of {expected_unit}."
                    )
                else:
                    raise ValueError(f"Invalid unit for '{param_name}'. Expected {expected_unit}.")
            else:
                # If parameter is unitless, ensure it's a number
                if not isinstance(v, (int, float)):
                    raise ValueError(f"Invalid bound for '{param_name}', must be a number.")

            validated_bounds.append(v)

        return validated_bounds


class FitFlagsModel(BaseModel):
    """
    Base model for fit flags with default values set to True.

    True indicates that the parameter should be fit.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    crval1: bool = True
    dispersion: bool = True
    incident_light_angle: bool = True
    resolving_power: bool = True
    opacity_factor: bool = True
    straylight_fraction: bool = True
    continuum_intensity_correction: bool = True


class WavelengthCalibrationParameters(BaseModel):
    """Object for initializing the fit parameters prior to a fit."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    crval1: Quantity
    dispersion: Quantity
    incident_light_angle: Quantity
    unfit_1d_spectrum: NDArray
    grating_constant: Quantity
    doppler_velocity: Quantity
    order: int
    resolving_power: int = 1
    opacity_factor: float = 1
    straylight_fraction: float = 0
    fit_flags: FitFlagsModel = FitFlagsModel()
    bounds: BoundsModel = BoundsModel()

    @field_validator("fit_flags", mode="before")
    @classmethod
    def validate_fit_flags(cls, value):
        """Ensure fit_flags is always an instance of FitFlagsModel, merging defaults if necessary."""
        if value is None:
            return FitFlagsModel()  # Default all to True

        if isinstance(value, dict):
            return FitFlagsModel(
                **{**FitFlagsModel().dict(), **value}
            )  # Merge defaults with user values

        if isinstance(value, FitFlagsModel):
            return value

        raise ValueError("fit_flags must be a dictionary or a FitFlagsModel instance.")

    @field_validator("crval1", mode="after")
    @classmethod
    def validate_crval1_units(cls, value):
        """Ensure crval1 has correct length units."""
        return value.to_value(u.nm)

    @field_validator("incident_light_angle")
    @classmethod
    def validate_incident_light_angle_units(cls, value):
        """Ensure incident_light_angle has correct length units."""
        return value.to_value(u.deg)

    @field_validator("dispersion")
    @classmethod
    def validate_dispersion_units(cls, value):
        """Ensure dispersion has correct length units."""
        return value.to_value(u.nm / u.pix)

    @field_validator("grating_constant")
    @classmethod
    def validate_grating_constant_units(cls, value):
        """Ensure grating_constant has correct length units."""
        return value.to_value(1 / u.mm)

    @field_validator("doppler_velocity")
    @classmethod
    def validate_doppler_velocity_units(cls, value):
        """Ensure doppler_velocity has correct length units."""
        return value.to_value(u.km / u.s)

    @field_validator("unfit_1d_spectrum", mode="after")
    @classmethod
    def validate_unfit_1d_spectrum(cls, value):
        """Ensure unfit_1d_spectrum is an NDArray."""
        if not isinstance(value, np.ndarray):
            raise TypeError("unfit_1d_spectrum must be a NumPy array.")
        return value

    def _get_bounds(self, param_name: str):
        """Retrieve bounds for a parameter, ensuring defaults exist."""
        bounds = getattr(self.bounds, param_name, None)

        if bounds is None:
            logger.info(f"Bounds for parameter '{param_name}' set to None.")
            return [0, 0]

        return [b.to_value() if isinstance(b, Quantity) else b for b in bounds]

    def _get_fit_flag(self, param_name: str) -> bool:
        """Retrieve the fit flag for a parameter, ensuring a default exists."""
        if not hasattr(self.fit_flags, param_name):
            raise KeyError(f"Fit flag for parameter '{param_name}' not found.")

        return getattr(self.fit_flags, param_name)

    @property
    def lmfit_parameters(self) -> Parameters:
        """Create a single lmfit `Parameters` object with custom bounds and free/fixed status."""
        param = Parameters()

        param.add(
            "crval1",
            vary=self._get_fit_flag("crval1"),
            value=self.crval1,
            min=self._get_bounds("crval1")[0],
            max=self._get_bounds("crval1")[1],
        )
        param.add(
            "dispersion",
            vary=self._get_fit_flag("dispersion"),
            value=self.dispersion,
            min=self.dispersion + self._get_bounds("dispersion")[0] * self.dispersion,
            max=self.dispersion + self._get_bounds("dispersion")[1] * self.dispersion,
        )
        param.add(
            "incident_light_angle",
            vary=self._get_fit_flag("incident_light_angle"),
            value=self.incident_light_angle,
            min=self._get_bounds("incident_light_angle")[0],
            max=self._get_bounds("incident_light_angle")[1],
        )
        param.add(
            "resolving_power",
            vary=self._get_fit_flag("resolving_power"),
            value=self.resolving_power,
            min=self._get_bounds("resolving_power")[0],
            max=self._get_bounds("resolving_power")[1],
        )
        param.add(
            "opacity_factor",
            vary=self._get_fit_flag("opacity_factor"),
            value=self.opacity_factor,
            min=self._get_bounds("opacity_factor")[0],
            max=self._get_bounds("opacity_factor")[1],
        )
        param.add(
            "straylight_fraction",
            vary=self._get_fit_flag("straylight_fraction"),
            value=self.straylight_fraction,
            min=self._get_bounds("straylight_fraction")[0],
            max=self._get_bounds("straylight_fraction")[1],
        )

        spectrum_percentile = np.nanpercentile(self.unfit_1d_spectrum, 75).to_value(u.nm)
        param.add(
            "continuum_intensity_correction",
            vary=self._get_fit_flag("continuum_intensity_correction"),
            value=spectrum_percentile,
            min=spectrum_percentile * self._get_bounds("continuum_intensity_correction")[0],
            max=spectrum_percentile * self._get_bounds("continuum_intensity_correction")[1],
        )
        return param

    @property
    def constant_parameters(self) -> dict:
        """Create a single dictionary with correct starting values for the constant instrument specific parameters that will not vary."""
        # Initialize a dict to hold the constant parameters and their values
        constant_params = {}
        constant_params["grating_constant"] = self.grating_constant
        constant_params["doppler_velocity"] = self.doppler_velocity
        constant_params["order"] = self.order
        return constant_params
