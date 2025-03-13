"""Objects for storing fit parameters."""
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
    fit_flags: Optional[dict[str, bool]] = None
    bounds: Optional[dict[str, list[Union[Quantity, float, int]]]] = None  # Mixed types

    @field_validator("crval1", "dispersion", "incident_light_angle", mode="before")
    @classmethod
    def validate_length_units(cls, value, info):
        """Ensure crval1, dispersion, and incident_light_angle have correct length units."""
        expected_units = {
            "crval1": u.nm,
            "dispersion": u.nm / u.pix,
            "incident_light_angle": u.deg,
        }

        field_name = info.field_name  # Correct way to get the field name

        if not isinstance(value, Quantity):
            raise TypeError(f"{field_name} must be an astropy Quantity.")

        expected_unit = expected_units[field_name]
        if not value.unit.is_equivalent(expected_unit):
            raise ValueError(f"{field_name} must be in {expected_unit}, but got {value.unit}.")

        return value

    @field_validator("unfit_1d_spectrum", mode="after")
    @classmethod
    def validate_unfit_1d_spectrum(cls, value):
        """Ensure unfit_1d_spectrum is an NDArray."""
        if not isinstance(value, np.ndarray):
            raise TypeError("unfit_1d_spectrum must be a NumPy array.")
        return value

    @field_validator("grating_constant", mode="before")
    @classmethod
    def validate_grating_constant(cls, value, info):
        """Ensure grating_constant has correct units (1/mm)."""
        if not isinstance(value, Quantity):
            raise TypeError(f"{info.field_name} must be an astropy Quantity.")

        if not value.unit.is_equivalent(1 / u.mm):
            raise ValueError(f"{info.field_name} must have units of 1/mm, but got {value.unit}.")

        return value

    @field_validator("doppler_velocity", mode="before")
    @classmethod
    def validate_doppler_velocity(cls, value, info):
        """Ensure doppler_velocity has correct units (km/s)."""
        if not isinstance(value, Quantity):
            raise TypeError(f"{info.field_name} must be an astropy Quantity.")

        if not value.unit.is_equivalent(u.km / u.s):
            raise ValueError(f"{info.field_name} must have units of km/s, but got {value.unit}.")

        return value

    @staticmethod
    def _default_fit_flags():
        """Return default fit flags. True indicates that the parameter should be fit."""
        return {
            "crval1": True,
            "dispersion": True,
            "incident_light_angle": True,
            "resolving_power": True,
            "opacity_factor": True,
            "straylight_fraction": True,
            "continuum_intensity_correction": True,
        }

    @staticmethod
    def _default_bounds():
        """Return default bounds for the fit parameters."""
        return {
            "crval1": [-0.5, 0.5] * u.nm,
            "dispersion": [-0.05, 0.05] * u.nm / u.pix,
            "incident_light_angle": [-180, 180] * u.deg,
            "resolving_power": [-50000, 50000],
            "opacity_factor": [-5, 5],
            "straylight_fraction": [-0.25, 0.25],
            "continuum_intensity_correction": [0.8, 1.2],
        }

    @field_validator("bounds", mode="before")
    @classmethod
    def validate_bounds(cls, value):
        """Ensure crval1, dispersion, and incident_light_angle have correct units, while others are dimensionless."""
        if value is None:
            return cls._default_bounds()

        default_bounds = cls._default_bounds()

        for key, bound in value.items():
            if key in ["crval1", "dispersion", "incident_light_angle"]:
                # These must be a Quantity and match expected units
                if not isinstance(bound[0], Quantity) or not isinstance(bound[1], Quantity):
                    raise ValueError(
                        f"Bounds for {key} must be an astropy Quantity with appropriate units."
                    )

                expected_unit = default_bounds[key][0].unit
                if not bound[0].unit.is_equivalent(expected_unit) or not bound[
                    1
                ].unit.is_equivalent(expected_unit):
                    raise ValueError(
                        f"Bounds for {key} must be in {expected_unit}, got {bound[0].unit} and {bound[1].unit}."
                    )

            else:
                # All other parameters must be dimensionless
                if isinstance(bound[0], Quantity) or isinstance(bound[1], Quantity):
                    raise ValueError(
                        f"Bounds for {key} must be dimensionless (int or float), but got {bound}."
                    )

        return value

    def __init__(self, **kwargs):
        """Merge user-provided fit_flags and bounds with defaults."""
        kwargs["fit_flags"] = {**self._default_fit_flags(), **(kwargs.get("fit_flags") or {})}
        kwargs["bounds"] = {**self._default_bounds(), **(kwargs.get("bounds") or {})}

        super().__init__(**kwargs)  # Initialize Pydantic model

    def _get_bounds(self, param_name: str):
        """Retrieve bounds for a parameter, ensuring defaults exist."""
        return self.bounds.get(param_name, self._default_bounds()[param_name])

    @property
    def lmfit_parameters(self) -> Parameters:
        """Create a single lmfit `Parameters` object with custom bounds and free/fixed status."""
        param = Parameters()

        param.add(
            "crval1",
            vary=self.fit_flags["crval1"],
            value=self.crval1.to_value(u.nm),
            min=(self.crval1 + self._get_bounds("crval1")[0]).to_value(u.nm),
            max=(self.crval1 + self._get_bounds("crval1")[1]).to_value(u.nm),
        )
        param.add(
            "dispersion",
            vary=self.fit_flags["dispersion"],
            value=self.dispersion.to_value(u.nm / u.pix),
            min=self.dispersion.to_value(u.nm / u.pix)
            + self._get_bounds("dispersion")[0].to_value(u.nm / u.pix)
            * self.dispersion.to_value(u.nm / u.pix),
            max=self.dispersion.to_value(u.nm / u.pix)
            + self._get_bounds("dispersion")[1].to_value(u.nm / u.pix)
            * self.dispersion.to_value(u.nm / u.pix),
        )
        param.add(
            "incident_light_angle",
            vary=self.fit_flags["incident_light_angle"],
            value=self.incident_light_angle.to_value(u.deg),
            min=(self.incident_light_angle + self._get_bounds("incident_light_angle")[0]).to_value(
                u.deg
            ),
            max=(self.incident_light_angle + self._get_bounds("incident_light_angle")[1]).to_value(
                u.deg
            ),
        )
        param.add(
            "resolving_power",
            vary=self.fit_flags["resolving_power"],
            value=self.resolving_power,
            min=self.resolving_power + self._get_bounds("resolving_power")[0],
            max=self.resolving_power + self._get_bounds("resolving_power")[1],
        )
        param.add(
            "opacity_factor",
            vary=self.fit_flags["opacity_factor"],
            value=self.opacity_factor,
            min=self.opacity_factor + self._get_bounds("opacity_factor")[0],
            max=self.opacity_factor + self._get_bounds("opacity_factor")[1],
        )
        param.add(
            "straylight_fraction",
            vary=self.fit_flags["straylight_fraction"],
            value=self.straylight_fraction,
            min=self.straylight_fraction + self._get_bounds("straylight_fraction")[0],
            max=self.straylight_fraction + self._get_bounds("straylight_fraction")[1],
        )
        spectrum_percentile = np.nanpercentile(self.unfit_1d_spectrum, 75)
        param.add(
            "continuum_intensity_correction",
            vary=self.fit_flags["continuum_intensity_correction"],
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
        constant_params["grating_constant"] = self.grating_constant.value
        constant_params["doppler_velocity"] = self.doppler_velocity.value
        constant_params["order"] = self.order
        return constant_params
