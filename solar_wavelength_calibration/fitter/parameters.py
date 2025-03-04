"""Objects for storing fit parameters."""
from typing import Optional

import astropy.units as u
import numpy as np
from lmfit import Parameters
from numpydantic import NDArray
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import field_validator


class WavelengthCalibrationParameters(BaseModel):
    """Object for initializing the fit parameters prior to a fit."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    crval1: u.nm
    dispersion: u.nm
    incident_light_angle: u.rad
    unfit_1d_spectrum: NDArray
    grating_constant: 1 / u.mm
    doppler_shift: u.km / u.s
    order: int
    resolving_power: int = 1
    opacity_factor: float = 1
    straylight_fraction: float = 0
    fit_flags: Optional[dict[str, bool]] = None
    bounds: Optional[dict[str, list[float]]] = None

    @field_validator("unfit_1d_spectrum", mode="after")
    @classmethod
    def validate_unfit_1d_spectrum(cls, value):
        """Ensure unfit_1d_spectrum is an NDArray with nanometers."""
        if not isinstance(value, u.Quantity):
            raise TypeError("unfit_1d_spectrum must be an astropy Quantity with units.")

        if not value.unit.is_equivalent(u.nm):
            raise ValueError(f"unfit_1d_spectrum must be in nanometers, got {value.unit} instead.")

        if not isinstance(value.value, np.ndarray):
            raise TypeError("unfit_1d_spectrum must be a NumPy array.")

        return value.to(u.nm)  # Convert to nanometers

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
            "crval1": [-0.5, 0.5],
            "dispersion": [-0.05, 0.05],
            "incident_light_angle": [-5, 5],
            "resolving_power": [-50000, 50000],
            "opacity_factor": [-5, 5],
            "straylight_fraction": [-0.25, 0.25],
            "continuum_intensity_correction": [0.8, 1.2],
        }

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
            value=self.crval1.value,
            min=self.crval1.value + self._get_bounds("crval1")[0],
            max=self.crval1.value + self._get_bounds("crval1")[1],
        )
        param.add(
            "dispersion",
            vary=self.fit_flags["dispersion"],
            value=self.dispersion.value,
            min=self.dispersion.value + self._get_bounds("dispersion")[0] * self.dispersion.value,
            max=self.dispersion.value + self._get_bounds("dispersion")[1] * self.dispersion.value,
        )
        param.add(
            "incident_light_angle",
            vary=self.fit_flags["incident_light_angle"],
            value=self.incident_light_angle.value,
            min=np.rad2deg(self.incident_light_angle.value)
            + self._get_bounds("incident_light_angle")[0],
            max=np.rad2deg(self.incident_light_angle.value)
            + self._get_bounds("incident_light_angle")[1],
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
        spectrum_percentile = np.nanpercentile(self.unfit_1d_spectrum.value, 75)
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
        constant_params["doppler_shift"] = self.doppler_shift.value
        constant_params["order"] = self.order
        return constant_params
