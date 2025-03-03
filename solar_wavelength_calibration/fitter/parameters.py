"""Objects for storing fit parameters."""
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
    fit_dispersion: bool = True

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

    @property
    def lmfit_parameters(self) -> Parameters:
        """Create a single lmfit `Parameters` object with correct starting values for the wavelength calibration parameters."""
        param = Parameters()
        param.add(
            "crval1",
            vary=True,
            value=self.crval1.value,
            min=self.crval1.value - 0.5,
            max=self.crval1.value + 0.5,
        )
        param.add(
            "dispersion",
            vary=self.fit_dispersion,
            value=self.dispersion.value,
            min=self.dispersion.value - 0.05 * self.dispersion.value,
            max=self.dispersion.value + 0.05 * self.dispersion.value,
        )
        param.add(
            "incident_light_angle",
            vary=True,
            value=self.incident_light_angle.value,
            min=np.rad2deg(self.incident_light_angle.value) - 5,
            max=np.rad2deg(self.incident_light_angle.value) + 5,
        )
        param.add(
            "resolving_power",
            vary=True,
            value=self.resolving_power,
            min=self.resolving_power - 50000,
            max=self.resolving_power + 50000,
        )
        param.add(
            "opacity_factor",
            vary=True,
            value=self.opacity_factor,
            min=self.opacity_factor - 5,
            max=self.opacity_factor + 5,
        )
        param.add(
            "straylight_fraction",
            vary=True,
            value=self.straylight_fraction,
            min=self.straylight_fraction - 0.25,
            max=self.straylight_fraction + 0.25,
        )
        spectrum_percentile = np.nanpercentile(self.unfit_1d_spectrum.value, 75)
        param.add(
            "continuum_intensity_correction",
            vary=True,
            value=spectrum_percentile,
            min=0.8 * spectrum_percentile,
            max=1.2 * spectrum_percentile,
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
