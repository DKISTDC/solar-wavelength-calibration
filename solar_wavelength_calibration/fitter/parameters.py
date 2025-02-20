"""Objects for storing fit parameters."""
from dataclasses import dataclass

import numpy as np
from lmfit import Parameters
from numpydantic import NDArray
from pydantic import BaseModel


@dataclass
class WavelengthCalibrationParameters(BaseModel):
    """Object for initializing the fit parameters prior to a fit."""

    crval1: float
    dispersion: float
    incident_light_angle: float
    unfit_1d_spectrum: NDArray
    grating_constant: float
    doppler_shift: float
    order: int
    resolving_power: int = 1
    opacity_factor: float = 1
    straylight_fraction: float = 1

    @property
    def lmfit_parameters(self) -> Parameters:
        """Create a single lmfit `Parameters` object with correct starting values for the wavelength calibration parameters."""
        param = Parameters()
        param.add(
            "cravl1", vary=True, value=self.crval1, min=self.crval1 - 0.5, max=self.crval1 + 0.5
        )
        param.add(
            "dispersion",
            vary=True,
            value=self.dispersion,
            min=self.dispersion - 0.05 * self.dispersion,
            max=self.dispersion + 0.05 * self.dispersion,
        )
        param.add(
            "incident_light_angle",
            vary=True,
            value=self.incident_light_angle,
            min=np.rad2deg(self.incident_light_angle) - 5,
            max=np.rad2deg(self.incident_light_angle) + 5,
        )
        # TODO: add default values for resolving power, opacity factor, and straylight fraction that 'do nothing' (i.e. multiply by 1 or add 0 or something)
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
        spectrum_percentile = np.nanpercentile(self.unfit_1d_spectrum, 75)
        param.add(
            "continuum_intensity_correction",
            vary=True,
            value=spectrum_percentile,
            min=0.8 * spectrum_percentile,
            max=1.2 * spectrum_percentile,
        )
        return param

    @property
    def constant_parameters(self) -> Parameters:
        """Create a single lmfit `Parameters` object with correct starting values for the constant instrument specific parameters that will not vary."""
        param = Parameters()
        param.add("grating_constant", vary=False, value=self.grating_constant)
        param.add("doppler_shift", vary=False, value=self.doppler_shift)
        param.add("order", vary=False, value=self.order)
        return param
