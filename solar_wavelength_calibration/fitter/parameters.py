"""Objects for storing fit parameters."""
import numpy as np
from lmfit import Parameters
from numpydantic import NDArray
from pydantic import BaseModel


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
    straylight_fraction: float = 0
    fit_dispersion: bool = True

    @property
    def lmfit_parameters(self) -> Parameters:
        """Create a single lmfit `Parameters` object with correct starting values for the wavelength calibration parameters."""
        param = Parameters()
        param.add(
            "cravl1", vary=True, value=self.crval1, min=self.crval1 - 0.5, max=self.crval1 + 0.5
        )
        param.add(
            "dispersion",
            vary=self.fit_dispersion,
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
    def constant_parameters(self) -> dict:
        """Create a single dictionary with correct starting values for the constant instrument specific parameters that will not vary."""
        # Initialize a dict to hold the constant parameters and their values
        constant_params = {}
        constant_params["grating_constant"] = self.grating_constant
        constant_params["doppler_shift"] = self.doppler_shift
        constant_params["order"] = self.order
        return constant_params
