"""Machinery to collect data and parameter objects and fit the wavelength axis."""
import logging
from keyword import kwlist

import astropy.units as u
import lmfit
import numpy as np
from lmfit import Parameters

from solar_wavelength_calibration.atlas.atlas_prep import PrepareAtlas
from solar_wavelength_calibration.atlas.base import AtlasBase
from solar_wavelength_calibration.fitter.fitting_core import fitness
from solar_wavelength_calibration.fitter.parameters import WavelengthCalibrationParameters

logger = logging.getLogger(__name__)


class WaveCalFitter:
    """Object that brings together data, models (atlases), and fit paramters to run fits."""

    def __init__(
        self,
        input_parameters: WavelengthCalibrationParameters,
        target_atlas: PrepareAtlas,
        unfit_1d_spectrum: np.ndarray,
    ):

        self.lmfit_parameters = input_parameters.lmfit_parameters
        self.constant_parameters = input_parameters.constant_parameters
        self.target_solar_atlas_wavelength = target_atlas.solar_atlas_wavelength
        self.target_telluric_atlas_wavelength = target_atlas.telluric_atlas_wavelength
        self.target_solar_atlas = target_atlas.solar_atlas
        self.target_telluric_atlas = target_atlas.telluric_atlas
        self.unfit_1d_spectrum = unfit_1d_spectrum

        self.run_fit()

    def run_fit(self) -> dict:
        """Docstring."""
        params_to_fit = self.lmfit_parameters
        constant_parameters = self.constant_parameters

        logger.info("Beginning fit.")
        minimizer = lmfit.Minimizer(
            fitness,
            params_to_fit,
            fcn_args=(
                self.unfit_1d_spectrum,
                self.target_solar_atlas_wavelength,
                self.target_telluric_atlas_wavelength,
                self.target_telluric_atlas,
                self.target_solar_atlas,
            ),
            fcn_kws=constant_parameters,
        )
        fit_result = minimizer.minimize(method="leastsq", params=params_to_fit)
        logger.info("Finished fit.")
        fit_result_params = fit_result.params.valuesdict()  # should be a dictionary

        logger.info("Fitted Values:")
        for (param, fitted_value) in fit_result_params.items():
            logger.info(f"Parameter: {param},       Fit Result: {fitted_value}")

        return fit_result_params
