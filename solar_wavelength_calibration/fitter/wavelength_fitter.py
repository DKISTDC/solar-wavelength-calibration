"""Machinery to collect data and parameter objects and fit the wavelength axis."""
import logging
from functools import cached_property

import astropy.units as u
import lmfit
import numpy as np
from astropy.units import Quantity
from astropy.wcs import WCS
from pydantic import BaseModel
from pydantic import ConfigDict

from solar_wavelength_calibration.atlas.atlas import CroppedAtlas
from solar_wavelength_calibration.atlas.base import AtlasBase
from solar_wavelength_calibration.fitter.fitting_core import fitness
from solar_wavelength_calibration.fitter.parameters import WavelengthCalibrationParameters

logger = logging.getLogger(__name__)


class WaveCalFitter(BaseModel):
    """Object that brings together data, models (atlases), and fit parameters to run fits."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_parameters: WavelengthCalibrationParameters
    atlas: AtlasBase

    def run_fit(
        self,
        expected_wavelength_vector: u.Quantity,
        unfit_1d_spectrum: np.ndarray,
        full: bool = False,
        method: str = "leastsq",
        **minimize_kwargs,
    ) -> dict:
        """Run data preparation and fitting."""
        cropped_atlas = self.prepare_atlas(expected_wavelength_vector)
        fit_result_params = self.fit_spectrum(
            cropped_atlas, unfit_1d_spectrum, full, method, **minimize_kwargs
        )
        return fit_result_params

    def prepare_atlas(
        self, expected_wavelength_vector: u.Quantity
    ) -> tuple[Quantity, Quantity, np.ndarray, np.ndarray]:
        """
        Crop and align the solar and telluric atlases to match the expected wavelength range.

        This is done by cropping the solar and telluric atlas data to a range
        slightly larger than the expected wavelength vector. The resulting cropped data
        is returned for use in fitting procedures.
        """
        solar_atlas_wavelength = self.atlas.solar_atlas_wavelength
        telluric_atlas_wavelength = self.atlas.telluric_atlas_wavelength

        expected_wavelength_range = (
            expected_wavelength_vector.max() - expected_wavelength_vector.min()
        )
        min_wavelength = expected_wavelength_vector.min() - 0.25 * expected_wavelength_range
        max_wavelength = expected_wavelength_vector.max() + 0.25 * expected_wavelength_range

        cropped_telluric_mask = (telluric_atlas_wavelength > min_wavelength) & (
            telluric_atlas_wavelength < max_wavelength
        )

        cropped_telluric_atlas_wavelength = telluric_atlas_wavelength[cropped_telluric_mask]
        cropped_telluric_atlas_transmission = self.atlas.telluric_atlas_transmission[
            cropped_telluric_mask
        ]

        cropped_solar_mask = (solar_atlas_wavelength > min_wavelength) & (
            solar_atlas_wavelength < max_wavelength
        )
        cropped_solar_atlas_wavelength = solar_atlas_wavelength[cropped_solar_mask]
        cropped_solar_atlas_transmission = self.atlas.solar_atlas_transmission[cropped_solar_mask]

        cropped_atlas = CroppedAtlas(
            cropped_solar_atlas_wavelength=cropped_solar_atlas_wavelength.to_value(u.nm),
            cropped_telluric_atlas_wavelength=cropped_telluric_atlas_wavelength.to_value(u.nm),
            cropped_solar_atlas_transmission=cropped_solar_atlas_transmission,
            cropped_telluric_atlas_transmission=cropped_telluric_atlas_transmission,
        )

        return cropped_atlas

    def fit_spectrum(
        self,
        cropped_atlas: CroppedAtlas,
        unfit_1d_spectrum: np.ndarray,
        full: bool,
        method: str,
        **minimize_kwargs,
    ) -> dict:
        """Send the fitter on its way."""
        params_to_fit = self.input_parameters.lmfit_parameters
        constant_parameters = self.input_parameters.constant_parameters
        number_of_wave_pix = np.size(unfit_1d_spectrum)

        logger.info("Beginning fit.")
        minimizer = lmfit.Minimizer(
            fitness,
            params_to_fit,
            fcn_args=(
                unfit_1d_spectrum,
                cropped_atlas,
                number_of_wave_pix,
            ),
            fcn_kws=constant_parameters,
        )
        fit_result = minimizer.minimize(params=params_to_fit, method=method, **minimize_kwargs)
        logger.info("Finished fit.")
        fit_result_params = fit_result.params.valuesdict()

        logger.info("Fitted Values:")
        for (param, fitted_value) in fit_result_params.items():
            logger.info(f"{param} = {fitted_value}")

        fit_result_header = {
            "CTYPE1": "AWAV-GRA",  # Grating dispersion function with air wavelengths
            "CUNIT1": "nm",  # Dispersion units
            "CRPIX1": number_of_wave_pix // 2 + 1,  # Reference pixel [pix]
            "PV1_0": constant_parameters["grating_constant"],  # Grating density
            "PV1_1": constant_parameters["order"],  # Diffraction order
            "CRVAL1": fit_result_params["crval1"],  # Reference value [nm]
            "CDELT1": fit_result_params["dispersion"],  # Linear dispersion [nm/pix]
            "PV1_2": fit_result_params["incident_light_angle"],  # Incident angle [deg]
        }

        if full:
            return fit_result_header, fit_result_params

        return fit_result_header
