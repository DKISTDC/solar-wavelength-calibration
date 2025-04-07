"""Machinery to collect data and parameter objects and fit the wavelength axis."""
import logging
from functools import cached_property

import astropy.units as u
import lmfit
import numpy as np
from astropy.wcs import WCS
from pydantic import BaseModel
from pydantic import ConfigDict

from solar_wavelength_calibration.atlas.base import AtlasBase
from solar_wavelength_calibration.fitter.fitting_core import fitness
from solar_wavelength_calibration.fitter.parameters import WavelengthCalibrationParameters

logger = logging.getLogger(__name__)


class WaveCalFitter(BaseModel):
    """Object that brings together data, models (atlases), and fit parameters to run fits."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_parameters: WavelengthCalibrationParameters
    atlas: AtlasBase
    expected_wavelength_vector: u.Quantity
    unfit_1d_spectrum: np.ndarray
    full: bool = False

    @cached_property
    def fit_result(self) -> dict:
        """Run data preparation and fitting."""
        (
            cropped_solar_atlas_wavelength,
            cropped_telluric_atlas_wavelength,
            cropped_solar_atlas_transmission,
            cropped_telluric_atlas_transmission,
        ) = self.prepare_atlas()
        fit_result_params = self.run_fit(
            cropped_solar_atlas_wavelength,
            cropped_telluric_atlas_wavelength,
            cropped_solar_atlas_transmission,
            cropped_telluric_atlas_transmission,
        )
        return fit_result_params

    def prepare_atlas(self):
        """Prepare the atlas by roughly matching the atlas to the raw spectrum.

        This is done by resampling both atlases to be on the same, linear wavelength grid and select the portion of atlas that
        pertains to bandpass used in the data.
        """
        # Resample atlases
        solar_atlas_wavelength = self.atlas.solar_atlas_wavelength
        telluric_atlas_wavelength = self.atlas.telluric_atlas_wavelength

        expected_wavelength_range = (
            self.expected_wavelength_vector.max() - self.expected_wavelength_vector.min()
        )
        min_wavelength = self.expected_wavelength_vector.min() - 0.25 * expected_wavelength_range
        max_wavelength = self.expected_wavelength_vector.max() + 0.25 * expected_wavelength_range

        cropped_telluric_mask = np.where(
            (telluric_atlas_wavelength > min_wavelength)
            & (telluric_atlas_wavelength < max_wavelength),
            True,
            False,
        )

        cropped_telluric_atlas_wavelength = telluric_atlas_wavelength[cropped_telluric_mask]
        cropped_telluric_atlas_transmission = self.atlas.telluric_atlas_transmission[
            cropped_telluric_mask
        ]

        cropped_solar_mask = (solar_atlas_wavelength > min_wavelength) * (
            solar_atlas_wavelength < max_wavelength
        )
        cropped_solar_atlas_wavelength = solar_atlas_wavelength[cropped_solar_mask]
        cropped_solar_atlas_transmission = self.atlas.solar_atlas_transmission[cropped_solar_mask]

        return (
            cropped_solar_atlas_wavelength.to_value(u.nm),
            cropped_telluric_atlas_wavelength.to_value(u.nm),
            cropped_solar_atlas_transmission,
            cropped_telluric_atlas_transmission,
        )

    def run_fit(self, fts_solar_wave, fts_telluric_wave, fts_solar, fts_telluric) -> dict:
        """Send the fitter on its way."""
        params_to_fit = self.input_parameters.lmfit_parameters
        constant_parameters = self.input_parameters.constant_parameters
        number_of_wave_pix = np.size(self.unfit_1d_spectrum)

        logger.info("Beginning fit.")
        minimizer = lmfit.Minimizer(
            fitness,
            params_to_fit,
            fcn_args=(
                self.unfit_1d_spectrum,
                fts_solar_wave,
                fts_telluric_wave,
                fts_telluric,
                fts_solar,
                number_of_wave_pix,
            ),
            fcn_kws=constant_parameters,
            # nan_policy="propagate",
        )
        fit_result = minimizer.minimize(method="leastsq", params=params_to_fit)
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

        if self.full:
            return WCS(fit_result_header), fit_result_params

        return WCS(fit_result_header)
