"""Machinery to collect data and parameter objects and fit the wavelength axis."""
import logging
from functools import cached_property

import astropy.units as u
import lmfit
import numpy as np
from numpydantic import NDArray
from pydantic import BaseModel
from pydantic import ConfigDict
from scipy.ndimage import gaussian_filter1d

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
    unfit_1d_spectrum: NDArray

    def __init__(self, input_parameters, atlas, expected_wavelength_vector, unfit_1d_spectrum):
        """Call Pydantic validation."""
        super().__init__(
            input_parameters=input_parameters,
            atlas=atlas,
            expected_wavelength_vector=expected_wavelength_vector,
            unfit_1d_spectrum=unfit_1d_spectrum,
        )

    @cached_property
    def fit_result(self) -> dict:
        """Run data preparation and fitting."""
        fts_wave, fts_solar, fts_telluric = self.prepare_atlas()
        fit_result_params = self.run_fit(fts_wave, fts_solar, fts_telluric)
        return fit_result_params

    def prepare_atlas(self):
        """Prepare the atlas by roughly matching the atlas to the raw spectrum. This is done in 2 steps.

        1. Resample both atlases to be on the same, linear wavelength grid and select the portion of atlas that
        pertains to bandpass used in the data.

        2. Determine a shift of the expected wavelength vector so that it generally aligns with the data itself.

        """
        # Extract parameters
        param_vals = self.input_parameters.lmfit_parameters.valuesdict()
        crval1 = param_vals["crval1"]
        dispersion = param_vals["dispersion"]
        resolving_power = param_vals["resolving_power"]
        opacity = param_vals["opacity_factor"]
        stray_light_fraction = param_vals["straylight_fraction"]
        continuum_intensity_correction = param_vals["continuum_intensity_correction"]

        # Resample atlases
        solar_atlas_wavelength = self.atlas.solar_atlas_wavelength
        telluric_atlas_wavelength = self.atlas.telluric_atlas_wavelength

        expected_wavelength_range = (
            self.expected_wavelength_vector.max() - self.expected_wavelength_vector.min()
        )
        min_wavelength = self.expected_wavelength_vector.min() - 0.25 * expected_wavelength_range
        max_wavelength = self.expected_wavelength_vector.max() + 0.25 * expected_wavelength_range

        cropped_telluric_mask = (telluric_atlas_wavelength > min_wavelength) * (
            telluric_atlas_wavelength < max_wavelength
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

        # Gaussian convolution of the FTS atlas
        fwhm_wavelength = crval1 / resolving_power  # both in parameters
        sigma_wavelength = fwhm_wavelength / (2.0 * np.sqrt(2.0 * np.log(2)))
        kern_pix = sigma_wavelength / np.abs(dispersion)

        # Determine shift
        shifts = np.linspace(-2, 2, 550) * u.nm
        merit = np.zeros(len(shifts))
        for n, shift in enumerate(shifts):
            preliminary_wavelength = self.expected_wavelength_vector + shift
            fts_solar = np.interp(
                preliminary_wavelength,
                cropped_solar_atlas_wavelength,
                cropped_solar_atlas_transmission,
            )
            fts_telluric = np.interp(
                preliminary_wavelength,
                cropped_telluric_atlas_wavelength,
                cropped_telluric_atlas_transmission,
            )
            # scale by opacity value
            fts_telluric_atmos_corr = fts_telluric**opacity

            # apply telluric absorption spectrum to solar spectrum
            fts_modulated = fts_telluric_atmos_corr * fts_solar
            # add flat value of straylight contamination
            fts_modulated_with_straylight = (fts_modulated + stray_light_fraction) / (
                1.0 + stray_light_fraction
            )
            # scale for total intensity of the continuum
            fit_amplitude = fts_modulated_with_straylight * continuum_intensity_correction
            # convolution for spectrograph line spread function
            fit_amplitude = gaussian_filter1d(fit_amplitude, kern_pix)
            # calculate a merit value to be minimized
            merit[n] = np.std(self.unfit_1d_spectrum - fit_amplitude)
            # merit[n] = np.std(self.unfit_1d_spectrum - fts_solar * fts_telluric)

        # get minimum
        shift = shifts[np.argmin(merit)]

        # recalculate spectral axis and atlas spectrum for the best shift value
        fts_wave = self.expected_wavelength_vector + shift
        fts_solar = np.interp(
            fts_wave, cropped_solar_atlas_wavelength, cropped_solar_atlas_transmission
        )
        fts_telluric = np.interp(
            fts_wave, cropped_telluric_atlas_wavelength, cropped_telluric_atlas_transmission
        )

        return fts_wave.value, fts_solar, fts_telluric

    def run_fit(self, fts_wave, fts_solar, fts_telluric) -> dict:
        """Send the fitter on its way."""
        params_to_fit = self.input_parameters.lmfit_parameters
        constant_parameters = self.input_parameters.constant_parameters

        logger.info("Beginning fit.")
        minimizer = lmfit.Minimizer(
            fitness,
            params_to_fit,
            fcn_args=(
                self.unfit_1d_spectrum,
                fts_wave,
                fts_telluric,
                fts_solar,
            ),
            fcn_kws=constant_parameters,
        )
        fit_result = minimizer.minimize(method="leastsq", params=params_to_fit)
        logger.info("Finished fit.")
        fit_result_params = fit_result.params.valuesdict()

        # assign units to fitted values
        fit_result_params["crval1"] = fit_result_params["crval1"] * u.nm
        fit_result_params["dispersion"] = fit_result_params["dispersion"] * u.nm
        fit_result_params["incident_light_angle"] = (
            fit_result_params["incident_light_angle"] * u.rad
        )

        logger.info("Fitted Values:")
        for (param, fitted_value) in fit_result_params.items():
            logger.info(f"{param} = {fitted_value}")

        return fit_result_params
