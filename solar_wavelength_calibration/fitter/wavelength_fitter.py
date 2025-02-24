"""Machinery to collect data and parameter objects and fit the wavelength axis."""
import logging
from typing import Any

import astropy.units as u
import lmfit
import numpy as np
from numpydantic import NDArray
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import PrivateAttr

from solar_wavelength_calibration.atlas.base import AtlasBase
from solar_wavelength_calibration.fitter.fitting_core import fitness
from solar_wavelength_calibration.fitter.parameters import WavelengthCalibrationParameters

logger = logging.getLogger(__name__)


class WaveCalFitter(BaseModel):
    """Object that brings together data, models (atlases), and fit parameters to run fits."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_parameters: WavelengthCalibrationParameters
    atlas: AtlasBase
    expected_wavelength_vector: NDArray
    unfit_1d_spectrum: NDArray

    # Declare private attributes (not part of Pydantic validation)
    _lmfit_parameters: Any = PrivateAttr()
    _constant_parameters: Any = PrivateAttr()
    _telluric_atlas_wavelength: Any = PrivateAttr()
    _telluric_atlas_transmission: Any = PrivateAttr()
    _solar_atlas_wavelength: Any = PrivateAttr()
    _solar_atlas_transmission: Any = PrivateAttr()
    _fit_result_params: Any = PrivateAttr()

    def __init__(self, input_parameters, atlas, expected_wavelength_vector, unfit_1d_spectrum):
        """Call Pydantic validation first, then extra logic in `model_post_init`."""
        super().__init__(
            input_parameters=input_parameters,
            atlas=atlas,
            expected_wavelength_vector=expected_wavelength_vector,
            unfit_1d_spectrum=unfit_1d_spectrum,
        )

    def model_post_init(self, __context: Any) -> None:
        """Post-validation initialization logic."""
        # Assign attributes to private attributes
        self._lmfit_parameters = self.input_parameters.lmfit_parameters
        self._constant_parameters = self.input_parameters.constant_parameters
        self._telluric_atlas_wavelength = self.atlas.telluric_atlas_wavelength
        self._telluric_atlas_transmission = self.atlas.telluric_atlas_transmission
        self._solar_atlas_wavelength = self.atlas.solar_atlas_wavelength
        self._solar_atlas_transmission = self.atlas.solar_atlas_transmission

        # Run data preparation and fitting
        fts_wave, fts_solar, fts_telluric = self.prepare_atlas()
        self._fit_result_params = self.run_fit(fts_wave, fts_solar, fts_telluric)

    def prepare_atlas(self):
        """Prepare the atlas by roughly matching the atlas to the raw spectrum. This is done in 2 steps.

        1. Resample both atlases to be on the same, linear wavelength grid and select the portion of atlas that
        pertains to bandpass used in the data.

        2. Determine a shift of the preliminary wavelength vector so that it generally aligns with the data itself.

        """
        # Resample atlases
        solar_atlas_wavelength = self._solar_atlas_wavelength * u.nm
        telluric_atlas_wavelength = self._telluric_atlas_wavelength * u.nm

        expected_wavelength_range = (
            self.expected_wavelength_vector.max() - self.expected_wavelength_vector.min()
        )
        min_wavelength = self.expected_wavelength_vector.min() - 0.25 * expected_wavelength_range
        max_wavelength = self.expected_wavelength_vector.max() + 0.25 * expected_wavelength_range

        cropped_telluric_mask = (telluric_atlas_wavelength > min_wavelength) * (
            telluric_atlas_wavelength < max_wavelength
        )
        cropped_telluric_atlas_wavelength = telluric_atlas_wavelength[cropped_telluric_mask]
        cropped_telluric_atlas_transmission = self._telluric_atlas_transmission[
            cropped_telluric_mask
        ]

        cropped_solar_mask = (solar_atlas_wavelength > min_wavelength) * (
            solar_atlas_wavelength < max_wavelength
        )
        cropped_solar_atlas_wavelength = solar_atlas_wavelength[cropped_solar_mask]
        cropped_solar_atlas_transmission = self._solar_atlas_transmission[cropped_solar_mask]

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
            # calculate a merit value to be minimized
            merit[n] = np.std(self.unfit_1d_spectrum - fts_solar * fts_telluric)

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
        params_to_fit = self._lmfit_parameters
        constant_parameters = self._constant_parameters

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
        _fit_result_params = fit_result.params.valuesdict()

        logger.info("Fitted Values:")
        for (param, fitted_value) in _fit_result_params.items():
            logger.info(f"Parameter: {param},       Fit Result: {fitted_value}")

        return _fit_result_params
