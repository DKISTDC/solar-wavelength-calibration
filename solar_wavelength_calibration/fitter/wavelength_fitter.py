"""Machinery to collect data and parameter objects and fit the wavelength axis."""
import logging

import astropy.units as u
import lmfit
import numpy as np
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

from solar_wavelength_calibration.atlas.atlas import Atlas
from solar_wavelength_calibration.atlas.atlas import AtlasSlice
from solar_wavelength_calibration.atlas.base import AtlasBase
from solar_wavelength_calibration.fitter.fitting_core import fitting_model
from solar_wavelength_calibration.fitter.parameters import WavelengthCalibrationParameters

logger = logging.getLogger(__name__)


class FitResultHeader(BaseModel):
    """Represents the Fitted WCS header information."""

    CTYPE1: str
    CUNIT1: str
    CRPIX1: int
    PV1_0: float
    PV1_1: int
    CRVAL1: float
    CDELT1: float
    PV1_2: float


class FitResult(BaseModel):
    """Represents the output of the fitting process, including the WCS header information and optional fit parameters."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    header: FitResultHeader
    parameters: lmfit.parameter.Parameters

    def __getattr__(self, item):
        """Return the fit result as a tuple of dictionaries."""
        return self.header, self.parameters


class WavelengthCalibrationFitter(BaseModel):
    """Object that brings together data, models (atlases), and fit parameters to run fits."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_parameters: WavelengthCalibrationParameters
    atlas: AtlasBase = Field(default_factory=Atlas)

    @model_validator(mode="after")
    def _validate_atlas_wavelength_monotonicity(self):
        """Ensure that the wavelength arrays in the atlas are monotonic."""
        if not self.atlas.telluric_atlas_wavelength_is_monotonic:
            raise ValueError("Telluric atlas wavelength array is not monotonic.")
        if not self.atlas.solar_atlas_wavelength_is_monotonic:
            raise ValueError("Solar atlas wavelength array is not monotonic.")
        return self

    def __call__(
        self,
        expected_wavelength_vector: u.Quantity,
        unfit_1d_spectrum: np.ndarray,
        method: str = "leastsq",
        **minimize_kwargs,
    ) -> FitResult:
        """Make the instance callable and run the fit."""
        return self.run_fit(
            expected_wavelength_vector,
            unfit_1d_spectrum,
            method,
            **minimize_kwargs,
        )

    def run_fit(
        self,
        expected_wavelength_vector: u.Quantity,
        unfit_1d_spectrum: np.ndarray,
        method: str = "leastsq",
        **minimize_kwargs,
    ) -> FitResult:
        """Run data preparation and fitting.

        Parameters
        ----------
        expected_wavelength_vector (u.Quantity)
            The expected wavelength vector for the fit.

        unfit_1d_spectrum (np.ndarray)
            The unfit 1D spectrum to be fitted.

        method (str, optional)
            The optimization method to use. Defaults to "leastsq".

        **minimize_kwargs
            Additional keyword arguments for the minimizer.

        Returns
        -------
        fit_result:
            An instance of `FitResult` containing:
            - `header` (dict[str, Quantity]): The WCS header information.
            - `parameters` (dict[str, float] | None): The fit parameters.
        """
        cropped_atlas = self._prepare_atlas(expected_wavelength_vector)
        fit_result = self._fit_spectrum(cropped_atlas, unfit_1d_spectrum, method, **minimize_kwargs)
        return fit_result

    def _prepare_atlas(self, expected_wavelength_vector: u.Quantity) -> AtlasSlice:
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

        cropped_telluric_atlas_wavelength = telluric_atlas_wavelength[
            cropped_telluric_mask
        ].to_value(u.nm)
        cropped_telluric_atlas_transmission = self.atlas.telluric_atlas_transmission[
            cropped_telluric_mask
        ]

        cropped_solar_mask = (solar_atlas_wavelength > min_wavelength) & (
            solar_atlas_wavelength < max_wavelength
        )
        cropped_solar_atlas_wavelength = solar_atlas_wavelength[cropped_solar_mask].to_value(u.nm)
        cropped_solar_atlas_transmission = self.atlas.solar_atlas_transmission[cropped_solar_mask]

        cropped_atlas = AtlasSlice(
            solar_atlas_wavelength=cropped_solar_atlas_wavelength,
            telluric_atlas_wavelength=cropped_telluric_atlas_wavelength,
            solar_atlas_transmission=cropped_solar_atlas_transmission,
            telluric_atlas_transmission=cropped_telluric_atlas_transmission,
        )

        return cropped_atlas

    def _fit_spectrum(
        self,
        cropped_atlas: AtlasSlice,
        unfit_1d_spectrum: np.ndarray,
        method: str,
        **minimize_kwargs,
    ) -> FitResult:
        """Send the fitter on its way."""
        params_to_fit = self.input_parameters.lmfit_parameters
        constant_parameters = self.input_parameters.constant_parameters
        number_of_wave_pix = np.size(unfit_1d_spectrum)

        logger.info("Beginning fit.")
        minimizer = lmfit.Minimizer(
            fitting_model,
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

        logger.info(f"{fit_result.params=}")

        fit_result_header = FitResultHeader(
            CTYPE1="AWAV-GRA",  # Grating dispersion function with air wavelengths
            CUNIT1="nm",  # Dispersion units
            CRPIX1=number_of_wave_pix // 2 + 1,  # Reference pixel [pix]
            PV1_0=constant_parameters["grating_constant"],  # Grating density
            PV1_1=constant_parameters["order"],  # Diffraction order
            CRVAL1=fit_result.params["crval1"].value,  # Reference value [nm]
            CDELT1=fit_result.params["dispersion"].value,  # Linear dispersion [nm/pix]
            PV1_2=fit_result.params["incident_light_angle"].value,  # Incident angle [deg]
        )

        return FitResult(header=fit_result_header, parameters=fit_result.params)
