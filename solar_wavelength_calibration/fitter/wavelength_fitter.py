"""Machinery to collect data and parameter objects and fit the wavelength axis."""
import logging

import astropy.units as u
import lmfit.parameter
import numpy as np
from astropy import constants as const
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.units import Quantity
from astropy.wcs import WCS
from lmfit import Parameters
from lmfit.minimizer import MinimizerResult
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator
from scipy.ndimage import gaussian_filter1d

from solar_wavelength_calibration.atlas.atlas import Atlas
from solar_wavelength_calibration.atlas.atlas import LocalAtlas
from solar_wavelength_calibration.atlas.base import AtlasBase
from solar_wavelength_calibration.fitter.parameters import WavelengthCalibrationParameters


logger = logging.getLogger(__name__)

__all__ = ["WavelengthCalibrationFitter"]


class WavelengthParameters(BaseModel):
    """Represents the Fitted WCS header information."""

    crpix: int
    crval: float
    dispersion: float
    grating_constant: float
    order: int
    incident_light_angle: float
    ctype: str = "AWAV-GRA"
    cunit: str = "nm"

    def to_header(self, axis_num: int) -> dict:
        """
        Convert the wavelength parameters to a WCS header dictionary.

        Parameters
        ----------
        axis_num : int
            The axis number for the WCS header (e.g., 1 for the first axis).

        Returns
        -------
        dict
            A dictionary representing the WCS header with keys corresponding to
            FITS header keywords and values derived from the wavelength parameters.
        """
        return {
            f"CTYPE{axis_num}": self.ctype,
            f"CUNIT{axis_num}": self.cunit,
            f"CRPIX{axis_num}": self.crpix,
            f"CRVAL{axis_num}": self.crval,
            f"CDELT{axis_num}": self.dispersion,
            f"PV{axis_num}_0": self.grating_constant,
            f"PV{axis_num}_1": self.order,
            f"PV{axis_num}_2": self.incident_light_angle,
        }


class FitResult(BaseModel):
    """Represents the output of the fitting process, including the WCS header information and optional fit parameters."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    wavelength_parameters: WavelengthParameters
    minimizer_result: MinimizerResult


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
        input_wavelength_vector: u.Quantity,
        input_spectrum: np.ndarray,
        method: str = "leastsq",
        **minimize_kwargs,
    ) -> FitResult:
        """Run data preparation and fitting.

        Parameters
        ----------
        input_wavelength_vector: (u.Quantity)
            The wavelength vector corresponding to the input spectrum.

        input_spectrum: (np.ndarray)
            The input 1D spectrum to be fitted.

        method: (str, optional)
            The optimization method to use. Defaults to "leastsq".

        **minimize_kwargs
            Additional keyword arguments for the minimizer.

        Returns
        -------
        fit_result:
            An instance of `FitResult` containing:
            - `wavelength_parameters` (WavelengthParameters): The fitted parameters.
            - `minimizer_result` (MinimizerResult): The result returned by the lmfit minimizer.

        """
        return self._run_fit(
            input_wavelength_vector,
            input_spectrum,
            method,
            **minimize_kwargs,
        )

    def _run_fit(
        self,
        input_wavelength_vector: u.Quantity,
        input_spectrum: np.ndarray,
        method: str = "leastsq",
        **minimize_kwargs,
    ) -> FitResult:
        """Run the fit."""
        cropped_atlas = self._prepare_atlas(input_wavelength_vector)
        fit_result = self._fit_spectrum(cropped_atlas, input_spectrum, method, **minimize_kwargs)
        return fit_result

    def _prepare_atlas(self, input_wavelength_vector: u.Quantity) -> LocalAtlas:
        """
        Crop and align the solar and telluric atlases to match the input wavelength range.

        This is done by cropping the solar and telluric atlas data to a range
        slightly larger than the input wavelength vector. The resulting cropped data
        is returned for use in fitting procedures.
        """
        solar_atlas_wavelength = self.atlas.solar_atlas_wavelength
        telluric_atlas_wavelength = self.atlas.telluric_atlas_wavelength

        input_wavelength_range = input_wavelength_vector.max() - input_wavelength_vector.min()
        min_wavelength = input_wavelength_vector.min() - 0.25 * input_wavelength_range
        max_wavelength = input_wavelength_vector.max() + 0.25 * input_wavelength_range

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

        cropped_atlas = LocalAtlas(
            solar_atlas_wavelength=cropped_solar_atlas_wavelength,
            telluric_atlas_wavelength=cropped_telluric_atlas_wavelength,
            solar_atlas_transmission=cropped_solar_atlas_transmission,
            telluric_atlas_transmission=cropped_telluric_atlas_transmission,
        )

        return cropped_atlas

    def _fit_spectrum(
        self,
        cropped_atlas: LocalAtlas,
        input_spectrum: np.ndarray,
        method: str,
        **minimize_kwargs,
    ) -> FitResult:
        """Send the fitter on its way."""
        params_to_fit = self.input_parameters.lmfit_parameters
        constant_parameters = self.input_parameters.constant_parameters
        number_of_wave_pix = np.size(input_spectrum)

        logger.info("Beginning fit.")
        minimizer = lmfit.Minimizer(
            fitting_model,
            params_to_fit,
            fcn_args=(
                input_spectrum,
                cropped_atlas,
                number_of_wave_pix,
            ),
            fcn_kws=constant_parameters,
        )
        minimizer_result = minimizer.minimize(
            params=params_to_fit, method=method, **minimize_kwargs
        )

        logger.info("Finished fit.")
        logger.info(minimizer_result.params.pretty_repr())

        wavelength_parameters = WavelengthParameters(
            crpix=number_of_wave_pix // 2 + 1,
            crval=minimizer_result.params["crval"].value,
            dispersion=minimizer_result.params["dispersion"].value,
            grating_constant=constant_parameters["grating_constant"],
            order=constant_parameters["order"],
            incident_light_angle=minimizer_result.params["incident_light_angle"].value,
            ctype="AWAV-GRA",
            cunit="nm",
        )

        return FitResult(
            wavelength_parameters=wavelength_parameters, minimizer_result=minimizer_result
        )


def calculate_linear_wave(
    params: lmfit.parameter.Parameters, number_of_wave_pix: int, grating_constant: float, order: int
) -> np.ndarray:
    """Calculate the linear wavelength vector.

    References about representing spectral coordinate in FITS:
    https://specreduce.readthedocs.io/en/latest/api/specreduce.utils.synth_data.make_2d_arc_image.html
    Greisen et al (2006) https://ui.adsabs.harvard.edu/abs/2006A%26A...446..747G/abstract (section 5)

    Parameters
    ----------
    params:
        Parameters object containing the fitting parameters.
    number_of_wave_pix:
        Number of wavelength pixels.
    grating_constant:
        Grating constant of the spectrograph.
    order:
        Diffraction order.

    Returns
    -------
    linear_wave: np.ndarray
        Linear wavelength vector in nanometers.
    """
    non_linear_header = WavelengthParameters(
        ctype="AWAV-GRA",
        cunit="nm",
        crpix=number_of_wave_pix // 2 + 1,
        grating_constant=grating_constant,
        order=order,
        crval=params["crval"].value,
        dispersion=params["dispersion"].value,
        incident_light_angle=params["incident_light_angle"].value,
    )

    non_linear_wcs = WCS(non_linear_header.to_header(axis_num=1))
    linear_wave = (non_linear_wcs.spectral.pixel_to_world(np.arange(number_of_wave_pix))).to_value(
        u.nm
    )

    return linear_wave


def resample_telluric_transmission_with_absorption_correction(
    linear_wave: np.ndarray, cropped_atlas: LocalAtlas, opacity_factor: float
) -> np.ndarray:
    """
    Resample the telluric transmission spectrum onto a linear wavelength grid and apply an absorption correction using the Beer–Lambert Law.

    This function interpolates the telluric transmission data from the provided atlas onto a linear wavelength vector, then modifies the transmission by applying an absorption scaling based on the
    Beer–Lambert Law:

        I(λ) = I(λ) ** opacity_factor

    This scaling simulates the attenuation of light due to atmospheric absorption, where the `opacity_factor` adjusts the strength of absorption across the spectrum.

    Parameters
    ----------
    linear_wave : np.ndarray
        1D array of linearly spaced wavelengths (in nanometers) onto which the telluric transmission will be interpolated.

    cropped_atlas : LocalAtlas
        Object containing the telluric atlas data, including wavelength and transmission arrays cropped to the relevant spectral region.

    opacity_factor : float
        Factor controlling the strength of telluric absorption. Values <1 reduce absorption, while values >1 increase it.

    Returns
    -------
    np.ndarray
        The resampled and absorption-corrected telluric transmission spectrum.
    """
    linear_fts_telluric_transmission = np.interp(
        linear_wave,
        cropped_atlas.telluric_atlas_wavelength.to_value(u.nm),
        cropped_atlas.telluric_atlas_transmission,
    )

    return linear_fts_telluric_transmission**opacity_factor


def resample_solar_transmission_with_doppler_shift(
    linear_wave: np.ndarray, cropped_atlas: LocalAtlas, doppler_velocity: Quantity, crval: float
) -> np.ndarray:
    """
    Interpolate the solar transmission onto the linear wavelength vector, and apply a doppler shift.

    Parameters
    ----------
    linear_wave : np.ndarray
        Linear wavelength vector in nanometers.
    cropped_atlas : LocalAtlas
        Cropped atlas containing telluric and solar atlas data.
    doppler_velocity : Quantity
        Doppler velocity in km/s.
    crval : float
        Reference wavelength in nanometers.

    Returns
    -------
    np.ndarray
        Solar transmission spectrum after applying the Doppler shift.
    """
    fts_wave_doppler_corrected = (
        cropped_atlas.solar_atlas_wavelength.to_value(u.nm)
        + doppler_velocity / (const.c.to("km/s")) * crval
    )
    return np.interp(
        linear_wave,
        fts_wave_doppler_corrected,
        cropped_atlas.solar_atlas_transmission,
    )


def combine_spectra(
    telluric_transmission: np.ndarray, solar_transmission: np.ndarray, straylight_fraction: float
) -> np.ndarray:
    """
    Combine the telluric and solar spectra with straylight contamination.

    Parameters
    ----------
    telluric_transmission : np.ndarray
        Telluric transmission spectrum.
    solar_transmission : np.ndarray
        Solar transmission spectrum.
    straylight_fraction : float
        Fraction of straylight contamination.

    Returns
    -------
    np.ndarray
        Combined spectrum with straylight contamination.
    """
    fts_solar_atmos_corr = telluric_transmission * solar_transmission
    return fts_solar_atmos_corr + straylight_fraction


def normalize_and_convolve(
    spectrum: np.ndarray,
    continuum_level: float,
    crval: float,
    resolving_power: float,
    dispersion: float,
) -> np.ndarray:
    """
    Normalize the spectrum and apply convolution for the spectrograph line spread function.

    Parameters
    ----------
    spectrum : np.ndarray
        The input spectrum to be normalized and convolved.
    continuum_level : float
        The continuum level to scale the spectrum.
    crval : float
        Reference wavelength in nanometers.
    resolving_power : float
        Resolving power of the spectrograph.
    dispersion : float
        Dispersion in nanometers per pixel.

    Returns
    -------
    np.ndarray
        The normalized and convolved spectrum.
    """
    fts_solar_normalized = spectrum * continuum_level
    sigma_wavelength = (crval / resolving_power) * gaussian_fwhm_to_sigma
    sigma_pix = sigma_wavelength / np.abs(dispersion)
    return gaussian_filter1d(fts_solar_normalized, np.abs(sigma_pix))


def fitting_model(
    params: Parameters,
    input_spectrum: np.ndarray,
    cropped_atlas: LocalAtlas,
    number_of_wave_pix: int,
    grating_constant: float,
    order: int,
    doppler_velocity: Quantity,
) -> np.ndarray:
    """
    Compute the fitting model function for fitting wavelength calibration parameters.

    Parameters
    ----------
    params : Parameters
        Parameters object containing the fitting parameters.
    input_spectrum : np.ndarray
        The observed 1D spectrum to be fitted.
    cropped_atlas : LocalAtlas
        Cropped atlas containing telluric and solar atlas data.
    number_of_wave_pix : int
        Number of wavelength pixels.
    grating_constant : float
        Grating constant of the spectrograph.
    order : int
        Diffraction order.
    doppler_velocity : Quantity
        Doppler velocity in km/s.

    Returns
    -------
    np.ndarray
        Residual amplitude between the observed spectrum and the fitted model.
    """
    crval = params["crval"].value
    dispersion = params["dispersion"].value
    opacity_factor = params["opacity_factor"].value
    straylight_fraction = params["straylight_fraction"].value
    continuum_level = params["continuum_level"].value
    resolving_power = params["resolving_power"].value

    # Calculate the linear wavelength vector
    linear_wave = calculate_linear_wave(params, number_of_wave_pix, grating_constant, order)

    # Apply telluric absorption
    telluric_transmission = resample_telluric_transmission_with_absorption_correction(
        linear_wave, cropped_atlas, opacity_factor
    )

    # Apply Doppler shift
    solar_transmission = resample_solar_transmission_with_doppler_shift(
        linear_wave, cropped_atlas, doppler_velocity, crval
    )

    # Combine spectra
    combined_spectrum = combine_spectra(
        telluric_transmission, solar_transmission, straylight_fraction
    )

    # Normalize and convolve
    fts_final_spectrum = normalize_and_convolve(
        combined_spectrum, continuum_level, crval, resolving_power, dispersion
    )

    # Calculate residual amplitude
    res_amplitude = input_spectrum - fts_final_spectrum

    return res_amplitude
