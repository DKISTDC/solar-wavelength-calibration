"""Machinery to collect data and parameter objects and fit the wavelength axis."""

import logging
from functools import cached_property
from typing import Callable

import astropy.units as u
import lmfit.parameter
import numpy as np
from astropy import constants as const
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.units import Quantity
from astropy.wcs import WCS
from lmfit.minimizer import MinimizerResult
from lmfit.parameter import Parameters
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator
from scipy.ndimage import gaussian_filter1d

from solar_wavelength_calibration.atlas.atlas import Atlas
from solar_wavelength_calibration.atlas.atlas import LocalAtlas
from solar_wavelength_calibration.atlas.base import AtlasBase
from solar_wavelength_calibration.fitter.parameters import WavelengthCalibrationParameters
from solar_wavelength_calibration.fitter.parameters import scalar_continuum_function

logger = logging.getLogger(__name__)


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

    def to_header(self, axis_num: int, add_alternate_keys: bool = False) -> dict:
        """
        Convert the wavelength parameters to a WCS header dictionary.

        Parameters
        ----------
        axis_num
            The axis number for the WCS header (e.g., 1 for the first axis).

        add_alternate_keys
            If True, include alternate header keywords (ending with 'A') for compatibility.

        Returns
        -------
        header dictionary
            A dictionary representing the WCS header with keys corresponding to FITS header keywords and values derived
            from the wavelength parameters.
        """
        header = {
            f"CTYPE{axis_num}": self.ctype,
            f"CUNIT{axis_num}": self.cunit,
            f"CRPIX{axis_num}": self.crpix,
            f"CRVAL{axis_num}": self.crval,
            f"CDELT{axis_num}": self.dispersion,
            f"PV{axis_num}_0": self.grating_constant,
            f"PV{axis_num}_1": self.order,
            f"PV{axis_num}_2": self.incident_light_angle,
        }
        if add_alternate_keys:
            header[f"CTYPE{axis_num}A"] = self.ctype
            header[f"CUNIT{axis_num}A"] = self.cunit
            header[f"CRPIX{axis_num}A"] = self.crpix
            header[f"CRVAL{axis_num}A"] = self.crval
            header[f"CDELT{axis_num}A"] = self.dispersion
            header[f"PV{axis_num}_0A"] = self.grating_constant
            header[f"PV{axis_num}_1A"] = self.order
            header[f"PV{axis_num}_2A"] = self.incident_light_angle
        return header


class FitResult(BaseModel):
    """Represents the output of the fitting process, including the WCS header information and optional fit parameters."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    wavelength_parameters: WavelengthParameters
    minimizer_result: MinimizerResult
    input_wavelength_vector: u.Quantity
    input_spectrum: np.ndarray
    spectral_weights: None | np.ndarray = None
    continuum_function: Callable[[np.ndarray, Parameters], np.ndarray] = scalar_continuum_function

    @cached_property
    def prepared_weights(self) -> np.ndarray:
        """
        Return the weights as-used for fitting.

        Specifically, return the square-root of the weights after normalizing their sum to 1. See
        `fitting_model` for more information.
        """
        weight_data = (
            np.ones(self.input_wavelength_vector.size)
            if self.spectral_weights is None
            else self.spectral_weights
        )
        prepared_weights = np.sqrt(weight_data / np.sum(weight_data))

        return prepared_weights

    @cached_property
    def num_wave_pix(self) -> int:
        """Return the number of spectral pixels in the input spectrum."""
        return self.input_spectrum.size

    @cached_property
    def best_fit_wavelength_vector(self) -> np.ndarray:
        """
        Compute a wavelength vector given the parameterization in `wavelength_parameters`.

        The parameterization of the grating equation is via `astropy.wcs.WCS`, which follows section 5 of
        `Greisen et al (2006) <https://ui.adsabs.harvard.edu/abs/2006A%26A...446..747G/abstract>`_.

        Returns
        -------
        best_fit_wavelength
            Wavelength vector corresponding to the `wavelength_parameters`. Units are nm.
        """
        wcs = WCS(self.wavelength_parameters.to_header(axis_num=1))
        return wcs.spectral.pixel_to_world(np.arange(self.num_wave_pix)).to_value(u.nm)

    @cached_property
    def best_fit_continuum(self) -> np.ndarray:
        """
        Return the best-fit continuum value(s).

        This is equal to `continuum_function(self.best_fit_wavelength_vector, self.minimizer_result.params)`
        """
        return self.continuum_function(
            self.best_fit_wavelength_vector, self.minimizer_result.params
        )

    @cached_property
    def best_fit_atlas(self) -> np.ndarray:
        """
        Return the best-fit combined Solar and Telluric atlas.

        This is the exact atlas combination (with absorption, doppler-shift, straylight, resolving power, and continuum
        corrections applied) that produced the best fit.
        """
        residuals = self.minimizer_result.residual / self.prepared_weights
        best_fit_atlas = self.input_spectrum - residuals

        return best_fit_atlas


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
        input_spectrum: np.ndarray,
        spectral_weights: None | np.ndarray = None,
        method: str = "leastsq",
        **minimize_kwargs,
    ) -> FitResult:
        """Run data preparation and fitting.

        Parameters
        ----------
        input_spectrum
            The input 1D spectrum to be fitted.

        spectral_weights
            Array of weights to apply to the residuals during fitting. If None, no weights are applied.

        method
            The optimization method to use. Defaults to "leastsq".

        spectral_weights
            Array of weights to apply to the residuals during fitting. If None, no weights are applied.

        **minimize_kwargs
            Additional keyword arguments for the minimizer.

        Returns
        -------
        fit_result:
            An instance of `FitResult`
        """
        if spectral_weights is None:
            spectral_weights = np.ones(len(input_spectrum))
        if len(spectral_weights) != len(input_spectrum):
            raise ValueError(
                f"Length of spectral_weights ({len(spectral_weights)}) does not match length of input_spectrum ({len(input_spectrum)})."
            )

        return self._run_fit(
            input_spectrum=input_spectrum,
            spectral_weights=spectral_weights,
            method=method,
            **minimize_kwargs,
        )

    def _run_fit(
        self,
        input_spectrum: np.ndarray,
        spectral_weights: np.ndarray,
        method: str = "leastsq",
        **minimize_kwargs,
    ) -> FitResult:
        """Run the fit."""
        input_wavelength_vector = self._compute_input_wavelength_vector(
            input_spectrum=input_spectrum
        )
        cropped_atlas = self._prepare_atlas(input_wavelength_vector)
        fit_result = self._fit_spectrum(
            cropped_atlas=cropped_atlas,
            input_wavelength_vector=input_wavelength_vector,
            input_spectrum=input_spectrum,
            method=method,
            spectral_weights=spectral_weights,
            **minimize_kwargs,
        )
        return fit_result

    def _compute_input_wavelength_vector(self, input_spectrum: np.ndarray) -> Quantity:
        """Compute a wavelength vector given an input spectrum and initial set of parameters."""
        num_wave_pix = input_spectrum.size
        order = self.input_parameters.order
        grating_constant_inverse_m = self.input_parameters.grating_constant.to_value(1 / u.m)
        init_params = self.input_parameters.lmfit_parameters

        raw_wave = calculate_linear_wave(
            params=init_params,
            number_of_wave_pix=num_wave_pix,
            grating_constant=grating_constant_inverse_m,
            order=order,
        )
        return raw_wave * u.nm

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
        input_wavelength_vector: u.Quantity,
        input_spectrum: np.ndarray,
        method: str,
        spectral_weights: np.ndarray,
        **minimize_kwargs,
    ) -> FitResult:
        """Send the fitter on its way."""
        params_to_fit = self.input_parameters.lmfit_parameters

        # Normalize weights to sum to 1, then take the square root.
        # This ensures that when res_amplitude is multiplied by these weights in fitting_model,
        # the default lmfit-computed chisq is equivalent to a weighted sum of squared residuals.
        normalized_weights = spectral_weights / np.sum(spectral_weights)
        prepared_weights = np.sqrt(normalized_weights)

        fitting_model_kwargs = self.input_parameters.constant_parameters | {
            "prepared_weights": prepared_weights,
        }

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
            fcn_kws=fitting_model_kwargs,
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
            grating_constant=self.input_parameters.grating_constant.value,
            order=self.input_parameters.order,
            incident_light_angle=minimizer_result.params["incident_light_angle"].value,
            ctype="AWAV-GRA",
            cunit="nm",
        )

        return FitResult(
            wavelength_parameters=wavelength_parameters,
            minimizer_result=minimizer_result,
            input_wavelength_vector=input_wavelength_vector,
            input_spectrum=input_spectrum,
            spectral_weights=spectral_weights,
            continuum_function=self.input_parameters.continuum_function,
        )


def calculate_linear_wave(
    params: lmfit.parameter.Parameters, number_of_wave_pix: int, grating_constant: float, order: int
) -> np.ndarray:
    """Calculate a linearly sampled wavelength vector given parameters describing a non-linear spectral WCS.

    We use `astropy.wcs.WCS`, which follows the parameterization of
    `Greisen et al (2006) <https://ui.adsabs.harvard.edu/abs/2006A%26A...446..747G/abstract>`_ (see section 5). In
    particular, the parameterization used here assumes a reflection grating in air so that higher-order terms above
    ``PVk_2`` can be ignored.

    Parameters
    ----------
    params
        Parameters object containing the fitting parameters.

    number_of_wave_pix
        Number of wavelength pixels.

    grating_constant
        Grating constant of the spectrograph. Must have units 1 / m

    order
        Diffraction order.

    Returns
    -------
    linear_wave
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
    r"""
    Resample the telluric transmission spectrum onto a linear wavelength grid and apply an absorption correction using the Beer–Lambert Law.

    This function interpolates the telluric transmission data from the provided atlas onto a linear wavelength vector,
    then scales the assumed optical depth by value of `opacity_factor`, :math:`T` such that

    .. math::

        I(\lambda) = I_0(\lambda)^T

    This scaling simulates the attenuation of light due to atmospheric absorption, where the `opacity_factor` adjusts the strength of absorption across the spectrum.

    Parameters
    ----------
    linear_wave
        1D array of linearly spaced wavelengths (in nanometers) onto which the telluric transmission will be interpolated.

    cropped_atlas
        Object containing the telluric atlas data, including wavelength and transmission arrays cropped to the relevant spectral region.

    opacity_factor
        Factor controlling the strength of telluric absorption. Values < 1 reduce absorption, while values > 1 increase it.

    Returns
    -------
    telluric spectrum
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
    linear_wave
        Linear wavelength vector in nanometers.

    cropped_atlas
        Cropped atlas containing telluric and solar atlas data.

    doppler_velocity
        Doppler velocity in km/s.

    crval
        Reference wavelength in nanometers.

    Returns
    -------
    solar spectrum
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


def apply_resolving_power(
    spectrum: np.ndarray,
    crval: float,
    resolving_power: float,
    dispersion: float,
) -> np.ndarray:
    """
    Apply convolution for the spectrograph line spread function.

    Parameters
    ----------
    spectrum
        The input spectrum to be normalized and convolved.

    crval
        Reference wavelength in nanometers.

    resolving_power
        Resolving power of the spectrograph.

    dispersion
        Dispersion in nanometers per pixel.

    Returns
    -------
    modified spectrum
        The normalized and convolved spectrum.
    """
    sigma_wavelength = (crval / resolving_power) * gaussian_fwhm_to_sigma
    sigma_pix = sigma_wavelength / np.abs(dispersion)
    return gaussian_filter1d(spectrum, np.abs(sigma_pix))


def fitting_model(
    params: Parameters,
    input_spectrum: np.ndarray,
    cropped_atlas: LocalAtlas,
    number_of_wave_pix: int,
    grating_constant: float,
    order: int,
    doppler_velocity: Quantity,
    prepared_weights: np.ndarray,
    continuum_function: Callable[[np.ndarray, Parameters], np.ndarray],
) -> np.ndarray:
    r"""
    Compute the fitting model function for fitting wavelength calibration parameters.

    See :doc:`this page </physical_model>` for more information.

    Parameters
    ----------
    params
        Parameters object containing the fitting parameters.

    input_spectrum
        The observed 1D spectrum to be fitted.

    cropped_atlas
        Cropped atlas containing telluric and solar atlas data.

    number_of_wave_pix
        Number of wavelength pixels.

    grating_constant
        Grating constant of the spectrograph.

    order
        Diffraction order.

    doppler_velocity
        Doppler velocity in km/s.

    prepared_weights
        Array of prepared weights (square roots of normalized spectral weights) to apply to the residuals to ensure proper weighting in the fit.
        The lmfit minimizer computes chi-squared as the sum of squared residuals, i.e., :math:`\chi^2 = \sum_i r_i^2`,
        where :math:`r`, the residuals, are the result of this function. To achieve a standard weighted chi-squared,
        (:math:`\chi^2 = \frac{\sum_i w_i r_i^2}{\sum_i w_i}`), we multiply each residual by the prepared weights such that
        :math:`r_w = r \sqrt{w_i}`. The lmfit goodness-of-fit then becomes :math:`\chi^2 = \sum_i r_{w,i}^2 = \sum_i w_i r_i^2`,
        which is equal to the standard weighted chi-squared when :math:`\sum_i w_i = 1`.

    continuum_function:
        Function that defines the continuum to be applied to the atlas spectrum. Must take the currently fit wavelength
        vector and fit `lmfit.Parameters` as its only two arguments and return a value that can be broadcast in
        multiplication to the combined atlas spectrum. Default is a scalar value (the value of the
        `~solar_wavelength_calibration.fitter.parameters.WavelengthCalibrationParameters.continuum_level` parameter).

    Returns
    -------
    residuals
        Residual amplitude between the observed spectrum and the fitted model.
    """
    crval = params["crval"].value
    dispersion = params["dispersion"].value
    opacity_factor = params["opacity_factor"].value
    straylight_fraction = params["straylight_fraction"].value
    resolving_power = params["resolving_power"].value

    # Calculate the linear wavelength vector
    linear_wave = calculate_linear_wave(params, number_of_wave_pix, grating_constant, order)

    # Resample telluric and solar atlases onto `linear_wave` and apply absorption and doppler shift corrections,
    # respectively
    telluric_transmission = resample_telluric_transmission_with_absorption_correction(
        linear_wave=linear_wave, cropped_atlas=cropped_atlas, opacity_factor=opacity_factor
    )
    solar_transmission = resample_solar_transmission_with_doppler_shift(
        linear_wave=linear_wave,
        cropped_atlas=cropped_atlas,
        doppler_velocity=doppler_velocity,
        crval=crval,
    )

    combined_atlas_spectrum = telluric_transmission * solar_transmission
    degraded_atlas_spectrum = apply_resolving_power(
        spectrum=combined_atlas_spectrum,
        crval=crval,
        resolving_power=resolving_power,
        dispersion=dispersion,
    )

    # Divide by 1 + straylight_fraction so the continuum level is preserved at 1. We don't want the straylight fraction
    # to couple with the continuum fit.
    atlas_spectrum_w_straylight = (degraded_atlas_spectrum + straylight_fraction) / (
        1 + straylight_fraction
    )

    try:
        continuum = continuum_function(linear_wave, params)
    except Exception as e:
        raise RuntimeError("The continuum function has produced an unrecoverable error.") from e
    final_atlas_spectrum = atlas_spectrum_w_straylight * continuum

    res_amplitude = input_spectrum - final_atlas_spectrum
    res_amplitude *= prepared_weights

    return res_amplitude


def calculate_initial_crval_guess(
    input_wavelength_vector: u.Quantity,
    input_spectrum: np.ndarray,
    atlas: AtlasBase,
    negative_limit=-2 * u.nm,
    positive_limit=2 * u.nm,
    num_steps=550,
) -> u.Quantity:
    """
    Estimate the initial guess for the `crval` parameter by aligning the input wavelength vector with the atlas data.

    This function calculates a shift for the input wavelength vector to minimize the
    difference between the input spectrum and the product of the solar and telluric
    atlas transmissions. The optimal shift is determined by evaluating a merit
    function over a range of possible shifts.

    Note
    ----
    This function does not perform any continuum matching between the input spectrum and the atlas spectra.

    Parameters
    ----------
    input_wavelength_vector
        The preliminary wavelength vector corresponding to the input spectrum.

    input_spectrum
        The observed 1D spectrum to be fitted.

    atlas
        The atlas object containing solar and telluric wavelength and transmission data.

    negative_limit
        The lower bound of the wavelength shift range for the initial guess (default: -2 nm).

    positive_limit
        The upper bound of the wavelength shift range for the initial guess (default: 2 nm).

    num_steps
        The number of steps to evaluate in the shift range (default: 550).

    Returns
    -------
    crval_init
        The estimated initial value for the `crval` parameter, representing the reference wavelength
    """
    shifts = np.linspace(negative_limit, positive_limit, num_steps)
    merit = np.zeros(len(shifts))
    for n, shift in enumerate(shifts):
        preliminary_wavelength = input_wavelength_vector + shift
        fts_solar = np.interp(
            preliminary_wavelength, atlas.solar_atlas_wavelength, atlas.solar_atlas_transmission
        )
        fts_telluric = np.interp(
            preliminary_wavelength,
            atlas.telluric_atlas_wavelength,
            atlas.telluric_atlas_transmission,
        )
        # calculate a merit value to be minimized
        merit[n] = np.sum((input_spectrum - fts_solar * fts_telluric) ** 2)

    # get minimum
    shift = shifts[np.argmin(merit)]

    # recalculate spectral axis and atlas spectrum for the best shift value
    fts_wave = input_wavelength_vector + shift

    crpix_updated = np.size(input_spectrum) // 2 + 1
    crval_initial_guess = fts_wave[crpix_updated]

    return crval_initial_guess
