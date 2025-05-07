"""Function used during fitting."""
import astropy.units as u
import lmfit.parameter
import numpy as np
from astropy import constants as const
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.wcs import WCS
from lmfit import Parameters
from scipy.ndimage import gaussian_filter1d

from solar_wavelength_calibration.atlas.atlas import LocalAtlas


def calculate_linear_wave(
    params: lmfit.parameter.Parameters, number_of_wave_pix: int, grating_constant: float, order: int
) -> np.ndarray:
    """Calculate the linear wavelength vector.

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
    # References about representing spectral coordinate in FITS:
    # https://ui.adsabs.harvard.edu/abs/2006A%26A...446..747G/abstract (section 5)
    # https://specreduce.readthedocs.io/en/latest/api/specreduce.utils.synth_data.make_2d_arc_image.html

    from .wavelength_fitter import WavelengthParameters

    non_linear_header = WavelengthParameters(
        ctype="AWAV-GRA",
        cunit="nm",
        crpix=number_of_wave_pix // 2 + 1,
        grating_constant=grating_constant,
        order=order,
        crval=params["crval1"].value,
        dispersion=params["dispersion"].value,
        incident_light_angle=params["incident_light_angle"].value,
    )

    non_linear_wcs = WCS(non_linear_header.to_header(axis_num=1))
    linear_wave = (non_linear_wcs.spectral.pixel_to_world(np.arange(number_of_wave_pix))).to_value(
        u.nm
    )

    return linear_wave


def apply_telluric_absorption(linear_wave, cropped_atlas, opacity_factor) -> np.ndarray:
    """
    Apply telluric absorption to the spectrum.

    Parameters
    ----------
    linear_wave : np.ndarray
        Linear wavelength vector in nanometers.
    cropped_atlas : LocalAtlas
        Cropped atlas containing telluric and solar atlas data.
    opacity_factor : float
        Factor to scale the telluric absorption.

    Returns
    -------
    np.ndarray
        Telluric transmission spectrum after applying absorption.
    """
    linear_fts_telluric_transmission = np.interp(
        linear_wave,
        cropped_atlas.telluric_atlas_wavelength,
        cropped_atlas.telluric_atlas_transmission,
    )
    return linear_fts_telluric_transmission**opacity_factor


def apply_doppler_shift(linear_wave, cropped_atlas, doppler_velocity, crval1) -> np.ndarray:
    """
    Apply Doppler shift to the solar spectrum.

    Parameters
    ----------
    linear_wave : np.ndarray
        Linear wavelength vector in nanometers.
    cropped_atlas : LocalAtlas
        Cropped atlas containing telluric and solar atlas data.
    doppler_velocity : float
        Doppler velocity in km/s.
    crval1 : float
        Reference wavelength in nanometers.

    Returns
    -------
    np.ndarray
        Solar transmission spectrum after applying the Doppler shift.
    """
    fts_wave_doppler_corrected = (
        cropped_atlas.solar_atlas_wavelength + doppler_velocity / (const.c.to("km/s")) * crval1
    )
    return np.interp(
        linear_wave,
        fts_wave_doppler_corrected,
        cropped_atlas.solar_atlas_transmission,
    )


def combine_spectra(telluric_transmission, solar_transmission, straylight_fraction) -> np.ndarray:
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
    spectrum, continuum_level, crval1, resolving_power, dispersion
) -> np.ndarray:
    """
    Normalize the spectrum and apply convolution for the spectrograph line spread function.

    Parameters
    ----------
    spectrum : np.ndarray
        The input spectrum to be normalized and convolved.
    continuum_level : float
        The continuum level to scale the spectrum.
    crval1 : float
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
    sigma_wavelength = (crval1 / resolving_power) * gaussian_fwhm_to_sigma
    sigma_pix = sigma_wavelength / np.abs(dispersion)
    return gaussian_filter1d(fts_solar_normalized, np.abs(sigma_pix))


def fitting_model(
    params: Parameters,
    unfit_1d_spectrum: np.ndarray,
    cropped_atlas: LocalAtlas,
    number_of_wave_pix: int,
    grating_constant: float,
    order: int,
    doppler_velocity: float,
) -> np.ndarray:
    """
    Compute the fitting model function for fitting wavelength calibration parameters.

    Parameters
    ----------
    params : Parameters
        Parameters object containing the fitting parameters.
    unfit_1d_spectrum : np.ndarray
        The observed 1D spectrum to be fitted.
    cropped_atlas : LocalAtlas
        Cropped atlas containing telluric and solar atlas data.
    number_of_wave_pix : int
        Number of wavelength pixels.
    grating_constant : float
        Grating constant of the spectrograph.
    order : int
        Diffraction order.
    doppler_velocity : float
        Doppler velocity in km/s.

    Returns
    -------
    np.ndarray
        Residual amplitude between the observed spectrum and the fitted model.
    """
    crval1 = params["crval1"].value
    dispersion = params["dispersion"].value
    opacity_factor = params["opacity_factor"].value
    straylight_fraction = params["straylight_fraction"].value
    continuum_level = params["continuum_level"].value
    resolving_power = params["resolving_power"].value

    # Calculate the linear wavelength vector
    linear_wave = calculate_linear_wave(params, number_of_wave_pix, grating_constant, order)

    # Apply telluric absorption
    telluric_transmission = apply_telluric_absorption(linear_wave, cropped_atlas, opacity_factor)

    # Apply Doppler shift
    solar_transmission = apply_doppler_shift(linear_wave, cropped_atlas, doppler_velocity, crval1)

    # Combine spectra
    combined_spectrum = combine_spectra(
        telluric_transmission, solar_transmission, straylight_fraction
    )

    # Normalize and convolve
    fts_final_spectrum = normalize_and_convolve(
        combined_spectrum, continuum_level, crval1, resolving_power, dispersion
    )

    # Calculate residual amplitude
    res_amplitude = unfit_1d_spectrum - fts_final_spectrum

    return res_amplitude
