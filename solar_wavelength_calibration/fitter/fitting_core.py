"""Function used during fitting."""
import astropy.units as u
import numpy as np
from astropy import constants as const
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.wcs import WCS
from lmfit import Parameters
from scipy.ndimage import gaussian_filter1d

from solar_wavelength_calibration.atlas.atlas import CroppedAtlas


def calculate_linear_wave(params, number_of_wave_pix, grating_constant, order):
    """Calculate the linear wavelength vector."""
    param_vals = params.valuesdict()
    crval1 = param_vals["crval1"]
    dispersion = param_vals["dispersion"]
    incident_light_angle = param_vals["incident_light_angle"]

    non_linear_header = {
        "CTYPE1": "AWAV-GRA",
        "CUNIT1": "nm",
        "CRPIX1": number_of_wave_pix // 2 + 1,
        "PV1_0": grating_constant,
        "PV1_1": order,
        "CRVAL1": crval1,
        "CDELT1": dispersion,
        "PV1_2": incident_light_angle,
    }

    non_linear_wcs = WCS(non_linear_header)
    linear_wave = (non_linear_wcs.spectral.pixel_to_world(np.arange(number_of_wave_pix))).to_value(
        u.nm
    )

    return linear_wave


def apply_telluric_absorption(linear_wave, cropped_atlas, opacity_factor):
    """Apply telluric absorption to the spectrum."""
    linear_fts_telluric_transmission = np.interp(
        linear_wave,
        cropped_atlas.telluric_atlas_wavelength,
        cropped_atlas.telluric_atlas_transmission,
    )
    return linear_fts_telluric_transmission**opacity_factor


def apply_doppler_shift(linear_wave, cropped_atlas, doppler_velocity, crval1):
    """Apply Doppler shift to the solar spectrum."""
    fts_wave_doppler_corrected = (
        cropped_atlas.solar_atlas_wavelength + doppler_velocity / (const.c.to("km/s")) * crval1
    )
    return np.interp(
        linear_wave,
        fts_wave_doppler_corrected,
        cropped_atlas.solar_atlas_transmission,
    )


def combine_spectra(telluric_transmission, solar_transmission, straylight_fraction):
    """Combine the telluric and solar spectra with straylight contamination."""
    fts_solar_atmos_corr = telluric_transmission * solar_transmission
    return fts_solar_atmos_corr + straylight_fraction


def normalize_and_convolve(spectrum, continuum_level, crval1, resolving_power, dispersion):
    """Normalize the spectrum and apply convolution for the spectrograph line spread function."""
    fts_solar_normalized = spectrum * continuum_level
    sigma_wavelength = (crval1 / resolving_power) * gaussian_fwhm_to_sigma
    sigma_pix = sigma_wavelength / np.abs(dispersion)
    return gaussian_filter1d(fts_solar_normalized, np.abs(sigma_pix))


def fitting_model(
    params: Parameters,
    unfit_1d_spectrum: np.ndarray,
    cropped_atlas: CroppedAtlas,
    number_of_wave_pix: int,
    grating_constant: float,
    order: int,
    doppler_velocity: float,
) -> np.ndarray:
    """Compute fitting_model function for fitting wavecal parameters."""
    param_vals = params.valuesdict()
    crval1 = param_vals["crval1"]
    dispersion = param_vals["dispersion"]
    opacity_factor = param_vals["opacity_factor"]
    straylight_fraction = param_vals["straylight_fraction"]
    continuum_level = param_vals["continuum_level"]
    resolving_power = param_vals["resolving_power"]

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
