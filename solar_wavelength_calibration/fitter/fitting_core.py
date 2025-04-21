"""Function used during fitting."""
import astropy.units as u
import numpy as np
from astropy import constants as const
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.wcs import WCS
from lmfit import Parameters
from scipy.ndimage import gaussian_filter1d

from solar_wavelength_calibration.atlas.atlas import CroppedAtlas


def _fitness(
    params: Parameters,
    unfit_1d_spectrum: np.ndarray,
    cropped_atlas: CroppedAtlas,
    number_of_wave_pix: int,
    grating_constant: float,
    order: int,
    doppler_velocity: float,
) -> np.ndarray:
    """Compute _fitness function for fitting wavecal parameters.

    Parameters
    ----------
    params
        Object containing information about the parameters to be used in the fit

    unfit_1d_spectrum
        An array with the input (unfit) 1D spectrum.

    cropped_atlas
        The Fourier Transform Spectrometer (FTS) atlas cropped to a range slightly larger than the expected wavelength vector. Cropped_atlas contains the solar wavelength cropped around the
        input data, the telluric wavelength cropped around the input data, and their corresponding transmission arrays.

    number_of_wave_pix
        The number of wavelength pixels in the unfit_1d_spectrum.

    grating_constant
        Instrument specific grating constant.

    order
        Instrument specific optical order.

    doppler_velocity
        Calculated speed at which instrument is moving relative to the Sun's center at time of observation. Positive values refer to when instrument is moving away from the sun.
    """
    param_vals = params.valuesdict()
    crval1 = param_vals["crval1"]
    dispersion = param_vals["dispersion"]
    incident_light_angle = param_vals["incident_light_angle"]
    resolving_power = param_vals["resolving_power"]
    opacity_factor = param_vals["opacity_factor"]
    straylight_fraction = param_vals["straylight_fraction"]
    continuum_level = param_vals["continuum_level"]

    # calculate the spectral axis
    # References about representing spectral coordinate in FITS:
    # https://ui.adsabs.harvard.edu/abs/2006A%26A...446..747G/abstract (section 5)
    # https://specreduce.readthedocs.io/en/latest/api/specreduce.utils.synth_data.make_2d_arc_image.html

    non_linear_header = {
        "CTYPE1": "AWAV-GRA",  # Grating dispersion function with air wavelengths
        "CUNIT1": "nm",  # Dispersion units
        "CRPIX1": number_of_wave_pix // 2 + 1,  # Reference pixel [pix]
        "PV1_0": grating_constant,  # Grating density
        "PV1_1": order,  # Diffraction order
        "CRVAL1": crval1,  # Reference value [nm] (<<<< TO BE OPTIMIZED <<<<<<)
        "CDELT1": dispersion,  # Linear dispersion [nm/pix] (<<<< TO BE OPTIMIZED <<<<<<)
        "PV1_2": incident_light_angle,  # Incident angle [deg] (<<<< TO BE OPTIMIZED <<<<<<)
    }

    non_linear_wcs = WCS(non_linear_header)
    linear_wave = (  # linear_wavelength_vector
        non_linear_wcs.spectral.pixel_to_world(np.arange(number_of_wave_pix))
    ).to_value(u.nm)

    # interpolate the telluric spectral atlas onto the new wavelength axis and scale by the opacity value that is being optimized
    linear_fts_telluric_transmission = np.interp(
        linear_wave,
        cropped_atlas.telluric_atlas_wavelength,
        cropped_atlas.telluric_atlas_transmission,
    )
    opacity_scaled_fts_telluric_transmission = (
        linear_fts_telluric_transmission
        ** opacity_factor  # relative scaling of each wavelength based on the atmosphere during the observations
    )  # in parameters

    # interpolate the solar spectral atlas onto the new wavelength axis and apply a shift according to the Doppler shift due to orbital motions (doppler velocity)
    fts_wave_doppler_corrected = (
        cropped_atlas.solar_atlas_wavelength + doppler_velocity / (const.c.to("km/s")) * crval1
    )
    linear_fts_solar_transmission = np.interp(
        linear_wave,
        fts_wave_doppler_corrected,
        cropped_atlas.solar_atlas_transmission,
    )

    # apply telluric absorption spectrum to solar spectrum
    fts_solar_atmos_corr = opacity_scaled_fts_telluric_transmission * linear_fts_solar_transmission
    # add flat value of straylight contamination
    fts_solar_with_straylight = fts_solar_atmos_corr + straylight_fraction

    # scale for total intensity of the continuum
    fts_solar_normalized = fts_solar_with_straylight * continuum_level

    # convolution for spectrograph line spread function
    sigma_wavelength = (crval1 / resolving_power) * gaussian_fwhm_to_sigma
    sigma_pix = sigma_wavelength / np.abs(dispersion)  # used to match FTS resolution to our data
    fts_final_spectrum = gaussian_filter1d(fts_solar_normalized, sigma_pix)

    res_amplitude = unfit_1d_spectrum - fts_final_spectrum

    return res_amplitude
