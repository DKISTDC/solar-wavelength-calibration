"""Function used during fitting."""
import astropy.units as u
import numpy as np
from astropy import constants as const
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.wcs import WCS
from lmfit import Parameters
from scipy.ndimage import gaussian_filter1d


def fitness(
    params: Parameters,
    unfit_1d_spectrum: np.ndarray,
    fts_wave: np.ndarray,
    fts_telluric: np.ndarray,
    fts_solar: np.ndarray,
    number_of_wave_pix: int,
    grating_constant: float,
    order: int,
    doppler_velocity: float,
) -> np.ndarray:
    """Compute fitness function for fitting wavecal parameters.

    Parameters
    ----------
    params
        Object containing information about the parameters to be used in the fit

    unfit_1d_spectrum
        An array with the input (unfit) 1D spectrum.

    fts_wave
        FTS Atlas wavelength array cropped around the input data.

    fts_telluric
        Telluric FTS atlas transmission array corresponding to fts_wave object.

    fts_solar
        Solar FTS atlas transmission array corresponding to fts_wave object.

    grating_constant
        Instrument specific grating constant.

    order
        Instrument specific optical order.

    doppler_velocity
        Calculated doppler velocity of DKIST at time of observation.
    """
    param_vals = params.valuesdict()
    crval1 = param_vals["crval1"]
    dispersion = param_vals["dispersion"]
    incident_light_angle = param_vals["incident_light_angle"]
    resolving_power = param_vals["resolving_power"]
    opacity = param_vals["opacity_factor"]
    stray_light_fraction = param_vals["straylight_fraction"]
    continuum_intensity_correction = param_vals["continuum_intensity_correction"]

    # calculate the spectral axis
    # Representations of spectral coordinates in FITS
    # https://ui.adsabs.harvard.edu/abs/2006A%26A...446..747G/abstract (section 5)
    # https://specreduce.readthedocs.io/en/latest/api/specreduce.utils.synth_data.make_2d_arc_image.html

    non_linear_header = {
        "CTYPE1": "AWAV-GRA",  # Grating dispersion function with air wavelengths
        "CUNIT1": "nm",  # Dispersion units
        "CRPIX1": number_of_wave_pix // 2 + 1,  # Reference pixel [pix]
        "PV1_0": grating_constant,  # Grating density
        "PV1_1": order,  # Diffraction order
        "CRVAL1": crval1,  # Reference value [nm] (<<<< TO BE OPTIMIZED <<<<<<)
        "CDELT1": dispersion,  # Linear dispersion [nm/pix] (<<<< TO BE OPTIMIZED <<<<<<) # TODO: dispersion param is in nm
        "PV1_2": np.rad2deg(
            incident_light_angle
        ),  # Incident angle [deg] (<<<< TO BE OPTIMIZED <<<<<<)
    }

    non_linear_wcs = WCS(non_linear_header)
    wavelength_vector = (  # linear_wavelength_vector
        non_linear_wcs.spectral.pixel_to_world(np.arange(number_of_wave_pix))
    ).to_value(u.nm)

    # interpolate the telluric spectral atlas onto the new wavelength axis and scale by the opacity value that is being optimized
    fts_telluric_transmission_interp = np.interp(wavelength_vector, fts_wave, fts_telluric)
    fts_telluric_transmission_atmos_corr = (
        fts_telluric_transmission_interp
        ** opacity  # relative scaling of each wavelength based on the atmosphere during the observations
    )  # in parameters

    # interpolate the solar spectral atlas onto the new wavelength axis and apply a shift according to the Doppler shift due to orbital motions (doppler velocity)
    fts_solar_interp = np.interp(
        wavelength_vector,
        fts_wave + doppler_velocity / (const.c.to_value("km/s")) * crval1,
        fts_solar,
    )

    # apply telluric absorption spectrum to solar spectrum
    fts_modulated = fts_telluric_transmission_atmos_corr * fts_solar_interp
    # add flat value of straylight contamination
    fts_modulated_with_straylight = fts_modulated + stray_light_fraction

    # scale for total intensity of the continuum
    fit_amplitude = fts_modulated_with_straylight * continuum_intensity_correction

    # convolution for spectrograph line spread function
    sigma_wavelength = (crval1 / resolving_power) * gaussian_fwhm_to_sigma
    kern_pix = sigma_wavelength / np.abs(dispersion)  # used to match FTS resolution to our data
    fit_amplitude = gaussian_filter1d(fit_amplitude, kern_pix)

    res_amplitude = unfit_1d_spectrum - fit_amplitude

    return res_amplitude
