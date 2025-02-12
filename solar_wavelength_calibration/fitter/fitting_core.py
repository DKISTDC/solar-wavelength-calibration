"""Function used during fitting."""
import astropy.units as u
import numpy as np
from astropy import constants as const
from astropy.wcs import WCS
from lmfit import Parameters
from scipy.ndimage import gaussian_filter1d


def fitness(
    params: Parameters,
    spectrum: np.ndarray,
    target_solar_atlas_wavelength: np.ndarray,
    target_telluric_atlas_wavelength: np.ndarray,
    target_telluric_atlas: np.ndarray,
    target_solar_atlas: np.ndarray,
    grating_constant: float,
    order: int,
    doppler_shift: float,
):
    """Compute fitness function for fitting wavecal parameters.

    Parameters
    ----------
    Params
        Object containing information about the parameters to be used in the fit

    Spectrum
        An object with the input (unfitted) 1D spectrum.

    grating_constant

    order

    target_atlas_wavelength

    target_telluric_atlas

    target_solar_atlas

    doppler_shift
    """
    # TODO: update these parameter names to the same as what is used across the module
    param_vals = params.valuesdict()
    crval1 = param_vals.get("cravl1")
    cdelt1 = param_vals.get("dispersion")
    incident_light_angle = param_vals.get("incident_light_angle")
    resolving_power = param_vals.get("resolving_power")
    opacity = param_vals.get("opacity_factor")
    stray_light_frac = param_vals.get("straylight_fraction")
    continuum_amplitude = param_vals.get("continuum_intensity_correction")

    # calculate the spectral axis
    # Representations of spectral coordinates in FITS
    # https://ui.adsabs.harvard.edu/abs/2006A%26A...446..747G/abstract
    # https://specreduce.readthedocs.io/en/latest/api/specreduce.utils.synth_data.make_2d_arc_image.html

    number_of_wave_pix = np.size(spectrum)

    non_linear_header = {
        "CTYPE1": "AWAV-GRA",  # Grating dispersion function with air wavelengths
        "CUNIT1": "nm",  # Dispersion units
        "CRPIX1": number_of_wave_pix // 2 + 1,  # Reference pixel [pix]
        "PV1_0": grating_constant.value,  # Grating density
        "PV1_1": order.value,  # Diffraction order
        "CRVAL1": crval1,  # Reference value [nm] (<<<< TO BE OPTMIZED <<<<<<)
        "CDELT1": cdelt1,  # Linear dispersion [nm/pix] (<<<< TO BE OPTMIZED <<<<<<)
        "PV1_2": incident_light_angle,  # Incident angle [deg] (<<<< TO BE OPTMIZED <<<<<<)
    }

    non_linear_wcs = WCS(non_linear_header)
    wavelength_vector = (
        (non_linear_wcs.spectral.pixel_to_world(np.arange(number_of_wave_pix))).to(u.nm).value
    )

    # Gaussian convolution of the FTS atlas
    fwhm_wavelength = np.divide(crval1, resolving_power)  # both in parameters
    sigma_wavelength = fwhm_wavelength / (2.0 * np.sqrt(2.0 * np.log(2)))
    kern_pix = sigma_wavelength / np.abs(cdelt1)

    # interpolate the telluric spectral atlas onto the new wavelength axis and scale by the opacity value that is being optimized
    fts_atmosphere_interp = np.interp(
        wavelength_vector, target_telluric_atlas_wavelength, target_telluric_atlas
    )
    fts_telluric_interp = np.exp(opacity * np.log(fts_atmosphere_interp))  # in parameters

    # interpolate the solar spectral atlas onto the new wavelength axis and apply a shift according to the Doppler shift due to orbital motions
    fts_solar_interp = np.interp(
        wavelength_vector,
        target_solar_atlas_wavelength + doppler_shift.value / (const.c.to("km/s")).value * crval1,
        target_solar_atlas,
    )

    # apply telluric absorption spectrum to solar spectrum
    fts_modulated = fts_telluric_interp * fts_solar_interp
    # add flat value of straylight contamination
    fts_modulated_with_straylight = (fts_modulated + stray_light_frac) / (1.0 + stray_light_frac)
    # scale for total intensity of the continuum
    fit_amplitude = fts_modulated_with_straylight * continuum_amplitude

    # convolution for spectrograph line spread function
    fit_amplitude = gaussian_filter1d(fit_amplitude, kern_pix)

    res_amplitude = spectrum - fit_amplitude

    return res_amplitude
