"""Wavelength calibration task."""
import astropy.units as u
import numpy as np
from astropy import constants as const
from astropy.wcs import WCS
from dkist_service_configuration.logging import logger
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import differential_evolution
from scipy.optimize import OptimizeResult

from solar_wavelength_calibration.atlas.atlas import Atlas
from solar_wavelength_calibration.atlas.base import AtlasBase

record_provenence = True


def calibrate_wavelength(
    spectrum: np.ndarray,
    expected_wavelength_vector: u.Quantity,
    dispersion: u.Quantity,
    alpha: float,
    doppler_shift: u.Quantity,
    order: int,
    grating_constant: float,
    atlas: AtlasBase | None = None,
):
    """
    Wavelength calibration task.

    Parameters
    ----------
    spectrum : np.ndarray
        The input spectrum to be calibrated.
    expected_wavelength_vector : u.Quantity
        The expected wavelength vector.
    dispersion : u.Quantity
        The dispersion of the grating.
    alpha : float
        The angle of incidence of the grating.
    doppler_shift : u.Quantity
        The doppler shift of the spectrum (from .
    order : int
        The order of the spectrum.
    grating_constant : float
        The grating constant.
    atlas : Atlas
        The FTS atlas to use for wavelength calibration.

    Returns
    -------
    fit_result
    """
    # Load atlas
    atlas = atlas or Atlas()

    # Resample atlas to be on the same, linear wavelength grid, select the portion of atlas that pertains to bandpass used
    (
        cropped_telluric_atlas_wave,
        cropped_telluric_atlas_trans,
        cropped_solar_atlas_wave_air,
        cropped_solar_atlas_trans_flipped,
    ) = resample_fts_atlases(
        atlas,
        expected_wavelength_vector,
    )

    # Determine a shift of the preliminary wavelength vector so that it generally aligns with the data itself.
    (fts_wave, fts_solar, fts_telluric) = initial_alignment(
        spectrum,
        expected_wavelength_vector,
        cropped_solar_atlas_wave_air,
        cropped_solar_atlas_trans_flipped,
        cropped_telluric_atlas_wave,
        cropped_telluric_atlas_trans,
    )

    # Define the bounds and send the fitting model on its way.
    fit_result = fit_dispersion_axis_to_FTS(
        fts_wave,
        fts_telluric,
        fts_solar,
        dispersion,
        alpha,
        doppler_shift,
        spectrum,
        order,
        grating_constant,
    )

    return fit_result


def resample_fts_atlases(
    atlas: AtlasBase,
    expected_wavelength_vector: u.Quantity,
) -> tuple[u.Quantity, np.ndarray, u.Quantity, np.ndarray]:
    """Resample both atlases to be on the same, linear wavelength grid and select the portion of atlas that pertains to bandpass used."""
    solar_atlas_wavelength = atlas.solar_atlas_wavelength * u.nm
    telluric_atlas_wavelength = atlas.telluric_atlas_wavelength * u.nm

    expected_wavelength_range = expected_wavelength_vector.max() - expected_wavelength_vector.min()
    min_wavelength = expected_wavelength_vector.min() - 0.25 * expected_wavelength_range
    max_wavelength = expected_wavelength_vector.max() + 0.25 * expected_wavelength_range

    cropped_telluric_mask = (telluric_atlas_wavelength > min_wavelength) * (
        telluric_atlas_wavelength < max_wavelength
    )
    cropped_telluric_atlas_wavelength = telluric_atlas_wavelength[cropped_telluric_mask]
    cropped_telluric_atlas_transmission = atlas.telluric_atlas_transmission[cropped_telluric_mask]

    cropped_solar_mask = (solar_atlas_wavelength > min_wavelength) * (
        solar_atlas_wavelength < max_wavelength
    )
    cropped_solar_atlas_wavelength = solar_atlas_wavelength[cropped_solar_mask]
    cropped_solar_atlas_transmission = atlas.solar_atlas_transmission[cropped_solar_mask]

    return (
        cropped_telluric_atlas_wavelength,
        cropped_telluric_atlas_transmission,
        cropped_solar_atlas_wavelength,
        cropped_solar_atlas_transmission,
    )


def initial_alignment(
    spectrum: np.ndarray,
    expected_wavelength_vector: u.Quantity,
    cropped_solar_atlas_wave_air: u.Quantity,
    cropped_solar_atlas_trans_flipped: np.ndarray,
    cropped_telluric_atlas_wave: u.Quantity,
    cropped_telluric_atlas_trans: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Determine a shift of the preliminary wavelength vector so that it generally aligns with the data itself."""
    shifts = np.linspace(-2, 2, 550) * u.nm
    merit = np.zeros(len(shifts))
    for n, shift in enumerate(shifts):
        preliminary_wavelength = expected_wavelength_vector + shift
        fts_solar = np.interp(
            preliminary_wavelength, cropped_solar_atlas_wave_air, cropped_solar_atlas_trans_flipped
        )
        fts_telluric = np.interp(
            preliminary_wavelength, cropped_telluric_atlas_wave, cropped_telluric_atlas_trans
        )
        # calculate a merit value to be minimized
        merit[n] = np.std(spectrum - fts_solar * fts_telluric)

    # get minimum
    shift = shifts[np.argmin(merit)]

    # recalculate spectral axis and atlas spectrum for the best shift value
    fts_wave = expected_wavelength_vector + shift
    fts_solar = np.interp(fts_wave, cropped_solar_atlas_wave_air, cropped_solar_atlas_trans_flipped)
    fts_telluric = np.interp(fts_wave, cropped_telluric_atlas_wave, cropped_telluric_atlas_trans)

    return fts_wave.value, fts_solar, fts_telluric


def fitness(
    parameters: np.ndarray,
    spectrum: np.ndarray,
    fts_wave: np.ndarray,
    fts_telluric: np.ndarray,
    fts_solar: np.ndarray,
    order: int,
    grating_constant: float,
    doppler_shift: float,
) -> float:
    """
    Model function for profile fitting.

    Parameters
    ----------
    crval1
        Wavelength at crpix1

    cdelt1
        Spectral dispersion at crpix1

    incident_light_angle
        Incident angle in degrees

    resolving_power
        Resolving power -- used to estimate the line spread function (may not be correct off limb)

    opacity
        Opacity scaling applied to telluric absorption

    stray_light_frac
        Inferred straylight fraction in the spectrograph --> This scales the lines non-linearly.

    continuum_amplitude
        Amplitude of the scattered light continuum
    """
    (
        crval1,
        cdelt1,
        incident_light_angle,
        resolving_power,
        opacity,
        stray_light_frac,
        continuum_amplitude,
    ) = parameters

    # calculate the spectral axis
    # Representations of spectral coordinates in FITS
    # https://ui.adsabs.harvard.edu/abs/2006A%26A...446..747G/abstract
    # https://specreduce.readthedocs.io/en/latest/api/specreduce.utils.synth_data.make_2d_arc_image.html

    number_of_wave_pix = np.size(spectrum)

    non_linear_header = {
        "CTYPE1": "AWAV-GRA",  # Grating dispersion function with air wavelengths
        "CUNIT1": "nm",  # Dispersion units
        "CRPIX1": number_of_wave_pix // 2 + 1,  # Reference pixel [pix]
        "PV1_0": grating_constant,  # Grating density
        "PV1_1": order,  # Diffraction order
        "CRVAL1": crval1,  # Reference value [nm] (<<<< TO BE OPTMIZED <<<<<<)
        "CDELT1": cdelt1,  # Linear dispersion [nm/pix] (<<<< TO BE OPTMIZED <<<<<<)
        "PV1_2": incident_light_angle,  # Incident angle [deg] (<<<< TO BE OPTMIZED <<<<<<)
    }

    non_linear_wcs = WCS(non_linear_header)
    wavelength_vector = (
        (non_linear_wcs.spectral.pixel_to_world(np.arange(number_of_wave_pix))).to(u.nm).value
    )

    # Gaussian convolution of the FTS atlas
    fwhm_wavelength = np.divide(crval1, resolving_power)
    sigma_wavelength = fwhm_wavelength / (2.0 * np.sqrt(2.0 * np.log(2)))
    kern_pix = sigma_wavelength / np.abs(cdelt1)

    # interpolate the telluric spectral atlas onto the new wavelength axis and scale by the opacity value that is being optimized
    fts_atmosphere_interp = np.interp(wavelength_vector, fts_wave, fts_telluric)
    fts_telluric_interp = np.exp(opacity * np.log(fts_atmosphere_interp))

    # interpolate the solar spectral atlas onto the new wavelength axis and apply a shift according to the Doppler shift due to orbital motions
    fts_solar_interp = np.interp(
        wavelength_vector,
        fts_wave + doppler_shift / (const.c.to("km/s")).value * crval1,
        fts_solar,
    )

    # apply telluric absorption spectrum to solar spectrum
    fts_modulated = fts_telluric_interp * fts_solar_interp
    # add flat value of straylight contamination
    fts_modulated_with_straylight = (fts_modulated + stray_light_frac) / (1.0 + stray_light_frac)
    # scale for total intensity of the continuum
    fit_amplitude = fts_modulated_with_straylight * continuum_amplitude

    # convolution for spectrograph line spread function
    fit_amplitude = gaussian_filter1d(fit_amplitude, kern_pix)

    # chisquare calculation for fit metric
    res_amplitude = np.sum((spectrum - fit_amplitude) ** 2)

    return res_amplitude


def fit_dispersion_axis_to_FTS(
    fts_wave: np.ndarray,
    fts_telluric: np.ndarray,
    fts_solar: np.ndarray,
    dispersion: u.Quantity,
    alpha: float,
    doppler_shift: u.Quantity,
    spectrum: np.ndarray,
    order: int,
    grating_constant: float,
) -> OptimizeResult:
    """Define the bounds and send the fitting model on its way."""
    parameter_names = (
        "crval1 (wavelength at crpix1)",
        "cdelt1 (spectral dispersion at crpix1)",
        "incident_light_angle",
        "resolving_power",
        "opacity",
        "stray_light_frac",
        "continuum_amplitude",
    )
    crpix1_updated = np.size(spectrum) // 2 + 1
    crval1 = fts_wave[crpix1_updated]  # initial guess
    bounds = [
        # [nm[ +\- 0.5 nm range used for finding CRVAL1
        (crval1 - 0.5, crval1 + 0.5),
        # [nm/pix] 5% bounds on the dispersion at CRPIX1
        (
            dispersion.value - 0.05 * dispersion.value,
            dispersion.value + 0.05 * dispersion.value,
        ),
        # [radian] Incident angle range is +/- 5 degree from value in header
        (np.rad2deg(alpha) - 5, np.rad2deg(alpha) + 5),
        # resolving power range
        (20000, 125000),
        # opacity factor bounds
        (0.0, 10),
        # straylight fraction
        (0.0, 0.5),
        # continuum intensity correction
        (0.8 * np.nanpercentile(spectrum, 75), 1.2 * np.nanpercentile(spectrum, 75)),
    ]

    for repeat_fit in range(5):  # repeat just in case the fitting gets stuck in a local minimum
        fit_result = differential_evolution(
            fitness,
            args=(
                spectrum,
                fts_wave,
                fts_telluric,
                fts_solar,
                order,
                grating_constant,
                doppler_shift.value,
            ),
            popsize=2,
            maxiter=300,
            bounds=bounds,
            disp=True,
            polish=True,
            tol=1.0e-9,
        )
        if fit_result.fun < 0.03:
            logger.info(" Convergence good based on fit func value")
            break

    logger.info("Fitted Values:")
    logger.info(" ")
    for p in range(len(parameter_names)):
        logger.info(
            f"Parameter: {parameter_names[p]},   Fit Result: {fit_result.x[p]},    Bounds: {bounds[p]}"
        )

    return fit_result
