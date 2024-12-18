"""Wavelength calibration task."""
import numpy as np
from astropy import constants as const
from astropy.units import Quantity as u
from astropy.wcs import WCS
from dkist_service_configuration.logging import logger
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import differential_evolution
from scipy.optimize import OptimizeResult


class WaveCal:
    """Task class for correcting the dispersion axis wavelength values."""

    def __init__(
        self,
        spectrum: np.ndarray,
        expected_wavelength_vector: u.Quantity,
        dispersion: u.Quantity,
        alpha: float,
        doppler_shift: u.Quantity,
        order: int,
        grating_constant: float,
    ):
        self.spectrum = spectrum
        self.expected_wavelength_vector = expected_wavelength_vector
        self.dispersion = dispersion
        self.alpha = alpha
        self.doppler_shift = doppler_shift
        self.order = order
        self.grating_constant = grating_constant

    record_provenence = True

    def run(self):
        """
        Run method for the task.

        Returns
        -------
        None
        """
        # Load telluric and non-telluric FTS atlas data, resample both atlases to be on the same, linear wavelength grid, select the portion of atlas that pertains to bandpass used
        (
            telluric_atlas_wave,
            telluric_atlas_trans,
            solar_atlas_wave_air,
            solar_atlas_trans_flipped,
        ) = self.load_and_resample_fts_atlas(self.expected_wavelength_vector)

        # Determine a shift of the preliminary wavelength vector so that it generally aligns with the data itself.
        (fts_wave, fts_solar, fts_telluric) = self.initial_alignment(
            self.spectrum,
            self.expected_wavelength_vector,
            solar_atlas_wave_air,
            solar_atlas_trans_flipped,
            telluric_atlas_wave,
            telluric_atlas_trans,
        )

        # Define the bounds and send the fitting model on its way.
        fit_result = self.fit_dispersion_axis_to_FTS(
            fts_wave,
            fts_telluric,
            fts_solar,
            self.dispersion,
            self.alpha,
            self.doppler_shift,
            self.spectrum,
            self.order,
            self.grating_constant,
        )

        # Save the fit results to disk to later be used to update the l1 headers."
        # self.write_fit_results(spectrum, order, fit_result)

    def load_and_resample_fts_atlas(
        self, expected_wavelength_vector: u.Quantity
    ) -> tuple[u.Quantity, np.ndarray, u.Quantity, np.ndarray]:
        """Load telluric and non-telluric FTS atlas data, resample both atlases to be on the same, linear wavelength grid, select the portion of atlas that pertains to bandpass used."""
        solar_atlas_wavelength, solar_atlas_transmission = self.parameters.solar_atlas
        solar_atlas_wavelength = solar_atlas_wavelength * u.nm
        telluric_atlas_wavelength, telluric_atlas_transmission = self.parameters.telluric_atlas
        telluric_atlas_wavelength = telluric_atlas_wavelength * u.nm

        expected_wavelength_range = (
            expected_wavelength_vector.max() - expected_wavelength_vector.min()
        )
        min_wavelength = expected_wavelength_vector.min() - 0.25 * expected_wavelength_range
        max_wavelength = expected_wavelength_vector.max() + 0.25 * expected_wavelength_range

        cropped_telluric_mask = (telluric_atlas_wavelength > min_wavelength) * (
            telluric_atlas_wavelength < max_wavelength
        )
        telluric_atlas_wavelength = telluric_atlas_wavelength[cropped_telluric_mask]
        telluric_atlas_transmission = telluric_atlas_transmission[cropped_telluric_mask]

        cropped_solar_mask = (solar_atlas_wavelength > min_wavelength) * (
            solar_atlas_wavelength < max_wavelength
        )
        solar_atlas_wavelength = solar_atlas_wavelength[cropped_solar_mask]
        solar_atlas_transmission = solar_atlas_transmission[cropped_solar_mask]

        return (
            telluric_atlas_wavelength,
            telluric_atlas_transmission,
            solar_atlas_wavelength,
            solar_atlas_transmission,
        )

    @staticmethod
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
        fts_modulated_with_straylight = (fts_modulated + stray_light_frac) / (
            1.0 + stray_light_frac
        )
        # scale for total intensity of the continuum
        fit_amplitude = fts_modulated_with_straylight * continuum_amplitude

        # convolution for spectrograph line spread function
        fit_amplitude = gaussian_filter1d(fit_amplitude, kern_pix)

        # chisquare calculation for fit metric
        res_amplitude = np.sum((spectrum - fit_amplitude) ** 2)

        return res_amplitude

    def fit_dispersion_axis_to_FTS(
        self,
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
                WaveCal.fitness,
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

        fit_amplitude = WaveCal.fitness(
            fit_result.x,
            spectrum,
            fts_wave,
            fts_telluric,
            fts_solar,
            order,
            grating_constant,
            doppler_shift.value,
        )

        return fit_result
