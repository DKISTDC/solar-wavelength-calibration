"""Wavelength calibration task."""


class WaveCal:
    """Task class for correcting the dispersion axis wavelength values."""

    record_provenence = True

    def run(self):
        """
        Run method for the task.

        Returns
        -------
        None
        """
        # Load intermediate 1d characteristic measured_spectra for beam 1."
        # spectrum = self.load_1d_char_spectrum()

        # Grab a header from a random solar gain frame to be used to find a rough initial wavelength estimate.
        # solar_header = self.load_header()

        # Compute theoretical dispersion value using the grating equation.
        # dispersion, order, alpha = self.get_theoretical_dispersion()

        # Compute the expected wavelength vector based on the header information.
        # expected_wavelength_vector = self.compute_expected_wavelength_vector(spectrum, solar_header)

        # Load telluric and non-telluric FTS atlas data, resample both atlases to be on the same, linear wavelength grid, select the portion of atlas that pertains to bandpass used
        """(
            telluric_atlas_wave,
            telluric_atlas_trans,
            solar_atlas_wave_air,
            solar_atlas_trans_flipped,
        ) = self.load_and_resample_fts_atlas(expected_wavelength_vector)"""

        # Determine a shift of the preliminary wavelength vector so that it generally aligns with the data itself.
        """(fts_wave, fts_solar, fts_telluric) = self.initial_alignment(
            spectrum,
            expected_wavelength_vector,
            solar_atlas_wave_air,
            solar_atlas_trans_flipped,
            telluric_atlas_wave,
            telluric_atlas_trans,
        )"""

        # Find the speed at which DKIST is moving relative to the Sun's center.
        # doppler_shift = self.get_doppler_shift()

        # Define the bounds and send the fitting model on its way.
        """fit_result = self.fit_dispersion_axis_to_FTS(
            fts_wave,
            fts_telluric,
            fts_solar,
            dispersion,
            alpha,
            doppler_shift,
            spectrum,
            order,
            self.constants.grating_constant,
        )"""

        # Save the fit results to disk to later be used to update the l1 headers."
        # self.write_fit_results(spectrum, order, fit_result)
