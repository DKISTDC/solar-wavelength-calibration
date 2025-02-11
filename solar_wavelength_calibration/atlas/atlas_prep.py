"""Machinery to prepare the FTS Atlas for the fit by resampling and aligning to the input data."""
import astropy.units as u
import numpy as np

from solar_wavelength_calibration.atlas.base import AtlasBase


class PrepareAtlas:
    """Object that prepares atlas before fitting."""

    def __init__(
        self,
        atlas: AtlasBase,
        expected_wavelength_vector: u.Quantity,
        unfit_1d_spectrum: np.ndarray,
    ):
        self.telluric_atlas = atlas.telluric_atlas_transmission  # fts_telluric
        self.solar_atlas = atlas.solar_atlas_transmission  # fts_solar
        self.solar_atlas_wavelength = atlas.solar_atlas_wavelength
        self.telluric_atlas_wavelength = atlas.telluric_atlas_wavelength  # fts_wave
        self.expected_wavelength_vector = expected_wavelength_vector
        self.unfit_1d_spectrum = unfit_1d_spectrum

        (
            cropped_telluric_atlas_wavelength,
            cropped_telluric_atlas_transmission,
            cropped_solar_atlas_wavelength,
            cropped_solar_atlas_transmission,
        ) = self.resample_atlas()

        self.initial_alignment(
            cropped_telluric_atlas_wavelength,
            cropped_telluric_atlas_transmission,
            cropped_solar_atlas_wavelength,
            cropped_solar_atlas_transmission,
        )

    def resample_atlas(self):
        """Resample both atlases to be on the same, linear wavelength grid and select the portion of atlas that pertains to bandpass used."""
        solar_atlas_wavelength = self.solar_atlas_wavelength * u.nm
        telluric_atlas_wavelength = self.telluric_atlas_wavelength * u.nm

        expected_wavelength_range = (
            self.expected_wavelength_vector.max() - self.expected_wavelength_vector.min()
        )
        min_wavelength = self.expected_wavelength_vector.min() - 0.25 * expected_wavelength_range
        max_wavelength = self.expected_wavelength_vector.max() + 0.25 * expected_wavelength_range

        cropped_telluric_mask = (telluric_atlas_wavelength > min_wavelength) * (
            telluric_atlas_wavelength < max_wavelength
        )
        cropped_telluric_atlas_wavelength = telluric_atlas_wavelength[cropped_telluric_mask]
        cropped_telluric_atlas_transmission = self.telluric_atlas[cropped_telluric_mask]

        cropped_solar_mask = (solar_atlas_wavelength > min_wavelength) * (
            solar_atlas_wavelength < max_wavelength
        )
        cropped_solar_atlas_wavelength = solar_atlas_wavelength[cropped_solar_mask]
        cropped_solar_atlas_transmission = self.solar_atlas[cropped_solar_mask]

        return (
            cropped_telluric_atlas_wavelength,
            cropped_telluric_atlas_transmission,
            cropped_solar_atlas_wavelength,
            cropped_solar_atlas_transmission,
        )

    def initial_alignment(
        self,
        cropped_telluric_atlas_wavelength,
        cropped_telluric_atlas_transmission,
        cropped_solar_atlas_wavelength,
        cropped_solar_atlas_transmission,
    ):
        """Determine a shift of the preliminary wavelength vector so that it generally aligns with the data itself."""
        shifts = np.linspace(-2, 2, 550) * u.nm
        merit = np.zeros(len(shifts))
        for n, shift in enumerate(shifts):
            preliminary_wavelength = self.expected_wavelength_vector + shift
            fts_solar = np.interp(
                preliminary_wavelength,
                cropped_solar_atlas_wavelength,
                cropped_solar_atlas_transmission,
            )
            fts_telluric = np.interp(
                preliminary_wavelength,
                cropped_telluric_atlas_wavelength,
                cropped_telluric_atlas_transmission,
            )
            # calculate a merit value to be minimized
            merit[n] = np.std(self.unfit_1d_spectrum - fts_solar * fts_telluric)

        # get minimum
        shift = shifts[np.argmin(merit)]

        # recalculate spectral axis and atlas spectrum for the best shift value
        fts_wave = self.expected_wavelength_vector + shift
        fts_solar = np.interp(
            fts_wave, cropped_solar_atlas_wavelength, cropped_solar_atlas_transmission
        )
        fts_telluric = np.interp(
            fts_wave, cropped_telluric_atlas_wavelength, cropped_telluric_atlas_transmission
        )

        return fts_wave.value, fts_solar, fts_telluric
