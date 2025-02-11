import astropy.units as u
import numpy as np
import pytest

from solar_wavelength_calibration.atlas.atlas import Atlas
from solar_wavelength_calibration.atlas.atlas import DownloadConfig
from solar_wavelength_calibration.atlas.base import AtlasBase
from solar_wavelength_calibration.fitter.wavelength_calibration import calibrate_wavelength
from solar_wavelength_calibration.fitter.wavelength_calibration import initial_alignment
from solar_wavelength_calibration.fitter.wavelength_calibration import resample_fts_atlases


@pytest.fixture(
    params=[
        pytest.param(None, id="default_config"),
    ]
)
def config(request) -> DownloadConfig | None:
    return request.param


@pytest.fixture()
def atlas(config: DownloadConfig | None) -> Atlas:
    return Atlas(config=config)


def test_resample_atlas_cropped(atlas: AtlasBase):
    """Given: An Atlas object
    When: The object is being resampled
    Then: The output object should be smaller than the input object.
    """
    expected_wavelength_vector = np.linspace(1500, 2000, 1000) * u.nm
    (
        cropped_telluric_atlas_wavelength,
        cropped_telluric_atlas_transmission,
        cropped_solar_atlas_wavelength,
        cropped_solar_atlas_transmission,
    ) = resample_fts_atlases(atlas, expected_wavelength_vector)

    # make sure that atlases were cropped
    assert cropped_solar_atlas_wavelength.shape < atlas.solar_atlas_wavelength.shape
    assert cropped_solar_atlas_transmission.shape < atlas.solar_atlas_transmission.shape
    assert cropped_telluric_atlas_wavelength.shape < atlas.telluric_atlas_wavelength.shape
    assert cropped_telluric_atlas_transmission.shape < atlas.telluric_atlas_transmission.shape


def test_resample_atlases_cover_expected_wavelength(atlas: AtlasBase):
    """Given: An Atlas object
    When: The object is being resampled
    Then: The atlases cover the range of the expected wavelength vector plus some
    """
    expected_wavelength_vector = np.linspace(1500, 2000, 1000) * u.nm
    (
        cropped_telluric_atlas_wavelength,
        cropped_telluric_atlas_transmission,
        cropped_solar_atlas_wavelength,
        cropped_solar_atlas_transmission,
    ) = resample_fts_atlases(atlas, expected_wavelength_vector)

    assert cropped_solar_atlas_wavelength.min().value < expected_wavelength_vector.min().value
    assert cropped_solar_atlas_wavelength.max().value > expected_wavelength_vector.max().value
    assert cropped_telluric_atlas_wavelength.min().value < expected_wavelength_vector.min().value
    assert cropped_telluric_atlas_wavelength.max().value > expected_wavelength_vector.max().value


def test_initial_alignment(atlas: AtlasBase):
    """Given:
    When: Aligning the atlas with the input data
    Then: The output vector has been shifted
    """
    spectrum = np.linspace(0, 1, 1000)
    expected_wavelength_vector = np.linspace(1500, 2000, 1000) * u.nm
    cropped_solar_atlas_wave_air = atlas.solar_atlas_wavelength[1000:1500] * u.nm
    cropped_solar_atlas_trans_flipped = atlas.solar_atlas_transmission[1000:1500]
    cropped_telluric_atlas_wave = atlas.telluric_atlas_wavelength[1000:1500] * u.nm
    cropped_telluric_atlas_trans = atlas.telluric_atlas_transmission[1000:1500]

    fts_wave, fts_solar, fts_telluric = initial_alignment(
        spectrum,
        expected_wavelength_vector,
        cropped_solar_atlas_wave_air,
        cropped_solar_atlas_trans_flipped,
        cropped_telluric_atlas_wave,
        cropped_telluric_atlas_trans,
    )
    # make sure that spectra shifted..
    assert fts_wave.min() != expected_wavelength_vector.min().value
    # but is still close to input vector
    assert np.isclose(fts_wave.min(), expected_wavelength_vector.min().value, rtol=5.0)
    assert np.isclose(fts_wave.max(), expected_wavelength_vector.max().value, rtol=5.0)


@pytest.fixture
def example_inputs():
    """Fixture for example inputs to the calibrate_wavelength function."""
    spectrum = np.linspace(0, 1, 1000)
    expected_wavelength_vector = np.linspace(1500, 2000, 1000) * u.nm
    dispersion = 0.1 * u.nm / u.pix
    alpha = 30.0  # in degrees
    doppler_shift = 0.02 * u.nm
    order = 1
    grating_constant = 1200.0  # grating lines/mm
    return (
        spectrum,
        expected_wavelength_vector,
        dispersion,
        alpha,
        doppler_shift,
        order,
        grating_constant,
    )


def test_calibrate_wavelength(example_inputs, atlas):
    """Given: Example inputs and an atlas
    When: Calibrating wavelength axis
    Then: Fit values should be close to calculated values
    """
    (
        spectrum,
        expected_wavelength_vector,
        dispersion,
        alpha,
        doppler_shift,
        order,
        grating_constant,
    ) = example_inputs

    fit_result = calibrate_wavelength(
        spectrum=spectrum,
        expected_wavelength_vector=expected_wavelength_vector,
        dispersion=dispersion,
        alpha=alpha,
        doppler_shift=doppler_shift,
        order=order,
        grating_constant=grating_constant,
        atlas=atlas,
    )
    assert np.isclose(fit_result.x[1], dispersion.value, rtol=0.1)


def test_incorrect_inputs_to_calibrate_wavelength(example_inputs):
    """Given: A very small input spectra
    When: Calibrating wavelength axis
    Then: Task should fail
    """
    """Test calibrate_wavelength with invalid inputs."""
    (
        spectrum,
        expected_wavelength_vector,
        dispersion,
        alpha,
        doppler_shift,
        order,
        grating_constant,
    ) = example_inputs

    with pytest.raises(ValueError):
        calibrate_wavelength(
            spectrum=np.random.random(3),
            expected_wavelength_vector=expected_wavelength_vector,
            dispersion=dispersion,
            alpha=alpha,
            doppler_shift=doppler_shift,
            order=order,
            grating_constant=grating_constant,
        )
