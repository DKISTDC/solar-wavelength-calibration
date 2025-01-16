import numpy as np

from solar_wavelength_calibration.wavelength_calibration import load_fts_atlases


def test_load_atlas():
    """Test that passes."""
    (
        telluric_atlas_wavelength,
        telluric_atlas_transmission,
        solar_atlas_wavelength,
        solar_atlas_transmission,
    ) = load_fts_atlases()

    assert isinstance(solar_atlas_wavelength, np.ndarray)
    assert isinstance(solar_atlas_transmission, np.ndarray)
    assert isinstance(telluric_atlas_wavelength, np.ndarray)
    assert isinstance(telluric_atlas_transmission, np.ndarray)
