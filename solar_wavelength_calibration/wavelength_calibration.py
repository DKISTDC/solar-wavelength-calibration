"""Wavelength calibration task."""
import os

import astropy.units as u
import numpy as np
import pooch

record_provenence = True


def load_fts_atlases():
    """Load telluric and non-telluric FTS atlas data."""
    # Define a Pooch object
    my_pooch = pooch.create(
        path=pooch.os_cache("pooch"),
        base_url="doi:10.5281/zenodo.14646787/",
        registry={
            "telluric_reference_atlas.npy": "md5:8db5e12508b293bca3495d81a0747447",
            "solar_reference_atlas.npy": "md5:84ab4c50689ef235fe5ed4f7ee905ca0",
        },
        env="FTS_ATLAS_DATA_DIR" if os.environ.get("FTS_ATLAS_DATA_DIR") is not None else None,
    )

    # Fetch the files using Pooch
    telluric_atlas = my_pooch.fetch("telluric_reference_atlas.npy")
    solar_atlas = my_pooch.fetch("solar_reference_atlas.npy")

    # Load the data
    telluric_atlas_wavelength, telluric_atlas_transmission = np.load(telluric_atlas)
    solar_atlas_wavelength, solar_atlas_transmission = np.load(solar_atlas)

    return (
        telluric_atlas_wavelength,
        telluric_atlas_transmission,
        solar_atlas_wavelength,
        solar_atlas_transmission,
    )
