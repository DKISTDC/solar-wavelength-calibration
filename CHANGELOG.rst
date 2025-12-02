v2.0.0 (2025-12-02)
===================

Features
--------

- Add the input data, prepared weights, and best-fit atlas as properties on `FitResult` to help user with post-fit analysis. (`#19 <https://bitbucket.org/dkistdc/solar-wavelength-calibration/pull-requests/19>`__)
- Add the ability to fit a user-parameterizable continuum function as part of the solar atlas fit.
  See :ref:`this page <custom_continuum_function>` for more information. (`#20 <https://bitbucket.org/dkistdc/solar-wavelength-calibration/pull-requests/20>`__)
- Add `best_fit_wavelength_vector` and `best_fit_continuum` properties to `FitResult`. (`#20 <https://bitbucket.org/dkistdc/solar-wavelength-calibration/pull-requests/20>`__)
- Change the physical meaning of the ``straylight_fraction`` fit parameter. It is now applied as
  `new_atlas = (old_atlas + straylight_fraction) / (1 + straylight_fraction)`. This renormalization helps decouple the effects
  of ``straylight_fraction`` from ``continuum_level`` or any custom continuum function. (`#20 <https://bitbucket.org/dkistdc/solar-wavelength-calibration/pull-requests/20>`__)
- Remove the `input_wavelength_vector` argument from the call to an instantiated `WavelengthCalibrationFitter` object.
  We now compute this vector from internally from the initial parameters used to set up `WavelengthCalibrationFitter` objects. (`#23 <https://bitbucket.org/dkistdc/solar-wavelength-calibration/pull-requests/23>`__)
- Add `normalization_percentile` kwarg to `calculate_initial_crval_guess` function. This allows a good guess when the atlas
  and input spectra have vastly different normalizations. (`#24 <https://bitbucket.org/dkistdc/solar-wavelength-calibration/pull-requests/24>`__)


Misc
----

- Update pre-commit hook versions and replace python-reorder-imports with isort. (`#18 <https://bitbucket.org/dkistdc/solar-wavelength-calibration/pull-requests/18>`__)
- Add tox tests with python 3.12 to BitBucket pipelines (`#21 <https://bitbucket.org/dkistdc/solar-wavelength-calibration/pull-requests/21>`__)
- Bump `pydantic` minimum version to v2.7.2 to avoid build error on python >= 3.12.
  See https://github.com/pydantic/pydantic-core/issues/1366 and https://github.com/pyupio/safety/issues/620 for more info. (`#21 <https://bitbucket.org/dkistdc/solar-wavelength-calibration/pull-requests/21>`__)
- Remove usage of `__all__` in modules. Instead, just import the given objects directly. This shouldn't change the API at all;
  it's required by Sphinx auto-api. (`#22 <https://bitbucket.org/dkistdc/solar-wavelength-calibration/pull-requests/22>`__)
- `BoundsModel`, `FitFlagsModel`, and `WavelengthCalibrationParameters` models no longer allow extra fields to be passed during instantiation.
  Any extra arguments should be defined explicitly on a subclass. (`#25 <https://bitbucket.org/dkistdc/solar-wavelength-calibration/pull-requests/25>`__)


Documentation
-------------

- Add online documentation pages. (`#22 <https://bitbucket.org/dkistdc/solar-wavelength-calibration/pull-requests/22>`__)


v1.0.1 (2025-07-28)
===================

Features
--------

- Enable retries in the download of atlas files in the event that the download fails. (`#17 <https://bitbucket.org/dkistdc/solar-wavelength-calibration/pull-requests/17>`__)


v1.0 (2025-06-23)
=================

Features
--------

- Added support for spectral weights in fitting, including residual scaling and corresponding tests. (`#14 <https://bitbucket.org/dkistdc/solar-wavelength-calibration/pull-requests/14>`__)
- Introduced `calculate_initial_crval_guess` helper function to estimate an initial CRVAL value by aligning the observed spectrum with solar and telluric atlas data. (`#15 <https://bitbucket.org/dkistdc/solar-wavelength-calibration/pull-requests/15>`__)
- Add support for alternate header keys in WavelengthParameters.to_header method. (`#16 <https://bitbucket.org/dkistdc/solar-wavelength-calibration/pull-requests/16>`__)


v0.1 (2025-05-27)
=================

No significant changes.


v0.1rc1 (2025-05-22)
====================

Features
--------

- Wavelength axis correction done by fitting the corresponding section of the FTS Atlas using a least-squares optimization. (`#3 <https://bitbucket.org/dkistdc/solar-wavelength-calibration/pull-requests/3>`__)


Misc
----

- Initial package setup. (`#1 <https://bitbucket.org/dkistdc/solar-wavelength-calibration/pull-requests/1>`__)
- Test minimum dependencies in addition to latest. (`#2 <https://bitbucket.org/dkistdc/solar-wavelength-calibration/pull-requests/2>`__)
- Adding framework to retrieve FTS Atlases from Zenodo using the `Pooch package <https://www.fatiando.org/pooch/latest/index.html>`_. Unless otherwise specified, the default retrieval location is https://zenodo.org/records/14674504. (`#5 <https://bitbucket.org/dkistdc/solar-wavelength-calibration/pull-requests/5>`__)
- Update Bitbucket pipelines to use execute script for standard steps. (`#7 <https://bitbucket.org/dkistdc/solar-wavelength-calibration/pull-requests/7>`__)
- Add code coverage badge to README.rst. (`#8 <https://bitbucket.org/dkistdc/solar-wavelength-calibration/pull-requests/8>`__)
