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
