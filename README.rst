solar-wavelength-calibration
----------------------------
|codecov|

A package for calibrating wavelength data.

Deployment
----------
solar-wavelength-calibration is deployed to `PyPI <https://pypi.org/project/solar-wavelength-calibration/>`_

Development
-----------
.. code-block:: bash

    git clone git@bitbucket.org:dkistdc/solar-wavelength-calibration.git
    cd solar-wavelength-calibration
    pre-commit install
    pip install -e .[test]
    pytest -v --cov solar-wavelength-calibration

Installation
------------
.. code:: bash

   pip install solar-wavelength-calibration

Usage
-----
.. code:: python

    >>> from solar_wavelength_calibration.atlas.atlas import Atlas, DownloadConfig
    >>> from solar_wavelength_calibration.fitter.parameters import AngleBoundRange, BoundsModel, DispersionBoundRange, LengthBoundRange, UnitlessBoundRange, WavelengthCalibrationParameters
    >>> from solar_wavelength_calibration.fitter.wavelength_fitter import WaveCalFitter

Load in the Telluric FTS Atlas and the Disk Center (Solar) FTS Atlas:

.. code:: python

    >>> atlas_config = DownloadConfig(
            base_url="doi:10.5281/zenodo.14646788/",
            telluric_reference_atlas_file_name="telluric_reference_atlas.npy",
            telluric_reference_atlas_hash_id="md5:8db5e12508b293bca3495d81a0747447",
            solar_reference_atlas_file_name="solar_reference_atlas.npy",
            solar_reference_atlas_hash_id="md5:84ab4c50689ef235fe5ed4f7ee905ca0",
        )

    >>> atlas = Atlas(atlas_config)

If desired Setup fitting bounds. If no bounds are specified, default values will be used:

.. code:: python

    >>> bounds = BoundsModel(
            crval1=LengthBoundRange(min=349.858 * u.nm, max=350.458 * u.nm),
            dispersion=DispersionBoundRange(
                min=dispersion - (0.05 * u.nm / u.pix), max=dispersion + (0.05 * u.nm / u.pix)
            ),
            incident_light_angle=AngleBoundRange(
                min=incident_light_angle - (180 * u.deg), max=incident_light_angle + (180 * u.deg)
            ),
            resolving_power=UnitlessBoundRange(
                min=resolving_power + resolving_power_bounds[0],
                max=resolving_power + resolving_power_bounds[1],
            ),
            opacity_factor=UnitlessBoundRange(min=0.0, max=10.0),
            straylight_fraction=UnitlessBoundRange(min=0.0, max=0.4),
        )

Initialize model parameters:

.. code:: python

    >>> input_parameters = WavelengthCalibrationParameters(
            crval1=atlas.telluric_atlas_wavelength[np.size(unfit_1d_spectrum) // 2 + 1],
            dispersion=dispersion,
            incident_light_angle=incident_light_angle,
            resolving_power=resolving_power,
            opacity_factor=5.0,
            straylight_fraction=0.2,
            grating_constant=self.constants.grating_constant,
            doppler_velocity=doppler_velocity,
            order=order,
            bounds=bounds,
        )

Calibrate the wavelength axis

.. code:: python

    >>> fitter = WaveCalFitter(
            input_parameters=input_parameters,
            atlas=atlas,
        )

Run the fit. If full flag is set to True, the fitter will return

.. code:: python

    >>> fit_result = fitter.run_fit(
            expected_wavelength_vector=expected_wavelength_vector,
            unfit_1d_spectrum=unfit_1d_spectrum,
        )

The full flag can also be set to True in order to return the fit result, as well as all other fit result parameters returned by the fitting algorithm.

.. code:: python

    >>> fit_result, fit_result_params = fitter.run_fit(
            expected_wavelength_vector=expected_wavelength_vector,
            unfit_1d_spectrum=unfit_1d_spectrum,
            full = True,
        )

Changelog
#########

When you make **any** change to this repository it **MUST** be accompanied by a changelog file.
The changelog for this repository uses the `towncrier <https://github.com/twisted/towncrier>`__ package.
Entries in the changelog for the next release are added as individual files (one per change) to the ``changelog/`` directory.

Writing a Changelog Entry
^^^^^^^^^^^^^^^^^^^^^^^^^

A changelog entry accompanying a change should be added to the ``changelog/`` directory.
The name of a file in this directory follows a specific template::

  <PULL REQUEST NUMBER>.<TYPE>[.<COUNTER>].rst

The fields have the following meanings:

* ``<PULL REQUEST NUMBER>``: This is the number of the pull request, so people can jump from the changelog entry to the diff on BitBucket.
* ``<TYPE>``: This is the type of the change and must be one of the values described below.
* ``<COUNTER>``: This is an optional field, if you make more than one change of the same type you can append a counter to the subsequent changes, i.e. ``100.bugfix.rst`` and ``100.bugfix.1.rst`` for two bugfix changes in the same PR.

The list of possible types is defined the the towncrier section of ``pyproject.toml``, the types are:

* ``feature``: This change is a new code feature.
* ``bugfix``: This is a change which fixes a bug.
* ``doc``: A documentation change.
* ``removal``: A deprecation or removal of public API.
* ``misc``: Any small change which doesn't fit anywhere else, such as a change to the package infrastructure.


Rendering the Changelog at Release Time
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When you are about to tag a release first you must run ``towncrier`` to render the changelog.
The steps for this are as follows:

* Run `towncrier build --version vx.y.z` using the version number you want to tag.
* Agree to have towncrier remove the fragments.
* Add and commit your changes.
* Tag the release.

**NOTE:** If you forget to add a Changelog entry to a tagged release (either manually or automatically with ``towncrier``)
then the Bitbucket pipeline will fail. To be able to use the same tag you must delete it locally and on the remote branch:

.. code-block:: bash

    # First, actually update the CHANGELOG and commit the update
    git commit

    # Delete tags
    git tag -d vWHATEVER.THE.VERSION
    git push --delete origin vWHATEVER.THE.VERSION

    # Re-tag with the same version
    git tag vWHATEVER.THE.VERSION
    git push --tags origin main

.. |codecov| image:: https://codecov.io/bb/dkistdc/solar-wavelength-calibration/branch/main/graph/badge.svg
    :target: https://codecov.io/bb/dkistdc/solar-wavelength-calibration
