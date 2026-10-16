solar-wavelength-calibration
----------------------------
|pypiversion| |pyversions| |codecov| |rtdstatus| |bbstatus| |pyOS| |zenodo|

A package for matching an observed solar spectrum to a reference solar atlas in order to produce a mapping from raw pixel
index to an absolute wavelength value. Please see the `online docs <https://docs.dkist.nso.edu/projects/solar-wavelength-calibration/en/latest/>`_
for detailed information.

Installation
------------
.. code:: bash

   pip install solar-wavelength-calibration

Development
-----------

The source repository can be cloned from BitBucket with the following command

.. code-block:: bash

    git clone git@ssh.bitbucket.org:dkistdc/solar-wavelength-calibration.git

If you don't have a BitBucket account use `https://bitbucket.org/dkistdc/solar-wavelength-calibration.git` as the URL
instead.

Linting
#######

To replicate the standard developement workflow you must also install `pre-commit <https://pre-commit.com/>`_ and set it
up to run whenever a git commit is made:

.. code-block:: bash

    cd solar-wavelength-calibration
    pre-commit install

To run `pre-commit` without having to make a commit simply run

.. code-block:: bash

    pre-commit run --all-files

This will check the formatting of all code against the constraints defined in ".pre-commit-config.yaml".

Unit Tests
###########

Reference atlas test data
^^^^^^^^^^^^^^^^^^^^^^^^^

Install `Git LFS <https://git-lfs.com/>`_ and run ``git lfs install`` once for
your user account before cloning. With the default LFS settings, ``git clone``
automatically downloads the complete atlas files.

Use the complete, solar and telluric atlases in
``solar_wavelength_calibration/tests/data/full-atlas/`` for scientifically
meaningful output. These canonical test data are stored with Git LFS.

Use the tiny ``test_*_reference_atlas.npy`` fixtures in
``solar_wavelength_calibration/tests/data/`` when scientific output is immaterial,
such as cache, download, configuration, and failure tests. Tiny fixtures remain
in ordinary Git.

Running Tests
^^^^^^^^^^^^^

Unit tests can be run by installing an editable version and running `pytest`:

.. code-block:: bash

    pip install -e .[test]
    pytest -v --cov solar-wavelength-calibration

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

Deployment
##########

solar-wavelength-calibration is deployed to `PyPI <https://pypi.org/project/solar-wavelength-calibration/>`_ as part of
BitBucket pipelines that are run on commits to the "main" branch tagged with a version tag (`vX.Y.Z`).

.. |codecov| image:: https://codecov.io/bb/dkistdc/solar-wavelength-calibration/branch/main/graph/badge.svg
    :target: https://codecov.io/bb/dkistdc/solar-wavelength-calibration
    :alt: Latest Codecov status
.. |rtdstatus| image:: https://app.readthedocs.com/projects/dkistdc-solar-wavelength-calibration/badge/?version=latest
    :target: https://docs.dkist.nso.edu/projects/solar-wavelength-calibration/en/latest/
    :alt: Documentation Status
.. |pypiversion| image:: https://img.shields.io/pypi/v/solar-wavelength-calibration.svg
    :target: https://pypi.org/project/solar-wavelength-calibration/
    :alt: PyPI package
.. |pyversions| image:: https://img.shields.io/pypi/pyversions/solar-wavelength-calibration
    :alt: Supported Python Versions
.. |bbstatus| image:: https://img.shields.io/bitbucket/pipelines/dkistdc/solar-wavelength-calibration/main
    :target: https://bitbucket.org/dkistdc/solar-wavelength-calibration/pipelines/results/branch/main/page/1
    :alt: Bitbucket Pipeline Status
.. |pyOS| image:: https://pyopensci.org/badges/peer-reviewed.svg
    :target: https://github.com/pyOpenSci/software-submission/issues/283
    :alt: pyOpenSci Peer-Reviewed
.. |zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.21420011.svg
    :target: https://doi.org/10.5281/zenodo.21420011
    :alt: Zenodo DOI
