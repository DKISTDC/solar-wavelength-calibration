solar-wavelength-calibration
----------------------------
|codecov|

A package for calibrating wavelength data. Please see the `online docs <https://docs.dkist.nso.edu/projects/solar-wavelength-calibration/en/latest/>`_
for detailed information.

Deployment
----------
solar-wavelength-calibration is deployed to `PyPI <https://pypi.org/project/solar-wavelength-calibration/>`_

Development
-----------
Install `Git LFS <https://git-lfs.com/>`_ and run ``git lfs install`` once for
your user account before cloning. With the default LFS settings, ``git clone``
automatically downloads the complete atlas files.

.. code-block:: bash

    git clone git@bitbucket.org:dkistdc/solar-wavelength-calibration.git
    cd solar-wavelength-calibration
    pre-commit install
    pip install -e .[test]
    pytest -v --cov solar-wavelength-calibration

Reference atlas test data
^^^^^^^^^^^^^^^^^^^^^^^^

Use the complete, unmodified solar and telluric atlases in
``solar_wavelength_calibration/tests/data/full-atlas/`` for scientifically
meaningful output. These canonical test data are stored with Git LFS; there is
no intermediate or reduced atlas.
Use the tiny ``test_*_reference_atlas.npy`` fixtures in
``solar_wavelength_calibration/tests/data/`` when scientific output is immaterial,
such as cache, download, configuration, and failure tests. Tiny fixtures remain
in ordinary Git, retaining the original NSO fixture bytes.

The full files come from
`doi:10.7910/DVN/JUSYJM <https://doi.org/10.7910/DVN/JUSYJM>`_, Dataverse dataset
version 1.1, released 2026-01-28, under CC0-1.0. Both upstream file versions are 1.
The tiny files came from the
`NSO test fixtures <https://share.nso.edu/shared/dkist/DC/solar_wavelength_calibration_atlas/>`_.

Tests download both fixture sizes over HTTP through Pooch, which follows LFS
redirects, verifies hashes, and caches the files. Their Bitbucket source API URLs
pin data commit ``ac3795de15003b7bd1e8c1a5304c294446c0cc62``, retained by the
permanent tag ``atlas-data-consolidated``. The URLs do not change after merging or
deleting the feature branch, including after a squash merge.
Local LFS hydration is not required for tests, so Pipelines disables clone-time
LFS downloads. Use ``git lfs pull`` if an existing checkout needs the full files.
Tests do not contact Dataverse; production endpoint availability is not tested
by this suite. The library's production Dataverse default is unchanged.

Package discovery includes ``solar_wavelength_calibration*`` except the full-atlas directory.
The ``exclude-package-data`` setting keeps ``tests/data/full-atlas/`` out of wheels;
``MANIFEST.in`` excludes that directory from sdists, whether the checkout contains
full files or LFS pointers. Tiny fixtures remain packaged.

Updating the atlases
^^^^^^^^^^^^^^^^^^^

1. On a feature branch, replace the full files with complete, unmodified upstream
   files. Verify their SHA-256 and MD5 against the upstream identifiers and update
   the provenance above. Keep the full files in LFS and tiny fixtures in ordinary
   Git. Add the required changelog entry.
2. Commit the data first and push the feature branch; the Git LFS pre-push hook
   uploads the full objects. Record that data commit's full SHA with
   ``git rev-parse HEAD``.
3. Create and push a new annotated, permanent data tag on that commit::

       git tag -a atlas-data-NAME DATA_COMMIT -m "Reference atlas data snapshot"
       git push origin refs/tags/atlas-data-NAME

   Replace ``NAME`` with a unique descriptive name containing no digits, and
   ``DATA_COMMIT`` with the full SHA. Digit-free names keep data tags out of
   setuptools-scm's package-version discovery; do not use release-style ``v*``
   names. Never move or delete an existing data tag: older tests depend on it
   retaining their pinned commit even after a squash merge and branch deletion.
4. In a subsequent commit, update ``test_full_atlas_config`` in
   ``solar_wavelength_calibration/tests/conftest.py`` and
   ``test_small_atlas_config`` in ``solar_wavelength_calibration/tests/test_atlas.py``
   to the published data commit. Keep the source API URL pattern
   ``https://api.bitbucket.org/2.0/repositories/dkistdc/solar-wavelength-calibration/src/COMMIT/PATH``.
   Update Pooch's expected hashes for any changed files and the commit/tag above.
   Do not change the production Dataverse configuration as part of a test-data update.
5. Verify anonymous downloads from the exact commit URLs, following redirects
   (for example, ``curl -q --fail --location``), and compare SHA-256, MD5, and sizes.
   Do not use repository archives, which may contain LFS pointers. Run the atlas
   and scientific tests with fresh caches so HTTP downloads are exercised, and
   check that the wheel and sdist exclude the full files and LFS pointers.
   Push the test changes for review; the commit-pinned URLs are final before merging.

Installation
------------
.. code:: bash

   pip install solar-wavelength-calibration


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
