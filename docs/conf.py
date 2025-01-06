"""Configuration file for the Sphinx documentation builder."""
# -- stdlib imports ------------------------------------------------------------
import importlib
import sys
import warnings
from importlib.metadata import distribution

from dkist_sphinx_theme.conf import *
from dkist_sphinx_theme.create_intersphinx_mapping import create_intersphinx_mapping
from packaging.version import Version

# Need a name for the overall repo
# __name__ where this code executes is "builtins" so that is no help
repo_name = "solar-wavelength-calibration"
package_name = repo_name.replace("-", "_")

dist = distribution(package_name)
package = importlib.import_module(package_name)

# -- Check for docs dependencies ----------------------------------------------------
missing_requirements = missing_dependencies_by_extra(package_name, extras=["docs"])
if missing_requirements["docs"]:
    print(
        f"The {' '.join(missing_requirements['docs'])} package(s) could not be found and "
        "is needed to build the documentation, please install the 'docs' requirements."
    )
    sys.exit(1)

# auto api parameters that cannot be moved into the theme:
autoapi_dirs = [Path(package.__file__).parent]
# Uncomment this for debugging
# autoapi_keep_files = True
suppress_warnings = ["autoapi.python_import_resolution"]

# -- Options for intersphinx extension -----------------------------------------
intersphinx_mapping = create_intersphinx_mapping(repo_name)
# Remaining sphinx settings are in dkist-sphinx-theme conf.py

# -- Project information -------------------------------------------------------
project = "SOLAR-WAVELENGTH-CALIBRATION"

# The full version, including alpha/beta/rc tags
dkist_version = Version(dist.version)
is_release = not (dkist_version.is_prerelease or dkist_version.is_devrelease)
# We want to ignore all warnings in a release version.
if is_release:
    warnings.simplefilter("ignore")
# Extensions so we can create the reqmts table
extensions += ["dkist_sphinx_theme.create_requirements_table"]
