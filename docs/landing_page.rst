solar-wavelength-calibration
============================

Accurately determine the wavelength axis of a solar spectrum.

Overview and Features
---------------------

This library is designed to match an observed solar spectrum to a reference solar atlas in order to produce a mapping
from raw pixel index to an absolute wavelength value. It has the following features:

* Physically motivated model of spectrograph optics that lead to non-linear wavelength solutions
* User-definable solar atlas or a sensible default
* Support for fitting telluric lines of variable strength with a Doppler offset
* Degradation of atlas spectral resolution to match spectrograph resolving power
* Fitted estimation of stray light in the observed spectrum
* Fully `astropy.units`-aware
* Uses `scipy.optimize` under the hood, and built on top of `lmfit <https://lmfit.github.io/lmfit-py/index.html>`_; easily change the fitting method, bounds, etc.
* Output that is fully compliant with FITS definitions for spectral WCS

.. _quickstart:

Quickstart / Example
====================

First, let's import packages we'll need. We will be using the `dkist <https://docs.dkist.nso.edu/projects/python-tools/en/stable/>`_
only to download example data. If you have your own spectra you can skip this import.

.. code:: python

    from solar_wavelength_calibration import Atlas, WavelengthCalibrationFitter, WavelengthCalibrationParameters
    from solar_wavelength_calibration import BoundsModel, FitFlagsModel, LengthBoundRange, DispersionBoundRange, AngleBoundRange, UnitlessBoundRange

    import matplotlib.pyplot as plt
    import astropy.units as u
    from astropy.wcs import WCS

    import numpy as np

    import dkist
    from dkist.data.sample import VISP_L1_KMUPT

Let's load and plot an example spectrum from the DKIST's ViSP instrument (a slit spectro-polarimeter)

.. code:: python

    # This slice grabs only Stokes I from a single frame
    dataset = dkist.load_dataset(VISP_L1_KMUPT)[0, 200]
    # Average along the slit to make a single spectrum
    spec = dataset.rebin((1, -1)).squeeze().data.compute()

    ax = plt.figure().add_subplot(111)
    ax.plot(spectrum)
    ax.set_xlabel("Pixel")
    ax.set_ylabel("Normalized Signal")

.. figure:: images/raw_visp_spectrum.jpg
    :alt: Raw spectrum

    Raw spectrum with no wavelength axis

Let's use a solar atlas to fit a wavelength solution to this spectrum. To begin with, we'll need some good initial guesses
for the offset and dispersion (called "CRVAL" and "CDELT" in FITS headers). Fortunately, these data came with headers
that have this info; you may need to source your initial guesses from somewhere else.

We also know the resolving power of our spectrograph, so let's define that here. This is not crucial for a good fit, but
does help match line shapes and depths. If you know it, use it!

NOTE: We'll start applying units at this point because all downstream machinery expects it.

.. code:: python

    crval_init = dataset.headers[0]["LINEWAV"] * u.nm
    cdelt_init = dataset.headers[0]["CDELT2"] * u.nm / u.pix
    crpix = dataset.headers[0]["CRPIX2"]
    resolving_power = 1.2e5

The full parameterization of a wavelength solution allows for non-linear terms that depend on the spectrograph's
incident light angle, grating constant, and spectral order, but in our case we're going to see how well we do with a
simple, linear solution. This can be achieved by setting the following values

.. code:: python

    incident_light_angle = 0 * u.deg
    order = 1
    grating_constant = 1 / u.m

To show off some additional features of our physical model let's also fit for varying strength in telluric lines, a
fraction contribution from stray light, and an overall continuum offset. We'll define our initial starting guesses for
these parameters like so:

.. code:: python

    initial_opacity_factor = 3
    initial_straylight_fraction = 0.01
    initial_continuum_level = 1.0

We also have the option to offset the telluric lines to account for the relative velocity between the Earth and the Sun
at the time of observations. For now we'll just set this velocity to 0.

.. code:: python

    doppler_velocity = 0 * u.km / u.s

Next let's define valid ranges for our fit parameters. This step isn't *strictly* necessary, but is generally good practice.
The more information we can give the fitter the better! Valid ranges are also required for certain fitting methods, like
`differential_evolution`. We define the ranges using the `~solar_wavelength_calibration.fitter.parameters.BoundsModel` class:

.. code:: python

    bounds = BoundsModel(
        crval=LengthBoundRange(min=crval_init - 0.2*u.nm, max=crval_init + 0.2*u.nm),
        dispersion=DispersionBoundRange(min=cdelt_init * 0.6, max=cdelt_init*1.4),
        resolving_power = UnitlessBoundRange(min=1e5, max=5e5),
        opacity_factor=UnitlessBoundRange(min=3, max=5),
        straylight_fraction=UnitlessBoundRange(min=0.0, max=0.8),
        continuum_level=UnitlessBoundRange(min=0.8, max=1.1)
        )

As discussed above, there are non-linear parameters that we are going to ignore for the time being. To do this we will
use the `~solar_wavelength_calibration.fitter.parameters.FitFlagsModel` to tell the fitter which parameters should be free or fixed in the fit.

.. code:: python

    fit_flags = FitFlagsModel(
        crval=True,
        dispersion=True,
        incident_light_angle=False,  # Set to False because we don't care about this higher-order parameter for now
        resolving_power=True,
        opacity_factor=True,
        straylight_fraction=True,
        continuum_level=True
        )

Now we can set up the final pieces needed to perform the fit. The `~solar_wavelength_calibration.fitter.parameters.WavelengthCalibrationParameters`
object holds all of our prior knowledge that we just discussed

.. code:: python

    params = WavelengthCalibrationParameters(
        crval=crval_init,
        dispersion=cdelt_init,
        incident_light_angle = incident_light_angle,
        grating_constant = grating_constant,
        doppler_velocity = 0 * u.km / u.s,
        order=order,
        resolving_power=resolving_power,
        opacity_factor=initial_opacity_factor,
        straylight_fraction=initial_straylight_fraction,
        continuum_level=initial_continuum_level,
        bounds=bounds,
        fit_flags=fit_flags
        )

At this point we also need a solar Atlas to serve as our reference spectrum. The default atlas that comes with this
package is a good starting point.

.. code:: python

    atlas = Atlas()

We can now initialize our fitter with the known priors (`params`) and our atlas (`atlas`)

.. code:: python

    fitter = WavelengthCalibrationFitter(
        input_parameters=params,
        atlas=atlas
        )

Let's do the fit! Because we're fitting a lot of parameters we'll use `differential_evolution` with a tight tolerance

.. code:: python

    fit_result = fitter(
        input_spectrum=spectrum,
        method="differential_evolution",
        tol=1e-10
        )

There are lots of ways to examine the fit; for now we'll take a look at the FITS header parameterization of our wavelength
solution.

.. code:: python

    header = fit_result.wavelength_parameters.to_header(axis_num=1)
    print(header)

    {'CTYPE1': 'AWAV-GRA',
     'CUNIT1': 'nm',
     'CRPIX1': 491,
     'CRVAL1': 630.1492172228517,
     'CDELT1': 0.001280789155906839,
     'PV1_0': 1.0,
     'PV1_1': 1,
     'PV1_2': 0.0}

We can use this header to generate the fit wavelength axis for our data. To do this we'll use the `astropy.wcs` package.
Let's also plot our spectrum (with fit wavelengths) against the best-fit atlas to see how we did.

.. code:: python

    wcs_obj = WCS(header)
    best_fit_wavelength = WCS.spectral.pixel_to_world(np.arange(spectrum.size)).to_value(u.nm)
    best_fit_atlas = fit_result.best_fit_atlas

    ax = plt.figure(figsize=(6,4)).add_subplot(111)
    ax.plot(best_fit_wavelength, spectrum, label="Observed")
    ax.plot(best_fit_wavelength, best_fit_atlas, label="Best Fit Atlas")
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Normalized Signal")
    ax.legend(loc=0)

.. figure:: images/fit_visp_spectrum.jpg
    :alt: Fit spectrum

    Input spectrum with fit wavelength axis and best-fit atlas spectrum

Looks good! We've missed a bit on the strength/resolution of some of the lines, but the wavelength solution looks spot on!
