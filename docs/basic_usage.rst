Basic Usage
===========

This page describes all the tools provided by `solar-wavelength-calibration` that are needed to set up and perform a fit.
For a quick introduction to most of these tools via example see the :ref:`quickstart <quickstart>`.

Defining Prior Knowledge
------------------------

The first step to setting up a fit is to collect as much known information about the problem as possible. Practically
speaking this looks like choosing sane initial guesses for the parameters to be fit, defining constant parameters
that describe the spectrograph setup, and, optionally, constraining the fit with bounds and fit flags.

Setting Parameters
^^^^^^^^^^^^^^^^^^

The `~solar_wavelength_calibration.fitter.parameters.WavelengthCalibrationParameters` object is a container for all
initial guesses and spectrograph configuration parameters. In other words, it contains all parameters needed to define
*one possible* wavelength solution. Once it has been set up it will be passed to the `fitter <solar_wavelength_calibration.fitter.wavelength_fitter.WavelengthCalibrationFitter>`
to produce the *best fit* wavelength solution.

`~solar_wavelength_calibration.fitter.parameters.WavelengthCalibrationParameters` has the following inputs:

.. _parameters_table:

======================= ======================================== ============== ========== ============== =========
argument                description                              units          constraint fit parameter? required?
======================= ======================================== ============== ========== ============== =========
`crval`                 Reference wavelength                     length         finite     yes            yes
`dispersion`            Spectral dispersion                      length / pix   finite     yes            yes
`incident_light_angle`  Grating incident light angle             angle          finite     yes            yes
`grating_constant`      Grating constant                         1 / length     finite     yes            yes
`doppler_velocity`      Relative speed between Sun and observer  length / time  finite     no             yes
`order`                 Spectral order                                          integer    no             yes
`continuum_level`       Overall continuum scaling                               float      yes            no
`resolving_power`       Spectrograph resolving power                            int        yes            no
`opacity_factor`        Scaling factor for telluric opacity                     finite     yes            no
`straylight_fraction`   Fractional additive stray light                         finite     yes            no
======================= ======================================== ============== ========== ============== =========

.. admonition:: Note on units

    This package uses `astropy.units` in an attempt to minimize (and perhaps even eliminate!) errors caused by incorrect
    units. All inputs that have physical units *must* have astropy units assigned to their variables. The exact unit doesn't
    matter as long as it is the correct type. For example, the `doppler_velocity` could be given as `2 * u.au / u.year`
    with no issue. Errors will be raised if the incorrect unit type is provided.

In addition to the parameters themselves, `~solar_wavelength_calibration.fitter.parameters.WavelengthCalibrationParameters`
also has `bounds` and `fit_flags` arguments that are described below.

Setting Fit Bounds
^^^^^^^^^^^^^^^^^^

Another way to define prior knowledge of the wavelength fitting problem is to set sensible bounds on the parameters to
be fit. We do this with the `~solar_wavelength_calibration.fitter.parameters.BoundsModel` class. This class has input
arguments corresponding to :ref:`parameters <parameters_table>` that are labeled as "fit parameters". The value assigned
to these inputs must be the `*BoundRange` classes appropriate for the given unit. For example, the bounds on `crval`, a
length, must be defined with the `~solar_wavelength_calibration.fitter.parameters.LengthBoundRange` class.

Valid usages of the `*BoundRange` classes are:

.. code-block:: python

    # With no arguments the default bounds are [-inf, inf]. AKA unbounded
    LengthBoundRange()

    # If either `max` or `min` are not given then the upper/lower limit is unbounded (± inf)
    UnitlessBoundRange(min=-23)
    DispersionBoundRange(max=34 * u.nm / u.pix)

To highlight some of the guard rails provided by the `*BoundRange` classes, here are some examples of range definitions
that will cause errors:

.. code-block:: python

    # This will error because the units do not describe an angle
    AngleBoundRange(max=180 * u.km)

    # This will error because min > max
    LengthBoundRange(min=1 * u.m, max=100 * u.mm)

The `~solar_wavelength_calibration.fitter.parameters.BoundsModel` class then takes a `*BoundRange` for each fit parameter.
*None* of the inputs to `~solar_wavelength_calibration.fitter.parameters.BoundsModel` are required. If a bound is not
provided the default range is `[0, inf]` (`[0 deg, 180 deg]` for `incident_light_angle`).

An instance of `~solar_wavelength_calibration.fitter.parameters.BoundsModel` is used in a fit by passing it to the
`bounds` argument of `~solar_wavelength_calibration.fitter.parameters.WavelengthCalibrationParameters`.

Setting Fit Parameters
^^^^^^^^^^^^^^^^^^^^^^

It is possible to only fit a subset of the :ref:`parameters <parameters_table>` that are "fit parameters". To do this
we use the `~solar_wavelength_calibration.fitter.parameters.FitFlagsModel`, which simply contains a boolean for each
fit parameter. If the value is `True` then that parameter will be free during the fit. If `False` then the parameter
will be fixed to the value input to `~solar_wavelength_calibration.fitter.parameters.WavelengthCalibrationParameters`.

An instance of `~solar_wavelength_calibration.fitter.parameters.FitFlagsModel` is used in a fit by passing it to the
`fit_flags` argument of `~solar_wavelength_calibration.fitter.parameters.WavelengthCalibrationParameters`.

Get a Reference Atlas
---------------------

At a bare minimum we need two reference arrays to fit a wavelength solution: the reference solar spectrum and
associated wavelengths. Unless you have space-based observations you will also need two arrays that define the telluric
absorption spectrum and wavelength.

This package uses the `~solar_wavelength_calibration.atlas.base.AtlasBase` abstract class to define how these four arrays
should be accessed. In its most basic form this base class simply requires the definition of four properties:
`telluric_atlas_wavelength`, `telluric_atlas_transmission`, `solar_atlas_wavelength`, and `solar_atlas_transmission`. A
user is free to define these properties however they see fit.

In addition to the base class, this package defines two `Atlas` types for common use patterns.

Download Atlas Data From Zenodo
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Atlas data that have been published on Zenodo can be automatically downloaded and cached with `pooch <https://www.fatiando.org/pooch/latest/>`_
by using the `~solar_wavelength_calibration.atlas.atlas.Atlas` class. The default configuration assumes the Zenodo record
contains two "\*.npy" files, one for the telluric atlas and one for the solar atlas. The default values correspond to
the `atlas provided by DKIST <https://zenodo.org/records/14674504>`_, which covers a wavelength range of 350 nm to 5000 nm.

To use this default atlas simply instantiate the `~solar_wavelength_calibration.atlas.atlas.Atlas` class with no parameters.

.. code-block:: python

    atlas = Atlas()

Define Atlas Arrays Locally
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Atlas arrays that are available locally can be placed in the `Atlas` framework using the `~solar_wavelength_calibration.atlas.atlas.LocalAtlas`
class. To do so, simply pass the appropriate values to the `telluric_atlas_wavelength`, `telluric_atlas_transmission`,
`solar_atlas_wavelength`, and `solar_atlas_transmission` arguments.

.. code-block:: python

    solar_wave, solar_spec = my_solar_loader_function()
    telluric_wave, telluric_spec = my_telluric_loader_function()

    atlas = LocalAtlas(
        telluric_atlas_wavelength=telluric_wave,
        telluric_atlas_transmission=telluric_spec,
        solar_atlas_wavelength=solar_wave,
        solar_atlas_transmission=solar_spec,
        )

.. note::

    The `telluric_atlas_wavelength` and `solar_atlas_wavelength` arrays must have `astropy length units<astropy.units>`
    applied.

Run The Fit
-----------

Once we have collected all our prior knowledge and a reference atlas, running a fit is fairly straightforward. First
we instantiate a `~solar_wavelength_calibration.fitter.wavelength_fitter.WavelengthCalibrationFitter` class with instances
of `~solar_wavelength_calibration.fitter.parameters.WavelengthCalibrationParameters` and a subclass of
`~solar_wavelength_calibration.atlas.base.AtlasBase`.

.. code-block:: python

    fitter = WavelengthCalibrationFitter(
        input_parameters=my_great_wavelength_calibration_parameters_object,
        atlas=atlas
        )

This fitter can then be used to fit any number of raw spectra by calling it once for each fit. There are many options
here, but the only required argument is the spectrum to be fit.

.. code-block:: python

    # This will run a fit
    fit_result = fitter(input_spectrum=raw_spec)

The two other named kwargs available are `method` and `spectral_weights`. `method` is used to define the minimization
algorithm used. See the `lmfit documentation <https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.minimize>`_
for all available methods; generally any method defined in `scipy.optimize` is available, along with some additional
methods like ``emcee``. `spectral_weights` can be an array with the same length as `input_spectrum` that defines relative
weights for each spectral pixel.

Any additional kwargs passed to the fitter will be passed directly to `lmfit.minimizer.minimize`. In many cases these
kwargs will be passed to the core fitting routine defined in `scipy.optimize`.

For example, we can use the ``differential_evolution`` method and pass kwargs that are unique to
`that method <scipy.optimize.differential_evolution>`

.. code-block:: python

    fit_result = fitter(
        input_spectrum=raw_spec,
        method="differential_evolution",
        popsize=1,
        tol=1e-10,
        init='halton'
        )

The same `fitter` can be used for multiple fits.

Use Fit Results
---------------

.. _fit_results:

The `~solar_wavelength_calibration.fitter.wavelength_fitter.FitResult` Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The output of a call to a `~solar_wavelength_calibration.fitter.wavelength_fitter.WavelengthCalibrationFitter` instance
is an instance of `~solar_wavelength_calibration.fitter.wavelength_fitter.FitResult`. This object contains the following
properties:

:wavelength_parameters: An instance of `~solar_wavelength_calibration.fitter.wavelength_fitter.WavelengthParameters`. See
  the :ref:`following section <wavelength_parameters>` for more information.

:minimizer_result: An instance of `lmfit.minimizer.MinimizerResult`. This object contains statistics and information about the running of the fit.

:input_wavelength_vector: Wavelength vector computed from the input parameter values.

:input_spectrum: The input spectrum. Just a copy of what was used in the call to `fitter`.

:spectral_weights: The input weights. Just a copy of what was used in the call to `fitter`.

:prepared_weights: The actual weights used in the fit. See :ref:`this section <weighting>` for more information.

:best_fit_atlas: The best-fit combined solar and telluric atlas with all instrumental effects applied.

.. _wavelength_parameters:

The `~solar_wavelength_calibration.fitter.wavelength_fitter.WavelengthParameters` Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One of the most useful fit outputs is the `~solar_wavelength_calibration.fitter.wavelength_fitter.WavelengthParameters`,
which contains all of the values needed by the FITS spectral WCS parameterization. In particular, the
`~solar_wavelength_calibration.fitter.wavelength_fitter.WavelengthParameters.to_header` method produces a `dict` of values
that can be inserted directly into an existing FITS header in order to apply the fit wavelength solution.

.. code-block:: python

    wave_params = fit_result.wavelength_parameters

    # Generate FITS keys assuming the spectral axis is the 2nd array dimension
    solution_header_entries = wave_params.to_header(axis_num=2)

    # Profit
    my_great_header.update(solution_header_entries)

Helper Functions
----------------

The following functions are not required to run a fit, but may be useful in setting everything up.

Computing An Initial CRVAL Guess
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Good initial guesses can greatly improve the performance and accuracy of a fit. In particular, the initial guess of the
reference wavelength (`crval`) can ruin a fit if it is too far off. In cases where the reference wavelength is not well
known the `~solar_wavelength_calibration.fitter.wavelength_fitter.calculate_initial_crval_guess` function may be helpful.
This function does a brute-force minimization of only a single parameter (`crval`) to match an input spectrum with a given
`Atlas`. The `negative_limit`, `positive_limit`, and `num_steps` arguments define the range of values considered.

Note that, because this function only "fits" `crval`, it requires a somewhat good match between the input data and `Atlas`
in terms of dispersion and spectral resolution.

Generate A Wavelength Vector
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are many cases where generating a wavelength vector from a given set of
`~solar_wavelength_calibration.fitter.parameters.WavelengthCalibrationParameters` is useful and the
`~solar_wavelength_calibration.fitter.wavelength_fitter.calculate_linear_wave` function does just this. This function is
primarily used internally, so it's calling signature is a little unusual. Given a
`~solar_wavelength_calibration.fitter.parameters.WavelengthCalibrationParameters` called `my_wavecal_parameters` here is
how to use it (note in particular the conversion of `grating_constant` to 1 / m):

.. code-block:: python

    num_wave = raw_spec.size
    init_wave_vec = calculate_linear_wave(
        params=my_wavecal_parameters_object.lmfit_parameters,
        number_of_wave_pix=num_wave,
        grating_constant=my_wavecal_parameters_object.grating_constant.to_value(1 / u.m),
        order=my_wavecal_parameters_object.order
        )
