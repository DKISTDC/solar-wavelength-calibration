Scientific Considerations
=========================

The purpose of this page is to provide useful context from a "Science use case" perspective that will hopefully help
you get the most our of this package.

Input Data
----------

The ideal input spectrum comes from flat-field calibration data taken soon after or before your science data. A flat-field
spectrum is preferred because it should be free of "scientific" impacts on the spectral shape and will have a higher
signal-to-noise (SNR) than science data. If the instrument in question takes two-dimensional spectra (e.g., a slit spectrograph)
then averaging some or all of the non-spectral dimension into a single spectrum will increase the SNR even further. Note
that care should be taken during this averaging because many instruments have slightly different line shapes in different regions
of the detector;an average may artificially degrade the spectral resolution.

If no flat-field spectra are available it is up to you to decide if calibrating directly on science data is acceptable.
For example, a science case where only the relative distances between spectral features is needed may be sufficiently
calibrated with science data. On the other hand, if your science relies on measuring the absolute location of a spectral
feature (e.g., to measure velocity) then you need to be very careful to not accidentally "calibrated out" your science!
In some cases this simply may not be possible without flat-field data.

Telluric Lines
--------------

The default `~solar_wavelength_calibration.atlas.atlas.Atlas` used for calibration contains both Solar and telluric spectra
and the strength of the telluric lines is directly controlled with the :ref:`opacity_factor <opacity_factor>` fit parameter.
If you want to remove telluric lines entirely it is recommended to either define a custom Atlas by overriding the
`~solar_wavelength_calibration.atlas.atlas.Atlas.telluric_atlas_transmission` property or simply setting ``opacity_factor = 0``
in your `~solar_wavelength_calibration.fitter.parameters.WavelengthCalibrationParameters` instance and ``opacity_factor = False``
in your `~solar_wavelength_calibration.fitter.parameters.FitFlagsModel` instance.

Relative Line Shapes
^^^^^^^^^^^^^^^^^^^^

The lines for both solar and telluric spectra in the default `~solar_wavelength_calibration.atlas.atlas.Atlas` are fully
resolved and thus the single ``resolving_power`` parameter can be used to degrade the resolution of the combined spectrum
to match the input data (see :ref:`here <spectral_resolution>` for more information).

If you think your represent a case where there is a *relative* difference in either line shape or resolution between the
solar and telluric lines the solution is to define a custom Atlas that overrides either the `solar <solar_wavelength_calibration.atlas.atlas.Atlas.solar_atlas_transmission>`
or `telluric <solar_wavelength_calibration.atlas.atlas.Atlas.telluric_atlas_transmission>` transmission properties to
account for this difference.

Fit Accuracy
------------

In this section we present general guidelines regarding the fitting accuracy of all the fit parameters. For more information,
including the code used, please see the accuracy tests page. Given that the results presented on that page represent a
only single instance of a contrived fit to "observed" atlas data care should be taken to not read too much into the following
guidelines. Variables specific to each fit (e.g., sampling resolution, spectral range, number of strong spectral features, etc.)
will affect the accuracy and precision of that fit. It is incumbent on the user, with their detailed knowledge of their
particular fitting scenario, to make their own determination of the quality of their fit results.

To help in assessing a fit, the ``minimizer_result`` attribute of a `~solar_wavelength_calibration.fitter.wavelength_fitter.FitResult`
instance is an ``lmfit`` `~lmfit.minimizer.MinimizerResult` object that holds many useful fit-quality metrics (the specific set depends on the
fitting method used). Furthermore, a basic bootstrapping algorithm can provide a good 0th-order estimation of fit uncertainties.

All of that said, these guidelines may help in getting good fits and assessing your own fit results:

#. Above all else, **provide as much knowledge as possible to the fitter**. This will be primarily in the form of good initial guesses and fit bounds.
#. Related, do not fit variables that you already know. For example, if the ``resolving_power`` is well known then turn off its fit in your `~solar_wavelength_calibration.fitter.parameters.FitFlagsModel` instance.
#. Provided decent initial guesses, the ``crval`` and ``dispersion`` parameters are generally very well fit, even in the presence of low signal-to-noise.
#. ``continuum_level``, which provides the overall scale factor, is generally well fit *provided the input spectrum has a flat continuum*. For examples of how to fit a non-flat continuum, see :ref:`here <custom_continuum_function>`.
#. ``incident_light_angle`` is a difficult parameter to fit accurately and the parameter whose bounds can ruin a fit. If possible, fix this parameter in the fit. If it must be fit then place tight bounds on it.
#. Line shape parameters (``resolving_power`` and ``opacity_factor``) are both very susceptible to noise, but rarely "ruin" fit. Detailed analysis of fit results should be made before inferring physical properties based on their best-fit values.

