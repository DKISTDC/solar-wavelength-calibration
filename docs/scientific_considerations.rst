Scientific Considerations
=========================

The purpose of this page is to provide useful context from a "Science use case" persepective that will hopefully help
you get the most our of this package.

Input Data
----------

The ideal input spectrum comes from flat-field calibration data taken soon after or before your science data. A flat-field
spectrum is preferred because it should be free of "scientific" impacts on the spectral shape and will have a higher
signal-to-noise (SNR) than science data. If the instrumet in question takes two-dimensional spectra (e.g., a slit spectrograph)
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
in your `~solar_wavelength_calibration.fitter.parameters.WavelengthCalibrationParameters` instance and ``opcaity_factor = False``
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


