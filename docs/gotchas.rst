Gotchas
=======

This page lists some common pitfalls that users may experience and enumerates some of the assumptions made by this library.

Atlas Assumptions
-----------------

Some aspects of the :doc:`physical model <physical_model>` make assumptions about the nature of the telluric and solar
atlas spectra provided to a `~solar_wavelength_calibration.fitter.wavelength_fitter.WavelengthCalibrationFitter` instance.
Violating these assumptions will probably not cause a failure when fitting a wavelength solution, but will change how
the :ref:`parameters related to spectral matching <spectral_parameters>` are interpreted. The assumptions are:

* The continuum level of both the telluric and solar atlases is 1.0
* Both atlases are fully resolved
* Neither atlas contains emission lines
* Both atlases cover the range of wavelengths spanned by the observed spectrum (violating this assumption *will* break wavelength fits)

Physical WCS Constraints and Bounds on CRVAL
--------------------------------------------

The full set of :ref:`FITS header parameters <header_parameters>` are not completely independent. In particular, the `crval`,
`incident_light_angle`, `order`, and `grating_constant` parameters are linked as described
`here <solar_wavelength_calibration.fitter.parameters.WavelengthCalibrationParameters.ensure_invertable_WCS>`. A consequence
of this constraint is that during the fit `crval` is actually fit with the
`~solar_wavelength_calibration.fitter.parameters.WavelengthCalibrationParameters.delta` parameter, which makes strictly
honoring the bounds on `crval` in `~solar_wavelength_calibration.fitter.parameters.BoundsModel` not completely possible.
Instead, the accuracy of the bounds on `crval` is determined by the width of the bounds on `incident_light_angle`
(see `~solar_wavelength_calibration.fitter.parameters.WavelengthCalibrationParameters.delta_bounds_given_crval_bounds`
for more information).

Given these constraints it is recommended to fix `incident_light_angle` in the fit. If this is not possible, the bounds
of `incident_light_angle` should be set as narrow as possible so that the bounds on `crval` will be as close to what was
asked for as possible.
