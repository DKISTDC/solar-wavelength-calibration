Physical Model
==============

This page describes the parameterization of the wavelength solution and the model used to degrade a solar atlas to match
given observing conditions.

Wavelength WCS Parameterization
-------------------------------

This package uses the parameterization of `Greisen et al (2006) <https://ui.adsabs.harvard.edu/abs/2006A%26A...446..747G/abstract>`_
(see section 5 in particular) to encode non-linear spectral axes in FITS headers. Specifically, the physical model used
here is currently limited to reflection gratings in air. This results in the standard linear parameters (CRVAL, CRPIX,
CDELT, etc.) combined with the non-linear parameters defined in Greisen's Table 6. The parameters that describe a
wavelength axis, along with constant values and units enforced by this package are found in the folloing table

.. _header_parameters:

========== ====================== =========================== ======= ==========
Header Key Variable Name          Description                 Units   Value
========== ====================== =========================== ======= ==========
CTYPEn                            Type of spectral coordinate         "AWAV-GRA"
CUNITn                            Units for CRVALn and CDELn          "nm"
CRPIXn                            Reference pixel
CRVALn     `crval`                Reference wavelength        nm
CDELTn     `dispersion`           Linear dispersion           nm / px
PVn_0      `grating_constant`     Grating constant            1 / m
PVn_1      `order`                Spectral order
PVn_2      `incident_light_angle` Incident light angle        deg
========== ====================== =========================== ======= ==========

.. note::

    The units of `PVn_0` are *always* 1 / m, regardless of the value of `CUNITn`.

.. note::

    By convention, this package *always* chooses CRPIXn to be the midpoint of the input spectrum. Thus, CRVALn will *always*
    be the central wavelength.

This package use the implementation `Greisen et al (2006) <https://ui.adsabs.harvard.edu/abs/2006A%26A...446..747G/abstract>`_
defined in `wcslib <https://www.atnf.csiro.au/computing/software/wcs/wcslib/index.html>`_ and accessed through the `astropy.wcs`
package, in particular the `astropy.wcs.WCS` object.

Matching Atlas Spectra to Observation Conditions
------------------------------------------------

In addition to computing a wavelength vector using the parmeterization above we also modify the atlas spectra to match
characteristics of the observed spectrum as closely as possible. Not only does this improve the accuracy of the wavelength
fit, but the fit parameter values can also yield insight into the quality of the input spectrum.

The "spectral" parameters used to modify the atlas spectra can be found in the following table

.. _spectral_parameters:

======================= ======================================== =============
Variable Name           Description                              Units
======================= ======================================== =============
`doppler_velocity`      Relative speed between Sun and observer  length / time
`continuum_level`       Overall continuum scaling
`resolving_power`       Spectrograph resolving power
`opacity_factor`        Scaling factor for telluric opacity
`straylight_fraction`   Fractional additive stray light
======================= ======================================== =============

For each fit iteration the atlas that is matched to the input spectrum is computed in the following steps, each described
in detail below:

#. Apply an opacity scaling factor to the telluric transmission
#. Apply a Doppler offset between the telluric and solar transmissions
#. Combine the telluric and solar transmissions
#. Smooth the spectrum with a Gaussian kernel to match the given spectral resolution
#. Apply an additive stray light factor
#. Apply a multiplicative scale factor to account for different continuum scalings

The single atlas spectrum that results from these steps is then subtracted from the input spectrum to compute the
goodness-of-fit (chi-squared) value.

Apply Telluric Opacity Factor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The telluric transmission, :math:`I_{tell}(\lambda)`, is assumed to be attenuated by a standard optical depth, :math:`\tau_0`,
that is a property of a specific atlas. Thus

.. math::

    I_{tell}(\lambda) = I_{tell,0}(\lambda)e^{\tau_0}.

The `opacity_factor` fit parameter allows us to modify the total optical depth and thus the strength of the telluric lines.
It is applied via

.. math::

    I_{tell,fit}(\lambda) = I_{tell}(\lambda)^T

where :math:`T` is the opacity scaling factor such that :math:`T * \tau_0` is the final total optical depth. Note that
we don't need to know the actual values of :math:`I_{tell,0}` and :math:`\tau_0`. :math:`I_{tell}` and :math:`T` are the
only things needed by the fit.

Apply Doppler Offset
^^^^^^^^^^^^^^^^^^^^

The relative motion of the Sun and the observer (really, the Earth's atmosphere) is accounted for by applying a Doppler
shift to the Atlas' solar transmission values. This is done by interpolating the raw transmission vector onto the fit
wavelength vector with an offset of :math:`v_{obs} \lambda_0 / c`, where :math:`v_{obs}` is the relative motion between
the Sun and observer (positive for when the observer is moving away from the Sun) and :math:`\lambda_0` is the current
value of CRVALn.

Combine Spectra
^^^^^^^^^^^^^^^

The telluric and Doppler-corrected solar transmissions are combined by simply multiplying them together.

Spectral Resolution
^^^^^^^^^^^^^^^^^^^

A large mismatch in resolution (:math:`\Delta\lambda/\lambda`) between the atlas and observed spectra causes a corresponding
mismatch in line shapes that can negatively impact the accuracy of the wavelength solution (especially the higher-order
wavelength parameters and the spectral parameters). Under the assumption that the combined atlas spectrum is fully
resolved, the resolution of the two spectra is matched by degrading the combined atlas spectrum to match the resolution
of the observed spectrum. This is done by convolving the combined atlas spectrum with a Gaussian filter whose width in
pixels is

.. math::

    w_{pix} = \frac{\lambda_0}{R d\, 2 \sqrt{\ln 2}}

where :math:`\lambda_0` is the current value of CRVALn, :math:`R` is the current value of `resolving_power`, and
:math:`d` is the current value of `dispersion`.

Stray light
^^^^^^^^^^^

A contribution from "stray light" is estimated by simply adding a constant factor to the combined atlas spectra. To avoid
changing the overall continuum level (and thus coupling the additive and multiplicative scale factors) we normalize
the continuum after adding the `straylight_fraction` fit parameter such that :math:`T = (T + s) / (1 + s)` where
:math:`T` is the combined atlas transmission and :math:`s` is the `straylight_fraction`. Note that this assumes the
continuum level of the atlas spectra is 1.

Continuum
^^^^^^^^^

Any multiplicative scaling differences between the atlas and observed spectrum are accounted for with an overall scaling
factor called `continuum_level`. The combined atlas spectra is simply multiplied by this value.

.. note::

    If a simple scalar does not accurately reflect the continuum of your data, it is also possilbe to
    :ref:`define and fit your own continuum function <custom_continuum_function>`.

.. _weighting:

Goodness of Fit and Weighting
-----------------------------

The final "residuals" vector is the difference of the input spectrum and the combined and modified atlas spectra.
A user can choose to apply different weights to different portions of the spectrum by passing an array to the ``weights``
kwarg of a call to a `WavelengthCalibrationFitter <solar_wavelength_calibration.fitter.wavelength_fitter.WavelengthCalibrationFitter.__call__>`.
These weights will be multiplied by the residuals array prior to computing the final goodness-of-fit (chi-squared) value.

Internally, the goodness-of-fit value is just `np.sum(residuals**2)` or :math:`\chi^2 = \sum_i r_i^2`, where :math:`r`
is `residuals`. To achieve a standard weighted chi-squared (:math:`\chi^2 = \frac{\sum_i w_i r_i^2}{\sum_i w_i}`) the
input weights are "prepared" by taking the square-root of the normalized weights:

.. math::

    w'_i = \sqrt{
              \frac
                {w_{0,i}}
                {\sum_i w_{0,i}}
              }

where :math:`w_{0}` are the raw `weights` input to the
`WavelengthCalibrationFitter <solar_wavelength_calibration.fitter.wavelength_fitter.WavelengthCalibrationFitter.__call__>`.
The residual values are then multiplied by the prepared weights such that :math:`r_w = r w'`.
Thus, the goodness-of-fit value then becomes

.. math::

    \chi^2 = \sum_i r_{w,i}^2 = \frac{\sum_i w_{0,i} r_i^2}{\sum_i w_{0,i}},

which is equal to the standard weighted chi-squared. The "prepared" weights used during the fit are made available as
a property on the :ref:`FitResults <fit_results>` class.
