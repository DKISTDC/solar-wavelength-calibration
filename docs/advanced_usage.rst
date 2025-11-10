Advanced Usage
==============

.. _custom_continuum_function:

Custom Continuum Parameterization
---------------------------------

Sometimes raw data have continuum variations that do not match the atlas continuum (for example, from optical effects
in the spectrograph) and for these cases this library allows a user to fit a completely custom continuum as a function of
wavelength simultaneously with the atlas fit. Here's how to do it.

`~solar_wavelength_calibration.fitter.parameters.WavelengthCalibrationParameters` contains all the hooks needed to define
and fit a custom continuum function, but you need to create a subclass to access them. The subclass needs to modify
two things: `~solar_wavelength_calibration.fitter.parameters.WavelengthCalibrationParameters.continuum_function` and
`~solar_wavelength_calibration.fitter.parameters.WavelengthCalibrationParameters.lmfit_parameters`

Custom `~solar_wavelength_calibration.fitter.parameters.WavelengthCalibrationParameters.continuum_function`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Choose how you want to parameterize your continuum. This is entirely up to you! Once you have a parameterization, write
a python function that computes a continuum array given the wavelength vector and fit parameters. The first two arguments
of this function *must* be the wavelength vector of the current fit iteration and a `~lmfit.parameter.Parameters` object
containing the current value of all fit parameters, in that order. All of the fit parameters for the current iteration
can be accessed via `parameters["PARAM_NAME"].value`. The continuum function also *must* return something that can be
multiplied by the combined atlas spectrum. The `~solar_wavelength_calibration.fitter.parameters.WavelengthCalibrationParameters.continuum_function`
property needs to return this custom function.

For example,

.. code-block:: python

    def my_great_continuum_function(wavelength: np.ndarray,
                                    parameters: Parameters) -> np.ndarray:

        value = parameters["new_continuum_parameter"].value
        ...

    class MyWavelengthCalibrationParameters(WavelengthCalibrationParameters):

        @property
        def continuum_function(self) -> Callable[[np.ndarray, Parameters], np.ndarray]:
            return my_great_continuum_function

You can also define the function directly in the body of the `continuum_function` property (as we'll see later). Just
make sure that this property returns the function itself, not the value of the function.

.. note::

    Even if your continuum doesn't depend on wavelength it still needs to accept the wavelength as the first parameter.

Custom `~solar_wavelength_calibration.fitter.parameters.WavelengthCalibrationParameters.lmfit_parameters`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `~solar_wavelength_calibration.fitter.parameters.WavelengthCalibrationParameters.lmfit_parameters` property is where
we define what parameters are sent to the minimization algorithm. If a parameter should be varied during the fit then
it needs to be represented in this property. When adding any new parameter, we need to make sure we also retain all
of the parameters that are provided by default, which can be accessed with `super().lmfit_parameters`. For example

.. code-block:: python

    class MyWavelengthCalibrationParameters(WavelengthCalibrationParameters):

        new_continuum_parameter: float

        @property
        def lmfit_parameters(self) -> Parameters:
            # Get the base parameters
            base_parameters = super().lmfit_parameters

            # Add a new parameter
            base_parameters.add("new_continuum_parameter",
                                value=self.new_continuum_parameter,
                                vary=True,
                                min=2.0,
                                max=10)

            return base_parameters

Once this is done, your continuum function will have access to `new_continuum_parameter` in its `parameters` argument.

Note in this example we have added `new_continuum_parameter` to the model fields of `MyWavelengthCalibrationParameters`
and used that value as the starting value for this parameter. This pattern is not strictly necessary, but does allow
the initial guess to be set on a per-instance basis of `MyWavelengthCalibrationParameters`, just like all of the base
parameters.

Continuum Functions With Extra Arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`~solar_wavelength_calibration.fitter.parameters.WavelengthCalibrationParameters.lmfit_parameters` should only contain
parameters that can vary during a fit. If your continuum function has extra arguments that are constant for a fit the
recommended pattern is to add these arguments as model fields on your subclass so they can be accessed in the definition
of `continuum_function`. For example

.. code-block:: python

    class MyWavelengthCalibrationParameters(WavelengthCalibrationParameters):

        new_continuum_parameter: float
        # Define an extra argument
        extra_continuum_arg: int

        @property
        def continuum_function(self) -> Callable[[np.ndarray, Parameters], np.ndarray]:

            def my_great_continuum_function(wavelength: np.ndarray,
                                            parameters: Parameters) -> np.ndarray:

                # Use the extra arg in the continuum function definition
                new_thing = self.extra_continuum_arg * wavelength
                fit_value = parameters["new_continuum_parameter"].value
                ...


        # Note that extra_continuum_arg is not used here
        @property
        def lmfit_parameters(self) -> Parameters:
            # Get the base parameters
            base_parameters = super().lmfit_parameters

            # Add a new parameter
            base_parameters.add("new_continuum_parameter",
                                value=self.new_continuum_parameter,
                                vary=True,
                                min=2.0,
                                max=10)

            return base_parameters

Full Example: Fit Continuum With a Polynomial
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example shows all of the concepts mentioned above in a realistic way. We're going to define our continuum as a
simply polynomial, but we're going to let the polynomial order be something we can set during instantiation so we can
easily experiment with different fit orders. Note the the polynomial order is *not* a fit parameter, but it is used
to construct the fit parameters (the actual polynomial coefficients).

.. code-block:: python

    class WavelengthCalibrationParametersWithContinuum(WavelengthCalibrationParameters):

        # Neither of these are actually fit parameters
        continuum_poly_fit_order: int
        normalized_abscissa: np.ndarray

        @property
        def continuum_function(self) -> Callable[[np.ndarray, Parameters], np.ndarray]:

            def polynomial_continuum(wavelength: np.ndarray,
                                     fit_parameters: Parameters) -> np.ndarray:
                # Notice that this function does NOT depend on wavelength:
                # by using the normalized abscissa we can be wavelength independent.

                # Note self.continuum_poly_fit_order and self.normalized_abscissa being used here
                coeffs = [
                    fit_parameters[f"poly_coeff_{i:02n}"].value
                    for i in range(self.continuum_poly_fit_order + 1)
                    ]
                return np.polyval(coeffs, self.normalized_abscissa)

            return polynomial_continuum

        @property
        def lmfit_parameters(self) -> Parameters:
            params = super().lmfit_parameters
            for o in range(self.continuum_poly_fit_order + 1):
                params.add(
                    f"poly_coeff_{o:02n}",
                    vary=True,
                    value=0,
                    min=-1,
                    max=1,
                )

            # This isn't strictly necessary, but it does ensure we don't accidentally
            # forget to turn off fitting of this parameter that is no longer used.
            del params["continuum_level"]

            return params

This class could be used in a fit like this

.. code-block:: python

    wavecal_params = WavelengthCalibrationParametersWithContinuum(
            continuum_poly_fit_order=3,
            normalized_abscissa=np.linspace(-1, 1, input_spec.size),
            crval=630 * u.nm,
            ... # Add the base parameters here
            )

    fitter = WavelengthCalibrationFitter(input_parameters=wavecal_params, atlas=atlas)

    results = fitter(input_spectrum=input_spec)

