from typing import Callable

import astropy.units as u
import numpy as np
import pytest
from astropy.units import Quantity
from astropy.wcs import WCS
from lmfit import Parameters
from lmfit.minimizer import MinimizerResult
from pydantic import ValidationError

from solar_wavelength_calibration import AngleBoundRange
from solar_wavelength_calibration import BoundsModel
from solar_wavelength_calibration import DispersionBoundRange
from solar_wavelength_calibration import FitFlagsModel
from solar_wavelength_calibration import LengthBoundRange
from solar_wavelength_calibration import UnitlessBoundRange
from solar_wavelength_calibration import WavelengthCalibrationFitter
from solar_wavelength_calibration import WavelengthCalibrationParameters
from solar_wavelength_calibration.atlas.atlas import LocalAtlas
from solar_wavelength_calibration.atlas.base import AtlasBase
from solar_wavelength_calibration.fitter.wavelength_fitter import WavelengthParameters
from solar_wavelength_calibration.fitter.wavelength_fitter import apply_resolving_power
from solar_wavelength_calibration.fitter.wavelength_fitter import calculate_initial_crval_guess
from solar_wavelength_calibration.fitter.wavelength_fitter import calculate_linear_wave
from solar_wavelength_calibration.fitter.wavelength_fitter import fitting_model
from solar_wavelength_calibration.fitter.wavelength_fitter import (
    resample_solar_transmission_with_doppler_shift,
)
from solar_wavelength_calibration.fitter.wavelength_fitter import (
    resample_telluric_transmission_with_absorption_correction,
)
from solar_wavelength_calibration.fitter.wavelength_fitter import scalar_continuum_function
from solar_wavelength_calibration.tests.conftest import test_incident_light_angle


@pytest.fixture
def synthetic_local_atlas():
    def _make(n=100):
        return LocalAtlas(
            solar_atlas_wavelength=np.linspace(1067.5, 1076.3, n) * u.nm,
            solar_atlas_transmission=np.random.uniform(0.8, 1.0, n),
            telluric_atlas_wavelength=np.linspace(1067.5, 1076.3, n) * u.nm,
            telluric_atlas_transmission=np.random.uniform(0.8, 1.0, n),
        )

    return _make


@pytest.fixture
def fitting_model_setup(parameters, observed_spectrum, synthetic_local_atlas):
    n = len(observed_spectrum)
    cropped_atlas = synthetic_local_atlas(n)
    params = parameters.lmfit_parameters
    grating_constant = parameters.constant_parameters["grating_constant"]
    order = parameters.constant_parameters["order"]
    doppler_velocity = parameters.constant_parameters["doppler_velocity"]
    return {
        "params": params,
        "input_spectrum": observed_spectrum,
        "cropped_atlas": cropped_atlas,
        "number_of_wave_pix": n,
        "grating_constant": grating_constant,
        "order": order,
        "doppler_velocity": doppler_velocity,
    }


@pytest.fixture
def wavecal_params_with_broken_continuum_func():
    # AKA an example of how a user might add a custom continuum function
    class NewWaveCalParams(WavelengthCalibrationParameters):

        continuum_exponent: float

        @property
        def continuum_function(self) -> Callable[[np.ndarray, Parameters, ...], np.ndarray]:
            def user_defined_continuum_function(input_wave, parameters):
                raise FileNotFoundError("Not sure why I was even looking for a file")

            return user_defined_continuum_function

        @property
        def lmfit_parameters(self):
            params = super().lmfit_parameters
            params.add("continuum_exponent", value=self.continuum_exponent, vary=True)
            return params

    return NewWaveCalParams


def test_wavecal_fitter(
    parameters: WavelengthCalibrationParameters,
    observed_spectrum,
    atlas: AtlasBase,
    test_dispersion,
    test_parameters_dict,
):
    """
    Given: The correct WavelengthCalibrationFitter inputs (data, models, and fit parameters)
    When: Initializing a WavelengthCalibrationFitter object with these inputs and running a fit
    Then: The fit runs and values are changed from their starting positions
    """
    fitter = WavelengthCalibrationFitter(
        input_parameters=parameters,
        atlas=atlas,
    )

    fit_result = fitter(input_spectrum=observed_spectrum)

    # test that values have been fit (i.e. changed)
    assert fit_result.wavelength_parameters.dispersion != test_dispersion
    assert (
        fit_result.wavelength_parameters.crval
        != atlas.telluric_atlas_wavelength[np.size(observed_spectrum) // 2 + 1]
    )


def test_calling_wavecal_fitter_bad_weights(
    parameters: WavelengthCalibrationParameters,
    observed_spectrum,
    atlas: AtlasBase,
    test_dispersion,
    test_parameters_dict,
    observed_wavelength_vector,
):
    """
    Given: The correct WavelengthCalibrationFitter inputs (data, models, and fit parameters)
    When: Initializing a WavelengthCalibrationFitter object with these inputs and calling the fit with wrong-sized weights
    Then: The correct error is raised
    """
    fitter = WavelengthCalibrationFitter(
        input_parameters=parameters,
        atlas=atlas,
    )

    with pytest.raises(ValueError, match="Length of spectral_weights"):
        fit_result = fitter(
            input_wavelength_vector=observed_wavelength_vector,
            input_spectrum=observed_spectrum,
            spectral_weights=np.random.random(observed_spectrum.size + 5),
        )


@pytest.mark.parametrize(
    "with_weights", [pytest.param(True, id="with_weights"), pytest.param(False, id="no_weights")]
)
def test_fit_results(
    parameters: WavelengthCalibrationParameters,
    observed_spectrum,
    atlas: AtlasBase,
    test_dispersion,
    test_parameters_dict,
    observed_wavelength_vector,
    observed_continuum_level,
    with_weights,
):
    """
    Given: The correct `WavelengthCalibrationFitter` inputs (data, models, and fit parameters)
    When: Producing a `FitResult` object by fitting
    Then: The properties of the `FitResult` object have the types we expect and the computed properties are calculated
           correctly.
    """
    fitter = WavelengthCalibrationFitter(
        input_parameters=parameters,
        atlas=atlas,
    )

    weights = np.random.random(observed_spectrum.size) if with_weights else None
    fit_result = fitter(
        input_spectrum=observed_spectrum,
        spectral_weights=weights,
    )

    assert isinstance(fit_result.wavelength_parameters, WavelengthParameters)
    assert isinstance(fit_result.wavelength_parameters.to_header(1), dict)
    assert isinstance(fit_result.minimizer_result, MinimizerResult)
    assert fit_result.num_wave_pix == observed_spectrum.size

    np.testing.assert_array_equal(fit_result.input_wavelength_vector, observed_wavelength_vector)
    np.testing.assert_array_equal(fit_result.input_spectrum, observed_spectrum)

    weight_data = weights if with_weights else np.ones_like(observed_spectrum)
    expected_prepared_weights = np.sqrt(weight_data / np.sum(weight_data))
    expected_best_fit_atlas = (
        observed_spectrum - fit_result.minimizer_result.residual / expected_prepared_weights
    )
    wcs = WCS(fit_result.wavelength_parameters.to_header(axis_num=1))
    expected_best_fit_wave = wcs.spectral.pixel_to_world(
        np.arange(observed_spectrum.size)
    ).to_value(u.nm)
    np.testing.assert_array_equal(fit_result.prepared_weights, expected_prepared_weights)
    np.testing.assert_almost_equal(
        fit_result.best_fit_continuum, observed_continuum_level, decimal=2
    )
    np.testing.assert_array_equal(fit_result.best_fit_wavelength_vector, expected_best_fit_wave)
    np.testing.assert_almost_equal(fit_result.best_fit_atlas, expected_best_fit_atlas)


def test_fit_with_custom_continuum_function(
    wavecal_params_with_extra_continuum_param,
    parameters: WavelengthCalibrationParameters,
    observed_spectrum,
    atlas: AtlasBase,
    test_dispersion,
    test_parameters_dict,
):
    """
    Given: A `WavelengthCalibrationFitter` with sanely populated data and parameters
    When: Calling the fitter with a custom continuum function
    Then: No errors occur
    """
    starting_value = 1.0
    new_parameters = wavecal_params_with_extra_continuum_param(
        continuum_exponent=starting_value, not_used=True, **parameters.model_dump()
    )

    fitter = WavelengthCalibrationFitter(
        input_parameters=new_parameters,
        atlas=atlas,
    )

    fit_result = fitter(
        input_spectrum=observed_spectrum,
    )
    assert "continuum_exponent" in fit_result.minimizer_result.params
    assert fit_result.minimizer_result.params["continuum_exponent"] != starting_value


def test_fit_with_custom_continuum_function_error(
    wavecal_params_with_broken_continuum_func,
    parameters: WavelengthCalibrationParameters,
    observed_spectrum,
    atlas: AtlasBase,
    test_dispersion,
    test_parameters_dict,
):
    """
    Given: A `WavelengthCalibrationFitter` with sanely populated data and parameters
    When: Calling the fitter with a custom continuum function that hits an Error
    Then: The error in the continuum function is correctly reported
    """
    new_parameters = wavecal_params_with_broken_continuum_func(
        continuum_exponent=1.0, not_used=True, **parameters.model_dump()
    )

    fitter = WavelengthCalibrationFitter(
        input_parameters=new_parameters,
        atlas=atlas,
    )

    with pytest.raises(
        RuntimeError, match="The continuum function has produced an unrecoverable error."
    ):
        fitter(
            input_spectrum=observed_spectrum,
        )


def test_incorrect_atlas(
    parameters: WavelengthCalibrationParameters,
    observed_spectrum,
    observed_wavelength_vector,
):
    """
    Given: An incorrect atlas (i.e. a random array of numbers)
    When: Initializing a WavelengthCalibrationFitter object
    Then: Raise a ValidationError
    """
    atlas = np.linspace(1000, 3000, 1000) * u.nm
    with pytest.raises(ValidationError, match="Input should be an instance of AtlasBase"):
        fitter = WavelengthCalibrationFitter(
            input_parameters=parameters,
            atlas=atlas,
        )


@pytest.mark.parametrize(
    "fixed_parameter",
    [
        "crval",
        "dispersion",
        "incident_light_angle",
        "resolving_power",
        "opacity_factor",
        "straylight_fraction",
        "continuum_level",
    ],
)
def test_fixing_parameters(
    fixed_parameter,
    test_parameters_dict,
    observed_wavelength_vector,
    observed_spectrum,
    observed_continuum_level,
    atlas: AtlasBase,
):
    """
    Given: A `WavelengthCalibrationFitter` initialized with a `WavelengthCalibrationParameters` with a single fixed parameter
    When: Running the fit
    Then: The "fit" value of the fixed parameter is the same as its initial value
    """
    fit_flag_args = {fixed_parameter: False}
    fit_flags = FitFlagsModel(**fit_flag_args)

    # Have starting values that are off of the true values so we can make sure free parameters change
    init_values = test_parameters_dict | {
        "crval": test_parameters_dict["crval"] + 3 * u.nm,
        "dispersion": test_parameters_dict["dispersion"] * 1.01,
        "incident_light_angle": test_parameters_dict["incident_light_angle"] + 4 * u.deg,
        "resolving_power": test_parameters_dict["resolving_power"] + 10,
        "opacity_factor": test_parameters_dict["opacity_factor"] + 0.1,
        "straylight_fraction": test_parameters_dict["straylight_fraction"] + 0.2,
        "continuum_level": observed_continuum_level * 0.98,
    }

    initialized_parameters = WavelengthCalibrationParameters(fit_flags=fit_flags, **init_values)

    fitter = WavelengthCalibrationFitter(
        input_parameters=initialized_parameters,
        atlas=atlas,
    )

    # We need to enter via the `._fit_spectrum` method because the init values will make an input_wavelength_vector
    # so wack it breaks old versions of numpy for some reason???
    prepared_atlas = fitter._prepare_atlas(input_wavelength_vector=observed_wavelength_vector)
    fit_result = fitter._fit_spectrum(
        cropped_atlas=prepared_atlas,
        input_wavelength_vector=observed_wavelength_vector,
        input_spectrum=observed_spectrum,
        method="leastsq",
        spectral_weights=np.ones_like(observed_spectrum),
    )
    for parameter in fit_result.minimizer_result.params:
        if parameter != fixed_parameter:
            continue

        init_value = init_values[parameter]

        if isinstance(init_value, Quantity):
            init_value = init_value.value

        np.testing.assert_allclose(init_value, fit_result.minimizer_result.params[parameter])


def test_fit_stays_valid_with_fixed_crval(
    test_parameters_dict, observed_spectrum, atlas: AtlasBase
):
    """
    Given: A set of input parameters that is on the edge of WCS validity and bounds on `incident_light_angle` that
      would take the fit to invalid WCS values
    When: Running the fit with `crval` fixed
    Then: The fit stays within valid ranges
    """

    input_crval = test_parameters_dict["crval"] + 10 * u.nm
    edge_angle = (
        np.rad2deg(
            np.arcsin(
                float(
                    input_crval
                    * test_parameters_dict["grating_constant"]
                    * test_parameters_dict["order"]
                )
                - 0.999999999
            )
        )
        * u.deg
    )

    # Heavily bias limits towards invalidity
    angle_min = edge_angle - 40.0 * u.deg
    angle_max = edge_angle + 0.01 * u.deg

    init_values = test_parameters_dict | {
        "crval": input_crval,
        "dispersion": test_parameters_dict["dispersion"] * 1.01,
        "incident_light_angle": edge_angle,
        "resolving_power": test_parameters_dict["resolving_power"] + 10,
        "opacity_factor": test_parameters_dict["opacity_factor"] + 0.1,
        "straylight_fraction": test_parameters_dict["straylight_fraction"] + 0.2,
    }

    parameters = WavelengthCalibrationParameters(
        fit_flags=FitFlagsModel(crval=False),
        bounds=BoundsModel(incident_light_angle=AngleBoundRange(min=angle_min, max=angle_max)),
        **init_values,
    )

    fitter = WavelengthCalibrationFitter(
        input_parameters=parameters,
        atlas=atlas,
    )
    _ = fitter(
        input_spectrum=observed_spectrum,
    )


def test_fit_is_correct(
    test_crval,
    test_dispersion,
    test_incident_light_angle,
    test_grating_constant,
    test_doppler_velocity,
    test_order,
    observed_spectrum,
    atlas: AtlasBase,
    test_parameters_dict,
):
    """
    Given: A simulated unfit 1D spectrum with a known shift and dispersion scale.
    When: The WavelengthCalibrationFitter is used to fit the spectrum.
    Then: The fit parameters that parametrize the WCS header are correct
    """

    parameters = WavelengthCalibrationParameters(
        crval=test_crval + 0.5 * u.nm,
        dispersion=test_dispersion * 1.01,
        incident_light_angle=test_incident_light_angle + 0.2 * u.deg,
        resolving_power=725001,
        opacity_factor=1,
        straylight_fraction=0.0,
        grating_constant=test_grating_constant,
        doppler_velocity=0 * u.km / u.s,
        order=test_order,
        bounds=BoundsModel(
            crval=LengthBoundRange(min=test_crval - 1 * u.nm, max=test_crval + 1 * u.nm),
            incident_light_angle=AngleBoundRange(
                min=test_incident_light_angle - 1 * u.deg, max=test_incident_light_angle + 1 * u.deg
            ),
            dispersion=DispersionBoundRange(
                min=test_dispersion - 0.001 * u.nm / u.pix,
                max=test_dispersion + 0.001 * u.nm / u.pix,
            ),
            resolving_power=UnitlessBoundRange(min=725000, max=725002),
            opacity_factor=UnitlessBoundRange(min=0.8, max=1),
            straylight_fraction=UnitlessBoundRange(min=0, max=0.2),
            continuum_level=UnitlessBoundRange(min=0.9, max=1),
        ),
    )

    fitter = WavelengthCalibrationFitter(
        input_parameters=parameters,
        atlas=atlas,
    )
    fit_result = fitter(
        input_spectrum=observed_spectrum,
        method="nelder-meade",
    )
    np.testing.assert_allclose(
        fit_result.wavelength_parameters.crval, test_crval.to_value(u.nm), atol=1, rtol=1e-4
    )
    np.testing.assert_allclose(
        fit_result.wavelength_parameters.dispersion,
        test_dispersion.to_value(u.nm / u.pix),
        atol=1e-5,
        rtol=1e-1,
    )

    fit_incident_light_angle = fit_result.wavelength_parameters.incident_light_angle
    # Incident light angle is pretty weakly constrained for our test setup so let's say we pass if we get within a
    # degree.
    assert abs(fit_incident_light_angle - test_incident_light_angle.to_value(u.deg)) < 0.5


def test_prepare_atlas(
    parameters: WavelengthCalibrationParameters,
    observed_spectrum,
    atlas: AtlasBase,
    observed_wavelength_vector,
):
    """
    Given: The correct WavelengthCalibrationFitter inputs (data, models, and fit parameters)
    When: Preparing the atlas for the fit
    Then: Make sure that a sample of the atlas is taken, and make sure that it is the correct sample
    """
    fitter = WavelengthCalibrationFitter(
        input_parameters=parameters,
        atlas=atlas,
    )
    cropped_atlas = fitter._prepare_atlas(input_wavelength_vector=observed_wavelength_vector)

    # test that we took out a chunk of the atlas
    assert len(cropped_atlas.solar_atlas_transmission) < len(atlas.solar_atlas_transmission)
    assert len(cropped_atlas.telluric_atlas_transmission) < len(atlas.telluric_atlas_transmission)

    # test that we took out the correct chunk for each atlas (i.e. mean of input wavelength vector is contained in fts_[solar/telluric]wavelength)
    assert (
        cropped_atlas.solar_atlas_wavelength.to_value(u.nm)[0]
        < np.mean(observed_wavelength_vector).value
        < cropped_atlas.solar_atlas_wavelength.to_value(u.nm)[
            len(cropped_atlas.solar_atlas_wavelength.to_value(u.nm)) - 1
        ]
    )

    assert (
        cropped_atlas.telluric_atlas_wavelength.to_value(u.nm)[0]
        < np.mean(observed_wavelength_vector).value
        < cropped_atlas.telluric_atlas_wavelength.to_value(u.nm)[
            len(cropped_atlas.telluric_atlas_wavelength.to_value(u.nm)) - 1
        ]
    )


def test_calculate_linear_wave(parameters):
    """
    Given: A set of valid parameters and a number of wavelength pixels.
    When: The `calculate_linear_wave` function is called.
    Then: It returns a linear wavelength vector as a numpy array with the correct length and positive values.
    """
    params = parameters.lmfit_parameters
    grating_constant = parameters.constant_parameters["grating_constant"]
    order = parameters.constant_parameters["order"]

    # Test with valid input
    number_of_wave_pix = 100
    linear_wave = calculate_linear_wave(params, number_of_wave_pix, grating_constant, order)
    assert isinstance(linear_wave, np.ndarray)
    assert len(linear_wave) == number_of_wave_pix
    assert np.all(linear_wave > 0)

    # Test with edge-case input
    number_of_wave_pix = 0
    linear_wave = calculate_linear_wave(params, number_of_wave_pix, grating_constant, order)
    assert isinstance(linear_wave, np.ndarray)
    assert len(linear_wave) == 0


@pytest.mark.usefixtures("synthetic_local_atlas")
def test_apply_telluric_absorption(parameters, synthetic_local_atlas):
    """
    Given: A linear wavelength vector, a cropped atlas containing telluric data, and an opacity factor.
    When: The `resample_telluric_transmission_with_absorption_correction` function is called.
    Then: It returns a telluric transmission spectrum as a numpy array with non-negative values and the same length as the input wavelength vector.
    """
    linear_wave = np.linspace(1067.5, 1076.3, 100)
    cropped_atlas = synthetic_local_atlas(100)

    # Test with valid opacity factor
    opacity_factor = parameters.lmfit_parameters["opacity_factor"].value
    telluric_transmission = resample_telluric_transmission_with_absorption_correction(
        linear_wave, cropped_atlas, opacity_factor
    )
    assert isinstance(telluric_transmission, np.ndarray)
    assert len(telluric_transmission) == len(linear_wave)
    assert np.all(telluric_transmission >= 0)

    # Test with extreme opacity factors
    for opacity_factor in [0, 1e6]:
        telluric_transmission = resample_telluric_transmission_with_absorption_correction(
            linear_wave, cropped_atlas, opacity_factor
        )
        assert isinstance(telluric_transmission, np.ndarray)
        assert len(telluric_transmission) == len(linear_wave)
        assert np.all(telluric_transmission >= 0)


@pytest.mark.usefixtures("synthetic_local_atlas")
def test_apply_doppler_shift(parameters, synthetic_local_atlas):
    """
    Given: A linear wavelength vector, a cropped atlas containing solar data, a Doppler velocity, and a reference wavelength.
    When: The `resample_solar_transmission_with_doppler_shift` function is called.
    Then: It returns a solar transmission spectrum as a numpy array with non-negative values and the same length as the input wavelength vector.
    """
    linear_wave = np.linspace(1067.5, 1076.3, 100)
    crval = parameters.lmfit_parameters["crval"].value
    cropped_atlas = synthetic_local_atlas(100)

    # Test with valid doppler velocity
    doppler_velocity = parameters.constant_parameters["doppler_velocity"]
    solar_transmission = resample_solar_transmission_with_doppler_shift(
        linear_wave, cropped_atlas, doppler_velocity, crval
    )
    assert isinstance(solar_transmission, np.ndarray)
    assert len(solar_transmission) == len(linear_wave)
    assert np.all(solar_transmission >= 0)

    # Test with edge-case doppler velocities
    for doppler_velocity in [0, 1e6, -1e6] * u.km / u.s:
        solar_transmission = resample_solar_transmission_with_doppler_shift(
            linear_wave, cropped_atlas, doppler_velocity, crval
        )
        assert isinstance(solar_transmission, np.ndarray)
        assert len(solar_transmission) == len(linear_wave)
        assert np.all(solar_transmission >= 0)


def test_apply_resolving_power(parameters):
    """
    Given: A spectrum, reference wavelength, resolving power, and dispersion.
    When: The `apply_resolving_power` function is called.
    Then: It returns a convolved spectrum as a numpy array with the same length as the input spectrum.
    """
    spectrum = np.linspace(0.8, 1.0, 100)
    crval = parameters.lmfit_parameters["crval"].value
    resolving_power = parameters.lmfit_parameters["resolving_power"].value
    dispersion = parameters.lmfit_parameters["dispersion"].value

    adjusted_spectrum = apply_resolving_power(spectrum, crval, resolving_power, dispersion)

    assert isinstance(adjusted_spectrum, np.ndarray)
    assert len(adjusted_spectrum) == len(spectrum)


def test_fitting_model(fitting_model_setup):
    """
    Given: A set of input parameters, an observed spectrum, and a cropped atlas.
    When: The `fitting_model` function is called with these inputs.
    Then: It returns the residual amplitude as a numpy array with the same length as the observed spectrum.
    """
    setup = fitting_model_setup
    weights = np.ones(setup["number_of_wave_pix"])
    residuals = fitting_model(
        setup["params"],
        setup["input_spectrum"],
        setup["cropped_atlas"],
        setup["number_of_wave_pix"],
        setup["grating_constant"],
        setup["order"],
        setup["doppler_velocity"],
        weights,
        continuum_function=scalar_continuum_function,
    )
    assert isinstance(residuals, np.ndarray)
    assert len(residuals) == len(setup["input_spectrum"])


def test_fitting_model_with_weights(fitting_model_setup):
    """
    Given: A set of input parameters and a synthetic spectrum
    When: The `fitting_model` is called with and without spectral weights
    Then: The residuals are scaled by the weights
    """
    setup = fitting_model_setup
    params = setup["params"]
    observed_spectrum = setup["input_spectrum"]
    cropped_atlas = setup["cropped_atlas"]
    number_of_wave_pix = setup["number_of_wave_pix"]
    grating_constant = setup["grating_constant"]
    order = setup["order"]
    doppler_velocity = setup["doppler_velocity"]

    spectral_weights = np.ones(len(observed_spectrum))
    normalized_weights = spectral_weights / np.sum(spectral_weights)
    prepared_weights = np.sqrt(normalized_weights)

    # Unweighted: weights are all 1, so residuals are unchanged
    residuals_unweighted = fitting_model(
        params,
        observed_spectrum,
        cropped_atlas,
        number_of_wave_pix,
        grating_constant,
        order,
        doppler_velocity,
        prepared_weights=prepared_weights,
        continuum_function=scalar_continuum_function,
    )

    # Undo the weighting caused by the default weights
    residuals_unweighted /= np.sqrt(1.0 / observed_spectrum.size)

    weights = np.linspace(0.5, 2.0, number_of_wave_pix)
    normalized_weights = weights / np.sum(weights)
    prepared_weights = np.sqrt(normalized_weights)

    residuals_weighted = fitting_model(
        params,
        observed_spectrum,
        cropped_atlas,
        number_of_wave_pix,
        grating_constant,
        order,
        doppler_velocity,
        prepared_weights=prepared_weights,
        continuum_function=scalar_continuum_function,
    )

    weighted_chisq = np.sum(residuals_weighted**2)
    np.testing.assert_allclose(
        weighted_chisq, np.sum((residuals_unweighted * np.sqrt(weights / np.sum(weights))) ** 2)
    )


def test_calculate_initial_crval_guess_returns_reasonable_value(
    atlas: AtlasBase,
    observed_spectrum: np.ndarray,
    observed_wavelength_vector: Quantity,
):
    """
    Given: A realistic observed spectrum and wavelength vector based on a shifted solar atlas,
           and a full atlas instance.
    When: The `calculate_initial_crval_guess` function is called.
    Then: The estimated CRVAL is within the range of the input wavelengths,
          and is greater than the center of the observed vector, consistent with shift correction.
    """
    estimated_crval = calculate_initial_crval_guess(
        input_wavelength_vector=observed_wavelength_vector,
        input_spectrum=observed_spectrum,
        atlas=atlas,
    )

    # Ensure output is a Quantity with correct unit
    assert isinstance(estimated_crval, Quantity)
    assert estimated_crval.unit.is_equivalent(u.nm)

    # Ensure estimated CRVAL lies within the input wavelength range
    assert observed_wavelength_vector.min() < estimated_crval < observed_wavelength_vector.max()

    # Check that estimated CRVAL is larger than the midpoint, which would correct the known synthetic shift
    midpoint_wave = observed_wavelength_vector[len(observed_wavelength_vector) // 2]
    assert estimated_crval > midpoint_wave


def test_to_header_add_alternate_keys(
    parameters: WavelengthCalibrationParameters,
    observed_spectrum,
    atlas: AtlasBase,
):

    fitter = WavelengthCalibrationFitter(
        input_parameters=parameters,
        atlas=atlas,
    )

    fit_result = fitter._run_fit(
        input_spectrum=observed_spectrum,
        spectral_weights=np.ones(len(observed_spectrum)),
    )

    # With add_alternate_keys=True
    header = fit_result.wavelength_parameters.to_header(axis_num=1, add_alternate_keys=True)
    # Check standard keys
    standard_keys = ["CTYPE1", "CUNIT1", "CRPIX1", "CRVAL1", "CDELT1", "PV1_0", "PV1_1", "PV1_2"]
    alternate_keys = [f"{key}A" for key in standard_keys]

    for key in standard_keys:
        assert key in header
    for key in alternate_keys:
        assert key in header
    for key in standard_keys:
        if f"{key}A" in header:
            assert header[key] == header[f"{key}A"]

    # With add_alternate_keys=False
    header_no_alt = fit_result.wavelength_parameters.to_header(axis_num=1)
    for key in standard_keys:
        assert key in header
    for key in alternate_keys:
        assert key not in header_no_alt
