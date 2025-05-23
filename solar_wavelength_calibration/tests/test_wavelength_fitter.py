import astropy.units as u
import numpy as np
import pytest
from astropy.units import Quantity
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
from solar_wavelength_calibration.fitter.wavelength_fitter import calculate_linear_wave
from solar_wavelength_calibration.fitter.wavelength_fitter import combine_spectra
from solar_wavelength_calibration.fitter.wavelength_fitter import fitting_model
from solar_wavelength_calibration.fitter.wavelength_fitter import normalize_and_convolve
from solar_wavelength_calibration.fitter.wavelength_fitter import (
    resample_solar_transmission_with_doppler_shift,
)
from solar_wavelength_calibration.fitter.wavelength_fitter import (
    resample_telluric_transmission_with_absorption_correction,
)
from solar_wavelength_calibration.tests.conftest import test_incident_light_angle


def test_wavecal_fitter(
    parameters: WavelengthCalibrationParameters,
    observed_spectrum,
    atlas: AtlasBase,
    test_dispersion,
    test_parameters_dict,
    observed_wavelength_vector,
):
    """
    Given: The correct WavelengthCalibrationFitter inputs (data, models, and fit parameters)
    When: Initializing a WavelengthCalibrationFitter object with these inputs
    Then: Everything needed for the fit is correctly set up
    """
    fitter = WavelengthCalibrationFitter(
        input_parameters=parameters,
        atlas=atlas,
    )

    fit_result = fitter._run_fit(
        input_wavelength_vector=observed_wavelength_vector,
        input_spectrum=observed_spectrum,
    )

    # test that values have been fit (i.e. changed)
    assert fit_result.wavelength_parameters.dispersion != test_dispersion
    assert (
        fit_result.wavelength_parameters.crval
        != atlas.telluric_atlas_wavelength[np.size(observed_spectrum) // 2 + 1]
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
        "continuum_intensity_correction",
    ],
)
def test_fixing_parameters(
    fixed_parameter,
    test_parameters_dict,
    observed_spectrum,
    observed_continuum_level,
    atlas: AtlasBase,
    observed_wavelength_vector,
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
    }

    initialized_parameters = WavelengthCalibrationParameters(fit_flags=fit_flags, **init_values)

    fitter = WavelengthCalibrationFitter(
        input_parameters=initialized_parameters,
        atlas=atlas,
    )

    fit_result = fitter._run_fit(
        input_wavelength_vector=observed_wavelength_vector,
        input_spectrum=observed_spectrum,
    )
    for parameter in fit_result.minimizer_result.params:
        if parameter != fixed_parameter:
            continue

        if parameter == "continuum_intensity_correction":
            init_value = observed_continuum_level
        else:
            init_value = init_values[parameter]

        if isinstance(init_value, Quantity):
            init_value = init_value.value

        np.testing.assert_allclose(init_value, fit_result.minimizer_result.params[parameter])


def test_fit_stays_valid_with_fixed_crval(
    test_parameters_dict, observed_wavelength_vector, observed_spectrum, atlas: AtlasBase
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
        **init_values
    )

    fitter = WavelengthCalibrationFitter(
        input_parameters=parameters,
        atlas=atlas,
    )
    _ = fitter(
        input_wavelength_vector=observed_wavelength_vector,
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
    observed_wavelength_vector,
    atlas: AtlasBase,
    test_parameters_dict,
):
    """
    Given: A simulated unfit 1D spectrum with a known shift and dispersion scale.
    When: The WavelengthCalibrationFitter is used to fit the spectrum.
    Then: The fit parameters that parameterize the WCS header are correct
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
        input_wavelength_vector=observed_wavelength_vector,
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


def test_apply_telluric_absorption(parameters):
    """
    Given: A linear wavelength vector, a cropped atlas containing telluric data, and an opacity factor.
    When: The `resample_telluric_transmission_with_absorption_correction` function is called.
    Then: It returns a telluric transmission spectrum as a numpy array with non-negative values and the same length as the input wavelength vector.
    """
    linear_wave = np.linspace(1067.5, 1076.3, 100)
    cropped_atlas = LocalAtlas(
        solar_atlas_wavelength=np.linspace(1067.5, 1076.3, 100) * u.nm,
        solar_atlas_transmission=np.random.uniform(0.8, 1.0, 100),
        telluric_atlas_wavelength=np.linspace(1067.5, 1076.3, 100) * u.nm,
        telluric_atlas_transmission=np.random.uniform(0.8, 1.0, 100),
    )

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


def test_apply_doppler_shift(parameters):
    """
    Given: A linear wavelength vector, a cropped atlas containing solar data, a Doppler velocity, and a reference wavelength.
    When: The `resample_solar_transmission_with_doppler_shift` function is called.
    Then: It returns a solar transmission spectrum as a numpy array with non-negative values and the same length as the input wavelength vector.
    """
    linear_wave = np.linspace(1067.5, 1076.3, 100)
    crval = parameters.lmfit_parameters["crval"].value
    cropped_atlas = LocalAtlas(
        solar_atlas_wavelength=np.linspace(1067.5, 1076.3, 100) * u.nm,
        solar_atlas_transmission=np.random.uniform(0.8, 1.0, 100),
        telluric_atlas_wavelength=np.linspace(1067.5, 1076.3, 100) * u.nm,
        telluric_atlas_transmission=np.random.uniform(0.8, 1.0, 100),
    )

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


def test_combine_spectra():
    """
    Given: Two transmission spectra (telluric and solar) and a straylight fraction.
    When: The `combine_spectra` function is called.
    Then: It returns a combined spectrum as a numpy array with non-negative values and the same length as the input spectra.
    """
    telluric_transmission = np.linspace(0.8, 1.0, 100)
    solar_transmission = np.linspace(0.9, 1.0, 100)

    # Test with valid straylight fraction
    straylight_fraction = 0.1
    combined_spectrum = combine_spectra(
        telluric_transmission, solar_transmission, straylight_fraction
    )
    assert isinstance(combined_spectrum, np.ndarray)
    assert len(combined_spectrum) == len(telluric_transmission)
    assert np.all(combined_spectrum >= 0)

    # Test with extreme straylight fractions
    for straylight_fraction in [0, 1]:
        combined_spectrum = combine_spectra(
            telluric_transmission, solar_transmission, straylight_fraction
        )
        assert isinstance(combined_spectrum, np.ndarray)
        assert len(combined_spectrum) == len(telluric_transmission)
        assert np.all(combined_spectrum >= 0)


def test_normalize_and_convolve(parameters):
    """
    Given: A spectrum, continuum level, reference wavelength, resolving power, and dispersion.
    When: The `normalize_and_convolve` function is called.
    Then: It returns a normalized and convolved spectrum as a numpy array with non-negative values and the same length as the input spectrum.
    """
    spectrum = np.linspace(0.8, 1.0, 100)
    continuum_level = 1.0
    crval = parameters.lmfit_parameters["crval"].value
    resolving_power = parameters.lmfit_parameters["resolving_power"].value
    dispersion = parameters.lmfit_parameters["dispersion"].value

    normalized_spectrum = normalize_and_convolve(
        spectrum, continuum_level, crval, resolving_power, dispersion
    )

    assert isinstance(normalized_spectrum, np.ndarray)
    assert len(normalized_spectrum) == len(spectrum)
    assert np.all(normalized_spectrum >= 0)


def test_fitting_model(parameters, atlas, observed_spectrum):
    """
    Given: A set of input parameters, an observed spectrum, and a cropped atlas.
    When: The `fitting_model` function is called with these inputs.
    Then: It returns the residual amplitude as a numpy array with the same length as the observed spectrum.
    """
    params = parameters.lmfit_parameters
    input_spectrum = observed_spectrum
    cropped_atlas = LocalAtlas(
        solar_atlas_wavelength=np.linspace(1067.5, 1076.3, 100) * u.nm,
        solar_atlas_transmission=np.random.uniform(0.8, 1.0, 100),
        telluric_atlas_wavelength=np.linspace(1067.5, 1076.3, 100) * u.nm,
        telluric_atlas_transmission=np.random.uniform(0.8, 1.0, 100),
    )
    number_of_wave_pix = len(input_spectrum)
    grating_constant = parameters.constant_parameters["grating_constant"]
    order = parameters.constant_parameters["order"]
    doppler_velocity = parameters.constant_parameters["doppler_velocity"]

    residuals = fitting_model(
        params,
        input_spectrum,
        cropped_atlas,
        number_of_wave_pix,
        grating_constant,
        order,
        doppler_velocity,
    )

    assert isinstance(residuals, np.ndarray)
    assert len(residuals) == len(input_spectrum)
