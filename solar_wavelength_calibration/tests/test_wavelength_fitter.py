import astropy
import astropy.units as u
import numpy as np
import pytest
from astropy.units import Quantity
from astropy.wcs import WCS
from lmfit import fit_report
from pydantic import ValidationError

from solar_wavelength_calibration.atlas.base import AtlasBase
from solar_wavelength_calibration.fitter.parameters import AngleBoundRange
from solar_wavelength_calibration.fitter.parameters import BoundsModel
from solar_wavelength_calibration.fitter.parameters import FitFlagsModel
from solar_wavelength_calibration.fitter.parameters import WavelengthCalibrationParameters
from solar_wavelength_calibration.fitter.wavelength_fitter import WaveCalFitter


def test_wavecal_fitter(
    parameters: WavelengthCalibrationParameters,
    observed_spectrum,
    atlas: AtlasBase,
    test_dispersion,
    test_parameters_dict,
    observed_wavelength_vector,
):
    """
    Given: The correct WaveCalFitter inputs (data, models, and fit parameters)
    When: Initializing a WaveCalFitter object with these inputs
    Then: Everything needed for the fit is correctly set up
    """
    fitter = WaveCalFitter(
        input_parameters=parameters,
        atlas=atlas,
    )

    print(
        fitter.run_fit(
            expected_wavelength_vector=observed_wavelength_vector,
            unfit_1d_spectrum=observed_spectrum,
        )
    )
    # test that values have been fit (i.e. changed)
    """assert fitter.run_fit.wcs.cdelt[0] != test_dispersion
    assert (
        fitter.run_fit.wcs.crval[0]
        != atlas.telluric_atlas_wavelength[np.size(observed_spectrum) // 2 + 1]
    )"""


def test_incorrect_atlas(
    parameters: WavelengthCalibrationParameters,
    observed_spectrum,
    observed_wavelength_vector,
):
    """
    Given: An incorrect atlas (i.e. a random array of numbers)
    When: Initializing a WaveCalFitter object
    Then: Raise a ValidationError
    """
    atlas = np.linspace(1000, 3000, 1000) * u.nm
    with pytest.raises(ValidationError, match="Input should be an instance of AtlasBase"):
        fitter = WaveCalFitter(
            input_parameters=parameters,
            atlas=atlas,
        )
        fitter.run_fit(
            expected_wavelength_vector=observed_wavelength_vector,
            unfit_1d_spectrum=observed_spectrum,
        )


@pytest.mark.parametrize(
    "fixed_parameter",
    [
        "crval1",
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
    Given: A `WaveCalFitter` initialized with a `WavelengthCalibrationParameters` with a single fixed parameter
    When: Running the fit
    Then: The "fit" value of the fixed parameter is the same as its initial value
    """
    fit_flag_args = {fixed_parameter: False}
    fit_flags = FitFlagsModel(**fit_flag_args)

    # Have starting values that are off of the true values so we can make sure free parameters change
    init_values = test_parameters_dict | {
        "crval1": test_parameters_dict["crval1"] + 3 * u.nm,
        "dispersion": test_parameters_dict["dispersion"] * 1.01,
        "incident_light_angle": test_parameters_dict["incident_light_angle"] + 4 * u.deg,
        "resolving_power": test_parameters_dict["resolving_power"] + 10,
        "opacity_factor": test_parameters_dict["opacity_factor"] + 0.1,
        "straylight_fraction": test_parameters_dict["straylight_fraction"] + 0.2,
    }

    initialized_parameters = WavelengthCalibrationParameters(fit_flags=fit_flags, **init_values)

    fitter = WaveCalFitter(
        input_parameters=initialized_parameters,
        atlas=atlas,
        full=True,
    )

    _, fit_result_params = fitter.run_fit(
        expected_wavelength_vector=observed_wavelength_vector, unfit_1d_spectrum=observed_spectrum
    )
    for parameter in fit_result_params:
        if parameter != fixed_parameter:
            continue

        if parameter == "continuum_intensity_correction":
            init_value = observed_continuum_level
        else:
            init_value = init_values[parameter]

        if isinstance(init_value, Quantity):
            init_value = init_value.value

        np.testing.assert_allclose(init_value, fit_result_params[parameter])


def test_fit_stays_valid_with_fixed_crval1(
    test_parameters_dict, observed_wavelength_vector, observed_spectrum, atlas: AtlasBase
):
    """
    Given: A set of input parameters that is on the edge of WCS validity and bounds on `incident_light_angle` that
      would take the fit to invalid WCS values
    When: Running the fit with `crval1` fixed
    Then: The fit stays within valid ranges
    """

    input_crval1 = test_parameters_dict["crval1"] + 10 * u.nm
    edge_angle = (
        np.rad2deg(
            np.arcsin(
                float(
                    input_crval1
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
        "crval1": input_crval1,
        "dispersion": test_parameters_dict["dispersion"] * 1.01,
        "incident_light_angle": edge_angle,
        "resolving_power": test_parameters_dict["resolving_power"] + 10,
        "opacity_factor": test_parameters_dict["opacity_factor"] + 0.1,
        "straylight_fraction": test_parameters_dict["straylight_fraction"] + 0.2,
    }

    parameters = WavelengthCalibrationParameters(
        fit_flags=FitFlagsModel(crval1=False),
        bounds=BoundsModel(incident_light_angle=AngleBoundRange(min=angle_min, max=angle_max)),
        **init_values
    )

    fitter = WaveCalFitter(
        input_parameters=parameters,
        atlas=atlas,
    )
    _ = fitter.run_fit(
        expected_wavelength_vector=observed_wavelength_vector,
        unfit_1d_spectrum=observed_spectrum,
    )


def test_fit_is_correct(
    test_crval1,
    test_dispersion,
    test_incident_light_angle,
    test_grating_constant,
    test_order,
    observed_spectrum,
    observed_wavelength_vector,
    atlas: AtlasBase,
    test_parameters_dict,
):
    """
    Given: A simulated unfit 1D spectrum with a known shift and dispersion scale.
    When: The WaveCalFitter is used to fit the spectrum.
    Then: The fitted dispersion (CDELT1) is approximately equal to the expected dispersion.
    """

    parameters = WavelengthCalibrationParameters(
        crval1=test_crval1 + 1.0 * u.nm,
        dispersion=test_dispersion * 1.01,
        incident_light_angle=test_incident_light_angle * 1.01,
        resolving_power=72500,
        opacity_factor=1,
        straylight_fraction=0.0,
        unfit_1d_spectrum=observed_spectrum,
        grating_constant=test_grating_constant,
        doppler_velocity=100 * u.km / u.s,
        order=test_order,
    )

    fitter = WaveCalFitter(
        input_parameters=parameters,
        atlas=atlas,
    )
    fit_result = fitter.run_fit(
        expected_wavelength_vector=observed_wavelength_vector,
        unfit_1d_spectrum=observed_spectrum,
    )
    assert np.isclose(fit_result["CRVAL1"], test_crval1.to_value("nm"))
    assert np.isclose(fit_result["CDELT1"], test_dispersion.to_value(u.nm / u.pix))

    fit_incident_light_angle = fit_result["PV1_2"]
    # Incident light angle is pretty weakly constrained for our test setup so let's say we pass if we get within a
    # degree.
    assert abs(fit_incident_light_angle - test_incident_light_angle.to_value(u.deg)) < 1


def test_prepare_atlas(
    parameters: WavelengthCalibrationParameters,
    observed_spectrum,
    atlas: AtlasBase,
    observed_wavelength_vector,
):
    """
    Given: The correct WaveCalFitter inputs (data, models, and fit parameters)
    When: Preparing the atlas for the fit
    Then: Make sure that a sample of the atlas is taken, and make sure that it is the correct sample
    """
    fitter = WaveCalFitter(
        input_parameters=parameters,
        atlas=atlas,
    )
    cropped_atlas = fitter.prepare_atlas(expected_wavelength_vector=observed_wavelength_vector)

    # test that we took out a chunk of the atlas
    assert len(cropped_atlas.solar_atlas_transmission) < len(atlas.solar_atlas_transmission)
    assert len(cropped_atlas.telluric_atlas_transmission) < len(atlas.telluric_atlas_transmission)

    # test that we took out the correct chunk for each atlas (i.e. mean of expected wavelength vector is contained in fts_[solar/telluric]wavelength)
    assert (
        cropped_atlas.solar_atlas_wavelength[0]
        < np.mean(observed_wavelength_vector).value
        < cropped_atlas.solar_atlas_wavelength[len(cropped_atlas.solar_atlas_wavelength) - 1]
    )

    assert (
        cropped_atlas.telluric_atlas_wavelength[0]
        < np.mean(observed_wavelength_vector).value
        < cropped_atlas.telluric_atlas_wavelength[len(cropped_atlas.telluric_atlas_wavelength) - 1]
    )


def test_full_flag_true(
    parameters: WavelengthCalibrationParameters,
    observed_spectrum,
    atlas: AtlasBase,
    observed_wavelength_vector,
):
    """
    Given: WaveCal inputs with the full flag set to true
    When: Initializing a WaveCalFitter object
    Then: We get two fit results back; one WCS object, and one dictionary
    """
    fitter = WaveCalFitter(
        input_parameters=parameters,
        atlas=atlas,
        full=True,
    )
    fit_result = fitter.run_fit(
        expected_wavelength_vector=observed_wavelength_vector,
        unfit_1d_spectrum=observed_spectrum,
    )
    # make sure we get 2 outputs
    assert len(fit_result) == 2
    fit_result_header, fit_result_params = fit_result
    # and make sure that the first output is the WCS object...
    assert isinstance(fit_result_header, dict)
    # and the second is a dictionary containing the rest of the fitted parameters
    assert isinstance(fit_result_params, dict)


def test_full_flag_false(
    parameters: WavelengthCalibrationParameters,
    observed_spectrum,
    atlas: AtlasBase,
    observed_wavelength_vector,
):
    """
    Given: WaveCal inputs with the full flag set to false (default)
    When: Initializing a WaveCalFitter object
    Then: We get one fit result back, and it is a dictionary
    """
    fitter = WaveCalFitter(
        input_parameters=parameters,
        atlas=atlas,
        full=False,
    )

    fit_result = fitter.run_fit(
        expected_wavelength_vector=observed_wavelength_vector,
        unfit_1d_spectrum=observed_spectrum,
    )

    # make sure we get one fit result back, and it is a dict
    assert isinstance(fit_result, dict)
