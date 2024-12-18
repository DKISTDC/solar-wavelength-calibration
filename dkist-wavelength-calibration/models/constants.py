"""Wavecal additions to common constants."""
from enum import Enum
from enum import unique

from dkist_processing_common.models.constants import ConstantsBase

@unique
class WavecalBudName(Enum):
    """Names to be used for Wavelength Calibration buds."""

    wave_min = "WAVE_MIN"
    wave_max = "WAVE_MAX"
    grating_position_deg = "GRATING_POSITION_DEG"
    grating_littrow_angle_deg = "GRATING_LITTROW_ANGLE_DEG"
    grating_constant = "GRATING_CONSTANT"
    solar_gain_ip_start_time = "SOLAR_GAIN_IP_START_TIME"

class WavecalConstants(ConstantsBase):
    """Wavelength calibration specific constants to add to the common constants."""

    @property
    def wave_min(self) -> float:
        """Wavelength minimum."""
        return self._db_dict[CryonirspBudName.wave_min.value]

    @property
    def wave_max(self) -> float:
        """Wavelength maximum."""
        return self._db_dict[CryonirspBudName.wave_max.value]

    @property
    def solar_gain_ip_start_time(self) -> str:
        """Solar gain IP start time."""
        return self._db_dict[CryonirspBudName.solar_gain_ip_start_time.value]

    @property
    def grating_position_deg(self) -> float:
        """Grating position angle (deg)."""
        return self._db_dict[CryonirspBudName.grating_position_deg.value]

    @property
    def grating_littrow_angle_deg(self) -> float:
        """Grating littrow angle (deg)."""
        return self._db_dict[CryonirspBudName.grating_littrow_angle_deg.value]

    @property
    def grating_constant(self) -> float:
        """Grating constant."""
        return self._db_dict[CryonirspBudName.grating_constant.value]