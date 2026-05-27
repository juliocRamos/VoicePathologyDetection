from dataclasses import dataclass
from typing import Optional


@dataclass
class AudioPreprocessConfig:
    target_sr: int = 16_000

    convert_to_mono: bool = True
    remove_dc: bool = True

    normalize_rms: bool = True
    target_dbfs: float = -20.0
    peak_limit: float = 0.99

    center_crop: bool = False
    crop_duration_sec: float = 2.0
    pad_if_short: bool = False

    min_duration_sec: Optional[float] = 0.5