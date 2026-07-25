from dataclasses import dataclass


@dataclass(frozen=True)
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

    def __post_init__(self) -> None:
        if self.target_sr <= 0:
            raise ValueError("target_sr must be positive.")

        if not 0 < self.peak_limit <= 1:
            raise ValueError(
                "peak_limit must be greater than 0 and at most 1."
            )

        if self.center_crop and self.crop_duration_sec <= 0:
            raise ValueError(
                "crop_duration_sec must be positive when center_crop "
                "is enabled."
            )
