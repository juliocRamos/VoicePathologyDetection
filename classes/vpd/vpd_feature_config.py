from dataclasses import dataclass

@dataclass
class VPDFeatureConfig:
    n_mfcc: int = 30
    n_fft: int = 1024
    hop_length: int = 128

    top_n_harmonics: int = 30
    harmonic_min_freq: float = 50.0
    harmonic_max_freq: float | None = None

    entropy_bins: int = 64
    zcr_percent: int = 20
    energy_percent_steps: int = 10

    include_mfcc_delta: bool = True

    # Glottal
    include_glottal_features: bool = True
    glottal_f0_min: float = 75.0
    glottal_f0_max: float = 600.0
    glottal_pitch_time_step: float = 0.01