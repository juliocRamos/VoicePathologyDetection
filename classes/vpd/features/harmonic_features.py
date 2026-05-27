import librosa
import numpy as np


def find_harmonics(
    y: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> np.ndarray:
    """
        Returns the [frequency, average magnitude] matrix of the harmonic portion of the signal.

        Note: This represents candidate spectral peaks after HPSS,
        non-glottal harmonics explicitly estimated from F0
    """
    y = np.asarray(y, dtype=np.float64)

    if y.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    harmonic, _ = librosa.effects.hpss(y)

    stft = librosa.stft(
        harmonic,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    magnitude = np.abs(stft)
    mean_magnitude = np.mean(magnitude, axis=1)

    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    return np.column_stack((frequencies, mean_magnitude))


def get_top_n_harmonics(
    harmonic_matrix: np.ndarray,
    top_n: int = 30,
    min_freq: float = 50.0,
    max_freq: float | None = None,
) -> np.ndarray:
    """
        Select the top N spectral bins by magnitude and returns its matrix
    """
    if harmonic_matrix is None or harmonic_matrix.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    matrix = np.asarray(harmonic_matrix, dtype=np.float64)

    if max_freq is not None:
        matrix = matrix[
            (matrix[:, 0] >= min_freq) &
            (matrix[:, 0] <= max_freq)
        ]
    else:
        matrix = matrix[matrix[:, 0] >= min_freq]

    if matrix.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    top = matrix[np.argsort(matrix[:, 1])[::-1]]
    top = top[:top_n]

    # Resort by frequency for collumn stability
    top = top[np.argsort(top[:, 0])]

    return top