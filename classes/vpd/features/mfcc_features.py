from librosa.feature import mfcc, delta
import numpy as np


def mfcc_features(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 30,
    n_fft: int = 1024,
    hop_length: int = 128,
    include_delta: bool = True,
    pre_emphasis_coef: float = 0.97,
) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)

    if y.size == 0:
        return {}

    # pre emphasis
    y = np.append(y[0], y[1:] - pre_emphasis_coef * y[:-1])

    mfcc_matrix = mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    matrices = [("mfcc", mfcc_matrix)]

    if include_delta and mfcc_matrix.shape[1] >= 3:
        d = delta(mfcc_matrix)
        d2 = delta(mfcc_matrix, order=2)

        matrices.append(("mfcc_delta", d))
        matrices.append(("mfcc_delta2", d2))

    features = {}

    for prefix, matrix in matrices:
        for i in range(matrix.shape[0]):
            values = matrix[i, :]

            features[f"{prefix}_{i + 1:02d}_mean"] = float(np.mean(values))
            features[f"{prefix}_{i + 1:02d}_std"] = float(np.std(values))
            features[f"{prefix}_{i + 1:02d}_min"] = float(np.min(values))
            features[f"{prefix}_{i + 1:02d}_max"] = float(np.max(values))
            features[f"{prefix}_{i + 1:02d}_median"] = float(np.median(values))

    return features