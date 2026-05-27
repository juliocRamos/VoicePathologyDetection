import numpy as np


def zcr_numpy(x: np.ndarray) -> int:
    x = np.asarray(x, dtype=np.float64)

    if x.size < 2:
        return 0

    return int(np.sum(np.diff(np.signbit(x)) != 0))

def get_zcr_B3_optimized(
    input_vec: np.ndarray,
    percent: int = 20,
) -> np.ndarray:
    """
        Returns normalized positions where the accumulated ZCR reaches
        20%, 40%, 60%, 80% of the total, by default.

        Default output with percent=20: 4 values.
    """
    x = np.asarray(input_vec, dtype=np.float64)

    if x.size == 0:
        return np.full(4, np.nan)

    x = x - np.mean(x)

    signs = np.diff(np.signbit(x)) != 0
    cumulative_zcr = np.cumsum(signs.astype(int))
    total_zcr = int(cumulative_zcr[-1]) if cumulative_zcr.size > 0 else 0

    thresholds = np.arange(percent, 100, percent) / 100.0

    if total_zcr == 0:
        return np.zeros(len(thresholds), dtype=np.float64)

    positions = []

    for threshold in thresholds:
        target = threshold * total_zcr
        index = int(np.searchsorted(cumulative_zcr, target, side="left"))
        positions.append(index / max(len(x) - 1, 1))

    return np.asarray(positions, dtype=np.float64)