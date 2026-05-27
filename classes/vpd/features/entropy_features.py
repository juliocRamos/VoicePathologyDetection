import numpy as np


def entropy_1d(
    input_vector: np.ndarray,
    bins: int = 64,
) -> float:
    """
        Shannon entropy estimated by histogram.
        More suitable for continuous signals than counting exact float values.
    """
    x = np.asarray(input_vector, dtype=np.float64)

    if x.size == 0:
        return np.nan

    if np.allclose(x, x[0]):
        return 0.0

    hist, _ = np.histogram(x, bins=bins, density=False)
    total = np.sum(hist)

    if total == 0:
        return 0.0

    p = hist / total
    p = p[p > 0]

    return float(-np.sum(p * np.log2(p)))

def get_entropy_C2(
    signal: np.ndarray,
    partitions: tuple[int, ...] = (2, 3, 5, 7, 11, 13, 17),
    bins: int = 64,
) -> np.ndarray:
    """
        Calculates normalized entropies by partitioning the signal.
        With partitions=(2,3,5,7,11,13,17), it returns 58 values.
    """
    x = np.asarray(signal, dtype=np.float64)

    if x.size == 0:
        return np.full(sum(partitions), np.nan)

    features = []

    for n_parts in partitions:
        segments = np.array_split(x, n_parts)

        entropies = np.array(
            [entropy_1d(segment, bins=bins) for segment in segments],
            dtype=np.float64,
        )

        max_entropy = np.nanmax(entropies)

        if max_entropy > 0:
            entropies = entropies / max_entropy

        features.extend(entropies.tolist())

    return np.asarray(features, dtype=np.float64)