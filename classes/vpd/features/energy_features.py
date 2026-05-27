import numpy as np


def get_energy_area(
    input_vector: np.ndarray,
    percent_step: int = 10,
) -> np.ndarray:
    """
        Returns the normalized positions in which the accumulated energies reach
        percentages from total: 10%, 20%, ..., 90%.

        Standard output: 9 values.
    """
    x = np.asarray(input_vector, dtype=np.float64)

    if x.size == 0:
        return np.full(9, np.nan)

    cumulative_energy = np.cumsum(x ** 2)
    total_energy = cumulative_energy[-1]

    if total_energy <= 0:
        return np.zeros(9, dtype=np.float64)

    thresholds = np.arange(percent_step, 100, percent_step) / 100.0
    positions = []

    for threshold in thresholds:
        target = threshold * total_energy
        index = int(np.searchsorted(cumulative_energy, target, side="left"))
        positions.append(index / max(len(x) - 1, 1))

    return np.asarray(positions, dtype=np.float64)