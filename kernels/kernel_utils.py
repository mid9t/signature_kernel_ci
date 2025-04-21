import numpy as np
from .signature_kernel import signature_kernel

def compute_kernel_matrix(
    paths: list[np.ndarray],
    level: int
) -> np.ndarray:
    """
    Given a list of N paths, compute the N×N kernel matrix where
    K[i, j] = signature_kernel(paths[i], paths[j], level).
    """
    N = len(paths)
    K = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(i, N):
            kij = signature_kernel(paths[i], paths[j], level)
            K[i, j] = K[j, i] = kij
    return K

def select_kernel_level(
    max_level: int = 4,
    data_dim: int = 1
) -> int:
    """
    Heuristic to choose the truncation level for signature computation.
    Common practice is to use levels up to 3–5 for small-dimensional paths.
    """
    # Example heuristic: level = min(max_level, data_dim + 2)
    return min(max_level, data_dim + 2)  # :contentReference[oaicite:4]{index=4}
