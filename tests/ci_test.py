# tests/ci_test.py

import numpy as np
from kernels.kernel_utils import compute_kernel_matrix
from kernels.signature_kernel import compute_signature
from utils.evaluation_metrics import centered_gram, hsic_statistic, bootstrap_pvalue

def conditional_independence_test(X_paths, Y_paths, Z_paths=None, level=3, n_bootstrap=1000, alpha=0.05):
    """
    Perform a signature‑kernel conditional independence test:
        H0: X ⟂ Y | Z

    Args:
        X_paths: list of np.ndarray, each shape (T, d) for variable X trajectories
        Y_paths: list of np.ndarray, each shape (T, d) for variable Y trajectories
        Z_paths: optional list of np.ndarray for conditioning variable Z (or None for unconditional)
        level: signature truncation level
        n_bootstrap: number of bootstrap samples for p-value
        alpha: significance level

    Returns:
        p_value: estimated p-value of the CI test
        reject: bool, True if H0 is rejected at level alpha
    """
    # 1. Compute Gram matrices K_XY, K_XZ, K_YZ (or just K_X, K_Y if Z_paths is None)
    Kx = compute_kernel_matrix(X_paths, level)
    Ky = compute_kernel_matrix(Y_paths, level)

    if Z_paths is not None:
        Kz = compute_kernel_matrix(Z_paths, level)
        # Centered conditional Gram matrices
        Kx_z = centered_gram(Kx, Kz)
        Ky_z = centered_gram(Ky, Kz)
    else:
        # Unconditional: just center each Gram
        Kx_z = centered_gram(Kx)
        Ky_z = centered_gram(Ky)

    # 2. Compute HSIC statistic
    stat = hsic_statistic(Kx_z, Ky_z)

    # 3. Estimate p-value via permutation bootstrap
    p_value = bootstrap_pvalue(Kx_z, Ky_z, stat, n_bootstrap=n_bootstrap)

    # 4. Decision
    reject = p_value < alpha

    return p_value, reject, stat


if __name__ == "__main__":
    # Example usage with simulated data
    from data.generate_sde_data import simulate_ou

    # Simulate X, Y, Z as OU processes with varying parameters
    N = 50
    paths_X = [simulate_ou(0.7, 1.0, 0.3, x0=0.0, T=1.0, N=N, seed=i)[1].reshape(-1, 1) for i in range(100)]
    paths_Y = [simulate_ou(0.5, 0.0, 0.5, x0=0.0, T=1.0, N=N, seed=100+i)[1].reshape(-1, 1) for i in range(100)]
    paths_Z = [simulate_ou(0.6, 0.5, 0.4, x0=0.0, T=1.0, N=N, seed=200+i)[1].reshape(-1, 1) for i in range(100)]

    p_val, rej, stat_val = conditional_independence_test(paths_X, paths_Y, Z_paths=paths_Z, level=3)
    print(f"CI test p-value: {p_val:.4f}, reject H0: {rej}, HSIC stat: {stat_val:.4f}")
 
def signature_kernel_ci_test(X, Y, Z=None, alpha=0.05):
    """
    Adapter for the causal_discovery.py import.
    """
    # Convert input arrays to the expected list-of-paths format
    # Each path should be shape (T, d); here, we assume (n_samples, T) input
    if X.ndim == 2:
        X_paths = [X[i].reshape(-1, 1) for i in range(X.shape[0])]
    else:
        X_paths = X
    if Y.ndim == 2:
        Y_paths = [Y[i].reshape(-1, 1) for i in range(Y.shape[0])]
    else:
        Y_paths = Y
    if Z is not None:
        if Z.ndim == 2:
            Z_paths = [Z[i].reshape(-1, 1) for i in range(Z.shape[0])]
        else:
            Z_paths = Z
    else:
        Z_paths = None

    p_value, reject, _ = conditional_independence_test(X_paths, Y_paths, Z_paths=Z_paths, alpha=alpha)
    return reject, p_value