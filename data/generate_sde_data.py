import numpy as np

def simulate_ou(
    theta: float, mu: float, sigma: float,
    x0: float, T: float, N: int, seed: int|None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate an Ornstein–Uhlenbeck process:
        dx_t = theta * (mu - x_t) dt + sigma dW_t
    using the Euler–Maruyama method :contentReference[oaicite:5]{index=5}.

    Args:
        theta: Mean-reversion speed.
        mu: Long-term mean.
        sigma: Volatility coefficient.
        x0: Initial state.
        T: Total time.
        N: Number of time steps.
        seed: Random seed (optional).

    Returns:
        t: Array of time points (length N+1).
        x: Simulated process values (length N+1).
    """
    if seed is not None:
        np.random.seed(seed)
    dt = T / N
    t = np.linspace(0, T, N+1)
    x = np.zeros(N+1)
    x[0] = x0
    for i in range(N):
        dW = np.random.normal(scale=np.sqrt(dt))
        x[i+1] = x[i] + theta * (mu - x[i]) * dt + sigma * dW
    return t, x

def simulate_gbm(
    S0: float, mu: float, sigma: float,
    T: float, N: int, seed: int|None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate Geometric Brownian Motion:
        dS_t = mu * S_t dt + sigma * S_t dW_t
    via the Euler–Maruyama method :contentReference[oaicite:6]{index=6}.

    Args:
        S0: Initial price.
        mu: Drift coefficient.
        sigma: Volatility coefficient.
        T: Total time.
        N: Number of time steps.
        seed: Random seed (optional).

    Returns:
        t: Array of time points (length N+1).
        S: Simulated price series (length N+1).
    """
    if seed is not None:
        np.random.seed(seed)
    dt = T / N
    t = np.linspace(0, T, N+1)
    S = np.zeros(N+1)
    S[0] = S0
    for i in range(N):
        dW = np.random.normal(scale=np.sqrt(dt))
        S[i+1] = S[i] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
    return t, S

def generate_sde_data(n_samples=100, n_variables=3, seed=42):
    """
    Generate synthetic SDE data for causal discovery experiments.
    Produces a simple linear causal chain: X → Y → Z.

    Args:
        n_samples: Number of samples (trajectories) to generate.
        n_variables: Number of variables (should be 3 for X,Y,Z).
        seed: Random seed for reproducibility.

    Returns:
        data: dict with keys 'X', 'Y', 'Z' and values as arrays of shape (n_samples, time_steps).
    """
    np.random.seed(seed)
    time_steps = 10  # Number of time points per sample

    X = np.zeros((n_samples, time_steps + 1))
    Y = np.zeros((n_samples, time_steps + 1))
    Z = np.zeros((n_samples, time_steps + 1))

    for i in range(n_samples):
        _, x = simulate_ou(theta=0.5, mu=0.0, sigma=1.0, x0=0.0, T=1.0, N=time_steps, seed=seed + i)
        _, y = simulate_ou(theta=0.5, mu=0.7 * np.mean(x), sigma=0.3, x0=0.0, T=1.0, N=time_steps, seed=seed + 1000 + i)
        _, z = simulate_ou(theta=0.5, mu=0.8 * np.mean(y), sigma=0.2, x0=0.0, T=1.0, N=time_steps, seed=seed + 2000 + i)
        X[i] = x
        Y[i] = y
        Z[i] = z

    return {"X": X, "Y": Y, "Z": Z}