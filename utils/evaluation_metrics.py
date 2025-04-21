import numpy as np

def centered_gram(K: np.ndarray, Kz: np.ndarray | None = None) -> np.ndarray:
    """
    Center (and optionally condition) a Gram matrix.
    - Unconditional centering uses H = I - (1/n)11^T to double‐center K. :contentReference[oaicite:0]{index=0}
    - Conditional centering projects out directions in RKHS induced by Kz:
      H_z = I - Kz (Kz + εI)^{-1}, then returns H_z @ K @ H_z. :contentReference[oaicite:1]{index=1}
    """
    n = K.shape[0]
    if Kz is None:
        H = np.eye(n) - np.ones((n, n)) / n  # centering matrix :contentReference[oaicite:2]{index=2}
        return H @ K @ H                    # double centering :contentReference[oaicite:3]{index=3}
    # Conditional centering (residualization in RKHS)
    reg = 1e-6 * np.eye(n)
    inv = np.linalg.inv(Kz + reg)           # regularized inverse :contentReference[oaicite:4]{index=4}
    H_z = np.eye(n) - Kz @ inv              # projector onto orthogonal complement :contentReference[oaicite:5]{index=5}
    return H_z @ K @ H_z

def hsic_statistic(K: np.ndarray, L: np.ndarray) -> float:
    """
    Compute the (V‐statistic) HSIC measure:
    HSIC = (1/(n-1)^2) * trace(KHLH), but here using center‐already Gram matrices:
    HSIC ≈ trace(K @ L) / (n-1)^2. :contentReference[oaicite:6]{index=6}
    """
    n = K.shape[0]
    return float(np.trace(K @ L) / (n - 1)**2)

def bootstrap_pvalue(
    K: np.ndarray,
    L: np.ndarray,
    stat: float,
    n_bootstrap: int = 1000
) -> float:
    """
    Estimate a p-value via permutation bootstrap:
    - Permute indices of K to mimic H0: independence. :contentReference[oaicite:7]{index=7}
    - Compute HSIC each time and compare to observed stat. :contentReference[oaicite:8]{index=8}
    """
    n = K.shape[0]
    null_stats = []
    for _ in range(n_bootstrap):
        perm = np.random.permutation(n)
        Kp = K[perm][:, perm]                # permuted Gram :contentReference[oaicite:9]{index=9}
        null_stats.append(hsic_statistic(Kp, L))
    null_stats = np.array(null_stats)
    # p-value: fraction of null ≥ observed, plus one to avoid zero. :contentReference[oaicite:10]{index=10}
    return float((np.sum(null_stats >= stat) + 1) / (n_bootstrap + 1))

def compute_shd(true_graph: np.ndarray, learned_graph: np.ndarray) -> float:
    """
    Compute Structural Hamming Distance between two graphs.
    
    The Structural Hamming Distance (SHD) is the number of edge insertions,
    deletions, and reversals needed to transform one graph into another.
    It's a standard metric for evaluating the quality of learned causal graphs.
    
    Args:
        true_graph: Adjacency matrix of the true causal graph
        learned_graph: Adjacency matrix of the learned causal graph
        
    Returns:
        The SHD value as a float
    """
    # Ensure both graphs have the same dimensions
    if true_graph.shape != learned_graph.shape:
        raise ValueError("Graph adjacency matrices must have the same dimensions")
    
    n = true_graph.shape[0]
    shd = 0
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue  # Skip self-loops
                
            # Different edge types contribute to SHD:
            # 1. Edge present in true but missing in learned (deletion)
            # 2. Edge missing in true but present in learned (insertion)
            # 3. Edge direction differs (reversal, counts as one edit)
            
            if true_graph[i, j] == 1 and learned_graph[i, j] == 0:
                # Edge deletion
                shd += 1
            elif true_graph[i, j] == 0 and learned_graph[i, j] == 1:
                # Edge insertion
                shd += 1
            elif true_graph[i, j] == 1 and learned_graph[i, j] == 1:
                # Edge exists in both, check if direction is correct
                if true_graph[j, i] != learned_graph[j, i]:
                    # Direction is different (one is directed, one is undirected)
                    shd += 1
    
    return float(shd)
