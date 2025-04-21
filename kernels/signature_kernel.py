import iisignature
import numpy as np

def compute_signature(path: np.ndarray, level: int) -> np.ndarray:
    """
    Compute the level‐`level` signature of a path using the iisignature library.
    The signature is the sequence of iterated integrals up to the given level,
    which provides a rich feature representation of the path. 
    """
    # Prepare the log‐signature helper (though we use raw signatures here)
    # The `sig` function computes the signature up to level `level`.
    sig = iisignature.sig(path, level)  # :contentReference[oaicite:0]{index=0}
    return sig

def signature_kernel(path1: np.ndarray,
                     path2: np.ndarray,
                     level: int) -> float:
    """
    Compute the signature kernel between two paths by taking the inner
    product of their signature feature vectors. This kernel implicitly
    maps paths into an RKHS where linear methods can be applied. 
    """
    sig1 = compute_signature(path1, level)
    sig2 = compute_signature(path2, level)
    # Inner product in signature feature space defines the kernel
    return float(np.dot(sig1, sig2))  # :contentReference[oaicite:1]{index=1}
