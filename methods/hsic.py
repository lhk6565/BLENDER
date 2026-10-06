import torch

# -----------------------------
# Utilities
# -----------------------------
def distmat(X: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distance matrix."""
    r = torch.sum(X * X, dim=1, keepdim=True)  # (m,1)
    a = X @ X.t()                              # (m,m)
    D2 = r - 2 * a + r.t()
    return D2.clamp_min_(0.0)

def centering_matrix(m: int, device=None, dtype=None) -> torch.Tensor:
    """H = I - (1/m) 11^T"""
    I = torch.eye(m, device=device, dtype=dtype)
    one = torch.ones((m, m), device=device, dtype=dtype)
    return I - (1.0 / m) * one

def double_center(K: torch.Tensor) -> torch.Tensor:
    """HKH with optional symmetrization for numerical stability."""
    m = K.size(0)
    H = centering_matrix(m, device=K.device, dtype=K.dtype)
    Kc = H @ K @ H
    # Enforce symmetry (helps when K is near-symmetric but accumulates fp error)
    return 0.5 * (Kc + Kc.t())

# -----------------------------
# Kernel Gram matrices
# -----------------------------
def gram_matrix(X: torch.Tensor,
                sigma: float = 1.0,
                k_type: str = "gaussian",
                eps: float = 1e-12) -> torch.Tensor:
    """
    Return the (uncentered) Gram matrix K.
    For delta kernel, X can be:
      - one-hot (m, K) float tensor, or
      - integer labels (m,) long tensor
    """
    if k_type == "gaussian":
        if X.dim() != 2:
            raise ValueError("Gaussian kernel expects X with shape (m, d).")
        D2 = distmat(X)
        # Keep your earlier scaling (variance = 2*sigma^2*d)
        variance = 2.0 * (sigma ** 2) * X.size(1)
        K = torch.exp(-D2 / (variance + eps))

    elif k_type == "linear":
        if X.dim() != 2:
            raise ValueError("Linear kernel expects X with shape (m, d).")
        K = X @ X.t()

    elif k_type == "delta":
        if X.dim() == 2:
            labels = X.argmax(dim=1)
        elif X.dim() == 1:
            labels = X
        else:
            raise ValueError("Delta kernel expects X with dim 1 (labels) or dim 2 (one-hot).")
        K = (labels[:, None] == labels[None, :]).to(dtype=torch.float32, device=X.device)

        # Match dtype with others if you want consistency
        if X.dtype.is_floating_point:
            K = K.to(dtype=X.dtype)

    else:
        raise ValueError(f"Unsupported k_type: {k_type}")

    return K

# -----------------------------
# Standard HSIC (biased) - stable & (theoretically) nonnegative
# -----------------------------
def hsic_biased(x: torch.Tensor,
                y: torch.Tensor,
                sigma_x: float,
                sigma_y: float = None,
                k_type_x: str = "gaussian",
                k_type_y: str = "delta",
                eps: float = 1e-12,
                clamp_nonneg: bool = True) -> torch.Tensor:
    """
    Standard (biased) HSIC estimator:
        HSIC = (1/(m-1)^2) * tr( Kc Lc )
    where Kc = HKH and Lc = HLH.

    This form is far less prone to negative values than the NOCCO/CCA-style score.
    Any tiny negative value can occur only from floating-point error; clamp_nonneg handles it.
    """
    m = x.size(0)
    if y.size(0) != m:
        raise ValueError("x and y must have the same number of samples (same first dimension).")

    if sigma_y is None:
        sigma_y = sigma_x

    K = gram_matrix(x, sigma=sigma_x, k_type=k_type_x, eps=eps)
    L = gram_matrix(y, sigma=sigma_y, k_type=k_type_y, eps=eps)

    Kc = double_center(K)
    Lc = double_center(L)

    hsic = torch.trace(Kc @ Lc) / ((m - 1.0) ** 2 + eps)

    if clamp_nonneg:
        hsic = hsic.clamp_min(0.0)

    return hsic

# -----------------------------
# Optional: NOCCO/Kernel-CCA style score, but fixed to use HKH
# (Still not guaranteed nonnegative in all cases, but much more stable than KH.)
# -----------------------------
def hsic_nocco_style(x: torch.Tensor,
                     y: torch.Tensor,
                     sigma_x: float,
                     sigma_y: float = None,
                     k_type_x: str = "gaussian",
                     k_type_y: str = "delta",
                     epsilon: float = 1e-5,
                     eps: float = 1e-12) -> torch.Tensor:
    """
    Your earlier 'normalized_cca' style score, but corrected to use double-centered Gram matrices.
    Kept for compatibility; prefer hsic_biased for nonnegativity.
    """
    m = x.size(0)
    if y.size(0) != m:
        raise ValueError("x and y must have the same number of samples.")

    if sigma_y is None:
        sigma_y = sigma_x

    K = gram_matrix(x, sigma=sigma_x, k_type=k_type_x, eps=eps)
    L = gram_matrix(y, sigma=sigma_y, k_type=k_type_y, eps=eps)

    Kc = double_center(K)
    Lc = double_center(L)

    I = torch.eye(m, device=x.device, dtype=x.dtype)

    # Solve (Kc + epsilon*m*I)^{-1} without explicit inverse
    Kc_inv = torch.linalg.solve(Kc + (epsilon * m) * I, I)
    Lc_inv = torch.linalg.solve(Lc + (epsilon * m) * I, I)

    Rx = Kc @ Kc_inv
    Ry = Lc @ Lc_inv

    # Equivalent to (Rx * Ry.t()).sum(), but explicit trace is clearer
    return torch.trace(Rx @ Ry)

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    m, dz, D = 64, 16, 4
    z_inv = torch.randn(m, dz)
    d_labels = torch.randint(0, D, (m,))          # (m,) labels
    # or d_onehot = torch.nn.functional.one_hot(d_labels, num_classes=D).float()

    # Recommended: standard HSIC with delta kernel on d
    hsic_val = hsic_biased(
        x=z_inv,
        y=d_labels,        # or d_onehot
        sigma_x=1.0,
        k_type_x="gaussian",
        k_type_y="delta",
        clamp_nonneg=True
    )
    print("HSIC(z_inv, d) =", hsic_val.item())
