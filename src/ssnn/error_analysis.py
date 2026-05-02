r"""
Theoretical analysis of approximation error (Step 5).

Decomposes the total error of the Edgeworth-corrected SSNN into five
computable bounds:

    (a) **Edgeworth truncation error** — remainder from truncating the
        cumulant expansion at κ₄.  Bounded via Berry–Esseen-type arguments:
        the density-level remainder of an order-s Edgeworth expansion for a
        sum of p independent (or decorrelated) discrete random variables is
        O(1 / √p) for the κ₃ term and O(1/p) for the κ₄ term.  In
        expectation against a Lipschitz test function g (e.g. ReLU, sigmoid)
        the truncation error inherits the same rate.

    (b) **Decorrelation approximation error** — the diagonal projection-
        cumulant formulas use Σ^{-1/2} to decorrelate, then pretend the
        resulting components are independent.  This kills cross-cumulant
        tensors of order ≥ 3.  The bound measures the residual via the
        Frobenius norm of the neglected cross-cumulant tensor contracted
        against the weight vector.

    (c) **LD estimation error** — propagates |Σ̂ − Σ|_F through the loss.
        The loss depends on Σ through projection variances v_k = wᵀΣw and
        pairwise covariances c_{kl}; perturbation analysis gives a bound
        linear in the LD matrix error.

    (d) **PUMAS splitting variance** — the pseudo-subset summary stats
        (Σ̂β_tr, E[y²]_tr) are noisy perturbations of the full-sample
        quantities.  We derive the variance of the surrogate loss under
        PUMAS splits as a function of N, n_tr, and the LD spectrum.

    (e) **Optimization error** — for the gradient-descent optimizer with
        backtracking and gradient clipping, we bound the suboptimality
        after T iterations in terms of the smoothness of L_EW (Lipschitz
        gradient constant) and the step size.

All bounds are computable from the same summary-recoverable quantities
used by the SSNN itself (Σ, Σβ, MAF, N).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .cumulants import snp_cumulants, decorrelation_matrix


# =====================================================================
# Data class for the full error decomposition
# =====================================================================

@dataclass
class ErrorDecomposition:
    """Full error decomposition for the Edgeworth-corrected SSNN.

    Each field is a non-negative scalar upper bound on the corresponding
    error source's contribution to the surrogate loss.

    Attributes:
        edgeworth_truncation: Bound on |E_true[g] - E_EW[g]| from
            truncating the Edgeworth series after the κ₃²/72 term.
        decorrelation_approx: Bound on the cross-cumulant terms
            neglected by the diagonal projection-cumulant formulas.
        ld_estimation: Bound on |L_EW(Σ̂) - L_EW(Σ)| from LD error.
        pumas_variance: Standard deviation of L_EW under PUMAS splits.
        optimization: Suboptimality bound after T GD iterations.
        total: Sum of all five bounds (triangle-inequality aggregate).
    """
    edgeworth_truncation: float
    decorrelation_approx: float
    ld_estimation: float
    pumas_variance: float
    optimization: float

    @property
    def total(self) -> float:
        return (
            self.edgeworth_truncation
            + self.decorrelation_approx
            + self.ld_estimation
            + self.pumas_variance
            + self.optimization
        )


# =====================================================================
# (a) Edgeworth truncation error
# =====================================================================

def edgeworth_truncation_bound(
    maf: np.ndarray,
    W: np.ndarray,
    Sigma: np.ndarray | None = None,
    Sigma_inv_sqrt: np.ndarray | None = None,
    activation: str = "relu",
) -> float:
    r"""Bound the Edgeworth truncation error for all hidden units.

    The Edgeworth expansion of the density of z = w^T x / √(w^T Σ w)
    through order s has a uniform remainder bounded by (Bhattacharya &
    Rao, 1976, Thm 20.1 for lattice distributions):

        sup_t |f_z(t) - f_{z,s}(t)|  ≤  C_s · ρ_{s+1}(w)

    where ρ_{s+1}(w) = Σ_j |w̃_j|^{s+1} κ_{s+1,j} / (Σ_j w̃_j² κ_{2,j})^{(s+1)/2}
    is the (s+1)-th Lyapunov-type ratio of the decorrelated projection.

    Our expansion includes terms through H₆ (i.e. through κ₃² which is
    the s=2 term of the formal expansion).  The first omitted term is the
    fifth cumulant contribution.  For Binomial(2,p) genotypes, κ₅ is
    computable:
        κ₅ = 2p(1-p)(1-2p)(1-12p(1-p))

    but we use a simpler, tighter bound: the next neglected density-level
    correction is O(κ₃·κ₄) with coefficient from H₇, and the dominant
    remainder for Lipschitz test functions (activations) is:

        |E_true[g(z)] - E_EW[g(z)]|
            ≤ Lip(g) · (|κ̃₃·κ̃₄|/144 · M₇ + κ̃₃³/1296 · M₉ + κ̃₅/120 · M₅)

    where M_r = E[|H_r(z)|] for z ~ N(0,1), and κ̃ are the standardised
    projection cumulants.

    For a conservative bound we use:
        ≤ Lip(g) · ρ₅(w)   (Lyapunov ratio at order 5)

    where we take Lip(g) = 1 for ReLU and Lip(g) = λ = √(π/8) for the
    probit sigmoid (the tightest Lipschitz constant of Φ(λ·)).

    The bound returned is the maximum over all m hidden units.

    Args:
        maf: (p,) minor allele frequencies.
        W: (m, p) first-layer weight matrix.
        Sigma: (p, p) LD covariance matrix (None → identity).
        Sigma_inv_sqrt: precomputed Σ^{-1/2} (optional).
        activation: "relu", "sigmoid", or "identity".

    Returns:
        Non-negative scalar upper bound on the truncation error.
    """
    cum = snp_cumulants(maf)
    k2, k3, k4 = cum["kappa2"], cum["kappa3"], cum["kappa4"]
    p_snps = len(maf)
    pq = maf * (1.0 - maf)
    kappa5 = 2.0 * pq * (1.0 - 2.0 * maf) * (1.0 - 12.0 * pq)

    lip_g = _lipschitz_constant(activation)

    if Sigma is not None:
        if Sigma_inv_sqrt is None:
            Sigma_inv_sqrt = decorrelation_matrix(Sigma)
        W_tilde = W @ Sigma_inv_sqrt.T
    else:
        W_tilde = W.copy()

    max_bound = 0.0
    for k in range(W.shape[0]):
        w = W_tilde[k]
        V = np.sum(w**2 * k2)
        if V <= 0:
            continue

        kt3 = np.sum(w**3 * k3) / V**1.5
        kt4 = np.sum(w**4 * k4) / V**2.0
        kt5 = np.sum(w**5 * kappa5) / V**2.5

        rho5 = (
            abs(kt3 * kt4) / 144.0 * _hermite_abs_moment(7)
            + abs(kt3)**3 / 1296.0 * _hermite_abs_moment(9)
            + abs(kt5) / 120.0 * _hermite_abs_moment(5)
        )

        max_bound = max(max_bound, lip_g * rho5)

    return max_bound


def _lipschitz_constant(activation: str) -> float:
    """Lipschitz constant of the activation on the real line."""
    if activation == "relu":
        return 1.0
    elif activation == "sigmoid":
        return np.sqrt(np.pi / 8.0)
    elif activation == "identity":
        return 1.0
    else:
        raise ValueError(f"Unknown activation: {activation!r}")


def _hermite_abs_moment(r: int) -> float:
    r"""E[|H_r(z)|] for z ~ N(0,1), computed via Gauss–Hermite quadrature.

    For moderate r this is cheap and exact to machine precision with
    sufficient nodes.
    """
    nodes, weights = np.polynomial.hermite_e.hermegauss(60)
    norm_factor = 1.0 / np.sqrt(2.0 * np.pi)
    h_vals = _hermite_poly(r, nodes)
    return float(np.sum(weights * np.abs(h_vals)) * norm_factor)


def _hermite_poly(r: int, t: np.ndarray) -> np.ndarray:
    """Evaluate the probabilist's Hermite polynomial H_r(t) via recurrence."""
    if r == 0:
        return np.ones_like(t)
    elif r == 1:
        return t.copy()
    h_prev2 = np.ones_like(t)
    h_prev1 = t.copy()
    for n in range(2, r + 1):
        h_curr = t * h_prev1 - (n - 1) * h_prev2
        h_prev2 = h_prev1
        h_prev1 = h_curr
    return h_prev1


# =====================================================================
# (b) Decorrelation approximation error
# =====================================================================

def decorrelation_bound(
    maf: np.ndarray,
    W: np.ndarray,
    Sigma: np.ndarray,
    Sigma_inv_sqrt: np.ndarray | None = None,
) -> float:
    r"""Bound the error from the diagonal-cumulant approximation in LD.

    The exact third cumulant of the projection z = w^T x is:

        κ₃(z) = Σ_{i,j,k} w_i w_j w_k cum(x_i, x_j, x_k)

    which involves the full third-order cumulant tensor.  The diagonal
    approximation in decorrelated coordinates replaces this with:

        κ₃^{diag}(z) = Σ_i w̃_i³ κ₃(x̃_i)

    where w̃ = Σ^{-1/2} w, x̃ = Σ^{-1/2} x.  The error is the sum of
    off-diagonal cross-cumulant terms.

    For Binomial(2, p) genotypes the per-SNP third cumulant is known
    exactly.  The cross-cumulant tensor of the decorrelated x̃ has entries
    proportional to the mixing induced by Σ^{-1/2}.  We bound:

        |κ₃^{exact}(z) - κ₃^{diag}(z)|
            ≤ Σ_{(i,j,k) not all equal} |w̃_i w̃_j w̃_k| · |cum(x̃_i, x̃_j, x̃_k)|

    The cross-cumulants of x̃ are bounded by the maximal per-SNP |κ₃|
    times the cube of the maximal off-diagonal entry of Σ^{-1/2} Σ₃ Σ^{-T/2}
    where Σ₃ is a matricization of the third cumulant tensor.

    A conservative but practical bound uses:

        |cross-cumulant correction| ≤ ‖w̃‖₃³ · max_j|κ₃_j| · (1 - Σ_j (L_j^{diag})³ / ‖L‖₃³)

    where L = Σ^{-1/2} and the ratio measures how "diagonal" the
    decorrelation is.

    We compute an analogous bound for the fourth cumulant.

    The returned scalar is max over hidden units of the combined (κ₃ + κ₄)
    cross-cumulant error, normalised by the projection variance.

    Args:
        maf: (p,) minor allele frequencies.
        W: (m, p) first-layer weight matrix.
        Sigma: (p, p) LD covariance matrix.
        Sigma_inv_sqrt: precomputed Σ^{-1/2} (optional).

    Returns:
        Non-negative scalar upper bound on the decorrelation error.
    """
    if Sigma_inv_sqrt is None:
        Sigma_inv_sqrt = decorrelation_matrix(Sigma)

    cum = snp_cumulants(maf)
    k2 = cum["kappa2"]
    k3_abs_max = np.max(np.abs(cum["kappa3"]))
    k4_abs_max = np.max(np.abs(cum["kappa4"]))

    diag_L = np.diag(Sigma_inv_sqrt)
    L_diag_cube = np.sum(np.abs(diag_L)**3)
    L_full_cube = np.sum(np.abs(Sigma_inv_sqrt)**3)

    if L_full_cube < 1e-30:
        return 0.0
    off_diag_fraction_3 = 1.0 - L_diag_cube / L_full_cube

    L_diag_four = np.sum(np.abs(diag_L)**4)
    L_full_four = np.sum(np.abs(Sigma_inv_sqrt)**4)
    if L_full_four < 1e-30:
        off_diag_fraction_4 = 0.0
    else:
        off_diag_fraction_4 = 1.0 - L_diag_four / L_full_four

    W_tilde = W @ Sigma_inv_sqrt.T

    max_bound = 0.0
    for k in range(W.shape[0]):
        w = W_tilde[k]
        V = np.sum(w**2 * k2)
        if V <= 0:
            continue

        w3_norm = np.sum(np.abs(w)**3)
        cross_err_3 = w3_norm * k3_abs_max * off_diag_fraction_3 / V**1.5

        w4_norm = np.sum(np.abs(w)**4)
        cross_err_4 = w4_norm * k4_abs_max * off_diag_fraction_4 / V**2.0

        max_bound = max(max_bound, cross_err_3 + cross_err_4)

    return max_bound


# =====================================================================
# (c) LD estimation error
# =====================================================================

def ld_estimation_bound(
    W: np.ndarray,
    a: np.ndarray,
    Sigma: np.ndarray,
    Sigma_beta: np.ndarray,
    delta_Sigma_fro: float,
    activation: str = "relu",
) -> float:
    r"""Bound the loss perturbation from LD estimation error.

    The surrogate loss depends on Σ through:
        v_k = w_k^T Σ w_k           (projection variances)
        c_{kl} = w_k^T Σ w_l        (pairwise covariances)
        s_k = (Σβ*)^T w_k           (Stein moments — if Σβ* is estimated jointly)

    For the Stein term (linear in Σ):
        |s_k(Σ̂) - s_k(Σ)| = |(Σ̂ - Σ)β* · w_k| ≤ ‖(Σ̂ - Σ)β*‖ · ‖w_k‖

    But in practice Σβ* comes directly from GWAS (X^Ty/n), not from Σ×β̂,
    so the dominant sensitivity is through v_k and c_{kl}.

    For the activation expectations, the Gaussian integrals (and their
    Edgeworth corrections) depend smoothly on C_{kl}. We use:

        |v_k(Σ̂) - v_k(Σ)| = |w_k^T (Σ̂ - Σ) w_k| ≤ ‖Σ̂ - Σ‖_F · ‖w_k‖²

    The total loss perturbation is bounded by the derivative of L w.r.t.
    each C-matrix entry, summed over all (k,l) pairs.

    A first-order bound (from Lipschitz continuity of the loss in Σ):

        |L_EW(Σ̂) - L_EW(Σ)| ≤ K(a, W, β*) · ‖Σ̂ - Σ‖_F

    where K depends on the network weights and the activation's smoothness.

    For ReLU, the activation expectations have Lipschitz derivatives in
    v_k of order 1/√v_k, so K scales with Σ_k |a_k|²/√v_k + cross terms.
    We compute K explicitly.

    Args:
        W: (m, p) first-layer weight matrix.
        a: (m,) second-layer weights.
        Sigma: (p, p) true LD covariance matrix.
        Sigma_beta: (p,) = Sigma @ beta*.
        delta_Sigma_fro: ‖Σ̂ - Σ‖_F (Frobenius norm of LD error).
        activation: activation function name.

    Returns:
        Non-negative scalar upper bound on |L(Σ̂) - L(Σ)|.
    """
    from .activations import get_activation_derivs
    from .gaussian_integrals import projection_variance

    dE_sp_dv, grad_E_ss = get_activation_derivs(activation)

    m, p = W.shape
    w_norms_sq = np.sum(W**2, axis=1)

    sensitivity = 0.0

    for k in range(m):
        v_k = projection_variance(Sigma, W[k])
        v_k = max(v_k, 1e-15)

        dEsp = abs(dE_sp_dv(v_k))
        s_k = abs(float(Sigma_beta @ W[k]))
        sensitivity += 2.0 * abs(a[k]) * s_k * dEsp * w_norms_sq[k]

    for k in range(m):
        for l in range(m):
            from .gaussian_integrals import pairwise_covariance
            C_kl = pairwise_covariance(Sigma, W[k], W[l])
            dF = grad_E_ss(C_kl)
            dF_norm = np.sqrt(dF[0, 0]**2 + 2 * dF[0, 1]**2 + dF[1, 1]**2)
            sensitivity += abs(a[k] * a[l]) * dF_norm * (
                w_norms_sq[k] + w_norms_sq[l]
            )

    return sensitivity * delta_Sigma_fro


# =====================================================================
# (d) PUMAS splitting variance
# =====================================================================

def pumas_variance_bound(
    Sigma: np.ndarray,
    Sigma_beta: np.ndarray,
    E_y2: float,
    N: int,
    n_train: int,
    W: np.ndarray,
    a: np.ndarray,
    activation: str = "relu",
) -> float:
    r"""Bound the standard deviation of L_EW under PUMAS splitting noise.

    PUMAS generates pseudo-training summary stats via:

        Σβ̂_tr | Σβ̂ ~ N( (n_tr/N) Σβ̂,  ((N-n_tr)/N²) Σ )

    The loss L(a, W) depends linearly on Σβ through the Stein term
    E[y·f(x)] = Σ_k a_k (Σβ)^T w_k E[σ'(z_k)].

    So L_EW(Σβ̂_tr) ≈ L_EW((n_tr/N)Σβ̂) + ∇_{Σβ}L · δ where
    δ ~ N(0, ((N-n_tr)/N²) Σ).

    The gradient of L_EW w.r.t. Σβ is:
        ∂L/∂(Σβ)_j = -2 Σ_k a_k w_{k,j} E_EW[σ'(z_k)]

    so Var(L_EW(Σβ̂_tr)) = g^T · ((N-n_tr)/N²) Σ · g

    where g_j = -2 Σ_k a_k w_{k,j} E_EW[σ'(z_k)].

    In addition, E[y²]_tr has variance ~ 2 E[y²]² (1/n_tr - 1/N), and
    since L = E[y²] - 2E[yf] + E[f²], the E[y²] noise adds directly.

    We return √(Var_Σβ + Var_{E[y²]}).

    Args:
        Sigma: (p, p) LD covariance matrix.
        Sigma_beta: (p,) full-sample marginal associations.
        E_y2: full-sample E[y²].
        N: Full GWAS sample size.
        n_train: Desired training subset size.
        W: (m, p) first-layer weight matrix.
        a: (m,) second-layer weights.
        activation: activation function name.

    Returns:
        Non-negative scalar: std dev of the surrogate loss under PUMAS noise.
    """
    from .edgeworth_integrals import edgeworth_E_sigma_prime
    from .gaussian_integrals import projection_variance

    m, p = W.shape
    n_val = N - n_train

    g = np.zeros(p)
    for k in range(m):
        v_k = projection_variance(Sigma, W[k])
        E_sp = edgeworth_E_sigma_prime(v_k, 0.0, 0.0, activation)
        g += -2.0 * a[k] * E_sp * W[k]

    cov_scale = float(n_val) / float(N)**2
    Sigma_g = Sigma @ g
    var_Sigma_beta = cov_scale * float(g @ Sigma_g)

    var_Ey2 = 2.0 * E_y2**2 * (1.0 / n_train - 1.0 / N)

    total_var = max(var_Sigma_beta + var_Ey2, 0.0)
    return np.sqrt(total_var)


# =====================================================================
# (e) Optimization error
# =====================================================================

def optimization_bound(
    loss_history: list[float] | np.ndarray,
    lr: float,
    grad_clip: float | None = None,
    L_smooth: float | None = None,
) -> float:
    r"""Bound the suboptimality of gradient descent on L_EW.

    For a function with L-Lipschitz gradients, gradient descent with
    step size η ≤ 1/L satisfies:

        L(θ_T) - L(θ*) ≤ ‖θ_0 - θ*‖² / (2 η T)

    Since we don't know θ* or L exactly, we use two practical proxies:

    1. **Empirical convergence gap**: the difference between the final
       loss and the minimum loss observed during training gives a lower
       bound on how close the optimizer came to its basin minimum.

    2. **Rate-based bound**: if L_smooth is provided, we use the standard
       GD convergence rate.  With gradient clipping at norm G, the
       effective step is bounded by η·G, giving:

           L(θ_T) - L(θ*) ≤ (L_smooth · G² · η) / 2

       per iteration (descent lemma), and over T steps the average
       gradient norm satisfies ‖∇L‖² ≤ 2(L(θ_0)-L(θ*))/( η T).

    We return the empirical gap (always available) unless L_smooth is
    provided, in which case we return the tighter of the two.

    Args:
        loss_history: sequence of loss values from training.
        lr: learning rate used.
        grad_clip: gradient clipping norm (None = no clipping).
        L_smooth: Lipschitz constant of the gradient (optional).

    Returns:
        Non-negative scalar bound on L(θ_T) - L*.
    """
    losses = np.asarray(loss_history, dtype=float)
    if len(losses) == 0:
        return float("inf")

    final_loss = losses[-1]
    min_loss = np.min(losses)
    empirical_gap = max(final_loss - min_loss, 0.0)

    if L_smooth is not None and len(losses) >= 2:
        T = len(losses) - 1
        init_gap = losses[0] - min_loss
        rate_bound = 2.0 * max(init_gap, 0.0) / (lr * T)
        if grad_clip is not None:
            rate_bound = min(rate_bound, L_smooth * grad_clip**2 * lr / 2.0)
        return min(empirical_gap, rate_bound)

    return empirical_gap


def estimate_smoothness(
    a: np.ndarray,
    W: np.ndarray,
    Sigma: np.ndarray,
    Sigma_beta: np.ndarray,
    E_y2: float,
    maf: np.ndarray,
    activation: str = "relu",
    eps: float = 1e-4,
    n_probes: int = 10,
    rng: np.random.Generator | None = None,
) -> float:
    r"""Estimate the local Lipschitz constant of ∇L_EW by finite differences.

    Probes the gradient at random perturbations around (a, W) and returns
    the maximum observed ‖∇L(θ₁) - ∇L(θ₂)‖ / ‖θ₁ - θ₂‖.

    This is a practical lower bound on the true smoothness constant.

    Args:
        a, W: current network parameters.
        Sigma, Sigma_beta, E_y2, maf: summary statistics.
        activation: activation function name.
        eps: perturbation scale.
        n_probes: number of random directions to probe.
        rng: random generator.

    Returns:
        Estimated L_smooth (non-negative).
    """
    from .edgeworth_risk import compute_edgeworth_gradients
    from .cumulants import decorrelation_matrix as dm

    if rng is None:
        rng = np.random.default_rng(0)

    Sigma_inv_sqrt = dm(Sigma)

    ga0, gW0 = compute_edgeworth_gradients(
        a, W, Sigma, Sigma_beta, E_y2, maf, activation, Sigma_inv_sqrt
    )
    g0 = np.concatenate([ga0.ravel(), gW0.ravel()])

    max_L = 0.0
    for _ in range(n_probes):
        da = rng.standard_normal(a.shape) * eps
        dW = rng.standard_normal(W.shape) * eps

        ga1, gW1 = compute_edgeworth_gradients(
            a + da, W + dW, Sigma, Sigma_beta, E_y2, maf,
            activation, Sigma_inv_sqrt
        )
        g1 = np.concatenate([ga1.ravel(), gW1.ravel()])

        dg = np.linalg.norm(g1 - g0)
        dtheta = np.linalg.norm(np.concatenate([da.ravel(), dW.ravel()]))
        if dtheta > 0:
            max_L = max(max_L, dg / dtheta)

    return max_L


# =====================================================================
# Full decomposition
# =====================================================================

def compute_error_decomposition(
    a: np.ndarray,
    W: np.ndarray,
    Sigma: np.ndarray,
    Sigma_beta: np.ndarray,
    E_y2: float,
    maf: np.ndarray,
    N: int,
    n_train: int,
    delta_Sigma_fro: float = 0.0,
    loss_history: list[float] | np.ndarray | None = None,
    lr: float = 0.01,
    grad_clip: float | None = 1.0,
    activation: str = "relu",
) -> ErrorDecomposition:
    """Compute the full five-component error decomposition.

    Args:
        a: (m,) second-layer weights.
        W: (m, p) first-layer weight matrix.
        Sigma: (p, p) LD covariance matrix.
        Sigma_beta: (p,) = Sigma @ beta*.
        E_y2: scalar E[y²].
        maf: (p,) minor allele frequencies.
        N: Full GWAS sample size.
        n_train: Training subset size for PUMAS.
        delta_Sigma_fro: ‖Σ̂ - Σ‖_F (LD estimation error). 0 if Σ is exact.
        loss_history: from optimizer (optional; [] gives optimization=0).
        lr: learning rate used in training.
        grad_clip: gradient clipping norm used in training.
        activation: activation function name.

    Returns:
        ErrorDecomposition with all five bounds.
    """
    Sigma_inv_sqrt = decorrelation_matrix(Sigma)

    trunc = edgeworth_truncation_bound(
        maf, W, Sigma, Sigma_inv_sqrt, activation
    )

    decorr = decorrelation_bound(maf, W, Sigma, Sigma_inv_sqrt)

    ld_err = ld_estimation_bound(
        W, a, Sigma, Sigma_beta, delta_Sigma_fro, activation
    )

    pumas_var = pumas_variance_bound(
        Sigma, Sigma_beta, E_y2, N, n_train, W, a, activation
    )

    if loss_history is not None and len(loss_history) > 0:
        opt_err = optimization_bound(loss_history, lr, grad_clip)
    else:
        opt_err = 0.0

    return ErrorDecomposition(
        edgeworth_truncation=trunc,
        decorrelation_approx=decorr,
        ld_estimation=ld_err,
        pumas_variance=pumas_var,
        optimization=opt_err,
    )
