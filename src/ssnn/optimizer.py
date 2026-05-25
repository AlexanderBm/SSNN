"""
Gradient descent optimizer for the summary-statistics neural network.

Trains a 1-hidden-layer network f(x) = sum_k a_k sigma(w_k^T x) by
minimizing the population risk L(a, W) computed entirely from summary
statistics (Sigma, Sigma_beta).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .population_risk import compute_loss, compute_grad_a, compute_grad_W


@dataclass
class TrainResult:
    """Result of training the summary-stat neural network."""
    a: np.ndarray
    W: np.ndarray
    loss_history: list[float] = field(default_factory=list)
    converged: bool = False
    n_iters: int = 0


def train(
    Sigma: np.ndarray,
    Sigma_beta: np.ndarray,
    E_y2: float,
    m: int = 5,
    activation: str = "relu",
    lr: float = 0.01,
    max_iters: int = 5000,
    tol: float = 1e-8,
    init_scale: float = 0.01,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
    reg_a: float = 0.0,
    reg_W: float = 0.0,
    Cov_ref: np.ndarray | None = None,
) -> TrainResult:
    """Train a 1-hidden-layer NN on summary statistics via gradient descent.

    Args:
        Sigma: (p, p) LD covariance matrix (used for Stein/E[yf] terms).
        Sigma_beta: (p,) = Sigma @ beta*.
        E_y2: scalar E[y^2].
        m: Number of hidden units.
        activation: Activation function name.
        lr: Learning rate.
        max_iters: Maximum number of gradient descent iterations.
        tol: Convergence tolerance on relative loss change.
        init_scale: Scale for random weight initialization.
        rng: Random generator for initialization.
        verbose: Print progress every 500 iterations.
        reg_a: L2 regularization strength for second-layer weights.
        reg_W: L2 regularization strength for first-layer weights.
        Cov_ref: (p, p) empirical covariance for E[f^2] term (corrects
            binomial bias). Uses Sigma when None.

    Returns:
        TrainResult with optimized (a, W) and loss history.
    """
    if rng is None:
        rng = np.random.default_rng()

    p = Sigma.shape[0]
    W = rng.standard_normal((m, p)) * init_scale
    a = rng.standard_normal(m) * init_scale

    loss_history = []
    converged = False

    for i in range(max_iters):
        loss = compute_loss(a, W, Sigma, Sigma_beta, E_y2, activation, reg_a, reg_W, Cov_ref)
        loss_history.append(loss)

        if verbose and i % 500 == 0:
            print(f"  iter {i:5d}  loss = {loss:.8f}")

        if i > 0:
            rel_change = abs(loss_history[-1] - loss_history[-2]) / (abs(loss_history[-2]) + 1e-30)
            if rel_change < tol:
                converged = True
                break

        grad_a = compute_grad_a(a, W, Sigma, Sigma_beta, activation, reg_a, Cov_ref)
        grad_W = compute_grad_W(a, W, Sigma, Sigma_beta, activation, reg_W, Cov_ref)

        a = a - lr * grad_a
        W = W - lr * grad_W

    if not converged:
        loss = compute_loss(a, W, Sigma, Sigma_beta, E_y2, activation, reg_a, reg_W, Cov_ref)
        loss_history.append(loss)

    return TrainResult(
        a=a,
        W=W,
        loss_history=loss_history,
        converged=converged,
        n_iters=len(loss_history),
    )
