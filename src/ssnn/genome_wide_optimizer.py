"""
Genome-wide wrapper for Gaussian NN training with per-block objective scaling.

Fixes the E[y²] cross-block dilution problem by scaling each block's training
objective to match its estimated heritability share rather than the total
phenotypic variance.
"""

from __future__ import annotations

import numpy as np

from .optimizer import train, TrainResult
from .population_risk import _compute_E_y_f, _compute_E_f_squared
from .utils import nn_predict


def compute_beta_hats(
    Sigma_blocks: list[np.ndarray],
    Sigma_beta_blocks: list[np.ndarray],
    ridge_lambda: float = 1.0,
) -> list[np.ndarray]:
    """Compute per-block ridge regression weights.

    beta_hat_b = (Sigma_b + lambda*I)^{-1} Sigma_beta_b
    """
    B = len(Sigma_blocks)
    beta_hats = []
    for b in range(B):
        p = Sigma_blocks[b].shape[0]
        beta_hats.append(
            np.linalg.solve(
                Sigma_blocks[b] + ridge_lambda * np.eye(p),
                Sigma_beta_blocks[b],
            )
        )
    return beta_hats


def train_genome_wide(
    Sigma_blocks: list[np.ndarray],
    Sigma_beta_blocks: list[np.ndarray],
    E_y2: float,
    Cov_ref_blocks: list[np.ndarray],
    beta_hat_blocks: list[np.ndarray],
    m: int = 3,
    activation: str = "relu",
    lr: float = 0.05,
    max_iters: int = 500,
    base_reg_W: float = 0.01,
    base_reg_a: float = 0.01,
    init_scale: float = 0.01,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
) -> tuple[list[TrainResult], float]:
    """Train per-block Gaussian NNs with per-block-scaled objectives.

    The four regularization mechanisms (in order of importance):

    1. Per-block E[y²] scaling: Train block b with E_y2_b = max(h2_b_est, 0.01*E_y2)
       instead of the global E_y2. This is the key fix.
    2. Adaptive regularization: reg_W and reg_a inversely proportional to block
       heritability fraction.
    3. Post-training global shrinkage: compute c* that minimizes E[(y - c*F)²].
    4. Per-block linear fallback: if NN loss > linear PRS loss for a block, replace
       that block's TrainResult with a linear-equivalent result.

    Args:
        Sigma_blocks: list of (p_b, p_b) LD covariance matrices.
        Sigma_beta_blocks: list of (p_b,) marginal association vectors E[x_b y].
        E_y2: global E[y²] (genome-wide phenotypic variance).
        Cov_ref_blocks: list of (p_b, p_b) empirical reference panel covariances.
        beta_hat_blocks: list of (p_b,) pre-computed ridge weights.
        m: hidden units per block.
        activation: activation function.
        lr, max_iters: passed to train().
        base_reg_W, base_reg_a: base regularization strengths (scaled by 1/h2_frac).
        init_scale: weight initialization scale.
        rng: random number generator.
        verbose: print per-block progress.

    Returns:
        (per_block_results, c_star) where c_star is the global shrinkage factor
        to apply as: final_prediction = c_star * sum_b f_b(x_b)
    """
    if rng is None:
        rng = np.random.default_rng()

    B = len(Sigma_blocks)
    per_block_results: list[TrainResult] = []

    for b in range(B):
        # Estimate per-block heritability
        h2_b_est = max(0.0, float(np.dot(Sigma_beta_blocks[b], beta_hat_blocks[b])))
        # Per-block E[y²]: clamp to at least 5% of global to avoid over-shrinking predictions
        E_y2_b = max(h2_b_est, 0.05 * E_y2)
        # Heritability fraction for adaptive regularization
        h2_frac_b = h2_b_est / max(E_y2, 1e-15)
        # Adaptive regularization: boost regularization for weak-signal blocks
        reg_W_b = base_reg_W / max(h2_frac_b, 0.01)
        reg_a_b = base_reg_a / max(h2_frac_b, 0.01)

        b_rng = np.random.default_rng(rng.integers(2**32) if rng is not None else b)

        result = train(
            Sigma=Sigma_blocks[b],
            Sigma_beta=Sigma_beta_blocks[b],
            E_y2=E_y2_b,          # key fix: per-block scaled objective
            m=m,
            activation=activation,
            lr=lr,
            max_iters=max_iters,
            init_scale=init_scale,
            rng=b_rng,
            reg_W=reg_W_b,
            reg_a=reg_a_b,
            Cov_ref=Cov_ref_blocks[b],
        )

        if verbose:
            final_loss = result.loss_history[-1] if result.loss_history else float("nan")
            print(
                f"  Block {b:3d}: h2_est={h2_b_est:.4f}  E_y2_b={E_y2_b:.4f}"
                f"  reg_W={reg_W_b:.4f}  loss={final_loss:.6f}"
                f"  converged={result.converged}"
            )

        # Linear fallback: check if the NN is actually helping this block.
        # Two conditions trigger the fallback:
        #   (a) NN training loss exceeds the linear PRS residual (NN worse than linear in objective)
        #   (b) E[y * f_b] <= 0 (NN is anti-correlated with y — always harmful)
        linear_loss_b = max(0.0, E_y2_b - h2_b_est)
        nn_loss_b = result.loss_history[-1] if result.loss_history else E_y2_b

        e_yf_b = _compute_E_y_f(
            result.a, result.W, Sigma_blocks[b], Sigma_beta_blocks[b], activation
        )
        nn_anticorrelated = e_yf_b <= 0.0

        if nn_loss_b > linear_loss_b + 1e-6 or nn_anticorrelated:
            # NN failed to beat linear or is anti-correlated with y.
            # Represent beta_hat^T x exactly using a 2-neuron relu NN:
            #   relu(w^T x) - relu(-w^T x) = w^T x  for any w (exact identity).
            # This works regardless of the activation used in predict_genome_wide.
            bh = beta_hat_blocks[b]
            a_lin = np.array([1.0, -1.0])
            W_lin = np.stack([bh, -bh], axis=0)  # shape (2, p_b)
            result = TrainResult(
                a=a_lin,
                W=W_lin,
                loss_history=[linear_loss_b],
                converged=True,
            )
            if verbose:
                reason = "anti-correlated" if nn_anticorrelated else f"nn_loss={nn_loss_b:.6f} > linear_loss={linear_loss_b:.6f}"
                print(f"    Block {b}: fallback to linear ({reason})")

        per_block_results.append(result)

    # Global shrinkage: find c* = argmin E[(y - c*F)²]
    # c* = E[y F] / E[F²]  where F = sum_b f_b(x_b)
    # Under independence across blocks:
    #   E[y F] = sum_b E[y f_b] (by Stein / linearity)
    #   E[F²] = sum_b E[f_b²]  (cross-block terms vanish when blocks are independent)
    numerator = 0.0
    denominator = 0.0
    for b, result in enumerate(per_block_results):
        numerator += _compute_E_y_f(
            result.a, result.W, Sigma_blocks[b], Sigma_beta_blocks[b], activation
        )
        denominator += _compute_E_f_squared(
            result.a, result.W, Sigma_blocks[b], activation, Cov_ref_blocks[b]
        )

    if denominator > 1e-15:
        c_star = float(np.clip(numerator / denominator, 0.0, 5.0))
    else:
        c_star = 1.0

    if verbose:
        print(f"  c_star = {c_star:.4f}  (numerator={numerator:.6f}, denominator={denominator:.6f})")

    return per_block_results, c_star


def predict_genome_wide(
    X_test_blocks: list[np.ndarray],
    per_block_results: list[TrainResult],
    c_star: float,
    activation: str = "relu",
) -> np.ndarray:
    """Generate genome-wide predictions applying the global shrinkage factor.

    Args:
        X_test_blocks: list of (n_test, p_b) genotype matrices (centered).
        per_block_results: per-block TrainResult objects from train_genome_wide.
        c_star: global shrinkage factor from train_genome_wide.
        activation: activation function name.

    Returns:
        (n_test,) array of predictions.
    """
    pred = sum(
        nn_predict(X_test_blocks[b], per_block_results[b].a, per_block_results[b].W, activation)
        for b in range(len(X_test_blocks))
    )
    return c_star * pred
