"""
Two-stage trainer combining per-block SSNNs with cross-block coefficients.

Stage 1 trains a Gaussian-population SSNN on each LD block independently
(using the standard `train` routine).  Stage 2 fits scalar coefficients
c_ab on top of the block predictors to capture cross-block epistatic
signal, restricted to a triage-filtered set of significant pairs.

Only the new cross-block modules and the existing `train`/`nn_predict`
helpers are used; no existing source files are modified.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .optimizer import train, TrainResult
from .utils import nn_predict
from .cross_block_stats import (
    compute_cross_block_scalars,
    triage_cross_pairs,
)
from .cross_block_risk import solve_cross_block_coefficients


def _per_block_prediction(
    X_block: np.ndarray,
    result: TrainResult,
    activation: str,
) -> np.ndarray:
    """Compute centered per-block prediction f_b(X_b)."""
    f = nn_predict(X_block, result.a, result.W, activation=activation)
    return f - f.mean()


def train_two_stage(
    Sigma_blocks: list[np.ndarray],
    Sigma_beta_blocks: list[np.ndarray],
    E_y2: float,
    Gamma_blocks: list[np.ndarray | None] | None,
    X_ref_blocks: list[np.ndarray],
    X_train_blocks: list[np.ndarray],
    y_train: np.ndarray,
    beta_hats: list[np.ndarray],
    n_train: int,
    m: int = 3,
    activation: str = "relu",
    lr: float = 0.05,
    max_iters: int = 300,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
) -> tuple[list[TrainResult], dict[tuple[int, int], float], np.ndarray]:
    """Train per-block SSNNs and then cross-block interaction coefficients.

    Args:
        Sigma_blocks: list of (p_b, p_b) LD covariances per block.
        Sigma_beta_blocks: list of (p_b,) marginal associations per block.
        E_y2: scalar phenotypic variance.
        Gamma_blocks: optional per-block interaction tensors (unused by the
            Stage-1 Gaussian trainer here, kept for API parity).
        X_ref_blocks: list of (n_ref, p_b) reference panels for Cov_ref.
        X_train_blocks: list of (n_train, p_b) training genotypes.
        y_train: (n_train,) centered training phenotype.
        beta_hats: list of (p_b,) ridge weights for the triage step.
        n_train: training sample size (matches len(y_train)).
        m: hidden units per block.
        activation: NN activation.
        lr, max_iters: passed to `train`.
        rng: numpy Generator for initialization.
        verbose: print Stage-1 progress.

    Returns:
        (per_block_results, c_dict, q_matrix).
    """
    if rng is None:
        rng = np.random.default_rng()

    B = len(Sigma_blocks)

    # Split training data: even indices → Stage 1 (NN assessment),
    # odd indices → Stage 2 (cross-block scalars, held-out to prevent overfitting).
    idx_s1 = np.arange(0, n_train, 2)
    idx_s2 = np.arange(1, n_train, 2)
    n_cross = len(idx_s2)
    X_s1_blocks = [X_train_blocks[b][idx_s1] for b in range(B)]
    X_s2_blocks = [X_train_blocks[b][idx_s2] for b in range(B)]
    y_s1 = y_train[idx_s1]
    y_s2 = y_train[idx_s2]

    per_block_results: list[TrainResult] = []
    per_block_Ef2: list[float] = []
    per_block_Eyf: list[float] = []

    for b in range(B):
        n_ref = len(X_ref_blocks[b])
        Cov_ref = X_ref_blocks[b].T @ X_ref_blocks[b] / max(n_ref, 1)
        result = train(
            Sigma=Sigma_blocks[b],
            Sigma_beta=Sigma_beta_blocks[b],
            E_y2=E_y2,
            m=m,
            activation=activation,
            lr=lr,
            max_iters=max_iters,
            rng=rng,
            verbose=False,
            Cov_ref=Cov_ref,
        )
        per_block_results.append(result)

        f_b_s1 = _per_block_prediction(X_s1_blocks[b], result, activation)
        per_block_Ef2.append(float(np.mean(f_b_s1 ** 2)))
        per_block_Eyf.append(float(np.mean(y_s1 * f_b_s1)))

        if verbose:
            print(f"  block {b:3d}  E[f^2]={per_block_Ef2[-1]:.4f}  E[yf]={per_block_Eyf[-1]:.4f}")

    # Stage 2: apply trained NNs to held-out data for unbiased cross-block estimates.
    f_s2_blocks = [
        _per_block_prediction(X_s2_blocks[b], per_block_results[b], activation)
        for b in range(B)
    ]
    per_block_Ef2_s2 = [float(np.mean(f ** 2)) for f in f_s2_blocks]
    E_y2_s2 = float(np.mean(y_s2 ** 2))

    # Triage: use LINEAR beta_hat projections so var_null formula is self-consistent.
    # q_linear[a,b] = E[y * (beta_a^T x_a) * (beta_b^T x_b)] on Stage 2 data.
    q_matrix_linear = compute_cross_block_scalars(X_s2_blocks, y_s2, beta_hats)
    linear_preds_s2 = [X_s2_blocks[b] @ beta_hats[b] for b in range(B)]
    per_block_Ef2_linear = [float(np.mean(lp ** 2)) for lp in linear_preds_s2]
    significant_pairs = triage_cross_pairs(
        q_matrix_linear, per_block_Ef2_linear, E_y2_s2, n_cross,
    )

    if verbose:
        print(f"  triage: {len(significant_pairs)} significant pair(s)")

    if not significant_pairs:
        return per_block_results, {}, q_matrix_linear

    # Coefficient solve: use NN predictions for both numerator and denominator.
    # c_ab* = E[y f_a^NN f_b^NN] / (E[f_a^NN^2] E[f_b^NN^2] + ridge).
    q_matrix_nn = np.zeros((B, B))
    for (a, b) in significant_pairs:
        q_ab = float(np.mean(y_s2 * f_s2_blocks[a] * f_s2_blocks[b]))
        q_matrix_nn[a, b] = q_ab
        q_matrix_nn[b, a] = q_ab

    c_dict = solve_cross_block_coefficients(
        per_block_Ef2_s2, q_matrix_nn, significant_pairs, E_y2=E_y2_s2, n=n_cross,
    )
    return per_block_results, c_dict, q_matrix_linear


def predict_two_stage(
    X_test_blocks: list[np.ndarray],
    per_block_results: list[TrainResult],
    c_dict: dict[tuple[int, int], float],
    activation: str = "relu",
) -> np.ndarray:
    """Predict using F(x) = sum_b f_b(x_b) + sum_{(a,b)} c_ab f_a f_b.

    Per-block predictions are individually mean-centered to match the
    centered-prediction assumption of the Stage-2 derivation.
    """
    B = len(X_test_blocks)
    n = X_test_blocks[0].shape[0]

    f_blocks = np.zeros((B, n))
    for b in range(B):
        f_blocks[b] = _per_block_prediction(X_test_blocks[b], per_block_results[b], activation)

    pred = f_blocks.sum(axis=0)
    for (a, b), c_ab in c_dict.items():
        pred = pred + c_ab * f_blocks[a] * f_blocks[b]
    return pred


@dataclass
class TwoStageResult:
    per_block_results: list[TrainResult]
    c_dict: dict[tuple[int, int], float]
    q_matrix: np.ndarray
