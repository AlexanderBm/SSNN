"""
Test: Nonlinear phenotype simulation.

Verifies that when the true data-generating process is nonlinear
(y = beta*^T x + gamma * relu(w*^T x) + eps), the SSNN codebase
behaves correctly:

1. **Oracle check**: An individual-level NN (trained on raw genotypes)
   achieves higher R^2 than the optimal linear PRS -- confirming the
   nonlinear signal is real and recoverable.

2. **Gaussian ceiling under nonlinear DGP**: The summary-stat Gaussian
   NN cannot meaningfully beat the linear model even when the true DGP
   is nonlinear, because the Gaussian population risk remains minimised
   at the best linear predictor (Proposition 2 extends to misspecified
   phenotype models as long as the genotype model is Gaussian).

3. **Edgeworth advantage**: With discrete Binomial(2,p) genotypes and
   mixed MAFs, the Edgeworth NN (which accounts for genotype skewness)
   achieves loss at least as low as the Gaussian NN.

4. **Sanity checks**: The nonlinear phenotype generator produces the
   intended variance decomposition.
"""

import numpy as np
import pytest

from ssnn.simulation import (
    generate_binomial_genotypes,
    generate_maf_spectrum,
    compute_summary_stats_from_genotypes,
    train_oracle_nn,
)
from ssnn.optimizer import train
from ssnn.edgeworth_optimizer import train_edgeworth
from ssnn.utils import (
    generate_ld_matrix,
    linear_prs_weights,
    prediction_r2,
    nn_prediction_r2,
)


# ===================================================================
# Helpers
# ===================================================================

def _generate_nonlinear_phenotype(
    X: np.ndarray,
    beta_star: np.ndarray,
    w_star: np.ndarray,
    gamma: float,
    sigma_eps: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """y = beta*^T x + gamma * relu(w*^T x) + eps."""
    linear_part = X @ beta_star
    nonlinear_part = gamma * np.maximum(0.0, X @ w_star)
    eps = rng.normal(0, sigma_eps, size=X.shape[0])
    return linear_part + nonlinear_part + eps


def _calibrate_gamma(
    X: np.ndarray,
    beta_star: np.ndarray,
    w_star: np.ndarray,
    target_nonlinear_frac: float,
) -> float:
    """Scale gamma so Var(gamma * relu(w*^T x)) / Var(beta*^T x) == target."""
    var_linear = np.var(X @ beta_star)
    relu_vals = np.maximum(0.0, X @ w_star)
    var_relu = np.var(relu_vals)
    if var_relu < 1e-15:
        return 0.0
    return np.sqrt(target_nonlinear_frac * var_linear / var_relu)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture(scope="module")
def nonlinear_setup():
    """Shared genotype and phenotype data for all nonlinear phenotype tests.

    Generates one dataset under a nonlinear DGP with both training and
    test splits, along with summary statistics and a linear-only
    reference dataset for comparison.
    """
    rng = np.random.default_rng(2024)
    p = 25
    m = 5
    n_train = 8000
    n_test = 3000
    heritability = 0.5
    nonlinear_frac = 0.25

    maf = generate_maf_spectrum(p, "mixed", rng)
    Sigma = generate_ld_matrix(p, decay=0.5)

    beta_star = rng.standard_normal(p) * 0.3
    w_star = rng.standard_normal(p) * 0.3

    X_train_raw = generate_binomial_genotypes(n_train, maf, Sigma, rng)
    X_test_raw = generate_binomial_genotypes(n_test, maf, Sigma, rng)

    train_means = X_train_raw.mean(axis=0)
    X_train = X_train_raw - train_means
    X_test = X_test_raw - train_means

    gamma = _calibrate_gamma(X_train, beta_star, w_star, nonlinear_frac)

    var_linear = np.var(X_train @ beta_star)
    var_nonlinear = np.var(gamma * np.maximum(0.0, X_train @ w_star))
    total_genetic_var = var_linear + var_nonlinear

    sigma_eps_lin = np.sqrt(var_linear * (1 - heritability) / heritability)
    sigma_eps_nl = np.sqrt(total_genetic_var * (1 - heritability) / heritability)

    # Linear phenotypes
    y_train_lin = X_train @ beta_star + rng.normal(0, sigma_eps_lin, n_train)
    y_test_lin = X_test @ beta_star + rng.normal(0, sigma_eps_lin, n_test)

    # Nonlinear phenotypes
    y_train_nl = _generate_nonlinear_phenotype(
        X_train, beta_star, w_star, gamma, sigma_eps_nl, rng,
    )
    y_test_nl = _generate_nonlinear_phenotype(
        X_test, beta_star, w_star, gamma, sigma_eps_nl, rng,
    )

    stats_lin = compute_summary_stats_from_genotypes(X_train_raw, y_train_lin, Sigma)
    stats_nl = compute_summary_stats_from_genotypes(X_train_raw, y_train_nl, Sigma)

    return {
        "p": p,
        "m": m,
        "Sigma": Sigma,
        "maf": maf,
        "X_train": X_train,
        "X_test": X_test,
        "y_train_lin": y_train_lin,
        "y_test_lin": y_test_lin,
        "y_train_nl": y_train_nl,
        "y_test_nl": y_test_nl,
        "stats_lin": stats_lin,
        "stats_nl": stats_nl,
        "gamma": gamma,
        "beta_star": beta_star,
        "w_star": w_star,
        "sigma_eps_nl": sigma_eps_nl,
    }


# ===================================================================
# Test 1: Oracle NN beats linear on nonlinear DGP
# ===================================================================

class TestOracleBeatsLinear:
    """An individual-level NN should beat linear PRS under nonlinear DGP."""

    def test_oracle_nn_outperforms_linear(self, nonlinear_setup):
        s = nonlinear_setup

        beta_lin = linear_prs_weights(s["Sigma"], s["stats_nl"]["Sigma_beta_hat"])
        r2_linear = prediction_r2(s["X_test"], s["y_test_nl"], beta_lin)

        oracle_a, oracle_W, _ = train_oracle_nn(
            s["X_train"], s["y_train_nl"],
            m=s["m"],
            activation="relu",
            lr=0.01,
            max_iters=5000,
            batch_size=256,
            rng=np.random.default_rng(42),
        )
        r2_oracle = nn_prediction_r2(
            s["X_test"], s["y_test_nl"], oracle_a, oracle_W, "relu",
        )

        assert r2_oracle > r2_linear + 0.02, (
            f"Oracle NN R^2 = {r2_oracle:.4f} should meaningfully beat "
            f"linear R^2 = {r2_linear:.4f}"
        )

    def test_oracle_nn_gains_larger_on_nonlinear_dgp(self, nonlinear_setup):
        """The oracle NN's advantage over linear should be larger under
        the nonlinear DGP than under the linear DGP.  Both can exceed
        linear (since discrete genotypes are non-Gaussian), but the
        nonlinear DGP gives the NN strictly more signal to exploit.
        """
        s = nonlinear_setup

        # Oracle on nonlinear DGP (already tested above, but compute the gap)
        beta_lin_nl = linear_prs_weights(s["Sigma"], s["stats_nl"]["Sigma_beta_hat"])
        r2_linear_nl = prediction_r2(s["X_test"], s["y_test_nl"], beta_lin_nl)
        oracle_a_nl, oracle_W_nl, _ = train_oracle_nn(
            s["X_train"], s["y_train_nl"],
            m=s["m"], activation="relu", lr=0.01, max_iters=5000,
            batch_size=256, rng=np.random.default_rng(42),
        )
        r2_oracle_nl = nn_prediction_r2(
            s["X_test"], s["y_test_nl"], oracle_a_nl, oracle_W_nl, "relu",
        )
        gap_nl = r2_oracle_nl - r2_linear_nl

        # Oracle on linear DGP
        beta_lin_lin = linear_prs_weights(s["Sigma"], s["stats_lin"]["Sigma_beta_hat"])
        r2_linear_lin = prediction_r2(s["X_test"], s["y_test_lin"], beta_lin_lin)
        oracle_a_lin, oracle_W_lin, _ = train_oracle_nn(
            s["X_train"], s["y_train_lin"],
            m=s["m"], activation="relu", lr=0.01, max_iters=5000,
            batch_size=256, rng=np.random.default_rng(42),
        )
        r2_oracle_lin = nn_prediction_r2(
            s["X_test"], s["y_test_lin"], oracle_a_lin, oracle_W_lin, "relu",
        )
        gap_lin = r2_oracle_lin - r2_linear_lin

        assert gap_nl > gap_lin, (
            f"Oracle-vs-linear gap should be larger for nonlinear DGP "
            f"({gap_nl:.4f}) than linear DGP ({gap_lin:.4f})"
        )


# ===================================================================
# Test 2: Gaussian NN ceiling under nonlinear DGP
# ===================================================================

class TestGaussianCeilingNonlinearDGP:
    """The Gaussian summary-stat NN should not beat linear, even on nonlinear DGP."""

    def test_gaussian_nn_does_not_beat_linear_on_nonlinear_dgp(self, nonlinear_setup):
        s = nonlinear_setup

        beta_lin = linear_prs_weights(s["Sigma"], s["stats_nl"]["Sigma_beta_hat"])
        r2_linear = prediction_r2(s["X_test"], s["y_test_nl"], beta_lin)

        n_reps = 3
        gaps = []
        for seed in range(n_reps):
            gauss = train(
                s["Sigma"], s["stats_nl"]["Sigma_beta_hat"], s["stats_nl"]["E_y2_hat"],
                m=s["m"],
                activation="relu",
                lr=0.005,
                max_iters=5000,
                tol=1e-10,
                init_scale=0.01,
                rng=np.random.default_rng(100 + seed * 17),
            )
            r2_gauss = nn_prediction_r2(
                s["X_test"], s["y_test_nl"], gauss.a, gauss.W, "relu",
            )
            gaps.append(r2_gauss - r2_linear)

        mean_gap = np.mean(gaps)
        assert mean_gap < 0.03, (
            f"Gaussian NN beat linear by {mean_gap:.4f} on average "
            f"under nonlinear DGP (should be < 0.03, i.e. ceiling holds). "
            f"Per-rep gaps: {gaps}"
        )

    def test_gaussian_nn_does_not_beat_linear_on_linear_dgp(self, nonlinear_setup):
        s = nonlinear_setup

        beta_lin = linear_prs_weights(s["Sigma"], s["stats_lin"]["Sigma_beta_hat"])
        r2_linear = prediction_r2(s["X_test"], s["y_test_lin"], beta_lin)

        gauss = train(
            s["Sigma"], s["stats_lin"]["Sigma_beta_hat"], s["stats_lin"]["E_y2_hat"],
            m=s["m"],
            activation="relu",
            lr=0.005,
            max_iters=5000,
            tol=1e-10,
            init_scale=0.01,
            rng=np.random.default_rng(42),
        )
        r2_gauss = nn_prediction_r2(
            s["X_test"], s["y_test_lin"], gauss.a, gauss.W, "relu",
        )

        assert r2_gauss <= r2_linear + 0.02, (
            f"Linear DGP: Gaussian NN R^2 = {r2_gauss:.4f} should not exceed "
            f"linear R^2 = {r2_linear:.4f} by more than 0.02"
        )


# ===================================================================
# Test 3: Edgeworth NN has lower loss than Gaussian on skewed genotypes
# ===================================================================

class TestEdgeworthAdvantage:
    """The Edgeworth NN should achieve loss at least as low as the Gaussian NN."""

    def test_edgeworth_loss_not_worse_than_gaussian(self, nonlinear_setup):
        """With mixed MAFs, the Edgeworth-corrected loss should be <= Gaussian."""
        s = nonlinear_setup

        gauss = train(
            s["Sigma"], s["stats_nl"]["Sigma_beta_hat"], s["stats_nl"]["E_y2_hat"],
            m=s["m"],
            activation="relu",
            lr=0.005,
            max_iters=5000,
            tol=1e-10,
            init_scale=0.01,
            rng=np.random.default_rng(42),
        )

        ew = train_edgeworth(
            s["Sigma"], s["stats_nl"]["Sigma_beta_hat"], s["stats_nl"]["E_y2_hat"],
            s["stats_nl"]["maf"],
            m=s["m"],
            activation="relu",
            lr=0.0005,
            max_iters=2500,
            tol=1e-10,
            rng=np.random.default_rng(43),
            loss_floor=0.0,
            grad_clip=0.5,
            max_backtracks=10,
            a_init=gauss.a,
            W_init=gauss.W,
        )

        assert ew.loss_history[-1] <= gauss.loss_history[-1] + 0.01, (
            f"Edgeworth final loss = {ew.loss_history[-1]:.6f} should be <= "
            f"Gaussian final loss = {gauss.loss_history[-1]:.6f} + tolerance"
        )

    def test_edgeworth_r2_not_worse_than_gaussian(self, nonlinear_setup):
        """Edgeworth NN should not lose to Gaussian NN on prediction R^2."""
        s = nonlinear_setup

        gauss = train(
            s["Sigma"], s["stats_nl"]["Sigma_beta_hat"], s["stats_nl"]["E_y2_hat"],
            m=s["m"],
            activation="relu",
            lr=0.005,
            max_iters=5000,
            tol=1e-10,
            init_scale=0.01,
            rng=np.random.default_rng(42),
        )

        ew = train_edgeworth(
            s["Sigma"], s["stats_nl"]["Sigma_beta_hat"], s["stats_nl"]["E_y2_hat"],
            s["stats_nl"]["maf"],
            m=s["m"],
            activation="relu",
            lr=0.0005,
            max_iters=2500,
            tol=1e-10,
            rng=np.random.default_rng(43),
            loss_floor=0.0,
            grad_clip=0.5,
            max_backtracks=10,
            a_init=gauss.a,
            W_init=gauss.W,
        )

        r2_gauss = nn_prediction_r2(
            s["X_test"], s["y_test_nl"], gauss.a, gauss.W, "relu",
        )
        r2_ew = nn_prediction_r2(
            s["X_test"], s["y_test_nl"], ew.a, ew.W, "relu",
        )

        assert r2_ew >= r2_gauss - 0.02, (
            f"Edgeworth R^2 = {r2_ew:.4f} should not be much worse than "
            f"Gaussian R^2 = {r2_gauss:.4f}"
        )


# ===================================================================
# Test 4: Sanity checks
# ===================================================================

class TestNonlinearPhenotypeGenerator:
    """Sanity checks on the nonlinear DGP setup."""

    def test_gamma_is_positive(self, nonlinear_setup):
        assert nonlinear_setup["gamma"] > 0.0

    def test_nonlinear_variance_fraction(self, nonlinear_setup):
        """The ReLU component should contribute ~25% of the genetic variance."""
        s = nonlinear_setup
        rng = np.random.default_rng(9999)
        X = generate_binomial_genotypes(5000, s["maf"], s["Sigma"], rng)
        X = X - X.mean(axis=0)

        var_lin = np.var(X @ s["beta_star"])
        var_nl = np.var(s["gamma"] * np.maximum(0.0, X @ s["w_star"]))
        frac = var_nl / (var_lin + 1e-30)

        assert 0.10 < frac < 0.50, (
            f"Nonlinear variance fraction = {frac:.3f}, expected ~0.25"
        )

    def test_nonlinear_dgp_has_higher_E_y2(self, nonlinear_setup):
        """E[y^2] under nonlinear DGP should be >= linear DGP (extra variance from relu)."""
        s = nonlinear_setup
        assert s["stats_nl"]["E_y2_hat"] >= s["stats_lin"]["E_y2_hat"] * 0.95, (
            f"E_y2 nonlinear = {s['stats_nl']['E_y2_hat']:.4f}, "
            f"E_y2 linear = {s['stats_lin']['E_y2_hat']:.4f}; "
            f"the nonlinear DGP adds variance so E_y2_nl should be >= E_y2_lin"
        )


# ===================================================================
# Test 5: Summary statistics sanity checks
# ===================================================================

class TestSummaryStatsSanity:
    """Verify that summary statistics from the nonlinear DGP are well-formed
    and that the nonlinear component actually perturbs them relative to the
    linear DGP.
    """

    def test_sigma_beta_hat_nonzero(self, nonlinear_setup):
        """Sigma_beta_hat from nonlinear DGP should not be degenerate."""
        s = nonlinear_setup
        sb = s["stats_nl"]["Sigma_beta_hat"]
        assert np.linalg.norm(sb) > 0.01, (
            f"Sigma_beta_hat norm = {np.linalg.norm(sb):.6f}, suspiciously small"
        )

    def test_sigma_beta_hat_correlates_with_truth(self, nonlinear_setup):
        """Sigma_beta_hat should correlate with Sigma @ beta_star (the linear
        component of the true signal dominates since nonlinear_frac = 0.25)."""
        s = nonlinear_setup
        sb_hat = s["stats_nl"]["Sigma_beta_hat"]
        sb_true = s["Sigma"] @ s["beta_star"]
        cosine = np.dot(sb_hat, sb_true) / (
            np.linalg.norm(sb_hat) * np.linalg.norm(sb_true) + 1e-30
        )
        assert cosine > 0.5, (
            f"Cosine similarity between Sigma_beta_hat and Sigma @ beta_star "
            f"= {cosine:.4f}, expected > 0.5 since linear component dominates"
        )

    def test_nonlinear_dgp_changes_summary_stats(self, nonlinear_setup):
        """The nonlinear DGP should produce different summary stats than the
        linear DGP -- confirming the relu component enters Sigma_beta_hat."""
        s = nonlinear_setup
        sb_nl = s["stats_nl"]["Sigma_beta_hat"]
        sb_lin = s["stats_lin"]["Sigma_beta_hat"]
        diff_norm = np.linalg.norm(sb_nl - sb_lin)
        avg_norm = 0.5 * (np.linalg.norm(sb_nl) + np.linalg.norm(sb_lin))
        rel_diff = diff_norm / (avg_norm + 1e-30)
        assert rel_diff > 0.01, (
            f"Relative difference in Sigma_beta_hat between nonlinear and "
            f"linear DGP = {rel_diff:.6f}, expected > 0.01"
        )


# ===================================================================
# Test 6: Gaussian NN loss convergence check
# ===================================================================

class TestGaussianNNConvergence:
    """Verify the Gaussian NN loss actually converges close to the
    irreducible linear-approximation error, not just that R^2 doesn't
    beat linear."""

    def test_gaussian_nn_loss_converges(self, nonlinear_setup):
        """The final Gaussian NN loss should be close to the best linear
        approximation error: L* = E[y^2] - Sigma_beta^T Sigma^{-1} Sigma_beta."""
        s = nonlinear_setup
        from ssnn.population_risk import compute_loss

        Sigma = s["Sigma"]
        sb = s["stats_nl"]["Sigma_beta_hat"]
        E_y2 = s["stats_nl"]["E_y2_hat"]

        beta_lin = linear_prs_weights(Sigma, sb)
        linear_loss = E_y2 - float(sb @ beta_lin)

        gauss = train(
            Sigma, sb, E_y2,
            m=s["m"],
            activation="relu",
            lr=0.005,
            max_iters=5000,
            tol=1e-10,
            init_scale=0.01,
            rng=np.random.default_rng(42),
        )

        nn_loss = compute_loss(gauss.a, gauss.W, Sigma, sb, E_y2, "relu")

        assert nn_loss < linear_loss * 1.15, (
            f"Gaussian NN final loss = {nn_loss:.6f} should be close to "
            f"linear loss = {linear_loss:.6f} (within 15%)"
        )


# ===================================================================
# Test 7: Variance decomposition on fixture data
# ===================================================================

class TestVarianceDecompositionOnFixture:
    """Verify the nonlinear variance fraction on the actual fixture data,
    not just a fresh sample."""

    def test_variance_fraction_on_fixture_data(self, nonlinear_setup):
        """The relu component should contribute ~25% of the genetic variance
        on the actual training data used by all other tests."""
        s = nonlinear_setup
        var_lin = np.var(s["X_train"] @ s["beta_star"])
        var_nl = np.var(s["gamma"] * np.maximum(0.0, s["X_train"] @ s["w_star"]))
        frac = var_nl / (var_lin + var_nl + 1e-30)
        assert 0.10 < frac < 0.50, (
            f"Nonlinear variance fraction on fixture data = {frac:.3f}, "
            f"expected ~0.20 (calibrated for 0.25 ratio to linear, "
            f"which is ~0.20 fraction of total)"
        )


# ===================================================================
# Test 8: Oracle NN approximates the true model
# ===================================================================

class TestOracleApproximatesTrueModel:
    """Verify the oracle NN is learning something close to the actual
    data-generating function, not just beating linear by accident."""

    def test_oracle_r2_approaches_true_model_ceiling(self, nonlinear_setup):
        """The true-model oracle predictor y_hat = beta*^T x + gamma * relu(w*^T x)
        achieves the heritability R^2 ceiling. The trained oracle NN should
        achieve at least 70% of this ceiling."""
        s = nonlinear_setup
        y_pred_true = (
            s["X_test"] @ s["beta_star"]
            + s["gamma"] * np.maximum(0.0, s["X_test"] @ s["w_star"])
        )
        ss_res_true = np.mean((s["y_test_nl"] - y_pred_true) ** 2)
        ss_tot = np.var(s["y_test_nl"])
        r2_true_model = 1.0 - ss_res_true / ss_tot

        oracle_a, oracle_W, _ = train_oracle_nn(
            s["X_train"], s["y_train_nl"],
            m=s["m"],
            activation="relu",
            lr=0.01,
            max_iters=5000,
            batch_size=256,
            rng=np.random.default_rng(42),
        )
        r2_oracle = nn_prediction_r2(
            s["X_test"], s["y_test_nl"], oracle_a, oracle_W, "relu",
        )

        assert r2_oracle > 0.70 * r2_true_model, (
            f"Oracle NN R^2 = {r2_oracle:.4f} should be >= 70% of "
            f"true model ceiling R^2 = {r2_true_model:.4f}"
        )


# ===================================================================
# Test 9: Edgeworth modifies the Stein cross-moment
# ===================================================================

class TestEdgeworthModifiesSteinCrossMoment:
    """Verify that the Edgeworth correction to E[sigma'(z)] is nonzero
    for skewed MAFs, confirming the theory about shifted optima."""

    def test_edgeworth_E_sigma_prime_differs_from_gaussian(self, nonlinear_setup):
        """With mixed MAFs, the Edgeworth-corrected E[sigma'(z)] should
        differ from the Gaussian E[sigma'(z)] = 0.5 for ReLU."""
        from ssnn.edgeworth_integrals import edgeworth_E_sigma_prime
        from ssnn.cumulants import projection_cumulants_ld, decorrelation_matrix
        from ssnn.gaussian_integrals import projection_variance

        s = nonlinear_setup
        Sigma = s["Sigma"]
        maf = s["stats_nl"]["maf"]
        S_inv_sqrt = decorrelation_matrix(Sigma)

        rng = np.random.default_rng(777)
        w_test = rng.standard_normal(s["p"]) * 0.3
        v = projection_variance(Sigma, w_test)
        kt3, kt4 = projection_cumulants_ld(w_test, maf, Sigma, S_inv_sqrt)

        E_sp_gauss = 0.5  # ReLU
        E_sp_ew = edgeworth_E_sigma_prime(v, kt3, kt4, "relu")

        assert E_sp_ew != pytest.approx(E_sp_gauss, abs=1e-6), (
            f"Edgeworth E[sigma'(z)] = {E_sp_ew:.8f} should differ from "
            f"Gaussian value 0.5 when MAFs are non-uniform "
            f"(kt3={kt3:.4f}, kt4={kt4:.4f})"
        )


# ===================================================================
# Test 10: _calibrate_gamma edge case
# ===================================================================

class TestCalibrateGammaEdgeCases:
    """Edge-case checks for the gamma calibration helper."""

    def test_near_zero_w_star_returns_zero(self):
        """When w_star ~ 0, relu(w*^T x) has near-zero variance,
        and _calibrate_gamma should return 0."""
        rng = np.random.default_rng(100)
        p = 10
        X = rng.standard_normal((500, p))
        beta = rng.standard_normal(p) * 0.3
        w_star_tiny = np.full(p, 1e-20)
        gamma = _calibrate_gamma(X, beta, w_star_tiny, 0.25)
        assert gamma == 0.0

    def test_calibration_gives_correct_ratio(self):
        """Calibrated gamma should produce the target variance ratio."""
        rng = np.random.default_rng(101)
        p = 10
        X = rng.standard_normal((5000, p))
        beta = rng.standard_normal(p) * 0.5
        w_star = rng.standard_normal(p) * 0.5
        target = 0.25

        gamma = _calibrate_gamma(X, beta, w_star, target)
        var_lin = np.var(X @ beta)
        var_nl = np.var(gamma * np.maximum(0.0, X @ w_star))
        achieved = var_nl / var_lin
        assert achieved == pytest.approx(target, rel=0.01), (
            f"Achieved ratio = {achieved:.4f}, target = {target:.4f}"
        )
