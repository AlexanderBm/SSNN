"""Tests for simulation.py: data generation, oracle NN, scenario runner."""

import numpy as np
import pytest

from ssnn.simulation import (
    generate_maf_spectrum,
    generate_binomial_genotypes,
    generate_effect_sizes,
    compute_summary_stats_from_genotypes,
    train_oracle_nn,
    SimulationScenario,
    ScenarioResult,
    _weight_cosine_similarity,
    run_single_rep,
    run_scenario,
)
from ssnn.edgeworth_optimizer import train_edgeworth
from ssnn.utils import generate_ld_matrix


# ===================================================================
# generate_maf_spectrum
# ===================================================================

class TestGenerateMAFSpectrum:

    @pytest.mark.parametrize("spectrum", ["common", "rare", "mixed"])
    def test_output_shape(self, spectrum):
        rng = np.random.default_rng(0)
        maf = generate_maf_spectrum(20, spectrum, rng)
        assert maf.shape == (20,)

    def test_common_range(self):
        rng = np.random.default_rng(1)
        maf = generate_maf_spectrum(100, "common", rng)
        assert np.all(maf >= 0.10)
        assert np.all(maf <= 0.50)

    def test_rare_range(self):
        rng = np.random.default_rng(2)
        maf = generate_maf_spectrum(100, "rare", rng)
        assert np.all(maf >= 0.01)
        assert np.all(maf <= 0.05)

    def test_mixed_has_all_bands(self):
        rng = np.random.default_rng(3)
        maf = generate_maf_spectrum(100, "mixed", rng)
        assert maf.shape == (100,)
        has_common = np.any(maf >= 0.10)
        has_lowfreq = np.any((maf >= 0.05) & (maf < 0.10))
        has_rare = np.any(maf < 0.05)
        assert has_common and has_lowfreq and has_rare

    def test_explicit_array_passthrough(self):
        explicit = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        rng = np.random.default_rng(4)
        result = generate_maf_spectrum(5, explicit, rng)
        np.testing.assert_array_equal(result, explicit)

    def test_explicit_array_wrong_shape_raises(self):
        explicit = np.array([0.1, 0.2, 0.3])
        rng = np.random.default_rng(5)
        with pytest.raises(ValueError, match="shape"):
            generate_maf_spectrum(5, explicit, rng)

    def test_unknown_spectrum_raises(self):
        rng = np.random.default_rng(6)
        with pytest.raises(ValueError, match="Unknown MAF spectrum"):
            generate_maf_spectrum(10, "ultracommon", rng)


# ===================================================================
# generate_binomial_genotypes
# ===================================================================

class TestGenerateBinomialGenotypes:

    def test_output_shape(self):
        rng = np.random.default_rng(10)
        maf = np.array([0.2, 0.3, 0.4])
        Sigma = generate_ld_matrix(3, n_blocks=1, decay=0.3)
        X = generate_binomial_genotypes(100, maf, Sigma, rng)
        assert X.shape == (100, 3)

    def test_values_in_012(self):
        rng = np.random.default_rng(11)
        p = 10
        maf = np.full(p, 0.3)
        Sigma = generate_ld_matrix(p, n_blocks=1, decay=0.3)
        X = generate_binomial_genotypes(500, maf, Sigma, rng)
        unique_vals = set(np.unique(X))
        assert unique_vals.issubset({0.0, 1.0, 2.0})

    def test_marginal_maf_matches_target(self):
        rng = np.random.default_rng(12)
        p = 5
        target_maf = np.array([0.1, 0.2, 0.3, 0.4, 0.45])
        Sigma = np.eye(p)
        n = 50_000
        X = generate_binomial_genotypes(n, target_maf, Sigma, rng)
        empirical_maf = X.mean(axis=0) / 2.0
        np.testing.assert_allclose(empirical_maf, target_maf, atol=0.02)

    def test_identity_sigma_uncorrelated(self):
        rng = np.random.default_rng(13)
        p = 5
        maf = np.full(p, 0.3)
        Sigma = np.eye(p)
        n = 20_000
        X = generate_binomial_genotypes(n, maf, Sigma, rng)
        corr = np.corrcoef(X.T)
        off_diag = corr[np.triu_indices(p, k=1)]
        assert np.all(np.abs(off_diag) < 0.05)

    def test_high_ld_nonzero_correlation(self):
        rng = np.random.default_rng(14)
        p = 5
        maf = np.full(p, 0.3)
        Sigma = generate_ld_matrix(p, n_blocks=1, decay=0.9)
        n = 20_000
        X = generate_binomial_genotypes(n, maf, Sigma, rng)
        corr = np.corrcoef(X.T)
        assert abs(corr[0, 1]) > 0.05


# ===================================================================
# generate_effect_sizes
# ===================================================================

class TestGenerateEffectSizes:

    def test_output_shape(self):
        rng = np.random.default_rng(20)
        p = 15
        Sigma = generate_ld_matrix(p, decay=0.5)
        beta, sigma_eps = generate_effect_sizes(p, Sigma, 0.5, 0.3, rng)
        assert beta.shape == (p,)
        assert isinstance(sigma_eps, float)

    def test_sparsity_pattern(self):
        rng = np.random.default_rng(21)
        p = 20
        sparsity = 0.25
        Sigma = generate_ld_matrix(p, decay=0.5)
        beta, _ = generate_effect_sizes(p, Sigma, 0.5, sparsity, rng)
        n_nonzero = np.count_nonzero(beta)
        expected_nonzero = max(1, int(sparsity * p))
        assert n_nonzero == expected_nonzero

    def test_minimum_one_causal(self):
        rng = np.random.default_rng(22)
        p = 10
        Sigma = generate_ld_matrix(p, decay=0.5)
        beta, _ = generate_effect_sizes(p, Sigma, 0.5, 0.0, rng)
        assert np.count_nonzero(beta) >= 1

    def test_achieved_heritability_matches_target(self):
        rng = np.random.default_rng(23)
        p = 15
        h2_target = 0.4
        Sigma = generate_ld_matrix(p, decay=0.5)
        beta, sigma_eps = generate_effect_sizes(p, Sigma, h2_target, 0.3, rng)
        genetic_var = beta @ Sigma @ beta
        h2_achieved = genetic_var / (genetic_var + sigma_eps ** 2)
        assert h2_achieved == pytest.approx(h2_target, rel=1e-10)

    @pytest.mark.parametrize("h2", [0.1, 0.3, 0.5, 0.8])
    def test_heritability_range(self, h2):
        rng = np.random.default_rng(24)
        p = 10
        Sigma = generate_ld_matrix(p, decay=0.5)
        beta, sigma_eps = generate_effect_sizes(p, Sigma, h2, 0.5, rng)
        genetic_var = beta @ Sigma @ beta
        h2_achieved = genetic_var / (genetic_var + sigma_eps ** 2)
        assert h2_achieved == pytest.approx(h2, rel=1e-10)


# ===================================================================
# compute_summary_stats_from_genotypes
# ===================================================================

class TestComputeSummaryStats:

    def test_returns_correct_keys(self):
        rng = np.random.default_rng(30)
        p = 5
        n = 100
        X = rng.choice([0, 1, 2], size=(n, p)).astype(float)
        y = rng.standard_normal(n)
        Sigma_ref = np.eye(p)
        stats = compute_summary_stats_from_genotypes(X, y, Sigma_ref)
        assert set(stats.keys()) == {"Sigma_beta_hat", "E_y2_hat", "Sigma", "maf", "Gamma_hat", "Cov_ref"}

    def test_sigma_beta_hat_shape(self):
        rng = np.random.default_rng(31)
        p = 8
        n = 200
        X = rng.choice([0, 1, 2], size=(n, p)).astype(float)
        y = rng.standard_normal(n)
        Sigma_ref = np.eye(p)
        stats = compute_summary_stats_from_genotypes(X, y, Sigma_ref)
        assert stats["Sigma_beta_hat"].shape == (p,)

    def test_large_n_sigma_beta_hat_approx_population(self):
        """With large n, Sigma_beta_hat should approximate Sigma @ beta_star."""
        rng = np.random.default_rng(32)
        p = 8
        n = 50_000
        maf = np.full(p, 0.3)
        Sigma = generate_ld_matrix(p, decay=0.5)
        beta_star = rng.standard_normal(p) * 0.3
        sigma_eps = 1.0

        X = generate_binomial_genotypes(n, maf, Sigma, rng)
        X_centered = X - X.mean(axis=0)
        y = X_centered @ beta_star + rng.normal(0, sigma_eps, n)

        stats = compute_summary_stats_from_genotypes(X, y, Sigma)

        empirical_Sigma = np.cov(X_centered.T)
        pop_value = empirical_Sigma @ beta_star
        np.testing.assert_allclose(
            stats["Sigma_beta_hat"], pop_value, atol=0.15
        )

    def test_maf_clipped(self):
        rng = np.random.default_rng(33)
        p = 3
        n = 100
        X = np.zeros((n, p))
        X[:, 2] = 2.0
        y = rng.standard_normal(n)
        Sigma_ref = np.eye(p)
        stats = compute_summary_stats_from_genotypes(X, y, Sigma_ref)
        assert np.all(stats["maf"] >= 0.01)
        assert np.all(stats["maf"] <= 0.99)


# ===================================================================
# train_oracle_nn
# ===================================================================

class TestTrainOracleNN:

    def test_loss_decreases(self):
        rng = np.random.default_rng(40)
        p = 10
        n = 500
        X = rng.standard_normal((n, p))
        beta = rng.standard_normal(p) * 0.5
        y = X @ beta + rng.normal(0, 1.0, n)

        a, W, loss_history = train_oracle_nn(
            X, y, m=5, activation="relu",
            lr=0.01, max_iters=300, batch_size=128, rng=np.random.default_rng(40),
        )
        assert loss_history[-1] < loss_history[0]

    def test_output_shapes(self):
        rng = np.random.default_rng(41)
        p, n, m = 8, 200, 4
        X = rng.standard_normal((n, p))
        y = rng.standard_normal(n)

        a, W, loss_history = train_oracle_nn(
            X, y, m=m, activation="relu",
            lr=0.01, max_iters=50, rng=np.random.default_rng(41),
        )
        assert a.shape == (m,)
        assert W.shape == (m, p)
        assert len(loss_history) == 50

    def test_identity_high_r2_on_easy_problem(self):
        """Identity activation on a large-signal linear problem should achieve high R^2."""
        rng = np.random.default_rng(42)
        p = 3
        n_train, n_test = 2000, 500
        Sigma = np.eye(p)
        X_train = rng.multivariate_normal(np.zeros(p), Sigma, size=n_train)
        beta = np.array([1.0, 0.5, -0.5])
        y_train = X_train @ beta + rng.normal(0, 0.1, n_train)

        a, W, _ = train_oracle_nn(
            X_train, y_train, m=p, activation="identity",
            lr=0.02, max_iters=500, batch_size=n_train,
            rng=np.random.default_rng(42), init_scale=0.1,
        )

        X_test = rng.multivariate_normal(np.zeros(p), Sigma, size=n_test)
        y_test = X_test @ beta + rng.normal(0, 0.1, n_test)

        y_pred = X_test @ W.T @ a
        ss_res = np.mean((y_test - y_pred) ** 2)
        ss_tot = np.var(y_test)
        r2 = 1.0 - ss_res / ss_tot
        assert r2 > 0.5

    @pytest.mark.parametrize("activation", ["relu", "sigmoid", "identity"])
    def test_all_activations_run(self, activation):
        rng = np.random.default_rng(43)
        p, n = 6, 200
        X = rng.standard_normal((n, p))
        y = rng.standard_normal(n)
        a, W, loss_history = train_oracle_nn(
            X, y, m=3, activation=activation,
            lr=0.005, max_iters=50, rng=np.random.default_rng(43),
        )
        assert len(loss_history) == 50
        assert np.all(np.isfinite(a))
        assert np.all(np.isfinite(W))


# ===================================================================
# SimulationScenario / ScenarioResult dataclasses
# ===================================================================

class TestDataclasses:

    def test_scenario_defaults(self):
        s = SimulationScenario()
        assert s.p == 50
        assert s.m == 5
        assert s.heritability == 0.5
        assert s.activation == "relu"
        assert s.maf_spectrum == "common"

    def test_scenario_override(self):
        s = SimulationScenario(p=20, m=3, heritability=0.3, activation="sigmoid")
        assert s.p == 20
        assert s.m == 3
        assert s.heritability == 0.3
        assert s.activation == "sigmoid"

    def test_scenario_result_fields(self):
        sr = ScenarioResult(method="test", r2=0.5, weight_cosine=0.9, mean_abs_kappa3=0.1)
        assert sr.method == "test"
        assert sr.r2 == 0.5
        assert sr.weight_cosine == 0.9
        assert sr.mean_abs_kappa3 == 0.1


# ===================================================================
# _weight_cosine_similarity
# ===================================================================

class TestWeightCosineSimilarity:

    def test_proportional_returns_one(self):
        beta_star = np.array([1.0, 2.0, 3.0])
        W = np.eye(3)
        a = beta_star * 2.0
        cos_sim = _weight_cosine_similarity(a, W, beta_star)
        assert cos_sim == pytest.approx(1.0, abs=1e-10)

    def test_anti_proportional_returns_neg_one(self):
        beta_star = np.array([1.0, 2.0, 3.0])
        W = np.eye(3)
        a = -beta_star * 3.0
        cos_sim = _weight_cosine_similarity(a, W, beta_star)
        assert cos_sim == pytest.approx(-1.0, abs=1e-10)

    def test_orthogonal_returns_zero(self):
        beta_star = np.array([1.0, 0.0, 0.0])
        W = np.eye(3)
        a = np.array([0.0, 1.0, 0.0])
        cos_sim = _weight_cosine_similarity(a, W, beta_star)
        assert cos_sim == pytest.approx(0.0, abs=1e-10)

    def test_zero_effective_weights(self):
        beta_star = np.array([1.0, 2.0, 3.0])
        W = np.zeros((2, 3))
        a = np.array([1.0, 1.0])
        cos_sim = _weight_cosine_similarity(a, W, beta_star)
        assert cos_sim == 0.0

    def test_zero_beta_star(self):
        W = np.eye(3)
        a = np.array([1.0, 2.0, 3.0])
        cos_sim = _weight_cosine_similarity(a, W, np.zeros(3))
        assert cos_sim == 0.0


# ===================================================================
# run_single_rep
# ===================================================================

class TestRunSingleRep:

    @pytest.fixture
    def small_scenario(self):
        return SimulationScenario(
            p=10, m=3, n_train=500, n_test=200,
            maf_spectrum="common", ld_decay=0.3,
            heritability=0.5, sparsity=0.3,
            activation="relu",
            sumstat_lr=0.01, sumstat_max_iters=200,
            oracle_lr=0.01, oracle_max_iters=300,
            oracle_batch_size=128,
        )

    def test_returns_four_results(self, small_scenario):
        rng = np.random.default_rng(50)
        results = run_single_rep(small_scenario, rng)
        assert len(results) == 5

    def test_all_methods_represented(self, small_scenario):
        rng = np.random.default_rng(51)
        results = run_single_rep(small_scenario, rng)
        methods = {r.method for r in results}
        assert methods == {"Linear PRS", "Gaussian NN", "Edgeworth NN", "Interaction NN", "Oracle NN"}

    def test_r2_values_finite(self, small_scenario):
        rng = np.random.default_rng(52)
        results = run_single_rep(small_scenario, rng)
        for r in results:
            assert np.isfinite(r.r2), f"{r.method} has non-finite R^2: {r.r2}"

    def test_all_results_are_scenario_result(self, small_scenario):
        rng = np.random.default_rng(53)
        results = run_single_rep(small_scenario, rng)
        for r in results:
            assert isinstance(r, ScenarioResult)

    def test_sigmoid_activation(self):
        scenario = SimulationScenario(
            p=8, m=3, n_train=400, n_test=150,
            maf_spectrum="common", ld_decay=0.3,
            heritability=0.5, sparsity=0.3,
            activation="sigmoid",
            sumstat_lr=0.01, sumstat_max_iters=200,
            oracle_lr=0.01, oracle_max_iters=300,
            oracle_batch_size=128,
        )
        rng = np.random.default_rng(54)
        results = run_single_rep(scenario, rng)
        assert len(results) == 5
        for r in results:
            assert np.isfinite(r.r2)


# ===================================================================
# run_scenario
# ===================================================================

class TestRunScenario:

    @pytest.fixture
    def tiny_scenario(self):
        return SimulationScenario(
            p=8, m=3, n_train=300, n_test=100,
            maf_spectrum="common", ld_decay=0.3,
            heritability=0.5, sparsity=0.3,
            activation="relu",
            sumstat_lr=0.01, sumstat_max_iters=100,
            oracle_lr=0.01, oracle_max_iters=150,
            oracle_batch_size=64,
        )

    def test_correct_number_of_rows(self, tiny_scenario):
        n_reps = 2
        rows = run_scenario(tiny_scenario, n_reps=n_reps, seed=60)
        assert len(rows) == 5 * n_reps

    def test_dict_keys(self, tiny_scenario):
        rows = run_scenario(tiny_scenario, n_reps=1, seed=61)
        expected_keys = {"method", "rep", "r2", "weight_cosine", "mean_abs_kappa3"}
        for row in rows:
            assert set(row.keys()) == expected_keys

    def test_deterministic_seeding(self, tiny_scenario):
        rows1 = run_scenario(tiny_scenario, n_reps=1, seed=62)
        rows2 = run_scenario(tiny_scenario, n_reps=1, seed=62)
        for r1, r2 in zip(rows1, rows2):
            assert r1["method"] == r2["method"]
            assert r1["r2"] == pytest.approx(r2["r2"], abs=1e-15)
            assert r1["weight_cosine"] == pytest.approx(r2["weight_cosine"], abs=1e-15)

    def test_different_seeds_differ(self, tiny_scenario):
        rows1 = run_scenario(tiny_scenario, n_reps=1, seed=63)
        rows2 = run_scenario(tiny_scenario, n_reps=1, seed=99)
        r2_vals_1 = [r["r2"] for r in rows1]
        r2_vals_2 = [r["r2"] for r in rows2]
        assert r2_vals_1 != r2_vals_2


# ===================================================================
# Warm-start for train_edgeworth (a_init / W_init)
# ===================================================================

class TestEdgeworthWarmStart:

    def test_warm_start_uses_provided_weights(self):
        """Passing a_init/W_init should start from those weights,
        not from random initialization."""
        rng = np.random.default_rng(70)
        p = 6
        m = 2
        Sigma = generate_ld_matrix(p, decay=0.5)
        beta_star = rng.standard_normal(p) * 0.3
        Sigma_beta = Sigma @ beta_star
        sigma_eps = 1.0
        E_y2 = float(beta_star @ Sigma @ beta_star + sigma_eps ** 2)
        maf = np.array([0.1, 0.2, 0.3, 0.35, 0.4, 0.45])

        gauss_result = train_edgeworth(
            Sigma, Sigma_beta, E_y2, maf,
            m=m, activation="relu", lr=0.005,
            max_iters=100, init_scale=0.01,
            rng=np.random.default_rng(42),
            loss_floor=0.0, grad_clip=1.0,
        )

        warm_result = train_edgeworth(
            Sigma, Sigma_beta, E_y2, maf,
            m=m, activation="relu", lr=0.001,
            max_iters=50,
            rng=np.random.default_rng(99),
            loss_floor=0.0, grad_clip=1.0,
            a_init=gauss_result.a,
            W_init=gauss_result.W,
        )

        random_result = train_edgeworth(
            Sigma, Sigma_beta, E_y2, maf,
            m=m, activation="relu", lr=0.001,
            max_iters=50, init_scale=0.01,
            rng=np.random.default_rng(99),
            loss_floor=0.0, grad_clip=1.0,
        )

        assert warm_result.loss_history[0] != pytest.approx(
            random_result.loss_history[0], abs=1e-6
        ), "Warm-started and random-init should have different initial losses"

    def test_warm_start_initial_loss_matches_provided_weights(self):
        """The first loss entry should correspond to the provided weights."""
        from ssnn.edgeworth_risk import compute_edgeworth_loss
        from ssnn.cumulants import decorrelation_matrix

        rng = np.random.default_rng(71)
        p = 6
        m = 2
        Sigma = generate_ld_matrix(p, decay=0.5)
        beta_star = rng.standard_normal(p) * 0.3
        Sigma_beta = Sigma @ beta_star
        sigma_eps = 1.0
        E_y2 = float(beta_star @ Sigma @ beta_star + sigma_eps ** 2)
        maf = np.array([0.1, 0.2, 0.3, 0.35, 0.4, 0.45])

        a_init = rng.standard_normal(m) * 0.05
        W_init = rng.standard_normal((m, p)) * 0.05

        result = train_edgeworth(
            Sigma, Sigma_beta, E_y2, maf,
            m=m, activation="relu", lr=0.001,
            max_iters=10,
            rng=np.random.default_rng(99),
            loss_floor=0.0, grad_clip=1.0,
            a_init=a_init,
            W_init=W_init,
        )

        S = decorrelation_matrix(Sigma)
        expected_loss = compute_edgeworth_loss(
            a_init, W_init, Sigma, Sigma_beta, E_y2,
            maf, "relu", S, loss_floor=0.0,
        )
        assert result.loss_history[0] == pytest.approx(expected_loss, rel=1e-10)
