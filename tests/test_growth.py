"""Wachstumspfade gegen Handrechnung (Faktoren 1,5 und 0,6), Binomialformeln, Prefix-Eigenschaft und die Zweige (Streuung unter/über der kritischen)."""

import math

import numpy as np
import pytest

import rgw_growth as G
import rgw_scenario as S


def test_path_value_by_hand():
    assert float(G.path_value(2, 4)) == pytest.approx(1.5 ** 2 * 0.6 ** 2) == pytest.approx(0.81)
    assert float(G.path_value(0, 1)) == pytest.approx(0.6) and float(G.path_value(1, 1)) == pytest.approx(1.5)
    assert G.path_value(np.array([0, 3]), 3).tolist() == pytest.approx([0.216, 3.375 * 1.0])


def test_expected_and_typical_value_by_hand():
    assert G.expected_value(1) == pytest.approx(1.05) and G.expected_value(2) == pytest.approx(1.1025)
    assert G.typical_value(2) == pytest.approx(0.9) and G.typical_value(4) == pytest.approx(0.81) and G.typical_value(100) == pytest.approx(0.9 ** 50)
    with pytest.raises(ValueError):
        G.typical_value(3)


def test_share_below_start_by_hand():
    """T = 1: nur die Abwärtsstufe 0,6 liegt unter 1 → 1/2. T = 2: Werte 0,36; 0,9; 0,9; 2,25 → 3/4. T = 4 (K = 0 bis 4 mit 1, 4, 6, 4, 1 Pfaden): K ≤ 2 liegt unter 1 → 11/16."""
    assert G.exact_share_below_start(1) == 0.5
    assert G.exact_share_below_start(2) == 0.75
    assert G.exact_share_below_start(4) == pytest.approx(11 / 16)


def test_share_above_expectation_by_hand():
    """T = 1: Erwartungswert 1,05, nur 1,5 liegt darüber → 1/2. T = 2: Erwartungswert 1,1025, nur 2,25 liegt darüber → 1/4."""
    assert G.exact_share_above_expectation(1) == 0.5
    assert G.exact_share_above_expectation(2) == 0.25


def test_prefix_paths_are_beginnings_of_the_same_path():
    u = S.Draws(1).uniform((5, 3)) < 0.5
    cum = np.cumsum(u, axis=1)
    out = G.ups_prefix(S.Draws(1), 5, [3, 2])
    assert out[2].tolist() == cum[:, 1].tolist() and out[3].tolist() == cum[:, 2].tolist()
    assert all(out[3] >= out[2])                                     # ein längerer Pfad hat mindestens so viele Aufwärtsschritte


def test_chunks_do_not_change_the_result(monkeypatch):
    big = G.ups_by_steps(S.Draws(2), 50, 6)
    monkeypatch.setattr(G, "CHUNK_DRAWS", 6 * 7)                    # 7 Pfade je Block
    small = G.ups_by_steps(S.Draws(2), 50, 6)
    assert big.tolist() == small.tolist()                           # die Gleichverteilten werden in derselben Reihenfolge gezogen


def test_binomial_draw_and_single_steps_agree_in_distribution():
    steps = G.ups_by_steps(S.Draws(3), 40000, 20)
    binom = G.ups_binomial(S.Draws(4), 40000, 20)
    assert abs(steps.mean() - 10) < 0.05 and abs(binom.mean() - 10) < 0.05
    assert abs(steps.var() - 5) < 0.15 and abs(binom.var() - 5) < 0.15
    assert abs(float((G.path_value(steps, 20) < 1).mean()) - float((G.path_value(binom, 20) < 1).mean())) < 0.01


def test_simulated_share_below_start_matches_the_binomial_sum():
    ups = G.ups_by_steps(S.Draws(5), 100_000, 100)
    assert abs(G.share_below_start(G.path_value(ups, 100)) - G.exact_share_below_start(100)) < 0.006      # 0,864 ± 0,001 Standardfehler
    assert G.exact_share_below_start(100) == pytest.approx(0.8644, abs=1e-4)


def test_lognormal_paths_keep_the_expected_factor_and_have_the_typical_one():
    sigma = 0.5
    v = G.lognormal_paths(S.Draws(6), 200_000, 1, sigma)
    assert v.mean() == pytest.approx(1.05, rel=0.01)
    assert G.typical_factor(v, 1) == pytest.approx(G.typical_factor_formula(sigma), rel=0.01) and G.typical_factor_formula(sigma) == pytest.approx(1.05 * math.exp(-0.125))


def test_critical_sigma_by_hand():
    """√(2 ln 1,05) = √0,09758 = 0,3124; dort ist der typische Faktor genau 1; größeres σ gibt einen schrumpfenden typischen Pfad, kleineres einen wachsenden."""
    c = G.critical_sigma()
    assert c == pytest.approx(math.sqrt(2 * math.log(1.05))) and c == pytest.approx(0.3124, abs=1e-4)
    assert G.typical_factor_formula(c) == pytest.approx(1.0)
    assert G.typical_factor_formula(0.2) > 1.0 > G.typical_factor_formula(0.4)


def test_the_branches_above_and_below_the_critical_sigma_differ():
    """Nullspalten-Regel: unter σ_krit liegt weniger als die Hälfte der Pfade unter dem Start, darüber mehr (exakt und simuliert)."""
    c = G.critical_sigma()
    assert G.lognormal_share_below_start(50, 0.9 * c) < 0.5 < G.lognormal_share_below_start(50, 1.1 * c)
    assert G.lognormal_share_below_start(50, c) == pytest.approx(0.5, abs=1e-9)
    low = G.share_below_start(G.lognormal_paths(S.Draws(7), 40000, 50, 0.2))
    high = G.share_below_start(G.lognormal_paths(S.Draws(7), 40000, 50, 0.5))
    assert low < 0.25 and high > 0.8


def test_sample_mean_ratio_and_shares_on_given_values():
    v = np.array([0.5, 1.0, 2.0, 3.0])
    assert G.share_below_start(v) == 0.25 and G.share_above(v, 1.0) == 0.5
    assert G.sample_mean_ratio(v, 1) == pytest.approx(1.625 / 1.05)
    assert G.sample_mean_ratio(v, 2) == pytest.approx(1.625 / 1.1025)


def test_exact_quantile_by_hand():
    """T = 2: Werte 0,36 (K = 0, Wahrscheinlichkeit 1/4), 0,9 (K = 1, 1/2) und 2,25 (K = 2, 1/4); die aufsummierte Verteilung ist 0,25 / 0,75 / 1.
    Stufen genau auf den Sprungstellen (0,25 und 0,75) sind in Gleitkommazahlen nicht eindeutig und werden nicht geprüft."""
    assert G.exact_quantile(2, 0.10) == pytest.approx(0.36) and G.exact_quantile(2, 0.24) == pytest.approx(0.36)
    assert G.exact_quantile(2, 0.26) == pytest.approx(0.9) and G.exact_quantile(2, 0.50) == pytest.approx(0.9) and G.exact_quantile(2, 0.74) == pytest.approx(0.9)
    assert G.exact_quantile(2, 0.76) == pytest.approx(2.25) and G.exact_quantile(2, 0.90) == pytest.approx(2.25) and G.exact_quantile(2, 1.0) == pytest.approx(2.25)
    assert G.exact_quantile(1, 0.4) == pytest.approx(0.6) and G.exact_quantile(1, 0.6) == pytest.approx(1.5)
    assert G.exact_quantile(2, 1.5) == pytest.approx(2.25)                         # eine Stufe über 1 bleibt beim größten Wert (kein Index über T hinaus)
    assert G.exact_quantile(100, 0.5) == pytest.approx(0.9 ** 50)


def test_lognormal_trajectories_have_the_log_drift_and_start_at_one():
    sigma = 0.5
    paths = G.lognormal_trajectories(S.Draws(31), 20_000, 2, sigma)
    assert paths.shape == (20_000, 3) and paths[:, 0].tolist() == [1.0] * 20_000
    first = np.log(paths[:, 1])
    assert first.mean() == pytest.approx(math.log(1.05) - sigma ** 2 / 2, abs=0.015) and first.std() == pytest.approx(sigma, abs=0.01)
    assert np.log(paths[:, 2]).mean() == pytest.approx(2 * (math.log(1.05) - sigma ** 2 / 2), abs=0.03)      # zwei Perioden: doppelter Drift


def test_two_point_trajectories_move_in_steps_of_the_two_factors():
    paths = G.two_point_trajectories(S.Draws(32), 500, 3)
    assert paths.shape == (500, 4) and set(np.round(paths[:, 1], 9)) == {1.5, 0.6}
    ratios = paths[:, 1:] / paths[:, :-1]
    assert np.all(np.isclose(ratios, 1.5) | np.isclose(ratios, 0.6)) and abs(np.isclose(ratios, 1.5).mean() - 0.5) < 0.05
