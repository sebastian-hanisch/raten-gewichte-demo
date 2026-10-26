"""Orakel Potenzmittel und Aufgaben (anderer Rechenweg): `scipy.stats.pmean`, `gmean`, `hmean` (mit Gewichten), dazu die Gesamtgröße aus Zähler und Nenner als Summenverhältnis
und die Reihenfolge HM ≤ GM ≤ AM ≤ QM mit der geschlossenen Form M_r/GM = exp(r·σ²/2) der Lognormalverteilung (Mittelwert über viele Ziehungen, mit Marge)."""

import math

import numpy as np
import pytest

import rgw_means as M
import rgw_scenario as S
import rgw_tasks as T

stats = pytest.importorskip("scipy.stats")


def _random_sets(seed, count=60):
    rng = np.random.default_rng(seed)       # nur im Test: Zufall der Prüfmengen, die Demo selbst zieht aus rgw_rng
    for _ in range(count):
        n = int(rng.integers(2, 31))
        yield np.exp(rng.uniform(0, 2) * rng.standard_normal(n)), rng.uniform(0.1, 5.0, n)


@pytest.mark.parametrize("r", [-5.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0, 10.0])
def test_power_mean_matches_scipy_pmean(r):
    for x, w in _random_sets(11):
        assert M.power_mean(x, r) == pytest.approx(stats.pmean(x, r), rel=1e-12)
        assert M.power_mean(x, r, w) == pytest.approx(stats.pmean(x, r, weights=w), rel=1e-12)


def test_geometric_and_harmonic_match_scipy():
    for x, w in _random_sets(12):
        assert M.geometric(x) == pytest.approx(stats.gmean(x), rel=1e-12)
        assert M.geometric(x, w) == pytest.approx(stats.gmean(x, weights=w), rel=1e-12)
        assert M.harmonic(x) == pytest.approx(stats.hmean(x), rel=1e-12)
        assert M.harmonic(x, w) == pytest.approx(stats.hmean(x, weights=w), rel=1e-12)
        assert M.arithmetic(x, w) == pytest.approx(float(np.average(x, weights=w)), rel=1e-12)


def test_the_four_means_are_ordered_on_random_sets():
    for x, _ in _random_sets(13, 200):
        m = M.four_means(x)
        values = [m["harmonisch"], m["geometrisch"], m["arithmetisch"], m["quadratisch"]]
        assert all(a <= b * (1 + 1e-12) for a, b in zip(values, values[1:]))
        assert M.power_mean(x, -np.inf) <= values[0] * (1 + 1e-12) and values[-1] <= M.power_mean(x, np.inf) * (1 + 1e-12)


def test_the_power_mean_is_non_decreasing_in_r():
    grid = [-np.inf, -5, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 5, 10, 50, np.inf]
    for x, _ in _random_sets(14, 100):
        values = np.array([M.power_mean(x, r) for r in grid])
        assert np.all(np.diff(values) >= -1e-12 * values[1:])


def test_constant_sets_give_the_constant_for_every_r():
    for r in (-np.inf, -3, -1, 0, 1, 2, 7, np.inf):
        assert M.power_mean(np.full(7, 3.7), r) == pytest.approx(3.7, rel=1e-14)


def test_lognormal_ratio_to_the_geometric_mean_is_exp_r_sigma_squared_over_two():
    """Für lognormale Werte mit Streuung σ ist M_r/GM = exp(r·σ²/2) (Erwartungswert über die Grundgesamtheit); mit 200 000 Ziehungen auf 1 % genau (σ = 0,5)."""
    sigma = 0.5
    x = np.exp(sigma * S.Draws(21).normal(200_000))
    gm = M.geometric(x)
    for r in (-1.0, 1.0, 2.0):
        assert M.power_mean(x, r) / gm == pytest.approx(math.exp(r * sigma ** 2 / 2), rel=0.01)


@pytest.mark.parametrize("key", ["strecke", "zeit", "verbrauch"])
def test_true_total_is_the_ratio_of_sums_and_the_right_mean_matches_it(key):
    """Unabhängig: Summe Zähler / Summe Nenner aus den Rohgrößen, ohne ein Mittel zu benutzen; das richtige Mittel der Aufgabe trifft sie auf 12 Stellen."""
    x, truth = T.make_batch(key, S.Draws(31), 100, 15, 0.9)
    num, den = T.raw_quantities(key, x)
    direct = num.sum(axis=1) / den.sum(axis=1)
    assert truth == pytest.approx(direct, rel=1e-14)
    assert M.power_mean(x, T.RATE_TASKS[key].right_r) == pytest.approx(direct, rel=1e-12)


def test_true_total_of_factors_is_the_root_of_the_product():
    x, truth = T.make_batch("faktoren", S.Draws(32), 100, 15, 0.4)
    assert truth == pytest.approx(np.prod(x, axis=1) ** (1 / 15), rel=1e-12)
    assert M.geometric(x) == pytest.approx(truth, rel=1e-12)
    # Gesamtfaktor über alle Perioden: Produkt = (geometrisches Mittel)^m, nicht (arithmetisches Mittel)^m
    assert np.prod(x[0]) == pytest.approx(M.geometric(x[0]) ** 15, rel=1e-10)
    assert abs(np.prod(x[0]) / M.arithmetic(x[0]) ** 15 - 1) > 0.05


def test_wrong_mean_costs_grow_with_the_spread():
    """Mittlere Abweichung des arithmetischen Mittels von der Gesamtgeschwindigkeit bei gleicher Strecke wächst mit σ (Handlungsorakel für die Messreihe E1)."""
    costs = []
    for sigma in (0.1, 0.4, 0.8):
        x, truth = T.make_batch("strecke", S.Draws(33), 500, 20, sigma)
        costs.append(float(np.mean(np.abs(M.arithmetic(x) / truth - 1))))
    assert costs[0] < costs[1] < costs[2] and costs[0] < 0.01 and costs[2] > 0.3
