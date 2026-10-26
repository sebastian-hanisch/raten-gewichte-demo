"""Orakel Wachstumspfade: Aufzählung aller 2^T Pfade (Produkt der Faktoren ohne Logarithmen) gegen die Binomialsummen, scipy-Normalverteilung für den lognormalen Fall,
Einzelschritte gegen die Quantilfunktion der Binomialverteilung (Verteilungsvergleich mit Marge), Erwartungswert als exakter Bruch."""

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest

import rgw_growth as G
import rgw_scenario as S


def _enumerate(T):
    values = []
    for steps in itertools.product((1.5, 0.6), repeat=T):
        v = 1.0
        for f in steps:
            v *= f
        values.append(v)
    return np.array(values)


@pytest.mark.parametrize("T", [1, 2, 3, 4, 6, 8, 10, 12, 14])
def test_exact_shares_match_the_enumeration_of_all_paths(T):
    v = _enumerate(T)
    assert G.exact_share_below_start(T) == pytest.approx(float((v < 1.0).mean()), abs=1e-12)
    assert G.exact_share_above_expectation(T) == pytest.approx(float((v > 1.05 ** T * (1 + 1e-12)).mean()), abs=1e-12)


@pytest.mark.parametrize("T", [2, 4, 6, 8, 10])
def test_expected_and_median_value_match_the_enumeration(T):
    v = _enumerate(T)
    assert G.expected_value(T) == pytest.approx(float(v.mean()), rel=1e-12)
    assert G.typical_value(T) == pytest.approx(float(np.median(v)), rel=1e-12)


def test_expected_value_as_an_exact_fraction():
    mean = Fraction(1, 2) * Fraction(3, 2) + Fraction(1, 2) * Fraction(3, 5)
    assert mean == Fraction(21, 20)
    assert G.expected_value(7) == pytest.approx(float(mean ** 7), rel=1e-14)
    # Varianz des Faktors 0,2025 = E[F²] − 1,05²; zweites Moment je Periode 0,5·2,25 + 0,5·0,36 = 1,305
    assert float(Fraction(1, 2) * Fraction(9, 4) + Fraction(1, 2) * Fraction(9, 25)) == pytest.approx(1.305)


def test_lognormal_share_below_start_matches_scipy_norm():
    norm = pytest.importorskip("scipy.stats").norm
    for T, sigma in ((10, 0.2), (50, 0.2), (50, 0.5), (100, 0.3124), (200, 0.8)):
        z = -T * (math.log(1.05) - sigma ** 2 / 2) / (sigma * math.sqrt(T))
        assert G.lognormal_share_below_start(T, sigma) == pytest.approx(float(norm.cdf(z)), abs=1e-12)


def test_closed_form_lognormal_paths_agree_with_summing_single_period_factors():
    """Pfad aus T Einzelfaktoren exp(μ + σZ) gegen die geschlossene Form mit einer Normalverteilten je Pfad: gleiche Verteilung (Mittel und Streuung der Logarithmen)."""
    T, sigma, N = 25, 0.4, 60_000
    mu = math.log(1.05) - sigma ** 2 / 2
    steps = np.exp((mu + sigma * S.Draws(41).normal((N, T))).sum(axis=1))
    closed = G.lognormal_paths(S.Draws(42), N, T, sigma)
    for sample in (steps, closed):
        logs = np.log(sample)
        assert abs(logs.mean() - T * mu) < 0.02 and abs(logs.std() - sigma * math.sqrt(T)) < 0.02


def test_single_steps_and_binomial_quantile_draw_agree_on_the_sample_mean_distribution():
    """Verhältnis Stichprobenmittel/Erwartungswert bei N = 1 000 Pfaden, T = 20: Mediane beider Ziehungen liegen beieinander (Marge 0,05)."""
    T, N, reps = 20, 1000, 300
    by_steps, by_binom = [], []
    for r in range(reps):
        by_steps.append(G.sample_mean_ratio(G.path_value(G.ups_by_steps(S.Draws(1000 + r), N, T), T), T))
        by_binom.append(G.sample_mean_ratio(G.path_value(G.ups_binomial(S.Draws(5000 + r), N, T), T), T))
    assert abs(np.median(by_steps) - np.median(by_binom)) < 0.05
    assert abs(np.mean(by_steps) - 1.0) < 0.03 and abs(np.mean(by_binom) - 1.0) < 0.03           # das Stichprobenmittel ist erwartungstreu
    assert np.median(by_steps) < 1.0 and np.median(by_binom) < 1.0                                  # aber schief: der typische Wert liegt darunter
