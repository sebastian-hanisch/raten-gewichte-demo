"""Orakel Größenverzerrung: Aufzählung kleiner Lose (jedes Stück einzeln), Formeln gegen scipy (Poisson, geometrisch, lognormal, Pareto), Rückrechnung als exakter Bruch,
Beobachtungsmodell aus der Formel gegen das Ziehen von Stücken aus einer Grundgesamtheit."""

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest

import rgw_scenario as S
import rgw_sizebias as Z

stats = pytest.importorskip("scipy.stats")


def test_enumerating_the_pieces_of_small_lots_gives_the_formulas():
    """Jede Kombination von drei Losgrößen aus 1 bis 5 (125 Fälle): Das Mittel über alle Stücke (jedes Stück einzeln aufgezählt) ist E[X²]/E[X], das Verhältnis zum Mittel pro Los 1 + CV²,
    und das harmonische Mittel der aufgezählten Stücke ist das Mittel pro Los (als Bruch exakt)."""
    for sizes in itertools.product(range(1, 6), repeat=3):
        pieces = [size for size in sizes for _ in range(size)]                       # ein Eintrag je Stück, Wert = Größe seines Loses
        lot_mean = Fraction(sum(sizes), 3)
        piece_mean = Fraction(sum(pieces), len(pieces))
        cv2 = Fraction(sum((Fraction(s) - lot_mean) ** 2 for s in sizes), 3) / lot_mean ** 2
        assert piece_mean / lot_mean == 1 + cv2
        assert Fraction(len(pieces)) / sum(Fraction(1, p) for p in pieces) == lot_mean
        assert Z.bias_factor(np.array(sizes, dtype=float)) == pytest.approx(float(1 + cv2), rel=1e-14)
        assert float(Z.harmonic_back(np.array(pieces, dtype=float))) == pytest.approx(float(lot_mean), rel=1e-12)


def test_moments_of_the_families_match_scipy():
    for lam in (1.0, 3.0, 10.0):
        mean, var = stats.poisson.stats(lam, moments="mv")
        assert Z.family_mean("poisson", lam) == pytest.approx(float(mean)) and Z.family_second_moment("poisson", lam) == pytest.approx(float(var + mean ** 2))
    for p in (0.2, 0.3, 0.7):
        mean, var = stats.geom.stats(p, moments="mv")                                # scipy: Träger {1, 2, …}, wie hier
        assert Z.family_mean("geometrisch", p) == pytest.approx(float(mean)) and Z.family_second_moment("geometrisch", p) == pytest.approx(float(var + mean ** 2))
    for sigma in (0.3, 0.5, 1.0):
        dist = stats.lognorm(s=sigma, scale=Z.LOGNORMAL_SCALE)
        assert Z.family_mean("lognormal", sigma) == pytest.approx(float(dist.mean())) and Z.family_second_moment("lognormal", sigma) == pytest.approx(float(dist.moment(2)))
    for alpha in (2.5, 3.0, 5.0):
        dist = stats.pareto(b=alpha)
        assert Z.family_mean("pareto", alpha) == pytest.approx(float(dist.mean())) and Z.family_second_moment("pareto", alpha) == pytest.approx(float(dist.moment(2)))


def test_lot_sampler_pmf_and_cdf_match_scipy():
    x = Z.lot_sizes("geometrisch", 0.3, S.Draws(61), 300_000)
    for k in (1, 2, 5):
        assert float((x == k).mean()) == pytest.approx(float(stats.geom.pmf(k, 0.3)), abs=0.004)
    y = Z.lot_sizes("pareto", 3.0, S.Draws(62), 300_000)
    for q in (1.5, 2.0, 4.0):
        assert float((y <= q).mean()) == pytest.approx(float(stats.pareto.cdf(q, 3.0)), abs=0.004)
    z = Z.lot_sizes("lognormal", 0.5, S.Draws(63), 300_000)
    for q in (6.0, 10.0, 15.0):
        assert float((z <= q).mean()) == pytest.approx(float(stats.lognorm.cdf(q, s=0.5, scale=10.0)), abs=0.004)


def test_closed_form_size_biased_law_matches_scipy_pmf_and_the_piece_sampler():
    """Stücksicht von Poisson(3): P(Y = k) = k·P(X = k)/λ = P(Poisson(λ) = k − 1); Zählung der geschlossenen Ziehung gegen scipy."""
    y = Z.size_biased_draw("poisson", 3.0, S.Draws(64), 300_000)
    for k in (1, 2, 4, 7):
        assert float((y == k).mean()) == pytest.approx(float(stats.poisson.pmf(k - 1, 3.0)), abs=0.004)
    g = Z.size_biased_draw("geometrisch", 0.3, S.Draws(65), 300_000)
    for k in (1, 2, 5):
        assert float((g == k).mean()) == pytest.approx(k * 0.3 * float(stats.geom.pmf(k, 0.3)), abs=0.004)             # P(Y = k) = k·p·(1 − p)^(k−1)·p
    # gegen das Ziehen von Stücken aus einer Grundgesamtheit
    lots = Z.lot_sizes("poisson", 3.0, S.Draws(66), 50_000)
    pieces = Z.pieces_from_population(lots, 300_000, S.Draws(67))
    for k in (1, 2, 4, 7):
        assert float((pieces == k).mean()) == pytest.approx(float((y == k).mean()), abs=0.006)


def test_lognormal_size_biased_is_lognormal_with_shifted_log_mean():
    y = Z.size_biased_draw("lognormal", 0.7, S.Draws(68), 300_000)
    expected_log_mean = math.log(Z.LOGNORMAL_SCALE) + 0.7 ** 2
    assert float(np.log(y).mean()) == pytest.approx(expected_log_mean, abs=0.01) and float(np.log(y).std()) == pytest.approx(0.7, abs=0.01)


def test_back_estimates_behave_as_measured_in_the_formulas():
    """lognormal σ = 1: Mittel der Beobachtungen = e·Mittel pro Los (+172 %), harmonisch trifft das Mittel pro Los."""
    y = Z.size_biased_draw("lognormal", 1.0, S.Draws(69), 1_000_000)
    lot_mean = Z.family_mean("lognormal", 1.0)
    assert float(Z.arithmetic_back(y)) / lot_mean == pytest.approx(math.e, rel=0.05)           # Tail: Standardfehler gut 1 %
    assert float(Z.harmonic_back(y)) / lot_mean == pytest.approx(1.0, rel=0.01)
