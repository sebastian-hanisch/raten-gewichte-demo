"""Größenverzerrung gegen Handrechnung (Lose 1, 1, 4), Formeln der Familien, Beobachtungsmodelle und Randfälle."""

import math

import numpy as np
import pytest

import rgw_scenario as S
import rgw_sizebias as Z


class _Fixed(S.Draws):
    """Strom mit fest vorgegebenen Gleichverteilten (für Handrechnungen)."""

    def __init__(self, values):
        super().__init__(0)
        self.values = np.asarray(values, dtype=float)

    def uniform(self, shape):
        return self.values.reshape(shape)


def test_pieces_from_the_three_lots_by_hand(lots):
    """Gleichverteilte (k + 0,5)/6, k = 0 bis 5: Die sechs Stücke gehören zu den Losen 1, 1, 4, 4, 4, 4 (Aufsummierte Größen 1, 2, 6)."""
    obs = Z.pieces_from_population(lots, 6, _Fixed([(k + 0.5) / 6 for k in range(6)]))
    assert obs.tolist() == [1.0, 1.0, 4.0, 4.0, 4.0, 4.0]
    assert Z.arithmetic_back(obs) == pytest.approx(3.0) and Z.harmonic_back(obs) == pytest.approx(2.0)


def test_bias_factor_of_the_three_lots_by_hand(lots):
    assert Z.bias_factor(lots) == pytest.approx(1.5) == pytest.approx(1 + 0.5)             # 1 + CV², CV² = 2/4
    assert float((lots ** 2).mean() / lots.mean()) == pytest.approx(3.0)                   # Mittel der Stücksicht E[X²]/E[X]
    assert float(lots.mean()) == pytest.approx(2.0)


def test_harmonic_back_by_hand():
    assert Z.harmonic_back([1.0, 1.0, 4.0, 4.0, 4.0, 4.0]) == pytest.approx(2.0)
    assert Z.harmonic_back([2.0, 2.0]) == pytest.approx(2.0)
    assert Z.harmonic_back(np.array([[1.0, 4.0], [2.0, 2.0]])).tolist() == pytest.approx([1.6, 2.0])    # 2/(1 + 0,25) = 1,6


def test_empty_lots_are_never_observed(lots):
    with pytest.raises(ValueError):
        Z.harmonic_back([0.0, 2.0])
    with pytest.raises(ValueError):
        Z.pieces_from_population(np.zeros(3), 2, S.Draws(1))
    assert 0.0 not in Z.pieces_from_population(np.array([0.0, 0.0, 3.0]), 50, S.Draws(2)).tolist()


def test_family_formulas_by_hand():
    assert Z.family_mean("poisson", 3.0) == 3.0 and Z.bias_factor_formula("poisson", 3.0) == pytest.approx(1 + 1 / 3)
    assert Z.piece_mean_formula("poisson", 3.0) == pytest.approx(4.0)                      # (λ + λ²)/λ = λ + 1
    assert Z.family_mean("geometrisch", 0.3) == pytest.approx(1 / 0.3) and Z.bias_factor_formula("geometrisch", 0.3) == pytest.approx(1.7)
    assert Z.piece_mean_formula("geometrisch", 0.3) == pytest.approx((2 - 0.3) / 0.3)
    assert Z.bias_factor_formula("lognormal", 1.0) == pytest.approx(math.e)
    assert Z.family_mean("lognormal", 1.0) == pytest.approx(10 * math.exp(0.5))
    assert Z.family_mean("pareto", 3.0) == pytest.approx(1.5) and Z.piece_mean_formula("pareto", 3.0) == pytest.approx(2.0)    # (α − 1)/(α − 2)
    assert Z.bias_factor_formula("pareto", 3.0) == pytest.approx(4 / 3)                    # (α − 1)²/(α(α − 2))


def test_heavy_tail_has_no_factor():
    """Pareto mit α ≤ 2: kein zweites Moment, kein Faktor; mit α ≤ 1 nicht einmal ein Mittel."""
    assert math.isinf(Z.bias_factor_formula("pareto", 2.0)) and math.isinf(Z.piece_mean_formula("pareto", 1.5))
    assert math.isinf(Z.family_mean("pareto", 1.0)) and math.isfinite(Z.family_mean("pareto", 1.5))
    with pytest.raises(ValueError):
        Z.size_biased_draw("pareto", 1.0, S.Draws(1), 3)


def test_unknown_family_is_an_error():
    for fn in (lambda: Z.family_mean("x", 1.0), lambda: Z.family_second_moment("x", 1.0), lambda: Z.lot_sizes("x", 1.0, S.Draws(1), 3), lambda: Z.size_biased_draw("x", 1.0, S.Draws(1), 3)):
        with pytest.raises(ValueError):
            fn()


@pytest.mark.parametrize("family, par", [("poisson", 3.0), ("geometrisch", 0.3), ("lognormal", 0.5), ("pareto", 3.0)])
def test_lot_sizes_have_the_family_mean(family, par):
    x = Z.lot_sizes(family, par, S.Draws(51), 400_000)
    assert x.mean() == pytest.approx(Z.family_mean(family, par), rel=0.02)


@pytest.mark.parametrize("family, par", [("poisson", 3.0), ("geometrisch", 0.3), ("lognormal", 0.5), ("pareto", 4.0)])
def test_closed_form_observations_have_the_piece_mean_and_the_harmonic_mean_recovers_the_lot_mean(family, par):
    y = Z.size_biased_draw(family, par, S.Draws(52), 400_000)
    assert y.mean() == pytest.approx(Z.piece_mean_formula(family, par), rel=0.02)
    target = Z.nonempty_mean_poisson(par) if family == "poisson" else Z.family_mean(family, par)       # leere Lose sind unsichtbar: Poisson liefert das Mittel der nichtleeren
    assert float(Z.harmonic_back(y)) == pytest.approx(target, rel=0.02)


def test_empty_lots_stay_invisible_for_poisson():
    """Poisson(3): harmonisches Mittel der Beobachtungen = Mittel der nichtleeren Lose λ/(1 − e^(−λ)) = 3,157 und nicht λ = 3."""
    lots = Z.lot_sizes("poisson", 3.0, S.Draws(53), 600_000)
    nonempty = lots[lots > 0]
    assert Z.nonempty_mean_poisson(3.0) == pytest.approx(3.157187, abs=1e-6)
    assert nonempty.mean() == pytest.approx(Z.nonempty_mean_poisson(3.0), rel=0.005)
    obs = Z.pieces_from_population(lots[:100_000], 400_000, S.Draws(54))
    assert float(Z.harmonic_back(obs)) == pytest.approx(Z.nonempty_mean_poisson(3.0), rel=0.02)
    assert float(Z.harmonic_back(obs)) / 3.0 - 1 == pytest.approx(0.052, abs=0.015)


@pytest.mark.parametrize("family, par", [("poisson", 10.0), ("lognormal", 0.5), ("geometrisch", 0.3)])
def test_measured_piece_to_lot_ratio_is_the_bias_factor_of_the_population(family, par):
    x = Z.lot_sizes(family, par, S.Draws(55), 5000)
    obs = Z.pieces_from_population(x, 400_000, S.Draws(56))
    assert obs.mean() / x.mean() == pytest.approx(Z.bias_factor(x), rel=0.02)


def test_every_family_changes_the_factor():
    """Nullspalten-Regel: Die Familien liefern verschiedene Faktoren (Poisson(3): 1,33; geometrisch(1/3): 1,67; lognormal mit σ = 1: e; Pareto(2,5): 1,8)."""
    factors = {Z.bias_factor_formula("poisson", 3.0), Z.bias_factor_formula("geometrisch", 1 / 3), Z.bias_factor_formula("lognormal", 1.0), Z.bias_factor_formula("pareto", 2.5)}
    assert len(factors) == 4
