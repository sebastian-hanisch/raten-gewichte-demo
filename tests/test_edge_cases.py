"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12, Mutationstests der Linie Mehrkriterien-Ranking): genau zwei Elemente, ein Element,
alle Werte gleich, Reglergrenzen, Gleichstände."""

import math

import numpy as np
import pytest

import rgw_constants as C
import rgw_evaluation as V
import rgw_growth as G
import rgw_means as M
import rgw_mix as X
import rgw_scenario as S
import rgw_sizebias as Z
import rgw_tasks as T


def test_exactly_two_elements_for_every_mean():
    """n = 2: der kleinste Fall, in dem die Mittel verschieden sind (Zweige für n < 2 dürfen nicht greifen)."""
    m = M.four_means([2.0, 8.0])
    assert m["harmonisch"] == pytest.approx(3.2) and m["geometrisch"] == pytest.approx(4.0) and m["arithmetisch"] == pytest.approx(5.0) and m["quadratisch"] == pytest.approx(math.sqrt(34))



def test_single_element_gives_that_element_for_every_r():
    for r in (-5.0, -1.0, 0.0, 1.0, 2.0, 5.0, np.inf, -np.inf):
        assert M.power_mean([3.5], r) == pytest.approx(3.5)
    assert M.total_rate([7.0], [2.0]) == pytest.approx(3.5)


def test_all_values_equal_means_every_mean_equal_and_the_verdict_is_calm():
    m = M.four_means([4.0] * 6)
    assert max(m.values()) - min(m.values()) < 1e-12
    kind, text = V.verdict_rates("strecke", 4.0, m, 1.0, 4.0)
    assert kind == "info" and "0,0 %" in text


@pytest.mark.parametrize("task", sorted(T.RATE_TASKS))
def test_two_posts_at_the_smallest_spread(task):
    """m = 2 und kleinste Streuung: die Aufgabe rechnet, die Mittel liegen eng beieinander (außer beim festen Wechsel 1,5/0,6)."""
    a = V.analyse(V.Settings(task=task, m=C.M_MIN, sigma=C.SIGMA_MIN))
    assert len(a.x) == 2 and np.isfinite(a.truth) and a.truth == pytest.approx(M.power_mean(a.x, T.RATE_TASKS[task].right_r), rel=1e-12)
    spread = max(a.means.values()) / min(a.means.values()) - 1
    if task == "wechsel":
        assert spread > 0.3
    else:
        assert spread < 0.02


def test_largest_spread_and_largest_exponent_stay_finite():
    a = V.analyse(V.Settings(task="strecke", m=C.M_MAX, sigma=C.SIGMA_MAX, r=C.R_MAX))
    assert np.all(np.isfinite(a.curve)) and np.all(np.diff(a.curve) >= -1e-9 * a.curve[1:]) and np.isfinite(a.selected)


def test_lots_with_two_observations_and_with_almost_equal_lots():
    a = V.analyse(V.Settings(task="lose", m=2, sigma=C.SIGMA_MIN))
    assert len(a.observed) == 2 and a.arithmetic / a.truth == pytest.approx(1.0, abs=0.2)
    near = V.analyse(V.Settings(task="lose", m=200, sigma=C.SIGMA_MIN))
    assert near.factor == pytest.approx(1.0, abs=0.01) and near.arithmetic == pytest.approx(near.harmonic, rel=0.05)


def test_the_table_has_no_regler_and_always_flips():
    a = V.analyse(V.Settings(task="tabelle", m=C.M_MIN, sigma=C.SIGMA_MIN))
    assert a.reversal and a.pooled.tolist() == pytest.approx([0.63, 0.76])


def test_constant_lots_have_no_bias():
    sizes = np.full(10, 4.0)
    assert Z.bias_factor(sizes) == 1.0
    obs = Z.pieces_from_population(sizes, 7, S.Draws(1))
    assert obs.tolist() == [4.0] * 7 and float(Z.harmonic_back(obs)) == 4.0 and float(Z.arithmetic_back(obs)) == 4.0


def test_one_lot_and_one_piece():
    assert Z.pieces_from_population(np.array([5.0]), 3, S.Draws(2)).tolist() == [5.0, 5.0, 5.0]
    assert Z.pieces_from_population(np.array([5.0, 1.0]), 1, S.Draws(2)).shape == (1,)


def test_ties_between_the_groups_are_not_a_reversal():
    """Gleiche Quoten in beiden Gruppen und Schichten: weder Vorsprung noch Umkehr, die strikte Ungleichung entscheidet (kein Gleichstand zählt)."""
    table = np.array([[[5, 5], [5, 5]], [[3, 7], [3, 7]]], dtype=float)
    assert not X.has_reversal(table)
    assert X.pooled_rates(table).tolist() == pytest.approx([0.4, 0.4])


def test_one_period_and_the_critical_boundary():
    assert G.exact_share_below_start(1) == 0.5 and G.path_value(np.array([1]), 1).tolist() == [1.5]
    c = G.critical_sigma()
    assert G.lognormal_share_below_start(50, c * (1 - 1e-6)) < 0.5 < G.lognormal_share_below_start(50, c * (1 + 1e-6))


def test_poisson_smallest_parameter_keeps_empty_lots_and_the_table_ends_cleanly():
    lots = S.poisson(S.Draws(3), 1000, 0.05)
    assert lots.min() == 0 and (lots == 0).mean() > 0.9
    assert S.poisson_pmf(0.05)[-1] < 1e-15
