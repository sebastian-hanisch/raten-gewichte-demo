"""Gewichtsmix gegen Handrechnung (Tabelle, die kippt), Gewichte, Standardgewichte, Zufallstabellen und die Kopplung von Nachfrage und Quote."""

import math

import numpy as np
import pytest

import rgw_mix as X
import rgw_scenario as S


def test_the_table_that_flips_by_hand(simpson_table):
    t = simpson_table
    assert X.stratum_rates(t).ravel().tolist() == pytest.approx([0.9, 0.8, 0.6, 0.4])         # A besser in beiden Schichten
    assert X.pooled_rates(t).tolist() == pytest.approx([0.63, 0.76])                      # insgesamt B besser
    assert X.group_weights(t).ravel().tolist() == pytest.approx([0.1, 0.9, 0.9, 0.1])
    assert bool(X.reversal_mask(t)) and X.has_reversal(t)


def test_the_total_is_the_weighted_mean_with_the_denominators_as_weights(simpson_table):
    """Rote Linie: Gesamtquote = Σ w_gs · Quote_gs mit w_gs = n_gs/n_g: 0,1·0,9 + 0,9·0,6 = 0,63 und 0,9·0,8 + 0,1·0,4 = 0,76."""
    rates, weights = X.stratum_rates(simpson_table), X.group_weights(simpson_table)
    assert X.weighted_total(rates, weights).tolist() == pytest.approx([0.1 * 0.9 + 0.9 * 0.6, 0.9 * 0.8 + 0.1 * 0.4])
    assert X.weighted_total(rates, weights).tolist() == pytest.approx(X.pooled_rates(simpson_table).tolist())


def test_standard_weights_remove_the_flip(simpson_table):
    rates, weights = X.stratum_rates(simpson_table), X.group_weights(simpson_table)
    assert X.standard_weights(weights).tolist() == pytest.approx([0.5, 0.5])
    totals = X.standardised_totals(rates, weights)
    assert totals.tolist() == pytest.approx([0.75, 0.60]) and totals[0] > totals[1]       # A wieder vorn
    assert float(X.total_variation(weights)) == pytest.approx(0.8)


def test_total_variation_by_hand():
    same = np.array([[0.3, 0.3], [0.7, 0.7]])
    apart = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert float(X.total_variation(same)) == 0.0 and float(X.total_variation(apart)) == pytest.approx(1.0)


def test_no_reversal_when_a_rate_is_equal_or_only_one_stratum_favours_a():
    equal_in_one = np.array([[[5, 5], [5, 5]], [[6, 4], [3, 7]]], dtype=float)         # Schicht 1 gleich: kein strikter Vorsprung in jeder Schicht
    assert not X.has_reversal(equal_in_one)
    mixed = np.array([[[9, 1], [2, 8]], [[1, 9], [8, 2]]], dtype=float)                 # A vorn in Schicht 1, B vorn in Schicht 2
    assert not X.has_reversal(mixed)
    consistent = np.array([[[9, 1], [2, 8]], [[8, 2], [1, 9]]], dtype=float)           # A in beiden vorn und insgesamt vorn
    assert not X.has_reversal(consistent)


def test_reversal_in_the_other_direction_counts_too(simpson_table):
    swapped = simpson_table[:, ::-1, :]                                                  # Gruppen vertauscht: B-Quoten in jeder Schicht vorn, insgesamt A
    assert X.has_reversal(swapped)


def test_empty_cell_means_no_reversal():
    table = np.array([[[0, 0], [3, 2]], [[5, 5], [4, 6]]], dtype=float)
    assert not X.has_reversal(table)


def test_random_tables_are_probabilities_with_the_right_shape():
    t = X.random_tables(S.Draws(1), 1000, 3)
    assert t.shape == (1000, 3, 2, 2) and t.min() > 0
    assert t.sum(axis=(1, 2, 3)) == pytest.approx(np.ones(1000))
    assert t.mean(axis=0).ravel() == pytest.approx(np.full(12, 1 / 12), abs=0.01)


def test_dominated_tables_have_a_better_in_every_layer_and_normalised_weights():
    rates, weights = X.dominated_rates_and_weights(S.Draws(2), 2000, 3, 1.0)
    assert rates.shape == (2000, 3, 2) and np.all(rates[..., 0] >= rates[..., 1])
    assert weights.sum(axis=1) == pytest.approx(np.ones((2000, 2)))


def test_equal_weights_never_flip_but_different_weights_can():
    """Nullspalten-Regel: τ = 0 gibt gleiche Gewichte 1/l und damit nie eine Umkehr (erzwungen); mit großem τ kippt ein Teil der Tabellen."""
    rates0, w0 = X.dominated_rates_and_weights(S.Draws(3), 20000, 3, 0.0)
    assert w0.min() == pytest.approx(1 / 3) and w0.max() == pytest.approx(1 / 3)
    assert float((X.weighted_total(rates0, w0)[:, 1] > X.weighted_total(rates0, w0)[:, 0]).mean()) == 0.0
    rates, w = X.dominated_rates_and_weights(S.Draws(3), 20000, 3, 1.5)
    tot = X.weighted_total(rates, w)
    assert float((tot[:, 1] > tot[:, 0]).mean()) > 0.02
    std = X.standardised_totals(rates, w)
    assert float((std[:, 1] > std[:, 0]).mean()) == 0.0                                  # gemeinsame Standardgewichte: nie


def test_coupled_demand_by_hand():
    z = np.array([1.0, -1.0])
    e = np.array([0.5, 0.5])
    assert X.coupled_demand(z, e, 0.0, 1.0).tolist() == pytest.approx((100 * np.exp(e)).tolist())             # κ = 0: nur der unabhängige Teil
    assert X.coupled_demand(z, e, 1.0, 1.0).tolist() == pytest.approx((100 * np.exp(z)).tolist())             # κ = 1: nur der gekoppelte Teil
    assert X.coupled_demand(z, e, 0.6, 0.5).tolist() == pytest.approx((100 * np.exp(0.5 * (0.6 * z + 0.8 * e))).tolist())
    assert X.coupled_demand(z, e, 0.0, 0.0).tolist() == pytest.approx([100.0, 100.0])


def test_pooled_rate_and_mean_rate_by_hand():
    q = np.array([[0.9, 0.5]])
    assert X.mean_rate(q).tolist() == [pytest.approx(0.7)]
    assert X.pooled_rate(q, np.array([[1.0, 1.0]])).tolist() == [pytest.approx(0.7)]
    assert X.pooled_rate(q, np.array([[3.0, 1.0]])).tolist() == [pytest.approx((2.7 + 0.5) / 4)]


def test_coupling_moves_the_pooled_rate_but_not_the_mean_rate():
    """Nullspalten-Regel für E5: Bei gleicher Standortquote verschiebt die Kopplung κ nur die Gesamtquote."""
    def cell(kappa, seed):
        d = S.Draws(seed)
        z = d.normal((20000, 10))
        e = d.normal((20000, 10))
        q = np.clip(0.8 + 0.08 * z, 0, 1)
        return float(X.mean_rate(q).mean()), float(X.pooled_rate(q, X.coupled_demand(z, e, kappa, 1.0)).mean())
    mean0, pooled0 = cell(0.0, 71)
    mean7, pooled7 = cell(0.7, 71)
    assert mean0 == pytest.approx(mean7, abs=1e-12)                                       # gleiche Standortquoten (gleiche z)
    assert pooled7 - pooled0 > 0.03 and abs(pooled0 - mean0) < 0.002
