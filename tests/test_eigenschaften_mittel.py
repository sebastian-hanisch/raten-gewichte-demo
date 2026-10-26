"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2, metamorph): Aussagen, die für Mittel, Aufgaben, Größenverzerrung und Tabellen gelten müssen, ohne dass man das richtige Ergebnis kennt.
Je Eigenschaft 30 bis 50 Zufallsinstanzen mit festem Seed."""

import numpy as np
import pytest

import rgw_growth as G
import rgw_means as M
import rgw_mix as X
import rgw_scenario as S
import rgw_sizebias as Z
import rgw_tasks as T

RNG = np.random.default_rng(2026)           # nur im Test: Prüfmengen
RS = (-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0)


def _sets(count=40):
    for _ in range(count):
        n = int(RNG.integers(2, 20))
        yield np.exp(RNG.uniform(0.1, 1.5) * RNG.standard_normal(n)), RNG.uniform(0.2, 4.0, n)


@pytest.mark.parametrize("r", RS)
def test_scale_equivariance_of_the_power_mean(r):
    for x, w in _sets():
        c = float(RNG.uniform(0.1, 50))
        assert M.power_mean(c * x, r, w) == pytest.approx(c * M.power_mean(x, r, w), rel=1e-11)


@pytest.mark.parametrize("r", RS)
def test_permutation_and_weight_scaling_do_not_change_the_mean(r):
    for x, w in _sets():
        perm = RNG.permutation(len(x))
        assert M.power_mean(x[perm], r, w[perm]) == pytest.approx(M.power_mean(x, r, w), rel=1e-12)
        assert M.power_mean(x, r, 7.5 * w) == pytest.approx(M.power_mean(x, r, w), rel=1e-12)


@pytest.mark.parametrize("r", RS)
def test_duality_of_the_exponent(r):
    """M_(−r)(x) = 1 / M_r(1/x): das harmonische Mittel ist der Kehrwert des arithmetischen Mittels der Kehrwerte, und so für jedes r."""
    for x, w in _sets():
        assert M.power_mean(x, -r, w) == pytest.approx(1.0 / M.power_mean(1.0 / x, r, w), rel=1e-11)


def test_a_mean_lies_between_the_smallest_and_the_largest_value_and_is_monotone_in_each_value():
    for x, w in _sets():
        for r in RS:
            v = M.power_mean(x, r, w)
            assert x.min() * (1 - 1e-12) <= v <= x.max() * (1 + 1e-12)
            bigger = x.copy()
            bigger[0] *= 1.5
            assert M.power_mean(bigger, r, w) > v


def test_task_truth_scales_with_the_rates_and_is_unchanged_by_the_order():
    """Alle Geschwindigkeiten mit c multipliziert: gleiche Strecke und Zeit/c, die Gesamtgeschwindigkeit wird mit c multipliziert; Umordnen der Fahrten ändert nichts."""
    for key in ("strecke", "zeit", "verbrauch"):
        x, truth = T.make_batch(key, S.Draws(5), 30, 8, 0.6)
        c = 1.7
        assert T.true_total(key, c * x) == pytest.approx(c * truth, rel=1e-12)
        assert T.true_total(key, x[:, ::-1]) == pytest.approx(truth, rel=1e-12)
    f, truth = T.make_batch("faktoren", S.Draws(6), 30, 8, 0.6)
    assert T.true_total("faktoren", f[:, ::-1]) == pytest.approx(truth, rel=1e-12) and T.true_total("faktoren", 2.0 * f) == pytest.approx(2.0 * truth, rel=1e-12)


def test_growth_same_seed_same_paths_and_more_periods_never_remove_up_steps():
    a = G.ups_prefix(S.Draws(9), 500, (10, 30))
    b = G.ups_prefix(S.Draws(9), 500, (10, 30))
    assert a[10].tolist() == b[10].tolist() and a[30].tolist() == b[30].tolist()
    assert np.all(a[30] >= a[10]) and np.all(a[30] - a[10] <= 20)


def test_exact_below_share_grows_over_the_even_horizons_used_in_the_demo():
    """Je länger der Zeitraum, desto größer der Anteil der Pfade unter dem Start, an den Horizonten der Demo (nicht über jede Periode: Die Zweipunktverteilung hat Gitterartefakte, die Werte springen)."""
    horizons = [10, 20, 50, 100, 200]
    values = [G.exact_share_below_start(t) for t in horizons]
    assert all(b > a for a, b in zip(values, values[1:])) and values[0] > 0.5 and values[-1] > 0.9


def test_size_bias_is_scale_free_and_the_harmonic_estimator_is_scale_equivariant():
    for family, par in (("poisson", 4.0), ("geometrisch", 0.4), ("lognormal", 0.8)):
        sizes = Z.lot_sizes(family, par, S.Draws(11), 400)
        sizes = sizes[sizes > 0]
        c = 3.0
        assert Z.bias_factor(c * sizes) == pytest.approx(Z.bias_factor(sizes), rel=1e-12)
        obs = Z.pieces_from_population(sizes, 60, S.Draws(12))
        obs_c = Z.pieces_from_population(c * sizes, 60, S.Draws(12))
        assert obs_c.tolist() == pytest.approx((c * obs).tolist()) and float(Z.harmonic_back(obs_c)) == pytest.approx(c * float(Z.harmonic_back(obs)))


def test_harmonic_never_exceeds_arithmetic_on_the_observations():
    for seed in range(40):
        sizes = Z.lot_sizes("lognormal", 1.0, S.Draws(seed), 300)
        obs = Z.pieces_from_population(sizes, 30, S.Draws(1000 + seed))
        assert float(Z.harmonic_back(obs)) <= float(Z.arithmetic_back(obs)) * (1 + 1e-12)


def test_swapping_the_groups_or_the_layers_does_not_change_whether_it_flips():
    tables = X.random_tables(S.Draws(21), 5000, 3)
    flips = X.reversal_mask(tables)
    assert flips.tolist() == X.reversal_mask(tables[:, :, ::-1, :]).tolist()                          # Gruppen vertauscht
    assert flips.tolist() == X.reversal_mask(tables[:, ::-1, :, :]).tolist()                          # Schichten umgeordnet
    assert flips.tolist() == X.reversal_mask(100.0 * tables).tolist()                                  # alle Zahlen mit demselben Faktor (Häufigkeiten statt Anteile)


def test_standard_weights_and_equal_weights_never_flip_on_random_dominated_tables():
    for tau in (0.5, 1.5):
        rates, w = X.dominated_rates_and_weights(S.Draws(22), 20000, 4, tau)
        std = X.standardised_totals(rates, w)
        assert np.all(std[:, 0] >= std[:, 1])
        same = X.weighted_total(rates, w[..., 0:1])
        assert np.all(same[:, 0] >= same[:, 1])


def test_pooled_rate_is_invariant_to_scaling_the_demand_and_the_mean_rate_ignores_the_demand():
    z, e = S.Draws(23).normal((200, 10)), S.Draws(24).normal((200, 10))
    q = X.service_levels(z, 0.8, 0.08)
    d = X.coupled_demand(z, e, 0.5, 1.0)
    assert X.pooled_rate(q, 5.0 * d) == pytest.approx(X.pooled_rate(q, d), rel=1e-12)
    assert X.mean_rate(q) == pytest.approx(X.pooled_rate(q, np.ones_like(q)), rel=1e-12)
    assert np.all((X.pooled_rate(q, d) >= q.min(axis=1)) & (X.pooled_rate(q, d) <= q.max(axis=1)))
