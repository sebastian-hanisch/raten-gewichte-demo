"""Szenario: der blockweise Strom ist der Strom der kopierten Klasse; die Verteilungen haben ihre dokumentierten Eigenschaften (Tabellen gegen scipy, Mittel, Grenzen)."""

import math

import numpy as np
import pytest

import rgw_scenario as S
from rgw_rng import SplitMix64


@pytest.mark.parametrize("seed", [0, 1, 7, 2**63 + 5, 999999])
def test_block_stream_equals_the_copied_splitmix64(seed):
    rng = SplitMix64(seed)
    expected = [rng.next() for _ in range(50)]
    assert S.Draws(seed).raw(50).tolist() == expected


def test_blocks_continue_the_stream_where_the_last_block_ended():
    d = S.Draws(5)
    first, second = d.raw(7), d.raw(9)
    assert np.concatenate([first, second]).tolist() == S.Draws(5).raw(16).tolist()
    assert S.Draws(5).raw(0).size == 0


def test_uniform_is_the_top_53_bits_of_the_stream():
    rng = SplitMix64(3)
    expected = [(rng.next() >> 11) / 2**53 for _ in range(10)]
    assert S.Draws(3).uniform(10).tolist() == expected
    u = S.Draws(4).uniform((1000, 3))
    assert u.min() >= 0 and u.max() < 1 and abs(u.mean() - 0.5) < 0.03


def test_normal_by_hand_from_two_uniforms():
    """Box-Muller: z0 = √(−2 ln(1 − u1))·cos(2π u2), z1 = … sin(2π u2); bei Größe 2 steht z0 zuerst."""
    u = S.Draws(9).uniform(2)
    r = math.sqrt(-2 * math.log(1 - u[0]))
    z = S.Draws(9).normal(2)
    assert z.tolist() == pytest.approx([r * math.cos(2 * math.pi * u[1]), r * math.sin(2 * math.pi * u[1])])


def test_normal_has_mean_zero_and_variance_one():
    z = S.Draws(10).normal(200000)
    assert abs(z.mean()) < 0.01 and abs(z.std() - 1) < 0.01
    assert S.Draws(10).normal(3).shape == (3,)
    assert S.Draws(10).normal((4, 2, 2)).shape == (4, 2, 2)


def test_exponential_has_mean_one_and_is_positive():
    e = S.Draws(11).exponential(200000)
    assert e.min() >= 0 and abs(e.mean() - 1) < 0.01 and abs(e.var() - 1) < 0.03


def test_cell_seed_is_stable_and_separates_cells():
    assert S.cell_seed(5, "a", 1.0) == S.cell_seed(5, "a", 1.0)
    assert len({S.cell_seed(5, "a", 1.0), S.cell_seed(5, "b", 1.0), S.cell_seed(5, "a", 2.0), S.cell_seed(6, "a", 1.0)}) == 4
    assert S.cell_seed(5) == 5


def test_poisson_table_matches_scipy_and_sums_to_one():
    stats = pytest.importorskip("scipy.stats")
    for lam in (0.5, 1.0, 3.0, 10.0, 30.0):
        pmf = S.poisson_pmf(lam)
        assert pmf.sum() == pytest.approx(1.0, abs=1e-12)
        assert pmf == pytest.approx(stats.poisson.pmf(np.arange(len(pmf)), lam), abs=1e-14)
    with pytest.raises(ValueError):
        S.poisson_pmf(0.0)


def test_binomial_table_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    for n, p in ((10, 0.5), (100, 0.5), (200, 0.5), (30, 0.2)):
        pmf = S.binomial_pmf(n, p)
        assert pmf.sum() == pytest.approx(1.0, abs=1e-12)
        assert pmf == pytest.approx(stats.binom.pmf(np.arange(n + 1), n, p), abs=1e-13)


def test_discrete_draws_have_the_right_mean_variance_and_support():
    x = S.poisson(S.Draws(12), 200000, 3.0)
    assert abs(x.mean() - 3.0) < 0.03 and abs(x.var() - 3.0) < 0.06 and x.min() == 0
    b = S.binomial(S.Draws(13), 200000, 100, 0.5)
    assert abs(b.mean() - 50) < 0.1 and abs(b.var() - 25) < 0.4 and b.min() >= 0 and b.max() <= 100
    g = S.geometric(S.Draws(14), 200000, 0.3)
    assert g.min() == 1 and abs(g.mean() - 1 / 0.3) < 0.05 and abs(g.var() - 0.7 / 0.09) < 0.2
    assert abs((g == 1).mean() - 0.3) < 0.005 and abs((g == 2).mean() - 0.21) < 0.005       # P(K = 1) = p, P(K = 2) = (1 − p)·p


def test_draw_from_pmf_by_hand():
    """Tabelle (0,25; 0,5; 0,25): Gleichverteilte 0,1 → 0; 0,25 → 1 (Rand gehört nach oben); 0,74 → 1; 0,75 → 2; 0,99 → 2."""
    class Fixed(S.Draws):
        def uniform(self, shape):
            return np.array([0.1, 0.25, 0.74, 0.75, 0.99])
    assert S.draw_from_pmf(Fixed(0), 5, np.array([0.25, 0.5, 0.25])).tolist() == [0, 1, 1, 2, 2]


def test_lognormal_and_pareto_by_the_formula():
    x = S.lognormal(S.Draws(15), 200000, 0.5, 10.0)
    assert abs(np.median(x) / 10.0 - 1) < 0.01 and abs(x.mean() / (10 * math.exp(0.125)) - 1) < 0.01
    p = S.pareto(S.Draws(16), 200000, 3.0, 1.0)
    assert p.min() >= 1.0 and abs(p.mean() / 1.5 - 1) < 0.02 and abs((p > 2.0).mean() - 0.125) < 0.003       # P(X > 2) = 2^(−3)


def test_invalid_parameters_are_errors():
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError):
            S.geometric(S.Draws(1), 3, bad)
    with pytest.raises(ValueError):
        S.pareto(S.Draws(1), 3, 0.0)
