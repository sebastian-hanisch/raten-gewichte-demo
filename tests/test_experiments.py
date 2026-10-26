"""Messfunktionen der Experimente in kleiner Fassung (wenige Seeds, wenige Wiederholungen): Struktur, Identitäten und die Zweige (Nullspalten-Regel).
Die großen Zahlen der Messreihen prüft tests/test_claims.py an der gespeicherten Datei."""

import math

import numpy as np
import pytest

import rgw_experiments as X
import rgw_tasks as T

FEW = X.SEEDS[:2]


def test_agg_keeps_the_per_seed_values():
    a = X.agg([1.0, 2.0, 3.0])
    assert a == {"mean": 2.0, "std": pytest.approx(math.sqrt(2 / 3)), "per_seed": [1.0, 2.0, 3.0]}


def test_source_hash_is_stable_and_covers_all_sources():
    assert X.source_hash() == X.source_hash() and len(X.source_hash()) == 16
    assert all((X.ROOT / name).exists() for name in X.SOURCES)
    assert set(X.PARTS) == {"m1", "m2", "m3", "m4", "m5"}


@pytest.mark.parametrize("task", X.E1_TASKS)
def test_cost_cell_right_mean_is_the_identity_and_the_others_cost(task):
    cell = X.cost_cell(task, 0.5, instances=300, seeds=FEW)
    right = T.RATE_TASKS[task].right_name
    assert cell["right"] == right and cell["abs"][right]["mean"] < 1e-9
    assert all(v["mean"] > 3.0 for k, v in cell["abs"].items() if k != right)             # bei σ = 0,5 kostet jedes andere Mittel mehr als 3 %
    assert list(cell["abs"]) == ["harmonisch", "geometrisch", "arithmetisch", "quadratisch"]


def test_cost_cell_costs_grow_with_the_spread_and_the_signs_are_those_of_the_order():
    low, high = X.cost_cell("strecke", 0.2, instances=300, seeds=FEW), X.cost_cell("strecke", 1.0, instances=300, seeds=FEW)
    for name in ("geometrisch", "arithmetisch", "quadratisch"):
        assert high["abs"][name]["mean"] > 5 * low["abs"][name]["mean"]
    # bei gleicher Strecke ist das harmonische Mittel das kleinste: alle anderen liegen darüber (positives Vorzeichen), das quadratische am weitesten
    s = high["signed"]
    assert s["geometrisch"]["mean"] > 0 and s["arithmetisch"]["mean"] > s["geometrisch"]["mean"] > 0 and s["quadratisch"]["mean"] > s["arithmetisch"]["mean"]
    zeit = X.cost_cell("zeit", 1.0, instances=300, seeds=FEW)["signed"]
    assert zeit["harmonisch"]["mean"] < 0 < zeit["quadratisch"]["mean"]


def test_cost_cell_verbrauch_and_zeit_have_the_same_harmonic_cost_within_noise():
    z, v = X.cost_cell("zeit", 0.5, instances=1500, seeds=FEW), X.cost_cell("verbrauch", 0.5, instances=1500, seeds=FEW)
    assert v["abs"]["harmonisch"]["mean"] == pytest.approx(z["abs"]["harmonisch"]["mean"], rel=0.05)


def test_cost_curve_has_its_minimum_at_the_right_exponent():
    for task in X.E1_TASKS:
        curve = X.cost_curve(task, 0.5, rs=(-2.0, -1.0, 0.0, 1.0, 2.0), instances=200, seeds=FEW)
        best = curve["rs"][int(np.argmin([c["mean"] for c in curve["cost"]]))]
        assert best == T.RATE_TASKS[task].right_r


def test_growth_cell_matches_the_binomial_sums_and_the_median():
    g = X.growth_cell(horizons=(10, 20, 50), N=20_000, seeds=FEW)
    assert [r["T"] for r in g["rows"]] == [10, 20, 50]
    for r in g["rows"]:
        assert r["below"]["mean"] == pytest.approx(r["exact_below"], abs=0.01) and r["above"]["mean"] == pytest.approx(r["exact_above"], abs=0.008)
        assert r["median"]["mean"] == pytest.approx(r["median_theory"], rel=1e-9) and r["median_theory"] == pytest.approx(0.9 ** (r["T"] / 2))
        assert r["expected"] == pytest.approx(1.05 ** r["T"])


def test_growth_cell_horizons_are_beginnings_of_the_same_path():
    """Gleiche Pfade für alle Horizonte: Der Anteil unter dem Start ändert sich zwischen T = 10 und T = 20 nur wie die exakte Binomialsumme."""
    g = X.growth_cell(horizons=(10, 20), N=30_000, seeds=X.SEEDS[:1])
    assert g["rows"][1]["below"]["mean"] - g["rows"][0]["below"]["mean"] == pytest.approx(G_exact(20) - G_exact(10), abs=0.012)


def G_exact(T_):
    import rgw_growth as G
    return G.exact_share_below_start(T_)


def test_lognormal_growth_cell_at_the_critical_sigma_has_half_the_paths_below_the_start():
    import rgw_growth as G
    c = X.lognormal_growth_cell(G.critical_sigma(), N=20_000, seeds=FEW)
    assert c["below"]["mean"] == pytest.approx(0.5, abs=0.015) and c["typical"]["mean"] == pytest.approx(1.0, abs=0.003) and c["exact_below"] == pytest.approx(0.5)
    assert X.lognormal_growth_cell(0.2, N=20_000, seeds=FEW)["typical"]["mean"] > 1.0 > X.lognormal_growth_cell(0.5, N=20_000, seeds=FEW)["typical"]["mean"]


def test_coverage_cell_improves_with_n_and_worsens_with_t_in_both_families():
    for family in ("zweipunkt", "lognormal"):
        by_n = [X.coverage_cell(family, 40, n, reps=60, seeds=FEW)["within"]["mean"] for n in (100, 1000, 10_000)]
        assert by_n[0] < by_n[1] < by_n[2]
        by_t = [X.coverage_cell(family, t, 1000, reps=60, seeds=FEW)["within"]["mean"] for t in (20, 60, 100)]
        assert by_t[0] > by_t[1] > by_t[2]
    with pytest.raises(ValueError):
        X.coverage_cell("gibt-es-nicht", 20, 100, reps=2, seeds=FEW)


def test_growth_live_returns_paths_and_the_same_numbers_for_the_same_seed():
    a = X.growth_live("zweipunkt", 20, 2000, 0.5, 7)
    b = X.growth_live("zweipunkt", 20, 2000, 0.5, 7)
    assert a["paths"].shape == (30, 21) and a["paths"][:, 0].tolist() == [1.0] * 30 and a["below"] == b["below"] and a["mean_ratio"] == b["mean_ratio"]
    log = X.growth_live("lognormal", 20, 2000, 0.5, 7)
    assert log["paths"].shape == (30, 21) and log["below"] != a["below"]
    with pytest.raises(ValueError):
        X.growth_live("x", 20, 100, 0.5, 1)


def test_bias_cell_matches_the_formula_for_light_tails_and_has_none_for_heavy_ones():
    c = X.bias_cell("poisson", 3.0, lots=2000, pieces=100_000, seeds=FEW)
    assert c["measured"]["mean"] == pytest.approx(c["formula"], rel=0.03) and c["population"]["mean"] == pytest.approx(c["formula"], rel=0.05)
    assert math.isinf(X.bias_cell("pareto", 2.0, lots=500, pieces=2000, seeds=FEW)["formula"])


def test_back_cell_arithmetic_is_biased_and_harmonic_is_not():
    cell = X.back_cell("lognormal", 1.0, 100, reps=300, seeds=FEW)
    assert cell["am_bias"]["mean"] == pytest.approx(172.0, abs=25.0) and abs(cell["hm_bias"]["mean"]) < 6.0
    assert cell["hm_rmse"]["mean"] < cell["am_rmse"]["mean"] / 5 and "hm_bias_nonempty" not in cell
    poisson = X.back_cell("poisson", 3.0, 1000, reps=200, seeds=FEW)
    assert poisson["hm_bias"]["mean"] == pytest.approx(5.2, abs=1.0) and abs(poisson["hm_bias_nonempty"]["mean"]) < 1.0 and poisson["nonempty"] == pytest.approx(3.157, abs=1e-3)


def test_back_cell_every_family_differs_in_the_arithmetic_bias():
    biases = {f: X.back_cell(f, p, 100, reps=100, seeds=X.SEEDS[:1])["am_bias"]["mean"] for f, p in (("poisson", 3.0), ("geometrisch", 0.3), ("lognormal", 0.5), ("pareto", 3.0))}
    assert len({round(b) for b in biases.values()}) >= 3 and all(b > 15 for b in biases.values())


def test_tail_cell_grows_without_bound_for_alpha_two_and_settles_for_alpha_three():
    t2 = X.tail_cell(2.0, runs=((100, 200), (100_000, 30)), seeds=X.SEEDS[:2])
    assert t2["formula"] is None and t2["rows"][1]["median"]["mean"] > 1.5 * t2["rows"][0]["median"]["mean"]
    t3 = X.tail_cell(3.0, runs=((100, 200), (100_000, 30)), seeds=X.SEEDS[:2])
    assert t3["formula"] == pytest.approx(2.0) and t3["rows"][1]["median"]["mean"] == pytest.approx(2.0, rel=0.05)


def test_reversal_cell_counts_and_falls_with_the_layers():
    c2, c4 = X.reversal_cell(2, 200_000, seeds=FEW), X.reversal_cell(4, 200_000, seeds=FEW)
    assert c2["hits"] == round(sum(c2["share"]["per_seed"]) * 200_000) and c2["share"]["mean"] > 5 * c4["share"]["mean"] > 0
    assert c2["standard_error"] == pytest.approx(math.sqrt(c2["share"]["mean"] * (1 - c2["share"]["mean"]) / 400_000), rel=0.01)
    assert X.reversal_live(3, 50_000, 5)["tables_per_seed"] == 50_000


def test_weight_cell_equal_weights_never_flip_and_spread_weights_do():
    zero = X.weight_cell(2, 0.0, 20_000, seeds=FEW)
    assert zero["share"]["mean"] == 0.0 and zero["tv_mean"]["mean"] == pytest.approx(0.0, abs=1e-12) and zero["tv_flipped_median"] is None
    wide = X.weight_cell(2, 1.5, 20_000, seeds=FEW)
    assert wide["share"]["mean"] > 0.03 and wide["tv_flipped_median"]["mean"] > wide["tv_mean"]["mean"]
    for cell in (zero, wide, X.weight_cell(5, 1.0, 20_000, seeds=FEW)):
        assert cell["equal_weight_flips"] == 0 and cell["standard_weight_flips"] == 0                       # erzwungen: Probe, kein Befund


def test_weight_cell_flips_fall_with_more_layers_at_the_same_spread():
    two, five = X.weight_cell(2, 1.0, 40_000, seeds=FEW), X.weight_cell(5, 1.0, 40_000, seeds=FEW)
    assert two["share"]["mean"] > five["share"]["mean"] > 0


def test_coupling_cell_only_the_pooled_rate_moves_with_kappa():
    zero = X.coupling_cell(0.0, 1.0, 4000, seeds=FEW)
    assert abs(zero["diff_mean_pp"]["mean"]) < 0.2 and abs(zero["diff_pooled_pp"]["mean"]) < 0.3 and zero["b_ahead_pooled"]["mean"] == pytest.approx(0.5, abs=0.03)
    pos = X.coupling_cell(0.7, 1.0, 4000, seeds=FEW)
    assert abs(pos["diff_mean_pp"]["mean"]) < 0.2 and pos["diff_pooled_pp"]["mean"] > 3.0 and pos["b_ahead_pooled"]["mean"] > 0.7 and pos["b_ahead_mean"]["mean"] == pytest.approx(0.5, abs=0.03)
    neg = X.coupling_cell(-0.7, 1.0, 4000, seeds=FEW)
    assert neg["diff_pooled_pp"]["mean"] < -3.0 and neg["b_ahead_pooled"]["mean"] < 0.3


def test_coupling_effect_grows_with_the_demand_spread():
    small, large = X.coupling_cell(0.7, 0.5, 4000, seeds=FEW), X.coupling_cell(0.7, 1.5, 4000, seeds=FEW)
    assert large["diff_pooled_pp"]["mean"] > small["diff_pooled_pp"]["mean"] > 1.0
