"""Die vorgerechneten Messreihen (data/rgw_results.json): vollständig, in sich stimmig, und die gespeicherten Zellen werden von den Live-Funktionen reproduziert."""

import json
import math

import pytest

import rgw_constants as C
import rgw_experiments as X
import rgw_growth as G
import rgw_results as R


def test_results_file_is_valid_utf8_json_with_all_parts():
    text = R.PATH.read_text(encoding="utf-8")
    data = json.loads(text)
    assert set(data) == {"meta", "m1", "m2", "m3", "m4", "m5"}
    assert "\r" not in text
    assert R.meta()["stand"] == "2026-10-26" and R.meta()["seeds"] == list(X.SEEDS)


def test_e1_tables_are_complete():
    assert [(c["task"], c["sigma"]) for c in R.cost_cells()] == [(t, s) for t in X.E1_TASKS for s in X.E1_SIGMAS]
    for c in R.cost_cells():
        assert c["m"] == X.E1_M and c["instances_per_seed"] == X.E1_INSTANCES and set(c["abs"]) == set(c["signed"]) == {"harmonisch", "geometrisch", "arithmetisch", "quadratisch"}
        assert all(len(v["per_seed"]) == len(X.SEEDS) for v in c["abs"].values())
    assert len(R.load()["m1"]["curves"]) == len(X.E1_TASKS) * len(X.E1_CURVE_SIGMAS)
    assert R.cost_curve("zeit", 0.5)["rs"] == list(X.E1_R_GRID)


def test_e2_tables_are_complete():
    assert [r["T"] for r in R.growth_rows()] == list(X.E2_HORIZONS) and R.load()["m2"]["growth"]["N"] == X.E2_N
    assert [round(r["sigma"], 6) for r in R.lognormal_rows()] == [round(s, 6) for s in X.E2_LOG_SIGMAS] + [round(G.critical_sigma(), 6)]
    assert len(R.coverage_rows()) == 2 * len(X.E2_COVER_T) * len(X.E2_COVER_N)
    assert R.coverage_row("zweipunkt", 60, 10_000)["reps_per_seed"] == X.E2_COVER_REPS


def test_e3_tables_are_complete():
    assert [(r["family"], r["par"]) for r in R.bias_rows()] == list(X.E3_BIAS_FAMILIES)
    assert [(r["family"], r["par"], r["n"]) for r in R.back_rows()] == [(f, p, n) for f, p in X.E3_BACK_FAMILIES for n in X.E3_NS]
    assert [r["alpha"] for r in R.tail_rows()] == list(X.E3_ALPHAS)
    assert [[x["n"] for x in r["rows"]] for r in R.tail_rows()] == [[n for n, _ in X.E3_TAIL_RUNS]] * len(X.E3_ALPHAS)
    assert R.back_row("poisson", 3.0, 1000)["hm_bias_nonempty"] and "hm_bias_nonempty" not in R.back_row("lognormal", 1.0, 100)


def test_e4_and_e5_tables_are_complete():
    assert [r["layers"] for r in R.reversal_rows()] == list(X.E4_LAYERS)
    assert [(r["k"], r["tau"]) for r in R.weight_rows()] == [(k, t) for k in X.E4_WEIGHT_K for t in X.E4_TAUS]
    assert [(c["kappa_b"], c["sigma_d"]) for c in R.coupling_cells()] == [(k, s) for k in X.E5_KAPPAS for s in X.E5_SIGMA_DS]
    assert all(r["equal_weight_flips"] == 0 and r["standard_weight_flips"] == 0 for r in R.weight_rows())


def test_shares_and_means_are_consistent():
    for r in R.growth_rows():
        assert 0 <= r["below"]["mean"] <= 1 and r["below"]["mean"] == pytest.approx(sum(r["below"]["per_seed"]) / 4)
    for r in R.reversal_rows():
        assert r["hits"] == round(sum(r["share"]["per_seed"]) * r["tables_per_seed"])
    for r in R.bias_rows():
        assert r["measured"]["mean"] > 1.0


def assert_same_cell(live, stored, rel=1e-9):
    """Gleiche Struktur, Zeichenketten und ganze Zahlen exakt, Gleitkommazahlen bis auf die letzte Binärstelle: `log`, `cos` und die Summation können je Plattform
    (Windows, Linux) in der letzten Stelle abweichen; die Messreihe wird auf einem anderen Rechner nachgerechnet."""
    if isinstance(stored, dict):
        assert isinstance(live, dict) and set(live) == set(stored)
        for key in stored:
            assert_same_cell(live[key], stored[key], rel)
    elif isinstance(stored, (list, tuple)):
        assert isinstance(live, (list, tuple)) and len(live) == len(stored)
        for a, b in zip(live, stored):
            assert_same_cell(a, b, rel)
    elif isinstance(stored, float):
        assert live == pytest.approx(stored, rel=rel, abs=1e-15) if math.isfinite(stored) else live == stored
    else:
        assert live == stored


def test_the_tolerant_comparison_distinguishes_a_real_difference():
    assert_same_cell({"a": [1.0, 2], "b": "x"}, {"a": [1.0 + 1e-13, 2], "b": "x"})
    for other in ({"a": [1.001, 2], "b": "x"}, {"a": [1.0, 3], "b": "x"}, {"a": [1.0, 2], "b": "y"}, {"a": [1.0, 2]}):
        with pytest.raises(AssertionError):
            assert_same_cell(other, {"a": [1.0, 2], "b": "x"})


def test_stored_cells_are_reproduced_by_the_live_functions():
    """Das Nachmessen in der App liefert dieselben Zahlen wie die vorgerechnete Zelle (gleiche Seeds, gleiche Läufe)."""
    assert_same_cell(X.cost_cell("strecke", 0.5), R.cost_cell("strecke", 0.5))
    assert_same_cell(X.cost_curve("faktoren", 0.2), R.cost_curve("faktoren", 0.2))
    assert_same_cell(X.lognormal_growth_cell(0.5), R.lognormal_row(0.5))
    assert_same_cell(X.coverage_cell("lognormal", 20, 100), R.coverage_row("lognormal", 20, 100))
    assert_same_cell(X.bias_cell("poisson", 1.0), R.bias_row("poisson", 1.0))
    assert_same_cell(X.back_cell("pareto", 2.2, 20), R.back_row("pareto", 2.2, 20))
    assert_same_cell(X.back_cell("poisson", 3.0, 20), R.back_row("poisson", 3.0, 20))
    assert_same_cell(X.weight_cell(3, 0.5), R.weight_row(3, 0.5))
    assert_same_cell(X.weight_cell(2, 0.0), R.weight_row(2, 0.0))
    assert_same_cell(X.coupling_cell(0.7, 1.0), R.coupling_cell(0.7, 1.0))


def test_stored_growth_table_is_reproduced():
    live = X.growth_cell()
    assert_same_cell(live["rows"], R.growth_rows())


def test_stored_reversal_and_tail_cells_are_reproduced():
    assert_same_cell(X.reversal_cell(4), R.reversal_row(4))
    assert_same_cell(X.tail_cell(3.0), R.tail_row(3.0))


def test_results_path_can_be_overridden(tmp_path, monkeypatch):
    import importlib
    monkeypatch.setenv("RGW_RESULTS_PATH", str(tmp_path / "x.json"))
    mod = importlib.reload(R)
    assert mod.PATH == tmp_path / "x.json"
    monkeypatch.delenv("RGW_RESULTS_PATH")
    importlib.reload(R)
    assert R.PATH.name == C.RESULTS_FILE
