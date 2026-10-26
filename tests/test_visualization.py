"""Plotly-Abbildungen: Jede baut für jede Aufgabe und jede gespeicherte Messreihe, alle Achsen sind fest, die Daten sind endlich, logarithmische Achsen bekommen nur positive Werte."""

import math

import numpy as np
import pytest

import rgw_constants as C
import rgw_evaluation as V
import rgw_experiments as X
import rgw_results as R
import rgw_visualization as W


def _check(fig, log_axes=()):
    assert fig.layout.dragmode is False
    for axis in (fig.layout.xaxis, fig.layout.yaxis):
        assert axis.fixedrange is True
    for trace in fig.data:
        for name in ("x", "y"):
            values = getattr(trace, name, None)
            if values is None or len(values) == 0 or isinstance(values[0], str):
                continue
            arr = np.array([v for v in values if v is not None], dtype=float)
            assert np.all(np.isfinite(arr)), (trace.name, name)
            if name in log_axes:
                assert np.all(arr > 0), (trace.name, name)
    return fig


@pytest.mark.parametrize("task", sorted(C.RATE_TASK_KEYS))
def test_rate_figures_for_every_task_and_the_extreme_exponents(task):
    for r in (C.R_MIN, 0.0, C.R_MAX):
        a = V.analyse(V.Settings(task=task, r=r, sigma=C.SIGMA_MAX if task != "wechsel" else 0.5))
        fig = _check(W.build_rate_curve(a))
        assert len(a.curve) == len(C.R_CURVE)
        names = [t.name for t in fig.data]
        assert "Potenzmittel M_r" in names and f"gewähltes r = {r:g}" in names
        _check(W.build_error_bars(V.mean_rows(a)))


def test_error_bars_mark_the_right_mean_and_use_one_bar_per_mean():
    a = V.analyse(V.Settings(task="zeit"))
    fig = W.build_error_bars(V.mean_rows(a))
    assert [t.name for t in fig.data] == ["Harmonisch", "Geometrisch", "Arithmetisch", "Quadratisch"]
    patterns = {t.name: t.marker.pattern.shape for t in fig.data}
    assert patterns["Arithmetisch"] == "/" and patterns["Harmonisch"] == ""


def test_lots_and_table_figures():
    for m in (C.M_MIN, 50, C.M_MAX):
        a = V.analyse(V.Settings(task="lose", m=m, sigma=1.0))
        fig = _check(W.build_lots(a), log_axes=("x",))
        assert fig.layout.xaxis.type == "log" and len(fig.data) == 2
    t = _check(W.build_table_chart(V.analyse(V.Settings(task="tabelle"))))
    assert [tr.name for tr in t.data] == ["Gruppe A", "Gruppe B"] and list(t.data[0].x) == list(W.LAYER_TITLES)
    assert [round(v) for v in t.data[0].y] == [90, 60, 63, 75] and [round(v) for v in t.data[1].y] == [80, 40, 76, 60]


def test_experiment_figures_from_the_stored_results():
    for task in X.E1_TASKS:
        cells = sorted([c for c in R.cost_cells() if c["task"] == task], key=lambda c: c["sigma"])
        _check(W.build_cost_by_sigma(cells, task))
        assert len(W.build_cost_by_sigma(cells, task).data) == 4
        _check(W.build_cost_by_r([c for c in R.load()["m1"]["curves"] if c["task"] == task], task))
    for horizon in X.E2_HORIZONS:
        _check(W.build_paths_chart(horizon, R.growth_rows(), None), log_axes=("y",))
    live = X.growth_live("zweipunkt", 20, 500, 0.5, 3)
    fig = _check(W.build_paths_chart(50, R.growth_rows(), live), log_axes=("y",))
    assert sum(1 for t in fig.data if t.name == "Pfade der Live-Messung" and t.showlegend) == 1
    _check(W.build_lognormal_chart(R.lognormal_rows()))
    for family in ("zweipunkt", "lognormal"):
        assert len(_check(W.build_coverage_chart(R.coverage_rows(), family)).data) == 3
    _check(W.build_bias_chart(R.bias_rows()), log_axes=("y",))
    for f, p in X.E3_BACK_FAMILIES:
        _check(W.build_back_chart([r for r in R.back_rows() if r["family"] == f and r["par"] == p]))
    _check(W.build_tail_chart(R.tail_rows()), log_axes=("x", "y"))
    _check(W.build_reversal_chart(R.reversal_rows()))
    _check(W.build_weights_chart(R.weight_rows()))
    for sd in X.E5_SIGMA_DS:
        fig = _check(W.build_coupling_chart([c for c in R.coupling_cells() if c["sigma_d"] == sd]))
        assert len(fig.data) == 2


def test_tail_chart_draws_the_formula_only_for_finite_means():
    fig = W.build_tail_chart(R.tail_rows())
    lines = [t for t in fig.data if t.mode == "lines"]
    assert len(lines) == sum(1 for r in R.tail_rows() if r["formula"] is not None) == 2
    assert all(math.isfinite(y) for t in lines for y in t.y)


def test_bias_chart_leaves_out_the_formula_where_it_does_not_exist():
    rows = R.bias_rows() + [{"family": "pareto", "par": 2.0, "formula": math.inf, "measured": {"mean": 5.0}}]
    fig = W.build_bias_chart(rows)
    assert fig.data[1].y[-1] is None and fig.data[1].y[0] == pytest.approx(2.0)
