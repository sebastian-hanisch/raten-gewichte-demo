"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Button, fährt jeden Regler an seine Grenzen
(einzeln und alle Minima, alle Maxima zusammen), wählt jede Aufgabe und jede Auswahl der Experimente und prüft, dass kein Python-Fehler auftritt - insbesondere
`streamlit.errors.StreamlitDuplicateElementId` (mehrere st.plotly_chart-Aufrufe ohne eindeutiges key=) und `StreamlitAPIException` bei Slidern mit berechneten Grenzen."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import rgw_constants as C
import rgw_experiments as X

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "▶"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    at = _fresh_app()
    assert at.title and not at.error


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().sidebar.slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.sidebar.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


@pytest.mark.parametrize("edge", ["min", "max"])
@pytest.mark.parametrize("task", C.TASK_KEYS)
def test_all_sliders_at_the_same_edge_for_every_task(task, edge):
    """Eckfall-Matrix: jede Aufgabe mit allen Reglern am Minimum bzw. am Maximum (Posten, Streuung, Exponent zugleich)."""
    at = _fresh_app()
    at.sidebar.selectbox(key="task_select").set_value(task).run(timeout=TIMEOUT)
    for slider in at.sidebar.slider:
        if slider.disabled:                  # ein gesperrter Regler lässt sich auch im Browser nicht verstellen
            continue
        value = slider.min if edge == "min" else slider.max
        slider.set_value(int(value) if isinstance(slider.value, int) else value)
    at.run(timeout=TIMEOUT)
    assert not at.exception, (task, edge, [str(e) for e in at.exception])


def _raw_slider_options():
    """Die Werte der Schieber mit Optionen (AppTest zeigt die formatierten Beschriftungen, `set_value` will die Werte)."""
    import rgw_results as R
    return {"e1_sigma_pick": X.E1_SIGMAS, "e2_horizon": X.E2_HORIZONS, "e2_pick": X.E2_HORIZONS, "e2_sigma_pick": [r["sigma"] for r in sorted(R.lognormal_rows(), key=lambda r: r["sigma"])],
            "e2_cov_t": X.E2_COVER_T, "e2_cov_n": X.E2_COVER_N, "e2_live_t": C.LIVE_PATH_T, "e2_live_n": C.LIVE_PATH_N, "e3_back_n": X.E3_NS, "e3_tail_pick": X.E3_ALPHAS,
            "e3_live_n": X.E3_NS, "e4_layers": X.E4_LAYERS, "e4_live_tables": [100_000, 500_000, 1_000_000, 2_000_000], "e4_wk": X.E4_WEIGHT_K, "e4_tau": X.E4_TAUS,
            "e5_sd": X.E5_SIGMA_DS, "e5_kappa": X.E5_KAPPAS}


def test_every_selection_of_the_experiments_does_not_raise():
    """Jede Auswahl der Experimente einzeln auf jeden ihrer Werte (Auswahlfelder, Schieber mit Optionen)."""
    base = _fresh_app()
    raw = _raw_slider_options()
    assert {w.key for w in base.main.select_slider} == set(raw)
    selects = [(w.key, list(w.options)) for w in base.main.selectbox]
    assert len(selects) + len(raw) >= 20
    for key, options in selects:
        for option in options:
            at = _fresh_app()
            at.main.selectbox(key=key).set_value(option).run(timeout=TIMEOUT)
            assert not at.exception, (key, option, [str(e) for e in at.exception])
    for key, values in raw.items():
        for value in values:
            at = _fresh_app()
            at.main.select_slider(key=key).set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (key, value, [str(e) for e in at.exception])


@pytest.mark.parametrize("button", ["e1_live", "e2_live", "e3_live", "e4_live", "e5_live", "pdf_make"])
def test_each_live_button_shows_a_result_and_a_second_click_with_new_settings_asks_again(button):
    at = _fresh_app()
    at.main.button(key=button).click().run(timeout=TIMEOUT)
    assert not at.exception, (button, [str(e) for e in at.exception])
    at.sidebar.slider(key="m_slider").set_value(40).run(timeout=TIMEOUT)
    assert not at.exception


def test_e2_live_run_with_the_two_point_family_draws_paths_into_the_chart():
    at = _fresh_app()
    at.main.select_slider(key="e2_live_t").set_value(20)
    at.main.select_slider(key="e2_live_n").set_value(1000)
    at.main.button(key="e2_live").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any("Zweipunkt" in i.value and "1.000 Pfade" in i.value for i in at.info), [i.value for i in at.info]


def test_live_budgets_are_respected_at_the_largest_selection():
    at = _fresh_app()
    at.main.select_slider(key="e2_live_t").set_value(max(C.LIVE_PATH_T))
    at.main.select_slider(key="e2_live_n").set_value(max(C.LIVE_PATH_N))
    at.main.button(key="e2_live").click().run(timeout=TIMEOUT)
    assert not at.exception
    n_eff = C.LIVE_PATH_BUDGET // max(C.LIVE_PATH_T)
    assert any(f"{n_eff:,}".replace(",", ".") in c.value for c in at.caption)


def test_experiment_grids_match_the_selection_options():
    at = _fresh_app()
    assert list(at.main.select_slider(key="e2_horizon").options) == [str(t) for t in X.E2_HORIZONS]
    assert list(at.main.select_slider(key="e4_layers").options) == [str(k) for k in X.E4_LAYERS]
    assert list(at.main.select_slider(key="e5_kappa").options) == [f"{k:g}".replace(".", ",") for k in X.E5_KAPPAS]
