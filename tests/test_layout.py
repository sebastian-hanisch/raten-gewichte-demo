"""Layout (DEMO-PLAYBOOK §2): fester Kopf, Schnellstart oben in der Seitenleiste, aktives Preset hervorgehoben.

Die Hervorhebung ist zustandslos (`aktives_preset` vergleicht jeden Regler aus SETTING_SPECS mit dem Preset). Der Test unten stellt
sicher, dass wirklich JEDER Regler der Seitenleiste dort steht: sonst bliebe das Preset trotz Änderung markiert."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import rgw_constants as C
import rgw_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")
REGLER_ARTEN = ("slider", "number_input", "select_slider", "selectbox", "radio", "toggle", "checkbox", "text_input")


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _typen(at) -> dict:
    """Button-Typ je Preset-Name (die Preset-Buttons der Seitenleiste)."""
    return {b.label: b.proto.type for b in at.sidebar.button if b.label in C.PRESETS}


def _regler(at, key):
    for art in REGLER_ARTEN:
        try:
            return getattr(at.sidebar, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(f"Regler {key} steht nicht in der Seitenleiste")


def test_presets_stehen_in_der_seitenleiste_und_nicht_im_hauptbereich():
    at = _app()
    assert [b.label for b in at.sidebar.button][: len(C.PRESETS)] == list(C.PRESETS)  # die ersten Buttons der Seitenleiste
    assert not [b for b in at.main.button if b.label in C.PRESETS]


def test_titel_steht_im_festen_kopf():
    at = _app()
    assert "st-key-kopf" in " ".join(h.proto.body for h in at.get("html"))  # das Stylesheet setzt an der Klasse des Containers an
    assert at.title[0].value


def test_beim_start_ist_genau_das_standard_preset_hervorgehoben():
    typen = _typen(_app())
    assert [name for name, typ in typen.items() if typ == "primary"] == [next(iter(C.PRESETS))]


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_angeklicktes_preset_ist_hervorgehoben_und_bleibt_es_im_naechsten_lauf(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]
    at.run()
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_setzt_alle_regler_auf_seine_werte(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    for key, state_key in P.PRESET_KEYS.items():
        assert _regler(at, state_key).value == C.PRESETS[name][key], (name, key)


def test_jeder_regler_der_seitenleiste_steht_in_setting_specs():
    at = _app()
    schluessel = {w.key for art in REGLER_ARTEN for w in getattr(at.sidebar, art) if w.key}
    fehlt = schluessel - set(P.SETTING_SPECS)
    assert not fehlt, f"Regler {sorted(fehlt)} fehlen in SETTING_SPECS: ihre Änderung hebt das aktive Preset nicht auf"
    assert set(P.SETTING_SPECS) <= schluessel


@pytest.mark.parametrize("state_key", list(P.SETTING_SPECS))
def test_jede_aenderung_eines_reglers_hebt_die_hervorhebung_auf(state_key):
    at = _app()
    spec = P.SETTING_SPECS[state_key]
    regler = _regler(at, state_key)
    if spec.options is not None:
        # „lose“ gehört nur zum Preset mit m = 100 und σ = 1: bei den Standardwerten der übrigen Regler passt kein Preset (bei „zeit“ oder „tabelle“ wäre das gleichnamige Preset zu Recht aktiv)
        neu = "lose" if regler.value != "lose" else "strecke"
    else:
        neu = spec.hi if regler.value != spec.hi else spec.lo
    regler.set_value(neu).run()
    assert not at.exception, [str(e) for e in at.exception]
    assert "primary" not in _typen(at).values()
    # erneutes Anklicken markiert wieder
    name = next(iter(C.PRESETS))
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert _typen(at)[name] == "primary"


def test_regler_ohne_wirkung_sind_sichtbar_gesperrt_statt_still_wirkungslos():
    """Kein stiller No-Op-Regler: Bei der Tabelle sind m, σ und r gesperrt, bei den Raten ist keiner gesperrt, bei den Losgrößen nur r."""
    at = _app()
    assert not [w for w in at.sidebar.slider if w.disabled]
    at.sidebar.button(key="preset_Die Tabelle, die kippt").click().run()
    assert {w.key for w in at.sidebar.slider if w.disabled} == {"m_slider", "sigma_slider", "r_slider"}
    at.sidebar.button(key="preset_Pro Los, pro Stück").click().run()
    assert {w.key for w in at.sidebar.slider if w.disabled} == {"r_slider"}
    at.sidebar.button(key="preset_Faktoren 1,5 und 0,6").click().run()
    assert {w.key for w in at.sidebar.slider if w.disabled} == {"sigma_slider"}
