"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Permalink-Begrenzung."""

import pytest

import rgw_constants as C
import rgw_evaluation as ev
import rgw_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_regler_bounds():
    for name, preset in C.PRESETS.items():
        assert preset["task"] in C.TASK_KEYS, name
        assert C.M_MIN <= preset["m"] <= C.M_MAX and C.SIGMA_MIN <= preset["sigma"] <= C.SIGMA_MAX and C.R_MIN <= preset["r"] <= C.R_MAX and 0 <= preset["seed"] <= C.SEED_MAX, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Gleiche Strecke (Standard)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_every_task_has_a_label_and_a_preset_or_is_reachable():
    assert set(C.TASK_LABELS) == set(C.TASK_KEYS)
    assert {p["task"] for p in C.PRESETS.values()} == set(C.TASK_KEYS)       # jede Aufgabe hat ein Preset
    assert len({p["task"] for p in C.PRESETS.values()}) == len(C.TASK_KEYS)


def test_presets_differ_in_what_they_show():
    """Nullspalten-Regel für die Presets: Kein Preset gleicht dem anderen."""
    settings = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(settings)) == len(settings)


def test_the_preset_help_numbers_match_the_hand_examples():
    """Die Zahlen im Hilfetext der Tabelle (63 % gegen 76 %) und der Faktoren (+5 %, 0,9) kommen aus der Handrechnung in tests/conftest.py."""
    a = ev.analyse(ev.Settings(task="tabelle"))
    assert f"{round(100 * a.pooled[0])} % gegen {round(100 * a.pooled[1])} %" in C.PRESET_HELP["Die Tabelle, die kippt"]
    assert "0,9" in C.PRESET_HELP["Faktoren 1,5 und 0,6"] and "+5 %" in C.PRESET_HELP["Faktoren 1,5 und 0,6"]


@pytest.mark.parametrize("state_key, raw, expected", [("m_slider", 10**6, C.M_MAX), ("m_slider", -4, C.M_MIN), ("r_slider", 99.0, C.R_MAX), ("sigma_slider", 0.0, C.SIGMA_MIN),
                                                      ("task_select", "gibt-es-nicht", C.DEFAULT_TASK), ("task_select", "lose", "lose")])
def test_permalink_values_are_clamped_or_replaced(state_key, raw, expected):
    assert P.clamp(P.SETTING_SPECS[state_key], raw) == expected
