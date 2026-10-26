"""Verdikte als reine Funktionen (DEMO-PLAYBOOK §12): Schwellen und jede Zahl im Text an handgesetzten Zeilen, nicht an Messdaten."""

import math

import pytest

import rgw_evaluation as V


def agg(mean, std=0.0):
    return {"mean": mean, "std": std, "per_seed": [mean]}


# ---------------------------------------------------------------- Zahlenformate

def test_number_formats():
    assert V.de(1.2345, 2) == "1,23" and V.de(-0.001, 2) == "0,00" and V.de(-1.5, 1) == "-1,5"
    assert V.pct(0.1234, 1) == "12,3" and V.pct(0.5, 0) == "50"
    assert V.signed_pct(0.125) == "+12,5 %" and V.signed_pct(-0.03) == "−3,0 %" and V.signed_pct(0.00001) == "0,0 %" and V.signed_pct(-0.00001) == "0,0 %"
    assert V.de_int(1234567) == "1.234.567" and V.de_int(999) == "999"
    assert V.de_g(1.1025, 3) == "1,10" and V.de_g(0.005154, 3) == "0,00515" and V.de_g(17292.58, 3) == "17.293" and V.de_g(132.0, 3) == "132"
    assert V.de_g(2.656e-05, 3) == "2,66·10⁻⁵" and V.de_g(4971923.7, 3) == "4,97·10⁶" and V.de_g(999999.0, 3) == "999.999" and V.de_g(2.2e7, 3) == "2,20·10⁷" and V.de_g(0, 3) == "0"
    assert V.family_name("poisson", 3.0) == "Poisson(λ = 3)" and V.family_name("lognormal", 0.5) == "lognormal(σ = 0,5)" and V.family_name("pareto", 2.2) == "Pareto(α = 2,2)"


# ---------------------------------------------------------------- Hauptansicht

MEANS = {"harmonisch": 66.6667, "geometrisch": 70.7107, "arithmetisch": 75.0, "quadratisch": 79.0569}


def test_verdict_rates_names_the_right_mean_and_the_misses():
    kind, text = V.verdict_rates("strecke", 66.6667, MEANS, 1.0, 75.0)
    assert kind == "warning"
    for expected in ("**66,67** km/h", "Das **harmonische** Mittel trifft sie (66,67)", "das geometrische 70,71 (+6,1 %)", "das arithmetische 75,00 (+12,5 %)", "das quadratische 79,06 (+18,6 %)", "r = 1,0 gibt 75,00 (+12,5 %)"):
        assert expected in text, expected


def test_verdict_rates_for_the_other_right_means():
    _, text = V.verdict_rates("zeit", 75.0, MEANS, -1.0, 66.6667)
    assert "Das **arithmetische** Mittel trifft sie (75,00)" in text and "das harmonische 66,67 (−11,1 %)" in text
    _, text = V.verdict_rates("wechsel", 0.948683, {"harmonisch": 0.857143, "geometrisch": 0.948683, "arithmetisch": 1.05, "quadratisch": 1.142366}, 0.0, 0.948683)
    assert "(Faktor je Periode)" in text and "Das **geometrische** Mittel trifft sie (0,949)" in text and "das arithmetische 1,050 (+10,7 %)" in text and "r = 0,0 gibt 0,949 (0,0 %)" in text


def test_verdict_rates_is_calm_when_all_means_are_close():
    kind, text = V.verdict_rates("strecke", 80.0, {"harmonisch": 80.0, "geometrisch": 80.1, "arithmetisch": 80.2, "quadratisch": 80.3}, 1.0, 80.2)
    assert kind == "info" and "kostet hier fast nichts" in text and "+0,2 %" in text and "+0,4 %" in text


def test_verdict_lots_warns_above_five_percent():
    kind, text = V.verdict_lots(10.0, 12.0, 10.5, 1.3, 40)
    assert kind == "warning"
    for expected in ("**10,0 Stück**", "40 beobachteten", "arithmetisches Mittel 12,0 (+20,0 %)", "harmonisches Mittel 10,5 (+5,0 %)", "Faktor 1,30", "2 000 Wiederholungen"):
        assert expected in text, expected
    assert V.verdict_lots(10.0, 10.4, 10.1, 1.01, 40)[0] == "info"


def test_verdict_table_for_the_flip_and_for_no_flip():
    import numpy as np
    rates = np.array([[0.9, 0.8], [0.6, 0.4]])
    kind, text = V.verdict_table(rates, np.array([0.63, 0.76]), np.array([0.75, 0.60]), 0.8, True)
    assert kind == "warning"
    for expected in ("große Standorte 90 gegen 80 %", "kleine Standorte 60 gegen 40 %", "**63 % gegen 76 %**", "Gesamtvariationsabstand 0,8", "75 % gegen 60 %"):
        assert expected in text, expected
    kind, text = V.verdict_table(rates, np.array([0.75, 0.6]), np.array([0.75, 0.6]), 0.0, False)
    assert kind == "info" and "Hier kippt nichts" in text and "75 % gegen 60 %" in text


# ---------------------------------------------------------------- Experimente

def _cost(task, sigma, right, abs_):
    return {"task": task, "sigma": sigma, "m": 20, "instances_per_seed": 2000, "right": right, "abs": {k: agg(v) for k, v in abs_.items()}}


def test_verdict_e1_lists_the_wrong_means_and_calls_the_right_one_an_identity():
    cell = _cost("strecke", 0.5, "harmonisch", {"harmonisch": 0.0, "geometrisch": 12.66, "arithmetisch": 27.08, "quadratisch": 42.76})
    kind, text = V.verdict_e1(cell)
    assert kind == "warning"
    for expected in ("Geschwindigkeit, gleiche Strecke je Fahrt", "σ = 0,50", "das harmonische Mittel ist hier das richtige", "Rechenidentität", "geometrisch 12,7 %", "arithmetisch 27,1 %", "quadratisch 42,8 %", "20 Posten", "2000 Instanzen"):
        assert expected in text, expected
    assert "harmonisch 0,0" not in text
    small = _cost("strecke", 0.1, "harmonisch", {"harmonisch": 0.0, "geometrisch": 0.48, "arithmetisch": 0.95, "quadratisch": 1.44})
    assert V.verdict_e1(small)[0] == "info"


def _growth_row(T, below, exact_below, above, median, mean_ratio, std):
    return {"T": T, "expected": 1.05 ** T, "exact_below": exact_below, "exact_above": above, "median_theory": 0.9 ** (T / 2) if T % 2 == 0 else None,
            "below": agg(below), "above": agg(above), "median": agg(median), "mean_ratio": agg(mean_ratio, std)}


def test_verdict_e2_paths_with_and_without_a_median():
    row = _growth_row(100, 0.864, 0.8644, 0.0103, 0.005154, 0.79, 0.30)
    kind, text = V.verdict_e2_paths(row, 100000)
    assert kind == "warning"
    for expected in ("Nach 100 Perioden", "**86,4 % der Pfade unter dem Start**", "Binomialsumme: 86,4 %", "Nur 1,03 %", "Erwartungswert 132", "Median ist 0,00515 (= 0,9^50)", "von 100.000 Pfaden", "0,79-Fache", "0,30"):
        assert expected in text, expected
    odd = _growth_row(11, 0.5, 0.5, 0.2, 0.7, 1.0, 0.0)
    assert "Median" not in V.verdict_e2_paths(odd, 100000)[1]


def test_verdict_e2_lognormal_shrinking_and_growing():
    shrink = {"sigma": 0.5, "T": 50, "typical": agg(0.9267), "typical_formula": 0.9266, "below": agg(0.86), "exact_below": 0.859}
    kind, text = V.verdict_e2_lognormal(shrink)
    assert kind == "warning" and "σ = 0,500" in text and "0,9267" in text and "0,9266" in text and "86,0 %" in text and "85,9 %" in text and "schrumpft" in text
    grow = {"sigma": 0.2, "T": 50, "typical": agg(1.0293), "typical_formula": 1.0292, "below": agg(0.155), "exact_below": 0.154}
    kind, text = V.verdict_e2_lognormal(grow)
    assert kind == "info" and "wächst noch" in text


def test_verdict_e2_coverage_and_live():
    row = {"family": "zweipunkt", "T": 60, "N": 10000, "reps_per_seed": 300, "within": agg(0.351), "median_ratio": agg(0.802)}
    kind, text = V.verdict_e2_coverage(row)
    assert kind == "warning" and "T = 60" in text and "10.000 Pfaden" in text and "35,1 % der Stichproben" in text and "0,80" in text and "300 Stichproben" in text
    assert V.verdict_e2_coverage({**row, "within": agg(0.999)})[0] == "info"
    live = {"family": "lognormal", "T": 50, "N": 10000, "below": 0.86, "above": 0.0123, "expected": 11.467, "median": 0.45, "mean_ratio": 0.6}
    kind, text = V.verdict_e2_live(live)
    assert kind == "info" and "10.000 Pfade" in text and "86,0 %" in text and "1,23 %" in text and "11,5" in text and "0,450" in text and "0,60" in text


def test_verdict_e3_factor_cases():
    row = {"family": "lognormal", "par": 1.0, "formula": math.e, "measured": agg(2.885, 0.31)}
    kind, text = V.verdict_e3_factor(row)
    assert kind == "warning" and "lognormal(σ = 1)" in text and "2,88-Fache" in text and "1 + CV² = 2,72" in text and "0,31" in text
    assert V.verdict_e3_factor({"family": "poisson", "par": 30.0, "formula": 1.0333, "measured": agg(1.033, 0.001)})[0] == "info"
    kind, text = V.verdict_e3_factor({"family": "pareto", "par": 2.0, "formula": math.inf, "measured": agg(5.0)})
    assert kind == "warning" and "Pareto(α = 2)" in text and "keinen Faktor" in text


def test_verdict_e3_back_with_and_without_empty_lots():
    row = {"family": "lognormal", "par": 1.0, "n": 100, "truth": 16.5, "am_bias": agg(171.5), "hm_bias": agg(1.6), "am_rmse": agg(175.1), "hm_rmse": agg(12.9)}
    kind, text = V.verdict_e3_back(row)
    assert kind == "warning" and "n = 100" in text and "+171,5 %" in text and "RMSE 175,1 %" in text and "+1,6 %" in text and "RMSE 12,9 %" in text and "nichtleeren" not in text
    poisson = {"family": "poisson", "par": 3.0, "n": 1000, "truth": 3.0, "nonempty": 3.1572, "am_bias": agg(33.3), "hm_bias": agg(5.3), "am_rmse": agg(33.4), "hm_rmse": agg(5.7), "hm_bias_nonempty": agg(0.04)}
    kind, text = V.verdict_e3_back(poisson)
    assert "n = 1.000" in text and "Leere Lose kommen in den Beobachtungen nie vor" in text and "3,16 statt 3,00" in text and "5,3 % über dem Mittel" in text
    assert V.verdict_e3_back({**row, "am_bias": agg(5.0)})[0] == "info"


def test_verdict_e3_tail_for_finite_and_infinite_means():
    rows = [{"n": 100, "median": agg(6.22)}, {"n": 10000, "median": agg(11.3)}, {"n": 1000000, "median": agg(15.2)}]
    kind, text = V.verdict_e3_tail({"alpha": 2.0, "formula": None, "rows": rows})
    assert kind == "warning" and "wächst mit n" in text and "n = 100: 6,22" in text and "n = 10.000: 11,3" in text and "n = 1.000.000: 15,2" in text and "keinen Faktor" in text
    kind, text = V.verdict_e3_tail({"alpha": 3.0, "formula": 2.0, "rows": rows})
    assert kind == "info" and "(α − 1)/(α − 2) = 2,00" in text and "nähert sich" in text


def test_verdict_e4_texts():
    kind, text = V.verdict_e4_reversal({"layers": 2, "tables_per_seed": 1000000, "share": agg(0.01675), "hits": 67011, "standard_error": 6.4e-5})
    assert kind == "info" and "2 Schichten" in text and "**1,675 % der Tabellen**" in text and "67.011 von 4.000.000" in text and "0,006 Prozentpunkte" in text
    row = {"k": 2, "tau": 1.0, "share": agg(0.0424), "tv_mean": agg(0.30), "tv_flipped_median": agg(0.49), "equal_weight_flips": 0, "standard_weight_flips": 0}
    kind, text = V.verdict_e4_weights(row)
    assert "jeder der 2 Schichten" in text and "τ = 1,00" in text and "**4,24 % der Tabellen**" in text and "0,30" in text and "Median 0,49" in text and "gleichen Gewichten 0 und mit gemeinsamen Standardgewichten 0" in text
    assert "keine Umkehr" in V.verdict_e4_weights({**row, "tau": 0.0, "share": agg(0.0), "tv_flipped_median": None})[1]


def test_verdict_e5_texts():
    cell = {"kappa_b": 0.7, "sigma_d": 1.0, "diff_mean_pp": agg(0.02), "diff_pooled_pp": agg(4.48), "b_ahead_mean": agg(0.502), "b_ahead_pooled": agg(0.813)}
    kind, text = V.verdict_e5(cell)
    assert kind == "warning" and "κ = 0,7" in text and "0,02 Prozentpunkte" in text and "**Gesamtquote 4,48**" in text and "50,2 %" in text and "81,3 %" in text
    kind, text = V.verdict_e5({**cell, "kappa_b": 0.0, "diff_pooled_pp": agg(-0.01), "b_ahead_pooled": agg(0.501)})
    assert kind == "info" and "κ = 0" in text and "50,1 %" in text and "Zufall: 50 %" in text and "-0,01" not in text.replace("0,01", "")
