"""Jede Zahl der README wird hier aus der Messdatei (data/rgw_results.json), den Mini-Instanzen oder den Formeln nachgerechnet und als Text in der README gesucht.
Die Zeichenketten entstehen mit denselben Formatfunktionen wie in der App (Dezimalkomma, Punkt als Tausendertrenner)."""

import math
from pathlib import Path

import numpy as np
import pytest

import rgw_evaluation as V
import rgw_experiments as X
import rgw_growth as G
import rgw_means as M
import rgw_results as R
import rgw_sizebias as Z

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")


def has(*texts):
    for t in texts:
        assert t in README, t


def d(x, n=1):
    return V.de(x, n)


# ---------------------------------------------------------------- Hand-Beispiele

def test_hand_examples():
    speeds = np.array([50.0, 100.0])
    assert M.harmonic(speeds) == pytest.approx(20 / 0.3) and M.arithmetic(speeds) == 75.0
    f = M.four_means([1.5, 0.6])
    assert (d(f["arithmetisch"], 2), d(f["geometrisch"], 4), d(f["harmonisch"], 4)) == ("1,05", "0,9487", "0,8571")
    lots = np.array([1.0, 1.0, 4.0])
    assert lots.mean() == 2.0 and (lots ** 2).sum() / lots.sum() == 3.0 and Z.bias_factor(lots) == 1.5 and (6 / (1 + 1 + 4 * 0.25)) == 2.0
    a = V.analyse(V.Settings(task="tabelle"))
    assert (round(100 * a.pooled[0]), round(100 * a.pooled[1]), round(100 * a.standard[0]), round(100 * a.standard[1])) == (63, 76, 75, 60) and a.reversal
    has("**66,67** km/h", "über je eine Stunde 75 (arithmetisch)", "arithmetisch 1,05, geometrisch 0,9487, harmonisch 0,8571", "Mittel pro Los 2, Mittel der Stücksicht 3 (Faktor 1,5 = 1 + CV²), harmonisch 2",
        "A 63 % gegen B 76 % insgesamt", "A 75 % gegen B 60 %", "20 km in 0,3 h")


# ---------------------------------------------------------------- E1

def test_e1_arithmetic_mean_for_equal_distance():
    cells = {s: R.cost_cell("strecke", s)["abs"] for s in (0.2, 0.5, 1.0)}
    ari = [cells[s]["arithmetisch"]["mean"] for s in (0.2, 0.5, 1.0)]
    assert [d(x) for x in ari] == ["3,9", "27,1", "162,6"]
    assert [d(cells[s]["geometrisch"]["mean"]) for s in (0.2, 0.5, 1.0)] == ["1,9", "12,7", "60,5"]
    assert [d(cells[s]["quadratisch"]["mean"]) for s in (0.2, 0.5, 1.0)] == ["5,9", "42,8", "297,2"]
    has("**3,9 % / 27,1 % / 162,6 %**", "geometrisch 1,9 / 12,7 / 60,5 %", "quadratisch 5,9 / 42,8 / 297,2 %", "um **27,1 %** über der Gesamtgeschwindigkeit, bei σ = 1,0 um **162,6 %**")


def test_e1_range_over_all_tasks():
    spans = {}
    for s in (0.2, 0.5, 1.0):
        vals = [v["mean"] for c in R.cost_cells() if c["sigma"] == s for k, v in c["abs"].items() if k != c["right"]]
        spans[s] = (min(vals), max(vals))
    assert spans[0.2][1] < 6.0 and round(spans[0.2][0], 1) == 1.9 and round(spans[0.2][1], 1) == 5.9
    assert (round(spans[0.5][0]), round(spans[0.5][1])) == (11, 43) and (round(spans[1.0][0]), round(spans[1.0][1])) == (36, 297)
    has("zwischen 1,9 und 5,9 %", "zwischen 11 und 43 %", "zwischen 36 und 297 %", "bei σ = 0,2 unter 6 %, bei 0,5 11 bis 43 %, bei 1,0 36 bis 297 %")


def test_e1_diagonal_is_an_identity_not_a_finding():
    for c in R.cost_cells():
        assert c["abs"][c["right"]]["mean"] < 1e-9
    has("Rechenidentität")


def test_e1_consumption_in_km_per_litre():
    c = R.cost_cell("verbrauch", 0.5)["abs"]["harmonisch"]["mean"]
    z = R.cost_cell("zeit", 0.5)["abs"]["harmonisch"]["mean"]
    assert (d(c), d(z)) == ("20,7", "20,8")
    has("**20,7 %**", "20,8 %")


def test_e1_growth_factors():
    f5, f10 = R.cost_cell("faktoren", 0.5)["abs"], R.cost_cell("faktoren", 1.0)["abs"]
    assert (d(f5["arithmetisch"]["mean"]), d(f10["arithmetisch"]["mean"]), d(f5["harmonisch"]["mean"]), d(f10["harmonisch"]["mean"])) == ("12,6", "61,3", "11,0", "36,2")
    has("**12,6 %** bei σ = 0,5 und **61,3 %** bei σ = 1,0; harmonisch 11,0 und 36,2 %")


# ---------------------------------------------------------------- E2

def test_e2_shares_below_start_and_above_expectation():
    rows = R.growth_rows()
    assert [d(100 * r["exact_below"]) for r in rows] == ["62,3", "74,8", "76,0", "86,4", "94,8"]
    assert max(abs(r["below"]["mean"] - r["exact_below"]) for r in rows) < 0.001           # höchstens 0,1 Prozentpunkte
    assert [d(100 * r["exact_above"], 1 if r["T"] < 200 else 2) for r in rows] == ["17,2", "13,2", "5,9", "1,0", "0,07"]
    for r in rows:
        assert r["median"]["mean"] == pytest.approx(0.9 ** (r["T"] / 2), rel=1e-12)
    has("**62,3 / 74,8 / 76,0 / 86,4 / 94,8 %**", "17,2 / 13,2 / 5,9 / **1,0** / 0,07 %", "Der Median ist 0,9^(T/2)", "höchstens 0,1 Prozentpunkte")


def test_e2_sample_mean_ratio_is_noisy():
    r100, r200 = R.growth_row(100)["mean_ratio"], R.growth_row(200)["mean_ratio"]
    assert (d(r100["mean"], 2), d(r100["std"], 2), d(min(r100["per_seed"]), 2), d(max(r100["per_seed"]), 2), d(r200["mean"], 2)) == ("0,79", "0,30", "0,49", "1,22", "0,05")
    assert r100["std"] > 0.3 * r100["mean"]                           # die Streuung zwischen den Seeds ist so groß wie ein Drittel des Mittelwerts: kein stabiler Wert
    has("**0,79 (Streuung zwischen den Seeds 0,30)**", "von 0,49 bis 1,22", "bei T = 200: 0,05", "**kein stabiler Befund**")


def test_e2_coverage_of_the_sample_mean():
    def cov(family, T, ns=(100, 1000, 10_000)):
        return " / ".join(d(100 * R.coverage_row(family, T, n)["within"]["mean"]) for n in ns)
    assert (cov("zweipunkt", 60), cov("zweipunkt", 100), cov("zweipunkt", 20)) == ("5,6 / 14,0 / 35,1", "3,5 / 3,7 / 6,5", "38,6 / 82,1 / 99,9")
    assert (cov("lognormal", 60), cov("lognormal", 100)) == ("4,7 / 9,2 / 15,8", "0,9 / 1,9 / 4,1")
    assert (d(R.coverage_row("zweipunkt", 100, 10_000)["median_ratio"]["mean"], 2), d(R.coverage_row("lognormal", 100, 10_000)["median_ratio"]["mean"], 2)) == ("0,41", "0,19")
    # in beiden Familien trifft das Mittel seltener, je länger der Horizont und je kleiner die Stichprobe
    for family in ("zweipunkt", "lognormal"):
        for n in (100, 1000, 10_000):
            vals = [R.coverage_row(family, T, n)["within"]["mean"] for T in (20, 40, 60, 100)]
            assert vals == sorted(vals, reverse=True)
    has("**5,6 / 14,0 / 35,1 %**", "T = 100: 3,5 / 3,7 / 6,5 %", "T = 20: 38,6 / 82,1 / 99,9 %", "T = 60: 4,7 / 9,2 / 15,8 %, T = 100: 0,9 / 1,9 / 4,1 %", "0,41 (Zweipunkt) und 0,19 (lognormal)",
        "in **6,5 %** der Stichproben", "35,1 % bei T = 60, N = 10⁴", "N = 10.000: 4,1 % gegen 6,5 %", "38,6 / 82,1 / 99,9 % und 3,5 / 3,7 / 6,5 %")


def test_e2_lognormal_growth_and_the_critical_sigma():
    assert d(G.critical_sigma(), 3) == "0,312"
    typ = {s: R.lognormal_row(s) for s in (0.2, 0.5)}
    assert (d(typ[0.2]["typical"]["mean"], 3), d(typ[0.5]["typical"]["mean"], 3), d(typ[0.2]["typical_formula"], 3), d(typ[0.5]["typical_formula"], 3)) == ("1,029", "0,927", "1,029", "0,927")
    assert [d(100 * R.lognormal_row(s)["below"]["mean"]) for s in (0.2, 0.35, 0.5, 0.8)] == ["15,5", "59,9", "86,0", "99,2"]
    crit = next(r for r in R.lognormal_rows() if abs(r["sigma"] - G.critical_sigma()) < 1e-9)
    assert d(crit["typical"]["mean"], 4) == "1,0001" and abs(crit["below"]["mean"] - 0.5) < 0.01
    has("**1,029 / 0,927**", "**0,312**", "15,5 / 59,9 / 86,0 / 99,2 %", "typischer Faktor 1,0001 bei σ = 0,3124 (gemessen); 0,927 bei σ = 0,5")


# ---------------------------------------------------------------- E3

def test_e3_bias_factor_confirms_the_neighbour_formula():
    m = {(f, p): R.bias_row(f, p) for f, p in X.E3_BIAS_FAMILIES}
    assert [d(m[("poisson", p)]["measured"]["mean"], 2) for p in (1.0, 3.0, 10.0, 30.0)] == ["1,99", "1,33", "1,10", "1,03"]
    assert [d(m[("poisson", p)]["formula"], 2) for p in (1.0, 3.0, 10.0, 30.0)] == ["2,00", "1,33", "1,10", "1,03"]
    assert (d(m[("geometrisch", 0.3)]["measured"]["mean"], 2), d(m[("geometrisch", 0.3)]["formula"], 2)) == ("1,69", "1,70")
    assert [d(m[("lognormal", p)]["measured"]["mean"], 2) for p in (0.5, 1.0, 1.5)] == ["1,29", "2,88", "9,19"]
    assert [d(m[("lognormal", p)]["formula"], 2) for p in (0.5, 1.0, 1.5)] == ["1,28", "2,72", "9,49"]
    assert (d(m[("pareto", 3.0)]["measured"]["mean"], 2), d(m[("pareto", 3.0)]["formula"], 2)) == ("1,37", "1,33")
    for (f, p), row in m.items():
        assert row["measured"]["mean"] == pytest.approx(row["formula"], rel=0.12)           # die Formel passt, bei σ ≥ 1 streuen die Seeds stark
    has("Poisson(1 / 3 / 10 / 30): 1,99 / 1,33 / 1,10 / 1,03 (Formel 1 + CV² = 2,00 / 1,33 / 1,10 / 1,03)", "geometrisch(0,3) 1,69 (1,70)", "1,29 / 2,88 / 9,19 (1,28 / 2,72 / 9,49", "Pareto(3) 1,37 (1,33)",
        "kein neuer Befund")


def test_e3_back_estimate_for_lognormal_lots():
    b100, b1000, b20 = R.back_row("lognormal", 1.0, 100), R.back_row("lognormal", 1.0, 1000), R.back_row("lognormal", 1.0, 20)
    assert (d(b100["am_bias"]["mean"]), d(b1000["am_bias"]["mean"]), d(b100["hm_bias"]["mean"]), d(b1000["hm_bias"]["mean"])) == ("171,5", "171,9", "1,6", "0,2")
    assert (d(b100["hm_rmse"]["mean"]), d(b1000["hm_rmse"]["mean"])) == ("12,9", "4,2")
    assert abs(b20["am_bias"]["mean"] - b1000["am_bias"]["mean"]) < 3.0                      # nicht kleiner mit mehr Daten
    assert (d(R.back_row("lognormal", 0.5, 100)["am_bias"]["mean"]), d(R.back_row("lognormal", 1.5, 1000)["am_bias"]["mean"])) == ("28,4", "848,4")
    # Verzerrung des arithmetischen Mittels = e^(σ²) − 1 in Prozent
    assert b1000["am_bias"]["mean"] == pytest.approx(100 * (math.exp(1.0) - 1), rel=0.01)
    has("**+171,5 %** (n = 100) und +171,9 % (n = 1.000)", "**+1,6 %** (n = 100), +0,2 % (n = 1.000), Streuung (RMSE) 12,9 und 4,2 %", "σ = 0,5: +28,4 %; σ = 1,5: +848,4 % (n = 1.000)", "um **171,5 %** über dem Mittel pro Los, das harmonische um **1,6 %** (n = 100)")


def test_e3_empty_lots_stay_invisible():
    row = R.back_row("poisson", 3.0, 1000)
    assert d(row["hm_bias"]["mean"]) == "5,3" and d(row["nonempty"], 3) == "3,157" and d(row["hm_bias_nonempty"]["mean"]) == "0,0"
    assert Z.nonempty_mean_poisson(3.0) / 3.0 - 1 == pytest.approx(0.0524, abs=1e-4)
    has("**+5,3 %** über dem Mittel (3,00)", "λ/(1 − e^(−λ)) = 3,157 (Abweichung 0,0 %)", "Fahrzeuge ohne Fahrgäste")


def test_e3_heavy_tail_and_geometric():
    p = R.back_row("pareto", 2.2, 1000)
    assert (d(p["am_bias"]["mean"], 0), d(p["am_bias"]["std"], 0), d(p["am_rmse"]["mean"], 0), d(p["hm_bias"]["mean"]), d(p["hm_rmse"]["mean"])) == ("212", "24", "718", "0,0", "1,6")
    g = R.back_row("geometrisch", 0.3, 1000)
    assert (d(g["am_bias"]["mean"]), d(g["hm_bias"]["mean"])) == ("70,0", "0,1")
    h9 = [R.back_row("lognormal", s, 100) for s in (0.5, 1.0, 1.5)]
    assert [d(r["am_bias"]["mean"]) for r in h9] == ["28,4", "171,5", "845,2"] and [d(r["hm_bias"]["mean"]) for r in h9] == ["0,3", "1,6", "6,6"]
    assert d(R.back_row("pareto", 2.2, 100)["hm_bias"]["mean"]) == "0,2"
    has("Arithmetisch +212 % (Streuung zwischen den Seeds 24, RMSE 718 %), harmonisch **0,0 %** mit RMSE 1,6 % (n = 1.000)", "+70,0 % gegen +0,1 %", "+28,4 / +171,5 / +845,2 % (n = 100) gegen +0,3 / +1,6 / +6,6 %", "Pareto(2,2): +0,2 %")


def test_e3_no_factor_for_alpha_at_most_two():
    def med(alpha, digits):
        return " / ".join(d(r["median"]["mean"], digits) for r in R.tail_row(alpha)["rows"])
    assert med(3.0, 2) == "1,94 / 2,00 / 2,00" and med(2.5, 2) == "2,76 / 2,93 / 2,98" and med(2.0, 1) == "6,2 / 11,3 / 15,2"
    assert R.tail_row(3.0)["formula"] == pytest.approx(2.0) and R.tail_row(2.5)["formula"] == pytest.approx(3.0) and R.tail_row(2.0)["formula"] is None and R.tail_row(1.5)["formula"] is None
    w = [r["median"]["mean"] for r in R.tail_row(2.0)["rows"]]
    assert w[0] < w[1] < w[2] and w[2] > 2 * w[0]                                             # wächst mit n
    a15 = [r["median"]["mean"] for r in R.tail_row(1.5)["rows"]]
    assert (round(a15[0]), round(a15[1] / 1e4, 1), round(a15[2] / 1e6, 1)) == (353, 3.4, 5.0)
    has("α = 3: 1,94 / 2,00 / 2,00 (Formel (α − 1)/(α − 2) = 2); α = 2,5: 2,76 / 2,93 / 2,98 (3); **α = 2: 6,2 / 11,3 / 15,2 (wächst)**; α = 1,5: 353 / 3,4·10⁴ / 5,0·10⁶", "α = 2: 6,2 / 11,3 / 15,2 bei n = 100 / 10⁴ / 10⁶")


# ---------------------------------------------------------------- E4

def test_e4_reversal_share_falls_with_the_layers():
    rows = {l: R.reversal_row(l) for l in (2, 3, 4, 6, 10)}
    assert [d(100 * rows[l]["share"]["mean"], 3) for l in (2, 3, 4)] == ["1,675", "0,570", "0,175"] and d(100 * rows[6]["share"]["mean"], 3) == "0,017"
    assert rows[10]["hits"] == 6 and rows[10]["tables_per_seed"] * 4 == 4_000_000 and rows[2]["hits"] == 67_011
    assert d(100 * rows[2]["standard_error"], 3) == "0,006"
    assert abs(rows[2]["share"]["mean"] - 1 / 60) < 2 * rows[2]["standard_error"]               # Vergleichswert 1/60 (Sekundärquelle, nicht hergeleitet)
    assert d(100 / 60, 3) == "1,667"
    has("**1,675 / 0,570 / 0,175 / 0,017 %**", "bei l = 10 in 6 von 4.000.000", "Standardfehler 0,006 Prozentpunkte bei l = 2", "1/60 = 1,667 %", "1,675 % (1/60 = 1,667 %), dann 0,570 / 0,175 / 0,017 %")


def test_e4_weight_spread_and_the_forced_zero():
    rows = [R.weight_row(2, t) for t in (0.25, 0.5, 1.0, 1.5)]
    assert [d(100 * r["share"]["mean"]) for r in rows] == ["0,7", "2,0", "4,2", "5,6"]
    assert [d(r["tv_mean"]["mean"], 2) for r in rows] == ["0,10", "0,18", "0,30", "0,37"] and [d(r["tv_flipped_median"]["mean"], 2) for r in rows] == ["0,17", "0,31", "0,49", "0,60"]
    assert d(100 * max(r["share"]["mean"] for r in R.weight_rows())) == "5,6"
    tv = [r["tv_flipped_median"]["mean"] for r in R.weight_rows() if r["tv_flipped_median"]]
    assert (d(min(tv), 2), d(max(tv), 2)) == ("0,17", "0,71")
    assert all(r["equal_weight_flips"] == 0 and r["standard_weight_flips"] == 0 for r in R.weight_rows())
    assert R.weight_row(2, 0.0)["share"]["mean"] == 0.0 and R.weight_row(2, 0.0)["tv_flipped_median"] is None
    has("**0,7 / 2,0 / 4,2 / 5,6 %**", "0,10 / 0,18 / 0,30 / 0,37", "0,17 / 0,31 / 0,49 / 0,60", "Höchstens 5,6 % im ganzen Raster", "0 Umkehrungen", "erzwungen", "höchstens 5,6 % im Raster", "0,17 bis 0,71")


# ---------------------------------------------------------------- E5

def test_e5_pooled_rate_against_mean_of_rates():
    c = R.coupling_cell(0.7, 1.0)
    assert (d(c["diff_mean_pp"]["mean"], 2), d(c["diff_pooled_pp"]["mean"], 2), d(100 * c["b_ahead_mean"]["mean"]), d(100 * c["b_ahead_pooled"]["mean"])) == ("0,02", "4,48", "50,2", "81,3")
    assert (d(100 * R.coupling_cell(0.3, 1.0)["b_ahead_pooled"]["mean"]), d(100 * R.coupling_cell(-0.7, 1.0)["b_ahead_pooled"]["mean"]), d(100 * R.coupling_cell(0.0, 1.0)["b_ahead_pooled"]["mean"])) == ("64,6", "19,0", "50,1")
    assert (d(100 * R.coupling_cell(0.7, 0.5)["b_ahead_pooled"]["mean"]), d(100 * R.coupling_cell(0.7, 1.5)["b_ahead_pooled"]["mean"])) == ("73,0", "83,6")
    for cell in R.coupling_cells():
        assert abs(cell["diff_mean_pp"]["mean"]) < 0.1 and abs(cell["b_ahead_mean"]["mean"] - 0.5) < 0.01          # ungewichtet: nur Zufall
    has("ungewichtet **0,02** Prozentpunkte, bei der Gesamtquote **4,48**", "in 50,2 %, nach Gesamtquote in **81,3 %**", "κ_B = 0,3: 64,6 %; κ_B = −0,7: 19,0 %; ohne Kopplung 50,1 %", "73,0 / 83,6 %", "liegt Gruppe B in **81,3 %**", "in 50,2 %.", "81,3 % gegen 50,2 %")


# ---------------------------------------------------------------- Hypothesen und Angaben, die aus Konstanten kommen

def test_numbers_from_constants_and_grids():
    assert X.E1_M == 20 and X.E1_INSTANCES == 2000 and X.E2_N == 100_000 and X.E2_COVER_REPS == 300 and len(X.SEEDS) == 4 and X.E5_LOCATIONS == 10 and X.E5_RATE == 0.8 and X.E5_SPREAD == 0.08
    assert V.de(X.E3_REPS, 0) == "2000" and G.MEAN_FACTOR == pytest.approx(0.5 * 1.5 + 0.5 * 0.6)
    has("20 Posten, 2.000 Instanzen je Seed", "300 Stichproben je Seed", "100.000 Pfaden", "10 Standorte je Gruppe", "Servicegrad 0,80 ± 0,08", "vier feste Zufalls-Seeds je Zelle", "höchstens 5 Millionen", "höchstens 2 Millionen", "höchstens 20 Millionen")


def test_the_order_and_closed_form_checked_in_the_oracle_tests_are_named_as_such():
    has("in 200 Zufallsmengen (Orakel) ohne Verletzung", "innerhalb 1 % der Formel", "Die 80.000 Mengen der Vormessung wurden nicht wiederholt")


def test_model_constants_named_in_the_readme():
    import rgw_tasks as T
    assert T.SPEED_SCALE == 80.0 and T.CONSUMPTION_SCALE == 8.0 and T.UP == 1.5 and T.DOWN == 0.6 and V.C.R_CURVE[0] == -5.0 and V.C.R_CURVE[-1] == 5.0
    assert M.power_mean([1.0, 1000.0], 300.0) == pytest.approx(1000.0 * 0.5 ** (1 / 300))            # numerisch stabil bis r = ±300 (Test test_huge_exponents_do_not_overflow)
    has("Median 80 km/h bzw. 8 l/100 km bzw. 1", "numerisch stabil bis r = ±300", "√0,9 = 0,9487", "Erwartungswert 1,05 je Periode")
