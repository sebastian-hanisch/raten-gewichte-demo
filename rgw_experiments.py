"""Messreihen der Experimente E1 bis E5: reine Funktionen über feste Zufalls-Seeds, die sowohl die vorgerechneten Tabellen (tools/make_results.py → data/rgw_results.json)
als auch die Live-Knöpfe der App erzeugen. Jede Zelle nutzt vier feste Seeds (SEEDS); angegeben wird der Mittelwert mit der Standardabweichung über die Seeds.

Zufall: `Draws(cell_seed(...))` aus rgw_scenario.py; ein Seed gehört zu einer Zelle (Seed, Marken), nie zu einer Instanz allein. Alle verglichenen Verfahren sehen dieselben Daten
(gepaarter Vergleich: vier Mittel derselben Instanzen, arithmetisches und harmonisches Mittel derselben Beobachtungen, ungewichtete und Gesamtquote derselben Standorte).

Die Raster und Seeds stehen hier und nicht in rgw_constants.py, damit das Ändern eines Hilfetextes die Prüfsumme der Erzeuger (`source_hash`) nicht verändert."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

import rgw_growth as G
import rgw_means as M
import rgw_mix as X
import rgw_scenario as S
import rgw_sizebias as Z
import rgw_tasks as T

ROOT = Path(__file__).resolve().parent
SOURCES = ("rgw_rng.py", "rgw_scenario.py", "rgw_means.py", "rgw_tasks.py", "rgw_growth.py", "rgw_sizebias.py", "rgw_mix.py", "rgw_experiments.py", "tools/make_results.py")

SEEDS = (8301, 8302, 8303, 8304)                       # vier feste Zufalls-Seeds je Zelle

E1_TASKS = ("strecke", "zeit", "verbrauch", "faktoren")
E1_SIGMAS = (0.1, 0.2, 0.35, 0.5, 0.75, 1.0)           # E1: Streuung der Logarithmen der Einzelraten
E1_M = 20                                              # E1: Posten je Instanz
E1_INSTANCES = 2000                                    # E1: Instanzen je Seed
E1_R_GRID = tuple(r / 2 for r in range(-8, 9))         # E1: Exponent r von −4 bis 4 in Schritten von 0,5
E1_CURVE_SIGMAS = (0.2, 0.5, 1.0)

E2_HORIZONS = (10, 20, 50, 100, 200)                   # E2: Perioden
E2_N = 100_000                                         # E2: Pfade je Seed
E2_LOG_SIGMAS = (0.1, 0.2, 0.3, 0.35, 0.5, 0.8)        # E2: Streuung lognormaler Faktoren bei Erwartungswert 1,05 je Periode
E2_LOG_T = 50
E2_LOG_N = 20_000
E2_COVER_T = (20, 40, 60, 100)                         # E2: Perioden der Stichprobenmittel
E2_COVER_N = (100, 1000, 10_000)                       # E2: Pfade je Stichprobe
E2_COVER_REPS = 300                                    # E2: Stichproben je Seed
E2_COVER_SIGMA = 0.5                                   # E2: Streuung der lognormalen Familie in der Abdeckungsreihe
E2_BAND = 0.2                                          # E2: „getroffen“ heißt Stichprobenmittel innerhalb ± 20 % des Erwartungswerts

E3_BIAS_FAMILIES = (("poisson", 1.0), ("poisson", 3.0), ("poisson", 10.0), ("poisson", 30.0), ("geometrisch", 0.3), ("lognormal", 0.5), ("lognormal", 1.0), ("lognormal", 1.5), ("pareto", 3.0))
E3_BACK_FAMILIES = (("lognormal", 0.5), ("lognormal", 1.0), ("lognormal", 1.5), ("poisson", 3.0), ("geometrisch", 0.3), ("pareto", 3.0), ("pareto", 2.2))
E3_LOTS = 5000                                         # E3: Lose der endlichen Grundgesamtheit
E3_PIECES = 400_000                                    # E3: gezogene Stücke
E3_NS = (20, 100, 1000)                                # E3: Beobachtungen je Rückrechnung
E3_REPS = 2000                                         # E3: Wiederholungen je Seed
E3_ALPHAS = (1.5, 2.0, 2.5, 3.0)                       # E3: Tail-Index der Pareto-Losgrößen
E3_TAIL_RUNS = ((100, 100), (10_000, 100), (1_000_000, 10))      # E3: (Beobachtungen, Wiederholungen je Seed)

E4_LAYERS = (2, 3, 4, 6, 10)                           # E4: Schichten der zufälligen Tabellen
E4_TABLES = 1_000_000                                  # E4: Tabellen je Seed
E4_CHUNK = 100_000
E4_WEIGHT_K = (2, 3, 5)                                # E4: Schichten bei vorgegebener Dominanz
E4_TAUS = (0.0, 0.25, 0.5, 1.0, 1.5)                   # E4: Streuung der Gewichte (lognormal)
E4_WEIGHT_TABLES = 200_000                             # E4: Tabellen je Seed und Zelle

E5_KAPPAS = (-0.7, 0.0, 0.3, 0.7)                      # E5: Kopplung von Nachfrage und Quote in Gruppe B (A ist unabhängig: κ = 0)
E5_SIGMA_DS = (0.5, 1.0, 1.5)                          # E5: Streuung der Nachfrage im Logarithmus
E5_COMPARISONS = 20_000                                # E5: Gruppenvergleiche je Seed
E5_LOCATIONS = 10                                      # E5: Standorte je Gruppe
E5_RATE, E5_SPREAD = 0.80, 0.08                        # E5: wahre Standortquote 0,80 mit Streuung 0,08 (in beiden Gruppen gleich)


def source_hash() -> str:
    """Prüfsumme der Quelldateien der Messreihen (Zeilenenden normalisiert): stimmt sie nicht mit der in der JSON-Datei überein, ist die Datei veraltet."""
    h = hashlib.sha256()
    for name in SOURCES:
        text = (ROOT / name).read_text(encoding="utf-8").replace("\r\n", "\n")
        h.update(name.encode("utf-8") + b"\0" + text.encode("utf-8") + b"\0")
    return h.hexdigest()[:16]


def agg(values) -> dict:
    """Mittel und Standardabweichung über die Seeds einer Zelle (die Einzelwerte bleiben erhalten)."""
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std()), "per_seed": [float(x) for x in v]}


# ================================================================== E1: Kosten des falschen Mittels

def cost_cell(task: str, sigma: float, m: int = E1_M, instances: int = E1_INSTANCES, seeds=SEEDS) -> dict:
    """Mittlere Abweichung |Mittel / wahre Gesamtgröße − 1| in Prozent und mittlere vorzeichenbehaftete Abweichung der vier Mittel; die vier Mittel sehen dieselben Instanzen."""
    absolute = {name: [] for name in M.MEAN_R}
    signed = {name: [] for name in M.MEAN_R}
    for seed in seeds:
        x, truth = T.make_batch(task, S.Draws(S.cell_seed(seed, "kosten", task, sigma, m)), instances, m, sigma)
        for name, r in M.MEAN_R.items():
            err = M.relative_error(M.power_mean(x, r), truth)
            absolute[name].append(100.0 * np.abs(err).mean())
            signed[name].append(100.0 * err.mean())
    return {"task": task, "sigma": sigma, "m": m, "instances_per_seed": instances, "right": T.RATE_TASKS[task].right_name,
            "abs": {k: agg(v) for k, v in absolute.items()}, "signed": {k: agg(v) for k, v in signed.items()}}


def cost_curve(task: str, sigma: float, rs=E1_R_GRID, m: int = E1_M, instances: int = E1_INSTANCES, seeds=SEEDS) -> dict:
    """Mittlere Abweichung des Potenzmittels M_r von der wahren Gesamtgröße in Prozent über den Exponenten r (dieselben Instanzen für alle r)."""
    cost = [[] for _ in rs]
    for seed in seeds:
        x, truth = T.make_batch(task, S.Draws(S.cell_seed(seed, "kurve", task, sigma, m)), instances, m, sigma)
        for i, r in enumerate(rs):
            cost[i].append(100.0 * np.abs(M.relative_error(M.power_mean(x, r), truth)).mean())
    return {"task": task, "sigma": sigma, "rs": list(rs), "cost": [agg(c) for c in cost]}


def run_m1() -> dict:
    return {"cells": [cost_cell(t, s) for t in E1_TASKS for s in E1_SIGMAS],
            "curves": [cost_curve(t, s) for t in E1_TASKS for s in E1_CURVE_SIGMAS]}


# ================================================================== E2: Pfade und Stichprobenmittel

def growth_cell(horizons=E2_HORIZONS, N: int = E2_N, seeds=SEEDS) -> dict:
    """Zweipunkt-Faktoren 1,5 / 0,6: Anteil unter dem Start, Anteil über dem Erwartungswert, Median des Endwerts (gerades T) und Stichprobenmittel durch Erwartungswert,
    dazu die exakten Binomialwerte. Die Horizonte sind Anfänge desselben Pfades."""
    cols = {t: {"below": [], "above": [], "median": [], "mean_ratio": []} for t in horizons}
    for seed in seeds:
        ups = G.ups_prefix(S.Draws(S.cell_seed(seed, "pfade", N)), N, horizons)
        for t in horizons:
            v = G.path_value(ups[t], t)
            cols[t]["below"].append(G.share_below_start(v))
            cols[t]["above"].append(G.share_above(v, G.expected_value(t)))
            cols[t]["median"].append(float(np.median(v)))
            cols[t]["mean_ratio"].append(G.sample_mean_ratio(v, t))
    rows = []
    for t in horizons:
        rows.append({"T": t, "expected": G.expected_value(t), "exact_below": G.exact_share_below_start(t), "exact_above": G.exact_share_above_expectation(t),
                     "median_theory": G.typical_value(t) if t % 2 == 0 else None, **{k: agg(v) for k, v in cols[t].items()}})
    return {"N": N, "rows": rows}


def lognormal_growth_cell(sigma: float, T_: int = E2_LOG_T, N: int = E2_LOG_N, seeds=SEEDS) -> dict:
    """Lognormale Faktoren mit Erwartungswert 1,05 je Periode: Anteil der Pfade unter dem Start, typischer Faktor je Periode und die Formeln."""
    below, typical = [], []
    for seed in seeds:
        v = G.lognormal_paths(S.Draws(S.cell_seed(seed, "lognormal", sigma, T_)), N, T_, sigma)
        below.append(G.share_below_start(v))
        typical.append(G.typical_factor(v, T_))
    return {"sigma": sigma, "T": T_, "N": N, "below": agg(below), "typical": agg(typical), "typical_formula": G.typical_factor_formula(sigma),
            "exact_below": G.lognormal_share_below_start(T_, sigma)}


def coverage_cell(family: str, T_: int, N: int, reps: int = E2_COVER_REPS, seeds=SEEDS) -> dict:
    """Wie oft trifft das Mittel von N Pfaden den Erwartungswert (innerhalb ± 20 %)? Anteil der Stichproben, in denen es trifft, und Median des Verhältnisses.
    Zweipunkt: eine Binomialziehung je Pfad; lognormal: eine Normalverteilte je Pfad (beide sind gegen Einzelschritte geprüft)."""
    within, median_ratio = [], []
    for seed in seeds:
        draws = S.Draws(S.cell_seed(seed, "abdeckung", family, T_, N))
        if family == "zweipunkt":
            v = G.path_value(G.ups_binomial(draws, reps * N, T_), T_)
        elif family == "lognormal":
            v = G.lognormal_paths(draws, reps * N, T_, E2_COVER_SIGMA)
        else:
            raise ValueError(f"unbekannte Familie: {family}")
        ratio = v.reshape(reps, N).mean(axis=1) / G.MEAN_FACTOR ** T_
        within.append(float((np.abs(ratio - 1.0) <= E2_BAND).mean()))
        median_ratio.append(float(np.median(ratio)))
    return {"family": family, "T": T_, "N": N, "reps_per_seed": reps, "within": agg(within), "median_ratio": agg(median_ratio)}


def run_m2() -> dict:
    return {"growth": growth_cell(),
            "lognormal": [lognormal_growth_cell(s) for s in E2_LOG_SIGMAS] + [lognormal_growth_cell(G.critical_sigma())],
            "coverage": [coverage_cell(f, t, n) for f in ("zweipunkt", "lognormal") for t in E2_COVER_T for n in E2_COVER_N]}


def growth_live(family: str, T_: int, N: int, sigma: float, seed: int, trajectories: int = 30) -> dict:
    """Live-Messung der App für eine Zelle (ein Seed): Anteil unter dem Start, Anteil über dem Erwartungswert, Median, Stichprobenmittel durch Erwartungswert
    und einige Pfade zum Zeichnen. Zweipunkt über Einzelschritte (N·T Gleichverteilte), lognormal in geschlossener Form."""
    base = S.cell_seed(seed, "live", family, T_, N, sigma)
    if family == "zweipunkt":
        v = G.path_value(G.ups_by_steps(S.Draws(base), N, T_), T_)
        paths = G.two_point_trajectories(S.Draws(base + 1), trajectories, T_)
    elif family == "lognormal":
        v = G.lognormal_paths(S.Draws(base), N, T_, sigma)
        paths = G.lognormal_trajectories(S.Draws(base + 1), trajectories, T_, sigma)
    else:
        raise ValueError(f"unbekannte Familie: {family}")
    return {"family": family, "T": T_, "N": N, "sigma": sigma, "expected": G.expected_value(T_), "below": G.share_below_start(v),
            "above": G.share_above(v, G.MEAN_FACTOR ** T_), "median": float(np.median(v)), "mean_ratio": G.sample_mean_ratio(v, T_), "paths": paths}


# ================================================================== E3: Größenverzerrung und Rückrechnung

def bias_cell(family: str, par: float, lots: int = E3_LOTS, pieces: int = E3_PIECES, seeds=SEEDS) -> dict:
    """Verhältnis Mittel der Stücksicht / Mittel pro Los: gemessen an gezogenen Stücken, aus den Losen derselben Grundgesamtheit (1 + CV² der gezogenen Lose) und aus der Formel der Familie."""
    measured, population = [], []
    for seed in seeds:
        draws = S.Draws(S.cell_seed(seed, "faktor", family, par))
        sizes = Z.lot_sizes(family, par, draws, lots)
        obs = Z.pieces_from_population(sizes, pieces, draws)
        measured.append(float(obs.mean() / sizes.mean()))
        population.append(Z.bias_factor(sizes))
    return {"family": family, "par": par, "lots": lots, "pieces": pieces, "formula": Z.bias_factor_formula(family, par), "measured": agg(measured), "population": agg(population)}


def back_cell(family: str, par: float, n: int, reps: int = E3_REPS, seeds=SEEDS) -> dict:
    """Rückrechnung auf das Mittel pro Los aus n beobachteten Größen (Stücksicht, aus der Formel gezogen): Verzerrung und Wurzel des mittleren quadratischen Fehlers in Prozent des wahren Mittels,
    für das arithmetische und das harmonische Mittel derselben Beobachtungen. Bei Poisson zusätzlich die Verzerrung des harmonischen Mittels gegen das Mittel der nichtleeren Lose."""
    truth = Z.family_mean(family, par)
    cols = {k: [] for k in ("am_bias", "hm_bias", "am_rmse", "hm_rmse", "hm_bias_nonempty")}
    nonempty = Z.nonempty_mean_poisson(par) if family == "poisson" else None
    for seed in seeds:
        y = Z.size_biased_draw(family, par, S.Draws(S.cell_seed(seed, "rueck", family, par, n)), (reps, n))
        am, hm = Z.arithmetic_back(y), Z.harmonic_back(y)
        cols["am_bias"].append(100.0 * (am.mean() / truth - 1.0))
        cols["hm_bias"].append(100.0 * (hm.mean() / truth - 1.0))
        cols["am_rmse"].append(100.0 * math.sqrt(float(((am / truth - 1.0) ** 2).mean())))
        cols["hm_rmse"].append(100.0 * math.sqrt(float(((hm / truth - 1.0) ** 2).mean())))
        if nonempty is not None:
            cols["hm_bias_nonempty"].append(100.0 * (hm.mean() / nonempty - 1.0))
    out = {"family": family, "par": par, "n": n, "reps_per_seed": reps, "truth": truth, "nonempty": nonempty}
    out.update({k: agg(v) for k, v in cols.items() if v})
    return out


def tail_cell(alpha: float, runs=E3_TAIL_RUNS, seeds=SEEDS) -> dict:
    """Pareto(α) als Losgröße: Median (über die Wiederholungen) des arithmetischen Mittels aus n beobachteten Größen der Stücksicht, je Beobachtungszahl n. Die Formel (α − 1)/(α − 2) gilt nur für α > 2."""
    formula = Z.piece_mean_formula("pareto", alpha)
    rows = []
    for n, reps in runs:
        medians = []
        for seed in seeds:
            draws = S.Draws(S.cell_seed(seed, "tail", alpha, n))
            means = [float(Z.size_biased_draw("pareto", alpha, draws, n).mean()) for _ in range(reps)]
            medians.append(float(np.median(means)))
        rows.append({"n": n, "reps_per_seed": reps, "median": agg(medians)})
    return {"alpha": alpha, "formula": None if math.isinf(formula) else formula, "rows": rows}


def run_m3() -> dict:
    return {"bias": [bias_cell(f, p) for f, p in E3_BIAS_FAMILIES],
            "back": [back_cell(f, p, n) for f, p in E3_BACK_FAMILIES for n in E3_NS],
            "tail": [tail_cell(a) for a in E3_ALPHAS]}


def back_live(family: str, par: float, n: int, reps: int, seed: int) -> dict:
    """Live-Messung der App für eine Zelle (ein Seed)."""
    return back_cell(family, par, n, reps, seeds=(seed,))


# ================================================================== E4: Wie selten ist Simpson?

def reversal_cell(layers: int, tables: int = E4_TABLES, seeds=SEEDS) -> dict:
    """Anteil der zufälligen 2×2×l-Tabellen (Zellwahrscheinlichkeiten gleichverteilt auf dem Simplex) mit Simpson-Umkehr."""
    shares, counts = [], []
    for seed in seeds:
        draws = S.Draws(S.cell_seed(seed, "tabellen", layers))
        hits = 0
        for start in range(0, tables, E4_CHUNK):
            hits += int(X.reversal_mask(X.random_tables(draws, min(E4_CHUNK, tables - start), layers)).sum())
        shares.append(hits / tables)
        counts.append(hits)
    total = tables * len(seeds)
    pooled = sum(counts) / total
    return {"layers": layers, "tables_per_seed": tables, "share": agg(shares), "hits": int(sum(counts)), "standard_error": math.sqrt(pooled * (1.0 - pooled) / total)}


def weight_cell(k: int, tau: float, tables: int = E4_WEIGHT_TABLES, seeds=SEEDS) -> dict:
    """Gruppe A hat in jeder der k Schichten die höhere Quote. Anteil der Tabellen, in denen die Gesamtquote von B trotzdem höher ist, bei lognormalen Gewichten mit Streuung τ,
    mittlerer Abstand der Gewichte (Gesamtvariationsabstand), sein Median unter den Umkehrungen und die Kontrollen: gleiche Gewichte und gemeinsame Standardgewichte (erzwungen null)."""
    share, tv_mean, tv_flip, equal_flips, standard_flips = [], [], [], 0, 0
    for seed in seeds:
        rates, w = X.dominated_rates_and_weights(S.Draws(S.cell_seed(seed, "gewichte", k, tau)), tables, k, tau)
        total = X.weighted_total(rates, w)
        flipped = total[:, 1] > total[:, 0]
        tv = X.total_variation(w)
        share.append(float(flipped.mean()))
        tv_mean.append(float(tv.mean()))
        if flipped.any():
            tv_flip.append(float(np.median(tv[flipped])))
        same = X.weighted_total(rates, w[..., 0:1])
        equal_flips += int((same[:, 1] > same[:, 0]).sum())
        std = X.standardised_totals(rates, w)
        standard_flips += int((std[:, 1] > std[:, 0]).sum())
    return {"k": k, "tau": tau, "tables_per_seed": tables, "share": agg(share), "tv_mean": agg(tv_mean), "tv_flipped_median": agg(tv_flip) if tv_flip else None,
            "equal_weight_flips": equal_flips, "standard_weight_flips": standard_flips}


def run_m4() -> dict:
    return {"reversal": [reversal_cell(l) for l in E4_LAYERS], "weights": [weight_cell(k, tau) for k in E4_WEIGHT_K for tau in E4_TAUS]}


def reversal_live(layers: int, tables: int, seed: int) -> dict:
    """Live-Messung der App (ein Seed)."""
    return reversal_cell(layers, tables, seeds=(seed,))


# ================================================================== E5: Gesamtquote gegen Mittel der Quoten

def coupling_cell(kappa_b: float, sigma_d: float, comparisons: int = E5_COMPARISONS, m: int = E5_LOCATIONS, seeds=SEEDS) -> dict:
    """Zwei Gruppen mit m Standorten und derselben wahren Standortquote (0,80 mit Streuung 0,08). Gruppe A hat unabhängige Nachfrage (κ = 0), Gruppe B die Kopplung κ_B von Nachfrage und Quote.
    Gemessen: Unterschied B − A in Prozentpunkten und Anteil der Vergleiche mit B > A, für das ungewichtete Mittel der Quoten und für die Gesamtquote (dieselben Standorte)."""
    cols = {k: [] for k in ("diff_mean_pp", "diff_pooled_pp", "b_ahead_mean", "b_ahead_pooled")}
    for seed in seeds:
        draws = S.Draws(S.cell_seed(seed, "kopplung", kappa_b, sigma_d))
        mean_rate, pooled = [], []
        for kappa in (0.0, kappa_b):
            z = draws.normal((comparisons, m))
            e = draws.normal((comparisons, m))
            q = X.service_levels(z, E5_RATE, E5_SPREAD)
            mean_rate.append(X.mean_rate(q))
            pooled.append(X.pooled_rate(q, X.coupled_demand(z, e, kappa, sigma_d)))
        d_mean, d_pooled = mean_rate[1] - mean_rate[0], pooled[1] - pooled[0]
        cols["diff_mean_pp"].append(100.0 * float(d_mean.mean()))
        cols["diff_pooled_pp"].append(100.0 * float(d_pooled.mean()))
        cols["b_ahead_mean"].append(float((d_mean > 0).mean()))
        cols["b_ahead_pooled"].append(float((d_pooled > 0).mean()))
    return {"kappa_b": kappa_b, "sigma_d": sigma_d, "comparisons_per_seed": comparisons, "locations": m, **{k: agg(v) for k, v in cols.items()}}


def run_m5() -> dict:
    return {"cells": [coupling_cell(k, s) for k in E5_KAPPAS for s in E5_SIGMA_DS]}


PARTS = {"m1": run_m1, "m2": run_m2, "m3": run_m3, "m4": run_m4, "m5": run_m5}
