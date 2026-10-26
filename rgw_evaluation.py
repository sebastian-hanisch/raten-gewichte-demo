"""Kennzahlen und Verdikte. Jedes Verdikt der App ist eine reine Funktion `verdict_*(...) -> (Art, Text)` (DEMO-PLAYBOOK §12, tests/test_verdicts.py):
Sie bekommen Zahlen, keine Seeds, und jede genannte Zahl steht im Text aus ihren Argumenten."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import rgw_constants as C
import rgw_means as M
import rgw_mix as X
import rgw_scenario as S
import rgw_sizebias as Z
import rgw_tasks as T

# Die Tabelle, die kippt (Handrechnung in tests/conftest.py): [Schicht groß, klein][Gruppe A, B][Erfolg, Misserfolg]
HAND_TABLE = np.array([[[9, 1], [72, 18]], [[54, 36], [4, 6]]], dtype=float)
LAYER_NAMES = ("große Standorte", "kleine Standorte")
MEAN_ADJECTIVE = {"harmonisch": "harmonische", "geometrisch": "geometrische", "arithmetisch": "arithmetische", "quadratisch": "quadratische"}
FAMILY_LABELS = {"zweipunkt": "Zweipunkt (1,5 / 0,6)", "lognormal": "lognormal"}
SHORT_FAMILY = {"poisson": "Poisson", "geometrisch": "geometrisch", "lognormal": "lognormal", "pareto": "Pareto"}


def family_name(family: str, par: float) -> str:
    """Familie mit Parameter: „Poisson(λ = 3)“, „lognormal(σ = 1)“."""
    return f"{SHORT_FAMILY[family]}({Z.FAMILIES[family][1]} = {de(par, 1).replace(',0', '')})"


@dataclass(frozen=True, kw_only=True)
class Settings:
    """Alle Regler der Seitenleiste (nur mit Schlüsselwörtern anlegen: ein eingefügtes Feld verschiebt sonst still Werte)."""
    task: str = C.DEFAULT_TASK
    m: int = C.DEFAULT_M
    sigma: float = C.DEFAULT_SIGMA
    r: float = C.DEFAULT_R
    seed: int = C.DEFAULT_SEED


# ---------------------------------------------------------------- Darstellung deutscher Zahlen

def de(x: float, digits: int = 2) -> str:
    """Zahl mit Dezimalkomma (eine negative Zahl, die auf null rundet, ohne Vorzeichen)."""
    text = f"{x:.{digits}f}"
    if text.startswith("-") and float(text) == 0.0:
        text = text[1:]
    return text.replace(".", ",")


def pct(x: float, digits: int = 1) -> str:
    """Anteil (0 bis 1) als Prozentzahl mit Dezimalkomma, ohne Prozentzeichen."""
    return de(100.0 * x, digits)


def signed_pct(x: float, digits: int = 1) -> str:
    """Abweichung (Anteil, 0,125 = 12,5 %) mit Vorzeichen: „+12,5 %“ oder „−3,0 %“; ein Wert, der auf null rundet, ohne Vorzeichen."""
    text = pct(abs(x), digits)
    if float(text.replace(",", ".")) == 0.0:
        return f"{text} %"
    return f"{'+' if x > 0 else '−'}{text} %"


def de_int(n: int) -> str:
    """Ganze Zahl mit Punkt als Tausendertrenner: 100.000."""
    return f"{int(n):,}".replace(",", ".")


def de_g(x: float, digits: int = 3) -> str:
    """Zahl mit `digits` gültigen Stellen und Dezimalkomma; sehr große und kleine Zahlen als „a,b·10^e“ mit hochgestelltem Exponenten."""
    if x == 0:
        return "0"
    exp = int(math.floor(math.log10(abs(x))))
    if -3 <= exp < 6:
        return de(x, max(0, digits - 1 - exp)) if exp < digits - 1 else f"{x:,.0f}".replace(",", ".")
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return f"{de(x / 10 ** exp, digits - 1)}·10{str(exp).translate(sup)}"


# ---------------------------------------------------------------- Hauptansicht

@dataclass(frozen=True)
class RatesAnalysis:
    settings: Settings
    task: T.RateTask
    x: np.ndarray                  # Einzelraten (m,)
    num: np.ndarray | None         # Zähler (Strecke, Liter) bzw. None bei Faktoren
    den: np.ndarray | None         # Nenner (Zeit, Hundert-km-Einheiten)
    truth: float                   # wahre Gesamtgröße aus den Rohgrößen
    means: dict                    # die vier Mittel
    selected: float                # M_r für das gewählte r
    curve: np.ndarray              # M_r auf R_CURVE


@dataclass(frozen=True)
class LotsAnalysis:
    settings: Settings
    sizes: np.ndarray              # Losgrößen der Grundgesamtheit
    observed: np.ndarray           # beobachtete Größen (ein Stück ziehen, nach der Größe seines Loses fragen)
    truth: float                   # Mittel pro Los
    factor: float                  # 1 + CV² der Grundgesamtheit
    arithmetic: float
    harmonic: float


@dataclass(frozen=True)
class TableAnalysis:
    table: np.ndarray              # (2 Schichten, 2 Gruppen, 2 Ausgänge)
    rates: np.ndarray              # (2, 2) Quote je Schicht und Gruppe
    weights: np.ndarray            # (2, 2) Gewicht der Schicht je Gruppe
    pooled: np.ndarray             # (2,) Gesamtquote je Gruppe
    standard: np.ndarray           # (2,) Quote mit gemeinsamen Standardgewichten
    tv: float
    reversal: bool


def analyse_rates(s: Settings) -> RatesAnalysis:
    # Der Seed hängt nicht von σ ab: Die Normalverteilten bleiben beim Verstellen von σ dieselben, nur die Streuung ändert sich.
    x, truth = T.make_batch(s.task, S.Draws(S.cell_seed(s.seed, "haupt", s.task, s.m)), 1, s.m, s.sigma)
    x = x[0]
    num = den = None
    if s.task != "wechsel" and s.task != "faktoren":
        num, den = T.raw_quantities(s.task, x)
    return RatesAnalysis(s, T.RATE_TASKS[s.task], x, num, den, float(truth[0]), M.four_means(x), float(M.power_mean(x, s.r)), np.array([M.power_mean(x, r) for r in C.R_CURVE]))


def analyse_lots(s: Settings) -> LotsAnalysis:
    sizes = Z.lot_sizes(C.LOTS_FAMILY, s.sigma, S.Draws(S.cell_seed(s.seed, "lose")), C.LOTS_POPULATION)
    observed = Z.pieces_from_population(sizes, s.m, S.Draws(S.cell_seed(s.seed, "stuecke", s.m)))
    return LotsAnalysis(s, sizes, observed, float(sizes.mean()), Z.bias_factor(sizes), float(Z.arithmetic_back(observed)), float(Z.harmonic_back(observed)))


def analyse_table() -> TableAnalysis:
    t = HAND_TABLE
    rates, weights = X.stratum_rates(t), X.group_weights(t)
    return TableAnalysis(t, rates, weights, X.pooled_rates(t), X.standardised_totals(rates, weights), float(X.total_variation(weights)), bool(X.reversal_mask(t)))


def analyse(s: Settings):
    """Die Hauptansicht der gewählten Aufgabe: Raten (Mittel und r-Kurve), Lose (Stücksicht und Rückrechnung) oder die Tabelle."""
    if s.task in T.RATE_TASKS:
        return analyse_rates(s)
    if s.task == "lose":
        return analyse_lots(s)
    if s.task == "tabelle":
        return analyse_table()
    raise ValueError(f"unbekannte Aufgabe: {s.task}")


def mean_rows(a: RatesAnalysis) -> list[dict]:
    """Zeilen der Mittel-Tabelle: Name, r, Wert, Abweichung von der wahren Gesamtgröße in Prozent, ob es das richtige Mittel der Aufgabe ist."""
    rows = []
    for name, r in M.MEAN_R.items():
        value = a.means[name]
        rows.append({"Mittel": name.capitalize(), "r": r, "Wert": value, "Abweichung (%)": 100.0 * float(M.relative_error(value, a.truth)), "richtig": name == a.task.right_name})
    return rows


# ---------------------------------------------------------------- Verdikte der Hauptansicht

def verdict_rates(task_key: str, truth: float, means: dict, r: float, selected: float) -> tuple[str, str]:
    """Verdikt der Raten-Aufgaben: die wahre Gesamtgröße, welches Mittel sie trifft und wie weit die anderen danebenliegen."""
    task = T.RATE_TASKS[task_key]
    digits = 3 if task.unit == "Faktor" else 2
    unit = "(Faktor je Periode)" if task.unit == "Faktor" else task.unit
    errors = {name: value / truth - 1.0 for name, value in means.items()}
    wrong = {name: e for name, e in errors.items() if name != task.right_name}
    parts = ", ".join(f"das {MEAN_ADJECTIVE[name]} {de(means[name], digits)} ({signed_pct(e)})" for name, e in wrong.items())
    worst = max(abs(e) for e in wrong.values())
    head = f"Die wahre Gesamtgröße ist **{de(truth, digits)}** {unit}, aus den Rohgrößen gerechnet. Das **{task.right_name}e** Mittel trifft sie ({de(means[task.right_name], digits)}); daneben liegen {parts}."
    tail = f" Das gewählte r = {de(r, 1)} gibt {de(selected, digits)} ({signed_pct(selected / truth - 1.0)})."
    if worst < 0.01:
        return "info", head + tail + " Bei dieser kleinen Streuung liegen die Mittel nahe beieinander: Die Wahl des Mittels kostet hier fast nichts, Experiment E1 zeigt, wie das mit der Streuung wächst."
    return "warning", head + tail


def verdict_lots(truth: float, arithmetic: float, harmonic: float, factor: float, n: int) -> tuple[str, str]:
    """Verdikt der Losgrößen: Mittel pro Los gegen die beiden Mittel der beobachteten Größen."""
    ea, eh = arithmetic / truth - 1.0, harmonic / truth - 1.0
    text = (f"Das Mittel pro Los ist **{de(truth, 1)} Stück**. Aus {n} beobachteten Stücken (jeweils die Größe des Loses, aus dem das Stück stammt) kommt: arithmetisches Mittel {de(arithmetic, 1)} ({signed_pct(ea)}), "
            f"harmonisches Mittel {de(harmonic, 1)} ({signed_pct(eh)}). Große Lose liefern mehr Stücke, deshalb liegt das arithmetische Mittel der Beobachtungen im Erwartungswert um den Faktor {de(factor, 2)} "
            f"(1 + CV² der Lose) über dem Mittel pro Los; das harmonische Mittel rechnet das zurück. Das ist eine einzelne Stichprobe: Experiment E3 zeigt die Streuung über 2 000 Wiederholungen.")
    return ("warning" if abs(ea) >= 0.05 else "info"), text


def verdict_table(rates: np.ndarray, pooled: np.ndarray, standard: np.ndarray, tv: float, reversal: bool) -> tuple[str, str]:
    """Verdikt der Tabelle: Quoten je Schicht, Gesamtquoten, Standardgewichte."""
    layers = ", ".join(f"{LAYER_NAMES[i]} {pct(rates[i, 0], 0)} gegen {pct(rates[i, 1], 0)} %" for i in range(2))
    if reversal:
        return "warning", (f"Gruppe A hat in beiden Schichten die höhere Quote ({layers}), insgesamt aber die niedrigere: **{pct(pooled[0], 0)} % gegen {pct(pooled[1], 0)} %**. "
                           f"Die Gesamtquote ist das mit den Gruppengrößen der Schichten gewichtete Mittel der Schichtquoten, und die Gewichte der beiden Gruppen unterscheiden sich (Gesamtvariationsabstand {de(tv, 1)}). "
                           f"Mit gemeinsamen Gewichten (½, ½) liegt A wieder vorn: {pct(standard[0], 0)} % gegen {pct(standard[1], 0)} %.")
    return "info", f"Hier kippt nichts ({layers}; insgesamt {pct(pooled[0], 0)} % gegen {pct(pooled[1], 0)} %)."


# ---------------------------------------------------------------- Verdikte der Experimente (Zeilen der vorgerechneten Messreihen)

def verdict_e1(cell: dict) -> tuple[str, str]:
    """E1: Kosten des falschen Mittels bei einer Aufgabe und einer Streuung. Das richtige Mittel hat die Abweichung null (Identität, kein Befund); der Befund ist der Preis der anderen."""
    task = T.RATE_TASKS[cell["task"]]
    wrong = {k: v["mean"] for k, v in cell["abs"].items() if k != cell["right"]}
    parts = ", ".join(f"{k} {de(v, 1)} %" for k, v in wrong.items())
    worst = max(wrong.values())
    text = (f"{task.label}, Streuung σ = {de(cell['sigma'], 2)}: das {cell['right']}e Mittel ist hier das richtige (Abweichung 0 % als Rechenidentität). "
            f"Die anderen weichen im Mittel von der wahren Gesamtgröße ab: {parts} ({cell['m']} Posten, {cell['instances_per_seed']} Instanzen je Seed, vier Seeds).")
    return ("warning" if worst >= 10.0 else "info"), text


def verdict_e2_paths(row: dict, N: int) -> tuple[str, str]:
    """E2: Pfade mit den Faktoren 1,5 und 0,6 nach T Perioden."""
    T_ = row["T"]
    text = (f"Nach {T_} Perioden liegen **{pct(row['below']['mean'], 1)} % der Pfade unter dem Start** (Binomialsumme: {pct(row['exact_below'], 1)} %), obwohl der Erwartungswert je Periode 1,05 beträgt. "
            f"Nur {pct(row['above']['mean'], 2)} % liegen über dem Erwartungswert {de_g(row['expected'], 3)}"
            + (f"; der Median ist {de_g(row['median']['mean'], 3)} (= 0,9^{T_ // 2})." if row["median_theory"] is not None else ".")
            + f" Das Stichprobenmittel von {de_int(N)} Pfaden ist das {de(row['mean_ratio']['mean'], 2)}-Fache des Erwartungswerts (Streuung zwischen den vier Seeds {de(row['mean_ratio']['std'], 2)}).")
    return "warning", text


def verdict_e2_lognormal(row: dict) -> tuple[str, str]:
    """E2: lognormale Faktoren mit Erwartungswert 1,05 je Periode, nach T Perioden."""
    shrinks = row["typical_formula"] < 1.0
    text = (f"Bei Streuung σ = {de(row['sigma'], 3)} der Logarithmen ist der typische Faktor je Periode {de(row['typical']['mean'], 4)} (Formel 1,05·exp(−σ²/2) = {de(row['typical_formula'], 4)}); "
            f"nach {row['T']} Perioden liegen {pct(row['below']['mean'], 1)} % der Pfade unter dem Start (exakt {pct(row['exact_below'], 1)} %). "
            + ("Der typische Pfad schrumpft, obwohl der Erwartungswert je Periode wächst." if shrinks else "Der typische Pfad wächst noch."))
    return ("warning" if shrinks else "info"), text


def verdict_e2_coverage(row: dict) -> tuple[str, str]:
    """E2: Wie oft trifft das Mittel von N Pfaden den Erwartungswert (± 20 %)?"""
    text = (f"{FAMILY_LABELS[row['family']].capitalize()}, T = {row['T']}: Das Mittel aus {de_int(row['N'])} Pfaden liegt in **{pct(row['within']['mean'], 1)} % der Stichproben** innerhalb von ± 20 % um den Erwartungswert; "
            f"der Median des Verhältnisses ist {de(row['median_ratio']['mean'], 2)} ({row['reps_per_seed']} Stichproben je Seed, vier Seeds).")
    return ("warning" if row["within"]["mean"] < 0.5 else "info"), text


def verdict_e2_live(res: dict) -> tuple[str, str]:
    """E2: Live-Messung der App (ein Seed)."""
    text = (f"{FAMILY_LABELS[res['family']].capitalize()}, {de_int(res['N'])} Pfade, T = {res['T']}: {pct(res['below'], 1)} % der Pfade liegen unter dem Start, {pct(res['above'], 2)} % über dem Erwartungswert {de_g(res['expected'], 3)}; "
            f"Median {de_g(res['median'], 3)}; Stichprobenmittel durch Erwartungswert: {de(res['mean_ratio'], 2)}.")
    return "info", text


def verdict_e3_factor(row: dict) -> tuple[str, str]:
    """E3: Verzerrungsfaktor Stücksicht / Lossicht einer Familie."""
    name = family_name(row["family"], row["par"])
    if math.isinf(row["formula"]):
        return "warning", f"{name}: Für α ≤ 2 gibt es keinen Faktor (kein zweites Moment)."
    text = (f"{name}: Das Mittel der Stücksicht ist im Mittel über vier Seeds das **{de(row['measured']['mean'], 2)}-Fache** des Mittels pro Los "
            f"(Formel 1 + CV² = {de(row['formula'], 2)}; Streuung zwischen den Seeds {de(row['measured']['std'], 2)}).")
    return ("warning" if row["formula"] >= 2.0 else "info"), text


def verdict_e3_back(row: dict) -> tuple[str, str]:
    """E3: Rückrechnung auf das Mittel pro Los aus n Beobachtungen."""
    name = family_name(row["family"], row["par"])
    text = (f"{name}, n = {de_int(row['n'])} Beobachtungen: arithmetisches Mittel {signed_pct(row['am_bias']['mean'] / 100.0)} Verzerrung (RMSE {de(row['am_rmse']['mean'], 1)} %), "
            f"harmonisches Mittel {signed_pct(row['hm_bias']['mean'] / 100.0)} (RMSE {de(row['hm_rmse']['mean'], 1)} %), jeweils in Prozent des wahren Mittels pro Los.")
    if row.get("nonempty") is not None:
        text += (f" Das harmonische Mittel bleibt {de(row['hm_bias']['mean'], 1)} % über dem Mittel: Leere Lose kommen in den Beobachtungen nie vor, es trifft das Mittel der nichtleeren Lose "
                 f"({de(row['nonempty'], 2)} statt {de(row['truth'], 2)}; Abweichung dazu {signed_pct(row['hm_bias_nonempty']['mean'] / 100.0)}).")
    return ("warning" if abs(row["am_bias"]["mean"]) >= 20.0 else "info"), text


def verdict_e3_tail(row: dict) -> tuple[str, str]:
    """E3: Pareto-Losgrößen mit Tail-Index α: Mittel der Stücksicht bei wachsendem n."""
    cells = ", ".join(f"n = {de_int(r['n'])}: {de_g(r['median']['mean'], 3)}" for r in row["rows"])
    if row["formula"] is None:
        return "warning", (f"Pareto(α = {de(row['alpha'], 1)}): Der Median des arithmetischen Mittels der Beobachtungen wächst mit n ({cells}). Es gibt keinen Faktor und kein Mittel der Stücksicht: "
                           "Mehr Beobachtungen machen die Schätzung nicht besser.")
    return "info", f"Pareto(α = {de(row['alpha'], 1)}): Der Median des arithmetischen Mittels der Beobachtungen nähert sich dem Wert (α − 1)/(α − 2) = {de(row['formula'], 2)} ({cells})."


def verdict_e4_reversal(row: dict) -> tuple[str, str]:
    """E4: Anteil der Tabellen mit Simpson-Umkehr bei zufälligen Zellwahrscheinlichkeiten."""
    return "info", (f"Zufällige Tabellen mit zwei Gruppen, zwei Ausgängen und {row['layers']} Schichten (Zellwahrscheinlichkeiten gleichverteilt): Umkehr in **{pct(row['share']['mean'], 3)} % der Tabellen** "
                    f"({de_int(row['hits'])} von {de_int(row['tables_per_seed'] * 4)}; Standardfehler {pct(row['standard_error'], 3)} Prozentpunkte).")


def verdict_e4_weights(row: dict) -> tuple[str, str]:
    """E4: Gruppe A ist in jeder Schicht besser; wie oft kippt die Gesamtquote bei Gewichtsstreuung τ?"""
    head = f"Gruppe A hat in jeder der {row['k']} Schichten die höhere Quote. Bei Gewichtsstreuung τ = {de(row['tau'], 2)} kippt die Gesamtquote in **{pct(row['share']['mean'], 2)} % der Tabellen** zugunsten von B"
    if row["tv_flipped_median"] is None:
        body = f" (mittlerer Gewichtsabstand {de(row['tv_mean']['mean'], 2)}; keine Umkehr)."
    else:
        body = f" (mittlerer Gewichtsabstand {de(row['tv_mean']['mean'], 2)}, unter den Umkehrungen im Median {de(row['tv_flipped_median']['mean'], 2)})."
    control = (f" Kontrolle (erzwungen, keine Messung): mit gleichen Gewichten {row['equal_weight_flips']} und mit gemeinsamen Standardgewichten {row['standard_weight_flips']} Umkehrungen.")
    return "info", head + body + control


def verdict_e5(cell: dict) -> tuple[str, str]:
    """E5: gleiche Standortquote in beiden Gruppen, verschiedene Kopplung von Nachfrage und Quote."""
    if cell["kappa_b"] == 0.0:
        text = (f"Beide Gruppen ohne Kopplung (κ = 0): Der Unterschied B − A ist ungewichtet {de(cell['diff_mean_pp']['mean'], 2)} Prozentpunkte, bei der Gesamtquote {de(cell['diff_pooled_pp']['mean'], 2)}; "
                f"B liegt in {pct(cell['b_ahead_mean']['mean'], 1)} % beziehungsweise {pct(cell['b_ahead_pooled']['mean'], 1)} % der Vergleiche vorn (Zufall: 50 %).")
        return "info", text
    text = (f"Gleiche wahre Standortquote in beiden Gruppen, aber in B Kopplung κ = {de(cell['kappa_b'], 1)} von Nachfrage und Quote (Streuung der Nachfrage {de(cell['sigma_d'], 1)}): "
            f"ungewichtet ist B − A {de(cell['diff_mean_pp']['mean'], 2)} Prozentpunkte, bei der **Gesamtquote {de(cell['diff_pooled_pp']['mean'], 2)}**. "
            f"B liegt ungewichtet in {pct(cell['b_ahead_mean']['mean'], 1)} % der Vergleiche vorn, nach Gesamtquote in {pct(cell['b_ahead_pooled']['mean'], 1)} %.")
    return "warning", text
