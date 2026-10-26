"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Komplexität am Slider-Maximum (DEMO-PLAYBOOK §1c): Die Hauptansicht rechnet m ≤ 200 Posten (O(m)) und eine r-Kurve mit 101 Stützstellen (O(101·m)). Die Live-Knöpfe der Experimente sind
durch Budgets begrenzt: Pfade N·T ≤ LIVE_PATH_BUDGET Gleichverteilte (E2), Rückrechnung reps·n ≤ LIVE_BACK_BUDGET Beobachtungen (E3), Tabellen 4·l·R ≤ LIVE_TABLE_BUDGET Gleichverteilte (E4)."""

import rgw_tasks as T

# --- Regler der Seitenleiste ---------------------------------------------------------------------------------------------------------------
TASK_KEYS = ("strecke", "zeit", "verbrauch", "wechsel", "lose", "tabelle")
TASK_LABELS = {
    "strecke": "Geschwindigkeit, gleiche Strecke je Fahrt",
    "zeit": "Geschwindigkeit, gleiche Zeit je Fahrt",
    "verbrauch": "Verbrauch in l/100 km, gleiche Strecke je Fahrzeug",
    "wechsel": "Wachstumsfaktoren 1,5 und 0,6 im Wechsel",
    "lose": "Losgrößen: pro Los oder pro Stück?",
    "tabelle": "Zwei Gruppen in zwei Schichten (Tabelle)",
}
RATE_TASK_KEYS = tuple(k for k in TASK_KEYS if k in T.RATE_TASKS)       # Aufgaben mit einer Liste von Raten und einem r-Diagramm
DEFAULT_TASK = "strecke"
M_MIN, M_MAX, DEFAULT_M = 2, 200, 20                  # Posten (Fahrten, Perioden) bzw. beobachtete Stücke bei den Losen
SIGMA_MIN, SIGMA_MAX, DEFAULT_SIGMA = 0.05, 1.5, 0.5  # Streuung der Logarithmen der Einzelraten bzw. der Losgrößen
R_MIN, R_MAX, DEFAULT_R = -5.0, 5.0, 1.0              # Mittelungsexponent r des Potenzmittels M_r (nicht p: siehe lp-mitte-demo)
SEED_MAX, DEFAULT_SEED = 999999, 35

R_CURVE = tuple(i / 10 for i in range(-50, 51))       # Stützstellen der r-Kurve: −5 bis 5 in Schritten von 0,1
LOTS_POPULATION = 2000                                # Lose der Grundgesamtheit in der Hauptansicht „Losgrößen“
LOTS_FAMILY = "lognormal"                             # Familie der Losgrößen in der Hauptansicht

# --- Live-Budgets der Experimente (Obergrenzen für die Knöpfe „Nachmessen“) -----------------------------------------------------------------
RESULTS_FILE = "rgw_results.json"
LIVE_PATH_BUDGET = 5_000_000                          # N · T Gleichverteilte je Live-Lauf (E2)
LIVE_PATH_N = (1000, 10_000, 100_000)
LIVE_PATH_T = (10, 20, 50, 100, 200)
LIVE_BACK_BUDGET = 2_000_000                          # Wiederholungen · n Beobachtungen (E3)
LIVE_BACK_REPS = 500
LIVE_TABLE_BUDGET = 20_000_000                        # 4 · l · Tabellen Gleichverteilte (E4)
LIVE_COMPARISONS = 20_000                             # Gruppenvergleiche (E5)

# --- Farben --------------------------------------------------------------------------------------------------------------------------------
COLOR_DATA = "rgba(120,120,120,0.55)"
COLOR_TRUE = "#222222"
MEAN_COLORS = {"harmonisch": "#8c6bb1", "geometrisch": "#2F6B65", "arithmetisch": "#d62728", "quadratisch": "#e08a1e"}
COLOR_A = "#2F6B65"
COLOR_B = "#8c6bb1"
COLOR_BAND = "rgba(76,120,168,0.18)"
COLOR_LINE = "#4c78a8"

# --- Presets -------------------------------------------------------------------------------------------------------------------------------
PRESETS = {
    "Gleiche Strecke (Standard)": {"task": "strecke", "m": DEFAULT_M, "sigma": DEFAULT_SIGMA, "r": 1.0, "seed": DEFAULT_SEED},
    "Gleiche Zeit": {"task": "zeit", "m": DEFAULT_M, "sigma": DEFAULT_SIGMA, "r": 1.0, "seed": DEFAULT_SEED},
    "Verbrauch l/100 km gegen km/l": {"task": "verbrauch", "m": DEFAULT_M, "sigma": DEFAULT_SIGMA, "r": -1.0, "seed": DEFAULT_SEED},
    "Faktoren 1,5 und 0,6": {"task": "wechsel", "m": 20, "sigma": DEFAULT_SIGMA, "r": 1.0, "seed": DEFAULT_SEED},
    "Pro Los, pro Stück": {"task": "lose", "m": 100, "sigma": 1.0, "r": DEFAULT_R, "seed": DEFAULT_SEED},
    "Die Tabelle, die kippt": {"task": "tabelle", "m": DEFAULT_M, "sigma": DEFAULT_SIGMA, "r": DEFAULT_R, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Gleiche Strecke (Standard)": "20 Fahrten über dieselbe Strecke mit verschiedenen Geschwindigkeiten (Streuung 0,5): Die Gesamtgeschwindigkeit ist Gesamtstrecke durch Gesamtzeit, das harmonische Mittel. Das arithmetische Mittel (r = 1) liegt deutlich darüber.",
    "Gleiche Zeit": "Dieselben Fahrten, aber jede dauert eine Stunde: Jetzt ist das arithmetische Mittel die Gesamtgeschwindigkeit, und das harmonische Mittel liegt darunter. Das Mittel gehört zum Nenner, nicht zur Zahl.",
    "Verbrauch l/100 km gegen km/l": "20 Fahrzeuge fahren je 100 km. Der Gesamtverbrauch in l/100 km ist das arithmetische Mittel der l/100 km. Wer km/l mittelt und zurückrechnet, landet auf dem harmonischen Mittel der l/100 km (r = −1).",
    "Faktoren 1,5 und 0,6": "Der Bedarf wächst abwechselnd um 50 % und fällt um 40 %: Im Durchschnitt +5 % je Periode (arithmetisch 1,05), aber nach zwei Perioden steht er bei 0,9. Das richtige Mittel für Wachstum ist das geometrische.",
    "Pro Los, pro Stück": "Lose mit lognormal verteilten Größen. Wer Stücke zieht und nach der Größe ihres Loses fragt, sieht große Lose häufiger: Das arithmetische Mittel der Beobachtungen liegt weit über dem Mittel pro Los, das harmonische trifft es.",
    "Die Tabelle, die kippt": "Gruppe A hat in jeder der beiden Schichten die höhere Quote und insgesamt die niedrigere (63 % gegen 76 %), weil die Gruppen verschieden auf die Schichten verteilt sind. Mit gleichen Gewichten kippt es zurück.",
}
