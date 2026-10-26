"""Aufgaben mit Zähler und Nenner (DEMO-PLAYBOOK §1b): Zu jeder Aufgabe gehört eine wahre Gesamtgröße, die aus den Rohgrößen gerechnet wird (Gesamtstrecke durch Gesamtzeit,
Gesamtliter durch Gesamtstrecke, Produkt der Faktoren), **nicht** aus einem Mittel. Das richtige Mittel ist das, was diese Gesamtgröße wiedergibt.

Rote Linie der Demo: Die Gesamtgröße ist das mit den Nennern gewichtete arithmetische Mittel der Einzelraten.
* gleiche Strecke je Fahrt: Nenner = Zeit = s/v, schnelle Fahrten haben wenig Gewicht, das gewichtete arithmetische Mittel ist das **harmonische** der Geschwindigkeiten;
* gleiche Zeit je Fahrt: Nenner = Zeit konstant, das **arithmetische** Mittel;
* Verbrauch l/100 km bei gleicher Strecke: Nenner = Strecke konstant, das **arithmetische** Mittel der l/100 km (das harmonische, das man mit km/l bekommt, ist hier falsch);
* Wachstumsfaktoren: Gesamtfaktor = Produkt, der gleichwertige Faktor je Periode ist das **geometrische** Mittel."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import rgw_means as M
import rgw_scenario as S

SPEED_SCALE = 80.0         # km/h, Median der Geschwindigkeiten
CONSUMPTION_SCALE = 8.0    # l/100 km, Median des Verbrauchs
LEG_KM = 10.0              # Strecke einer Fahrt bei „gleiche Strecke“
LEG_HOURS = 1.0            # Dauer einer Fahrt bei „gleiche Zeit“
CONSUMPTION_KM = 100.0     # Strecke je Fahrzeug beim Verbrauch
UP, DOWN = 1.5, 0.6        # Wachstumsfaktoren des Wechsels


@dataclass(frozen=True)
class RateTask:
    key: str
    label: str
    unit: str
    right_r: float          # Exponent des Mittels, das die wahre Gesamtgröße wiedergibt
    right_name: str         # Name dieses Mittels
    uses_sigma: bool        # hängt die Aufgabe von der Streuung ab?


RATE_TASKS = {
    "strecke": RateTask("strecke", "Geschwindigkeit, gleiche Strecke je Fahrt", "km/h", -1.0, "harmonisch", True),
    "zeit": RateTask("zeit", "Geschwindigkeit, gleiche Zeit je Fahrt", "km/h", 1.0, "arithmetisch", True),
    "verbrauch": RateTask("verbrauch", "Verbrauch in l/100 km, gleiche Strecke je Fahrzeug", "l/100 km", 1.0, "arithmetisch", True),
    "faktoren": RateTask("faktoren", "Wachstumsfaktoren je Periode (zufällig, lognormal)", "Faktor", 0.0, "geometrisch", True),
    "wechsel": RateTask("wechsel", "Wachstumsfaktoren 1,5 und 0,6 im Wechsel", "Faktor", 0.0, "geometrisch", False),
}


def raw_quantities(task: str, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zähler und Nenner zu den Einzelraten x (letzte Achse) einer Aufgabe mit Gesamtrate: (Strecke, Zeit) bzw. (Liter, Hundert-km-Einheiten)."""
    x = np.asarray(x, dtype=float)
    if task == "strecke":
        km = np.full(x.shape, LEG_KM)
        return km, km / x
    if task == "zeit":
        hours = np.full(x.shape, LEG_HOURS)
        return x * hours, hours
    if task == "verbrauch":
        per_100 = np.full(x.shape, CONSUMPTION_KM / 100.0)
        return x * per_100, per_100
    raise ValueError(f"{task} hat keine Gesamtrate aus Zähler und Nenner")


def true_total(task: str, x) -> np.ndarray | float:
    """Wahre Gesamtgröße einer Aufgabe aus den Rohgrößen: Gesamtrate bei Geschwindigkeit und Verbrauch, Produkt^(1/m) bei Faktoren (letzte Achse)."""
    x = np.asarray(x, dtype=float)
    if task in ("faktoren", "wechsel"):
        out = np.prod(x, axis=-1) ** (1.0 / x.shape[-1])
        return float(out) if out.ndim == 0 else out
    return M.total_rate(*raw_quantities(task, x))


def alternating_factors(m: int) -> np.ndarray:
    """m Perioden im Wechsel 1,5 und 0,6 (beginnend mit 1,5)."""
    if m < 1:
        raise ValueError("mindestens eine Periode")
    return np.where(np.arange(m) % 2 == 0, UP, DOWN)


def make_batch(task: str, draws: S.Draws, runs: int, m: int, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """`runs` Instanzen mit je m Einzelraten und ihre wahren Gesamtgrößen: (x mit Form (runs, m), Wahrheit mit Form (runs,))."""
    if task == "wechsel":
        x = np.tile(alternating_factors(m), (runs, 1))
    elif task == "faktoren":
        x = S.lognormal(draws, (runs, m), sigma)
    elif task in ("strecke", "zeit"):
        x = S.lognormal(draws, (runs, m), sigma, SPEED_SCALE)
    elif task == "verbrauch":
        x = S.lognormal(draws, (runs, m), sigma, CONSUMPTION_SCALE)
    else:
        raise ValueError(f"unbekannte Aufgabe: {task}")
    return x, np.asarray(true_total(task, x))
