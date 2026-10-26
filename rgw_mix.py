"""Gewichtsmix als Einheiten (DEMO-PLAYBOOK §1b): Gesamtquote gegen Quote je Schicht, gleiche Gewichte, Standardgewichte, Gewichtsunterschied.

Eine Tabelle hat l Schichten, zwei Gruppen (A, B) und zwei Ausgänge (Erfolg, Misserfolg): Form (l, 2, 2), Achsen [Schicht, Gruppe, Ausgang], Ausgang 0 = Erfolg.
Die **Gesamtquote** einer Gruppe ist das mit den Gruppenzahlen der Schichten gewichtete Mittel der Schichtquoten (Nenner als Gewichte):
    Gesamtquote_g = Σ_s w_gs · Quote_gs     mit  w_gs = n_gs / Σ_s n_gs
**Simpson-Umkehr**: Gruppe A hat in jeder Schicht eine höhere (oder in jeder eine niedrigere) Quote als B, aber die Gesamtquote steht umgekehrt. Das geht nur, wenn die Gewichte w_As und w_Bs verschieden sind.
Mit **gemeinsamen Standardgewichten** (dieselben Gewichte für beide Gruppen) ist die Umkehr ausgeschlossen: Beide Gesamtgrößen sind Konvexkombinationen derselben Gewichte.
Die Kausalmechanik (eine gemeinsame Ursache kippt das Vorzeichen) steht in `konfundierung-demo`; hier wird gezählt, wie häufig und bei welchem Gewichtsunterschied es kippt."""

from __future__ import annotations

import math

import numpy as np

import rgw_scenario as S


def random_tables(draws: S.Draws, runs: int, layers: int) -> np.ndarray:
    """`runs` Tabellen 2×2×l mit gleichverteilten Zellwahrscheinlichkeiten (Dirichlet(1, …, 1)): normierte Exponentialverteilte. Form (runs, l, 2, 2)."""
    e = draws.exponential((runs, layers, 2, 2))
    return e / e.sum(axis=(1, 2, 3), keepdims=True)


def stratum_rates(cells: np.ndarray) -> np.ndarray:
    """Erfolgsquote je Schicht und Gruppe, Form (..., l, 2). Eine leere Zelle gibt nan (nicht definiert)."""
    cells = np.asarray(cells, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return cells[..., 0] / cells.sum(axis=-1)


def group_weights(cells: np.ndarray) -> np.ndarray:
    """Gewichte der Schichten je Gruppe (Anteil der Gruppe, der in der Schicht steht), Form (..., l, 2); Summe über die Schichten je Gruppe ist 1."""
    n = np.asarray(cells, dtype=float).sum(axis=-1)
    return n / n.sum(axis=-2, keepdims=True)


def pooled_rates(cells: np.ndarray) -> np.ndarray:
    """Gesamtquote je Gruppe aus den Summen über die Schichten, Form (..., 2)."""
    cells = np.asarray(cells, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return cells[..., 0].sum(axis=-2) / cells.sum(axis=-1).sum(axis=-2)


def weighted_total(rates: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Gewichtetes Mittel der Schichtquoten über die Schichtachse (vorletzte): Σ_s w_s · Quote_s. Form (..., 2) bei Eingabe (..., l, 2)."""
    return (np.asarray(weights, dtype=float) * np.asarray(rates, dtype=float)).sum(axis=-2)


def standard_weights(weights: np.ndarray) -> np.ndarray:
    """Gemeinsame Standardgewichte: das Mittel der beiden Gruppengewichte, (w_A + w_B)/2, für beide Gruppen gleich. Form (..., l)."""
    return np.asarray(weights, dtype=float).mean(axis=-1)


def total_variation(weights: np.ndarray) -> np.ndarray:
    """Gesamtvariationsabstand der Gewichte der beiden Gruppen: ½ Σ_s |w_As − w_Bs| (0 bei gleichen, 1 bei getrennten Gewichten). Form (...,)."""
    w = np.asarray(weights, dtype=float)
    return 0.5 * np.abs(w[..., 0] - w[..., 1]).sum(axis=-1)


def reversal_mask(cells: np.ndarray) -> np.ndarray:
    """Simpson-Umkehr in jeder Tabelle (Form (runs, l, 2, 2) oder (l, 2, 2)): Die Differenz A − B der Schichtquoten hat in allen Schichten dasselbe Vorzeichen (nicht null),
    und die Differenz der Gesamtquoten hat das entgegengesetzte (nicht null)."""
    rate = stratum_rates(cells)
    diff = rate[..., 0] - rate[..., 1]
    pooled = pooled_rates(cells)
    total = pooled[..., 0] - pooled[..., 1]
    all_pos = (diff > 0).all(axis=-1)
    all_neg = (diff < 0).all(axis=-1)
    return (all_pos & (total < 0)) | (all_neg & (total > 0))


def has_reversal(table) -> bool:
    """Umkehr in einer einzelnen Tabelle (Form (l, 2, 2)), etwa einer Tabelle ganzzahliger Zählungen."""
    return bool(reversal_mask(np.asarray(table, dtype=float)))


def dominated_rates_and_weights(draws: S.Draws, runs: int, layers: int, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Tabellen, in denen A in jeder Schicht die höhere Quote hat: Quoten (runs, l, 2) aus zwei gleichverteilten Ziehungen je Schicht (A bekommt die größere, B die kleinere),
    Gewichte (runs, l, 2) je Gruppe lognormal(τ) und auf Summe 1 normiert. τ = 0 gibt gleiche Gewichte 1/l, je größer τ, desto verschiedener die Gruppen."""
    u = draws.uniform((runs, layers, 2))
    rates = np.stack([u.max(axis=-1), u.min(axis=-1)], axis=-1)
    raw = np.exp(tau * draws.normal((runs, layers, 2)))
    return rates, raw / raw.sum(axis=-2, keepdims=True)


def service_levels(z: np.ndarray, base: float, spread: float) -> np.ndarray:
    """Servicegrad je Standort aus der Standardnormalverteilten z: base + spread·z, auf [0, 1] begrenzt."""
    return np.clip(base + spread * np.asarray(z, dtype=float), 0.0, 1.0)


def coupled_demand(z: np.ndarray, e: np.ndarray, kappa: float, sigma_d: float, scale: float = 100.0) -> np.ndarray:
    """Nachfrage je Standort, gekoppelt an die Standardnormalverteilte z des Servicegrads: scale·exp(σ_d·(κ·z + √(1−κ²)·e)); κ = 0 unabhängig, κ > 0 haben starke Standorte mehr Nachfrage."""
    return scale * np.exp(sigma_d * (kappa * z + math.sqrt(1.0 - kappa * kappa) * e))


def pooled_rate(rate: np.ndarray, demand: np.ndarray) -> np.ndarray:
    """Gesamtquote über Standorte (letzte Achse): Σ Quote·Nachfrage / Σ Nachfrage."""
    return (rate * demand).sum(axis=-1) / demand.sum(axis=-1)


def mean_rate(rate: np.ndarray) -> np.ndarray:
    """Ungewichtetes Mittel der Standortquoten (letzte Achse)."""
    return rate.mean(axis=-1)


def standardised_totals(rates: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Gesamtquoten beider Gruppen mit den gemeinsamen Standardgewichten (w_A + w_B)/2. Form (..., 2)."""
    return weighted_total(rates, standard_weights(weights)[..., None])
