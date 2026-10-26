"""Mittel als Einheiten (DEMO-PLAYBOOK §1b): das Potenzmittel M_r und das gewichtete Mittel, dazu die Gesamtgrößen aus Zähler und Nenner.

Der Exponent heißt hier **r** (Mittelungsexponent). In `lp-mitte-demo` ist p der Exponent der Strecke (die Lp-Mitte), in `p-center-demo` die Zahl der Mittelpunkte.

    M_r(x; w) = ( Σ w_i x_i^r )^(1/r)     mit Σ w_i = 1,   M_0 = exp(Σ w_i ln x_i),   M_(−∞) = min,   M_(+∞) = max

r = −1 harmonisch, 0 geometrisch, 1 arithmetisch, 2 quadratisch. Alle Werte müssen positiv sein (ein Mittel von Raten, Faktoren und Größen).
Reine numpy-Funktionen über die letzte Achse, ohne scipy; tests/test_oracle_means.py prüft sie gegen `scipy.stats`."""

from __future__ import annotations

import numpy as np

# Die vier Mittel der Demo mit ihrem Exponenten r (Reihenfolge = Reihenfolge der Mittel: kleinstes zuerst)
MEAN_R = {"harmonisch": -1.0, "geometrisch": 0.0, "arithmetisch": 1.0, "quadratisch": 2.0}


def _positive(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError("ein Mittel braucht mindestens einen Wert")
    if np.any(x <= 0.0) or not np.all(np.isfinite(x)):
        raise ValueError("Potenzmittel brauchen endliche positive Werte")
    return x


def normalised_weights(x: np.ndarray, w) -> np.ndarray:
    """Gewichte auf Summe 1 (letzte Achse); ohne Angabe gleiche Gewichte. Negative oder alle-null-Gewichte sind ein Fehler."""
    if w is None:
        return np.full(x.shape, 1.0 / x.shape[-1])
    w = np.broadcast_to(np.asarray(w, dtype=float), x.shape)
    total = w.sum(axis=-1, keepdims=True)
    if np.any(w < 0.0) or np.any(total <= 0.0):
        raise ValueError("Gewichte müssen nichtnegativ sein und eine positive Summe haben")
    return w / total


def power_mean(x, r: float, w=None):
    """Gewichtetes Potenzmittel M_r über die letzte Achse. r = 0 ist das geometrische Mittel, r = ±inf Maximum bzw. Minimum (Gewichte spielen dort keine Rolle).
    Numerisch stabil: für r > 0 wird durch das Maximum, für r < 0 durch das Minimum geteilt, bevor potenziert wird."""
    x = _positive(x)
    if r == np.inf:
        return x.max(axis=-1)
    if r == -np.inf:
        return x.min(axis=-1)
    wn = normalised_weights(x, w)
    if r == 0.0:
        return np.exp((wn * np.log(x)).sum(axis=-1))
    anchor = x.max(axis=-1, keepdims=True) if r > 0 else x.min(axis=-1, keepdims=True)
    return anchor[..., 0] * (wn * (x / anchor) ** r).sum(axis=-1) ** (1.0 / r)


def harmonic(x, w=None):
    return power_mean(x, -1.0, w)


def geometric(x, w=None):
    return power_mean(x, 0.0, w)


def arithmetic(x, w=None):
    """Gewichtetes arithmetisches Mittel Σ w_i x_i / Σ w_i (Werte dürfen hier auch null sein: es wird nicht potenziert)."""
    x = np.asarray(x, dtype=float)
    return (normalised_weights(x, w) * x).sum(axis=-1)


def quadratic(x, w=None):
    return power_mean(x, 2.0, w)


def four_means(x, w=None) -> dict:
    """Die vier Mittel harmonisch, geometrisch, arithmetisch, quadratisch einer Menge."""
    return {name: float(power_mean(x, r, w)) for name, r in MEAN_R.items()}


def total_rate(numerator, denominator) -> float:
    """Gesamtgröße einer Rate: Summe der Zähler durch Summe der Nenner (Gesamtstrecke / Gesamtzeit, Gesamtliter / Gesamtstrecke). Der unabhängige Rechenweg zu jedem Mittel."""
    out = np.asarray(numerator, dtype=float).sum(axis=-1) / np.asarray(denominator, dtype=float).sum(axis=-1)
    return float(out) if out.ndim == 0 else out


def relative_error(value, truth):
    """Abweichung eines Wertes von der wahren Größe, bezogen auf die wahre Größe: value / truth − 1 (Vorzeichen: positiv heißt zu groß)."""
    return np.asarray(value, dtype=float) / np.asarray(truth, dtype=float) - 1.0
