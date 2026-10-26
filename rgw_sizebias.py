"""Größenverzerrung als Einheiten (DEMO-PLAYBOOK §1b): Wer Stücke zieht (Fahrgäste, Kunden, Stück aus dem Lager), sieht große Lose häufiger, in genau dem Verhältnis ihrer Größe.

Lose haben Größen X (Fahrgäste je Fahrzeug, Stück je Los). Zieht man ein **Stück** gleichverteilt unter allen Stücken und fragt nach der Größe seines Loses, ist die Wahrscheinlichkeit für ein
Los proportional zu seiner Größe: Das Mittel dieser Beobachtungen ist E[X²]/E[X] = E[X]·(1 + CV²), um den Faktor 1 + CV² zu groß gegenüber dem Mittel pro Los.
Zwei Rechnungen, die in der Demo gemessen werden:
* Das **harmonische Mittel** der beobachteten Größen ist (bei Losen ohne leere Lose) das Mittel pro Los: E[1/Y] = Σ_x (x p(x)/E[X])·(1/x) = 1/E[X].
* **Leere Lose** (X = 0) haben in der Stücksicht die Wahrscheinlichkeit 0: Das harmonische Mittel liefert dann das Mittel der **nichtleeren** Lose.
Die Wartezeit-Formel für den Takt (Verhältnis 1 + CV², Wartezeit pro Fahrgast statt pro Bus) steht in `bunching-demo`; hier geht es um die Rückrechnung auf die Größe pro Los."""

from __future__ import annotations

import math

import numpy as np

import rgw_scenario as S

# Verteilungsfamilien der Losgrößen: Schlüssel -> (Name, Name des Parameters)
FAMILIES = {
    "poisson": ("Poisson (Fahrgäste je Fahrzeug)", "λ"),
    "geometrisch": ("geometrisch (Länge einer Kolonne)", "p"),
    "lognormal": ("lognormal (Stück je Los)", "σ"),
    "pareto": ("Pareto (Stück je Los, schwerer Tail)", "α"),
}
LOGNORMAL_SCALE = 10.0     # Median der lognormalen Losgröße in Stück
PARETO_MIN = 1.0           # kleinste Losgröße der Pareto-Verteilung in Stück


def lot_sizes(family: str, par: float, draws: S.Draws, shape) -> np.ndarray:
    """Losgrößen aus einer Familie (Poisson darf 0 liefern, die anderen sind positiv)."""
    if family == "poisson":
        return S.poisson(draws, shape, par).astype(float)
    if family == "geometrisch":
        return S.geometric(draws, shape, par).astype(float)
    if family == "lognormal":
        return S.lognormal(draws, shape, par, LOGNORMAL_SCALE)
    if family == "pareto":
        return S.pareto(draws, shape, par, PARETO_MIN)
    raise ValueError(f"unbekannte Familie: {family}")


def family_mean(family: str, par: float) -> float:
    """Mittel pro Los E[X] (unendlich, wo es nicht existiert)."""
    if family == "poisson":
        return par
    if family == "geometrisch":
        return 1.0 / par
    if family == "lognormal":
        return LOGNORMAL_SCALE * math.exp(0.5 * par * par)
    if family == "pareto":
        return PARETO_MIN * par / (par - 1.0) if par > 1.0 else math.inf
    raise ValueError(f"unbekannte Familie: {family}")


def family_second_moment(family: str, par: float) -> float:
    """E[X²] (unendlich, wo es nicht existiert)."""
    if family == "poisson":
        return par + par * par
    if family == "geometrisch":
        return (2.0 - par) / (par * par)
    if family == "lognormal":
        return LOGNORMAL_SCALE ** 2 * math.exp(2.0 * par * par)
    if family == "pareto":
        return PARETO_MIN ** 2 * par / (par - 2.0) if par > 2.0 else math.inf
    raise ValueError(f"unbekannte Familie: {family}")


def piece_mean_formula(family: str, par: float) -> float:
    """Mittel der Stücksicht E[X²]/E[X] (unendlich für Pareto mit α ≤ 2)."""
    second = family_second_moment(family, par)
    return math.inf if math.isinf(second) else second / family_mean(family, par)


def bias_factor_formula(family: str, par: float) -> float:
    """Verzerrungsfaktor Stücksicht / Lossicht = E[X²]/E[X]² = 1 + CV² (unendlich für Pareto mit α ≤ 2)."""
    second = family_second_moment(family, par)
    return math.inf if math.isinf(second) else second / family_mean(family, par) ** 2


def size_biased_draw(family: str, par: float, draws: S.Draws, shape) -> np.ndarray:
    """Beobachtete Größen in der Stücksicht, aus der Formel gezogen (unendliche Grundgesamtheit):
    Poisson(λ) → 1 + Poisson(λ); geometrisch(p) → Summe zweier geometrischer minus 1; lognormal(μ, σ) → lognormal(μ + σ², σ); Pareto(α) → Pareto(α − 1)."""
    if family == "poisson":
        return 1.0 + S.poisson(draws, shape, par)
    if family == "geometrisch":
        return (S.geometric(draws, shape, par) + S.geometric(draws, shape, par) - 1).astype(float)
    if family == "lognormal":
        return LOGNORMAL_SCALE * np.exp(par * par + par * draws.normal(shape))
    if family == "pareto":
        if par <= 1.0:
            raise ValueError("die Stücksicht der Pareto-Verteilung braucht α > 1")
        return S.pareto(draws, shape, par - 1.0, PARETO_MIN)
    raise ValueError(f"unbekannte Familie: {family}")


def pieces_from_population(sizes: np.ndarray, n: int, draws: S.Draws) -> np.ndarray:
    """Zieht n Stücke gleichverteilt unter allen Stücken einer endlichen Grundgesamtheit von Losen und gibt die Größe des jeweiligen Loses zurück.
    Ein Los der Größe x wird mit Wahrscheinlichkeit x/Σx gezogen (aufsummierte Größen, eine Gleichverteilte je Stück)."""
    sizes = np.asarray(sizes, dtype=float)
    cum = np.cumsum(sizes)
    if cum[-1] <= 0.0:
        raise ValueError("alle Lose sind leer: es gibt kein Stück zu ziehen")
    index = np.searchsorted(cum, draws.uniform(n) * cum[-1], side="right").clip(0, len(sizes) - 1)
    return sizes[index]


def bias_factor(sizes) -> float:
    """Verzerrungsfaktor einer endlichen Grundgesamtheit: Mittel der Quadrate durch Quadrat des Mittels."""
    sizes = np.asarray(sizes, dtype=float)
    return float((sizes ** 2).mean() / sizes.mean() ** 2)


def arithmetic_back(observed) -> np.ndarray:
    """Arithmetisches Mittel der beobachteten Größen (letzte Achse): das Mittel der Stücksicht, nicht das pro Los."""
    return np.asarray(observed, dtype=float).mean(axis=-1)


def harmonic_back(observed) -> np.ndarray:
    """Harmonisches Mittel der beobachteten Größen (letzte Achse): n / Σ(1/y), die Rückrechnung auf das Mittel pro Los."""
    observed = np.asarray(observed, dtype=float)
    if np.any(observed <= 0.0):
        raise ValueError("beobachtete Größen sind positiv (ein Los der Größe 0 wird nie beobachtet)")
    return observed.shape[-1] / (1.0 / observed).sum(axis=-1)


def nonempty_mean_poisson(lam: float) -> float:
    """Mittel der nichtleeren Lose einer Poisson(λ)-Verteilung: λ/(1 − e^(−λ))."""
    return lam / (1.0 - math.exp(-lam))
