"""Wachstumspfade als Einheiten (DEMO-PLAYBOOK §1b): Das Produkt von Perioden-Faktoren, der Erwartungswert und der typische Pfad.

Zweipunkt-Faktoren: jede Periode wächst der Wert mit Wahrscheinlichkeit p um den Faktor `up` (hier 1,5) und fällt sonst auf `down` (hier 0,6). Nach T Perioden ist der Wert
up^K · down^(T−K) mit K ~ Binomial(T, p). Erwartungswert je Periode 0,5·1,5 + 0,5·0,6 = 1,05; geometrisches Mittel √(1,5·0,6) = √0,9 = 0,9487.

Lognormale Faktoren: log-Faktor ~ N(ln m − σ²/2, σ²), damit der Erwartungswert je Periode m bleibt (1,05) und nur die Streuung wächst; der typische Faktor ist dann m·exp(−σ²/2),
der Pfad schrumpft typisch ab σ = √(2 ln m)."""

from __future__ import annotations

import math

import numpy as np

import rgw_scenario as S

UP, DOWN, P_UP = 1.5, 0.6, 0.5
MEAN_FACTOR = 1.05            # Erwartungswert je Periode (0,5·1,5 + 0,5·0,6)
CHUNK_DRAWS = 2_000_000       # so viele Gleichverteilte höchstens auf einmal im Speicher


def path_value(ups, T: int, up: float = UP, down: float = DOWN):
    """Endwert nach T Perioden bei K Aufwärtsschritten: up^K · down^(T−K) (über Logarithmen, damit T = 200 nicht überläuft)."""
    ups = np.asarray(ups, dtype=float)
    return np.exp(ups * math.log(up) + (T - ups) * math.log(down))


def expected_value(T: int, up: float = UP, down: float = DOWN, p: float = P_UP) -> float:
    """Erwartungswert des Endwerts: (p·up + (1−p)·down)^T."""
    return (p * up + (1.0 - p) * down) ** T


def typical_value(T: int, up: float = UP, down: float = DOWN) -> float:
    """Median des Endwerts für gerades T und p = 1/2: (up·down)^(T/2), das ist 0,9^(T/2) (bei ungeradem T liegt der Median des Zweipunkt-Gitters auf einem Gitterpunkt, keine glatte Formel)."""
    if T % 2:
        raise ValueError("die Formel gilt nur für gerades T")
    return (up * down) ** (T // 2)


def ups_by_steps(draws: S.Draws, N: int, T: int, p: float = P_UP) -> np.ndarray:
    """Zahl der Aufwärtsschritte von N Pfaden aus Einzelschritten (N·T Gleichverteilte, blockweise)."""
    chunk = max(1, CHUNK_DRAWS // T)
    out = np.empty(N, dtype=np.int64)
    for start in range(0, N, chunk):
        stop = min(N, start + chunk)
        out[start:stop] = (draws.uniform((stop - start, T)) < p).sum(axis=1)
    return out


def ups_prefix(draws: S.Draws, N: int, horizons) -> dict:
    """Aufwärtsschritte von N Pfaden zu mehreren Horizonten: Die Pfade sind Anfänge desselben Pfades (T = 10 ist der Anfang von T = 20 …), das wachsende Beispiel."""
    horizons = sorted(int(t) for t in horizons)
    T_max = horizons[-1]
    chunk = max(1, CHUNK_DRAWS // T_max)
    out = {t: np.empty(N, dtype=np.int64) for t in horizons}
    for start in range(0, N, chunk):
        stop = min(N, start + chunk)
        cum = np.cumsum(draws.uniform((stop - start, T_max)) < P_UP, axis=1)
        for t in horizons:
            out[t][start:stop] = cum[:, t - 1]
    return out


def ups_binomial(draws: S.Draws, N: int, T: int, p: float = P_UP) -> np.ndarray:
    """Aufwärtsschritte von N Pfaden mit einer einzigen Gleichverteilten je Pfad (Quantilfunktion der Binomialverteilung); gleiche Verteilung wie `ups_by_steps`."""
    return S.binomial(draws, N, T, p)


def lognormal_paths(draws: S.Draws, N: int, T: int, sigma: float, mean_factor: float = MEAN_FACTOR) -> np.ndarray:
    """Endwerte von N Pfaden mit lognormalen Faktoren (Erwartungswert je Periode mean_factor): ln v = T·(ln m − σ²/2) + σ·√T·Z, eine Normalverteilte je Pfad."""
    return np.exp(T * (math.log(mean_factor) - 0.5 * sigma * sigma) + sigma * math.sqrt(T) * draws.normal(N))


def critical_sigma(mean_factor: float = MEAN_FACTOR) -> float:
    """Streuung, ab der der typische Pfad schrumpft, obwohl der Erwartungswert je Periode wächst: m·exp(−σ²/2) = 1, also σ = √(2 ln m)."""
    return math.sqrt(2.0 * math.log(mean_factor))


def typical_factor_formula(sigma: float, mean_factor: float = MEAN_FACTOR) -> float:
    """Typischer (geometrischer) Faktor je Periode bei lognormalen Faktoren mit Erwartungswert m: m·exp(−σ²/2)."""
    return mean_factor * math.exp(-0.5 * sigma * sigma)


def typical_factor(values: np.ndarray, T: int) -> float:
    """Typischer Faktor je Periode aus gezogenen Endwerten: (Median des Endwerts)^(1/T), über Logarithmen."""
    return math.exp(float(np.median(np.log(values))) / T)


def share_below_start(values: np.ndarray) -> float:
    """Anteil der Pfade, die unter dem Startwert 1 liegen."""
    return float((np.asarray(values) < 1.0).mean())


def share_above(values: np.ndarray, level: float) -> float:
    """Anteil der Pfade über einer Marke (etwa dem Erwartungswert)."""
    return float((np.asarray(values) > level).mean())


def sample_mean_ratio(values: np.ndarray, T: int, mean_factor: float = MEAN_FACTOR) -> float:
    """Stichprobenmittel der Endwerte im Verhältnis zum Erwartungswert m^T."""
    return float(np.asarray(values).mean() / mean_factor ** T)


def exact_share_below_start(T: int, up: float = UP, down: float = DOWN) -> float:
    """Zweipunkt, p = 1/2: P(Endwert < 1) als Summe der Binomialwahrscheinlichkeiten C(T,k)/2^T über alle k mit up^k·down^(T−k) < 1."""
    return sum(math.comb(T, k) for k in range(T + 1) if k * math.log(up) + (T - k) * math.log(down) < 0.0) / 2.0 ** T


def exact_share_above_expectation(T: int, up: float = UP, down: float = DOWN) -> float:
    """Zweipunkt, p = 1/2: P(Endwert > Erwartungswert) als Binomialsumme."""
    level = T * math.log(P_UP * up + (1.0 - P_UP) * down)
    return sum(math.comb(T, k) for k in range(T + 1) if k * math.log(up) + (T - k) * math.log(down) > level) / 2.0 ** T


def lognormal_share_below_start(T: int, sigma: float, mean_factor: float = MEAN_FACTOR) -> float:
    """Lognormale Faktoren: P(Endwert < 1) = Φ(−T·(ln m − σ²/2)/(σ·√T)) mit der Normalverteilung Φ (über math.erfc)."""
    z = -T * (math.log(mean_factor) - 0.5 * sigma * sigma) / (sigma * math.sqrt(T))
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def exact_quantile(T: int, level: float, up: float = UP, down: float = DOWN) -> float:
    """Zweipunkt, p = 1/2: Quantil des Endwerts zur Stufe `level` aus der aufsummierten Binomialverteilung (kleinstes K mit P(K' ≤ K) ≥ level), ohne Simulation."""
    cdf = np.cumsum(S.binomial_pmf(T, P_UP))
    k = int(np.searchsorted(cdf / cdf[-1], level, side="left"))
    return float(path_value(min(k, T), T, up, down))


def two_point_trajectories(draws: S.Draws, count: int, T: int) -> np.ndarray:
    """`count` Pfade mit Zweipunkt-Faktoren, Form (count, T + 1): Spalte 0 ist der Startwert 1, Spalte t der Wert nach t Perioden."""
    steps = np.where(draws.uniform((count, T)) < P_UP, math.log(UP), math.log(DOWN))
    return np.exp(np.concatenate([np.zeros((count, 1)), np.cumsum(steps, axis=1)], axis=1))


def lognormal_trajectories(draws: S.Draws, count: int, T: int, sigma: float, mean_factor: float = MEAN_FACTOR) -> np.ndarray:
    """`count` Pfade mit lognormalen Faktoren (Erwartungswert je Periode mean_factor), Form (count, T + 1), Spalte 0 gleich 1."""
    steps = (math.log(mean_factor) - 0.5 * sigma * sigma) + sigma * draws.normal((count, T))
    return np.exp(np.concatenate([np.zeros((count, 1)), np.cumsum(steps, axis=1)], axis=1))
