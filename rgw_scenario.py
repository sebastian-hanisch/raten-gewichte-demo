"""Szenario: der Zufallsstrom und die Verteilungen, aus denen alle Instanzen der Demo gezogen werden (eigener SplitMix64-Strom, `rgw_rng.py`).

`Draws` zieht Zahlen blockweise (numpy, uint64 rechnet modulo 2^64) und liefert exakt den Strom der wörtlich kopierten Klasse `SplitMix64`;
tests/test_scenario.py vergleicht beide. Damit sind alle Instanzen auf jeder Plattform gleich, und Messreihen über viele Instanzen bleiben schnell.

Verteilungen (alle aus gleichverteilten Zahlen des Stroms, ohne `numpy.random`):
* normal (Box-Muller), lognormal exp(σ·Z) mit Median 1,
* Poisson und Binomial über die Quantilfunktion einer exakt aufsummierten Wahrscheinlichkeitstabelle (eine Gleichverteilte je Ziehung),
* geometrisch auf {1, 2, …} (Zahl der Versuche bis zum ersten Erfolg), Pareto mit kleinstem Wert 1,
* Exponential (für die gleichverteilte Zerlegung des Simplex: Dirichlet(1, …, 1))."""

from __future__ import annotations

import math
import zlib

import numpy as np

from rgw_rng import MASK

GAMMA = 0x9E3779B97F4A7C15
TWO_PI = 2.0 * math.pi


class Draws:
    """SplitMix64-Strom, blockweise gezogen: gleicher Seed, gleiche Zahlen wie `rgw_rng.SplitMix64`, aber ohne Python-Schleife."""

    def __init__(self, seed: int):
        self.state = int(seed) & MASK

    def raw(self, count: int) -> np.ndarray:
        """Die nächsten `count` Ausgaben des Stroms als uint64-Feld."""
        count = int(count)
        if count <= 0:
            return np.zeros(0, dtype=np.uint64)
        with np.errstate(over="ignore"):
            z = np.arange(1, count + 1, dtype=np.uint64) * np.uint64(GAMMA) + np.uint64(self.state)
            z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            z = z ^ (z >> np.uint64(31))
        self.state = (self.state + count * GAMMA) & MASK
        return z

    def uniform(self, shape) -> np.ndarray:
        """Gleichverteilt auf [0, 1) mit 53 Bit."""
        size = int(np.prod(shape))
        return ((self.raw(size) >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))).reshape(shape)

    def normal(self, shape) -> np.ndarray:
        """Standardnormal nach Box-Muller: zwei gleichverteilte Zahlen ergeben ein Paar (Kosinus-Wert zuerst)."""
        size = int(np.prod(shape))
        pairs = (size + 1) // 2
        u = self.uniform((2, pairs))
        radius = np.sqrt(-2.0 * np.log(1.0 - u[0]))        # 1 − u liegt in (0, 1]: kein log(0)
        angle = TWO_PI * u[1]
        z = np.concatenate([radius * np.cos(angle), radius * np.sin(angle)])
        return z[:size].reshape(shape)

    def exponential(self, shape) -> np.ndarray:
        """Exponentialverteilt mit Mittel 1: −ln(1 − u)."""
        return -np.log(1.0 - self.uniform(shape))


def cell_seed(base: int, *tags) -> int:
    """Stabiler Seed einer Messzelle aus dem Basis-Seed und beliebig vielen Marken (Zahlen oder Texte); ein Seed gehört zu einer Zelle, nie zu einer Instanz allein."""
    value = int(base)
    for tag in tags:
        part = zlib.crc32(tag.encode("utf-8")) if isinstance(tag, str) else int(round(float(tag) * 1000))
        value = (value * 1_000_003 + part) % (1 << 61)
    return value


# ---------------------------------------------------------------- diskrete Verteilungen über Wahrscheinlichkeitstabellen

def poisson_pmf(lam: float) -> np.ndarray:
    """Wahrscheinlichkeiten P(K = k) der Poisson-Verteilung für k = 0 bis kmax (kmax so, dass der Rest unter 1e-15 liegt), rekursiv in Logarithmen aufgebaut."""
    if lam <= 0:
        raise ValueError("Poisson braucht λ > 0")
    kmax = int(lam + 12.0 * math.sqrt(lam) + 30)
    k = np.arange(kmax + 1)
    log_pmf = -lam + k * math.log(lam) - np.array([math.lgamma(i + 1.0) for i in k])
    return np.exp(log_pmf)


def binomial_pmf(n: int, p: float) -> np.ndarray:
    """Wahrscheinlichkeiten P(K = k) der Binomialverteilung Bin(n, p) für k = 0 bis n (Logarithmen, damit auch n = 200 stabil bleibt)."""
    k = np.arange(n + 1)
    log_pmf = np.array([math.lgamma(n + 1.0) - math.lgamma(i + 1.0) - math.lgamma(n - i + 1.0) for i in k])
    log_pmf = log_pmf + k * math.log(p) + (n - k) * math.log1p(-p)
    return np.exp(log_pmf)


def draw_from_pmf(draws: Draws, shape, pmf: np.ndarray) -> np.ndarray:
    """Ganzzahlige Ziehung 0, 1, 2, … aus einer Wahrscheinlichkeitstabelle (Quantilfunktion der aufsummierten Tabelle; eine Gleichverteilte je Wert)."""
    cdf = np.cumsum(pmf)
    cdf = cdf / cdf[-1]
    return np.searchsorted(cdf, draws.uniform(shape), side="right").clip(0, len(pmf) - 1)


def poisson(draws: Draws, shape, lam: float) -> np.ndarray:
    return draw_from_pmf(draws, shape, poisson_pmf(lam))


def binomial(draws: Draws, shape, n: int, p: float) -> np.ndarray:
    return draw_from_pmf(draws, shape, binomial_pmf(n, p))


def geometric(draws: Draws, shape, p: float) -> np.ndarray:
    """Zahl der Versuche bis zum ersten Erfolg (Werte 1, 2, 3, …), P(K > k) = (1 − p)^k; Mittel 1/p."""
    if not 0.0 < p < 1.0:
        raise ValueError("geometrisch braucht 0 < p < 1")
    u = draws.uniform(shape)
    return 1 + np.floor(np.log1p(-u) / math.log1p(-p)).astype(np.int64)


# ---------------------------------------------------------------- stetige Verteilungen

def lognormal(draws: Draws, shape, sigma: float, scale: float = 1.0) -> np.ndarray:
    """Lognormal mit Median `scale` und Streuung σ der Logarithmen: scale · exp(σ·Z); Mittel scale · exp(σ²/2)."""
    return scale * np.exp(sigma * draws.normal(shape))


def pareto(draws: Draws, shape, alpha: float, scale: float = 1.0) -> np.ndarray:
    """Pareto mit kleinstem Wert `scale` und Tail-Index α: P(X > x) = (scale/x)^α. Mittel endlich ab α > 1, Varianz ab α > 2."""
    if alpha <= 0:
        raise ValueError("Pareto braucht α > 0")
    return scale * (1.0 - draws.uniform(shape)) ** (-1.0 / alpha)
