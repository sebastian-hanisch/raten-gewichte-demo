"""Orakel Gewichtsmix: Aufzählung aller ganzzahligen 2×2×l-Tabellen kleiner Summe mit exakter Bruchrechnung (`fractions`) gegen `reversal_mask`;
dazu die Umkehr in zufälligen Tabellen (Vergleichswert 1/60 für 2×2×2 aus einer Sekundärquelle, nicht selbst hergeleitet und hier nur als Marke benutzt)."""

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest

import rgw_mix as X
import rgw_scenario as S


def _compositions(total, parts):
    """Alle Zerlegungen von `total` in `parts` nichtnegative ganze Zahlen."""
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first,) + rest


def _reference_has_reversal(table):
    """Unabhängig und exakt: Quoten als Brüche, Simpson-Umkehr nach Definition. Tabelle als verschachtelte Zählungen [Schicht][Gruppe][Erfolg, Misserfolg]."""
    layers = len(table)
    rates = []
    for s in range(layers):
        row = []
        for g in range(2):
            succ, fail = table[s][g]
            if succ + fail == 0:
                return False                                    # Quote nicht definiert
            row.append(Fraction(int(succ), int(succ + fail)))
        rates.append(row)
    pooled = []
    for g in range(2):
        succ = sum(int(table[s][g][0]) for s in range(layers))
        tot = sum(int(table[s][g][0] + table[s][g][1]) for s in range(layers))
        pooled.append(Fraction(succ, tot))
    diffs = [r[0] - r[1] for r in rates]
    total = pooled[0] - pooled[1]
    return (all(d > 0 for d in diffs) and total < 0) or (all(d < 0 for d in diffs) and total > 0)


@pytest.mark.parametrize("layers, total", [(2, 7), (2, 9), (3, 5)])
def test_reversal_mask_matches_the_exact_reference_on_all_small_integer_tables(layers, total):
    cells = 4 * layers
    count_reversal = count_all = 0
    for comp in _compositions(total, cells):
        table = np.array(comp, dtype=float).reshape(layers, 2, 2)
        expected = _reference_has_reversal(table.astype(int).tolist())
        assert X.has_reversal(table) == expected, comp
        count_reversal += expected
        count_all += 1
    assert count_all == math.comb(total + cells - 1, cells - 1)
    assert count_reversal >= 0


def test_a_reversal_needs_a_minimum_table_size():
    """Auf allen 2×2×2-Tabellen mit Summe 7 (und weniger) gibt es keine Umkehr, bei 20 gibt es sie (die Tabelle von Hand: 9/10, 54/90, 72/90, 4/10 hat Summe 200)."""
    small = sum(_reference_has_reversal(np.array(c).reshape(2, 2, 2).tolist()) for c in _compositions(7, 8))
    assert small == 0
    witness = [[[9, 1], [72, 18]], [[54, 36], [4, 6]]]
    assert _reference_has_reversal(witness)
    smallest = next((c for total in range(8, 16) for c in _compositions(total, 8) if _reference_has_reversal(np.array(c).reshape(2, 2, 2).tolist())), None)
    assert smallest is not None and sum(smallest) < 16                                       # eine kleine Tabelle mit Umkehr gibt es schon unter der Summe 16
    assert X.has_reversal(np.array(smallest, dtype=float).reshape(2, 2, 2))


def test_random_2x2x2_tables_flip_in_about_one_sixtieth():
    """Zufällige Tabellen mit gleichverteilten Zellwahrscheinlichkeiten (Dirichlet(1, …, 1)): Anteil der Umkehr, Marge 5 Standardfehler. Vergleichswert 1/60, nicht selbst hergeleitet."""
    runs = 400_000
    share = float(X.reversal_mask(X.random_tables(S.Draws(81), runs, 2)).mean())
    se = math.sqrt(share * (1 - share) / runs)
    assert abs(share - 1 / 60) < 5 * se + 1e-4
    assert 0.0155 < share < 0.0180


def test_more_layers_make_the_flip_rarer():
    shares = [float(X.reversal_mask(X.random_tables(S.Draws(82 + l), 200_000, l)).mean()) for l in (2, 3, 4)]
    assert shares[0] > shares[1] > shares[2] > 0


def test_reversal_share_equals_the_enumeration_on_a_rescaled_grid():
    """Gleichverteilte ganzzahlige Zerlegungen von N = 40 in 8 Zellen nähern die Gleichverteilung auf dem Simplex; der Anteil der Umkehr liegt nahe am Zufallsanteil (Marge 0,004)."""
    rng = np.random.default_rng(5)                                         # nur im Test: gleichverteilte Zerlegung über Sternchen und Striche
    N, cells, runs = 40, 8, 40000
    flips = 0
    for _ in range(runs):
        pos = np.sort(rng.choice(N + cells - 1, size=cells - 1, replace=False))            # cells − 1 verschiedene Trennstellen aus N + cells − 1 Plätzen
        parts = np.diff(np.concatenate([[-1], pos, [N + cells - 1]])) - 1
        assert parts.min() >= 0 and parts.sum() == N
        flips += _reference_has_reversal(parts.reshape(2, 2, 2).tolist())
    assert abs(flips / runs - 1 / 60) < 0.004
