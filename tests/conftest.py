"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b). Die Erwartungswerte stehen im Test als Rechnung.

Handrechnungen:
* **Geschwindigkeiten 50 und 100 km/h, je 10 km:** Gesamtstrecke 20 km, Zeit 10/50 + 10/100 = 0,2 + 0,1 = 0,3 h, Gesamtgeschwindigkeit 20/0,3 = 66,67 km/h (das harmonische Mittel
  2/(1/50 + 1/100)). **Je 1 Stunde:** 150 km in 2 h = 75 km/h (das arithmetische Mittel). Die Gesamtrate ist das mit der Zeit gewichtete arithmetische Mittel:
  (0,2·50 + 0,1·100)/0,3 = 20/0,3 = 66,67.
* **Faktoren 1,5 und 0,6:** arithmetisch (1,5 + 0,6)/2 = 1,05; geometrisch √(1,5·0,6) = √0,9 = 0,948683; harmonisch 2/(1/1,5 + 1/0,6) = 2/(0,666667 + 1,666667) = 0,857143;
  quadratisch √((2,25 + 0,36)/2) = √1,305 = 1,142366. Im Wechsel 1,5, 0,6, 1,5, 0,6: Produkt 0,81 = 0,9².
* **Lose der Größe 1, 1, 4:** Mittel pro Los 6/3 = 2; Stücksicht (ein Stück gleichverteilt unter 6 Stücken): Lose 1, 1 mit je einem Stück, Los 4 mit vier Stücken, Beobachtungen 1, 1, 4, 4, 4, 4,
  Mittel 18/6 = 3; Varianz der Lose ((1−2)² + (1−2)² + (4−2)²)/3 = 2, CV² = 2/4 = 0,5, Faktor 1 + CV² = 1,5 = 3/2. Harmonisches Mittel der Beobachtungen 6/(1 + 1 + 4·0,25) = 6/3 = 2.
* **Tabelle, die kippt** (Spalte Erfolg, Misserfolg; Schicht groß, klein): A groß 9 von 10, A klein 54 von 90; B groß 72 von 90, B klein 4 von 10. Quoten je Schicht: groß 0,9 gegen 0,8, klein 0,6 gegen 0,4,
  A also in beiden Schichten besser. Gesamt: A 63/100 = 0,63, B 76/100 = 0,76: B besser. Gewichte A (0,1; 0,9), B (0,9; 0,1), Gesamtvariationsabstand ½(0,8 + 0,8) = 0,8.
  Mit gemeinsamen Standardgewichten (0,5; 0,5): A (0,9 + 0,6)/2 = 0,75, B (0,8 + 0,4)/2 = 0,60.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def speeds():
    """Geschwindigkeiten 50 und 100 km/h."""
    return np.array([50.0, 100.0])


@pytest.fixture
def factors():
    """Wachstumsfaktoren 1,5 und 0,6."""
    return np.array([1.5, 0.6])


@pytest.fixture
def lots():
    """Lose der Größen 1, 1, 4."""
    return np.array([1.0, 1.0, 4.0])


@pytest.fixture
def simpson_table():
    """Tabelle [Schicht groß, klein][Gruppe A, B][Erfolg, Misserfolg] mit Zählungen."""
    return np.array([[[9, 1], [72, 18]], [[54, 36], [4, 6]]], dtype=float)
