"""Vorgerechnete Messreihen (data/rgw_results.json, erzeugt von tools/make_results.py aus rgw_experiments.py): Laden und Abfragen.

Die Datei trägt die Prüfsumme der Quelldateien; `is_current()` vergleicht sie mit dem heutigen Stand (tests/test_results_current.py)."""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path

import rgw_constants as C

PATH = Path(os.environ.get("RGW_RESULTS_PATH") or Path(__file__).resolve().parent / "data" / C.RESULTS_FILE)


@functools.lru_cache(maxsize=1)
def load() -> dict:
    return json.loads(PATH.read_text(encoding="utf-8"))


def meta() -> dict:
    return load()["meta"]


def is_current() -> bool:
    import rgw_experiments as X
    return meta()["erzeuger_hash"] == X.source_hash()


def _one(rows: list[dict], **match) -> dict:
    (row,) = [r for r in rows if all(r[k] == v for k, v in match.items())]
    return row


# E1
def cost_cells() -> list[dict]:
    return load()["m1"]["cells"]


def cost_cell(task: str, sigma: float) -> dict:
    return _one(cost_cells(), task=task, sigma=sigma)


def cost_curve(task: str, sigma: float) -> dict:
    return _one(load()["m1"]["curves"], task=task, sigma=sigma)


# E2
def growth_rows() -> list[dict]:
    return load()["m2"]["growth"]["rows"]


def growth_row(T: int) -> dict:
    return _one(growth_rows(), T=T)


def lognormal_rows() -> list[dict]:
    return load()["m2"]["lognormal"]


def lognormal_row(sigma: float) -> dict:
    return _one(lognormal_rows(), sigma=sigma)


def coverage_rows() -> list[dict]:
    return load()["m2"]["coverage"]


def coverage_row(family: str, T: int, N: int) -> dict:
    return _one(coverage_rows(), family=family, T=T, N=N)


# E3
def bias_rows() -> list[dict]:
    return load()["m3"]["bias"]


def bias_row(family: str, par: float) -> dict:
    return _one(bias_rows(), family=family, par=par)


def back_rows() -> list[dict]:
    return load()["m3"]["back"]


def back_row(family: str, par: float, n: int) -> dict:
    return _one(back_rows(), family=family, par=par, n=n)


def tail_rows() -> list[dict]:
    return load()["m3"]["tail"]


def tail_row(alpha: float) -> dict:
    return _one(tail_rows(), alpha=alpha)


# E4
def reversal_rows() -> list[dict]:
    return load()["m4"]["reversal"]


def reversal_row(layers: int) -> dict:
    return _one(reversal_rows(), layers=layers)


def weight_rows() -> list[dict]:
    return load()["m4"]["weights"]


def weight_row(k: int, tau: float) -> dict:
    return _one(weight_rows(), k=k, tau=tau)


# E5
def coupling_cells() -> list[dict]:
    return load()["m5"]["cells"]


def coupling_cell(kappa_b: float, sigma_d: float) -> dict:
    return _one(coupling_cells(), kappa_b=kappa_b, sigma_d=sigma_d)
