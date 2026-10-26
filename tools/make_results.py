"""Erzeugt data/rgw_results.json (die vorgerechneten Messreihen der Experimente E1 bis E5) aus rgw_experiments.py.

    python tools/make_results.py --part m1          # eine Messreihe nach data/_parts/m1.json (läuft allein, mehrere Teile können parallel laufen)
    python tools/make_results.py --merge            # alle Teile zu data/rgw_results.json zusammenfügen, mit Prüfsumme der Quellen

Teile: m1 (Kosten des falschen Mittels), m2 (Pfade und Stichprobenmittel), m3 (Größenverzerrung und Rückrechnung), m4 (Wie selten ist Simpson),
m5 (Gesamtquote gegen Mittel der Quoten).
Die Prüfsumme (`erzeuger_hash`) deckt rgw_rng, rgw_scenario, rgw_means, rgw_tasks, rgw_growth, rgw_sizebias, rgw_mix, rgw_experiments und dieses Skript ab;
tests/test_results_current.py schlägt fehl, wenn sich eine dieser Dateien nach dem Erzeugen geändert hat."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rgw_constants as C  # noqa: E402
import rgw_experiments as X  # noqa: E402

PARTS_DIR = ROOT / "data" / "_parts"
STAND = "2026-10-26"


def run_part(name: str) -> None:
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    result = X.PARTS[name]()
    out = PARTS_DIR / f"{name}.json"
    out.write_text(json.dumps({"hash": X.source_hash(), "data": result}, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print(f"{name}: fertig in {time.perf_counter() - t0:.0f} s -> {out}", flush=True)


def merge() -> None:
    parts, hashes = {}, set()
    for name in X.PARTS:
        raw = json.loads((PARTS_DIR / f"{name}.json").read_text(encoding="utf-8"))
        parts[name] = raw["data"]
        hashes.add(raw["hash"])
    if hashes != {X.source_hash()}:
        sys.exit(f"Teile sind mit anderem Quellstand erzeugt ({sorted(hashes)} gegen {X.source_hash()}); Teile neu erzeugen")
    meta = {"stand": STAND, "erzeuger_hash": X.source_hash(), "seeds": list(X.SEEDS), "quellen": list(X.SOURCES)}
    (ROOT / "data" / C.RESULTS_FILE).write_text(json.dumps({"meta": meta, **parts}, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print(f"data/{C.RESULTS_FILE} geschrieben", X.source_hash())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=sorted(X.PARTS))
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.part:
        run_part(args.part)
    elif args.merge:
        merge()
    else:
        ap.error("--part oder --merge angeben")
