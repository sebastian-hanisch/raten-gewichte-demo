"""Die Messdatei ist so aktuell wie ihre Erzeuger: Prüfsumme der Quelldateien (eigene Datei, damit die Mutationsprüfung sie auslassen kann: jede Mutation eines Erzeugers ändert die Prüfsumme)."""

import rgw_experiments as X
import rgw_results as R


def test_results_file_is_current_with_the_generator_sources():
    """Schlägt fehl, sobald sich rgw_rng, rgw_scenario, rgw_means, rgw_tasks, rgw_growth, rgw_sizebias, rgw_mix, rgw_experiments oder tools/make_results.py ändern:
    dann `python tools/make_results.py --part <m1|m2|m3|m4|m5>` und `--merge` neu laufen lassen."""
    assert R.meta()["erzeuger_hash"] == X.source_hash()
    assert R.meta()["quellen"] == list(X.SOURCES) and R.meta()["seeds"] == list(X.SEEDS)
    assert R.is_current()
