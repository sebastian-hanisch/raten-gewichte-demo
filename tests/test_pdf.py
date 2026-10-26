"""PDF-Export: erzeugt gültige PDF-Bytes mit Umlauten und Sonderzeichen, für jedes Preset und jede Aufgabe."""

import rgw_constants as C
import rgw_evaluation as V
import rgw_pdf_export as P


def _pdf(settings):
    a = V.analyse(settings)
    if isinstance(a, V.RatesAnalysis):
        verdict = V.verdict_rates(settings.task, a.truth, a.means, settings.r, a.selected)
    elif isinstance(a, V.LotsAnalysis):
        verdict = V.verdict_lots(a.truth, a.arithmetic, a.harmonic, a.factor, settings.m)
    else:
        verdict = V.verdict_table(a.rates, a.pooled, a.standard, a.tv, a.reversal)
    return P.build_pdf(a, settings, [verdict])


def test_pdf_for_every_preset_is_a_pdf():
    for name, preset in C.PRESETS.items():
        data = _pdf(V.Settings(**preset))
        assert data[:5] == b"%PDF-" and len(data) > 1500, name


def test_pdf_at_the_regler_extremes():
    for task in C.TASK_KEYS:
        for edge in (dict(m=C.M_MIN, sigma=C.SIGMA_MIN, r=C.R_MIN), dict(m=C.M_MAX, sigma=C.SIGMA_MAX, r=C.R_MAX)):
            assert _pdf(V.Settings(task=task, **edge))[:5] == b"%PDF-"


def test_latin_replaces_characters_the_core_fonts_do_not_know():
    assert P._latin("R ≤ √(3/n) – σ = 0,3 · τ ½ **fett** ∞ ± 1 κ α λ") == "R <= Wurzel(3/n) - sigma = 0,3 * tau 1/2 fett unendlich +- 1 kappa alpha lambda"
    assert P._latin("Größe ä ö ü ß 10⁻⁵") == "Größe ä ö ü ß 10-5"


def test_result_lines_by_hand():
    a = V.analyse(V.Settings(task="tabelle"))
    lines = P.result_lines(a)
    assert lines[0] == "große Standorte: Gruppe A 90 %, Gruppe B 80 %" and "Gesamtquote: Gruppe A 63 %, Gruppe B 76 %" in lines
    assert "Mit gemeinsamen Gewichten (1/2, 1/2): Gruppe A 75 %, Gruppe B 60 %" in lines and lines[-1] == "Gesamtvariationsabstand der Gewichte: 0,8"
    rates = P.result_lines(V.analyse(V.Settings(task="zeit")))
    assert rates[0].startswith("Wahre Gesamtgröße: ") and sum("(richtig für diese Aufgabe)" in l for l in rates) == 1 and any("Arithmetisch (r = 1)" in l and "richtig" in l for l in rates)
    lots = P.result_lines(V.analyse(V.Settings(task="lose", m=30)))
    assert lots[1] == "Beobachtete Stücke: 30" and lots[2].startswith("Arithmetisches Mittel der Beobachtungen: ")
