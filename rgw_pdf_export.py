"""PDF-Export (fpdf2, Kernschriften): Einstellungen, Zahlen der gewählten Aufgabe und die Verdikte. Erst nach Klick erzeugt (DEMO-PLAYBOOK §1c)."""

from __future__ import annotations

from fpdf import FPDF

import rgw_constants as C
import rgw_evaluation as V

_REPLACE = {"–": "-", "—": "-", "≤": "<=", "≥": ">=", "√": "Wurzel", "‖": "|", "²": "2", "Σ": "Summe", "°": " Grad", "·": "*", "−": "-", "„": '"', "“": '"',
            "ε": "eps", "ρ": "rho", "σ": "sigma", "τ": "tau", "κ": "kappa", "λ": "lambda", "α": "alpha", "∞": "unendlich", "±": "+-", "→": "->", "μ": "mu", "Δ": "Delta", "½": "1/2",
            "⁻": "-", "⁰": "0", "¹": "1", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}


def _latin(text: str) -> str:
    """Kernschriften kennen nur Latin-1: Sonderzeichen ersetzen, Markdown-Sterne entfernen."""
    for a, b in _REPLACE.items():
        text = text.replace(a, b)
    return text.replace("**", "").replace("`", "").encode("latin-1", "replace").decode("latin-1")


def result_lines(a) -> list[str]:
    """Die Zahlen der Hauptansicht als Zeilen: Mittel und Abweichungen bei Raten, Mittel pro Los und Rückrechnung bei Losen, Quoten bei der Tabelle."""
    if isinstance(a, V.RatesAnalysis):
        lines = [f"Wahre Gesamtgröße: {V.de(a.truth, 3)} {a.task.unit}"]
        for row in V.mean_rows(a):
            mark = " (richtig für diese Aufgabe)" if row["richtig"] else ""
            lines.append(f"{row['Mittel']} (r = {V.de(row['r'], 0)}): {V.de(row['Wert'], 3)}, Abweichung {V.de(row['Abweichung (%)'], 1)} %{mark}")
        lines.append(f"Gewähltes r = {V.de(a.settings.r, 1)}: {V.de(a.selected, 3)}")
        return lines
    if isinstance(a, V.LotsAnalysis):
        return [f"Mittel pro Los: {V.de(a.truth, 2)} Stück (Grundgesamtheit von {len(a.sizes)} Losen)",
                f"Beobachtete Stücke: {len(a.observed)}",
                f"Arithmetisches Mittel der Beobachtungen: {V.de(a.arithmetic, 2)} ({V.signed_pct(a.arithmetic / a.truth - 1)})",
                f"Harmonisches Mittel der Beobachtungen: {V.de(a.harmonic, 2)} ({V.signed_pct(a.harmonic / a.truth - 1)})",
                f"Verzerrungsfaktor 1 + CV^2 der Grundgesamtheit: {V.de(a.factor, 2)}"]
    return [f"{V.LAYER_NAMES[i]}: Gruppe A {V.pct(a.rates[i, 0], 0)} %, Gruppe B {V.pct(a.rates[i, 1], 0)} %" for i in range(2)] + [
        f"Gesamtquote: Gruppe A {V.pct(a.pooled[0], 0)} %, Gruppe B {V.pct(a.pooled[1], 0)} %",
        f"Mit gemeinsamen Gewichten (1/2, 1/2): Gruppe A {V.pct(a.standard[0], 0)} %, Gruppe B {V.pct(a.standard[1], 0)} %",
        f"Gesamtvariationsabstand der Gewichte: {V.de(a.tv, 1)}"]


def build_pdf(a, settings: V.Settings, verdicts: list[tuple[str, str]]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _latin("Raten, Gewichte, Größenverzerrung"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _latin(f"Aufgabe: {C.TASK_LABELS[settings.task]}. Posten bzw. Beobachtungen m = {settings.m}, Streuung sigma = {V.de(settings.sigma, 2)}, r = {V.de(settings.r, 1)}, Zufalls-Seed {settings.seed}."))
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Zahlen", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for line in result_lines(a):
        pdf.multi_cell(0, 5.5, _latin(line), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    for _, text in verdicts:
        pdf.multi_cell(0, 5, _latin(text), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _latin("Alle Daten sind erzeugt (lognormale Raten, Lose, ein Handbeispiel); die Aufgaben sind gewählt, keine Messung an echten Daten."))
    return bytes(pdf.output())
