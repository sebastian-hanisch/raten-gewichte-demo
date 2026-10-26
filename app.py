"""Raten, Gewichte, Größenverzerrung: Welches Mittel gehört wozu? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 7 der Reihe „Mittelwerte und Lagemaße“ (optional). Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import rgw_constants as C
import rgw_evaluation as V
import rgw_experiments as X
import rgw_results as R
import rgw_sizebias as Z
import rgw_tasks as T
from rgw_layout import dunkles_design, kopf_css
from rgw_pdf_export import build_pdf
from rgw_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from rgw_visualization import (
    build_back_chart,
    build_bias_chart,
    build_cost_by_r,
    build_cost_by_sigma,
    build_coupling_chart,
    build_coverage_chart,
    build_error_bars,
    build_lognormal_chart,
    build_lots,
    build_paths_chart,
    build_rate_curve,
    build_reversal_chart,
    build_table_chart,
    build_tail_chart,
    build_weights_chart,
)

st.set_page_config(page_title="Raten, Gewichte, Größenverzerrung – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return V.analyse(settings)


@st.cache_data(show_spinner=False)
def _cost_live(task, sigma, m, instances):
    return X.cost_cell(task, sigma, m, instances)


@st.cache_data(show_spinner=False)
def _growth_live(family, T_, N, sigma, seed):
    return X.growth_live(family, T_, N, sigma, seed)


@st.cache_data(show_spinner=False)
def _back_live(family, par, n, reps, seed):
    return X.back_live(family, par, n, reps, seed)


@st.cache_data(show_spinner=False)
def _reversal_live(layers, tables, seed):
    return X.reversal_live(layers, tables, seed)


@st.cache_data(show_spinner=False)
def _coupling_live(kappa_b, sigma_d, comparisons, seed):
    return X.coupling_cell(kappa_b, sigma_d, comparisons, seeds=(seed,))


def _verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


def _fmt(x, d=2):
    return V.de(x, d)


def _md(text):
    """Kennzahl-Differenz mit Bindestrich als Minuszeichen: Streamlit zeigt den Pfeil nach unten nur bei einem führenden „-“, nicht bei „−“."""
    return text.replace("−", "-")


def _de_label(x):
    """Zahl als Auswahl-Beschriftung mit Dezimalkomma (0.5 → 0,5)."""
    return f"{x:g}".replace(".", ",")


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in rgw_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("⚖️ Raten, Gewichte, Größenverzerrung: Welches Mittel gehört wozu?")
st.markdown(
    """
**Stück 7 der Reihe „Mittelwerte und Lagemaße“ (optional).** Die Reihe fragt, welche Mitte zu welcher Frage passt. Hier geht es um Zahlen, die **durch etwas geteilt** sind (km/h, l/100 km, Wachstumsfaktor je Periode, Stück je Los, Quote je Standort).
Rote Linie: **Die Gesamtgröße ist das gewichtete Mittel der Einzelwerte, und die Nenner sind die Gewichte.** Das Beispiel: 50 und 100 km/h über je 10 km sind **66,67 km/h**, nicht 75 (Gesamtstrecke 20 km durch Gesamtzeit 0,3 h); über je eine Stunde sind es 75.
Je nach Frage legt der Nenner fest, welches Mittel (harmonisch, geometrisch, arithmetisch) die richtige Gesamtgröße liefert. Gemessen wird, **was das falsche Mittel kostet**, warum **Wachstumspfade** dem Erwartungswert nicht folgen, wie sich **Größenverzerrung** zurückrechnen lässt und wie oft eine **Gesamtquote gegen alle Teilquoten** kippt.
Alle Aufgaben sind **gewählt, keine Messung an echten Daten**.
Abgrenzung: [bunching-demo](https://github.com/sebastian-hanisch/bunching-demo) (Wartezeit pro Fahrgast statt pro Bus, die Formel 1 + CV² steht dort), [konfundierung-demo](https://github.com/sebastian-hanisch/konfundierung-demo) (warum eine gemeinsame Ursache das Vorzeichen kippt),
`lp-mitte-demo` (Stück 3: dort ist **p** der Exponent der Strecke; hier heißt der Exponent des Mittels **r**) und `mittelwert-median-demo` (Wurzel der Reihe: Mittel und Median).

Weiter unten: fünf Experimente (**🔬 E1 bis E5**), **🚧 Wo die Annahmen enden** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("Was ein Mittel ist", expanded=True):
    st.markdown(
        r"""
1. **Das Potenzmittel $M_r$** der Werte $x_1, \dots, x_n$ mit Gewichten $w_i$ (Summe 1) ist $\big(\sum_i w_i x_i^{\,r}\big)^{1/r}$; $r = 1$ ist das arithmetische, $r = 0$ (als Grenzwert) das geometrische, $r = -1$ das harmonische, $r = 2$ das quadratische Mittel.
   Es wächst mit $r$: harmonisch ≤ geometrisch ≤ arithmetisch ≤ quadratisch. Der Buchstabe ist hier **$r$**, nicht $p$.
2. **Jedes Mittel einer Rate ist ein Gewichtungsproblem.** Die Gesamtrate ist Summe der Zähler durch Summe der Nenner, das ist das mit den Nennern gewichtete arithmetische Mittel der Raten. Ist der Nenner bei allen Posten gleich, ist es das einfache arithmetische Mittel;
   steckt der Nenner im Kehrwert der Rate (Strecke fest, Zeit = Strecke/Geschwindigkeit), ist es das harmonische.
3. **Wachstum multipliziert.** Das Produkt der Faktoren ist die $T$-te Potenz des geometrischen Mittels. Das arithmetische Mittel der Faktoren beschreibt den Erwartungswert, nicht den typischen Verlauf.

| Mittel | $r$ | Richtig für | Falsch für |
|---|---|---|---|
| Harmonisch | −1 | Geschwindigkeit über **gleiche Strecke**; km/l über gleiche Strecke | Geschwindigkeit über gleiche Zeit; l/100 km bei gleicher Strecke |
| Geometrisch | 0 | **Wachstumsfaktoren** (Produkt); Verhältnisse | Summen und Mengen |
| Arithmetisch | 1 | Geschwindigkeit über **gleiche Zeit**; l/100 km bei gleicher Strecke; Mengen | Geschwindigkeit über gleiche Strecke; Wachstumsfaktoren; Stücksicht |
| Quadratisch | 2 | Streuungsmaße (Wurzel des mittleren Quadrats) | die Gesamtgröße einer Rate |

Der Median steht in [mittelwert-median-demo](https://github.com/sebastian-hanisch/mittelwert-median-demo), die Lp-Mitte mit Exponent $p$ für Punkte in der Ebene in `lp-mitte-demo` (Stück 3).
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    task = st.selectbox("Aufgabe", C.TASK_KEYS, format_func=C.TASK_LABELS.get, key="task_select",
                        help="Welche Größe wird gemittelt? Die Aufgabe legt fest, was der Nenner ist und damit, welches Mittel die Gesamtgröße liefert.")
    m = st.slider("Posten m", *bounds("m_slider"), key="m_slider", disabled=task == "tabelle",
                  help="Fahrten, Fahrzeuge oder Perioden bei den Raten; bei den Losgrößen die Zahl der beobachteten Stücke. Die Tabelle ist ein festes Beispiel.")
    sigma = st.slider("Streuung σ", *bounds("sigma_slider"), key="sigma_slider", step=0.05, disabled=task in ("wechsel", "tabelle"),
                      help="Streuung der Logarithmen der Einzelraten bzw. der Losgrößen (lognormal). Je größer, desto mehr kostet das falsche Mittel. Die Faktoren 1,5/0,6 und die Tabelle sind fest.")
    r = st.slider("Mittelungsexponent r", *bounds("r_slider"), key="r_slider", step=0.5, disabled=task not in C.RATE_TASK_KEYS,
                  help="Das Potenzmittel M_r mit diesem Exponenten wird in der Kurve markiert: −1 harmonisch, 0 geometrisch, 1 arithmetisch, 2 quadratisch. Nur bei den Raten.")
    st.number_input("Zufalls-Seed der Daten", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Daten generieren", width="stretch", on_click=randomize_seed)

settings = V.Settings(task=str(task), m=int(m), sigma=float(sigma), r=float(r), seed=int(st.session_state["seed_input"]))
sync_query_params({k: st.session_state[k] for k in ("task_select", "m_slider", "sigma_slider", "r_slider", "seed_input")})

a = _analysis(settings)

# --- Hauptansicht --------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Welches Mittel gehört zu dieser Aufgabe?")
st.markdown(f"**Aufgabe:** {C.TASK_LABELS[settings.task]}")

if isinstance(a, V.RatesAnalysis):
    digits = 3 if a.task.unit == "Faktor" else 2
    unit = "" if a.task.unit == "Faktor" else f" {a.task.unit}"
    m1, m2, m3 = st.columns(3)
    m1.metric("Wahre Gesamtgröße", f"{_fmt(a.truth, digits)}{unit}", delta="aus den Rohgrößen gerechnet", delta_color="off")
    m2.metric(f"Gewähltes Mittel M_r (r = {_fmt(settings.r, 1)})", f"{_fmt(a.selected, digits)}{unit}", delta=f"{_md(V.signed_pct(a.selected / a.truth - 1.0))} gegen die wahre Gesamtgröße", delta_color="off")
    m3.metric("Richtiges Mittel für diese Aufgabe", a.task.right_name.capitalize(), delta=f"r = {_fmt(a.task.right_r, 0)}", delta_color="off")
    st.plotly_chart(build_rate_curve(a), width="stretch", key="main_chart")
    st.caption("Links die Einzelraten (grau). Die blaue Kurve ist das Potenzmittel $M_r$ über dem Exponenten $r$; die vier Rauten sind harmonisch, geometrisch, arithmetisch, quadratisch. "
               "Die gestrichelte Linie ist die wahre Gesamtgröße, aus Zähler und Nenner gerechnet: Wo die Kurve sie schneidet, liegt das richtige Mittel.")
    rows = V.mean_rows(a)
    t1, t2 = st.columns([3, 2])
    t1.dataframe(rows, hide_index=True, width="stretch", column_order=["Mittel", "r", "Wert", "Abweichung (%)", "richtig"],
                 column_config={"r": st.column_config.NumberColumn(format="%d"), "Wert": st.column_config.NumberColumn(format="%.3f"), "Abweichung (%)": st.column_config.NumberColumn(format="%.1f %%"),
                                "richtig": st.column_config.CheckboxColumn("richtig für die Aufgabe")})
    t2.plotly_chart(build_error_bars(rows), width="stretch", key="main_errors")
    with st.expander("Die Einzelposten ansehen"):
        if a.num is None:
            detail = [{"Periode": i + 1, "Faktor": float(a.x[i])} for i in range(len(a.x))]
        else:
            nums, dens = {"strecke": ("Strecke (km)", "Zeit (h)"), "zeit": ("Strecke (km)", "Zeit (h)"), "verbrauch": ("Liter", "Strecke (je 100 km)")}[settings.task]
            detail = [{"Nr.": i + 1, a.task.unit: float(a.x[i]), nums: float(a.num[i]), dens: float(a.den[i])} for i in range(len(a.x))]
        st.dataframe(detail, hide_index=True, width="stretch")
        st.caption("Zähler und Nenner sind die Rohgrößen; die Gesamtrate ist die Summe der Zähler durch die Summe der Nenner." if a.num is not None else
                   "Das Produkt aller Faktoren ist der Gesamtfaktor; der gleichwertige Faktor je Periode ist die m-te Wurzel daraus, das geometrische Mittel.")
    main_verdict = V.verdict_rates(settings.task, a.truth, a.means, settings.r, a.selected)
    pdf_verdicts = [main_verdict]
elif isinstance(a, V.LotsAnalysis):
    m1, m2, m3 = st.columns(3)
    m1.metric("Mittel pro Los (wahr)", f"{_fmt(a.truth, 1)} Stück", delta=f"Grundgesamtheit von {len(a.sizes)} Losen", delta_color="off")
    m2.metric("Arithmetisches Mittel der Beobachtungen", f"{_fmt(a.arithmetic, 1)} Stück", delta=f"{_md(V.signed_pct(a.arithmetic / a.truth - 1.0))} gegen das Mittel pro Los", delta_color="off")
    m3.metric("Harmonisches Mittel der Beobachtungen", f"{_fmt(a.harmonic, 1)} Stück", delta=f"{_md(V.signed_pct(a.harmonic / a.truth - 1.0))} gegen das Mittel pro Los", delta_color="off")
    st.plotly_chart(build_lots(a), width="stretch", key="main_chart")
    st.caption(f"Grün: die Losgrößen der Grundgesamtheit (Stück je Los, lognormal mit Median {Z.LOGNORMAL_SCALE:g} und Streuung σ). Violett: die Größe des Loses, aus dem ein gezogenes Stück stammt ({settings.m} Stück); große Lose kommen häufiger vor, weil sie mehr Stücke haben. "
               "Die senkrechten Linien: Mittel pro Los (schwarz), arithmetisches (rot) und harmonisches Mittel (violett) der Beobachtungen. In der Fahrgastbefragung entspricht ein Los einem Fahrzeug und ein Stück einem Fahrgast.")
    main_verdict = V.verdict_lots(a.truth, a.arithmetic, a.harmonic, a.factor, settings.m)
    pdf_verdicts = [main_verdict]
else:
    m1, m2, m3 = st.columns(3)
    m1.metric("Gesamtquote Gruppe A", f"{V.pct(a.pooled[0], 0)} %", delta=f"{V.pct(a.pooled[0] - a.pooled[1], 0)} Prozentpunkte gegen B")      # mein Wert minus Referenz; höher ist besser, also normale Färbung
    m2.metric("Gesamtquote Gruppe B", f"{V.pct(a.pooled[1], 0)} %", delta="Referenz", delta_color="off")
    m3.metric("Gesamtvariationsabstand der Gewichte", _fmt(a.tv, 1), delta="0 = gleiche Gewichte, 1 = getrennt", delta_color="off")
    st.plotly_chart(build_table_chart(a), width="stretch", key="main_chart")
    cells = []
    for i, layer in enumerate(V.LAYER_NAMES):
        for g, group in enumerate(("A", "B")):
            cells.append({"Schicht": layer, "Gruppe": group, "Erfolge": int(a.table[i, g, 0]), "Gesamt": int(a.table[i, g].sum()), "Quote (%)": 100.0 * float(a.rates[i, g]), "Gewicht der Schicht (%)": 100.0 * float(a.weights[i, g])})
    st.dataframe(cells, hide_index=True, width="stretch", column_config={"Quote (%)": st.column_config.NumberColumn(format="%.0f %%"), "Gewicht der Schicht (%)": st.column_config.NumberColumn(format="%.0f %%")})
    st.caption("Die Gesamtquote einer Gruppe ist die Summe ihrer Quoten, gewichtet mit dem Anteil ihrer Fälle in der jeweiligen Schicht. Ein Standort mit einem Servicegrad (Erfolg = pünktlich geliefert) steht für einen Fall. "
               "Mit gemeinsamen Gewichten (½, ½) für beide Gruppen sind die Gesamtquoten Mittel derselben Gewichte der Schichtquoten, und A liegt vorn.")
    main_verdict = V.verdict_table(a.rates, a.pooled, a.standard, a.tv, a.reversal)
    pdf_verdicts = [main_verdict]

# PDF erst auf Klick
pdf_key = repr(settings)
if st.button("📄 PDF erstellen", key="pdf_make"):
    st.session_state["pdf"] = (pdf_key, build_pdf(a, settings, pdf_verdicts))
if st.session_state.get("pdf", (None, None))[0] == pdf_key:
    st.download_button("⬇️ PDF herunterladen", data=st.session_state["pdf"][1], file_name="raten-gewichte.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Welches Mittel gehört zu welchem „pro“?")
_verdict(main_verdict)

with st.expander("🧰 Die vier Mittel erklärt"):
    st.markdown(
        r"""
| Mittel | Was es tut | Wann es die Gesamtgröße ist | Wer setzt an |
|---|---|---|---|
| **Harmonisch** ($r = -1$) | Kehrwert des Mittels der Kehrwerte | Raten über **gleiche Strecke** (Zeit = Strecke/Geschwindigkeit): Gesamtstrecke durch Gesamtzeit | Rückrechnung der Stücksicht auf das Mittel pro Los (E3) |
| **Geometrisch** ($r = 0$) | Wurzel aus dem Produkt | **Wachstumsfaktoren**: Produkt = (geometrisches Mittel)$^T$ | typischer Pfad statt Erwartungswert (E2) |
| **Arithmetisch** ($r = 1$) | Summe durch Zahl | Raten über **gleiche Zeit**; l/100 km bei gleicher Strecke; jede Gesamtgröße mit konstantem Nenner | falsch, sobald die Nenner verschieden sind (E1, E5) |
| **Quadratisch** ($r = 2$) | Wurzel aus dem mittleren Quadrat | kein Mittel einer Rate; Streuungsmaß | Standardabweichung, Effektivwert |
| **Gewichtetes arithmetisches** | Summe der Zähler durch Summe der Nenner | **jede** Gesamtrate; mit gleichen Nennern das einfache arithmetische Mittel | Gesamtquote gegen Mittel der Quoten (E5) |
        """
    )

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.caption(f"Die Messreihen der Experimente sind vorgerechnet (vier feste Zufalls-Seeds je Zelle, Stand {R.meta()['stand']}); die Knöpfe „Nachmessen“ rechnen neu und nehmen dafür Ihre Einstellungen.")

# E1 ---------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 E1: Was kostet das falsche Mittel?")
st.markdown(f"Je Aufgabe und Streuung {V.de_int(X.E1_INSTANCES)} Instanzen mit {X.E1_M} Posten (lognormale Einzelraten, Streuung σ der Logarithmen), die vier Mittel auf denselben Instanzen. "
            "Gemessen: die mittlere Abweichung des Mittels von der wahren Gesamtgröße in Prozent. Das **richtige** Mittel hat die Abweichung null (eine Rechenidentität, kein Befund); "
            "der Befund ist der Preis der anderen. „Verbrauch“ ist dieselbe Mathematik wie „gleiche Zeit“.")
e1_task = st.selectbox("Aufgabe", X.E1_TASKS, format_func=lambda k: T.RATE_TASKS[k].label, key="e1_task")
e1_cells = sorted([c for c in R.cost_cells() if c["task"] == e1_task], key=lambda c: c["sigma"])
c1, c2 = st.columns(2)
c1.plotly_chart(build_cost_by_sigma(e1_cells, T.RATE_TASKS[e1_task].label), width="stretch", key="e1_sigma")
c2.plotly_chart(build_cost_by_r([c for c in R.load()["m1"]["curves"] if c["task"] == e1_task], e1_task), width="stretch", key="e1_r")
st.caption("Links: mittlere Abweichung der vier Mittel über der Streuung. Rechts: dieselbe Abweichung für das Potenzmittel $M_r$ über $r$ bei drei Streuungen; das Minimum (null) liegt beim richtigen Mittel der Aufgabe.")
e1_sigma = st.select_slider("Streuung σ", options=list(X.E1_SIGMAS), value=0.5, key="e1_sigma_pick", format_func=_de_label)
_verdict(V.verdict_e1(R.cost_cell(e1_task, e1_sigma)))
live_key = (e1_task, settings.sigma, settings.m, X.E1_INSTANCES)
if st.button(f"Nachmessen mit σ = {_fmt(settings.sigma, 2)} und m = {settings.m} aus der Seitenleiste ({V.de_int(X.E1_INSTANCES)} Instanzen je Seed, vier Seeds, unter einer Sekunde)", key="e1_live"):
    with st.spinner("Rechne vier Seeds …"):
        st.session_state["e1_result"] = (live_key, _cost_live(*live_key))
res = st.session_state.get("e1_result")
if res is not None and res[0] == live_key:
    _verdict(V.verdict_e1(res[1]))
elif res is not None:
    st.caption("Die Einstellungen haben sich geändert: Knopf erneut drücken.")

# E2 ---------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 E2: Wachstumspfade und das Stichprobenmittel")
st.markdown("Der Bedarf wächst je Periode um den Faktor 1,5 oder fällt auf 0,6, beides gleich wahrscheinlich: Erwartungswert 1,05 je Periode, geometrisches Mittel $\\sqrt{0{,}9} = 0{,}9487$. "
            "Im Wechsel 1,5, 0,6, 1,5, … steht der Bedarf nach zwei Perioden bei 0,9. Gemessen wird der Zufallspfad: wo die Pfade stehen und ob das **Stichprobenmittel** den Erwartungswert trifft. "
            f"{V.de_int(X.E2_N)} Pfade je Seed; die Horizonte sind Anfänge desselben Pfades.")
horizon = st.select_slider("Perioden bis T", options=list(X.E2_HORIZONS), value=100, key="e2_horizon")
live_paths = st.session_state.get("e2_result")
st.plotly_chart(build_paths_chart(horizon, R.growth_rows(), live_paths[1] if live_paths is not None and live_paths[1]["family"] == "zweipunkt" and live_paths[1]["T"] >= 1 else None), width="stretch", key="e2_paths")
st.caption("Blaues Band: 10 % bis 90 % der Pfade (exakt aus der Binomialverteilung), blaue Linie: Median. Rot gestrichelt: der Erwartungswert 1,05^T. Orange Punkte: Stichprobenmittel von je 100.000 Pfaden, ein Punkt je Seed. "
           "Die y-Achse ist logarithmisch; die Pfade der Live-Messung (grau) erscheinen, wenn sie die Zweipunkt-Faktoren nutzt.")
row_pick = st.select_slider("Horizont T für das Verdikt", options=list(X.E2_HORIZONS), value=100, key="e2_pick")
_verdict(V.verdict_e2_paths(R.growth_row(row_pick), X.E2_N))

st.markdown("**Mehr Streuung statt Zweipunkt:** lognormale Faktoren mit festem Erwartungswert 1,05 je Periode; nach 50 Perioden.")
st.plotly_chart(build_lognormal_chart(R.lognormal_rows()), width="stretch", key="e2_lognormal")
sig_options = [r_["sigma"] for r_ in sorted(R.lognormal_rows(), key=lambda r_: r_["sigma"])]
sig_pick = st.select_slider("Streuung σ der Faktoren", options=sig_options, value=0.5, key="e2_sigma_pick", format_func=lambda x: f"{x:.4g}".replace(".", ","))
_verdict(V.verdict_e2_lognormal(R.lognormal_row(sig_pick)))

st.markdown("**Wie oft trifft das Stichprobenmittel?** Anteil der Stichproben (je N Pfade), deren Mittel den Erwartungswert auf ± 20 % trifft.")
fam_pick = st.selectbox("Pfadfamilie", ["zweipunkt", "lognormal"], format_func=lambda k: V.FAMILY_LABELS[k] + (f", σ = {X.E2_COVER_SIGMA:g}".replace(".", ",") if k == "lognormal" else ""), key="e2_family")
st.plotly_chart(build_coverage_chart(R.coverage_rows(), fam_pick), width="stretch", key="e2_coverage")
k1, k2 = st.columns(2)
cov_t = k1.select_slider("Perioden T", options=list(X.E2_COVER_T), value=60, key="e2_cov_t")
cov_n = k2.select_slider("Pfade je Stichprobe N", options=list(X.E2_COVER_N), value=10_000, key="e2_cov_n", format_func=V.de_int)
_verdict(V.verdict_e2_coverage(R.coverage_row(fam_pick, cov_t, cov_n)))

st.markdown("**Nachmessen:**")
l1, l2, l3 = st.columns(3)
live_family = l1.selectbox("Familie", ["zweipunkt", "lognormal"], format_func=lambda k: V.FAMILY_LABELS[k], key="e2_live_family",
                           help="Lognormal nimmt die Streuung σ der Seitenleiste.")
live_t = l2.select_slider("Perioden T", options=list(C.LIVE_PATH_T), value=50, key="e2_live_t")
live_n = l3.select_slider("Pfade N", options=list(C.LIVE_PATH_N), value=10_000, key="e2_live_n", format_func=V.de_int)
n_eff = int(min(live_n, C.LIVE_PATH_BUDGET // live_t)) if live_family == "zweipunkt" else int(live_n)
e2_key = (live_family, int(live_t), n_eff, float(settings.sigma) if live_family == "lognormal" else 0.0, settings.seed)
if n_eff < live_n:
    st.caption(f"Bei {live_t} Perioden begrenzt das Rechenbudget die Pfade auf {V.de_int(n_eff)} (N · T höchstens {V.de_int(C.LIVE_PATH_BUDGET)} Zufallszahlen).")
if st.button("Pfade nachmessen (Zufalls-Seed der Seitenleiste, einige Sekunden)", key="e2_live"):
    with st.spinner("Ziehe Pfade …"):
        st.session_state["e2_result"] = (e2_key, _growth_live(*e2_key))
    st.rerun()      # das Diagramm oben zeichnet die Pfade der Live-Messung erst im nächsten Lauf
res = st.session_state.get("e2_result")
if res is not None and res[0] == e2_key:
    _verdict(V.verdict_e2_live(res[1]))
elif res is not None:
    st.caption("Die Einstellungen haben sich geändert: Knopf erneut drücken.")

# E3 ---------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 E3: Von der Befragung auf das Los zurückrechnen")
st.markdown("Wer ein Stück zieht (einen Fahrgast, einen Kunden, ein Teil aus dem Lager) und nach der Größe seines Loses fragt, sieht große Lose häufiger, im Verhältnis zu ihrer Größe. "
            "Der Faktor $1 + \\mathrm{CV}^2$ ist die Wartezeit-Formel aus `bunching-demo` und wird hier nur gemessen, nicht hergeleitet. Neu ist die **Rückrechnung**: "
            "Das **harmonische** Mittel der beobachteten Größen ist das Mittel pro Los, solange es keine leeren Lose gibt (leere Lose werden nie beobachtet).")
st.plotly_chart(build_bias_chart(R.bias_rows()), width="stretch", key="e3_bias")
bias_labels = [f"{V.family_name(f, p)}" for f, p in X.E3_BIAS_FAMILIES]
bias_pick = st.selectbox("Familie der Losgrößen", range(len(X.E3_BIAS_FAMILIES)), format_func=lambda i: bias_labels[i], key="e3_bias_pick")
_verdict(V.verdict_e3_factor(R.bias_row(*X.E3_BIAS_FAMILIES[bias_pick])))
back_labels = [f"{V.family_name(f, p)}" for f, p in X.E3_BACK_FAMILIES]
back_pick = st.selectbox("Rückrechnung für", range(len(X.E3_BACK_FAMILIES)), index=1, format_func=lambda i: back_labels[i], key="e3_back_pick")
back_family, back_par = X.E3_BACK_FAMILIES[back_pick]
st.plotly_chart(build_back_chart([r_ for r_ in R.back_rows() if r_["family"] == back_family and r_["par"] == back_par]), width="stretch", key="e3_back")
back_n = st.select_slider("Beobachtete Stücke n", options=list(X.E3_NS), value=100, key="e3_back_n", format_func=V.de_int)
_verdict(V.verdict_e3_back(R.back_row(back_family, back_par, back_n)))
st.markdown(f"**Schwere Tails:** Pareto-Losgrößen mit Tail-Index $\\alpha$. Für $\\alpha > 2$ gibt es das Mittel der Stücksicht $(\\alpha-1)/(\\alpha-2)$, für $\\alpha \\le 2$ nicht.")
st.plotly_chart(build_tail_chart(R.tail_rows()), width="stretch", key="e3_tail")
tail_pick = st.select_slider("Tail-Index α", options=list(X.E3_ALPHAS), value=2.0, key="e3_tail_pick", format_func=_de_label)
_verdict(V.verdict_e3_tail(R.tail_row(tail_pick)))
n_live = st.select_slider("Beobachtete Stücke n der Live-Messung", options=list(X.E3_NS), value=100, key="e3_live_n", format_func=V.de_int)
reps_live = int(max(20, min(C.LIVE_BACK_REPS, C.LIVE_BACK_BUDGET // n_live)))
back_key = (back_family, float(back_par), int(n_live), reps_live, settings.seed)
if st.button(f"Rückrechnung nachmessen (Familie oben, {reps_live} Wiederholungen, Zufalls-Seed der Seitenleiste, wenige Sekunden)", key="e3_live"):
    with st.spinner("Rechne Wiederholungen …"):
        st.session_state["e3_result"] = (back_key, _back_live(*back_key))
res = st.session_state.get("e3_result")
if res is not None and res[0] == back_key:
    _verdict(V.verdict_e3_back(res[1]))
    st.caption(f"Die Live-Messung nimmt {res[1]['reps_per_seed']} Wiederholungen und einen Seed; sie schwankt stärker als die vorgerechnete Reihe ({V.de_int(X.E3_REPS)} Wiederholungen, vier Seeds).")
elif res is not None:
    st.caption("Die Einstellungen haben sich geändert: Knopf erneut drücken.")

# E4 ---------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 E4: Wie selten ist Simpson?")
st.markdown("Gruppe A hat in jeder Schicht die höhere Quote und insgesamt die niedrigere: die Umkehr, die das Beispiel „Die Tabelle, die kippt“ zeigt. Die Mechanik (eine gemeinsame Ursache) steht in `konfundierung-demo`. "
            "Hier wird gezählt, **wie oft** es in zufälligen Tabellen passiert und **bei welchem Gewichtsunterschied**.")
st.plotly_chart(build_reversal_chart(R.reversal_rows()), width="stretch", key="e4_reversal")
layers_pick = st.select_slider("Schichten", options=list(X.E4_LAYERS), value=2, key="e4_layers")
_verdict(V.verdict_e4_reversal(R.reversal_row(layers_pick)))
st.markdown("**Bei gegebener Dominanz:** A hat in jeder Schicht die höhere Quote (zufällig gezogen); die Gewichte der Schichten sind in beiden Gruppen verschieden (lognormal mit Streuung τ).")
st.plotly_chart(build_weights_chart(R.weight_rows()), width="stretch", key="e4_weights")
w1, w2 = st.columns(2)
w_k = w1.select_slider("Schichten k", options=list(X.E4_WEIGHT_K), value=2, key="e4_wk")
w_tau = w2.select_slider("Streuung τ der Gewichte", options=list(X.E4_TAUS), value=1.0, key="e4_tau", format_func=_de_label)
_verdict(V.verdict_e4_weights(R.weight_row(w_k, w_tau)))
tables_live = st.select_slider("Tabellen der Live-Messung", options=[100_000, 500_000, 1_000_000, 2_000_000], value=500_000, key="e4_live_tables", format_func=V.de_int)
tables_eff = int(min(tables_live, C.LIVE_TABLE_BUDGET // (4 * layers_pick)))
rev_key = (int(layers_pick), tables_eff, settings.seed)
if tables_eff < tables_live:
    st.caption(f"Bei {layers_pick} Schichten begrenzt das Rechenbudget die Tabellen auf {V.de_int(tables_eff)}.")
if st.button(f"Tabellen nachzählen ({layers_pick} Schichten, {V.de_int(tables_eff)} Tabellen, Zufalls-Seed der Seitenleiste, wenige Sekunden)", key="e4_live"):
    with st.spinner("Zähle Tabellen …"):
        st.session_state["e4_result"] = (rev_key, _reversal_live(*rev_key))
res = st.session_state.get("e4_result")
if res is not None and res[0] == rev_key:
    live = res[1]
    st.info(f"Nachgezählt: {V.pct(live['share']['mean'], 3)} % der {V.de_int(live['tables_per_seed'])} Tabellen mit {live['layers']} Schichten haben eine Umkehr ({V.de_int(live['hits'])} Treffer, Standardfehler {V.pct(live['standard_error'], 3)} Prozentpunkte).")
elif res is not None:
    st.caption("Die Einstellungen haben sich geändert: Knopf erneut drücken.")

# E5 ---------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 E5: Gesamtquote gegen Mittel der Quoten")
st.markdown(f"Zwei Gruppen mit je {X.E5_LOCATIONS} Standorten; jeder Standort hat einen Servicegrad (wahre Standortquote {X.E5_RATE:.2f} mit Streuung {X.E5_SPREAD:.2f}, in **beiden Gruppen gleich**) und eine Nachfrage. "
            "In Gruppe A ist die Nachfrage unabhängig vom Servicegrad, in Gruppe B hängt sie mit der Kopplung $\\kappa_B$ davon ab (starke Standorte haben mehr Nachfrage). "
            "Gemessen: Welche Gruppe liegt vorn, wenn man das **ungewichtete Mittel der Standortquoten** nimmt, und welche, wenn man die **Gesamtquote** (mit Nachfrage gewichtet) nimmt?")
sd_pick = st.select_slider("Streuung der Nachfrage σ_d (Logarithmus)", options=list(X.E5_SIGMA_DS), value=1.0, key="e5_sd", format_func=_de_label)
st.plotly_chart(build_coupling_chart([c for c in R.coupling_cells() if c["sigma_d"] == sd_pick]), width="stretch", key="e5_chart")
kap_pick = st.select_slider("Kopplung κ_B in Gruppe B", options=list(X.E5_KAPPAS), value=0.7, key="e5_kappa", format_func=_de_label)
_verdict(V.verdict_e5(R.coupling_cell(kap_pick, sd_pick)))
cpl_key = (float(kap_pick), float(sd_pick), C.LIVE_COMPARISONS, settings.seed)
if st.button(f"Nachmessen ({V.de_int(C.LIVE_COMPARISONS)} Vergleiche, ein Seed, unter einer Sekunde)", key="e5_live"):
    with st.spinner("Vergleiche Gruppen …"):
        st.session_state["e5_result"] = (cpl_key, _coupling_live(*cpl_key))
res = st.session_state.get("e5_result")
if res is not None and res[0] == cpl_key:
    _verdict(V.verdict_e5(res[1]))
elif res is not None:
    st.caption("Die Einstellungen haben sich geändert: Knopf erneut drücken.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Lognormale Einzelraten, Losgrößen und Faktoren** | Die Kosten des falschen Mittels hängen an der Streuung der gewählten Verteilung; echte Fahrten, Verbräuche und Lose haben andere Formen. Alle Prozentzahlen gelten für den Erzeuger, nicht für Ihre Daten. | – |
| **Der Nenner ist bekannt** | Das richtige Mittel folgt dem Nenner. Wer nur die Raten kennt, aber nicht weiß, ob Strecke oder Zeit gleich war, kann das Mittel nicht wählen; die Demo zeigt, was die Wahl kostet, nicht, wie man sie trifft. | – |
| **Unabhängige Faktoren mit gleicher Verteilung** | Echte Wachstumsraten sind abhängig und ändern ihre Verteilung; der Befund „das Stichprobenmittel trifft den Erwartungswert nicht“ gilt für unabhängige Pfade mit zwei Familien (Zweipunkt, lognormal). | – |
| **Stücke werden gleichverteilt gezogen** | Wird nicht jedes Stück gleich wahrscheinlich gezogen (etwa Fahrgäste, die selten fahren, antworten seltener), gilt der Faktor $1 + \\mathrm{CV}^2$ nicht. | Wartezeit pro Fahrgast: [bunching-demo](https://github.com/sebastian-hanisch/bunching-demo) |
| **Keine leeren Lose** | Leere Lose sind in der Stücksicht unsichtbar: Das harmonische Mittel liefert das Mittel der **nichtleeren** Lose, nicht des Mittels über alle Lose. | – |
| **Endliche zweite Momente** | Bei Pareto-Losgrößen mit $\\alpha \\le 2$ gibt es keinen Faktor und kein Mittel der Stücksicht; das arithmetische Mittel der Beobachtungen wächst mit der Zahl der Beobachtungen, mehr Daten helfen nicht. | – |
| **Zufällige Tabellen mit gleichverteilten Zellen** | Der Anteil der Umkehr (E4) gilt für diesen Erzeuger. Echte Tabellen haben meist ungleiche Gewichte, weil die Gruppen verschieden in die Schichten laufen; das ist die Bedingung für die Umkehr. | Kausale Deutung: [konfundierung-demo](https://github.com/sebastian-hanisch/konfundierung-demo) |
| **Kopplung von Nachfrage und Quote im Modell** | E5 setzt die Kopplung κ_B und die Streuung der Nachfrage als Modellannahme. Ob sie in Ihren Daten besteht, zeigt die Demo nicht; sie zeigt nur, dass Gesamtquote und Mittel der Quoten dann verschieden ordnen. | – |
| **Simulierte Daten, vier Seeds je Zelle** | Die angegebene Streuung ist die zwischen vier festen Zufalls-Seeds, kein Konfidenzintervall. | – |
    """
)
st.markdown(
    """
**Bekannte Probleme und Unsicherheiten**

- Die Zahlen sind **Messungen an erzeugten Daten**; die Aufgaben (Fahrten, Lose, Standorte) sind gewählt. Dass sie einem echten Betrieb ähneln, ist nicht gezeigt.
- Das **Stichprobenmittel der Pfade** schwankt bei langen Horizonten stark (T = 100 und 200: die Streuung zwischen den vier Seeds ist so groß wie der Mittelwert); belastbar ist deshalb die Trefferquote über 300 Stichproben je Seed, nicht der Wert einzelner Stichproben.
- Der **Vergleichswert 1/60** für zufällige 2×2×2-Tabellen steht in der Literatur; die Quelle wurde für diese Demo **nicht eingesehen**. Die Messung hier trifft die Zahl, ohne sie herzuleiten.
- Literatur zu Potenzmitteln, Längenverzerrung und Wachstumsverlust (Zeitmittel gegen Erwartungswert) wurde **nicht gelesen**; alle Aussagen sind eigene Messungen und Umformungen, geprüft gegen Handrechnung, Aufzählung und `scipy`.
- Mausklicks auf die Diagramme (Auswahl) sind nicht Teil der Demo.
    """
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** *Gegeben:* Posten $i = 1, \dots, n$ mit Zähler $a_i$ und Nenner $b_i > 0$ (Strecke und Zeit, Liter und Strecke) und der Einzelrate $x_i = a_i / b_i$. *Gesucht:* die Gesamtgröße und das Mittel, das sie wiedergibt.

| Symbol | Bedeutung |
|---|---|
| $a_i$, $b_i$, $x_i$ | Zähler, Nenner und Rate des Postens $i$ |
| $w_i$ | Gewicht, Summe 1 |
| $r$, $M_r$ | Mittelungsexponent und Potenzmittel |
| $f_t$, $V_T$ | Faktor der Periode $t$ und Wert nach $T$ Perioden |
| $m$, $\sigma$ | Erwartungswert je Periode und Streuung der Logarithmen der Faktoren |
| $X$, $Y$ | Größe eines Loses und Größe des Loses eines gezogenen Stücks |
| $\mathrm{CV}$ | Variationskoeffizient der Losgrößen |
| $q_{gs}$, $w_{gs}$ | Quote und Gewicht der Schicht $s$ in Gruppe $g$ |

**Gesamtrate.** Die Gesamtgröße ist die Summe der Zähler durch die Summe der Nenner:

$$\bar x = \frac{\sum_i a_i}{\sum_i b_i}.$$

Das ist das gewichtete arithmetische Mittel der Raten mit den Nennern als Gewichten:

$$\bar x = \sum_i w_i x_i, \qquad w_i = \frac{b_i}{\sum_j b_j}.$$

**Potenzmittel.**

$$M_r(x; w) = \Big(\sum_i w_i\, x_i^{\,r}\Big)^{1/r}, \qquad M_0(x; w) = \exp\Big(\sum_i w_i \ln x_i\Big).$$

Es gilt $M_{-\infty} \le M_{-1} \le M_0 \le M_1 \le M_2 \le M_{\infty}$. Für lognormale Werte mit Streuung $\sigma$ ist

$$\frac{M_r}{M_0} = e^{\,r\sigma^{2}/2}.$$

**Gleiche Strecke $s$ je Fahrt:** $b_i = s / v_i$, also

$$\bar v = \frac{n\,s}{\sum_i s / v_i} = M_{-1}(v).$$

**Gleiche Zeit $t$ je Fahrt:** $a_i = v_i\, t$, also

$$\bar v = \frac{\sum_i v_i\, t}{n\, t} = M_{1}(v).$$

**Wachstum.** Das Produkt der Faktoren ist die $T$-te Potenz des geometrischen Mittels:

$$V_T = \prod_{t=1}^{T} f_t = \big(M_0(f)\big)^{T}.$$

Zweipunkt-Faktoren $1{,}5$ und $0{,}6$ mit Wahrscheinlichkeit $\tfrac12$: Erwartungswert und Median (gerades $T$)

$$\mathbb{E}[V_T] = 1{,}05^{\,T}, \qquad \operatorname{Median}(V_T) = (1{,}5 \cdot 0{,}6)^{T/2} = 0{,}9^{\,T/2}.$$

Der Anteil der Pfade unter dem Start ist eine Binomialsumme über alle $k$ Aufwärtsschritte mit $1{,}5^{k}\, 0{,}6^{T-k} < 1$:

$$\Pr(V_T < 1) = 2^{-T} \sum_{k:\ 1{,}5^{k} 0{,}6^{T-k} < 1} \binom{T}{k}.$$

Lognormale Faktoren mit Erwartungswert $m$ je Periode:

$$\ln V_T \sim \mathcal{N}\Big(T\big(\ln m - \tfrac{\sigma^{2}}{2}\big),\ T\sigma^{2}\Big).$$

Der typische Faktor je Periode ist $m\, e^{-\sigma^{2}/2}$; er fällt unter 1 ab

$$\sigma^{*} = \sqrt{2 \ln m}.$$

**Größenverzerrung.** Wer ein Stück gleichverteilt unter allen Stücken zieht und die Größe $Y$ seines Loses notiert, sieht ein Los der Größe $x$ mit Wahrscheinlichkeit

$$\Pr(Y = x) = \frac{x\, \Pr(X = x)}{\mathbb{E}[X]}.$$

Das Mittel der Beobachtungen und das der Kehrwerte sind

$$\mathbb{E}[Y] = \frac{\mathbb{E}[X^{2}]}{\mathbb{E}[X]} = \mathbb{E}[X]\,\big(1 + \mathrm{CV}^{2}\big), \qquad \mathbb{E}\Big[\frac{1}{Y}\Big] = \frac{\Pr(X > 0)}{\mathbb{E}[X]}.$$

Das harmonische Mittel der Beobachtungen ist also das Mittel pro Los, bei leeren Losen das Mittel der nichtleeren Lose $\mathbb{E}[X] / \Pr(X > 0)$. Bei Pareto-Losgrößen mit Tail-Index $\alpha > 2$ ist

$$\mathbb{E}[Y] = \frac{\alpha - 1}{\alpha - 2},$$

für $\alpha \le 2$ ist $\mathbb{E}[X^{2}]$ unendlich.

**Gesamtquote und Umkehr.** Die Gesamtquote der Gruppe $g$ über die Schichten $s$ ist

$$q_g = \sum_s w_{gs}\, q_{gs}, \qquad w_{gs} = \frac{n_{gs}}{\sum_t n_{gt}}.$$

Umkehr heißt $q_{As} > q_{Bs}$ für alle $s$ und $q_A < q_B$ (oder umgekehrt). Mit gemeinsamen Standardgewichten

$$\tilde q_g = \sum_s \tfrac12 (w_{As} + w_{Bs})\, q_{gs}$$

ist die Umkehr ausgeschlossen. Der Abstand der Gewichte ist

$$\mathrm{TV} = \tfrac12 \sum_s \big|w_{As} - w_{Bs}\big|.$$

**Literatur.** Potenzmittel, Längenverzerrung, Wachstumsverlust und das Simpson-Paradox sind Lehrbuchstoff; die Quellen wurden für diese Demo **nicht eingesehen**. Ein in der Literatur genannter Vergleichswert für zufällige 2×2×2-Tabellen (1/60) wird nur als Marke in einem Test benutzt.
Alle Aussagen der Demo sind eigene Messungen an erzeugten Daten und Umformungen, die in `tests/` gegen Handrechnung, Aufzählung, exakte Brüche und `scipy` geprüft sind.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
