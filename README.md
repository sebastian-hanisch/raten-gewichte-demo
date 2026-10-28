# Raten, Gewichte, Größenverzerrung: Welches Mittel gehört wozu? – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-raten-gewichte-demo.streamlit.app/)** -->

Stück 7 der Reihe **Mittelwerte und Lagemaße** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Kante im Abhängigkeitsgraphen: Stück 1 → Stück 7 (`mittelwert-median-demo` → hier), die Schwäche des Mittels lautet: **Es passt nicht zu Quotienten und nicht zu ungleichen Gewichten.** Das Stück fragt nicht, wie sich ein Mittel gegen Ausreißer wehrt, sondern **welches Mittel zu welchem „pro“ gehört** und was es kostet, das falsche zu nehmen.

Vehikel: Fahrten mit Geschwindigkeit (gleiche Strecke, gleiche Zeit), Fahrzeuge mit Verbrauch (l/100 km gegen km/l), Bedarf mit Wachstumsfaktoren je Periode (1,5 und 0,6, ohne Finanzsprache), Lose mit Stück-Sicht (Fahrgäste je Fahrzeug, Stück je Los), Standorte mit Nachfrage und Servicegrad und zwei Gruppen in zwei Schichten. Alle Aufgaben und Verteilungen sind **gewählt, keine Messung an echten Daten**.

Abgrenzung (im Abgleich der READMEs gelesen): [bunching-demo](https://github.com/sebastian-hanisch/bunching-demo) misst die Wartezeit pro Fahrgast statt pro Bus (Verhältnis 1 + CV²), [konfundierung-demo](https://github.com/sebastian-hanisch/konfundierung-demo) erklärt, warum eine gemeinsame Ursache das Vorzeichen kippt, `lp-mitte-demo` (Stück 3) nutzt den Buchstaben **p** für den Exponenten der Strecke, `p-center-demo` für die Zahl der Mittelpunkte; hier heißt der Exponent des Mittels **r**. Die Wartezeit-Formel und die Kausalmechanik werden **nicht wiederholt**, nur gemessen (Faktor 1 + CV²) bzw. verwiesen.

## Warum dieses Problem

Jedes Mittel ist ein gewichtetes Mittel, und die Frage legt die Gewichte fest. **Die Gesamtgröße einer Rate ist das mit den Nennern gewichtete arithmetische Mittel der Einzelraten**: Zeit, Strecke, Größe, Nachfrage, Gruppenanteil. Zwei Fahrten mit 50 und 100 km/h über je 10 km sind 20 km in 0,3 h, also **66,67 km/h**; über je eine Stunde sind es **75 km/h**. Dasselbe Datenpaar, zwei Antworten, und beide sind richtig für ihre Frage.

Kernfrage: *Welches Mittel gehört zu welchem „pro“, und warum gibt dieselbe Tabelle je nach Gewicht zwei Antworten?* Fünf gemessene Haken tragen die Demo:

1. **Das falsche Mittel kostet mit der Streuung überproportional** (als Faustformel etwa wie e^(σ²) − 1: das Verhältnis M₁/M₋₁ für sehr viele lognormale Posten nach dem Verhältnis M_r/M_0 des Expanders „📐 Mathematische Formulierung“; mit 20 Posten gemessen liegen die Werte darunter). Bei gleicher Strecke liegt das arithmetische Mittel der Geschwindigkeiten bei Streuung σ = 0,5 um **27,1 %** über der Gesamtgeschwindigkeit, bei σ = 1,0 um **162,6 %**.
2. **Das Stichprobenmittel trifft den Erwartungswert nicht.** Bei den Faktoren 1,5 und 0,6 liegen nach 100 Perioden **86,4 %** der Pfade unter dem Start; das Mittel aus 10.000 Pfaden trifft den Erwartungswert (± 20 %) nur in **6,5 %** der Stichproben (Messwert aus 1.200 Stichproben mit einem Standardfehler von etwa 0,7 Prozentpunkten; eine unabhängige Nachrechnung mit mehr Stichproben liegt höher, gehört aber nicht zur Messreihe der Demo).
3. **Das harmonische Mittel rechnet die Größenverzerrung zurück.** Bei lognormalen Losgrößen (σ = 1) liegt das arithmetische Mittel der beobachteten Größen um **171,5 %** über dem Mittel pro Los, das harmonische um **1,6 %** (n = 100). Leere Lose bleiben dabei unsichtbar.
4. **Bei Pareto-Losgrößen mit α ≤ 2 gibt es keine Rückrechnung über das arithmetische Mittel:** es wächst mit der Zahl der Beobachtungen.
5. **Gesamtquote und Mittel der Quoten ordnen verschieden.** Bei gleicher wahrer Standortquote liegt Gruppe B in **81,3 %** der Vergleiche vorn, wenn man die Gesamtquote nimmt und die Nachfrage mit der Quote gekoppelt ist (κ_B = 0,7, Gruppe A ohne Kopplung, je 10 Standorte, σ_d = 1), im ungewichteten Mittel der Quoten in 50,2 %.

## Trägt das Stück über die Nachbarn hinaus?

**Teilweise, und ehrlich nur als Sammelstück.** Neu im Portfolio (Abgleich der READMEs, Stand 2026-10-24): die Potenzmittel-Familie und die Zuordnung „welches Mittel zu welcher Aufgabe“, die Kosten des falschen Mittels als Kurve (E1), das Stichprobenmittel der Pfade neben dem Erwartungswert (E2), die Rückrechnung von der Stücksicht auf die Größe pro Los (E3) und die Häufigkeit der Simpson-Umkehr (E4). Nicht neu sind der Verzerrungsfaktor 1 + CV² (steht in `bunching-demo`) und die Mechanik der Umkehr (`konfundierung-demo`). Es gibt **keinen einzelnen großen Befund** auf der Höhe der Stücke 3 bis 5, sondern einen gemeinsamen Blick (jedes Mittel ist ein gewichtetes Mittel, die Frage legt die Gewichte fest) und mehrere kleine gemessene Haken darin. Die Ampel bleibt gelb; die Zusammenlegung mit Stück 1 (ein Abschnitt „Mittel sind Gewichtung“ mit E1 bis E3) bliebe eine ehrliche Alternative.

## Modell

- **Aufgaben mit Nenner** (`rgw_tasks.py`): Die wahre Gesamtgröße wird aus den Rohgrößen gerechnet (Gesamtstrecke durch Gesamtzeit, Gesamtliter durch Gesamtstrecke, Produkt der Faktoren), nicht aus einem Mittel. Das richtige Mittel ist das, was sie wiedergibt: **harmonisch** (gleiche Strecke), **arithmetisch** (gleiche Zeit; l/100 km bei gleicher Strecke), **geometrisch** (Wachstumsfaktoren). Einzelraten lognormal (Median 80 km/h bzw. 8 l/100 km bzw. 1), Streuung σ der Logarithmen als Regler.
- **Potenzmittel** (`rgw_means.py`): M_r mit r = −1 harmonisch, 0 geometrisch, 1 arithmetisch, 2 quadratisch; gewichtet; numerisch stabil bis r = ±300. Der Exponent heißt **r**, nicht p.
- **Wachstumspfade** (`rgw_growth.py`): Faktor 1,5 oder 0,6, je 50 %, unabhängig: Erwartungswert 1,05 je Periode, geometrisches Mittel √0,9 = 0,9487, im Wechsel 0,9 je Doppelschritt; zweite Familie lognormal mit festem Erwartungswert 1,05 und Streuung σ.
- **Größenverzerrung** (`rgw_sizebias.py`): Lose aus Poisson, geometrisch, lognormal oder Pareto; ein gezogenes Stück erscheint mit der Wahrscheinlichkeit seines Loses proportional zur Losgröße. Beobachtete Größen aus der Formel gezogen (Poisson → 1 + Poisson, geometrisch → Summe zweier minus 1, lognormal → μ + σ², Pareto(α) → Pareto(α − 1)) oder aus einer endlichen Grundgesamtheit.
- **Gewichtsmix** (`rgw_mix.py`): Tabellen 2 × 2 × l (zwei Gruppen, Erfolg/Misserfolg, l Schichten); Zellwahrscheinlichkeiten gleichverteilt (Dirichlet(1, …, 1), normierte Exponentialverteilte) oder A in jeder Schicht besser mit lognormalen Gewichten (Streuung τ); Standorte mit Servicegrad 0,80 ± 0,08 und mit Quote gekoppelter Nachfrage (Kopplung κ, Streuung σ_d).
- Zufall: ein eigener SplitMix64-Strom (`rgw_rng.py`, wörtlich kopiert), blockweise mit numpy gezogen; Verteilungen aus den Gleichverteilten (Box-Muller, Quantilfunktion der aufsummierten Wahrscheinlichkeitstabellen), kein `numpy.random`. Die Formeln stehen im 📐-Expander der App.

## Methodik

- **Gepaarte Vergleiche:** Alle verglichenen Verfahren sehen dieselben Daten: die vier Mittel dieselben Instanzen, arithmetisches und harmonisches Mittel dieselben Beobachtungen, ungewichtete und Gesamtquote dieselben Standorte.
- **Messzellen:** vier feste Zufalls-Seeds je Zelle, angegeben als Mittel mit Streuung zwischen den Seeds (kein Konfidenzintervall); ein Seed gehört zu einer Zelle, nie zu einer Instanz allein. Die Messreihen sind vorgerechnet (`tools/make_results.py` → `data/rgw_results.json` mit Prüfsumme der Quelldateien); die Knöpfe „Nachmessen“ rechnen mit eigenen Einstellungen neu und sind durch Budgets begrenzt.
- **Exakte Gegenwerte:** Der Anteil der Pfade unter dem Start und über dem Erwartungswert ist eine Binomialsumme und wird neben die Simulation gestellt; der lognormale Anteil unter dem Start ist eine Normalverteilungsfunktion.
- **Rechenbudget:** Pfade N · T höchstens 5 Millionen Zufallszahlen live, Rückrechnung höchstens 2 Millionen Beobachtungen, Tabellen höchstens 20 Millionen Zufallszahlen.

## Befunde (gemessen, keine Behauptungen)

Vier feste Zufalls-Seeds je Zelle; jede Zahl steht in `tests/test_claims.py` und wird dort aus `data/rgw_results.json` bzw. den Live-Funktionen nachgerechnet. Alle Zahlen gelten für den Erzeuger dieser Demo, nicht für echte Daten.

| Frage | Befund | Test |
|---|---|---|
| Hand-Beispiele | 50 und 100 km/h über je 10 km: **66,67** km/h (harmonisch), über je eine Stunde 75 (arithmetisch). Faktoren 1,5 und 0,6: arithmetisch 1,05, geometrisch 0,9487, harmonisch 0,8571. Lose 1, 1, 4: Mittel pro Los 2, Mittel der Stücksicht 3 (Faktor 1,5 = 1 + CV²), harmonisch 2. Tabelle: A 63 % gegen B 76 % insgesamt, in beiden Schichten A vorn; mit Gewichten (½, ½) A 75 % gegen B 60 %. | `test_hand_examples` |
| E1 Preis des falschen Mittels bei gleicher Strecke (20 Posten, 2.000 Instanzen je Seed) | Mittlere Abweichung des **arithmetischen** Mittels von der Gesamtgeschwindigkeit: **3,9 % / 27,1 % / 162,6 %** bei σ = 0,2 / 0,5 / 1,0; geometrisch 1,9 / 12,7 / 60,5 %; quadratisch 5,9 / 42,8 / 297,2 %. | `test_e1_arithmetic_mean_for_equal_distance` |
| E1 Spanne über alle Aufgaben | Über alle vier Aufgaben (gleiche Strecke, gleiche Zeit, Verbrauch, Faktoren) liegt jedes falsche Mittel bei σ = 0,2 zwischen 1,9 und 5,9 %, bei σ = 0,5 zwischen 11 und 43 %, bei σ = 1,0 zwischen 36 und 297 % neben der wahren Gesamtgröße. Das richtige Mittel hat die Abweichung 0 (Rechenidentität, kein Befund). | `test_e1_range_over_all_tasks` |
| E1 Verbrauch | Wer km/l mittelt und zurückrechnet, landet beim harmonischen Mittel der l/100 km: **20,7 %** neben dem Gesamtverbrauch bei σ = 0,5 (gleiche Aufgabe „gleiche Zeit“: 20,8 %). | `test_e1_consumption_in_km_per_litre` |
| E1 Wachstumsfaktoren (lognormal) | Arithmetisch statt geometrisch: **12,6 %** bei σ = 0,5 und **61,3 %** bei σ = 1,0; harmonisch 11,0 und 36,2 %. | `test_e1_growth_factors` |
| E2 Pfade 1,5 / 0,6 | Anteil unter dem Start nach T = 10 / 20 / 50 / 100 / 200 Perioden **62,3 / 74,8 / 76,0 / 86,4 / 94,8 %** (Binomialsumme); die Simulation von 100.000 Pfaden liegt höchstens 0,1 Prozentpunkte daneben. Der Median ist 0,9^(T/2). Über dem Erwartungswert 1,05^T liegen nur 17,2 / 13,2 / 5,9 / **1,0** / 0,07 % der Pfade. | `test_e2_shares_below_start_and_above_expectation` |
| E2 Stichprobenmittel, N = 100.000 | Verhältnis zum Erwartungswert bei T = 100: **0,79 (Streuung zwischen den Seeds 0,30)**, die vier Seeds reichen von 0,49 bis 1,22; bei T = 200: 0,05. Der Wert ist **kein stabiler Befund** (Heavy Tail), die Trefferquote darunter schon. | `test_e2_sample_mean_ratio_is_noisy` |
| E2 Trefferquote des Stichprobenmittels | Anteil der Stichproben, deren Mittel den Erwartungswert auf ± 20 % trifft, Zweipunkt T = 60 bei N = 100 / 1.000 / 10.000 Pfaden: **5,6 / 14,0 / 35,1 %**; T = 100: 3,5 / 3,7 / 6,5 %; T = 20: 38,6 / 82,1 / 99,9 %. Lognormal (σ = 0,5) noch schlechter: T = 60: 4,7 / 9,2 / 15,8 %, T = 100: 0,9 / 1,9 / 4,1 %. Der Median des Verhältnisses ist bei T = 100, N = 10.000: 0,41 (Zweipunkt) und 0,19 (lognormal). Die Prozentwerte sind Messwerte aus 1.200 Stichproben je Zelle (4 Seeds · 300) mit einem Standardfehler bis etwa 1,4 Prozentpunkte; auf die zweite Stelle sind sie nicht verlässlich. | `test_e2_coverage_of_the_sample_mean` |
| E2 Streuung statt Zweipunkt | Lognormale Faktoren mit Erwartungswert 1,05: typischer Faktor je Periode **1,029 / 0,927** bei σ = 0,2 / 0,5 (Formel 1,05·exp(−σ²/2) = 1,029 / 0,927); kritisch ab σ = √(2 ln 1,05) = **0,312**. Nach 50 Perioden liegen 15,5 / 59,9 / 86,0 / 99,2 % der Pfade unter dem Start bei σ = 0,2 / 0,35 / 0,5 / 0,8. | `test_e2_lognormal_growth_and_the_critical_sigma` |
| E3 Verzerrungsfaktor (Stücksicht gegen Lossicht) | Poisson(1 / 3 / 10 / 30): 1,99 / 1,33 / 1,10 / 1,03 (Formel 1 + CV² = 2,00 / 1,33 / 1,10 / 1,03); geometrisch(0,3) 1,69 (1,70); lognormal(σ = 0,5 / 1 / 1,5) 1,29 / 2,88 / 9,19 (1,28 / 2,72 / 9,49, die Seeds streuen bei σ ≥ 1 stark); Pareto(3) 1,37 (1,33). **Bestätigt die Formel aus `bunching-demo`, kein neuer Befund.** | `test_e3_bias_factor_confirms_the_neighbour_formula` |
| E3 Rückrechnung lognormal (σ = 1) | Auf das Mittel pro Los: arithmetisches Mittel der Beobachtungen **+171,5 %** (n = 100) und +171,9 % (n = 1.000), nicht kleiner mit mehr Daten; harmonisches Mittel **+1,6 %** (n = 100), +0,2 % (n = 1.000), Streuung (RMSE) 12,9 und 4,2 %. σ = 0,5: +28,4 %; σ = 1,5: +848,4 % (n = 1.000). | `test_e3_back_estimate_for_lognormal_lots` |
| E3 Leere Lose | Poisson(3), n = 1.000: Das harmonische Mittel liegt **+5,3 %** über dem Mittel (3,00), und genau auf dem Mittel der nichtleeren Lose λ/(1 − e^(−λ)) = 3,157 (Abweichung 0,0 %): Fahrzeuge ohne Fahrgäste kommen in einer Fahrgastbefragung nicht vor. | `test_e3_empty_lots_stay_invisible` |
| E3 Schwerer Tail, Pareto(2,2) | Arithmetisch +212 % (Streuung zwischen den Seeds 24, RMSE 718 %), harmonisch **0,0 %** mit RMSE 1,6 % (n = 1.000). Bei geometrisch(0,3) +70,0 % gegen +0,1 %. Die +212 % sind ein Mittel über Wiederholungen, das von wenigen Ausreißern getrieben wird (unendliche Varianz der Stücksicht); ein typischer Lauf liegt nach einer unabhängigen Nachrechnung niedriger, die nicht zur Messreihe der Demo gehört. | `test_e3_heavy_tail_and_geometric` |
| E3 Kein Faktor bei α ≤ 2 | Median des arithmetischen Mittels der Beobachtungen bei n = 100 / 10.000 / 1.000.000: α = 3: 1,94 / 2,00 / 2,00 (Formel (α − 1)/(α − 2) = 2); α = 2,5: 2,76 / 2,93 / 2,98 (3); **α = 2: 6,2 / 11,3 / 15,2 (wächst)**; α = 1,5: 353 / 3,4·10⁴ / 5,0·10⁶. | `test_e3_no_factor_for_alpha_at_most_two` |
| E4 Wie selten ist Simpson? | Zufällige 2 × 2 × l-Tabellen (Zellwahrscheinlichkeiten gleichverteilt): Umkehr bei l = 2 / 3 / 4 / 6 Schichten in **1,675 / 0,570 / 0,175 / 0,017 %** der Tabellen (4.000.000 je Zeile, Standardfehler 0,006 Prozentpunkte bei l = 2); bei l = 10 in 6 von 4.000.000. Die Zahl bei l = 2 liegt bei 1/60 = 1,667 % (Vergleichswert aus einer Sekundärquelle, hier nicht selbst hergeleitet). | `test_e4_reversal_share_falls_with_the_layers` |
| E4 Gewichtsunterschied | A in jeder Schicht besser, Gewichte lognormal (τ): bei k = 2 Schichten kippt die Gesamtquote in **0,7 / 2,0 / 4,2 / 5,6 %** der Tabellen bei τ = 0,25 / 0,5 / 1,0 / 1,5 (mittlerer Abstand der Gewichte 0,10 / 0,18 / 0,30 / 0,37; unter den Umkehrungen im Median 0,17 / 0,31 / 0,49 / 0,60). Höchstens 5,6 % im ganzen Raster. Mit **gleichen Gewichten** oder **gemeinsamen Standardgewichten** gibt es in allen Zellen 0 Umkehrungen (erzwungen, Probe, kein Befund). | `test_e4_weight_spread_and_the_forced_zero` |
| E5 Gesamtquote gegen Mittel der Quoten | Gleiche wahre Standortquote 0,80 in beiden Gruppen, Streuung der Nachfrage σ_d = 1: Unterschied B − A ungewichtet **0,02** Prozentpunkte, bei der Gesamtquote **4,48** (κ_B = 0,7); B liegt ungewichtet in 50,2 %, nach Gesamtquote in **81,3 %** der Vergleiche vorn (κ_B = 0,3: 64,6 %; κ_B = −0,7: 19,0 %; ohne Kopplung 50,1 %). Bei σ_d = 0,5 / 1,5: 73,0 / 83,6 %. | `test_e5_pooled_rate_against_mean_of_rates` |

## Hypothesen des Einzelplans (Ja/Nein nach eigener Messung)

Nummern wie in `mittelwerte-planung/STUECK-7-raten-gewichte.md`. Alle Zahlen wurden im Bau mit dem **eigenen SplitMix64-Strom** neu gemessen; die Wegwerf-Vormessung des Plans (numpy-Zufall) liefert keine README-Zahl.

| | Hypothese | Ergebnis | Zahl (diese Demo) |
|---|---|---|---|
| H1 | 50 und 100 km/h über gleiche Strecke: 66,67 statt 75 | **Ja** | Handrechnung und Mini-Instanz: 66,67 und 75 (`tests/conftest.py`, `test_means.py`) |
| H2 | Reihenfolge HM ≤ GM ≤ AM ≤ QM, und M_r/GM = exp(rσ²/2) bei lognormal | **Ja** | in 200 Zufallsmengen (Orakel) ohne Verletzung; M_r gegen `scipy.stats.pmean` auf 12 Stellen; lognormal σ = 0,5, 200.000 Werte: innerhalb 1 % der Formel. Die 80.000 Mengen der Vormessung wurden nicht wiederholt. |
| H3 | Das falsche Mittel kostet bei σ = 0,2 unter 6 %, bei 0,5 11 bis 43 %, bei 1,0 36 bis 297 % | **Ja** | siehe E1 |
| H4 | km/l gemittelt und zurückgerechnet liegt 20,7 % (σ = 0,5) neben dem wahren l/100 km | **Ja** | 20,7 % (Aufgabe Verbrauch), 20,8 % (Aufgabe gleiche Zeit) |
| H5 | Faktoren 1,5/0,6: Median-Pfad 0,9^(T/2), Pfade unter Start 62 / 75 / 86 / 95 % nach 10 / 20 / 100 / 200 Perioden | **Ja** | 62,3 / 74,8 / 86,4 / 94,8 % (exakt), Simulation höchstens 0,1 Prozentpunkte daneben |
| H6 | Das Stichprobenmittel trifft 1,05^T nicht: 0,54 bei T = 100, 0,08 bei T = 200 (N = 10⁵); bei T = 60 trifft ein Mittel aus 10⁴ Pfaden in 33 % der Läufe ± 20 % | **Teilweise** | Der Wert bei T = 100 reproduziert sich nicht als Zahl: 0,79 mit Streuung 0,30 zwischen den vier Seeds (0,49 bis 1,22); T = 200: 0,05. Die **Trefferquote** trägt: 35,1 % bei T = 60, N = 10⁴, in zwei Pfadfamilien (Zweipunkt und lognormal). |
| H7 | Ab σ = √(2 ln 1,05) = 0,312 schrumpft der typische Pfad trotz + 5 % im Mittel | **Ja** | kritisches σ = 0,312; typischer Faktor 1,0001 bei σ = 0,3124 (gemessen); 0,927 bei σ = 0,5 |
| H8 | Größenverzerrung = 1 + CV² in allen Kulissen | **Ja** (kein Neuwert) | siehe E3, Tabelle der Faktoren |
| H9 | Das harmonische Mittel der beobachteten Größen stellt das Mittel pro Los her (arithmetisch + 28 / + 172 / + 850 %), auch bei Pareto(2,2) | **Ja** | +28,4 / +171,5 / +845,2 % (n = 100) gegen +0,3 / +1,6 / +6,6 %; Pareto(2,2): +0,2 % |
| H10 | Leere Lose sind in der Stücksicht unsichtbar (Poisson(3): + 5,3 %) | **Ja** | +5,3 % bei n = 1.000, auf dem Mittel der nichtleeren Lose |
| H11 | Bei Pareto α ≤ 2 existiert die Stücksicht-Größe nicht (das Mittel wächst mit n) | **Ja** | α = 2: 6,2 / 11,3 / 15,2 bei n = 100 / 10⁴ / 10⁶ |
| H12 | Simpson-Umkehr in 1/60 der zufälligen 2 × 2 × 2-Tabellen, mit der Schichtzahl stark fallend | **Ja** | 1,675 % (1/60 = 1,667 %), dann 0,570 / 0,175 / 0,017 % |
| H13 | Selbst bei Dominanz in jeder Schicht kippt die Gesamtordnung nur bei großem Gewichtsunterschied, höchstens 7,7 % der Tabellen | **Teilweise** (der Anteil ja, mit anderem Gewichtsmodell; „nur bei großem Gewichtsunterschied“ nein) | Hier lognormale statt Dirichlet-Gewichte: höchstens 5,6 % im Raster; im Median der Umkehrungen ein Abstand der Gewichte von 0,17 bis 0,71, die Umkehrung tritt also schon bei mittlerem Gewichtsabstand auf. Die 7,7 % des Plans gehören zu den Dirichlet-Gewichten und werden **nicht** übernommen. |
| H14 | Gesamtquote und Mittel der Quoten ordnen Gruppen mit gleicher Standortquote verschieden: 81 % B > A bei κ_B = 0,7 | **Ja** | 81,3 % gegen 50,2 % (σ_d = 1, 10 Standorte je Gruppe) |
| H15 | Mit Standardgewichten kippt nichts | **Ja** (erzwungen, Probe) | 0 Umkehrungen in allen Zellen; mathematisch erzwungen (Konvexkombination derselben Gewichte), kein Befund |

## Befunde und Korrekturen gegenüber dem Plan

- **Wartezeit-Formel nicht wiederholt.** Der Plan (Stück 7, H2) hatte „Wartezeit/Abstand 0,50 / 1,00 / 2,50“; das gehört zu `bunching-demo`. Hier nur der Verzerrungsfaktor als Messung (E3) und der Verweis.
- **M4d (Mittel der Quoten gegen Gesamtquote mit zufälliger Nachfrage) nicht übernommen.** Die Vormessung zeigte nur Rauschen der Gewichte; ersetzt durch E5 (gleiche wahre Standortquote, nur die Kopplung unterscheidet sich).
- **Die Diagonale von E1 und die Standardgewichte in E4 sind Identitäten, keine Befunde.** Das richtige Mittel hat die Abweichung 0 (Rechenidentität); die Standardgewichte kippen nie (Konvexkombination). Beides steht als solches in App und Tests.
- **M2b auf mehr als eine Pfadfamilie.** Die Plan-Vormessung nutzte nur den Zweipunkt-Pfad; hier zusätzlich lognormale Faktoren. Ergebnis: Das Stichprobenmittel trifft den Erwartungswert auch dort selten, bei langen Horizonten noch seltener (T = 100, N = 10.000: 4,1 % gegen 6,5 %). Die Trefferquoten des Plans für Zweipunkt (T = 20: 37,5 / 80,7 / 99,9 %, T = 100: 3,7 / 2,5 / 8,1 %) reproduzieren sich mit anderem Zufallsstrom nur im Rahmen der Streuung (hier 38,6 / 82,1 / 99,9 % und 3,5 / 3,7 / 6,5 %).
- **Das Stichprobenmittel bei T = 100 und N = 10⁵ ist kein stabiler Wert.** Der Plan nannte 0,54, hier 0,79 mit Streuung 0,30 zwischen den vier Seeds; belastbar ist die Trefferquote über 300 Stichproben je Seed und Zelle (siehe H6).
- **Gewichtsmodell für E4 ist ein anderes.** Der Plan zog Gewichte aus Dirichlet-Verteilungen (dafür wäre ein Gamma-Sampler nötig); hier sind es lognormale Gewichte mit Streuung τ. Die Zahlen sind deshalb **nicht** die des Plans (dort bis 7,7 %), sondern eigene (bis 5,6 %).
- **Abweichung der Umsetzung:** Die Hauptansicht rechnet je Aufgabe die vier Mittel auf einer einzigen Instanz; die Mittelwerte über 2.000 Instanzen stehen in E1. Der Regler „Aufgabe“ enthält sechs Aufgaben (gleiche Strecke, gleiche Zeit, Verbrauch, Faktoren im Wechsel, Losgrößen, Tabelle); zufällige lognormale Faktoren gibt es nur in E1.
- **Der Vergleichswert 1/60 ist nicht hergeleitet.** Er gilt für gleichverteilte 2 × 2 × 2-Tabellen und stammt aus einer Sekundärquelle (laut Auszug, nicht aus dem Volltext); keine Quelle wird hier zitiert, die Messung trifft die Zahl (1,675 % gegen 1,667 %, innerhalb von zwei Standardfehlern), ohne sie herzuleiten.

## Ehrliche Grenzen

- **Erzeugte Daten.** Alle Prozentzahlen gelten für lognormale Raten, Poisson-, geometrische, lognormale und Pareto-Lose, Zweipunkt- und lognormale Faktoren und zufällige Tabellen. Echte Fahrten, Verbräuche und Lose haben andere Formen.
- **Der Nenner muss bekannt sein.** Das Mittel folgt dem Nenner. Die Demo zeigt, was die falsche Wahl kostet, nicht, wie man den Nenner herausfindet.
- **Das Stichprobenmittel der Pfade streut stark** (Heavy Tail): Der Wert eines einzelnen Mittels ist kein Befund, die Trefferquote über viele Stichproben schon.
- **Leere Lose und schwere Tails brechen die Rückrechnung:** Das harmonische Mittel liefert das Mittel der nichtleeren Lose, bei Pareto α ≤ 2 hilft kein Mittel der Beobachtungen allein.
- **E5 setzt die Kopplung als Modellannahme.** Ob Nachfrage und Quote in Ihren Daten gekoppelt sind, zeigt die Demo nicht.
- **Keine Literatur gelesen.** Potenzmittel, Längenverzerrung, Wachstumsverlust (Zeitmittel gegen Erwartungswert) und der Anteil der Simpson-Umkehr sind Lehrbuchstoff bzw. stehen in der Literatur; die Quellen wurden für diese Demo **nicht eingesehen**. Es gibt deshalb keinen Literatur-Referenztest (`tests/test_literatur_*.py`): offen. Alle Aussagen sind eigene Messungen und Umformungen, geprüft gegen Handrechnung, Aufzählung, exakte Brüche und `scipy`.
- **Mausklicks auf die Diagramme** (Auswahl) sind nicht Teil der Demo und nicht getestet.
- **Streamlit Cloud nicht gemessen.** Die Live-Budgets stammen aus Messungen auf einem Entwicklungsrechner; auf der Cloud sind sie nicht geprüft.

## Verifikation

- **Kern:** Mittel, Aufgaben, Pfade, Größenverzerrung und Gewichtsmix je als kleine Einheit gegen Handrechnung auf den Mini-Instanzen in `tests/conftest.py` (`test_means.py`, `test_growth.py`, `test_sizebias.py`, `test_mix.py`); Randfälle (genau zwei Elemente, ein Element, alle Werte gleich, Reglergrenzen, Gleichstände in `test_edge_cases.py`).
- **Orakel (anderer Rechenweg):** `test_oracle_means.py` (M_r gegen `scipy.stats.pmean`/`gmean`/`hmean`, mit Gewichten, Gesamtrate als Summenverhältnis), `test_oracle_growth.py` (Aufzählung aller 2^T Pfade, exakter Bruch, Normalverteilung aus `scipy`, Einzelschritte gegen Binomialziehung), `test_oracle_sizebias.py` (Aufzählung kleiner Lose als exakte Brüche, Momente gegen `scipy.stats`, Verteilungen gegen scipy-Wahrscheinlichkeiten), `test_oracle_mix.py` (Aufzählung aller ganzzahligen 2 × 2 × l-Tabellen kleiner Summe mit `fractions`, Anteil der Umkehr gegen 1/60 mit Marge).
- **Zweig-Tests (Nullspalten-Regel):** Jedes Mittel weicht auf mindestens einer Aufgabe vom richtigen ab, jede Aufgabe hat ein anderes richtiges Mittel, jede Verteilungsfamilie ändert den Faktor, die Kopplung κ verschiebt nur die Gesamtquote, gleiche und gemeinsame Gewichte kippen nie (ausdrücklich als erzwungene Probe), Streuung unter und über dem kritischen σ (`test_means.py`, `test_experiments.py`, `test_mix.py`).
- **Eigenschaftstests:** Skalenverhalten, Permutation, Dualität M_(−r)(x) = 1/M_r(1/x), Monotonie, Gruppen- und Schichttausch (`test_eigenschaften_mittel.py`).
- **Messreihen:** Struktur, Aktualität (Prüfsumme der Quelldateien in `test_results_current.py`) und Reproduktion gespeicherter Zellen durch die Live-Funktionen, mit Toleranz für die letzte Binärstelle (`test_results.py`).
- **Zahlen im README:** `tests/test_claims.py` rechnet jede Zahl aus der Messdatei nach und sucht sie im README.
- **Verdikte** als reine Funktionen an handgesetzten Zeilen (`test_verdicts.py`), Presets, Layout (Schnellstart oben, fester Kopf, aktives Preset hervorgehoben, gesperrte Regler), AppTest-Rauchtest mit Eckfall-Matrix, jeder Auswahl und jedem Knopf, PDF, Footer, Laufzeit-Importe, Worst-Case am Slider-Maximum im eigenen Prozess und gleiche Zahlen unter zwei Hash-Seeds.
- **Kopie:** `rgw_rng.py` ist wörtlich aus `reduktionskette-demo/red_scenario.py` kopiert und wird gegen das Original verglichen (`test_copies.py`); der blockweise Strom gegen die Klasse in `test_scenario.py`.
- **Mutationsprüfung** (`tools/mutate_demo.py` aus dem Website-Repo, inkrementell, Zustand in `tests/mutation_state.json`; `--exclude` für `rgw_rng`, `rgw_layout`, `rgw_pdf_export`, `rgw_visualization`, `rgw_constants`, `rgw_presets`, `rgw_scenario`, ohne `test_results_current.py`, `test_layout.py`, `test_pdf.py`): Vollauf über alle **182 Stellen** in `rgw_means.py` (22), `rgw_tasks.py` (17), `rgw_growth.py` (56), `rgw_sizebias.py` (61) und `rgw_mix.py` (26). **Stand beim Abschluss: 161 gefangen, 21 äquivalent, 0 offen.** Im ersten Lauf überlebten 28 Mutanten (Rate der ganzen Suite 85 %, nur der Orakel-Tests 57 %); sieben waren echte Lücken und sind mit Tests geschlossen: die exakten Quantile der Pfade (`exact_quantile`, Randstufe und Obergrenze), die Drift der lognormalen Pfade, der Typ des Ergebnisses von `total_rate` und `true_total` (Python-Zahl für eine Instanz, Feld für einen Stapel) und eine einzelne Periode im Wechsel der Faktoren. Die 21 äquivalenten sind festgehalten: Gleichheit zweier Gleitkommazahlen in den Binomialsummen (der Logarithmus von 1,5 durch den von 0,6 ist irrational, nie genau gleich) und Summanden mit dem Binomialkoeffizienten 0 bzw. Pfade, die nie unter dem Start liegen; `<` gegen `<=` bei einer Gleichverteilten genau 0,5 (Wahrscheinlichkeit 2^−53); die Blockgröße beim Ziehen der Pfade (ändert nur Speicher und Laufzeit); der Anker (Maximum oder Minimum) in `power_mean` für r zwischen −1 und 1 (ändert nur die Überlaufgrenze); `PARETO_MIN ** 2` gegen `** 3` bei PARETO_MIN = 1; die unerreichbare Obergrenze des `clip` in `pieces_from_population` (die Gleichverteilte ist kleiner als 1, das Produkt mit der Summe rundet nie auf die Summe); das Vorzeichen der Normalverteilten in `size_biased_draw` und `lognormal_trajectories` (Spiegelung, gleiche Verteilung) und `α <= 1` gegen `α < 1` (α = 1 löst in der Pareto-Ziehung ohnehin einen Fehler aus). **Nicht mutiert** sind `rgw_experiments.py` und `rgw_evaluation.py` (das Werkzeug erreicht sie über die Orakel nicht; sie sind durch `test_experiments.py`, `test_results.py`, `test_verdicts.py` und `test_claims.py` geprüft, nicht durch Mutation) sowie Szenario, Darstellung und die kopierte `rgw_rng.py`. „Mutation sauber“ gilt nur für die fünf gezogenen Module. Schleifen mit Abbruchbedingung gibt es nicht (nur Schleifen über feste Blöcke), deshalb kein Test mit Zeitgrenze für einen Endlos-Mutanten; Zeitüberschreitungen traten nicht auf.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Hauptansicht je Aufgabe, Experimente E1 bis E5, Grenzen, Mathematische Formulierung |
| `rgw_means.py` | Potenzmittel, gewichtetes Mittel, Gesamtrate |
| `rgw_tasks.py` | Aufgaben mit Zähler und Nenner, wahre Gesamtgröße, Instanzen |
| `rgw_growth.py` | Wachstumspfade, exakte Binomialsummen, typischer Faktor |
| `rgw_sizebias.py` | Verteilungsfamilien der Lose, Stücksicht, Rückrechnung |
| `rgw_mix.py` | Tabellen, Gesamtquote, Umkehr, Standardgewichte, Kopplung |
| `rgw_scenario.py` | Blockweiser SplitMix64-Strom, Verteilungen aus Gleichverteilten |
| `rgw_rng.py` | wörtliche Kopie des Generators (Quelle im Kopf der Datei) |
| `rgw_experiments.py` | Messfunktionen der Experimente E1 bis E5, Prüfsumme der Erzeuger |
| `rgw_results.py` | Zugriff auf die vorgerechneten Messreihen |
| `rgw_evaluation.py` | Einstellungen, Hauptansicht-Analysen, Verdikte als reine Funktionen |
| `rgw_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `rgw_pdf_export.py` | PDF-Export (fpdf2), erst nach Klick |
| `rgw_constants.py`, `rgw_presets.py`, `rgw_layout.py` | Reglergrenzen, Presets, Permalink, fester Kopf |
| `tools/make_results.py` | erzeugt `data/rgw_results.json` aus den Messfunktionen (Teile m1 bis m5, parallel möglich) |
| `data/rgw_results.json` | die vorgerechneten Messreihen mit Prüfsumme der Erzeuger |
| `tests/` | Mini-Instanzen und Einheitentests, Orakel, Eigenschaften, Zweige, Randfälle, Messreihen, Zahlen im README, Verdikte, App, Layout, PDF, Worst-Case |
| `requirements.txt` / `requirements-dev.txt` | Laufzeit (streamlit, numpy, plotly, fpdf2) bzw. mit pytest und scipy (nur Gegenproben) |

## Bewusst nicht umgesetzt

- **Die Wartezeit-Formel für den Takt** (Wartezeit pro Fahrgast, 1 + CV², Orakel B/(B + 1)): `bunching-demo`.
- **Die Kausalmechanik der Simpson-Umkehr** (gemeinsame Ursache, Adjustierung, Standardisierung als Eingriff): `konfundierung-demo` und die Linie Kausale Diagramme und Adjustierung.
- **Korrelation und Abhängigkeitsmaße:** die Linie Korrelation.
- **Kelly-Kriterium, Wachstumsrate unter Einsatz, Ergodizitätsökonomie und Finanzsprache.** Die Faktoren sind Bedarfsfaktoren; die Demo misst Pfade, keine Strategie.
- **Gamma- und Dirichlet-Gewichte** (wie in der Plan-Vormessung): lognormale Gewichte genügen für die Frage nach dem Gewichtsunterschied.
- **Zeitgewichtete Mittel in Zeitreihen** (Zeitmittel eines Prozesses) und **Literatur-Referenztests:** keine Quelle eingesehen.
- **Mittel in anderen Räumen** (Kreis, Verteilungen): Stück 8 (`mitte-im-raum-demo`).
- **Median und Lp-Mitte:** `mittelwert-median-demo` bzw. `lp-mitte-demo` (je ein Satz im Expander der App).

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Die Streamlit-Cloud-Einstellung für die Python-Version ist nicht festgelegt; die Tests laufen in der CI mit Python 3.12 (`.github/workflows/tests.yml`).

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und fpdf2; `scipy` nur für die Gegenproben in `tests/`.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
