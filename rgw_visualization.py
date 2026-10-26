"""Plotly-Abbildungen: alle Achsen fest (`fixedrange`), Beschriftungen als Annotationen, Farben aus rgw_constants.py."""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go

import rgw_constants as C
import rgw_growth as G
import rgw_means as M
import rgw_sizebias as Z

FAMILY_SHORT = {"poisson": "Poisson", "geometrisch": "geometrisch", "lognormal": "lognormal", "pareto": "Pareto"}
LAYER_TITLES = ("Große Standorte", "Kleine Standorte", "Gesamt (eigene Gewichte)", "Gesamt (gemeinsame Gewichte)")


def _layout(fig: go.Figure, xtitle: str = "", ytitle: str = "", height: int = 380, legend_below: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=55, r=20, t=30, b=95 if legend_below else 40), xaxis_title=xtitle, yaxis_title=ytitle,
        legend=dict(orientation="h", yanchor="top", y=-0.22, x=0) if legend_below else None, dragmode=False,
    )
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


# ---------------------------------------------------------------- Hauptansicht: Raten

def build_rate_curve(a) -> go.Figure:
    """Das Potenzmittel M_r über dem Exponenten r, die vier Mittel als Rauten, das gewählte r als Kreis und die wahre Gesamtgröße als Linie; links die Einzelraten."""
    fig = go.Figure()
    left = C.R_MIN - 0.45
    fig.add_trace(go.Scatter(x=[left] * len(a.x), y=a.x, mode="markers", name="Einzelraten", marker=dict(size=7, color=C.COLOR_DATA), hovertemplate="%{y:.3g}<extra>Einzelrate</extra>"))
    fig.add_trace(go.Scatter(x=list(C.R_CURVE), y=a.curve, mode="lines", name="Potenzmittel M_r", line=dict(color=C.COLOR_LINE, width=2), hovertemplate="r = %{x:.1f}: %{y:.4g}<extra></extra>"))
    fig.add_hline(y=a.truth, line=dict(color=C.COLOR_TRUE, width=2, dash="dash"))
    fig.add_annotation(x=C.R_MAX, y=a.truth, text="wahre Gesamtgröße", showarrow=False, xanchor="right", yanchor="bottom", font=dict(color=C.COLOR_TRUE, size=12))
    for name, r in M.MEAN_R.items():
        fig.add_trace(go.Scatter(x=[r], y=[a.means[name]], mode="markers", name=f"{name} (r = {r:g})",
                                 marker=dict(size=13, symbol="diamond", color=C.MEAN_COLORS[name], line=dict(color="white", width=1)),
                                 hovertemplate=f"{name}: %{{y:.4g}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[a.settings.r], y=[a.selected], mode="markers", name=f"gewähltes r = {a.settings.r:g}", marker=dict(size=11, symbol="circle-open", color=C.COLOR_TRUE, line=dict(width=3)),
                             hovertemplate="r = %{x:.1f}: %{y:.4g}<extra></extra>"))
    fig.update_xaxes(range=[left - 0.3, C.R_MAX + 0.2], tickmode="array", tickvals=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    return _layout(fig, "Mittelungsexponent r", f"Mittel ({a.task.unit})", 420)


def build_error_bars(rows: list[dict]) -> go.Figure:
    """Abweichung der vier Mittel von der wahren Gesamtgröße in Prozent (Balken); das richtige Mittel der Aufgabe ist schraffiert."""
    fig = go.Figure()
    for row in rows:
        fig.add_trace(go.Bar(x=[row["Abweichung (%)"]], y=[row["Mittel"]], orientation="h", name=row["Mittel"], marker=dict(color=C.MEAN_COLORS[row["Mittel"].lower()],
                             pattern=dict(shape="/" if row["richtig"] else "")), showlegend=False,
                             hovertemplate="%{x:.1f} %<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, "Abweichung von der wahren Gesamtgröße (%)", "", 260, legend_below=False)


# ---------------------------------------------------------------- Hauptansicht: Lose

def build_lots(a) -> go.Figure:
    """Losgrößen der Grundgesamtheit gegen die beobachteten Größen (Stücksicht), logarithmische x-Achse; Linien: Mittel pro Los, arithmetisches und harmonisches Mittel der Beobachtungen."""
    lo = max(float(a.sizes.min()), 1e-9)
    hi = float(max(a.sizes.max(), a.observed.max()))
    edges = np.geomspace(lo, hi * 1.0001, 31)
    centre, width = 0.5 * (edges[1:] + edges[:-1]), np.diff(edges)
    fig = go.Figure()
    for data, name, color in ((a.sizes, "Lose (Grundgesamtheit)", C.COLOR_A), (a.observed, "beobachtet: Größe des Loses eines gezogenen Stücks", C.COLOR_B)):
        counts, _ = np.histogram(data, bins=edges)
        fig.add_trace(go.Bar(x=centre, y=counts / counts.sum(), width=width, name=name, marker=dict(color=color, opacity=0.55), hovertemplate="%{x:.3g}: %{y:.1%}<extra></extra>"))
    for value, label, color, dash in ((a.truth, "Mittel pro Los", C.COLOR_TRUE, "solid"), (a.arithmetic, "arithmetisch", C.MEAN_COLORS["arithmetisch"], "dash"), (a.harmonic, "harmonisch", C.MEAN_COLORS["harmonisch"], "dot")):
        fig.add_vline(x=value, line=dict(color=color, width=2, dash=dash))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(type="log")
    fig.add_annotation(x=math.log10(a.truth), y=1.0, yref="paper", text="Mittel pro Los", showarrow=False, yanchor="bottom", font=dict(size=11))
    fig.add_annotation(x=math.log10(a.arithmetic), y=0.93, yref="paper", text="arithmetisch", showarrow=False, yanchor="bottom", xanchor="left", font=dict(size=11, color=C.MEAN_COLORS["arithmetisch"]))
    fig.add_annotation(x=math.log10(a.harmonic), y=0.86, yref="paper", text="harmonisch", showarrow=False, yanchor="bottom", xanchor="right", font=dict(size=11, color=C.MEAN_COLORS["harmonisch"]))
    return _layout(fig, "Losgröße (Stück, logarithmisch)", "Anteil", 400)


# ---------------------------------------------------------------- Hauptansicht: Tabelle

def build_table_chart(a) -> go.Figure:
    """Quoten von A und B je Schicht und insgesamt, mit eigenen und mit gemeinsamen Gewichten."""
    groups = [a.rates[0], a.rates[1], a.pooled, a.standard]
    fig = go.Figure()
    for g, (name, color) in enumerate((("Gruppe A", C.COLOR_A), ("Gruppe B", C.COLOR_B))):
        values = [float(100.0 * x[g]) for x in groups]
        fig.add_trace(go.Bar(x=list(LAYER_TITLES), y=values, name=name, marker=dict(color=color), text=[f"{v:.0f} %" for v in values], textposition="outside", hovertemplate="%{y:.1f} %<extra>" + name + "</extra>"))
    fig.update_yaxes(range=[0, 100])
    fig.update_layout(barmode="group")
    return _layout(fig, "", "Erfolgsquote (%)", 380)


# ---------------------------------------------------------------- E1

def build_cost_by_sigma(cells: list[dict], task_label: str) -> go.Figure:
    """Mittlere Abweichung der vier Mittel von der wahren Gesamtgröße über der Streuung σ (Linien); das richtige Mittel liegt auf der Nulllinie."""
    fig = go.Figure()
    for name in M.MEAN_R:
        fig.add_trace(go.Scatter(x=[c["sigma"] for c in cells], y=[c["abs"][name]["mean"] for c in cells], mode="lines+markers", name=name, line=dict(color=C.MEAN_COLORS[name], width=2),
                                 hovertemplate="σ = %{x}: %{y:.1f} %<extra>" + name + "</extra>"))
    return _layout(fig, "Streuung σ der Logarithmen", "mittlere Abweichung (%)", 380)


def build_cost_by_r(curves: list[dict], task: str) -> go.Figure:
    """Mittlere Abweichung des Potenzmittels M_r über dem Exponenten r für drei Streuungen: das Minimum liegt beim richtigen Mittel der Aufgabe."""
    fig = go.Figure()
    for curve, color in zip(curves, ("#4c78a8", "#e08a1e", "#d62728")):
        fig.add_trace(go.Scatter(x=curve["rs"], y=[c["mean"] for c in curve["cost"]], mode="lines", name=f"σ = {curve['sigma']:g}", line=dict(color=color, width=2),
                                 hovertemplate="r = %{x}: %{y:.1f} %<extra>σ = " + f"{curve['sigma']:g}" + "</extra>"))
    return _layout(fig, "Mittelungsexponent r", "mittlere Abweichung (%)", 380)


# ---------------------------------------------------------------- E2

def build_paths_chart(horizon: int, rows: list[dict], live: dict | None) -> go.Figure:
    """Zweipunkt-Faktoren 1,5 / 0,6 (logarithmische y-Achse): Erwartungswert 1,05^T, exakte Quantile 10 / 50 / 90 % und der Startwert 1; Punkte: Stichprobenmittel von 100 000 Pfaden
    (vorgerechnet, ein Punkt je Seed); Linien: Pfade der Live-Messung, falls vorhanden."""
    ts = np.arange(0, horizon + 1)
    fig = go.Figure()
    q10 = [G.exact_quantile(int(t), 0.10) if t > 0 else 1.0 for t in ts]
    q50 = [G.exact_quantile(int(t), 0.50) if t > 0 else 1.0 for t in ts]
    q90 = [G.exact_quantile(int(t), 0.90) if t > 0 else 1.0 for t in ts]
    fig.add_trace(go.Scatter(x=ts, y=q90, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ts, y=q10, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=C.COLOR_BAND, name="10 % bis 90 % der Pfade (exakt)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ts, y=q50, mode="lines", name="Median (exakt)", line=dict(color=C.COLOR_LINE, width=2), hovertemplate="T = %{x}: %{y:.3g}<extra>Median</extra>"))
    fig.add_trace(go.Scatter(x=ts, y=[G.expected_value(int(t)) for t in ts], mode="lines", name="Erwartungswert 1,05^T", line=dict(color=C.MEAN_COLORS["arithmetisch"], width=2, dash="dash"),
                             hovertemplate="T = %{x}: %{y:.3g}<extra>Erwartungswert</extra>"))
    fig.add_hline(y=1.0, line=dict(color="rgba(120,120,120,0.7)", width=1))
    if live is not None:
        for i, path in enumerate(live["paths"][:12]):
            fig.add_trace(go.Scatter(x=np.arange(len(path)), y=path, mode="lines", line=dict(color="rgba(120,120,120,0.35)", width=1), name="Pfade der Live-Messung", showlegend=i == 0, hoverinfo="skip"))
    points = [(r["T"], seed_ratio * r["expected"]) for r in rows if r["T"] <= horizon for seed_ratio in r["mean_ratio"]["per_seed"]]
    if points:
        fig.add_trace(go.Scatter(x=[p[0] for p in points], y=[p[1] for p in points], mode="markers", name="Stichprobenmittel von 100.000 Pfaden (ein Punkt je Seed)",
                                 marker=dict(size=8, color=C.MEAN_COLORS["quadratisch"], line=dict(color="white", width=1)), hovertemplate="T = %{x}: %{y:.3g}<extra>Stichprobenmittel</extra>"))
    fig.update_yaxes(type="log")
    return _layout(fig, "Perioden T", "Wert (Start = 1, logarithmisch)", 440)


def build_lognormal_chart(rows: list[dict]) -> go.Figure:
    """Anteil der Pfade unter dem Start (Balken) und typischer Faktor je Periode (Linie) über der Streuung σ der lognormalen Faktoren."""
    rows = sorted(rows, key=lambda r: r["sigma"])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"{r['sigma']:.3g}".replace(".", ",") for r in rows], y=[100 * r["below"]["mean"] for r in rows], name="Pfade unter dem Start (%)", marker=dict(color=C.COLOR_LINE),
                         hovertemplate="%{y:.1f} %<extra></extra>"))
    fig.add_trace(go.Scatter(x=[f"{r['sigma']:.3g}".replace(".", ",") for r in rows], y=[r["typical"]["mean"] for r in rows], mode="lines+markers", name="typischer Faktor je Periode", yaxis="y2",
                             line=dict(color=C.MEAN_COLORS["arithmetisch"], width=2), hovertemplate="%{y:.4f}<extra></extra>"))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0.7, 1.1], fixedrange=True, title="Faktor je Periode"))
    _layout(fig, "Streuung σ (Erwartungswert je Periode 1,05)", "Pfade unter dem Start (%)", 380)
    fig.update_layout(yaxis=dict(range=[0, 100], fixedrange=True))
    return fig


def build_coverage_chart(rows: list[dict], family: str) -> go.Figure:
    """Anteil der Stichproben, deren Mittel den Erwartungswert auf ± 20 % trifft, über den Perioden T für drei Stichprobengrößen."""
    fig = go.Figure()
    for n, color in zip(sorted({r["N"] for r in rows}), ("#4c78a8", "#e08a1e", "#2F6B65")):
        sel = sorted([r for r in rows if r["family"] == family and r["N"] == n], key=lambda r: r["T"])
        fig.add_trace(go.Scatter(x=[r["T"] for r in sel], y=[100 * r["within"]["mean"] for r in sel], mode="lines+markers", name=f"{n:,} Pfade".replace(",", "."), line=dict(color=color, width=2),
                                 hovertemplate="T = %{x}: %{y:.1f} %<extra></extra>"))
    fig.update_yaxes(range=[0, 105])
    return _layout(fig, "Perioden T", "Stichproben, die den Erwartungswert treffen (%)", 380)


# ---------------------------------------------------------------- E3

def build_bias_chart(rows: list[dict]) -> go.Figure:
    """Verzerrungsfaktor Stücksicht / Lossicht je Familie: gemessen (Balken) und 1 + CV² der Formel (Raute); Pareto mit α ≤ 2 hat keinen."""
    labels = [f"{FAMILY_SHORT[r['family']]} {r['par']:g}".replace(".", ",") for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[r["measured"]["mean"] for r in rows], name="gemessen (vier Seeds)", marker=dict(color=C.COLOR_B), hovertemplate="%{y:.2f}<extra>gemessen</extra>"))
    fig.add_trace(go.Scatter(x=labels, y=[None if math.isinf(r["formula"]) else r["formula"] for r in rows], mode="markers", name="Formel 1 + CV²",
                             marker=dict(size=11, symbol="diamond", color=C.COLOR_TRUE), hovertemplate="%{y:.2f}<extra>Formel</extra>"))
    fig.update_yaxes(type="log")
    return _layout(fig, "Familie und Parameter", "Verhältnis Stücksicht / Lossicht (logarithmisch)", 380)


def build_back_chart(rows: list[dict]) -> go.Figure:
    """Verzerrung der Rückrechnung in Prozent des wahren Mittels über der Zahl der Beobachtungen: arithmetisches (rot) und harmonisches Mittel (violett) mit RMSE als Band."""
    rows = sorted(rows, key=lambda r: r["n"])
    ns = [r["n"] for r in rows]
    fig = go.Figure()
    for key, rmse, name, color in (("am_bias", "am_rmse", "arithmetisch", C.MEAN_COLORS["arithmetisch"]), ("hm_bias", "hm_rmse", "harmonisch", C.MEAN_COLORS["harmonisch"])):
        fig.add_trace(go.Scatter(x=ns, y=[r[key]["mean"] for r in rows], mode="lines+markers", name=f"{name}: Verzerrung", line=dict(color=color, width=2), hovertemplate="n = %{x}: %{y:.1f} %<extra>" + name + "</extra>"))
        fig.add_trace(go.Scatter(x=ns, y=[r[rmse]["mean"] for r in rows], mode="lines+markers", name=f"{name}: RMSE", line=dict(color=color, width=1.5, dash="dot"), hovertemplate="n = %{x}: %{y:.1f} %<extra>RMSE " + name + "</extra>"))
    fig.update_xaxes(type="log", tickvals=ns)
    fig.update_yaxes(type="log")
    return _layout(fig, "Beobachtete Stücke n (logarithmisch)", "Prozent des wahren Mittels pro Los (logarithmisch)", 400)


def build_tail_chart(rows: list[dict]) -> go.Figure:
    """Pareto-Losgrößen: Median des arithmetischen Mittels der Beobachtungen über n (beide Achsen logarithmisch); für α > 2 strichliert der Wert (α − 1)/(α − 2)."""
    fig = go.Figure()
    for row, color in zip(rows, ("#d62728", "#e08a1e", "#4c78a8", "#2F6B65")):
        ns = [r["n"] for r in row["rows"]]
        fig.add_trace(go.Scatter(x=ns, y=[r["median"]["mean"] for r in row["rows"]], mode="lines+markers", name=f"α = {row['alpha']:g}".replace(".", ","), line=dict(color=color, width=2),
                                 hovertemplate="n = %{x}: %{y:.3g}<extra>α = " + f"{row['alpha']:g}" + "</extra>"))
        if row["formula"] is not None:
            fig.add_trace(go.Scatter(x=[ns[0], ns[-1]], y=[row["formula"]] * 2, mode="lines", line=dict(color=color, width=1, dash="dash"), showlegend=False, hoverinfo="skip"))
    fig.update_xaxes(type="log", tickvals=[100, 10_000, 1_000_000])
    fig.update_yaxes(type="log")
    return _layout(fig, "Beobachtete Stücke n", "Median des arithmetischen Mittels (logarithmisch)", 380)


# ---------------------------------------------------------------- E4

def build_reversal_chart(rows: list[dict]) -> go.Figure:
    """Anteil der zufälligen Tabellen mit Simpson-Umkehr über der Zahl der Schichten (logarithmische y-Achse; null fehlt)."""
    rows = sorted(rows, key=lambda r: r["layers"])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[str(r["layers"]) for r in rows], y=[100 * r["share"]["mean"] for r in rows], marker=dict(color=C.COLOR_LINE), name="Tabellen mit Umkehr (%)",
                         hovertemplate="%{y:.4f} %<extra></extra>"))
    fig.update_yaxes(type="log")
    return _layout(fig, "Schichten", "Tabellen mit Umkehr (%, logarithmisch)", 360, legend_below=False)


def build_weights_chart(rows: list[dict]) -> go.Figure:
    """Anteil der Tabellen, in denen die Gesamtquote trotz Dominanz von A in jeder Schicht zugunsten von B kippt, über der Gewichtsstreuung τ; je eine Linie je Schichtzahl."""
    fig = go.Figure()
    for k, color in zip(sorted({r["k"] for r in rows}), ("#4c78a8", "#e08a1e", "#2F6B65")):
        sel = sorted([r for r in rows if r["k"] == k], key=lambda r: r["tau"])
        fig.add_trace(go.Scatter(x=[r["tau"] for r in sel], y=[100 * r["share"]["mean"] for r in sel], mode="lines+markers", name=f"{k} Schichten", line=dict(color=color, width=2),
                                 hovertemplate="τ = %{x}: %{y:.2f} %<extra></extra>"))
    return _layout(fig, "Streuung τ der Gewichte (0 = gleiche Gewichte)", "Tabellen mit Umkehr (%)", 360)


# ---------------------------------------------------------------- E5

def build_coupling_chart(cells: list[dict]) -> go.Figure:
    """Anteil der Vergleiche, in denen Gruppe B vorn liegt, über der Kopplung κ_B: ungewichtetes Mittel der Quoten gegen Gesamtquote (gleiche wahre Standortquote in beiden Gruppen)."""
    cells = sorted(cells, key=lambda c: c["kappa_b"])
    labels = [f"{c['kappa_b']:g}".replace(".", ",") for c in cells]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[100 * c["b_ahead_mean"]["mean"] for c in cells], name="Mittel der Quoten (ungewichtet)", marker=dict(color=C.COLOR_DATA)))
    fig.add_trace(go.Bar(x=labels, y=[100 * c["b_ahead_pooled"]["mean"] for c in cells], name="Gesamtquote", marker=dict(color=C.COLOR_B)))
    fig.add_hline(y=50, line=dict(color=C.COLOR_TRUE, width=1, dash="dash"))
    fig.update_yaxes(range=[0, 100])
    fig.update_layout(barmode="group")
    return _layout(fig, "Kopplung κ_B von Nachfrage und Quote in Gruppe B", "B liegt vorn in (%) der Vergleiche", 360)
