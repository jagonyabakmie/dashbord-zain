"""Plotly helpers: one transparent theme and a few reusable statistical chart forms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from . import styles as S

CONFIG = {"displaylogo": False, "responsive": True,
          "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d", "toggleSpikelines"]}


def base_layout(fig: go.Figure, height: int = 360, legend: bool = False, **kw) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, Segoe UI, system-ui, sans-serif", size=12, color=S.INK_2),
        margin=dict(l=64, r=18, t=24, b=56),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="rgba(15,138,95,0.35)",
                        font=dict(family="Plus Jakarta Sans, system-ui, sans-serif", size=12, color=S.INK)),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=S.INK_2)),
        bargap=0.06,
    )
    axis = dict(automargin=True, gridcolor=S.GRID, zeroline=False, linecolor="rgba(11,61,44,0.18)", ticks="",
                title_font=dict(size=12, color=S.MUTED), tickfont=dict(size=11, color=S.MUTED))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    if kw:
        fig.update_layout(**kw)
    return fig


def forest(df: pd.DataFrame, label_col: str, est_col: str, lo_col: str | None, hi_col: str | None,
           color_col: str | None = None, x_title: str = "Estimate", height: int | None = None,
           hover_extra: str | None = None, ref: float = 0.0) -> go.Figure:
    """Horizontal coefficient plot; df is plotted top-to-bottom in given order."""
    d = df.iloc[::-1].reset_index(drop=True)
    colors = d[color_col] if color_col else [S.EMERALD] * len(d)
    fig = go.Figure()
    if lo_col and hi_col and d[lo_col].notna().any():
        for i, r in d.iterrows():
            if pd.notna(r[lo_col]) and pd.notna(r[hi_col]):
                fig.add_trace(go.Scatter(x=[r[lo_col], r[hi_col]], y=[r[label_col]] * 2, mode="lines",
                                         line=dict(color=colors[i], width=3), hoverinfo="skip", showlegend=False))
    custom = np.stack([d[lo_col] if lo_col else np.full(len(d), np.nan),
                       d[hi_col] if hi_col else np.full(len(d), np.nan),
                       d[hover_extra] if hover_extra else np.full(len(d), "")], axis=1)
    fig.add_trace(go.Scatter(
        x=d[est_col], y=d[label_col], mode="markers",
        marker=dict(size=13, color=list(colors), line=dict(color="#FFFFFF", width=2)),
        customdata=custom,
        hovertemplate="<b>%{y}</b><br>Estimate: %{x:.4f}<br>95% CI: [%{customdata[0]:.4f}, %{customdata[1]:.4f}]"
                      + ("<br>%{customdata[2]}" if hover_extra else "") + "<extra></extra>",
        showlegend=False))
    fig.add_vline(x=ref, line=dict(color="rgba(11,61,44,0.35)", width=1, dash="dot"))
    base_layout(fig, height=height or max(220, 64 * len(d) + 60))
    fig.update_xaxes(title_text=x_title)
    fig.update_yaxes(showgrid=False)
    return fig


def qq(values: np.ndarray, labels: list[str] | None = None,
       color: str = S.EMERALD, height: int = 340, name: str = "Value") -> go.Figure:
    """Normal Q-Q plot using Blom plotting positions; reference line through quartiles."""
    v = np.asarray(values, dtype=float)
    ok = ~np.isnan(v)
    v = v[ok]
    labs = np.asarray(labels)[ok] if labels is not None else None
    order = np.argsort(v)
    sv = v[order]
    n = len(sv)
    p = (np.arange(1, n + 1) - 0.375) / (n + 0.25)
    theo = stats.norm.ppf(p)
    q1, q3 = np.percentile(sv, [25, 75])
    z1, z3 = stats.norm.ppf([0.25, 0.75])
    slope = (q3 - q1) / (z3 - z1)
    icpt = q1 - slope * z1
    fig = go.Figure()
    xr = np.array([theo.min() - 0.2, theo.max() + 0.2])
    fig.add_trace(go.Scatter(x=xr, y=icpt + slope * xr, mode="lines",
                             line=dict(color=S.GOLD, width=2, dash="dash"), name="Normal reference",
                             hoverinfo="skip"))
    sl = labs[order] if labs is not None else np.array([""] * n)
    marker_size = 9 if n <= 60 else 5
    fig.add_trace(go.Scatter(x=theo, y=sv, mode="markers", name=name,
                             marker=dict(size=marker_size, color=color, opacity=0.8 if n > 60 else 0.95,
                                         line=dict(color="#FFFFFF", width=1 if n <= 60 else 0)),
                             customdata=sl,
                             hovertemplate="%{customdata}<br>Theoretical: %{x:.2f}<br>Observed: %{y:.3f}<extra></extra>"))
    base_layout(fig, height=height)
    fig.update_xaxes(title_text="Theoretical normal quantile")
    fig.update_yaxes(title_text="Sample quantile")
    return fig


def add_qq_highlight(fig: go.Figure, values, labels, target_label: str) -> None:
    """Mark one labeled point (e.g. a selected school) on a Q-Q plot in gold."""
    v = np.asarray(values, dtype=float)
    labs = np.asarray(labels)
    ok = ~np.isnan(v)
    v, labs = v[ok], labs[ok]
    order = np.argsort(v)
    n = len(v)
    theo = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
    for rank, idx in enumerate(order):
        if labs[idx] == target_label:
            fig.add_trace(go.Scatter(x=[theo[rank]], y=[v[idx]], mode="markers+text", text=[target_label],
                                     textposition="top left", textfont=dict(color=S.INK, size=11),
                                     marker=dict(size=15, color=S.GOLD, line=dict(color="#FFFFFF", width=2)),
                                     hoverinfo="skip", showlegend=False))


def kde_curve(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 3 or np.std(x) == 0:
        return np.zeros_like(grid)
    return stats.gaussian_kde(x)(grid)
