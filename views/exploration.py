"""Page 2: Exploratory Relationship Analysis (raw relationships; filters subset student rows).

School-level predictors (facility, teacher experience) are analysed on 30 school rows,
never on 900 pseudo-replicated student rows.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils import charts as C
from utils import data as D
from utils import interpretations as I
from utils import styles as S
from views.common import chart, empty_note, filter_students, gender_filtered_students, selected_school_id

PREDICTORS = {
    "SES": ("ses", "student"),
    "Prior Score": ("prior.score", "student"),
    "Study Hours": ("study.hours", "student"),
    "Facility": ("facility", "school"),
    "Teacher Experience": ("teacher.exp", "school"),
}


def _fit_line(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0:
        return None
    b1, b0 = np.polyfit(x, y, 1)
    return b0, b1


def render(ctx) -> None:
    df, desc = filter_students(ctx)
    sid = selected_school_id(ctx)

    actions = S.header_row(
        "Exploring the Patterns Behind Achievement",
        "Exploring relationships before formal multilevel inference.",
        chips=["Raw associations", "Student & school level kept separate"],
        gold_chip=None if desc == "All students" else f"Filter: {desc}",
        deco="26i_pencil.png", actions_width=2,
    )
    with actions:
        st.selectbox("Predictor", list(PREDICTORS), key="ex_pred",
                     help="Predictor shown in the explorer and the KPI strip")
    pred = st.session_state.get("ex_pred", "SES")
    col, level = PREDICTORS[pred]

    if level == "student":
        xy = df[["student_id", "school_id", col, "math.score"]].dropna()
        xs, ys = xy[col], xy["math.score"]
        n_obs, unit = len(xy), "students"
    else:
        agg = D.school_means_from_students(gender_filtered_students(ctx))
        xs, ys = agg[col], agg["mean_math"]
        n_obs, unit = len(agg), "schools"
    r = I.safe_corr(xs, ys)

    S.kpi_strip([
        dict(label="Selected Predictor", value=pred, icon="tune", tone="forest",
             sub="School level (30 rows)" if level == "school" else "Student level"),
        dict(label="N", value=f"{n_obs:,}", icon="groups", tone="teal", sub=unit),
        dict(label="Correlation with Math", value=I.num(r, 3, na="–"), icon="query_stats", tone="emerald",
             help="Pearson correlation in the current selection. School-level predictors are correlated with school mean math."),
        dict(label=f"Mean {pred}", value=I.num(xs.mean(), na="–"), icon="functions", tone="gold"),
        dict(label=f"SD {pred}", value=I.num(xs.std() if n_obs > 1 else np.nan, na="–"), icon="stacked_bar_chart", tone="blue"),
    ])

    # ---------------- A. Predictor explorer + B. Raw slopes ----------------
    c1, c2 = st.columns(2, gap="small")
    with c1, S.card("ex_scatter"):
        S.card_head("Explorer", f"How Does {pred} Track Achievement?",
                    ("Each point is a student · least-squares line" if level == "student"
                     else "Each point is a school (30 rows, no pseudo-replication) · least-squares line"),
                    tone="emerald", pill_icon="scatter_plot")
        if n_obs < 3:
            empty_note("Select at least three observations to draw a relationship.")
        else:
            fig = go.Figure()
            if level == "student":
                fig.add_trace(go.Scattergl(
                    x=xs, y=ys, mode="markers", name="Students",
                    marker=dict(size=5, color="rgba(15,138,95,0.42)", line=dict(width=0)),
                    customdata=np.stack([xy["student_id"], xy["school_id"]], axis=1),
                    hovertemplate="%{customdata[0]} · School %{customdata[1]}<br>" + pred + ": %{x:.2f}<br>Math: %{y:.2f}<extra></extra>"))
            else:
                colors = [S.GOLD if (sid is not None and s == sid) else S.EMERALD for s in agg["school_id"]]
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="markers", name="Schools",
                    marker=dict(size=[14 if c == S.GOLD else 10 for c in colors], color=colors, line=dict(color="#FFFFFF", width=1.2)),
                    customdata=agg["school_id"],
                    hovertemplate="School %{customdata}<br>" + pred + ": %{x:.2f}<br>Mean math: %{y:.2f}<extra></extra>"))
            line = _fit_line(xs, ys)
            if line is not None:
                gx = np.linspace(xs.min(), xs.max(), 50)
                fig.add_trace(go.Scatter(x=gx, y=line[0] + line[1] * gx, mode="lines", name="Linear fit",
                                         line=dict(color=S.GOLD if level == "student" else S.FOREST_2, width=2.2), hoverinfo="skip"))
            C.base_layout(fig)
            fig.update_xaxes(title_text=pred)
            fig.update_yaxes(title_text="Math score" if level == "student" else "School mean math")
            chart(fig, "ex_scatter")
            slope_txt = f"{line[1]:+.3f} points per unit" if line is not None else "undefined"
            S.insight(f"{pred} shows a <b>{I.corr_strength(r)} {I.direction(r)}</b> raw association with "
                      f"{'math score' if level == 'student' else 'school mean math'} (r = {r:.2f}, n = {n_obs} {unit}; raw slope "
                      f"{slope_txt}). This is unadjusted; adjusted estimates are on the Research Questions page.")

    with c2, S.card("ex_slopes"):
        rs = D.sheet(D.S_RAW_SLOPES).sort_values("raw_hours_slope", ascending=False).reset_index(drop=True)
        rs["lo"] = rs["raw_hours_slope"] - 1.96 * rs["raw_hours_se"]
        rs["hi"] = rs["raw_hours_slope"] + 1.96 * rs["raw_hours_se"]
        rs["label"] = rs["school_id"].map(lambda s: f"S{int(s):02d}")
        S.card_head("Random Slope Motivation", "Does Study Time Pay Off Equally Everywhere?",
                    "Raw within-school slopes of math on study hours (±1.96 SE) · workbook values, full sample",
                    tone="blue", pill_icon="trending_up")
        colors = [S.GOLD if (sid is not None and s == sid) else (S.EMERALD if v >= 0 else S.TERRA)
                  for s, v in zip(rs["school_id"], rs["raw_hours_slope"])]
        fig = go.Figure(go.Scatter(
            x=rs["label"], y=rs["raw_hours_slope"], mode="markers",
            marker=dict(size=[12 if c == S.GOLD else 7 for c in colors], color=colors, line=dict(color="#FFFFFF", width=1)),
            error_y=dict(type="data", symmetric=False, array=rs["hi"] - rs["raw_hours_slope"],
                         arrayminus=rs["raw_hours_slope"] - rs["lo"], color="rgba(62,120,178,0.35)", thickness=1.4, width=0),
            customdata=np.stack([rs["raw_hours_se"], rs["raw_hours_p"], rs["lo"], rs["hi"]], axis=1),
            hovertemplate="<b>School %{x}</b><br>Raw slope: %{y:.3f}<br>SE: %{customdata[0]:.3f}<br>p = %{customdata[1]:.3f}"
                          "<br>±1.96 SE: [%{customdata[2]:.3f}, %{customdata[3]:.3f}]<extra></extra>"))
        fig.add_hline(y=0, line=dict(color="rgba(11,61,44,0.4)", width=1, dash="dot"))
        mslope = rs["raw_hours_slope"].mean()
        fig.add_hline(y=mslope, line=dict(color=S.GOLD, width=1.5, dash="dash"), annotation_text=f"mean {mslope:.2f}",
                      annotation_position="top right", annotation_font=dict(size=10, color=S.MUTED))
        C.base_layout(fig)
        fig.update_xaxes(tickangle=-90, tickfont=dict(size=9), showgrid=False)
        fig.update_yaxes(title_text="Points per extra hour")
        chart(fig, "ex_slopes")
        npos = int((rs["raw_hours_slope"] > 0).sum())
        nsig = int((rs["raw_hours_p"] < 0.05).sum())
        txt = (f"Raw slopes range from <b>{rs['raw_hours_slope'].min():.2f}</b> to <b>{rs['raw_hours_slope'].max():.2f}</b>; "
               f"{npos} of {len(rs)} are positive and {nsig} reach p &lt; 0.05 alone. The spread motivates a random "
               "study-hours slope, though part of it is sampling noise with 30 students per school.")
        if sid is not None:
            rr = rs[rs["school_id"] == sid].iloc[0]
            txt = f"<b>School {sid:02d}</b>: raw slope {rr['raw_hours_slope']:+.3f} ({I.pval(rr['raw_hours_p'])}). " + txt
        S.insight(txt)

    # ---------------- C. Gender + D. Correlations ----------------
    c3, c4 = st.columns([5, 7], gap="small")
    with c3, S.card("ex_gender"):
        S.card_head("Control Variable", "Mathematics Achievement by Gender",
                    f"Violin + box with jittered students · {desc}", tone="mint", pill_icon="wc")
        if len(df) < 2:
            empty_note("Select at least two students to compare distributions.")
        else:
            fig = go.Figure()
            gcol = {"Laki-laki": S.BLUE, "Perempuan": S.EMERALD}
            for g in sorted(df["gender"].unique()):
                sub = df[df["gender"] == g]
                fig.add_trace(go.Violin(
                    y=sub["math.score"], x=[g] * len(sub), name=g, side="positive", width=0.7,
                    box_visible=True, meanline_visible=True, points="all", pointpos=-0.45, jitter=0.25,
                    line=dict(color=gcol.get(g, S.EMERALD), width=1.4), fillcolor=gcol.get(g, S.EMERALD), opacity=0.6,
                    marker=dict(size=3, color=gcol.get(g, S.EMERALD), opacity=0.5),
                    customdata=sub["student_id"], spanmode="hard",
                    hovertemplate="%{customdata}<br>Math: %{y:.2f}<extra>" + g + "</extra>"))
            C.base_layout(fig)
            fig.update_yaxes(title_text="Math score")
            fig.update_xaxes(showgrid=False)
            chart(fig, "ex_gender")
            means = df.groupby("gender")["math.score"].agg(["mean", "count"])
            fx = D.sheet(D.S_FINAL_FIXED)
            fem = fx[fx["predictor"] == "female"]
            if {"Perempuan", "Laki-laki"} <= set(means.index):
                d = means.loc["Perempuan", "mean"] - means.loc["Laki-laki", "mean"]
                txt = (f"Raw means: Perempuan <b>{means.loc['Perempuan', 'mean']:.2f}</b> vs Laki-laki "
                       f"<b>{means.loc['Laki-laki', 'mean']:.2f}</b> (gap {d:+.2f}). ")
            else:
                g0 = means.index[0]
                txt = f"Only {g0} students are selected (mean {means.loc[g0, 'mean']:.2f}). "
            if not fem.empty:
                txt += (f"Adjusted female coefficient in the final model: {fem['estimate'].iloc[0]:+.3f}. "
                        "Gender is observed, so neither gap is causal.")
            S.insight(txt)

    with c4, S.card("ex_corr"):
        S.card_head("Correlation", "How Are the Variables Related?",
                    "Pearson correlations · student level (filtered rows) or school level (30 school rows)",
                    tone="slate", pill_icon="grid_on")
        lvl = st.segmented_control("Level", ["Student Level", "School Level"], default="Student Level",
                                   key="ex_corr_level", label_visibility="collapsed") or "Student Level"
        if lvl == "Student Level":
            labels = list(D.STUDENT_VARS.values())
            if desc == "All students":
                cm = D.sheet(D.S_CORR_STUDENT).set_index("variable").loc[list(D.STUDENT_VARS), list(D.STUDENT_VARS)]
                src = "workbook, full sample (n = 900)"
            elif len(df) >= 3:
                cm = df[list(D.STUDENT_VARS)].corr()
                src = f"{len(df)} filtered students"
            else:
                cm, src = None, ""
        else:
            labels = list(D.SCHOOL_VARS.values())
            cm = D.sheet(D.S_CORR_SCHOOL).set_index("variable").loc[list(D.SCHOOL_VARS), list(D.SCHOOL_VARS)]
            src = "workbook, 30 schools (filters do not apply)"
        if cm is None:
            empty_note("Select at least three students for a student-level correlation matrix.")
        else:
            z = cm.to_numpy(dtype=float)
            fig = go.Figure(go.Heatmap(
                z=z, x=labels, y=labels, zmin=-1, zmax=1, zmid=0, xgap=2, ygap=2,
                colorscale=[[0, "#B9573E"], [0.5, "#F4F2EC"], [1, "#0B6B48"]],
                text=np.vectorize(lambda v: "" if np.isnan(v) else f"{v:.2f}")(z), texttemplate="%{text}",
                textfont=dict(size=10), colorbar=dict(thickness=8, outlinewidth=0, tickfont=dict(size=9)),
                hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>"))
            C.base_layout(fig)
            fig.update_layout(margin=dict(l=80))
            fig.update_yaxes(autorange="reversed", showgrid=False)
            fig.update_xaxes(showgrid=False)
            chart(fig, "ex_corr")
            others = [(labels[i], z[0, i]) for i in range(1, len(labels)) if not np.isnan(z[0, i])]
            best = max(others, key=lambda t: abs(t[1])) if others else None
            txt = f"Source: {src}. "
            if best:
                txt += f"{labels[0]} correlates most with <b>{best[0]}</b> (r = {best[1]:.2f}). "
            txt += ("With only 30 schools these correlations are uncertain and must not be read as 900 observations."
                    if lvl == "School Level" else "These raw associations ignore the nesting of students in schools.")
            S.insight(txt)
