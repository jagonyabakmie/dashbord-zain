"""Page 2: Exploratory Relationship Analysis (raw relationships; filters subset student rows).

School-level predictors (facility, teacher experience) are analysed on 30 school rows,
never on 900 pseudo-replicated student rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
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

    S.page_header(
        "Exploring the Patterns Behind Achievement",
        "Exploring relationships before formal multilevel inference.",
        kicker="Exploratory Relationship Analysis",
        chips=["Raw associations", "Student & school level kept separate"],
        gold_chip=None if desc == "All students" else f"Filter: {desc}",
        deco="26i_pencil.png",
    )

    pred = st.session_state.get("ex_pred", "SES")
    col, level = PREDICTORS[pred]

    # Data for the selected predictor at its proper analytical level
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

    # ---------------- A. Predictor explorer ----------------
    c1, c2 = st.columns([7, 5], gap="medium")
    with c1, S.card("card_ex_scatter"):
        S.card_head("Explorer", "How Does Each Predictor Track Achievement?",
                    "Scatter with least-squares line · student predictors use student rows; school predictors use one row per school",
                    tone="emerald", pill_icon="scatter_plot")
        st.selectbox("Predictor", list(PREDICTORS), key="ex_pred")
        fig = go.Figure()
        if level == "student":
            if n_obs < 3:
                empty_note("Select at least three students to draw a relationship.")
            else:
                fig.add_trace(go.Scattergl(
                    x=xs, y=ys, mode="markers", name="Students",
                    marker=dict(size=6, color="rgba(15,138,95,0.45)", line=dict(width=0)),
                    customdata=np.stack([xy["student_id"], xy["school_id"]], axis=1),
                    hovertemplate="%{customdata[0]} · School %{customdata[1]}<br>" + pred + ": %{x:.2f}<br>Math: %{y:.2f}<extra></extra>"))
        else:
            colors = [S.GOLD if (sid is not None and s == sid) else S.EMERALD for s in agg["school_id"]]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers", name="Schools",
                marker=dict(size=[16 if c == S.GOLD else 11 for c in colors], color=colors, line=dict(color="#FFFFFF", width=1.5)),
                customdata=agg["school_id"],
                hovertemplate="School %{customdata}<br>" + pred + ": %{x:.2f}<br>Mean math: %{y:.2f}<extra></extra>"))
        line = _fit_line(xs, ys) if n_obs >= 3 else None
        if line is not None:
            gx = np.linspace(xs.min(), xs.max(), 50)
            fig.add_trace(go.Scatter(x=gx, y=line[0] + line[1] * gx, mode="lines", name="Linear fit",
                                     line=dict(color=S.GOLD if level == "student" else S.FOREST_2, width=2.5), hoverinfo="skip"))
        if n_obs >= 3:
            C.base_layout(fig, height=360)
            fig.update_xaxes(title_text=pred)
            fig.update_yaxes(title_text="Mathematics score" if level == "student" else "School mean mathematics score")
            chart(fig, "ex_scatter")
            slope_txt = f"{line[1]:+.3f} points per unit" if line is not None else "undefined"
            lvl_txt = ("Each point is a student." if level == "student"
                       else "Each point is a school, because this characteristic varies only between schools; using 900 student rows would overstate the evidence.")
            S.insight(
                f"In this selection, {pred} shows a <b>{I.corr_strength(r)} {I.direction(r)}</b> raw association with "
                f"{'mathematics score' if level == 'student' else 'school mean mathematics'} (r = {r:.2f}, n = {n_obs} {unit}); "
                f"the fitted raw slope is {slope_txt}. {lvl_txt} This is an unadjusted pattern; adjusted estimates are on the Research Questions page."
            )

    # ---------------- B. Raw study-hours slopes ----------------
    with c2, S.card("card_ex_slopes"):
        rs = D.sheet(D.S_RAW_SLOPES).sort_values("raw_hours_slope").reset_index(drop=True)
        rs["lo"] = rs["raw_hours_slope"] - 1.96 * rs["raw_hours_se"]
        rs["hi"] = rs["raw_hours_slope"] + 1.96 * rs["raw_hours_se"]
        rs["label"] = rs["school_id"].map(lambda s: f"School {int(s):02d}")
        S.card_head("Random Slope Motivation", "Does Study Time Pay Off Equally Everywhere?",
                    "Raw within-school OLS slopes of math on study hours (±1.96 SE), workbook values · full sample",
                    tone="blue", pill_icon="trending_up")
        colors = [S.GOLD if (sid is not None and s == sid) else (S.EMERALD if v >= 0 else S.TERRA)
                  for s, v in zip(rs["school_id"], rs["raw_hours_slope"])]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rs["raw_hours_slope"], y=rs["label"], mode="markers",
            marker=dict(size=[13 if c == S.GOLD else 8 for c in colors], color=colors, line=dict(color="#FFFFFF", width=1)),
            error_x=dict(type="data", symmetric=False, array=rs["hi"] - rs["raw_hours_slope"],
                         arrayminus=rs["raw_hours_slope"] - rs["lo"], color="rgba(62,120,178,0.35)", thickness=1.6, width=0),
            customdata=np.stack([rs["raw_hours_se"], rs["raw_hours_p"], rs["lo"], rs["hi"]], axis=1),
            hovertemplate="<b>%{y}</b><br>Raw slope: %{x:.3f}<br>SE: %{customdata[0]:.3f}<br>p = %{customdata[1]:.3f}"
                          "<br>±1.96 SE: [%{customdata[2]:.3f}, %{customdata[3]:.3f}]<extra></extra>"))
        fig.add_vline(x=0, line=dict(color="rgba(11,61,44,0.4)", width=1, dash="dot"))
        mslope = rs["raw_hours_slope"].mean()
        fig.add_vline(x=mslope, line=dict(color=S.GOLD, width=1.5, dash="dash"),
                      annotation_text=f"mean {mslope:.2f}", annotation_font=dict(size=10, color=S.MUTED))
        C.base_layout(fig, height=470)
        fig.update_yaxes(tickfont=dict(size=9), showgrid=False)
        fig.update_xaxes(title_text="Points per extra weekly study hour")
        chart(fig, "ex_slopes")
        npos = int((rs["raw_hours_slope"] > 0).sum())
        nsig = int((rs["raw_hours_p"] < 0.05).sum())
        txt = (f"Raw slopes range from <b>{rs['raw_hours_slope'].min():.2f}</b> to <b>{rs['raw_hours_slope'].max():.2f}</b>; "
               f"{npos} of {len(rs)} schools show a positive slope and {nsig} reach p < 0.05 on their own (30 students each). "
               "This spread visually motivates a random study-hours slope, though part of it is sampling noise in small groups.")
        if sid is not None:
            rr = rs[rs["school_id"] == sid].iloc[0]
            txt += f" <b>School {sid:02d}</b>: raw slope {rr['raw_hours_slope']:+.3f} ({I.pval(rr['raw_hours_p'])})."
        S.insight(txt, kind="green")

    # ---------------- C. Gender ----------------
    c3, c4 = st.columns([5, 7], gap="medium")
    with c3, S.card("card_ex_gender"):
        gdf = df
        S.card_head("Control Variable", "Mathematics Achievement by Gender",
                    f"Violin + box with jittered students · {desc}", tone="mint", pill_icon="wc")
        if len(gdf) < 2:
            empty_note("Select at least two students to compare distributions.")
        else:
            fig = go.Figure()
            gcol = {"Laki-laki": S.BLUE, "Perempuan": S.EMERALD}
            for g in sorted(gdf["gender"].unique()):
                sub = gdf[gdf["gender"] == g]
                fig.add_trace(go.Violin(
                    y=sub["math.score"], x=[g] * len(sub), name=g, side="positive", width=0.7,
                    box_visible=True, meanline_visible=True, points="all", pointpos=-0.45, jitter=0.25,
                    line=dict(color=gcol.get(g, S.EMERALD), width=1.5), fillcolor=gcol.get(g, S.EMERALD), opacity=0.6,
                    marker=dict(size=3.5, color=gcol.get(g, S.EMERALD), opacity=0.5),
                    customdata=sub["student_id"], spanmode="hard",
                    hovertemplate="%{customdata}<br>Math: %{y:.2f}<extra>" + g + "</extra>"))
            C.base_layout(fig, height=360)
            fig.update_yaxes(title_text="Mathematics score")
            fig.update_xaxes(showgrid=False)
            chart(fig, "ex_gender")
            means = gdf.groupby("gender")["math.score"].agg(["mean", "count"])
            fx = D.sheet(D.S_FINAL_FIXED)
            fem = fx[fx["predictor"] == "female"]
            if len(means) == 2 and "Perempuan" in means.index and "Laki-laki" in means.index:
                d = means.loc["Perempuan", "mean"] - means.loc["Laki-laki", "mean"]
                txt = (f"Raw means: Perempuan <b>{means.loc['Perempuan', 'mean']:.2f}</b> (n = {int(means.loc['Perempuan', 'count'])}) vs "
                       f"Laki-laki <b>{means.loc['Laki-laki', 'mean']:.2f}</b> (n = {int(means.loc['Laki-laki', 'count'])}), "
                       f"a raw gap of {d:+.2f} points. ")
            else:
                g0 = means.index[0]
                txt = f"Only {g0} students are in this selection (mean {means.loc[g0, 'mean']:.2f}). "
            if not fem.empty:
                txt += (f"After adjustment in the locked final model, the female coefficient is {fem['estimate'].iloc[0]:+.3f} points. "
                        "Gender is an observed characteristic, so neither gap implies a causal effect.")
            S.insight(txt)

    # ---------------- D. Correlations ----------------
    with c4, S.card("card_ex_corr"):
        S.card_head("Correlation", "How Are the Variables Related?",
                    "Pearson correlations · student level (filtered rows) or school level (30 school rows)",
                    tone="slate", pill_icon="grid_on")
        lvl = st.segmented_control("Level", ["Student Level", "School Level"], default="Student Level",
                                   key="ex_corr_level", label_visibility="collapsed") or "Student Level"
        if lvl == "Student Level":
            labels = list(D.STUDENT_VARS.values())
            if desc == "All students":
                cm = D.sheet(D.S_CORR_STUDENT).set_index("variable").loc[list(D.STUDENT_VARS), list(D.STUDENT_VARS)]
                src = "workbook (full sample, n = 900)"
            elif len(df) >= 3:
                cm = df[list(D.STUDENT_VARS)].corr()
                src = f"computed on {len(df)} filtered students"
            else:
                cm = None
                src = ""
        else:
            labels = list(D.SCHOOL_VARS.values())
            cm = D.sheet(D.S_CORR_SCHOOL).set_index("variable").loc[list(D.SCHOOL_VARS), list(D.SCHOOL_VARS)]
            src = "workbook (30 schools; filters do not apply)"
        if cm is None:
            empty_note("Select at least three students for a student-level correlation matrix.")
        else:
            z = cm.to_numpy(dtype=float)
            fig = go.Figure(go.Heatmap(
                z=z, x=labels, y=labels, zmin=-1, zmax=1, zmid=0, xgap=3, ygap=3,
                colorscale=[[0, "#B9573E"], [0.5, "#F4F2EC"], [1, "#0B6B48"]],
                text=np.vectorize(lambda v: "" if np.isnan(v) else f"{v:.2f}")(z), texttemplate="%{text}",
                textfont=dict(size=11),
                colorbar=dict(thickness=10, outlinewidth=0, tickfont=dict(size=10)),
                hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>"))
            C.base_layout(fig, height=360)
            fig.update_yaxes(autorange="reversed", showgrid=False)
            fig.update_xaxes(showgrid=False)
            chart(fig, "ex_corr")
            target = labels[0]
            others = [(labels[i], z[0, i]) for i in range(1, len(labels)) if not np.isnan(z[0, i])]
            best = max(others, key=lambda t: abs(t[1])) if others else None
            txt = f"Source: {src}. "
            if best:
                txt += f"{target} correlates most strongly with <b>{best[0]}</b> (r = {best[1]:.2f}). "
            if lvl == "School Level":
                txt += ("With only 30 schools, school-level correlations carry wide uncertainty and must not be read as if "
                        "there were 900 independent observations.")
            else:
                txt += "These are raw student-level associations that ignore the nesting of students in schools."
            S.insight(txt, kind="green")
