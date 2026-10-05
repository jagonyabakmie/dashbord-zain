"""Page 1: Overview & Descriptive Profile (exploratory; filters truly subset student rows)."""

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

CHAR_VARS = {"SES": "ses", "Prior Score": "prior.score", "Study Hours": "study.hours"}


def render(ctx) -> None:
    df, desc = filter_students(ctx)
    full = D.students()
    sid = selected_school_id(ctx)

    actions = S.header_row(
        "Overview & Descriptive Profile",
        "A first look at mathematics achievement and the characteristics of students and schools.",
        chips=[f"{len(full)} Students", f"{full['school_id'].nunique()} Schools", "Two-Level Analysis"],
        gold_chip=None if desc == "All students" else f"Filter: {desc}",
        deco="26h_books.png", actions_width=1.6,
    )
    with actions:
        cols = ["student_id", "school_id", "gender", "ses", "prior.score", "study.hours", "facility", "teacher.exp", "math.score"]
        st.download_button("Filtered data", df[cols].to_csv(index=False).encode("utf-8"), file_name="filtered_students.csv",
                           mime="text/csv", icon=":material/download:", width="stretch",
                           help=f"Download the {len(df)} filtered student rows as CSV")

    n = len(df)
    sd = df["math.score"].std() if n > 1 else np.nan
    S.kpi_strip([
        dict(label="Students", value=f"{n:,}", icon="groups", tone="forest"),
        dict(label="Schools", value=f"{df['school_id'].nunique():,}", icon="apartment", tone="teal"),
        dict(label="Average Math Score", value=I.num(df["math.score"].mean()), icon="calculate", tone="emerald"),
        dict(label="Score SD", value=I.num(sd, na="–"), icon="stacked_bar_chart", tone="gold",
             help="Standard deviation of mathematics scores in the current selection (needs at least 2 students)."),
        dict(label="Average Prior Score", value=I.num(df["prior.score"].mean()), icon="menu_book", tone="mint"),
        dict(label="Study Hours / week", value=I.num(df["study.hours"].mean()), icon="schedule", tone="blue"),
    ])

    # ---------------- Row 1: narrative + distribution + school means ----------------
    c1, c2, c3 = st.columns([3, 4, 5], gap="small")
    with c1, S.card("narr_overview"):
        books = S.img_uri(S.deco_b64("26h_books.png"))
        gm = full["math.score"].mean()
        if desc == "All students" or not n:
            sel = f"The overall average score is <b>{gm:.2f}</b>."
        else:
            diff = df["math.score"].mean() - gm
            sel = (f"The current selection averages <b>{df['math.score'].mean():.2f}</b>, {abs(diff):.2f} points "
                   f"{'above' if diff >= 0 else 'below'} the overall {gm:.2f}.")
        st.markdown(
            f"""<span class="pill">{S.icon('diversity_3')}Overview</span><img class="narr-books" src="{books}"/>
<div class="narr-title">Nilai Siswa Tidak Berdiri Sendiri</div>
<div class="narr-body">{len(full)} students are nested in {full['school_id'].nunique()} schools. Students in one school share
teachers and peers, so their scores are not independent; a multilevel model separates variation <b>within</b> and
<b>between</b> schools.<br/><br/>Raw gaps between school averages are <b>not automatically school effects</b>, because
schools also enrol different students. {sel}</div>""",
            unsafe_allow_html=True,
        )

    with c2, S.card("ov_dist"):
        S.card_head("Distribution", "Mathematics Scores Distribution",
                    f"Histogram with density curve · {desc} (n = {n:,})", tone="emerald", pill_icon="bar_chart")
        if n < 3:
            empty_note("Select at least three students to draw a distribution. A single student's score is "
                       f"<b>{I.num(df['math.score'].iloc[0]) if n else '–'}</b>.")
        else:
            x = df["math.score"].to_numpy()
            mean, med = x.mean(), np.median(x)
            nbins = int(np.clip(np.sqrt(n) * 1.3, 8, 36))
            counts, edges = np.histogram(x, bins=nbins)
            bw = edges[1] - edges[0]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=counts, width=bw * 0.94,
                                 marker=dict(color="rgba(63,174,128,0.62)", line=dict(width=0)), name="Students",
                                 customdata=np.stack([edges[:-1], edges[1:]], axis=1),
                                 hovertemplate="Score %{customdata[0]:.1f}–%{customdata[1]:.1f}<br>%{y} students<extra></extra>"))
            grid = np.linspace(x.min() - 2, x.max() + 2, 200)
            fig.add_trace(go.Scatter(x=grid, y=C.kde_curve(x, grid) * n * bw, mode="lines", name="Density",
                                     line=dict(color=S.FOREST_2, width=2), hoverinfo="skip"))
            fig.add_vline(x=mean, line=dict(color=S.GOLD, width=2, dash="dash"))
            fig.add_vline(x=med, line=dict(color=S.BLUE, width=2, dash="dot"))
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=S.GOLD, dash="dash", width=2),
                                     name=f"Mean {mean:.2f}"))
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=S.BLUE, dash="dot", width=2),
                                     name=f"Median {med:.2f}"))
            C.base_layout(fig, legend=True)
            fig.update_layout(margin=dict(t=22))
            fig.update_xaxes(title_text="Mathematics score")
            fig.update_yaxes(title_text="Students")
            chart(fig, "ov_dist")
            skew = pd.Series(x).skew()
            shape = "roughly symmetric" if abs(skew) < 0.3 else ("right-skewed" if skew > 0 else "left-skewed")
            S.insight(f"Scores centre on <b>{mean:.2f}</b> (median {med:.2f}, SD <b>{x.std(ddof=1):.2f}</b>, range "
                      f"{x.min():.1f}–{x.max():.1f}) and are {shape} (skewness {skew:.2f}). This is the variation the "
                      "multilevel model splits into within- and between-school parts.")

    with c3, S.card("ov_school"):
        g_df = gender_filtered_students(ctx)
        if ctx["gender"] == ctx["ALL"]:
            sm = D.schools()[["school_id", "n_students", "mean_math", "ci_math_low", "ci_math_high"]].copy()
            src = "workbook school summary"
        else:
            sm = D.school_means_from_students(g_df)
            src = f"recomputed for {ctx['gender']} students"
        sm = sm.sort_values("mean_math", ascending=False).reset_index(drop=True)
        sm["label"] = sm["school_id"].map(lambda s: f"S{int(s):02d}")
        S.card_head("School Profile", "Rata-rata Capaian Matematika per Sekolah",
                    f"{len(sm)} schools sorted by raw mean with 95% CI · {src}", tone="blue", pill_icon="apartment")
        colors = [S.GOLD if (sid is not None and s == sid) else S.EMERALD for s in sm["school_id"]]
        fig = go.Figure(go.Scatter(
            x=sm["label"], y=sm["mean_math"], mode="markers",
            marker=dict(size=[13 if c == S.GOLD else 8 for c in colors], color=colors, line=dict(color="#FFFFFF", width=1.2)),
            error_y=dict(type="data", symmetric=False, array=sm["ci_math_high"] - sm["mean_math"],
                         arrayminus=sm["mean_math"] - sm["ci_math_low"], color="rgba(15,138,95,0.45)", thickness=1.6, width=2),
            customdata=np.stack([sm["ci_math_low"], sm["ci_math_high"], sm["n_students"], np.arange(1, len(sm) + 1)], axis=1),
            hovertemplate="<b>School %{x}</b><br>Mean: %{y:.2f}<br>95% CI: [%{customdata[0]:.2f}, %{customdata[1]:.2f}]"
                          "<br>n = %{customdata[2]}<br>Raw rank: %{customdata[3]}<extra></extra>"))
        grand = g_df["math.score"].mean()
        fig.add_hline(y=grand, line=dict(color=S.MUTED, width=1, dash="dot"),
                      annotation_text=f"overall {grand:.2f}", annotation_position="top right",
                      annotation_font=dict(size=10, color=S.MUTED))
        C.base_layout(fig)
        fig.update_xaxes(tickangle=-90, tickfont=dict(size=9), showgrid=False)
        fig.update_yaxes(title_text="Mean math score")
        chart(fig, "ov_school")
        top, bot = sm.iloc[0], sm.iloc[-1]
        txt = (f"School means range from <b>{bot['mean_math']:.2f}</b> (S{int(bot['school_id']):02d}) to "
               f"<b>{top['mean_math']:.2f}</b> (S{int(top['school_id']):02d}); SD of school means {sm['mean_math'].std():.2f}. ")
        if sid is not None and sid in set(sm["school_id"]):
            r = sm[sm["school_id"] == sid].iloc[0]
            pos = int(sm.index[sm["school_id"] == sid][0]) + 1
            rel = "above" if r["ci_math_low"] > grand else "below" if r["ci_math_high"] < grand else "not clearly different from"
            txt += f"<b>School {sid:02d}</b> ranks {I.ordinal(pos)} ({r['mean_math']:.2f}), {rel} the overall mean. "
        txt += "These raw averages are not yet adjusted for student composition."
        S.insight(txt)

    # ---------------- Row 2: characteristics + heatmap ----------------
    c4, c5 = st.columns([4, 8], gap="small")
    with c4, S.card("ov_chars"):
        S.card_head("Student Level", "Student Characteristics Distribution",
                    f"Selected predictor by gender · {desc}", tone="mint", pill_icon="person")
        choice = st.segmented_control("Characteristic", list(CHAR_VARS), default="SES", key="ov_char",
                                      label_visibility="collapsed") or "SES"
        col = CHAR_VARS[choice]
        if n < 2:
            empty_note(f"One student selected: {choice} = <b>{I.num(df[col].iloc[0]) if n else '–'}</b>.")
        else:
            fig = go.Figure()
            gcol = {"Laki-laki": S.BLUE, "Perempuan": S.EMERALD}
            for g in sorted(df["gender"].unique()):
                sub = df[df["gender"] == g]
                fig.add_trace(go.Violin(
                    x=[g] * len(sub), y=sub[col], name=g, box_visible=True, meanline_visible=True, points=False,
                    line=dict(color=gcol.get(g, S.EMERALD), width=1.4), fillcolor=gcol.get(g, S.EMERALD), opacity=0.55,
                    spanmode="hard", hovertemplate=f"{choice}: %{{y:.2f}}<extra>{g}</extra>"))
            C.base_layout(fig, violingap=0.3)
            fig.update_yaxes(title_text=choice)
            fig.update_xaxes(showgrid=False)
            chart(fig, "ov_chars")
            v = df[col]
            r = I.safe_corr(df[col], df["math.score"])
            q1, q3 = v.quantile([0.25, 0.75])
            S.insight(f"{choice}: mean <b>{v.mean():.2f}</b>, SD {v.std():.2f}, IQR {q1:.1f}–{q3:.1f}. Raw correlation with "
                      f"math <b>r = {r:.2f}</b> ({I.corr_strength(r)}); a descriptive association, not an adjusted effect.")

    with c5, S.card("ov_heat"):
        prof = D.sheet(D.S_PROFILE_Z).copy().sort_values("mean_math_z", ascending=False).reset_index(drop=True)
        zcols = ["mean_math_z", "mean_ses_z", "mean_prior_z", "mean_hours_z", "facility_z", "teacher_exp_z"]
        names = ["Mean Math", "Mean SES", "Mean Prior", "Mean Hours", "Facility", "Teacher Exp"]
        xlab = [f"S{int(s):02d}" for s in prof["school_id"]]
        S.card_head("School Characteristics", "School Profile Heatmap",
                    "Standardized school characteristics (z-scores), sorted by mean math · 30 school rows from the workbook",
                    tone="slate", pill_icon="grid_view")
        fig = go.Figure(go.Heatmap(
            z=prof[zcols].to_numpy().T, x=xlab, y=names, zmin=-2.5, zmax=2.5, zmid=0,
            colorscale=[[0, "#B9573E"], [0.3, "#E8B9A8"], [0.5, "#F4F2EC"], [0.7, "#9FD8BD"], [1, "#0B6B48"]],
            xgap=2, ygap=2, colorbar=dict(title=dict(text="z", font=dict(size=10)), thickness=8, len=0.95, outlinewidth=0,
                                          tickfont=dict(size=9)),
            hovertemplate="School %{x}<br>%{y}: z = %{z:.2f}<extra></extra>"))
        C.base_layout(fig)
        fig.update_layout(margin=dict(l=74))
        fig.update_xaxes(tickangle=-90, tickfont=dict(size=9), showgrid=False)
        fig.update_yaxes(showgrid=False, autorange="reversed")
        if sid is not None:
            idx = xlab.index(f"S{sid:02d}")
            fig.add_shape(type="rect", x0=idx - 0.5, x1=idx + 0.5, y0=-0.5, y1=len(names) - 0.5, line=dict(color=S.GOLD, width=3))
        chart(fig, "ov_heat")
        cs = D.sheet(D.S_CORR_SCHOOL).set_index("variable")
        top = prof.iloc[0]
        txt = (f"Mean math tracks mean SES most closely (r = {cs.loc['mean_math', 'mean_ses']:.2f}); its raw correlation with "
               f"facility (r = {cs.loc['mean_math', 'facility']:.2f}) and teacher experience (r = {cs.loc['mean_math', 'teacher.exp']:.2f}) "
               f"is weaker. The top school (S{int(top['school_id']):02d}) has SES z = {top['mean_ses_z']:+.2f} but facility "
               f"z = {top['facility_z']:+.2f}, so high achievement does not mean the highest value on every characteristic.")
        if sid is not None:
            r = prof[prof["school_id"] == sid].iloc[0]
            hi = max(zip(names, r[zcols]), key=lambda t: t[1])
            lo = min(zip(names, r[zcols]), key=lambda t: t[1])
            txt = (f"<b>School {sid:02d}</b> (gold frame) is highest on {hi[0]} (z = {hi[1]:+.2f}) and lowest on {lo[0]} "
                   f"(z = {lo[1]:+.2f}). " + txt)
        S.insight(txt)
