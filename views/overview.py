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

    S.page_header(
        "Overview & Descriptive Profile",
        "A first look at mathematics achievement and the characteristics of students and schools.",
        kicker="Mathematics Achievement Analytics",
        chips=[f"{len(full)} Students", f"{full['school_id'].nunique()} Schools", "Two-Level Analysis"],
        gold_chip=None if desc == "All students" else f"Filter: {desc}",
        deco="26h_books.png",
    )

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

    # ---------------- Row 1: narrative + distribution ----------------
    c1, c2 = st.columns([5, 7], gap="medium")
    with c1, S.card("cardnarr_overview"):
        books = S.img_uri(S.deco_b64("26h_books.png"))
        n_sc = full["school_id"].nunique()
        st.markdown(
            f"""<span class="pill">{S.icon('diversity_3')}Overview</span>
<img class="narr-books" src="{books}"/>
<div class="narr-title">Nilai Siswa Tidak Berdiri Sendiri</div>
<div class="narr-lead">Students' mathematics scores are shaped both by who they are and by the school they attend.</div>
<div class="narr-body">This dashboard follows <b>{len(full)} students nested in {n_sc} schools</b>. Students in the same school
share teachers, facilities and peers, so their scores are not independent observations. A multilevel model separates
variation <b>within</b> schools from variation <b>between</b> schools.<br/><br/>
Raw differences between school averages are <b>not automatically school effects</b>: schools also differ in the students
they enrol (SES, prior achievement, study habits). The later pages compare raw rankings with performance relative to
model expectation.</div>""",
            unsafe_allow_html=True,
        )
        gm = full["math.score"].mean()
        if n:
            diff = df["math.score"].mean() - gm
            sel_txt = ("the full sample" if desc == "All students"
                       else f"the current selection ({desc}), which is {abs(diff):.2f} points {'above' if diff >= 0 else 'below'} the overall mean of {gm:.2f}")
            S.insight(
                f"The average mathematics score is <b>{df['math.score'].mean():.2f}</b> for {sel_txt}. "
                "Whether such gaps reflect the school or the students it serves is a question for the adjusted model, not the raw mean.",
                title="Interpretasi",
            )

    with c2, S.card("card_ov_dist"):
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
                                 marker=dict(color="rgba(63,174,128,0.62)", line=dict(width=0)),
                                 name="Students",
                                 customdata=np.stack([edges[:-1], edges[1:]], axis=1),
                                 hovertemplate="Score %{customdata[0]:.1f}–%{customdata[1]:.1f}<br>%{y} students<extra></extra>"))
            grid = np.linspace(x.min() - 2, x.max() + 2, 200)
            fig.add_trace(go.Scatter(x=grid, y=C.kde_curve(x, grid) * n * bw, mode="lines", name="Density",
                                     line=dict(color=S.FOREST_2, width=2.5), hoverinfo="skip"))
            fig.add_vline(x=mean, line=dict(color=S.GOLD, width=2, dash="dash"))
            fig.add_vline(x=med, line=dict(color=S.BLUE, width=2, dash="dot"))
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=S.GOLD, dash="dash", width=2),
                                     name=f"Mean ({mean:.2f})"))
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=S.BLUE, dash="dot", width=2),
                                     name=f"Median ({med:.2f})"))
            C.base_layout(fig, height=330, legend=True)
            fig.update_xaxes(title_text="Mathematics score")
            fig.update_yaxes(title_text="Students")
            chart(fig, "ov_dist")
            skew = pd.Series(x).skew()
            shape = ("roughly symmetric" if abs(skew) < 0.3 else
                     ("right-skewed (a longer tail of high scores)" if skew > 0 else "left-skewed (a longer tail of low scores)"))
            S.insight(
                f"Scores centre on <b>{mean:.2f}</b> (median {med:.2f}) with a standard deviation of <b>{x.std(ddof=1):.2f}</b>, "
                f"ranging from {x.min():.1f} to {x.max():.1f}. The distribution is {shape} (skewness {skew:.2f}). "
                "This is the baseline variation that the multilevel model later splits into within- and between-school parts."
            )

    # ---------------- Row 2: school means + characteristics ----------------
    c3, c4 = st.columns([7, 5], gap="medium")
    with c3, S.card("card_ov_school"):
        g_df = gender_filtered_students(ctx)
        if ctx["gender"] == ctx["ALL"]:
            sm = D.schools()[["school_id", "n_students", "mean_math", "ci_math_low", "ci_math_high"]].copy()
            src = "Workbook school summary"
        else:
            sm = D.school_means_from_students(g_df)
            src = f"Recomputed for {ctx['gender']} students"
        sm = sm.sort_values("mean_math", ascending=False).reset_index(drop=True)
        sm["label"] = sm["school_id"].map(lambda s: f"S{int(s):02d}")
        S.card_head("School Profile", "Rata-rata Capaian Matematika per Sekolah",
                    f"{len(sm)} schools sorted by raw mean score with 95% CI · {src}", tone="blue", pill_icon="apartment")
        colors = [S.GOLD if (sid is not None and s == sid) else S.EMERALD for s in sm["school_id"]]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sm["label"], y=sm["mean_math"], mode="markers",
            marker=dict(size=[15 if c == S.GOLD else 10 for c in colors], color=colors, line=dict(color="#FFFFFF", width=1.5)),
            error_y=dict(type="data", symmetric=False, array=sm["ci_math_high"] - sm["mean_math"],
                         arrayminus=sm["mean_math"] - sm["ci_math_low"], color="rgba(15,138,95,0.45)", thickness=2, width=3),
            customdata=np.stack([sm["ci_math_low"], sm["ci_math_high"], sm["n_students"], np.arange(1, len(sm) + 1)], axis=1),
            hovertemplate="<b>School %{x}</b><br>Mean: %{y:.2f}<br>95% CI: [%{customdata[0]:.2f}, %{customdata[1]:.2f}]"
                          "<br>n = %{customdata[2]}<br>Raw rank: %{customdata[3]}<extra></extra>"))
        grand = g_df["math.score"].mean()
        fig.add_hline(y=grand, line=dict(color=S.MUTED, width=1, dash="dot"),
                      annotation_text=f"Overall mean {grand:.2f}", annotation_position="top right",
                      annotation_font=dict(size=11, color=S.MUTED))
        C.base_layout(fig, height=360)
        fig.update_xaxes(tickangle=-90, tickfont=dict(size=10), showgrid=False)
        fig.update_yaxes(title_text="Mean mathematics score")
        chart(fig, "ov_school")
        top, bot = sm.iloc[0], sm.iloc[-1]
        txt = (f"Raw school means range from <b>{bot['mean_math']:.2f}</b> (School {int(bot['school_id']):02d}) to "
               f"<b>{top['mean_math']:.2f}</b> (School {int(top['school_id']):02d}), a gap of {top['mean_math'] - bot['mean_math']:.2f} points; "
               f"the SD of school means is {sm['mean_math'].std():.2f}. ")
        if sid is not None and sid in set(sm["school_id"]):
            r = sm[sm["school_id"] == sid].iloc[0]
            pos = int(sm.index[sm["school_id"] == sid][0]) + 1
            rel = ("above" if r["ci_math_low"] > grand else "below" if r["ci_math_high"] < grand else "not clearly different from")
            txt += (f"<b>School {sid:02d}</b> ranks {I.ordinal(pos)} of {len(sm)} with a mean of {r['mean_math']:.2f}; "
                    f"its 95% CI is {rel} the overall mean. ")
        txt += "These are raw averages that have not yet been adjusted for student composition."
        S.insight(txt)

    with c4, S.card("card_ov_chars"):
        S.card_head("Student Level", "Student Characteristics Distribution",
                    f"Distribution of a selected student-level predictor by gender · {desc}", tone="mint", pill_icon="person")
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
                    x=[g] * len(sub), y=sub[col], name=g, box_visible=True, meanline_visible=True, points="outliers",
                    line=dict(color=gcol.get(g, S.EMERALD), width=1.5), fillcolor=gcol.get(g, S.EMERALD), opacity=0.55,
                    marker=dict(size=4, color=gcol.get(g, S.EMERALD)), spanmode="hard",
                    customdata=sub["student_id"],
                    hovertemplate=f"%{{customdata}}<br>{choice}: %{{y:.2f}}<extra>{g}</extra>"))
            C.base_layout(fig, height=318, legend=False, violingap=0.25)
            fig.update_yaxes(title_text=choice)
            fig.update_xaxes(showgrid=False)
            chart(fig, "ov_chars")
            v = df[col]
            r = I.safe_corr(df[col], df["math.score"])
            q1, q3 = v.quantile([0.25, 0.75])
            S.insight(
                f"{choice} has mean <b>{v.mean():.2f}</b> and SD <b>{v.std():.2f}</b> (IQR {q1:.1f}–{q3:.1f}). "
                f"Its raw correlation with mathematics score in this selection is <b>r = {r:.2f}</b> ({I.corr_strength(r)}). "
                "This is a descriptive association; adjusted effects appear on the Research Questions page.",
                kind="green",
            )

    # ---------------- Row 3: heatmap ----------------
    with S.card("card_ov_heat"):
        prof = D.sheet(D.S_PROFILE_Z).copy()
        prof = prof.sort_values("mean_math_z", ascending=False).reset_index(drop=True)
        zcols = ["mean_math_z", "mean_ses_z", "mean_prior_z", "mean_hours_z", "facility_z", "teacher_exp_z"]
        names = ["Mean Math", "Mean SES", "Mean Prior", "Mean Hours", "Facility", "Teacher Exp"]
        ylab = [("► " if sid is not None and s == sid else "") + f"School {int(s):02d}" for s in prof["school_id"]]
        S.card_head("School Characteristics", "School Profile Heatmap",
                    "Standardized school characteristics (z-scores across the 30 schools), sorted by mean math · "
                    "school-level rows from the workbook, highlight only", tone="slate", pill_icon="grid_view")
        z = prof[zcols].to_numpy()
        fig = go.Figure(go.Heatmap(
            z=z.T, x=ylab, y=names, zmin=-2.5, zmax=2.5, zmid=0,
            colorscale=[[0, "#B9573E"], [0.3, "#E8B9A8"], [0.5, "#F4F2EC"], [0.7, "#9FD8BD"], [1, "#0B6B48"]],
            xgap=2, ygap=2,
            colorbar=dict(title=dict(text="z", font=dict(size=11)), thickness=10, len=0.9, outlinewidth=0,
                          tickfont=dict(size=10)),
            hovertemplate="%{x}<br>%{y}: z = %{z:.2f}<extra></extra>"))
        C.base_layout(fig, height=300)
        fig.update_xaxes(tickangle=-90, tickfont=dict(size=10), showgrid=False)
        fig.update_yaxes(showgrid=False, autorange="reversed")
        if sid is not None:
            idx = ylab.index(next(l for l in ylab if l.startswith("►")))
            fig.add_shape(type="rect", x0=idx - 0.5, x1=idx + 0.5, y0=-0.5, y1=len(names) - 0.5,
                          line=dict(color=S.GOLD, width=3))
        chart(fig, "ov_heat")
        cs = D.sheet(D.S_CORR_SCHOOL).set_index("variable")
        r_ses = cs.loc["mean_math", "mean_ses"]
        r_fac = cs.loc["mean_math", "facility"]
        r_tch = cs.loc["mean_math", "teacher.exp"]
        top = prof.iloc[0]
        txt = (f"Across the 30 schools, mean math tracks mean SES most closely (r = {r_ses:.2f}), while its raw correlation "
               f"with facility (r = {r_fac:.2f}) and teacher experience (r = {r_tch:.2f}) is weaker. The top-scoring school "
               f"(School {int(top['school_id']):02d}) has SES z = {top['mean_ses_z']:+.2f} and facility z = {top['facility_z']:+.2f}, "
               "so high achievement does not coincide with the highest value on every characteristic.")
        if sid is not None:
            r = prof[prof["school_id"] == sid].iloc[0]
            hi = max(zip(names, r[zcols]), key=lambda t: t[1])
            lo = min(zip(names, r[zcols]), key=lambda t: t[1])
            txt += f" <b>School {sid:02d}</b> stands out most on {hi[0]} (z = {hi[1]:+.2f}) and lowest on {lo[0]} (z = {lo[1]:+.2f})."
        S.insight(txt)

    with st.expander("Download the filtered student data"):
        st.caption(f"{len(df)} rows · {desc}")
        cols = ["student_id", "school_id", "gender", "ses", "prior.score", "study.hours", "facility", "teacher.exp", "math.score"]
        st.download_button("Download CSV", df[cols].to_csv(index=False).encode("utf-8"),
                           file_name="filtered_students.csv", mime="text/csv", icon=":material/download:")
