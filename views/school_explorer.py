"""Page 5: School Performance Explorer. School-specific values from the locked final model (13_final_school)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import charts as C
from utils import data as D
from utils import interpretations as I
from utils import styles as S
from views.common import chart, selected_school_id


def _coef(fx: pd.DataFrame, term: str) -> float:
    return float(fx.loc[fx["predictor"] == term, "estimate"].iloc[0])


def render(ctx) -> None:
    fs = D.final_school().copy()
    fx = D.sheet(D.S_FINAL_FIXED)
    mid = D.final_model_id()
    ids = sorted(fs["school_id"].astype(int).tolist())
    labels = [D.school_label(s) for s in ids]

    pre = selected_school_id(ctx)
    if "se_school" not in st.session_state or st.session_state["se_school"] not in labels:
        st.session_state["se_school"] = D.school_label(pre) if pre in ids else labels[0]

    S.page_header(
        "Beyond the Raw School Ranking",
        "Comparing raw achievement with performance relative to model expectation.",
        kicker="School Performance Explorer",
        chips=[f"Locked final model {mid}", "Random intercept + random study-hours slope"],
        deco="26b_leaf_top_right.png",
    )

    with S.card("card_se_pick"):
        a, b = st.columns([2, 5], vertical_alignment="center")
        with a:
            st.selectbox("Select a school", labels, key="se_school")
        with b:
            st.markdown('<div class="card-sub" style="margin:0">Raw values describe what students in the school scored. '
                        '<b>Adjusted effects</b> are the school\'s random intercept: how far its students score from what the '
                        'model expects given their SES, prior score, study hours, gender and the school\'s facility and teacher '
                        'experience. They describe performance relative to model expectation, not direct school quality.</div>',
                        unsafe_allow_html=True)
    sid = int(st.session_state["se_school"].split()[-1])
    r = fs[fs["school_id"] == sid].iloc[0]
    n_sc = len(fs)

    S.kpi_strip(items=[
        dict(label="Raw Mean", value=I.num(r["mean_math"]), icon="calculate", tone="forest",
             sub=f"95% CI {r['ci_math_low']:.1f}–{r['ci_math_high']:.1f}"),
        dict(label="Raw Rank", value=f"{int(r['raw_rank'])} / {n_sc}", icon="format_list_numbered", tone="teal"),
        dict(label="Adjusted Rank", value=f"{int(r['adjusted_rank'])} / {n_sc}", icon="swap_vert", tone="emerald",
             sub=(f"{'up' if r['rank_change'] > 0 else 'down'} {abs(int(r['rank_change']))} place(s)" if r["rank_change"] != 0 else "unchanged")),
        dict(label="Adjusted Effect", value=I.signed(r["adjusted_intercept"], 3), icon="target", tone="gold",
             sub=f"95% CI {r['adjusted_intercept_low']:+.2f} to {r['adjusted_intercept_high']:+.2f}",
             help="School random intercept (BLUP) from the final model: points above/below model expectation."),
        dict(label="Facility", value=I.num(r["facility"], 1), icon="domain", tone="blue"),
        dict(label="Teacher Experience", value=I.num(r["teacher.exp"], 1), icon="school", tone="mint", sub="years"),
        dict(label="Study-Hours Slope", value=I.num(r["hours_total_slope"], 3), icon="trending_up", tone="slate",
             sub="fixed + school deviation", help="School-specific total slope = fixed study-hours slope + this school's random slope."),
    ])

    # ---------------- A. Slopegraph + B. Caterpillar ----------------
    c1, c2 = st.columns([5, 7], gap="medium")
    with c1, S.card("card_se_slope"):
        S.card_head("Ranking", "Does Raw School Ranking Change After Adjustment?",
                    "Raw rank (left) vs adjusted rank (right) · 1 = highest", tone="gold", pill_icon="swap_vert")
        fig = go.Figure()
        for _, s in fs.iterrows():
            if int(s["school_id"]) == sid:
                continue
            ch = s["rank_change"]
            col = "rgba(15,138,95,0.45)" if ch > 0 else ("rgba(185,87,62,0.45)" if ch < 0 else "rgba(107,123,115,0.35)")
            fig.add_trace(go.Scatter(x=["Raw rank", "Adjusted rank"], y=[s["raw_rank"], s["adjusted_rank"]], mode="lines+markers",
                                     line=dict(color=col, width=1.5), marker=dict(size=6, color=col),
                                     hovertemplate=f"School {int(s['school_id']):02d}<br>%{{x}}: %{{y}}<extra></extra>", showlegend=False))
        fig.add_trace(go.Scatter(x=["Raw rank", "Adjusted rank"], y=[r["raw_rank"], r["adjusted_rank"]], mode="lines+markers+text",
                                 line=dict(color=S.GOLD, width=4), marker=dict(size=14, color=S.GOLD, line=dict(color="#FFFFFF", width=2)),
                                 text=[f"S{sid:02d} · {int(r['raw_rank'])}", f"{int(r['adjusted_rank'])} · S{sid:02d}"],
                                 textposition=["middle left", "middle right"], textfont=dict(color=S.INK, size=12),
                                 hovertemplate=f"School {sid:02d}<br>%{{x}}: %{{y}}<extra></extra>", showlegend=False))
        C.base_layout(fig, height=470)
        fig.update_yaxes(autorange="reversed", title_text="Rank", dtick=5, range=[n_sc + 1, 0])
        fig.update_xaxes(showgrid=False, range=[-0.45, 1.45])
        chart(fig, "se_slope")
        moved = fs.loc[fs["rank_change"].abs().idxmax()]
        S.insight(
            f"<b>School {sid:02d}</b> moves from raw rank {int(r['raw_rank'])} to adjusted rank {int(r['adjusted_rank'])}. "
            f"Across schools, {int((fs['rank_change'] != 0).sum())} of {n_sc} change position; the largest move is School "
            f"{int(moved['school_id']):02d} ({int(moved['raw_rank'])} → {int(moved['adjusted_rank'])}). Green lines rise after "
            "adjustment and red lines fall, meaning the raw rank partly reflected student composition."
        )

    with c2, S.card("card_se_cat"):
        d = fs.sort_values("adjusted_intercept").reset_index(drop=True)
        d["label"] = d["school_id"].map(lambda s: f"School {int(s):02d}")
        S.card_head("Adjusted Performance", "Which Schools Score Above Model Expectation?",
                    "School random intercepts (adjusted school performance) with 95% intervals · zero = model expectation",
                    tone="emerald", pill_icon="target")
        colors = [S.GOLD if s == sid else (S.EMERALD if lo > 0 else (S.TERRA if hi < 0 else S.NEUTRAL))
                  for s, lo, hi in zip(d["school_id"], d["adjusted_intercept_low"], d["adjusted_intercept_high"])]
        fig = go.Figure(go.Scatter(
            x=d["adjusted_intercept"], y=d["label"], mode="markers",
            marker=dict(size=[14 if c == S.GOLD else 9 for c in colors], color=colors, line=dict(color="#FFFFFF", width=1)),
            error_x=dict(type="data", symmetric=False, array=d["adjusted_intercept_high"] - d["adjusted_intercept"],
                         arrayminus=d["adjusted_intercept"] - d["adjusted_intercept_low"], color="rgba(11,61,44,0.3)",
                         thickness=1.6, width=0),
            customdata=np.stack([d["adjusted_intercept_low"], d["adjusted_intercept_high"], d["raw_rank"], d["adjusted_rank"]], axis=1),
            hovertemplate="<b>%{y}</b><br>Adjusted effect: %{x:+.3f}<br>95% interval: [%{customdata[0]:+.3f}, %{customdata[1]:+.3f}]"
                          "<br>Raw rank %{customdata[2]} → adjusted rank %{customdata[3]}<extra></extra>"))
        fig.add_vline(x=0, line=dict(color="rgba(11,61,44,0.45)", width=1.2, dash="dot"))
        C.base_layout(fig, height=520)
        fig.update_yaxes(showgrid=False, tickfont=dict(size=10))
        fig.update_xaxes(title_text="Points relative to model expectation")
        chart(fig, "se_cat")
        above = int((d["adjusted_intercept_low"] > 0).sum())
        below = int((d["adjusted_intercept_high"] < 0).sum())
        pos = "above" if r["adjusted_intercept_low"] > 0 else ("below" if r["adjusted_intercept_high"] < 0 else "not clearly different from")
        S.insight(
            f"{above} schools sit clearly above and {below} clearly below model expectation (intervals excluding zero; green/red). "
            f"<b>School {sid:02d}</b> has an adjusted effect of {r['adjusted_intercept']:+.3f} points, {pos} expectation. "
            "This is performance relative to model expectation for comparable students, not a direct measure of school quality.",
            kind="green",
        )

    # ---------------- C. Study-hours slopes + D. Response curve ----------------
    b_hours = _coef(fx, "hours_c")
    c3, c4 = st.columns([6, 6], gap="medium")
    with c3, S.card("card_se_hours"):
        d = fs.sort_values("hours_total_slope").reset_index(drop=True)
        d["label"] = d["school_id"].map(lambda s: f"School {int(s):02d}")
        d["lo"] = b_hours + d["hours_random_low"]
        d["hi"] = b_hours + d["hours_random_high"]
        S.card_head("Random Slope", "Is Study Time Linked to Scores Equally in Every School?",
                    "School-specific total study-hours slopes (fixed + random deviation, interval of the deviation)",
                    tone="blue", pill_icon="trending_up")
        colors = [S.GOLD if s == sid else S.BLUE for s in d["school_id"]]
        fig = go.Figure(go.Scatter(
            x=d["hours_total_slope"], y=d["label"], mode="markers",
            marker=dict(size=[14 if c == S.GOLD else 8 for c in colors], color=colors, line=dict(color="#FFFFFF", width=1)),
            error_x=dict(type="data", symmetric=False, array=d["hi"] - d["hours_total_slope"],
                         arrayminus=d["hours_total_slope"] - d["lo"], color="rgba(62,120,178,0.3)", thickness=1.6, width=0),
            customdata=np.stack([d["hours_random_effect"], d["lo"], d["hi"]], axis=1),
            hovertemplate="<b>%{y}</b><br>Total slope: %{x:.3f}<br>Random deviation: %{customdata[0]:+.3f}"
                          "<br>Interval: [%{customdata[1]:.3f}, %{customdata[2]:.3f}]<extra></extra>"))
        fig.add_vline(x=b_hours, line=dict(color=S.EMERALD, width=2, dash="dash"),
                      annotation_text=f"average fixed slope {b_hours:.3f}", annotation_font=dict(size=10, color=S.MUTED))
        fig.add_vline(x=0, line=dict(color="rgba(11,61,44,0.35)", width=1, dash="dot"))
        C.base_layout(fig, height=520)
        fig.update_yaxes(showgrid=False, tickfont=dict(size=10))
        fig.update_xaxes(title_text="Points per extra weekly study hour")
        chart(fig, "se_hours")
        rank_h = int((fs["hours_total_slope"] > r["hours_total_slope"]).sum()) + 1
        S.insight(
            f"School-specific slopes range from {fs['hours_total_slope'].min():.3f} to {fs['hours_total_slope'].max():.3f} "
            f"around the average fixed slope of {b_hours:.3f} points per hour. <b>School {sid:02d}</b> has a total slope of "
            f"{r['hours_total_slope']:.3f} ({I.ordinal(rank_h)} steepest), i.e. {r['hours_random_effect']:+.3f} relative to the average school. "
            "Slopes are adjusted associations, not the effect of assigning more study time."
        )

    with c4, S.card("card_se_curve"):
        stu = D.students()
        hours_mean = float((stu["study.hours"] - stu["hours_c"]).mean())
        p_female = float(stu["female"].mean())
        b0 = _coef(fx, "(Intercept)")
        b_fem, b_fac, b_tch = _coef(fx, "female"), _coef(fx, "facility_c"), _coef(fx, "teacher_c")
        h = np.linspace(stu["study.hours"].min(), stu["study.hours"].max(), 60)
        hc = h - hours_mean
        avg = b0 + b_fem * p_female + b_hours * hc
        sel = (b0 + b_fem * p_female + b_fac * r["facility_c"] + b_tch * r["teacher_c"] + r["adjusted_intercept"]
               + (b_hours + r["hours_random_effect"]) * hc)
        S.card_head("Response Curve", f"How Would School {sid:02d} Respond to More Study Time?",
                    "Model-implied math score vs weekly study hours · SES and prior score at the sample mean, gender at the sample mix",
                    tone="gold", pill_icon="show_chart")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=h, y=avg, mode="lines", name="Average school",
                                 line=dict(color=S.EMERALD, width=2.5, dash="dash"),
                                 hovertemplate="Average school<br>%{x:.1f} h/week → %{y:.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=h, y=sel, mode="lines", name=f"School {sid:02d}",
                                 line=dict(color=S.GOLD, width=3.5),
                                 hovertemplate=f"School {sid:02d}<br>%{{x:.1f}} h/week → %{{y:.2f}}<extra></extra>"))
        C.base_layout(fig, height=400, legend=True)
        fig.update_xaxes(title_text="Weekly study hours")
        fig.update_yaxes(title_text="Predicted mathematics score")
        chart(fig, "se_curve")
        gap_mean = float(sel[np.argmin(np.abs(hc))] - avg[np.argmin(np.abs(hc))])
        S.insight(
            f"At the average study time ({hours_mean:.1f} h/week), School {sid:02d}'s curve sits {gap_mean:+.2f} points from the "
            f"average school, combining its facility ({r['facility']:.1f}), teacher experience ({r['teacher.exp']:.1f} yrs) and "
            f"adjusted effect. Each extra hour is associated with {r['hours_total_slope']:+.3f} points here vs {b_hours:+.3f} on "
            f"average. Curves use only the locked model's fixed effects ({mid}) and this school's stored random effects.",
            kind="green",
        )

    with st.expander("All schools: raw and adjusted values (workbook)"):
        cols = ["school_id", "mean_math", "raw_rank", "adjusted_intercept", "adjusted_intercept_low", "adjusted_intercept_high",
                "adjusted_rank", "rank_change", "hours_total_slope", "facility", "teacher.exp", "mean_ses", "mean_prior", "mean_hours"]
        st.dataframe(fs[cols].sort_values("adjusted_rank"), hide_index=True, width="stretch")
        st.download_button("Download school table (CSV)", fs[cols].to_csv(index=False).encode(), file_name="school_effects_final.csv",
                           mime="text/csv", icon=":material/download:")
