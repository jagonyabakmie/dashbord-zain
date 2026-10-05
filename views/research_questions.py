"""Page 3: Research Questions. Locked full-sample results only; filters never subset anything here."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import charts as C
from utils import data as D
from utils import interpretations as I
from utils import styles as S
from views.common import chart

L1_TERMS = ["ses_c", "prior_c", "hours_c", "female"]
L2_TERMS = ["facility_c", "teacher_c"]


def _std_ci(fx: pd.DataFrame) -> pd.DataFrame:
    """Standardized CI = raw CI scaled by the workbook's standardized_beta / estimate ratio."""
    fx = fx.copy()
    ratio = fx["standardized_beta"] / fx["estimate"]
    fx["std_low"] = fx["ci_low"] * ratio
    fx["std_high"] = fx["ci_high"] * ratio
    lo = fx[["std_low", "std_high"]].min(axis=1)
    hi = fx[["std_low", "std_high"]].max(axis=1)
    fx["std_low"], fx["std_high"] = lo, hi
    fx["label"] = fx["predictor"].map(D.TERM_LABELS).fillna(fx["predictor"])
    return fx


def render(ctx) -> None:
    mid = D.final_model_id()
    rq = D.sheet(D.S_RQ).set_index("rq")
    vp = D.sheet(D.S_VAR_PATH)
    fx = _std_ci(D.sheet(D.S_FINAL_FIXED))
    fx_model = fx["model_id"].iloc[0]
    inference = fx["inference"].iloc[0] if "inference" in fx else ""

    S.page_header(
        "What Explains Mathematics Achievement?",
        "Four research questions, four direct statistical answers.",
        kicker="Research Questions",
        chips=[f"Locked final model {fx_model}", inference, "Full sample: 900 students · 30 schools"],
        deco="26k_sparkles.png",
    )

    l1 = fx[fx["predictor"].isin(L1_TERMS)].copy()
    l1["abs"] = l1["standardized_beta"].abs()
    l1 = l1.sort_values("abs", ascending=False)
    strongest = l1.iloc[0]

    icc = rq.loc["RQ1", "metric_1"]
    within = rq.loc["RQ1", "metric_2"]
    within_red = rq.loc["RQ2", "metric_1"]
    add_between_red = rq.loc["RQ3", "metric_1"]
    between_red = rq.loc["RQ4", "metric_1"]
    remaining = rq.loc["RQ4", "metric_2"]

    S.kpi_strip([
        dict(label="Between-school share (ICC)", value=I.pct(icc), icon="apartment", tone="forest", sub="RQ1",
             help="Intraclass correlation from the null model: share of total score variance lying between schools."),
        dict(label="Within-school share", value=I.pct(within), icon="groups", tone="teal", sub="RQ1"),
        dict(label="Strongest student predictor", value=strongest["label"], icon="military_tech", tone="gold",
             sub=f"standardized β = {strongest['standardized_beta']:.3f} · RQ2"),
        dict(label=rq.loc["RQ2", "metric_1_label"], value=I.pct(within_red), icon="trending_down", tone="emerald", sub="RQ2",
             help="Reduction in within-school (residual) variance relative to the null model."),
        dict(label=rq.loc["RQ3", "metric_1_label"], value=I.pct(add_between_red), icon="domain", tone="blue", sub="RQ3",
             help="Additional reduction in between-school variance when school characteristics are added."),
        dict(label="Initial school variance remaining", value=I.pct(remaining), icon="hourglass_bottom", tone="terra", sub="RQ4",
             help="Share of the null-model between-school variance that remains in the final model."),
    ])

    # ---------------- RQ1 + RQ2 ----------------
    c1, c2 = st.columns(2, gap="medium")
    with c1, S.card("card_rq1"):
        S.card_head("RQ 01", "Where Does the Variation Come From?",
                    "Null-model variance partition · within vs between schools", tone="blue", pill_icon="pie_chart")
        null = vp.iloc[0]
        tb, tw = null["between_school_variance"], null["within_student_variance"]
        fig = go.Figure()
        fig.add_trace(go.Bar(y=["Total variance"], x=[within], orientation="h", name="Within schools",
                             marker=dict(color=S.EMERALD, line=dict(color="#FFFFFF", width=2)),
                             text=[f"Within {I.pct(within)}"], textposition="inside", insidetextanchor="middle",
                             textfont=dict(color="#FFFFFF", size=13),
                             customdata=[[tw]], hovertemplate="Within schools: %{x:.1%}<br>Variance: %{customdata[0]:.3f}<extra></extra>"))
        fig.add_trace(go.Bar(y=["Total variance"], x=[icc], orientation="h", name="Between schools (ICC)",
                             marker=dict(color=S.GOLD, line=dict(color="#FFFFFF", width=2)),
                             text=[I.pct(icc)], textposition="outside", textfont=dict(color=S.INK, size=13),
                             customdata=[[tb]], hovertemplate="Between schools: %{x:.1%}<br>Variance: %{customdata[0]:.3f}<extra></extra>"))
        C.base_layout(fig, height=190, legend=True, barmode="stack")
        fig.update_xaxes(tickformat=".0%", range=[0, 1.16], title_text="Share of total variance")
        fig.update_yaxes(showticklabels=False, showgrid=False)
        chart(fig, "rq1")
        S.kpi_strip([
            dict(label="Between-school variance (null)", value=I.num(tb, 3), icon="apartment", tone="gold"),
            dict(label="Within-school variance (null)", value=I.num(tw, 3), icon="person", tone="emerald"),
        ])
        S.insight(
            f"{rq.loc['RQ1', 'answer']}. About <b>{I.pct(icc)}</b> of the variation in mathematics scores lies between schools "
            f"(ICC = {icc:.3f}), and <b>{I.pct(within)}</b> lies between students within the same school. This is a variance "
            "partition, not a causal share: it says where scores differ, not what causes the differences. Even a modest ICC "
            "means students in a school are not independent, which justifies the multilevel model.",
            title="Interpretasi",
        )

    with c2, S.card("card_rq2"):
        S.card_head("RQ 02 · Student Level", "Which Student Characteristics Matter Most?",
                    f"Standardized coefficients with 95% CI · {fx_model} ({inference})", tone="mint", pill_icon="person_search")
        d = l1.copy()
        d["color"] = [S.EMERALD if v >= 0 else S.TERRA for v in d["standardized_beta"]]
        d["hover"] = [f"Raw estimate {e:+.4f} (95% CI {lo:.4f} to {hi:.4f})<br>{I.pval(p)}"
                      for e, lo, hi, p in zip(d["estimate"], d["ci_low"], d["ci_high"], d["p_value"])]
        fig = C.forest(d, "label", "standardized_beta", "std_low", "std_high", color_col="color",
                       x_title="Standardized β (SD change in math per SD of predictor)", hover_extra="hover", height=300)
        chart(fig, "rq2")
        sec = l1.iloc[1]
        S.insight(
            f"Holding the other predictors and school characteristics constant, <b>{strongest['label']}</b> shows the strongest "
            f"adjusted association (β = {strongest['standardized_beta']:.3f}, {I.pval(strongest['p_value'])}), followed by "
            f"{sec['label']} (β = {sec['standardized_beta']:.3f}). "
            + "; ".join(f"{r['label']} β = {r['standardized_beta']:+.3f}" for _, r in l1.iloc[2:].iterrows())
            + f". Together the student-level terms reduce within-school variance by {I.pct(within_red)} relative to the null model. "
            "These are adjusted associations in observational data, not causal effects.",
            title="Interpretasi",
        )

    # ---------------- RQ3 + RQ4 ----------------
    c3, c4 = st.columns(2, gap="medium")
    with c3, S.card("card_rq3"):
        S.card_head("RQ 03 · School Level", "Do School Characteristics Matter?",
                    f"Level-2 fixed effects with 95% CI (Kenward-Roger df) · {fx_model}", tone="slate", pill_icon="domain")
        d = fx[fx["predictor"].isin(L2_TERMS)].copy()
        sc = D.schools()
        sd_map = {"facility_c": sc["facility"].std(), "teacher_c": sc["teacher.exp"].std()}
        unit = {"facility_c": "per facility point", "teacher_c": "per year of experience"}
        d["color"] = [S.BLUE] * len(d)
        d["label"] = [f"{D.TERM_LABELS[p]}<br><span style='font-size:10px'>{unit[p]}</span>" for p in d["predictor"]]
        d["hover"] = [f"df = {df_:.1f}, {I.pval(p)}<br>Standardized β = {b:.3f}"
                      for df_, p, b in zip(d["df"], d["p_value"], d["standardized_beta"])]
        fig = C.forest(d, "label", "estimate", "ci_low", "ci_high", color_col="color",
                       x_title="Change in math score per unit (raw scale)", hover_extra="hover", height=250)
        chart(fig, "rq3")
        parts = []
        for _, r in d.iterrows():
            p = r["predictor"]
            parts.append(
                f"<b>{D.TERM_LABELS[p]}</b>: {r['estimate']:+.3f} points {unit[p]} (95% CI {r['ci_low']:.3f} to {r['ci_high']:.3f}, "
                f"{I.pval(r['p_value'])}, df ≈ {r['df']:.0f}); a 1-SD difference between schools ({sd_map[p]:.2f}) corresponds to "
                f"about {r['estimate'] * sd_map[p]:+.2f} points")
        S.insight(
            "; ".join(parts) + f". Both intervals exclude zero, but the magnitudes are small next to SES. Adding these school "
            f"characteristics reduces between-school variance by a further {I.pct(add_between_red)}. With only 30 schools, "
            "these are school-level associations, not proof that facilities or experience raise scores.",
            title="Interpretasi",
        )

    with c4, S.card("card_rq4"):
        S.card_head("RQ 04", "What Still Remains Unexplained?",
                    "Between-school (intercept) variance from null to final model · workbook variance path",
                    tone="terra", pill_icon="waterfall_chart")
        stages = vp["model_stage"].tolist()
        bv = vp["between_school_variance"].to_numpy(dtype=float)
        step_names = {1: "+ Student characteristics", 2: "+ School characteristics", 3: "+ Random study-hours slope"}
        x_lab = [f"{stages[0]} model"] + [step_names.get(i, stages[i]) for i in range(1, len(bv))] + ["Remaining (final)"]
        measures = ["absolute"] + ["relative"] * (len(bv) - 1) + ["total"]
        y_vals = [bv[0]] + list(np.diff(bv)) + [None]
        fig = go.Figure(go.Waterfall(
            x=x_lab, y=[v if v is not None else 0 for v in y_vals], measure=measures,
            text=[f"{bv[0]:.3f}"] + [f"{v:+.3f}" for v in np.diff(bv)] + [f"{bv[-1]:.3f}"], textposition="outside",
            textfont=dict(size=11, color=S.INK),
            connector=dict(line=dict(color="rgba(11,61,44,0.25)", width=1, dash="dot")),
            increasing=dict(marker=dict(color=S.TERRA)), decreasing=dict(marker=dict(color=S.EMERALD)),
            totals=dict(marker=dict(color=S.GOLD)),
            hovertemplate="%{x}<br>%{y:.3f}<extra></extra>"))
        C.base_layout(fig, height=330)
        fig.update_yaxes(title_text="Between-school variance", range=[0, bv[0] * 1.2])
        fig.update_xaxes(showgrid=False, tickfont=dict(size=10))
        chart(fig, "rq4")
        last = vp.iloc[-1]
        slope_txt = (f" The final model also carries a random study-hours slope variance of {last['random_slope_variance']:.4f}."
                     if pd.notna(last.get("random_slope_variance")) else "")
        S.insight(
            f"{rq.loc['RQ4', 'answer']} Between-school variance moves from <b>{bv[0]:.3f}</b> in the null model to "
            f"<b>{bv[-1]:.3f}</b> in the final model, a {I.pct(between_red)} reduction, so <b>{I.pct(remaining)}</b> of the initial "
            f"school-level variance is still unexplained by the measured characteristics.{slope_txt} In the random-slope model the "
            "intercept variance refers to a student at the average study time.",
            title="Interpretasi",
        )

    with st.expander("Research-question summary as stored in the workbook"):
        tbl = D.sheet(D.S_RQ).copy()
        st.dataframe(tbl, hide_index=True, width="stretch")
        st.caption(f"Fixed effects: {D.S_FINAL_FIXED} · Variance path: {D.S_VAR_PATH} · "
                   "Results are fixed to the locked full-sample model and do not respond to sidebar filters.")
        st.dataframe(D.sheet(D.S_FINAL_FIXED), hide_index=True, width="stretch")
