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
    a, b = fx["ci_low"] * ratio, fx["ci_high"] * ratio
    fx["std_low"], fx["std_high"] = np.minimum(a, b), np.maximum(a, b)
    fx["label"] = fx["predictor"].map(D.TERM_LABELS).fillna(fx["predictor"])
    return fx


def render(ctx) -> None:
    rq = D.sheet(D.S_RQ).set_index("rq")
    vp = D.sheet(D.S_VAR_PATH)
    fx = _std_ci(D.sheet(D.S_FINAL_FIXED))
    fx_model = fx["model_id"].iloc[0]
    inference = fx["inference"].iloc[0] if "inference" in fx else ""

    actions = S.header_row(
        "What Explains Mathematics Achievement?",
        "Four research questions, four direct statistical answers.",
        chips=[f"Locked final model {fx_model}", inference, "Full sample"],
        deco="26k_sparkles.png", actions_width=2,
    )
    with actions, st.popover("Workbook tables", icon=":material/table:", width="stretch"):
        st.caption("Research-question summary (07_RQ_summary) and locked fixed effects (09_final_fixed_KR). "
                   "These results do not respond to sidebar filters.")
        st.dataframe(D.sheet(D.S_RQ), hide_index=True, width="stretch")
        st.dataframe(D.sheet(D.S_FINAL_FIXED), hide_index=True, width="stretch")

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
             sub=f"β = {strongest['standardized_beta']:.3f} · RQ2"),
        dict(label=rq.loc["RQ2", "metric_1_label"], value=I.pct(within_red), icon="trending_down", tone="emerald", sub="RQ2",
             help="Reduction in within-school (residual) variance relative to the null model."),
        dict(label=rq.loc["RQ3", "metric_1_label"], value=I.pct(add_between_red), icon="domain", tone="blue", sub="RQ3",
             help="Additional reduction in between-school variance when school characteristics are added."),
        dict(label="School variance remaining", value=I.pct(remaining), icon="hourglass_bottom", tone="terra", sub="RQ4",
             help="Share of the null-model between-school variance that remains in the final model."),
    ])

    # ---------------- RQ1 + RQ2 ----------------
    c1, c2 = st.columns(2, gap="small")
    with c1, S.card("rq1"):
        S.card_head("RQ 01", "Where Does the Variation Come From?",
                    "Null-model variance partition · within vs between schools", tone="blue", pill_icon="pie_chart")
        null = vp.iloc[0]
        tb, tw = null["between_school_variance"], null["within_student_variance"]
        fig = go.Figure()
        fig.add_trace(go.Bar(y=["Variance"], x=[within], orientation="h", name="Within schools",
                             marker=dict(color=S.EMERALD, line=dict(color="#FFFFFF", width=2)),
                             text=[f"Within schools {I.pct(within)} (σ² = {tw:.2f})"], textposition="inside",
                             insidetextanchor="middle", textfont=dict(color="#FFFFFF", size=12),
                             hovertemplate="Within schools: %{x:.1%}<extra></extra>"))
        fig.add_trace(go.Bar(y=["Variance"], x=[icc], orientation="h", name="Between schools (ICC)",
                             marker=dict(color=S.GOLD, line=dict(color="#FFFFFF", width=2)),
                             text=[f"{I.pct(icc)}"], textposition="outside", textfont=dict(color=S.INK, size=12),
                             hovertemplate=f"Between schools: %{{x:.1%}}<br>σ² = {tb:.3f}<extra></extra>"))
        C.base_layout(fig, legend=True, barmode="stack")
        fig.update_layout(margin=dict(t=24, l=10))
        fig.update_xaxes(tickformat=".0%", range=[0, 1.14], title_text="Share of total variance")
        fig.update_yaxes(showticklabels=False, showgrid=False)
        chart(fig, "rq1")
        S.insight(f"About <b>{I.pct(icc)}</b> of score variation lies between schools (ICC = {icc:.3f}; σ² {tb:.3f}) and "
                  f"<b>{I.pct(within)}</b> within schools (σ² {tw:.3f}). This is a variance partition, not a causal share; "
                  "even a modest ICC means students in a school are not independent, which justifies the multilevel model.",
                  lines=4)

    with c2, S.card("rq2"):
        S.card_head("RQ 02 · Student Level", "Which Student Characteristics Matter Most?",
                    f"Standardized coefficients with 95% CI · {fx_model} ({inference})", tone="mint", pill_icon="person_search")
        d = l1.copy()
        d["color"] = [S.EMERALD if v >= 0 else S.TERRA for v in d["standardized_beta"]]
        d["hover"] = [f"Raw estimate {e:+.4f} (95% CI {lo:.4f} to {hi:.4f})<br>{I.pval(p)}"
                      for e, lo, hi, p in zip(d["estimate"], d["ci_low"], d["ci_high"], d["p_value"])]
        fig = C.forest(d, "label", "standardized_beta", "std_low", "std_high", color_col="color",
                       x_title="Standardized β", hover_extra="hover")
        fig.update_layout(margin=dict(l=110))
        chart(fig, "rq2")
        sec = l1.iloc[1]
        rest = "; ".join(f"{r['label']} {r['standardized_beta']:+.3f}" for _, r in l1.iloc[2:].iterrows())
        S.insight(f"Holding other predictors constant, <b>{strongest['label']}</b> has the strongest adjusted association "
                  f"(β = {strongest['standardized_beta']:.3f}, {I.pval(strongest['p_value'])}), then {sec['label']} "
                  f"(β = {sec['standardized_beta']:.3f}); {rest}. Student terms cut within-school variance by "
                  f"{I.pct(within_red)}. These are associations, not causal effects.", lines=4)

    # ---------------- RQ3 + RQ4 ----------------
    c3, c4 = st.columns(2, gap="small")
    with c3, S.card("rq3"):
        S.card_head("RQ 03 · School Level", "Do School Characteristics Matter?",
                    f"Level-2 fixed effects with 95% CI (Kenward-Roger df) · {fx_model}", tone="slate", pill_icon="domain")
        d = fx[fx["predictor"].isin(L2_TERMS)].copy()
        sc = D.schools()
        sd_map = {"facility_c": sc["facility"].std(), "teacher_c": sc["teacher.exp"].std()}
        unit = {"facility_c": "per facility point", "teacher_c": "per year"}
        d["color"] = [S.BLUE] * len(d)
        d["label"] = [f"{D.TERM_LABELS[p]} ({unit[p]})" for p in d["predictor"]]
        d["hover"] = [f"df = {df_:.1f}, {I.pval(p)}<br>Standardized β = {b:.3f}"
                      for df_, p, b in zip(d["df"], d["p_value"], d["standardized_beta"])]
        fig = C.forest(d, "label", "estimate", "ci_low", "ci_high", color_col="color",
                       x_title="Change in math score per unit", hover_extra="hover")
        fig.update_layout(margin=dict(l=170))
        chart(fig, "rq3")
        parts = [f"<b>{D.TERM_LABELS[r['predictor']]}</b> {r['estimate']:+.3f} {unit[r['predictor']]} "
                 f"(95% CI {r['ci_low']:.3f} to {r['ci_high']:.3f}, {I.pval(r['p_value'])}; 1 SD ≈ "
                 f"{r['estimate'] * sd_map[r['predictor']]:+.2f} points)" for _, r in d.iterrows()]
        S.insight("; ".join(parts) + f". Both intervals exclude zero but the effects are small next to SES; adding them cuts "
                  f"between-school variance by a further {I.pct(add_between_red)}. With 30 schools these are associations only.",
                  lines=4)

    with c4, S.card("rq4"):
        S.card_head("RQ 04", "What Still Remains Unexplained?",
                    "Between-school (intercept) variance from null to final model · workbook variance path",
                    tone="terra", pill_icon="waterfall_chart")
        stages = vp["model_stage"].tolist()
        bv = vp["between_school_variance"].to_numpy(dtype=float)
        step_names = {1: "+ Student", 2: "+ School", 3: "+ Random slope"}
        x_lab = [f"{stages[0]} model"] + [step_names.get(i, stages[i]) for i in range(1, len(bv))] + ["Remaining"]
        measures = ["absolute"] + ["relative"] * (len(bv) - 1) + ["total"]
        fig = go.Figure(go.Waterfall(
            x=x_lab, y=[bv[0]] + list(np.diff(bv)) + [0], measure=measures,
            text=[f"{bv[0]:.3f}"] + [f"{v:+.3f}" for v in np.diff(bv)] + [f"{bv[-1]:.3f}"], textposition="outside",
            textfont=dict(size=10, color=S.INK),
            connector=dict(line=dict(color="rgba(11,61,44,0.25)", width=1, dash="dot")),
            increasing=dict(marker=dict(color=S.TERRA)), decreasing=dict(marker=dict(color=S.EMERALD)),
            totals=dict(marker=dict(color=S.GOLD)), hovertemplate="%{x}<br>%{y:.3f}<extra></extra>"))
        C.base_layout(fig)
        fig.update_yaxes(title_text="Between-school σ²", range=[0, bv[0] * 1.22])
        fig.update_xaxes(showgrid=False, tickfont=dict(size=10))
        chart(fig, "rq4")
        last = vp.iloc[-1]
        slope_txt = (f" The final model also has a random study-hours slope variance of {last['random_slope_variance']:.4f}."
                     if pd.notna(last.get("random_slope_variance")) else "")
        S.insight(f"Between-school variance falls from <b>{bv[0]:.3f}</b> to <b>{bv[-1]:.3f}</b> ({I.pct(between_red)} lower), "
                  f"so <b>{I.pct(remaining)}</b> of the initial school-level variance remains unexplained.{slope_txt} "
                  "With a random slope, the intercept variance refers to a student at average study time.", lines=4)
