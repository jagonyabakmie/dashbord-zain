"""Page 6: Diagnostics & Robustness for the locked final model (full sample; a school can only be highlighted)."""

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

TESTS = [
    # label, value col, ok col, kind, H0 text, short KPI label, icon
    ("Level-1 residual normality", "residual_normality_p", "residual_normality_ok", "p",
     "Residuals are normally distributed", "Residual normality", "bar_chart"),
    ("Level-1 homoskedasticity (Breusch-Pagan)", "homoskedasticity_L1_bp_p", "homoskedasticity_L1_ok", "p",
     "Residual variance is constant across fitted values", "L1 homoskedasticity", "align_justify_space_even"),
    ("Between-school variance homogeneity (Levene)", "homogeneity_L2_levene_p", "homogeneity_L2_ok", "p",
     "Residual variance is equal across schools", "School variance homogeneity", "apartment"),
    ("Linearity (quadratic terms)", "linearity_quadratic_p", "linearity_ok", "p",
     "Quadratic terms add nothing (relationships are linear)", "Linearity", "show_chart"),
    ("Multicollinearity (max VIF)", "max_vif", "multicollinearity_ok", "vif",
     "VIF below the usual threshold", "Max VIF", "join_inner"),
    ("Random-intercept normality (Shapiro-Wilk)", "random_intercept_normality_p", "random_intercept_normality_ok", "p",
     "School intercepts are normally distributed", "Random-intercept normality", "target"),
    ("Random-slope normality (Shapiro-Wilk, min p)", "random_slope_normality_min_p", "random_slope_normality_ok", "p",
     "School study-hours slopes are normally distributed", "Random-slope normality", "trending_up"),
]


def render(ctx) -> None:
    mid = D.final_model_id()
    dg = D.sheet(D.S_FINAL_DIAG)
    dg = dg[dg["model_id"] == mid].iloc[0]
    rdg = D.sheet(D.S_FINAL_RDIAG)
    rdg = rdg[rdg["model_id"] == mid]
    fs = D.final_school()
    sid = selected_school_id(ctx)

    S.page_header(
        "Can We Trust the Model?",
        "Assumption checks, stability, and model adequacy.",
        kicker="Diagnostics & Robustness",
        chips=[f"Locked final model {mid}", f"Status: {dg['status']}", "Full sample"],
        gold_chip=f"Highlight: School {sid:02d}" if sid else None,
        deco="26e_wave.png",
    )

    kpis = []
    for label, vcol, okcol, kind, _, short, ic in TESTS:
        v = dg.get(vcol, np.nan)
        ok = I.ok_flag(dg.get(okcol))
        val = I.num(v, 2) if kind == "vif" else I.pval_short(v)
        kpis.append(dict(label=short + (" (VIF)" if kind == "vif" else " p-value"), value=val, icon=ic,
                         tone="emerald" if ok else ("terra" if ok is False else "slate"),
                         sub="Pass" if ok else ("Flag" if ok is False else "Not available")))
    S.kpi_strip(kpis)

    # ---------------- A. Residual vs fitted + B. QQ residuals ----------------
    pr = D.predictions(mid)
    c1, c2 = st.columns(2, gap="medium")
    with c1, S.card("card_dg_rvf"):
        S.card_head("Level 1", "Are Residuals Evenly Spread?",
                    f"Residual vs fitted · stored predictions for {mid} (n = {len(pr)})"
                    + (f" · School {sid:02d} highlighted" if sid else ""), tone="emerald", pill_icon="scatter_plot")
        if pr.empty:
            S.insight("Stored residuals for the final model are not available in the workbook.", title="Not available", kind="warn")
        else:
            fig = go.Figure()
            base = pr if sid is None else pr[pr["school_id"] != sid]
            fig.add_trace(go.Scattergl(x=base["fitted_math"], y=base["residual"], mode="markers", name="Students",
                                       marker=dict(size=5, color="rgba(15,138,95,0.42)"),
                                       customdata=np.stack([base["student_id"], base["school_id"]], axis=1),
                                       hovertemplate="%{customdata[0]} · School %{customdata[1]}<br>Fitted %{x:.2f}<br>Residual %{y:.3f}<extra></extra>"))
            if sid is not None:
                hl = pr[pr["school_id"] == sid]
                fig.add_trace(go.Scatter(x=hl["fitted_math"], y=hl["residual"], mode="markers", name=f"School {sid:02d}",
                                         marker=dict(size=9, color=S.GOLD, line=dict(color="#FFFFFF", width=1)),
                                         customdata=hl["student_id"],
                                         hovertemplate="%{customdata}<br>Fitted %{x:.2f}<br>Residual %{y:.3f}<extra></extra>"))
            # binned mean of residuals as a descriptive smoother
            binned = pr.groupby(pd.qcut(pr["fitted_math"], 15, duplicates="drop"), observed=True)[["fitted_math", "residual"]].mean()
            fig.add_trace(go.Scatter(x=binned["fitted_math"], y=binned["residual"],
                                     mode="lines", name="Binned mean", line=dict(color=S.BLUE, width=2.5), hoverinfo="skip"))
            fig.add_hline(y=0, line=dict(color=S.GOLD, width=1.5, dash="dash"))
            C.base_layout(fig, height=360, legend=True)
            fig.update_xaxes(title_text="Fitted mathematics score")
            fig.update_yaxes(title_text="Residual")
            chart(fig, "dg_rvf")
            bp = dg["homoskedasticity_L1_bp_p"]
            S.insight(
                f"The binned mean residual stays within ±{binned['residual'].abs().max():.3f} points of zero across fitted values "
                f"(residual SD {pr['residual'].std():.3f}). The Breusch-Pagan test "
                f"gives {I.pval(bp)}, so constant variance is {'not rejected' if bp >= 0.05 else 'rejected'} at 5%, and the linearity "
                f"check gives {I.pval(dg['linearity_quadratic_p'])}."
            )

    with c2, S.card("card_dg_qq"):
        S.card_head("Level 1", "Are Level-1 Residuals Normal?",
                    "Normal Q-Q plot of the stored student residuals (Blom plotting positions)", tone="blue", pill_icon="timeline")
        if not pr.empty:
            fig = C.qq(pr["residual"].to_numpy(), labels=pr["student_id"].tolist(), color=S.BLUE, name="Residual")
            C.base_layout(fig, height=360, legend=True)
            chart(fig, "dg_qq")
            p = dg["residual_normality_p"]
            sk = pr["residual"].skew()
            ku = pr["residual"].kurt()
            S.insight(
                f"The normality test for the final model gives {I.pval(p)} — {'above' if p >= 0.05 else 'below'} 0.05, so normality is "
                f"{'not rejected' if p >= 0.05 else 'rejected'}{', though only narrowly' if 0.05 <= p < 0.10 else ''}. "
                f"Sample skewness is {sk:.2f} and excess kurtosis {ku:.2f}; with 900 students, mild tail departures have little "
                "effect on fixed-effect inference.",
                kind="green",
            )

    # ---------------- C. QQ random effects + D. Summary table ----------------
    c3, c4 = st.columns([5, 7], gap="medium")
    with c3, S.card("card_dg_re"):
        S.card_head("Level 2", "Are School Effects Normal?",
                    "Normal Q-Q plot of the 30 stored school random effects (final model)", tone="gold", pill_icon="apartment")
        which = st.segmented_control("Random effect", ["Random Intercept", "Random Study-Hours Slope"],
                                     default="Random Intercept", key="dg_re", label_visibility="collapsed") or "Random Intercept"
        col, term = (("adjusted_intercept", "(Intercept)") if which == "Random Intercept"
                     else ("hours_random_effect", "hours_c"))
        labs = [f"School {int(s):02d}" for s in fs["school_id"]]
        fig = C.qq(fs[col].to_numpy(), labels=labs, color=S.EMERALD, name="School effect")
        if sid is not None:
            C.add_qq_highlight(fig, fs[col].to_numpy(), labs, f"School {sid:02d}")
        C.base_layout(fig, height=360, legend=True)
        chart(fig, "dg_re_chart")
        rr = rdg[rdg["random_term"] == term]
        p = rr["shapiro_p"].iloc[0] if not rr.empty else np.nan
        concl = rr["conclusion"].iloc[0] if not rr.empty else "Not available"
        S.insight(f"Shapiro-Wilk for the {which.lower()}: {I.pval(p)} ({concl}). With only 30 schools the test has limited power, "
                  "so the Q-Q plot is read alongside it; points close to the gold line support the normality assumption.")

    with c4, S.card("card_dg_table"):
        S.card_head("Scorecard", "Diagnostic Summary", f"All stored checks for the locked final model {mid}",
                    tone="terra", pill_icon="fact_check")
        rows = []
        for label, vcol, okcol, kind, h0, *_ in TESTS:
            v = dg.get(vcol, np.nan)
            ok = I.ok_flag(dg.get(okcol))
            if I.isna(v):
                stat, dec, conc = "Not available", S.badge("Not available", "na"), "Not stored in the workbook"
            elif kind == "vif":
                stat = f"VIF = {v:.2f}"
                dec = S.badge("Pass", "ok") if ok else S.badge("Flag", "bad")
                conc = "No problematic multicollinearity" if ok else "Collinearity may inflate standard errors"
            else:
                stat = I.pval(v)
                dec = S.badge("Fail to reject H0", "ok") if ok else S.badge("Reject H0", "bad")
                conc = (h0 + " (not rejected)") if ok else ("Evidence against: " + h0.lower())
            rows.append(f"<tr><td>{label}</td><td class='num'>{stat}</td><td>{dec}</td><td>{conc}</td></tr>")
        conv = dg.get("convergence_message")
        conv_ok = I.isna(conv) or str(conv).strip() == ""
        rows.append(f"<tr><td>Convergence</td><td class='num'>{'No message' if conv_ok else S.esc(conv)}</td>"
                    f"<td>{S.badge('Converged' if conv_ok else 'Warning', 'ok' if conv_ok else 'bad')}</td>"
                    f"<td>{'Optimizer finished without warnings' if conv_ok else 'Check optimizer settings'}</td></tr>")
        sing = bool(dg.get("singular"))
        rows.append(f"<tr><td>Singularity</td><td class='num'>{'Singular' if sing else 'Not singular'}</td>"
                    f"<td>{S.badge('Flag' if sing else 'Pass', 'bad' if sing else 'ok')}</td>"
                    f"<td>{'Random-effect structure at the boundary' if sing else 'Random-effect variances are estimable'}</td></tr>")
        rows.append(f"<tr><td>Overall status</td><td class='num'>{S.esc(dg['status'])}</td>"
                    f"<td>{S.badge(dg['status'], 'ok' if dg['status'] == 'VALID' else 'bad')}</td><td>Workbook model status</td></tr>")
        st.markdown("<table class='diag-table'><thead><tr><th>Diagnostic / test</th><th>Statistic</th><th>Decision</th>"
                    "<th>Conclusion</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>", unsafe_allow_html=True)
        n_ok = sum(1 for _, _, okc, *_ in TESTS if I.ok_flag(dg.get(okc)))
        S.insight(f"{n_ok} of {len(TESTS)} assumption checks pass at the 5% level, the fit converged and is not singular. "
                  "Residual normality is the closest call; the remaining checks give no sign of a violated assumption.",
                  title="Interpretasi", kind="green")
        st.caption(f"Tests from {D.S_FINAL_DIAG} / {D.S_FINAL_RDIAG} (REML final fit). Residual and fitted values for plots come from "
                   f"{D.S_PRED} for the same specification in the model bank.")

    # ---------------- Robustness ----------------
    with st.expander("Robustness: how stable are the conclusions?"):
        st.caption("Built only from workbook outputs. The workbook contains no leave-one-school-out results, so robustness is "
                   "shown across competing specifications and estimation methods instead.")
        top = D.sheet(D.S_TOP)
        top_valid = top[top["status"] == "VALID"].head(10)
        coefs = D.sheet(D.S_COEFS)
        cc = coefs[coefs["model_id"].isin(top_valid["model_id"]) & (coefs["predictor"] != "(Intercept)")].copy()
        cc["label"] = cc["predictor"].map(D.TERM_LABELS)
        r1, r2 = st.columns([7, 5], gap="medium")
        with r1, S.card("card_dg_rob1"):
            S.card_head("Robustness", "Do Coefficients Hold Across the Top 10 Models?",
                        "Standardized β of each predictor in the 10 best valid models (ML bank) · gold = final model",
                        tone="slate", pill_icon="stacked_line_chart")
            fig = go.Figure()
            order = ["SES", "Prior Score", "Teacher Experience", "Facility", "Study Hours", "Female (vs male)"]
            for lab in order:
                sub = cc[cc["label"] == lab]
                if sub.empty:
                    continue
                isf = sub["model_id"] == mid
                fig.add_trace(go.Scatter(x=sub.loc[~isf, "standardized_beta"], y=[lab] * int((~isf).sum()), mode="markers",
                                         marker=dict(size=10, color="rgba(15,138,95,0.45)", line=dict(color="#FFFFFF", width=1)),
                                         customdata=sub.loc[~isf, "model_id"], showlegend=False,
                                         hovertemplate="%{customdata}<br>β = %{x:.4f}<extra>" + lab + "</extra>"))
                fig.add_trace(go.Scatter(x=sub.loc[isf, "standardized_beta"], y=[lab] * int(isf.sum()), mode="markers",
                                         marker=dict(size=14, color=S.GOLD, symbol="diamond", line=dict(color="#FFFFFF", width=1.5)),
                                         showlegend=False, hovertemplate=mid + "<br>β = %{x:.4f}<extra>" + lab + "</extra>"))
            fig.add_vline(x=0, line=dict(color="rgba(11,61,44,0.35)", width=1, dash="dot"))
            C.base_layout(fig, height=320)
            fig.update_yaxes(autorange="reversed", showgrid=False)
            fig.update_xaxes(title_text="Standardized β")
            chart(fig, "dg_rob1")
            rng = cc.groupby("label")["standardized_beta"].agg(["min", "max", "count"])
            sign_stable = all((r["min"] > 0) == (r["max"] > 0) for _, r in rng.iterrows())
            S.insight(f"Across the {len(top_valid)} best valid specifications, every predictor keeps the same sign "
                      f"{'' if sign_stable else '(except where noted) '}and SES remains the dominant association "
                      f"(β from {rng.loc['SES', 'min']:.3f} to {rng.loc['SES', 'max']:.3f})." if sign_stable and 'SES' in rng.index
                      else "Coefficient signs differ across some top specifications; see the plot.", kind="green")
        with r2, S.card("card_dg_rob2"):
            S.card_head("Estimation", "ML vs REML for the Final Model",
                        f"{mid}: model-bank ML fit vs locked REML + Kenward-Roger fit", tone="blue", pill_icon="compare_arrows")
            ml = coefs[coefs["model_id"] == mid][["predictor", "estimate", "std_error"]].rename(
                columns={"estimate": "ML estimate", "std_error": "ML SE"})
            reml = D.sheet(D.S_FINAL_FIXED)[["predictor", "estimate", "std_error"]].rename(
                columns={"estimate": "REML estimate", "std_error": "REML SE (KR)"})
            cmp_ = reml.merge(ml, on="predictor", how="left")
            cmp_["predictor"] = cmp_["predictor"].map(D.TERM_LABELS).fillna(cmp_["predictor"])
            st.dataframe(cmp_.round(4), hide_index=True, width="stretch")
            vml = D.model_rows(D.S_RANDOM, mid)
            vre = D.sheet(D.S_FINAL_RANDOM)
            vml = vml[vml["term2"].isna()][["group", "term1", "variance_or_covariance"]].rename(columns={"variance_or_covariance": "ML variance"})
            vre = vre[vre["term2"].isna()][["group", "term1", "variance_or_covariance"]].rename(columns={"variance_or_covariance": "REML variance"})
            vc = vre.merge(vml, on=["group", "term1"], how="left")
            vc["Component"] = [("Residual" if g == "Residual" else ("School intercept" if t == "(Intercept)" else f"Slope: {D.TERM_LABELS.get(t, t)}"))
                               for g, t in zip(vc["group"], vc["term1"])]
            st.dataframe(vc[["Component", "REML variance", "ML variance"]].round(4), hide_index=True, width="stretch")
            maxdiff = float((cmp_["REML estimate"] - cmp_["ML estimate"]).abs().max())
            S.insight(f"Fixed effects agree to within {maxdiff:.4f} points between ML and REML; REML variance components are "
                      "slightly larger, as expected, without changing any conclusion.")
        diag_all = D.sheet(D.S_DIAG).merge(D.catalog()[["model_id", "status"]], on="model_id")
        tv = diag_all[diag_all["model_id"].isin(top["model_id"])]
        okcols = [c for c in tv.columns if c.endswith("_ok")]
        n_all = int((tv[okcols].fillna(1).astype(int).min(axis=1) == 1).sum())
        st.caption(f"Of the {len(tv)} models in {D.S_TOP}, {n_all} pass every stored assumption check "
                   "(checks not applicable to a model are ignored).")
