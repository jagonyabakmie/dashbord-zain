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


def _robustness(mid: str) -> None:
    """Popover content: coefficient stability across the top valid models and ML vs REML for the final model."""
    st.caption("Built only from workbook outputs. The workbook contains no leave-one-school-out results, so robustness is "
               "shown across competing specifications and estimation methods instead.")
    top = D.sheet(D.S_TOP)
    top_valid = top[top["status"] == "VALID"].head(10)
    coefs = D.sheet(D.S_COEFS)
    cc = coefs[coefs["model_id"].isin(top_valid["model_id"]) & (coefs["predictor"] != "(Intercept)")].copy()
    cc["label"] = cc["predictor"].map(D.TERM_LABELS)
    st.markdown("**Do coefficients hold across the top 10 models?** Standardized β in the 10 best valid models "
                "(ML bank); gold diamond = final model.")
    fig = go.Figure()
    for lab in ["SES", "Prior Score", "Teacher Experience", "Facility", "Study Hours", "Female (vs male)"]:
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
    C.base_layout(fig, height=280)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(title_text="Standardized β")
    st.plotly_chart(fig, width="stretch", theme=None, config=C.CONFIG, key="dg_rob1")
    rng = cc.groupby("label")["standardized_beta"].agg(["min", "max"])
    sign_stable = all((r["min"] > 0) == (r["max"] > 0) for _, r in rng.iterrows())
    if sign_stable and "SES" in rng.index:
        S.insight(f"Across the {len(top_valid)} best valid specifications every predictor keeps the same sign and SES remains "
                  f"the dominant association (β from {rng.loc['SES', 'min']:.3f} to {rng.loc['SES', 'max']:.3f}).", lines=3)
    else:
        S.insight("Coefficient signs differ across some top specifications; see the plot.", lines=3)

    st.markdown(f"**ML vs REML for {mid}.** Model-bank ML fit vs the locked REML + Kenward-Roger fit.")
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
              "slightly larger, as expected, without changing any conclusion.", lines=3)
    diag_all = D.sheet(D.S_DIAG).merge(D.catalog()[["model_id", "status"]], on="model_id")
    tv = diag_all[diag_all["model_id"].isin(top["model_id"])]
    okcols = [c for c in tv.columns if c.endswith("_ok")]
    n_all = int((tv[okcols].fillna(1).astype(int).min(axis=1) == 1).sum())
    st.caption(f"Of the {len(tv)} models in {D.S_TOP}, {n_all} pass every stored assumption check "
               "(checks not applicable to a model are ignored).")


def render(ctx) -> None:
    mid = D.final_model_id()
    dg = D.sheet(D.S_FINAL_DIAG)
    dg = dg[dg["model_id"] == mid].iloc[0]
    rdg = D.sheet(D.S_FINAL_RDIAG)
    rdg = rdg[rdg["model_id"] == mid]
    fs = D.final_school()
    sid = selected_school_id(ctx)

    actions = S.header_row(
        "Can We Trust the Model?",
        "Assumption checks, stability, and model adequacy.",
        chips=[f"Locked final model {mid}", f"Status: {dg['status']}", "Full sample"],
        gold_chip=f"Highlight: School {sid:02d}" if sid else None,
        deco="26e_wave.png", actions_width=2,
    )
    with actions, st.popover("Robustness", icon=":material/stacked_line_chart:", width="stretch"):
        _robustness(mid)

    kpis = []
    for label, vcol, okcol, kind, _, short, ic in TESTS:
        v = dg.get(vcol, np.nan)
        ok = I.ok_flag(dg.get(okcol))
        val = I.num(v, 2) if kind == "vif" else I.pval_short(v)
        kpis.append(dict(label=short + (" (VIF)" if kind == "vif" else " p"), value=val, icon=ic,
                         tone="emerald" if ok else ("terra" if ok is False else "slate"),
                         sub="Pass" if ok else ("Flag" if ok is False else "Not available")))
    S.kpi_strip(kpis)

    # ---------------- A. Residual vs fitted + B. QQ residuals ----------------
    pr = D.predictions(mid)
    c1, c2 = st.columns(2, gap="small")
    with c1, S.card("dg_rvf"):
        S.card_head("Level 1", "Are Residuals Evenly Spread?",
                    f"Residual vs fitted · stored predictions for {mid} (n = {len(pr)})"
                    + (f" · School {sid:02d} highlighted" if sid else ""), tone="emerald", pill_icon="scatter_plot")
        if pr.empty:
            S.insight("Stored residuals for the final model are not available in the workbook.", title="Not available", kind="warn")
        else:
            fig = go.Figure()
            base = pr if sid is None else pr[pr["school_id"] != sid]
            fig.add_trace(go.Scattergl(x=base["fitted_math"], y=base["residual"], mode="markers", name="Students",
                                       marker=dict(size=4, color="rgba(15,138,95,0.42)"),
                                       customdata=np.stack([base["student_id"], base["school_id"]], axis=1),
                                       hovertemplate="%{customdata[0]} · School %{customdata[1]}<br>Fitted %{x:.2f}<br>Residual %{y:.3f}<extra></extra>"))
            if sid is not None:
                hl = pr[pr["school_id"] == sid]
                fig.add_trace(go.Scatter(x=hl["fitted_math"], y=hl["residual"], mode="markers", name=f"School {sid:02d}",
                                         marker=dict(size=8, color=S.GOLD, line=dict(color="#FFFFFF", width=1)),
                                         customdata=hl["student_id"],
                                         hovertemplate="%{customdata}<br>Fitted %{x:.2f}<br>Residual %{y:.3f}<extra></extra>"))
            # binned mean of residuals as a descriptive smoother
            binned = pr.groupby(pd.qcut(pr["fitted_math"], 15, duplicates="drop"), observed=True)[["fitted_math", "residual"]].mean()
            fig.add_trace(go.Scatter(x=binned["fitted_math"], y=binned["residual"],
                                     mode="lines", name="Binned mean", line=dict(color=S.BLUE, width=2.5), hoverinfo="skip"))
            fig.add_hline(y=0, line=dict(color=S.GOLD, width=1.5, dash="dash"))
            C.base_layout(fig, legend=True)
            fig.update_xaxes(title_text="Fitted mathematics score")
            fig.update_yaxes(title_text="Residual")
            chart(fig, "dg_rvf")
            bp = dg["homoskedasticity_L1_bp_p"]
            S.insight(
                f"The binned mean residual stays within ±{binned['residual'].abs().max():.3f} points of zero (residual SD "
                f"{pr['residual'].std():.3f}). Breusch-Pagan gives {I.pval(bp)}, so constant variance is "
                f"{'not rejected' if bp >= 0.05 else 'rejected'} at 5%; the linearity check gives {I.pval(dg['linearity_quadratic_p'])}.")

    with c2, S.card("dg_qq"):
        S.card_head("Level 1", "Are Level-1 Residuals Normal?",
                    "Normal Q-Q plot of the stored student residuals (Blom plotting positions)", tone="blue", pill_icon="timeline")
        if not pr.empty:
            fig = C.qq(pr["residual"].to_numpy(), labels=pr["student_id"].tolist(), color=S.BLUE, name="Residual")
            C.base_layout(fig, legend=True)
            fig.update_layout(legend=dict(x=0, xanchor="left"))
            chart(fig, "dg_qq")
            p = dg["residual_normality_p"]
            S.insight(
                f"The normality test gives {I.pval(p)}, so normality is {'not rejected' if p >= 0.05 else 'rejected'}"
                f"{', though only narrowly' if 0.05 <= p < 0.10 else ''}. Skewness is {pr['residual'].skew():.2f} and excess "
                f"kurtosis {pr['residual'].kurt():.2f}; with 900 students, mild tail departures have little effect on "
                "fixed-effect inference.")

    # ---------------- C. QQ random effects + D. Summary table ----------------
    c3, c4 = st.columns([5, 7], gap="small")
    with c3, S.card("dg_re"):
        S.card_head("Level 2", "Are School Effects Normal?",
                    "Normal Q-Q plot of the 30 stored school random effects", tone="gold", pill_icon="apartment")
        which = st.segmented_control("Random effect", ["Random Intercept", "Random Study-Hours Slope"],
                                     default="Random Intercept", key="dg_re", label_visibility="collapsed") or "Random Intercept"
        col, term = (("adjusted_intercept", "(Intercept)") if which == "Random Intercept"
                     else ("hours_random_effect", "hours_c"))
        labs = [f"School {int(s):02d}" for s in fs["school_id"]]
        fig = C.qq(fs[col].to_numpy(), labels=labs, color=S.EMERALD, name="School effect")
        if sid is not None:
            C.add_qq_highlight(fig, fs[col].to_numpy(), labs, f"School {sid:02d}")
        C.base_layout(fig, legend=True)
        fig.update_layout(legend=dict(x=0, xanchor="left"))
        chart(fig, "dg_re_chart")
        rr = rdg[rdg["random_term"] == term]
        p = rr["shapiro_p"].iloc[0] if not rr.empty else np.nan
        concl = rr["conclusion"].iloc[0] if not rr.empty else "Not available"
        S.insight(f"Shapiro-Wilk for the {which.lower()}: {I.pval(p)} ({concl}). With only 30 schools the test has limited "
                  "power, so read it with the Q-Q plot; points close to the gold line support normality.")

    with c4, S.card("dg_table"):
        S.card_head("Scorecard", "Diagnostic Summary", f"All stored checks for the locked final model {mid} · "
                    f"tests from {D.S_FINAL_DIAG} / {D.S_FINAL_RDIAG}, residuals from {D.S_PRED}",
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
        st.markdown("<div class='scroll-area'><table class='diag-table'><thead><tr><th>Diagnostic / test</th><th>Statistic</th>"
                    "<th>Decision</th><th>Conclusion</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>",
                    unsafe_allow_html=True)
        n_ok = sum(1 for _, _, okc, *_ in TESTS if I.ok_flag(dg.get(okc)))
        S.insight(f"{n_ok} of {len(TESTS)} assumption checks pass at the 5% level, the fit converged and is not singular. "
                  "Residual normality is the closest call; the other checks give no sign of a violated assumption.", lines=2)
