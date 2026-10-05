"""Page 4: Multilevel Model Laboratory. Every specification maps to a precomputed model in 14_model_catalog."""

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

L1_OPTS = {0: "Exclude", 1: "Fixed", 2: "Fixed + Random slope"}
L2_OPTS = {0: "Exclude", 1: "Fixed"}
STATE_COLS = [c for c, *_ in D.L1_PREDICTORS] + [c for c, *_ in D.L2_PREDICTORS]

TEX = {"ses_c": r"\mathrm{SES}", "prior_c": r"\mathrm{Prior}", "hours_c": r"\mathrm{Hours}", "female": r"\mathrm{Female}",
       "facility_c": r"\mathrm{Facility}", "teacher_c": r"\mathrm{TeacherExp}"}


def _apply_states(states: dict) -> None:
    for col, val in states.items():
        st.session_state[f"lab_{col}"] = int(val)


def _states_of(model_id: str) -> dict:
    row = D.catalog().set_index("model_id").loc[model_id]
    return {c: int(row[c]) for c in STATE_COLS}


def _equations(states: dict) -> tuple[str, str, str]:
    l1 = [(t, states[c]) for c, t, *_ in D.L1_PREDICTORS if states[c] > 0]
    l2 = [(t, states[c]) for c, t, *_ in D.L2_PREDICTORS if states[c] > 0]
    # Level 1
    rhs = [r"\beta_{0j}"] + [rf"\beta_{{{k}j}}\,{TEX[t]}_{{ij}}" for k, (t, _) in enumerate(l1, start=1)]
    eq1 = r"\mathrm{Math}_{ij} = " + " + ".join(rhs) + r" + e_{ij},\quad e_{ij}\sim N(0,\sigma^2)"
    # Level 2
    g0 = [r"\gamma_{00}"] + [rf"\gamma_{{0{m}}}\,{TEX[t]}_{{j}}" for m, (t, _) in enumerate(l2, start=1)] + [r"u_{0j}"]
    lines = [r"\beta_{0j} = " + " + ".join(g0)]
    for k, (t, s) in enumerate(l1, start=1):
        lines.append(rf"\beta_{{{k}j}} = \gamma_{{{k}0}}" + (rf" + u_{{{k}j}}" if s == 2 else ""))
    eq2 = r"\begin{aligned}" + r" \\ ".join(l.replace(" = ", " &= ", 1) for l in lines) + r"\end{aligned}"
    # Combined
    fixed = [r"\gamma_{00}"] + [rf"\gamma_{{{k}0}}\,{TEX[t]}_{{ij}}" for k, (t, _) in enumerate(l1, start=1)] \
        + [rf"\gamma_{{0{m}}}\,{TEX[t]}_{{j}}" for m, (t, _) in enumerate(l2, start=1)]
    rand = [r"u_{0j}"] + [rf"u_{{{k}j}}\,{TEX[t]}_{{ij}}" for k, (t, s) in enumerate(l1, start=1) if s == 2]
    eq3 = (r"\mathrm{Math}_{ij} = \underbrace{" + " + ".join(fixed) + r"}_{\text{fixed}} + \underbrace{"
           + " + ".join(rand) + r"}_{\text{random (school)}} + e_{ij}")
    return eq1, eq2, eq3


def render(ctx) -> None:
    cat = D.catalog()
    final_id = D.final_model_id()
    best_id = D.best_model_id()
    meta = D.readme()

    if "lab_ses_state" not in st.session_state:
        _apply_states(_states_of(final_id))

    S.page_header(
        "Build Your Multilevel Model",
        "Select predictors, compare specifications, and inspect model quality.",
        kicker="Multilevel Model Laboratory",
        chips=[f"{len(cat)} precomputed models", f"{int((cat['status'] == 'VALID').sum())} valid · "
                                                 f"{int((cat['status'] != 'VALID').sum())} singular",
               "ML estimation for AIC/AICc comparison"],
        gold_chip=f"Best valid: {best_id}",
        deco="26j_decorative_circles.png",
    )

    # ---------------- Controls ----------------
    with S.card("card_lab_controls"):
        S.card_head("Specification", "Choose How Each Predictor Enters the Model",
                    "Random intercept for school is always on. Your choice is matched to a stored model; nothing is re-fitted.",
                    tone="emerald", pill_icon="tune")
        b1, b2, b3, _ = st.columns([1.3, 1.3, 1.1, 3])
        b1.button("Load final model", icon=":material/lock:", on_click=_apply_states, args=(_states_of(final_id),),
                  width="stretch")
        b2.button("Load best valid model", icon=":material/emoji_events:", on_click=_apply_states,
                  args=(_states_of(best_id),), width="stretch")
        b3.button("Null model", icon=":material/restart_alt:", on_click=_apply_states,
                  args=({c: 0 for c in STATE_COLS},), width="stretch")
        st.markdown('<div class="card-sub" style="margin-top:.6rem"><b>Level 1 · student predictors</b></div>',
                    unsafe_allow_html=True)
        cols = st.columns(2) * 2
        for colw, (sc, term, _, label) in zip(cols, D.L1_PREDICTORS):
            with colw:
                st.segmented_control(label, list(L1_OPTS), format_func=lambda v: L1_OPTS[v], key=f"lab_{sc}",
                                     required=True)
        st.markdown('<div class="card-sub"><b>Level 2 · school predictors</b></div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for colw, (sc, term, _, label) in zip(cols, D.L2_PREDICTORS):
            with colw:
                st.segmented_control(label, list(L2_OPTS), format_func=lambda v: L2_OPTS[v], key=f"lab_{sc}",
                                     required=True)

    states = {c: int(st.session_state.get(f"lab_{c}", 0) or 0) for c in STATE_COLS}
    m = D.find_model(states)
    if m is None:
        S.insight("This combination is not in the precomputed model bank.", title="Model not available", kind="warn")
        return
    mid = m["model_id"]
    is_valid = m["status"] == "VALID"
    warn = m["warnings"] if isinstance(m["warnings"], str) and m["warnings"].strip() else None
    conv = m["convergence_message"] if isinstance(m["convergence_message"], str) else None

    # ---------------- KPIs + status ----------------
    rank_txt = f"rank {int(m['AICc_rank_valid'])} of {int((cat['status'] == 'VALID').sum())}" if I.is_finite(m["AICc_rank_valid"]) else "not ranked"
    S.kpi_strip([
        dict(label="Model ID", value=mid, icon="fingerprint", tone="forest",
             sub="Final model" if mid == final_id else ("Best valid" if mid == best_id else f"{int(m['npar'])} parameters")),
        dict(label="AIC", value=I.num(m["AIC"], 1), icon="analytics", tone="teal",
             help="Akaike Information Criterion (ML fit). Lower is better."),
        dict(label="AICc", value=I.num(m["AICc"], 1), icon="analytics", tone="emerald",
             help="Small-sample corrected AIC. Lower is better; used for ranking."),
        dict(label="ΔAICc", value=I.num(m["delta_AICc_valid"], 2, na="Not ranked"), icon="leaderboard", tone="gold",
             sub=rank_txt, help="Difference from the best valid model's AICc. Singular models are not ranked."),
        dict(label="BIC", value=I.num(m["BIC"], 1), icon="analytics", tone="blue",
             help="Bayesian Information Criterion. Penalises extra parameters more strongly than AIC."),
        dict(label="Marginal R²", value=I.num(m["R2_marginal"], 3), icon="donut_large", tone="mint",
             help="Variance explained by fixed effects only (Nakagawa)."),
        dict(label="Conditional R²", value=I.num(m["R2_conditional"], 3), icon="donut_small", tone="mint",
             help="Variance explained by fixed + random effects. Not available for singular fits."),
        dict(label="ICC", value=I.num(m["ICC"], 3), icon="apartment", tone="slate",
             help="Adjusted intraclass correlation of this model (share of residual variance at school level). Not available for singular fits."),
        dict(label="Status", value=m["status"], icon="check_circle" if is_valid and not warn else "warning",
             tone="emerald" if is_valid and not warn else "terra"),
    ], cols=5)

    if is_valid and not warn:
        st.markdown(f'<div class="status-banner ok">{S.icon("verified")}<div><b>{mid} is a valid fit.</b> '
                    f'Non-singular, no convergence warning recorded. Formula: <span class="eq">{m["formula"]}</span></div></div>',
                    unsafe_allow_html=True)
    else:
        msgs = []
        if bool(m["singular"]):
            msgs.append("<b>Singular fit</b>: at least one random-effect variance or correlation sits at the boundary, so this "
                        "random structure is not supported by the data. ΔAICc, conditional R² and ICC are not reported.")
        if warn:
            msgs.append(f"<b>Convergence warning</b>: {S.esc(warn)}")
        if conv:
            msgs.append(f"<b>Convergence message</b>: {S.esc(conv)}")
        st.markdown(f'<div class="status-banner bad">{S.icon("error")}<div>{"<br/>".join(msgs)}<br/>'
                    f'Formula: <span class="eq">{m["formula"]}</span></div></div>', unsafe_allow_html=True)

    # ---------------- A. Leaderboard + B. Coefficients ----------------
    c1, c2 = st.columns([6, 6], gap="medium")
    with c1, S.card("card_lab_board"):
        S.card_head("Model Comparison", "Which Model Receives the Strongest Support?",
                    "Top 15 valid models by ΔAICc · gold = best valid, green = your selection", tone="gold",
                    pill_icon="emoji_events")
        valid = cat[cat["status"] == "VALID"].sort_values("delta_AICc_valid")
        top = valid.head(15)
        if is_valid and mid not in set(top["model_id"]):
            top = pd.concat([top, valid[valid["model_id"] == mid]])
        top = top.reset_index(drop=True)
        colors = [S.GOLD if r == best_id else (S.FOREST_2 if r == mid else S.MINT) for r in top["model_id"]]
        ylab = [f"{r}{' ★' if r == best_id else ''}{' ◀' if r == mid else ''}" for r in top["model_id"]]
        fig = go.Figure(go.Bar(
            x=top["delta_AICc_valid"], y=ylab, orientation="h",
            marker=dict(color=colors, line=dict(color="#FFFFFF", width=1)),
            customdata=np.stack([top["formula"], top["AICc"], top["R2_marginal"], top["R2_conditional"]], axis=1),
            text=[f"{v:.1f}" for v in top["delta_AICc_valid"]], textposition="outside", textfont=dict(size=10, color=S.INK_2),
            hovertemplate="<b>%{y}</b><br>ΔAICc = %{x:.2f}<br>AICc = %{customdata[1]:.1f}<br>R²m = %{customdata[2]:.3f} · "
                          "R²c = %{customdata[3]:.3f}<br>%{customdata[0]}<extra></extra>"))
        C.base_layout(fig, height=440)
        fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(size=10))
        fig.update_xaxes(title_text="ΔAICc (0 = best valid model)", range=[0, top["delta_AICc_valid"].max() * 1.18 + 1])
        chart(fig, "lab_board")
        sec = valid.iloc[1] if len(valid) > 1 else None
        txt = (f"<b>{best_id}</b> has the lowest AICc among {len(valid)} valid models"
               + (f"; the runner-up {sec['model_id']} is {sec['delta_AICc_valid']:.1f} AICc units behind, so the support for "
                  f"{best_id} is {'strong (ΔAICc > 10)' if sec['delta_AICc_valid'] > 10 else 'moderate'}" if sec is not None else "")
               + ". ")
        if mid != best_id:
            txt += (f"Your selection {mid} is {I.num(m['delta_AICc_valid'], 1)} units behind." if is_valid
                    else f"Your selection {mid} is singular and therefore excluded from the ranking.")
        S.insight(txt)

    with c2, S.card("card_lab_coef"):
        coefs = D.model_rows(D.S_COEFS, mid)
        coefs = coefs[coefs["predictor"] != "(Intercept)"].copy()
        S.card_head("Selected Model", "What Does This Model Estimate?",
                    f"Fixed-effect coefficients with 95% CI · {mid}", tone="emerald", pill_icon="ssid_chart")
        scale = st.segmented_control("Scale", ["Standardized β", "Raw estimate"], default="Standardized β",
                                     key="lab_scale", label_visibility="collapsed") or "Standardized β"
        if coefs.empty:
            empty = D.model_rows(D.S_COEFS, mid)
            icpt = empty["estimate"].iloc[0] if not empty.empty else np.nan
            S.insight(f"The null model has no predictors; its intercept (grand mean) is <b>{I.num(icpt)}</b>.",
                      title="No predictors selected", kind="green")
        else:
            coefs["label"] = coefs["predictor"].map(D.TERM_LABELS).fillna(coefs["predictor"])
            ratio = coefs["standardized_beta"] / coefs["estimate"]
            coefs["s_lo"] = np.minimum(coefs["ci_low"] * ratio, coefs["ci_high"] * ratio)
            coefs["s_hi"] = np.maximum(coefs["ci_low"] * ratio, coefs["ci_high"] * ratio)
            coefs["color"] = [S.EMERALD if s else S.NEUTRAL for s in coefs["significant_5pct"]]
            coefs["hover"] = [f"Raw {e:+.4f} [{lo:.4f}, {hi:.4f}]<br>{I.pval(p)} · df = {d_:.1f}"
                              for e, lo, hi, p, d_ in zip(coefs["estimate"], coefs["ci_low"], coefs["ci_high"],
                                                          coefs["p_value"], coefs["df"])]
            if scale == "Standardized β":
                coefs = coefs.reindex(coefs["standardized_beta"].abs().sort_values(ascending=False).index)
                fig = C.forest(coefs, "label", "standardized_beta", "s_lo", "s_hi", color_col="color",
                               x_title="Standardized β", hover_extra="hover", height=max(250, 58 * len(coefs) + 70))
            else:
                fig = C.forest(coefs, "label", "estimate", "ci_low", "ci_high", color_col="color",
                               x_title="Raw estimate (points per unit)", hover_extra="hover",
                               height=max(250, 58 * len(coefs) + 70))
            chart(fig, "lab_coef")
            nsig = int(coefs["significant_5pct"].sum())
            lead = coefs.loc[coefs["standardized_beta"].abs().idxmax()]
            S.insight(
                f"{nsig} of {len(coefs)} fixed effects have 95% CIs excluding zero (green; grey = CI includes zero). "
                f"The largest standardized association is <b>{lead['label']}</b> (β = {lead['standardized_beta']:.3f}). "
                "Bank models are ML fits for comparison; the locked REML + Kenward-Roger results are on the Research Questions page.",
                kind="green",
            )
        st.download_button("Download coefficients (CSV)", D.model_rows(D.S_COEFS, mid).to_csv(index=False).encode(),
                           file_name=f"{mid}_coefficients.csv", mime="text/csv", icon=":material/download:")

    # ---------------- C. Variance + D. Observed vs predicted ----------------
    c3, c4 = st.columns([5, 7], gap="medium")
    with c3, S.card("card_lab_var"):
        rnd = D.model_rows(D.S_RANDOM, mid)
        S.card_head("Random Effects", "How Is the Variance Structured?",
                    f"Random intercept, random slope and residual variances · {mid}", tone="blue", pill_icon="layers")
        var = rnd[rnd["term2"].isna()].copy()
        var["label"] = [("Residual (within)" if g == "Residual" else
                         ("School intercept" if t == "(Intercept)" else f"Slope: {D.TERM_LABELS.get(t, t)}"))
                        for g, t in zip(var["group"], var["term1"])]
        var["color"] = [S.EMERALD if g == "Residual" else (S.GOLD if t == "(Intercept)" else S.BLUE)
                        for g, t in zip(var["group"], var["term1"])]
        fig = go.Figure(go.Bar(
            x=var["variance_or_covariance"], y=var["label"], orientation="h",
            marker=dict(color=var["color"], line=dict(color="#FFFFFF", width=1)),
            text=[f"{v:.4f}" for v in var["variance_or_covariance"]], textposition="outside",
            textfont=dict(size=11, color=S.INK_2),
            customdata=var["stddev_or_correlation"],
            hovertemplate="%{y}<br>Variance: %{x:.4f}<br>SD: %{customdata:.4f}<extra></extra>"))
        C.base_layout(fig, height=max(200, 60 * len(var) + 60))
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_xaxes(title_text="Variance", range=[0, var["variance_or_covariance"].max() * 1.3])
        chart(fig, "lab_var")
        cov = rnd[rnd["term2"].notna()]
        if not cov.empty:
            cv = cov.assign(Pair=[f"{D.TERM_LABELS.get(a, a)} × {D.TERM_LABELS.get(b, b)}" for a, b in zip(cov["term1"], cov["term2"])])
            st.dataframe(cv[["Pair", "variance_or_covariance", "stddev_or_correlation"]].rename(
                columns={"variance_or_covariance": "Covariance", "stddev_or_correlation": "Correlation"}),
                hide_index=True, width="stretch")
            if (cov["stddev_or_correlation"].abs() > 0.95).any():
                S.insight("At least one random-effect correlation is above |0.95|, close to the boundary; interpret the "
                          "slope–intercept correlation cautiously.", title="Boundary check", kind="warn")
        icpt = var[var["label"] == "School intercept"]["variance_or_covariance"]
        resid = var[var["label"] == "Residual (within)"]["variance_or_covariance"]
        if len(icpt) and len(resid):
            S.insight(f"School intercept variance {icpt.iloc[0]:.3f} vs residual variance {resid.iloc[0]:.3f}. "
                      f"{'Slope variances are shown in blue.' if (var['color'] == S.BLUE).any() else 'No random slopes are active.'}",
                      kind="green")

    with c4, S.card("card_lab_pred"):
        pr = D.predictions(mid)
        S.card_head("Model Fit", "How Close Are Predictions to Observed Scores?",
                    f"Observed vs fitted mathematics score (stored predictions, n = {len(pr)}) · {mid}",
                    tone="mint", pill_icon="scatter_plot")
        if pr.empty:
            S.insight("Predictions for this model are not available in the workbook.", title="Not available", kind="warn")
        else:
            lo = min(pr["observed_math"].min(), pr["fitted_math"].min()) - 1
            hi = max(pr["observed_math"].max(), pr["fitted_math"].max()) + 1
            fig = go.Figure()
            fig.add_trace(go.Scattergl(
                x=pr["fitted_math"], y=pr["observed_math"], mode="markers",
                marker=dict(size=5, color="rgba(15,138,95,0.45)"),
                customdata=np.stack([pr["student_id"], pr["school_id"], pr["residual"]], axis=1),
                hovertemplate="%{customdata[0]} · School %{customdata[1]}<br>Fitted: %{x:.2f}<br>Observed: %{y:.2f}"
                              "<br>Residual: %{customdata[2]:.2f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=S.GOLD, width=2, dash="dash"),
                                     hoverinfo="skip"))
            C.base_layout(fig, height=380)
            fig.update_xaxes(title_text="Fitted (predicted) mathematics score", range=[lo, hi])
            fig.update_yaxes(title_text="Observed mathematics score", range=[lo, hi])
            chart(fig, "lab_pred")
            rmse = float(np.sqrt(np.mean(pr["residual"] ** 2)))
            r = I.safe_corr(pr["fitted_math"], pr["observed_math"])
            S.insight(f"Root mean squared residual is <b>{rmse:.3f}</b> points and the observed–fitted correlation is "
                      f"{r:.3f}. Points close to the gold diagonal are predicted well; fitted values include the school "
                      "random effects, so they describe in-sample fit rather than out-of-sample accuracy.")

    # ---------------- Equations + diagnostics ----------------
    c5, c6 = st.columns([6, 6], gap="medium")
    with c5, S.card("card_lab_eq"):
        S.card_head("Equations", "Model Written Out", "Generated from the selected fixed and random specification",
                    tone="slate", pill_icon="function")
        eq1, eq2, eq3 = _equations(states)
        with st.expander("Level 1 equation (students)", expanded=True):
            st.latex(eq1)
        with st.expander("Level 2 equations (schools)"):
            st.latex(eq2)
        with st.expander("Combined equation"):
            st.latex(eq3)
        st.caption("Predictors are grand-mean centred (ses_c, prior_c, hours_c, facility_c, teacher_c); Female is a 0/1 indicator.")

    with c6, S.card("card_lab_diag"):
        S.card_head("Diagnostics", "Does This Model Meet Its Assumptions?",
                    f"Stored diagnostics for {mid}; missing checks are shown as Not available", tone="terra",
                    pill_icon="health_and_safety")
        dg = D.model_rows(D.S_DIAG, mid)
        rd = D.model_rows(D.S_RDIAG, mid)
        st.markdown(_diag_table(dg.iloc[0] if not dg.empty else None, rd, m), unsafe_allow_html=True)


DIAG_SPEC = [
    ("Level-1 residual normality", "residual_normality_p", "residual_normality_ok", "p"),
    ("Level-1 homoskedasticity (Breusch-Pagan)", "homoskedasticity_L1_bp_p", "homoskedasticity_L1_ok", "p"),
    ("Between-school variance homogeneity (Levene)", "homogeneity_L2_levene_p", "homogeneity_L2_ok", "p"),
    ("Linearity (quadratic terms)", "linearity_quadratic_p", "linearity_ok", "p"),
    ("Multicollinearity (max VIF)", "max_vif", "multicollinearity_ok", "vif"),
    ("Random-intercept normality", "random_intercept_normality_p", "random_intercept_normality_ok", "p"),
    ("Random-slope normality (min p)", "random_slope_normality_min_p", "random_slope_normality_ok", "p"),
]


def _diag_table(dg, rd: pd.DataFrame, m) -> str:
    rows = []
    for name, vcol, okcol, kind in DIAG_SPEC:
        v = dg[vcol] if dg is not None and vcol in dg else np.nan
        ok = I.ok_flag(dg[okcol]) if dg is not None and okcol in dg else None
        if I.isna(v):
            val, b = "Not available", S.badge("Not available", "na")
        else:
            val = f"VIF = {v:.2f}" if kind == "vif" else I.pval(v)
            b = S.badge("Pass", "ok") if ok else (S.badge("Flag", "bad") if ok is False else S.badge("Not available", "na"))
        rows.append(f"<tr><td>{name}</td><td class='num'>{val}</td><td>{b}</td></tr>")
    for _, r in rd.iterrows():
        if r["random_term"] == "(Intercept)":
            continue
        okr = "Normal" in str(r["conclusion"]) and "Depart" not in str(r["conclusion"])
        rows.append(f"<tr><td>Slope normality: {D.TERM_LABELS.get(r['random_term'], r['random_term'])}</td>"
                    f"<td class='num'>{I.pval(r['shapiro_p'])}</td><td>{S.badge('Pass' if okr else 'Flag', 'ok' if okr else 'bad')}</td></tr>")
    sing = bool(m["singular"])
    rows.append(f"<tr><td>Singularity</td><td class='num'>{'Singular' if sing else 'Not singular'}</td>"
                f"<td>{S.badge('Flag' if sing else 'Pass', 'bad' if sing else 'ok')}</td></tr>")
    w = m["warnings"] if isinstance(m["warnings"], str) and m["warnings"].strip() else None
    rows.append(f"<tr><td>Convergence</td><td>{S.esc(w) if w else 'No warning recorded'}</td>"
                f"<td>{S.badge('Flag' if w else 'Pass', 'bad' if w else 'ok')}</td></tr>")
    return ("<table class='diag-table'><thead><tr><th>Check</th><th>Statistic</th><th>Decision</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")
