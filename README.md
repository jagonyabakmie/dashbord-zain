# Mathematics Achievement Analytics

A six-page Streamlit dashboard for the RASIO 10.0 Statistics Olympiad final round: 900 students nested in 30 schools, analysed with two-level (multilevel) models.

Every number comes from `MASTER_STREAMLIT_MULTILEVEL.xlsx`. No model is re-fitted inside Streamlit.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Keep `app.py`, `utils/`, `views/`, `assets/`, `.streamlit/` and `MASTER_STREAMLIT_MULTILEVEL.xlsx` in the same folder.

The one-screen layout needs Streamlit 1.50 or newer (charts use `height="stretch"`). If an older version is installed, run `pip install -U streamlit`.

## Layout

Each page fits one browser screen on laptops and desktops, with no page scroll: a slim header, a KPI strip and a grid of equal-height cards whose charts stretch to fill them. Every interpretation box uses one flat green tint and sits at the bottom of its card, so boxes in the same row line up. Long interpretations are shortened to a few lines; hover over a box to read the full text. Secondary tables (workbook tables, the all-schools table, robustness checks) open from a button in the header.

Below 1100px wide or 600px tall the cards stack vertically and the page scrolls normally, so the dashboard stays usable on tablets and phones. On screens up to 720px tall the sidebar notes are hidden to keep the sidebar from scrolling.

## Project structure

```
app.py                     page config, sidebar navigation, filters, routing
.streamlit/config.toml     light theme, hides default page navigation
requirements.txt
utils/
  data.py                  workbook access: sheet constants, cached per-sheet loading, model lookup
  styles.py                palette, global CSS, decorative-asset processing, HTML components (hero, KPI, cards, insights)
  charts.py                transparent Plotly theme, forest plot, Q-Q plot, density helper
  interpretations.py       number / p-value formatting and wording helpers
views/
  common.py                exploratory filtering, chart wrapper
  overview.py              1. Overview & Descriptive Profile
  exploration.py           2. Exploratory Relationship Analysis
  research_questions.py    3. Research Questions
  model_lab.py             4. Multilevel Model Laboratory
  school_explorer.py       5. School Performance Explorer
  diagnostics.py           6. Diagnostics & Robustness
```

## Filtering logic

| Page | Sidebar behaviour |
|---|---|
| Overview, Exploration | School / Gender / Student ID filters truly subset the 900 student rows. KPIs, charts and insight text update. School-level views (school means, Facility, Teacher Experience) stay at one row per school; a selected school is highlighted. |
| Research Questions | No filters. Always the locked final model (M293, REML + Kenward-Roger) on the full sample. |
| Model Laboratory | Predictor states (Exclude / Fixed / Fixed + Random slope) map to one of the 324 precomputed models in `14_model_catalog`. |
| School Explorer | School selector on the page; values come from the locked final model (`13_final_school`). |
| Diagnostics | Locked final model; a selected school is only highlighted. |

## Workbook sheets used

`00_README`, `01_students`, `02_schools`, `03_school_profiles_z`, `04_raw_hours_slopes`, `05_corr_student`, `06_corr_school`, `07_RQ_summary`, `08_variance_path`, `09_final_fixed_KR`, `10_final_random`, `11_final_diag`, `12_final_random_diag`, `13_final_school`, `14_model_catalog`, `15_model_coefficients`, `16_model_random`, `17_model_diag`, `18_random_diag`, `20_top_models`, `24_model_predictions`.

Not used: `19_school_effects` (per-model BLUPs; the locked model's school effects come from `13_final_school`), `21_dashboard_map`, `22_data_dictionary`, `23_relationships` (documentation sheets that guided the build).

## Notes

- **Final model.** `00_README` names `M293` as both the best valid model and the final report model; `09_final_fixed_KR` marks it `LOCKED FINAL REPORT`.
- **Speed.** Sheets load one at a time and are cached with `st.cache_data`. The four largest sheets are also mirrored to `.cache/*.parquet` on first use, keyed on the workbook's size and modification time, so later restarts are fast. Delete `.cache/` at any time; it is rebuilt automatically.
- **Plots for the final model's residuals** use `24_model_predictions` (the ML model-bank fit of the same specification), because the workbook stores student-level residuals only there. The diagnostic tests use the REML results in `11_final_diag`.
- **Standardized confidence intervals** are the workbook's raw CIs multiplied by `standardized_beta / estimate`.
- **Response curve** (School Explorer) uses only the locked fixed effects and the school's stored random effects, with SES and prior score at the sample mean and gender at the sample proportion female.
- **Decorative assets.** Leaves, books, pencil, sparkles, waves, header and background art are used for decoration. Their caption strip is cropped off and the white backdrop is made transparent at runtime. KPI, card and chart PNGs that contain baked-in numbers are never displayed; all KPIs and charts are built from data.
- **Robustness.** The workbook has no leave-one-school-out results. The Robustness section instead compares coefficients across the top-10 valid models and ML vs REML estimates for M293.
