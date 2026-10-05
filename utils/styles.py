"""Visual system: palette, global CSS, decorative assets and small HTML components.

Only decorative PNGs (background, header art, sidebar art, leaves, books, sparkles…)
are used. KPI / chart PNGs with baked-in numbers are never rendered: every KPI,
badge, title and insight below is generated from workbook data at runtime.
"""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import numpy as np
import streamlit as st

ASSET_DIR = Path(__file__).resolve().parent.parent / "assets"

# ----------------------------------------------------------------------------
# Palette
# ----------------------------------------------------------------------------
FOREST = "#0B3D2C"
FOREST_2 = "#12583E"
EMERALD = "#0F8A5F"
EMERALD_SOFT = "#5FBF95"
MINT = "#A7E3C9"
MINT_LIGHT = "#E6F6EE"
BLUE = "#3E78B2"
BLUE_SOFT = "#9DBEDF"
GOLD = "#C9962B"
GOLD_SOFT = "#F3DFA8"
TERRA = "#B9573E"
INK = "#17261F"
INK_2 = "#3C4E45"
MUTED = "#6B7B73"
GRID = "rgba(11,61,44,0.08)"
NEUTRAL = "#C9D3CE"

FONT = "'Plus Jakarta Sans', 'Segoe UI', system-ui, -apple-system, sans-serif"


# ----------------------------------------------------------------------------
# Assets
# ----------------------------------------------------------------------------
@st.cache_data
def asset_b64(name: str) -> str:
    """Base64 of an asset PNG, untouched."""
    p = ASSET_DIR / name
    if not p.exists():
        return ""
    return base64.b64encode(p.read_bytes()).decode()


@st.cache_data
def deco_b64(name: str, top: float = 0.10, bottom: float = 0.74) -> str:
    """Decorative element cut out of the asset sheet crop.

    The supplied decorative PNGs carry a near-white backdrop and a caption under
    the drawing; this crops the caption off and keys the white backdrop to
    transparency so the element floats over cards and backgrounds.
    """
    from PIL import Image

    p = ASSET_DIR / name
    if not p.exists():
        return ""
    im = np.asarray(Image.open(p).convert("RGB")).astype(float)
    h = im.shape[0]
    im = im[int(h * top): int(h * bottom)]
    mn = im.min(axis=2)
    alpha = np.clip((255 - mn) * 3.2, 0, 255)
    alpha[(255 - mn) < 14] = 0
    rgba = np.dstack([im, alpha]).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def img_uri(b64: str) -> str:
    return f"data:image/png;base64,{b64}" if b64 else ""


# ----------------------------------------------------------------------------
# Global CSS
# ----------------------------------------------------------------------------
def inject_css() -> None:
    bg = img_uri(asset_b64("01_background_main.png"))
    header_art = img_uri(asset_b64("02_header_banner.png"))
    side_leaf = img_uri(deco_b64("26c_leaf_bottom_left.png"))
    leaf_br = img_uri(deco_b64("26d_leaf_bottom_right.png"))
    leaf_tl = img_uri(deco_b64("26a_leaf_top_left.png"))

    css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,400..600,0..1,0&display=swap');
:root {{
  --forest: {FOREST}; --forest-2: {FOREST_2}; --emerald: {EMERALD}; --mint: {MINT};
  --mint-light: {MINT_LIGHT}; --blue: {BLUE}; --gold: {GOLD}; --terra: {TERRA};
  --ink: {INK}; --ink-2: {INK_2}; --muted: {MUTED};
  --card-bg: rgba(255,255,255,0.80);
  --card-border: rgba(15,138,95,0.18);
  --shadow: 0 10px 30px -18px rgba(11,61,44,0.45), 0 2px 6px -2px rgba(11,61,44,0.08);
}}
html, body, [class*="css"], .stApp, .stMarkdown, button, input, select, textarea {{
  font-family: {FONT};
}}
.stApp {{
  background-color: #F3F8F5;
  background-image: linear-gradient(180deg, rgba(246,250,248,0.90) 0%, rgba(246,250,248,0.93) 45%, rgba(246,250,248,0.96) 100%), url("{bg}");
  background-size: cover; background-attachment: fixed; background-position: center top;
  color: var(--ink);
}}
/* ---------- Streamlit chrome ---------- */
#MainMenu, footer, [data-testid="stStatusWidget"], [data-testid="stDecoration"], .stDeployButton,
[data-testid="stAppDeployButton"], [data-testid="stToolbarActions"] {{ display: none !important; visibility: hidden; }}
[data-testid="stHeader"] {{ background: transparent; height: 2.5rem; }}
.block-container {{ padding-top: 1.4rem !important; padding-bottom: 3rem !important; max-width: 1480px; }}
h1, h2, h3, h4 {{ font-family: {FONT}; color: var(--ink); letter-spacing: -0.01em; }}
a {{ color: var(--emerald); }}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, #0A3627 0%, #0D4631 55%, #0B3D2C 100%) !important;
  border-right: 0;
}}
section[data-testid="stSidebar"]::after {{ content:""; position:absolute; left:-8px; bottom:-6px; width:150px; height:150px; pointer-events:none;
  background: url("{side_leaf}") no-repeat left bottom / contain; opacity:.35; }}
section[data-testid="stSidebar"] * {{ color: #E9F5EF; }}
section[data-testid="stSidebar"] .stMarkdown p {{ color: #CFE5DA; }}
section[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.14); margin: 0.8rem 0; }}
.brand {{ display:flex; align-items:center; gap:.75rem; padding: .2rem .1rem .3rem .1rem; }}
.brand-mark {{ width:52px; height:52px; border-radius: 16px; background: rgba(255,255,255,0.08);
  display:flex; align-items:center; justify-content:center; border:1px solid rgba(255,255,255,0.15); }}
.brand-mark .material-symbols-rounded {{ font-size: 30px; color: #FFFFFF; }}
.brand-title {{ font-weight: 800; font-size: 1.02rem; line-height:1.15; color:#FFFFFF; letter-spacing:.02em; text-transform: uppercase; }}
.brand-sub {{ font-weight: 500; font-size: .86rem; color:#BFE3D1; }}
.side-label {{ font-size:.72rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; color:#8FC9AE !important; margin:.4rem 0 .35rem .1rem; display:flex; align-items:center; gap:.35rem; }}
.side-footer {{ margin-top: 1.0rem; padding: .85rem .9rem; border-radius: 14px; background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.12); font-size: .80rem; line-height: 1.55; color:#CFE5DA; }}
.side-footer b {{ color:#FFFFFF; }}
.side-note {{ font-size:.78rem; line-height:1.45; padding:.6rem .7rem; border-radius:12px; background: rgba(201,150,43,0.16);
  border:1px solid rgba(201,150,43,0.35); color:#F6E7C0 !important; }}

/* navigation radio -> menu */
section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap: 4px; }}
section[data-testid="stSidebar"] div[role="radiogroup"] > label {{
  width: 100%; margin: 0; padding: .62rem .75rem; border-radius: 12px; cursor: pointer;
  transition: background .15s ease; border: 1px solid transparent;
}}
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {{ display: none; }}
section[data-testid="stSidebar"] div[role="radiogroup"] > label p {{ font-size: .92rem; font-weight: 500; color: #D7EBE1; }}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {{ background: rgba(255,255,255,0.07); }}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {{
  background: linear-gradient(90deg, #14A06E 0%, #0F8A5F 100%);
  box-shadow: 0 8px 20px -10px rgba(20,160,110,0.9); border-color: rgba(255,255,255,0.18);
}}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p {{ color: #FFFFFF; font-weight: 700; }}
/* Streamlit >= 1.5x react-aria radio */
section[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div {{ width: 100%; }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"] {{
  width: 100%; margin: 0 0 3px 0; padding: .62rem .75rem; border-radius: 12px; cursor: pointer;
  transition: background .15s ease; border: 1px solid transparent; box-sizing: border-box;
}}
section[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child:not([data-testid]) {{ display: none; }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"] p {{ font-size: .92rem; font-weight: 500; color: #D7EBE1; }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"] p span[role="img"] {{ margin-right: .45rem; font-size: 1.15rem; }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover {{ background: rgba(255,255,255,0.07); }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {{
  background: linear-gradient(90deg, #14A06E 0%, #0F8A5F 100%);
  box-shadow: 0 8px 20px -10px rgba(20,160,110,0.9); border-color: rgba(255,255,255,0.18);
}}
section[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p {{ color: #FFFFFF; font-weight: 700; }}
/* sidebar selectboxes (new + legacy DOM) */
section[data-testid="stSidebar"] [data-testid="stSelectbox"] [role="group"] {{
  background: rgba(255,255,255,0.08) !important; border: 1px solid rgba(255,255,255,0.22) !important; border-radius: 10px;
}}
section[data-testid="stSidebar"] [data-testid="stSelectbox"] input {{ color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF; background: transparent !important; }}
section[data-testid="stSidebar"] [data-testid="stSelectbox"] button svg {{ color: #CFE5DA; fill: #CFE5DA; }}
section[data-testid="stSidebar"] .stButton button {{ background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.25); border-radius: 10px; }}
section[data-testid="stSidebar"] .stButton button:hover {{ border-color: #5FBF95; background: rgba(95,191,149,0.15); }}
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{ color:#CFE5DA; font-weight:600; font-size:.84rem; }}
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
  background: rgba(255,255,255,0.08); border-color: rgba(255,255,255,0.22); border-radius: 10px;
}}
section[data-testid="stSidebar"] div[data-baseweb="select"] svg {{ fill: #CFE5DA; }}

/* ---------- Page hero ---------- */
.hero {{
  position: relative; overflow: hidden; border-radius: 22px; padding: 1.35rem 1.6rem 1.25rem 1.6rem;
  background: linear-gradient(115deg, rgba(10,70,48,0.96) 0%, rgba(15,138,95,0.92) 55%, rgba(95,191,149,0.80) 100%), url("{header_art}");
  background-size: cover; background-position: center; box-shadow: var(--shadow); margin-bottom: 1.0rem;
}}
.hero::after {{ content:""; position:absolute; inset:0; background: url("{header_art}") right center / auto 100% no-repeat; opacity:.55; mix-blend-mode: screen; pointer-events:none; }}
.hero-top {{ display:flex; align-items:center; gap: .7rem; position:relative; z-index:2; }}
.hero-kicker {{ color:#CFF0E0; font-size:.78rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; }}
.hero h1 {{ color:#FFFFFF !important; font-size: 1.95rem; font-weight: 800; margin: .1rem 0 .15rem 0; padding:0; position:relative; z-index:2; line-height:1.2; }}
.hero p {{ color:#E3F6EC; font-size: 1.0rem; margin:0; position:relative; z-index:2; max-width: 760px; }}
.hero-chips {{ display:flex; gap:.45rem; flex-wrap:wrap; margin-top:.8rem; position:relative; z-index:2; }}
.hero-chip {{ background: rgba(255,255,255,0.16); border:1px solid rgba(255,255,255,0.28); color:#FFFFFF; font-size:.76rem; font-weight:600;
  padding:.28rem .7rem; border-radius:999px; backdrop-filter: blur(4px); }}
.hero-chip.gold {{ background: rgba(201,150,43,0.35); border-color: rgba(243,223,168,0.6); }}
.hero-deco {{ position:absolute; right: 1.2rem; top: 50%; transform: translateY(-50%); height: 92px; z-index:2; opacity:.95; }}

/* ---------- KPI strip ---------- */
.kpi-grid {{ display:grid; gap: .75rem; margin: .2rem 0 1rem 0; grid-template-columns: repeat(var(--n), minmax(0, 1fr)); }}
.kpi-grid.dense {{ gap: .55rem; }}
.kpi-grid.dense .kpi {{ padding: .7rem .65rem; gap: .5rem; flex-direction: column; align-items: flex-start; }}
.kpi-grid.dense .kpi-icon {{ flex: 0 0 32px; width: 32px; height: 32px; border-radius: 10px; }}
.kpi-grid.dense .kpi-icon .material-symbols-rounded {{ font-size: 19px; }}
.kpi-grid.dense .kpi-value {{ font-size: 1.2rem; }}
.kpi-grid.dense .kpi-body {{ width: 100%; }}
@media (max-width: 1250px) {{ .kpi-grid {{ grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)) !important; }} }}
[data-testid="stColumn"] .kpi-grid {{ grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); }}
.kpi {{
  position:relative; background: var(--card-bg); border:1px solid var(--card-border); border-radius:18px;
  padding:.85rem .95rem; display:flex; align-items:center; gap:.75rem; box-shadow: var(--shadow);
  backdrop-filter: blur(8px); min-width:0;
}}
.kpi-icon {{ flex: 0 0 44px; height:44px; border-radius:13px; display:flex; align-items:center; justify-content:center; }}
.kpi-icon .material-symbols-rounded {{ font-size: 25px; color:#FFFFFF; }}
.kpi-body {{ min-width:0; }}
.kpi-value {{ font-size: 1.38rem; font-weight: 800; color: var(--ink); line-height:1.1; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.kpi-label {{ font-size: .76rem; color: var(--muted); font-weight:600; margin-top:.15rem; line-height:1.25; }}
.kpi-sub {{ font-size: .70rem; color: var(--muted); margin-top:.1rem; line-height:1.25; }}
.kpi .help {{ position:absolute; top:.45rem; right:.55rem; font-size: 15px; color: #9AB0A5; cursor: help; }}
.tone-forest {{ background: linear-gradient(135deg, #0F5A3F, #0B3D2C); }}
.tone-emerald {{ background: linear-gradient(135deg, #22B07C, #0F8A5F); }}
.tone-teal {{ background: linear-gradient(135deg, #1C9AA0, #10717A); }}
.tone-blue {{ background: linear-gradient(135deg, #5C93CA, #3E78B2); }}
.tone-gold {{ background: linear-gradient(135deg, #E0B44E, #C9962B); }}
.tone-terra {{ background: linear-gradient(135deg, #D07A61, #B9573E); }}
.tone-mint {{ background: linear-gradient(135deg, #7FD1AE, #3FAE80); }}
.tone-slate {{ background: linear-gradient(135deg, #8A9A93, #5E6E66); }}

/* ---------- Cards (keyed st.container) ---------- */
div[class*="st-key-card"] {{
  position: relative; background: var(--card-bg) !important; border: 1px solid var(--card-border) !important;
  border-radius: 20px !important; box-shadow: var(--shadow); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
  padding: 1.05rem 1.15rem 1.0rem 1.15rem !important; overflow: hidden;
}}
div[class*="st-key-card"]::after {{
  content: ""; position:absolute; right:-6px; bottom:-8px; width: 74px; height: 74px; pointer-events:none;
  background: url("{leaf_br}") no-repeat right bottom / contain; opacity: .55;
}}
div[class*="st-key-cardnarr"] {{ background: linear-gradient(160deg, rgba(255,255,255,0.92), rgba(230,246,238,0.88)) !important; }}
div[class*="st-key-cardnarr"]::before {{
  content: ""; position:absolute; left:-8px; top:-10px; width: 70px; height: 70px; pointer-events:none;
  background: url("{leaf_tl}") no-repeat left top / contain; opacity: .55;
}}
.card-head {{ position:relative; z-index:2; margin-bottom:.25rem; }}
.pill {{ display:inline-flex; align-items:center; gap:.3rem; font-size:.68rem; font-weight:800; letter-spacing:.09em; text-transform:uppercase;
  padding:.24rem .65rem; border-radius:999px; color:#FFFFFF; background: var(--forest-2); }}
.pill .material-symbols-rounded {{ font-size: 15px; }}
.pill.emerald {{ background: #0F8A5F; }} .pill.blue {{ background: #3E78B2; }} .pill.gold {{ background: #C9962B; }}
.pill.mint {{ background: #3FAE80; }} .pill.slate {{ background: #6F8C9D; }} .pill.terra {{ background: #B9573E; }}
.pill.ghost {{ background: rgba(15,138,95,0.10); color: var(--forest-2); }}
.card-title {{ font-size: 1.12rem; font-weight: 800; color: var(--ink); margin: .45rem 0 .1rem 0; line-height:1.25; }}
.card-sub {{ font-size: .84rem; color: var(--muted); margin: 0 0 .2rem 0; line-height:1.4; }}
.narr-title {{ padding-right: 92px; font-size: 1.55rem; font-weight: 800; color: var(--forest); line-height:1.2; margin:.55rem 0 .4rem 0; }}
.narr-lead {{ font-size: 1.0rem; color: var(--ink-2); font-weight:600; line-height:1.45; margin-bottom:.5rem; }}
.narr-body {{ font-size: .92rem; color: var(--ink-2); line-height:1.6; }}
.narr-books {{ position:absolute; right: .2rem; top: 2.3rem; height: 78px; opacity:.95; }}

/* ---------- Insight ---------- */
.insight {{ position:relative; z-index:2; display:flex; gap:.7rem; padding:.75rem .9rem; border-radius:14px; margin-top:.35rem;
  background: linear-gradient(135deg, rgba(255,248,226,0.95), rgba(255,251,240,0.92)); border:1px solid rgba(201,150,43,0.30);
  border-left: 4px solid {GOLD}; }}
.insight.green {{ background: linear-gradient(135deg, rgba(230,246,238,0.96), rgba(245,251,248,0.92)); border-color: rgba(15,138,95,0.22); border-left-color:{EMERALD}; }}
.insight.warn {{ background: linear-gradient(135deg, rgba(250,233,227,0.96), rgba(253,245,242,0.92)); border-color: rgba(185,87,62,0.28); border-left-color:{TERRA}; }}
.insight .material-symbols-rounded {{ font-size: 22px; color: {GOLD}; flex: 0 0 auto; margin-top:1px; }}
.insight.green .material-symbols-rounded {{ color: {EMERALD}; }}
.insight.warn .material-symbols-rounded {{ color: {TERRA}; }}
.insight-title {{ font-weight:800; font-size:.86rem; color: var(--ink); margin-bottom:.15rem; }}
.insight-body {{ font-size:.85rem; color: var(--ink-2); line-height:1.55; }}
.insight-body b {{ color: var(--ink); }}

/* ---------- Status badge / tables ---------- */
.badge {{ display:inline-flex; align-items:center; gap:.3rem; padding:.22rem .6rem; border-radius:999px; font-size:.74rem; font-weight:700; }}
.badge.ok {{ background: rgba(15,138,95,0.12); color:#0B6B48; }}
.badge.bad {{ background: rgba(185,87,62,0.14); color:#9A3F28; }}
.badge.warn {{ background: rgba(201,150,43,0.18); color:#8A6413; }}
.badge.na {{ background: rgba(107,123,115,0.14); color:#4F5E57; }}
.badge .material-symbols-rounded {{ font-size: 15px; }}
.diag-table {{ width:100%; border-collapse: separate; border-spacing: 0; font-size:.85rem; }}
.diag-table th {{ text-align:left; font-size:.72rem; letter-spacing:.08em; text-transform:uppercase; color: var(--muted); font-weight:700;
  padding:.5rem .6rem; border-bottom: 1px solid rgba(11,61,44,0.12); }}
.diag-table td {{ padding:.55rem .6rem; border-bottom: 1px solid rgba(11,61,44,0.07); color: var(--ink-2); vertical-align: top; }}
.diag-table tr:last-child td {{ border-bottom: 0; }}
.diag-table td.num {{ font-variant-numeric: tabular-nums; font-weight:600; color: var(--ink); white-space:nowrap; }}
.eq {{ font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace; font-size:.84rem; }}
.section-label {{ display:flex; align-items:center; gap:.55rem; margin: .4rem 0 .6rem .1rem; }}
.section-label .bar {{ width: 5px; height: 22px; border-radius: 3px; background: linear-gradient(180deg, {EMERALD}, {MINT}); }}
.section-label span.t {{ font-weight:800; font-size: 1.05rem; color: var(--forest); }}
.section-label span.s {{ font-size:.82rem; color: var(--muted); }}
.status-banner {{ display:flex; align-items:center; gap:.6rem; padding:.7rem .9rem; border-radius:14px; margin-bottom: .8rem; font-size:.88rem; }}
.status-banner.ok {{ background: rgba(15,138,95,0.10); border:1px solid rgba(15,138,95,0.25); color:#0B5A3D; }}
.status-banner.bad {{ background: rgba(185,87,62,0.10); border:1px solid rgba(185,87,62,0.30); color:#8E3A24; }}
.status-banner .material-symbols-rounded {{ font-size: 22px; }}

/* ---------- Widgets in main area ---------- */
[data-testid="stMain"] div[data-baseweb="select"] > div {{ border-radius: 10px; border-color: rgba(15,138,95,0.25); background: rgba(255,255,255,0.9); }}
div[data-testid="stSegmentedControl"] button[aria-checked="true"], div[data-testid="stSegmentedControl"] button[kind="segmented_controlActive"] {{
  background: {EMERALD} !important; color: #FFFFFF !important; border-color: {EMERALD} !important; }}
div[data-testid="stExpander"] details {{ border-radius: 14px; border: 1px solid var(--card-border); background: rgba(255,255,255,0.75); }}
div[data-testid="stExpander"] summary p {{ font-weight: 700; color: var(--forest); }}
.stDownloadButton button {{ border-radius: 10px; border: 1px solid rgba(15,138,95,0.35); color: var(--forest-2); background: rgba(255,255,255,0.85); }}
.stDownloadButton button:hover {{ border-color: {EMERALD}; color: {EMERALD}; }}
[data-testid="stPlotlyChart"] {{ position: relative; z-index: 2; }}
.material-symbols-rounded {{ font-family: 'Material Symbols Rounded'; font-weight: normal; font-style: normal; line-height: 1;
  letter-spacing: normal; text-transform: none; display: inline-block; white-space: nowrap; direction: ltr; -webkit-font-smoothing: antialiased; }}
@media (max-width: 900px) {{
  .hero h1 {{ font-size: 1.5rem; }} .hero-deco {{ display:none; }} .narr-books {{ display:none; }}
}}
</style>
"""
    st.markdown(css, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# HTML components
# ----------------------------------------------------------------------------
def esc(s) -> str:
    return html.escape(str(s))


def icon(name: str) -> str:
    return f'<span class="material-symbols-rounded">{name}</span>'


def page_header(title: str, subtitle: str, kicker: str, chips: list[str] | None = None,
                deco: str = "26h_books.png", gold_chip: str | None = None) -> None:
    deco_uri = img_uri(deco_b64(deco))
    chip_html = "".join(f'<span class="hero-chip">{esc(c)}</span>' for c in (chips or []))
    if gold_chip:
        chip_html += f'<span class="hero-chip gold">{esc(gold_chip)}</span>'
    st.markdown(
        f"""<div class="hero">
  <div class="hero-top"><span class="hero-kicker">{esc(kicker)}</span></div>
  <h1>{esc(title)}</h1>
  <p>{esc(subtitle)}</p>
  <div class="hero-chips">{chip_html}</div>
  {'<img class="hero-deco" src="' + deco_uri + '"/>' if deco_uri else ''}
</div>""",
        unsafe_allow_html=True,
    )


def kpi_strip(items: list[dict], cols: int | None = None) -> None:
    """items: dict(label, value, icon, tone, sub=None, help=None)."""
    n = cols or len(items)
    cards = []
    for it in items:
        help_html = (f'<span class="material-symbols-rounded help" title="{esc(it["help"])}">info</span>'
                     if it.get("help") else "")
        sub = f'<div class="kpi-sub">{esc(it["sub"])}</div>' if it.get("sub") else ""
        cards.append(
            f'<div class="kpi">{help_html}<div class="kpi-icon tone-{it.get("tone", "emerald")}">{icon(it.get("icon", "insights"))}</div>'
            f'<div class="kpi-body"><div class="kpi-value" title="{esc(it["value"])}">{esc(it["value"])}</div>'
            f'<div class="kpi-label">{esc(it["label"])}</div>{sub}</div></div>'
        )
    st.markdown(
        f'<div class="kpi-grid{" dense" if n >= 7 else ""}" style="--n:{n};">'
        + "".join(cards) + "</div>",
        unsafe_allow_html=True,
    )


def card_head(pill: str, title: str, subtitle: str = "", tone: str = "emerald", pill_icon: str | None = None) -> None:
    ic = icon(pill_icon) if pill_icon else ""
    sub = f'<div class="card-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="card-head"><span class="pill {tone}">{ic}{esc(pill)}</span>'
        f'<div class="card-title">{esc(title)}</div>{sub}</div>',
        unsafe_allow_html=True,
    )


def insight(body_html: str, title: str = "Key Insight", kind: str = "gold") -> None:
    ic = {"gold": "lightbulb", "green": "insights", "warn": "warning"}.get(kind, "lightbulb")
    cls = "" if kind == "gold" else kind
    st.markdown(
        f'<div class="insight {cls}">{icon(ic)}<div><div class="insight-title">{esc(title)}</div>'
        f'<div class="insight-body">{body_html}</div></div></div>',
        unsafe_allow_html=True,
    )


def section_label(title: str, sub: str = "") -> None:
    st.markdown(
        f'<div class="section-label"><div class="bar"></div><span class="t">{esc(title)}</span>'
        f'<span class="s">{esc(sub)}</span></div>',
        unsafe_allow_html=True,
    )


def badge(text: str, kind: str = "ok") -> str:
    ic = {"ok": "check_circle", "bad": "error", "warn": "warning", "na": "help"}.get(kind, "info")
    return f'<span class="badge {kind}">{icon(ic)}{esc(text)}</span>'


def card(key: str):
    """Glass card container. Keys starting with 'cardnarr' get the narrative styling."""
    return st.container(key=key)
