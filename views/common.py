"""Shared page helpers: exploratory filtering and selected-school resolution."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from utils import data as D
from utils import styles as S


def selected_school_id(ctx) -> int | None:
    lab = ctx.get("school")
    if not lab or lab == ctx["ALL_SCHOOLS"]:
        return None
    return int(lab.split()[-1])


def filter_students(ctx) -> tuple[pd.DataFrame, str]:
    """Apply exploratory filters (school, gender, student) to the 900 student rows."""
    df = D.students()
    parts = []
    sid = selected_school_id(ctx)
    if sid is not None:
        df = df[df["school_id"] == sid]
        parts.append(D.school_label(sid))
    if ctx["gender"] != ctx["ALL"]:
        df = df[df["gender"] == ctx["gender"]]
        parts.append(ctx["gender"])
    if ctx["student"] != ctx["ALL"]:
        df = df[df["student_id"] == ctx["student"]]
        parts.append(f"Student {ctx['student']}")
    desc = " · ".join(parts) if parts else "All students"
    return df.reset_index(drop=True), desc


def gender_filtered_students(ctx) -> pd.DataFrame:
    """Students filtered by gender only, used to build school-level (30-row) aggregates."""
    df = D.students()
    if ctx["gender"] != ctx["ALL"]:
        df = df[df["gender"] == ctx["gender"]]
    return df


def empty_note(msg: str) -> None:
    S.insight(msg, title="Not enough data", kind="warn")


def chart(fig, key: str | None = None) -> None:
    """Render a chart that stretches to the free height of its card."""
    from utils.charts import CONFIG
    fig.update_layout(height=None, autosize=True)
    try:
        st.plotly_chart(fig, width="stretch", height="stretch", theme=None, config=CONFIG, key=key)
    except TypeError:  # Streamlit without height="stretch": fall back to a fixed height
        fig.update_layout(height=260)
        st.plotly_chart(fig, use_container_width=True, theme=None, config=CONFIG, key=key)
