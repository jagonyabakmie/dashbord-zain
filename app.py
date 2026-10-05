"""Mathematics Achievement Analytics: six-page multilevel dashboard.

Run with:  streamlit run app.py
All statistics are read from MASTER_STREAMLIT_MULTILEVEL.xlsx; nothing is re-fitted here.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Mathematics Achievement Analytics",
    page_icon="📗",
    layout="wide",
    initial_sidebar_state="auto",
)

from utils import data as D  # noqa: E402
from utils import styles as S  # noqa: E402
from views import (diagnostics, exploration, model_lab, overview,  # noqa: E402
                   research_questions, school_explorer)

PAGES = {
    "overview": (":material/home: Overview & Descriptive Profile", overview.render),
    "exploration": (":material/travel_explore: Exploratory Relationship Analysis", exploration.render),
    "rq": (":material/lightbulb: Research Questions", research_questions.render),
    "lab": (":material/science: Multilevel Model Laboratory", model_lab.render),
    "school": (":material/school: School Performance Explorer", school_explorer.render),
    "diag": (":material/verified_user: Diagnostics & Robustness", diagnostics.render),
}
EXPLORATORY = {"overview", "exploration"}

ALL_SCHOOLS = "All Schools"
ALL = "All"


def _init_state() -> None:
    st.session_state.setdefault("page", "overview")
    st.session_state.setdefault("f_school", ALL_SCHOOLS)
    st.session_state.setdefault("f_gender", ALL)
    st.session_state.setdefault("f_student", ALL)


def _reset_filters() -> None:
    st.session_state["f_school"] = ALL_SCHOOLS
    st.session_state["f_gender"] = ALL
    st.session_state["f_student"] = ALL


def _persist_select(label: str, options: list, store_key: str, **kw) -> None:
    """Selectbox whose value survives pages where the widget is not rendered."""
    wkey = "w_" + store_key
    if st.session_state[store_key] not in options:
        st.session_state[store_key] = options[0]
    st.session_state[wkey] = st.session_state[store_key]

    def _sync():
        st.session_state[store_key] = st.session_state[wkey]

    st.selectbox(label, options, key=wkey, on_change=_sync, **kw)


def sidebar(students) -> str:
    with st.sidebar:
        st.markdown(
            f"""<div class="brand"><div class="brand-mark">{S.icon('menu_book')}</div>
<div><div class="brand-title">Math Achievement</div><div class="brand-sub">Multilevel Analytics</div></div></div>""",
            unsafe_allow_html=True,
        )
        st.markdown("<hr/>", unsafe_allow_html=True)
        keys = list(PAGES)
        page = st.radio(
            "Navigation", keys, key="page", label_visibility="collapsed",
            format_func=lambda k: PAGES[k][0],
        )
        st.markdown("<hr/>", unsafe_allow_html=True)

        school_ids = sorted(students["school_id"].unique())
        school_opts = [ALL_SCHOOLS] + [D.school_label(s) for s in school_ids]
        if page in EXPLORATORY:
            st.markdown(f'<div class="side-label">{S.icon("tune")} Filters</div>', unsafe_allow_html=True)
            _persist_select("School", school_opts, "f_school")
            genders = [ALL] + sorted(students["gender"].dropna().unique().tolist())
            _persist_select("Gender", genders, "f_gender")
            pool = students
            if st.session_state["f_school"] != ALL_SCHOOLS:
                pool = pool[pool["school_label"] == st.session_state["f_school"]]
            if st.session_state["f_gender"] != ALL:
                pool = pool[pool["gender"] == st.session_state["f_gender"]]
            stud_opts = [ALL] + sorted(pool["student_id"].tolist())
            _persist_select("Student ID", stud_opts, "f_student",
                            help="Optional. Narrows student-level views to a single student.")
            st.button("Reset filters", on_click=_reset_filters, width="stretch", icon=":material/restart_alt:")
        elif page == "rq":
            st.markdown(
                f'<div class="side-note">Research questions use the locked final model '
                f'<b>{D.final_model_id()}</b> on the full sample (900 students, 30 schools). '
                f'Sidebar filters do not change these results.</div>',
                unsafe_allow_html=True,
            )
        elif page == "diag":
            st.markdown(f'<div class="side-label">{S.icon("my_location")} Highlight</div>', unsafe_allow_html=True)
            _persist_select("Highlight school", school_opts, "f_school",
                            help="Highlights a school in school-level visuals only. Inferential results stay on the full sample.")
            st.markdown(
                f'<div class="side-note">Inferential results use the locked final model '
                f'<b>{D.final_model_id()}</b> on the full sample (900 students, 30 schools). '
                f'Selecting a school only highlights it.</div>',
                unsafe_allow_html=True,
            )
        elif page == "lab":
            st.markdown(
                '<div class="side-note">Model specifications are read from the precomputed '
                'model bank. Nothing is re-fitted in the browser.</div>',
                unsafe_allow_html=True,
            )
        elif page == "school":
            st.markdown(
                f'<div class="side-note">School effects come from the locked final model '
                f'<b>{D.final_model_id()}</b>. Choose a school on the page.</div>',
                unsafe_allow_html=True,
            )

        meta = D.readme()
        n_st = int(meta.get("Students", len(students)))
        n_sc = int(meta.get("Schools", students["school_id"].nunique()))
        st.markdown(
            f'<div class="side-footer"><b>RASIO 10.0 • Final Round</b><br/>{n_st} Students • {n_sc} Schools</div>',
            unsafe_allow_html=True,
        )
    return page


def main() -> None:
    _init_state()
    S.inject_css()
    students = D.students()
    page = sidebar(students)
    ctx = {
        "school": st.session_state["f_school"],
        "gender": st.session_state["f_gender"],
        "student": st.session_state["f_student"],
        "ALL_SCHOOLS": ALL_SCHOOLS,
        "ALL": ALL,
    }
    PAGES[page][1](ctx)


main()
