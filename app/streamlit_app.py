"""
SkillMine dashboard (Phase 6).

Launch from the project root:
    streamlit run app/streamlit_app.py

Page implementations are in app/views.py.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import views

st.set_page_config(page_title="SkillMine", page_icon="⛏️", layout="wide")

nav = st.navigation({
    "Historical analysis": [
        st.Page(views.overview_page,  title="Overview",          icon="🏠", url_path="overview", default=True),
        st.Page(views.rules_page,     title="Association Rules", icon="🔗", url_path="rules"),
        st.Page(views.network_page,   title="Skill Network",     icon="🕸️", url_path="network"),
        st.Page(views.clusters_page,  title="Posting Clusters",  icon="🧩", url_path="clusters"),
        st.Page(views.explorer_page,  title="Skill Explorer",    icon="🔍", url_path="explorer"),
        st.Page(views.emerging_page,  title="Emerging Skills",   icon="📈", url_path="emerging"),
    ],
})
nav.run()
