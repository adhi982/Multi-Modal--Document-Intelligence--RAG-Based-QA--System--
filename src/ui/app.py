"""Main Streamlit app."""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

st.set_page_config(
    page_title="Multi-Modal Document QA",
    page_icon="📋",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Query Interface", "Metrics Dashboard", "Example Queries"]
)

# Page routing
if page == "Query Interface":
    from pages import query
    query.show()

elif page == "Metrics Dashboard":
    from pages import dashboard
    dashboard.show()

elif page == "Example Queries":
    from pages import examples
    examples.show()
