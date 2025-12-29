import streamlit as st


def render_sidebar_nav() -> None:
    """Render the top-of-sidebar page navigation with custom labels."""
    st.sidebar.page_link("app.py", label="View per Item")
    st.sidebar.page_link("pages/1_Scoreboard.py", label="Scoreboard")
    st.sidebar.page_link("pages/2_Data_Import.py", label="Data Import")
    st.sidebar.markdown("---")
