import streamlit as st


def render_sidebar_nav(*, entrypoint: str | None = None) -> None:
    """Render the top-of-sidebar page navigation with custom labels.

    Streamlit's multipage navigation only supports linking to:
    - the active entrypoint file (the file passed to `streamlit run`)
    - files inside `pages/`

    So we must link to the *current* entrypoint, not hardcode app.py.
    """

    entrypoint = entrypoint or st.session_state.get("_entrypoint", "app.py")
    st.sidebar.page_link(str(entrypoint), label="View per Item")
    st.sidebar.page_link("pages/1_Scoreboard.py", label="Scoreboard")
    st.sidebar.page_link("pages/3_Items.py", label="Items")
    st.sidebar.page_link("pages/2_Data_Import.py", label="Data Import")
    st.sidebar.markdown("---")
