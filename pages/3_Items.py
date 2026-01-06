import os

import pandas as pd
import streamlit as st
import sqlalchemy as sa

from db import get_connection, get_mysql_engine
from nav import render_sidebar_nav


st.set_page_config(page_title="Items", layout="wide")

render_sidebar_nav()


def _use_mysql() -> bool:
    if st.session_state.get("_entrypoint") == "app_mysql.py":
        return True
    return (os.getenv("BENCHMARK_DB_BACKEND") or "").lower() == "mysql"


@st.cache_resource
def _mysql_engine() -> sa.Engine:
    return get_mysql_engine()


def _mysql_read_df(sql: str, params: dict | None = None) -> pd.DataFrame:
    eng = _mysql_engine()
    with eng.connect() as conn:
        return pd.read_sql(sa.text(sql), conn, params=params or {})


@st.cache_data(ttl=60)
def _get_projects_with_items() -> list[str]:
    if _use_mysql():
        df = _mysql_read_df("SELECT DISTINCT project FROM item_features ORDER BY project")
    else:
        con = get_connection()
        try:
            df = con.execute("SELECT DISTINCT project FROM item_features ORDER BY project").fetchdf()
        finally:
            con.close()
    if df.empty:
        return []
    return df["project"].dropna().astype(str).tolist()


@st.cache_data(ttl=60)
def _load_items(project: str) -> pd.DataFrame:
    if _use_mysql():
        return _mysql_read_df(
            """
            SELECT project, item_id, name, item_type, flavour, size
            FROM item_features
            WHERE project = :project
            ORDER BY item_id
            """,
            {"project": project},
        )

    con = get_connection()
    try:
        return con.execute(
            """
            SELECT project, item_id, name, item_type, flavour, size
            FROM item_features
            WHERE project = ?
            ORDER BY item_id
            """,
            [project],
        ).fetchdf()
    finally:
        con.close()


def main() -> None:
    st.title("Items")
    st.caption("Items from the item_features table.")

    try:
        projects = _get_projects_with_items()
    except Exception as e:
        st.error(f"Could not read item_features: {e}")
        return

    if not projects:
        st.info("No rows found in item_features.")
        return

    project = st.sidebar.selectbox("Project", projects, key="items_project")

    try:
        df = _load_items(project)
    except Exception as e:
        st.error(f"Could not load items: {e}")
        return

    if df.empty:
        st.info("No items for this project in item_features.")
        return

    df["project"] = df["project"].astype(str)
    df["item_id"] = df["item_id"].astype(str)

    st.subheader(f"{project} — {len(df)} items")
    st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()
