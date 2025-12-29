from pathlib import Path

import streamlit as st

from db import init_db
from ingest import import_sales_history, import_forecasts, import_sales_history_horizontal_months
from db import get_connection

from nav import render_sidebar_nav


st.set_page_config(page_title="Data Import", layout="wide")

render_sidebar_nav()


def _repo_root() -> Path:
    # pages/2_Data_Import.py -> repo root
    return Path(__file__).resolve().parents[1]


def main() -> None:
    init_db()

    st.title("Data Import")
    st.caption("Reload benchmark data from CSV files into DuckDB.")

    data_dir = _repo_root() / "Data"
    sales_csv = data_dir / "sales_history.csv"
    forecasts_csv = data_dir / "forecasts.csv"

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sales history")
        st.write(str(sales_csv))
        st.write("Exists:", sales_csv.exists())
        if sales_csv.exists():
            st.write("Size (bytes):", sales_csv.stat().st_size)

    with col2:
        st.subheader("Forecasts")
        st.write(str(forecasts_csv))
        st.write("Exists:", forecasts_csv.exists())
        if forecasts_csv.exists():
            st.write("Size (bytes):", forecasts_csv.stat().st_size)

    st.markdown("---")

    if st.button("Reload Data from CSV", type="primary"):
        with st.spinner("Reloading data..."):
            try:
                if not sales_csv.exists():
                    raise FileNotFoundError(f"Missing {sales_csv}")
                if not forecasts_csv.exists():
                    raise FileNotFoundError(f"Missing {forecasts_csv}")

                import_sales_history(sales_csv)
                import_forecasts(forecasts_csv)
                st.cache_data.clear()
                st.success("Data reloaded successfully.")
            except Exception as e:
                st.error(f"Error reloading data: {e}")

    st.markdown("---")

    st.subheader("Import HORIZONTAL monthly sales CSV")
    st.caption(
        "Upload a wide monthly file (last 12 columns = months). "
        "Values are imported on the 1st of each month into the selected project."
    )

    existing_projects: list[str] = []
    try:
        con = get_connection()
        try:
            existing_projects = (
                con.execute(
                    "SELECT DISTINCT project FROM sales_history WHERE project IS NOT NULL ORDER BY project"
                )
                .fetchdf()["project"]
                .dropna()
                .astype(str)
                .tolist()
            )
        finally:
            con.close()
    except Exception:
        existing_projects = []

    project = st.text_input(
        "Project to import into",
        key='import_project',
        help=(
            "Required. This will be written to sales_history.project for all imported rows. "
            "Existing projects: " + (", ".join(existing_projects) if existing_projects else "(none yet)")
        ),
    )

    overwrite = st.checkbox(
        "Overwrite existing sales for this project in imported date range",
        value=True,
        key='import_overwrite',
    )

    uploaded = st.file_uploader(
        "HORIZONTAL monthly CSV",
        type=["csv"],
        accept_multiple_files=False,
    )

    disabled = (uploaded is None) or (not project.strip())
    if st.button("Import HORIZONTAL Monthly Sales", disabled=disabled):
        with st.spinner("Importing monthly sales..."):
            try:
                stats = import_sales_history_horizontal_months(
                    uploaded.getvalue(),
                    project=project,
                    overwrite_project_date_range=overwrite,
                )
                st.cache_data.clear()
                st.success(
                    f"Imported {stats.get('rows_inserted', 0)} rows "
                    f"({stats.get('items', 0)} items, {stats.get('years', 0)} years)."
                )
                if stats.get("min_date") and stats.get("max_date"):
                    st.caption(f"Date range: {stats['min_date']} to {stats['max_date']}")
            except Exception as e:
                st.error(f"Error importing HORIZONTAL monthly sales: {e}")

    st.info(
        "After reloading, go back to the main page to explore items and forecasts. "
        "If metrics look stale, recompute them from the main page sidebar."
    )


if __name__ == "__main__":
    main()
