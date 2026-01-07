# ingest.py
import pandas as pd
import os
from pathlib import Path
from io import BytesIO
import re
from datetime import date
from db import get_connection

DATA_DIR = Path(os.getenv("BENCHMARK_DATA_DIR", str(Path(__file__).parent / "Data")))

def import_sales_history(csv_path: Path):
    df = pd.read_csv(csv_path, sep=';')

    # Normalize column names
    df = df.rename(columns={
        "saledate": "sale_date",
        "sales": "sales",
        "project": "project",
        "item_id": "item_id"
    })

    # Ensure date type
    df["sale_date"] = pd.to_datetime(df["sale_date"]).dt.date

    con = get_connection()
    con.execute("DELETE FROM sales_history WHERE TRUE;")  # optional: wipe before load
    con.register("sales_df", df)
    con.execute("""
        INSERT INTO sales_history
        SELECT project, item_id, sale_date, sales
        FROM sales_df
    """)
    con.close()

def import_forecasts(csv_path: Path):
    df = pd.read_csv(csv_path, sep=';')

    df = df.rename(columns={
        "forecastmethod": "forecast_method",
        "forecastdate": "forecast_date",
        "forecast": "forecast",
        "project": "project",
        "item_id": "item_id"
    })

    df["forecast_date"] = pd.to_datetime(df["forecast_date"]).dt.date

    con = get_connection()
    con.execute("DELETE FROM forecasts WHERE TRUE;")  # optional: wipe before load
    con.register("forecasts_df", df)
    con.execute("""
        INSERT INTO forecasts
        SELECT project, forecast_method, item_id, forecast_date, forecast
        FROM forecasts_df
    """)
    con.close()


_YEAR_RE = re.compile(r"^\s*(\d{4})\s*$")


def _try_read_csv_bytes(data: bytes, *, sep: str = ";") -> pd.DataFrame:
    """Read CSV bytes with a couple of encoding fallbacks."""
    last_error: Exception | None = None
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(BytesIO(data), sep=sep, dtype=str, encoding=enc)
        except Exception as e:  # pragma: no cover
            last_error = e
    assert last_error is not None
    raise last_error


def _parse_decimal_comma(value: str) -> float:
    v = value.strip()
    v = v.replace(" ", "")
    v = v.replace("\u00a0", "")
    v = v.replace(",", ".")
    return float(v)


def import_sales_history_horizontal_months(
    csv_bytes: bytes,
    *,
    project: str,
    overwrite_project_date_range: bool = True,
) -> dict:
    """Import wide monthly CSV into sales_history.

    Expected shape: one row per item/year with the last 12 fields as month values.
    Each month value is imported on the first of that month.

    Rules:
    - `project` is provided by the user (not from file).
    - Keep negatives.
    - Skip blanks.
    - Import 0 as 0.
    """

    if not project or not project.strip():
        raise ValueError("project is required")
    project = project.strip()

    df = _try_read_csv_bytes(csv_bytes, sep=";")
    if df.empty:
        return {"rows_inserted": 0, "items": 0, "years": 0}

    rows_out: list[dict] = []
    item_ids: set[str] = set()
    years: set[int] = set()

    # Iterate as raw strings; the file often has irregular headers/encodings.
    for _, row in df.iterrows():
        values = ["" if pd.isna(v) else str(v) for v in row.tolist()]

        # Skip header-ish row(s)
        if values and str(values[0]).strip().upper().startswith("HORIZONTAL"):
            continue

        if len(values) < 13:
            continue

        month_cells = values[-12:]
        meta = values[:-12]

        # Find year: first 4-digit number in the metadata cells
        year: int | None = None
        for cell in meta:
            m = _YEAR_RE.match(str(cell))
            if m:
                year = int(m.group(1))
                break
        if year is None:
            continue

        # Item id: first non-empty cell before the year cell (or first non-empty meta cell)
        item_id: str | None = None
        for cell in meta:
            s = str(cell).strip()
            if not s:
                continue
            # Skip the year itself
            if _YEAR_RE.match(s):
                continue
            item_id = s
            break
        if item_id is None:
            continue

        item_ids.add(item_id)
        years.add(year)

        for month_index, cell in enumerate(month_cells, start=1):
            s = str(cell).strip()
            if not s:
                continue  # skip blanks
            sales = _parse_decimal_comma(s)
            rows_out.append(
                {
                    "project": project,
                    "item_id": item_id,
                    "sale_date": date(year, month_index, 1),
                    "sales": sales,
                }
            )

    if not rows_out:
        return {"rows_inserted": 0, "items": 0, "years": 0}

    out_df = pd.DataFrame(rows_out)

    con = get_connection()
    try:
        if overwrite_project_date_range:
            min_date = out_df["sale_date"].min()
            max_date = out_df["sale_date"].max()
            con.execute(
                """
                DELETE FROM sales_history
                WHERE project = ?
                  AND sale_date BETWEEN ? AND ?
                """,
                [project, min_date, max_date],
            )

        con.register("sales_df", out_df)
        con.execute(
            """
            INSERT INTO sales_history
            SELECT project, item_id, sale_date, sales
            FROM sales_df
            """
        )
    finally:
        con.close()

    return {
        "rows_inserted": int(len(out_df)),
        "items": int(len(item_ids)),
        "years": int(len(years)),
        "min_date": str(out_df["sale_date"].min()),
        "max_date": str(out_df["sale_date"].max()),
    }


def import_forecasts_horizontal_months(
    csv_bytes: bytes,
    *,
    project: str,
    forecast_method: str,
    overwrite_project_date_range: bool = True,
) -> dict:
    """Import wide monthly forecast CSV into forecasts.

    Expected shape: one row per item/year with the last 12 fields as month values.
    Each month value is imported on the first of that month.

    Rules:
    - `project` and `forecast_method` are provided by the user (not from file).
    - Skip blanks.
    - Import 0 as 0.
    """

    if not project or not project.strip():
        raise ValueError("project is required")
    if not forecast_method or not forecast_method.strip():
        raise ValueError("forecast_method is required")

    project = project.strip()
    forecast_method = forecast_method.strip()

    df = _try_read_csv_bytes(csv_bytes, sep=";")
    if df.empty:
        return {"rows_inserted": 0, "items": 0, "years": 0}

    rows_out: list[dict] = []
    item_ids: set[str] = set()
    years: set[int] = set()

    for _, row in df.iterrows():
        values = ["" if pd.isna(v) else str(v) for v in row.tolist()]

        # Skip header-ish row(s)
        if values and str(values[0]).strip().upper().startswith("HORIZONTAL"):
            continue

        if len(values) < 13:
            continue

        month_cells = values[-12:]
        meta = values[:-12]

        year: int | None = None
        for cell in meta:
            m = _YEAR_RE.match(str(cell))
            if m:
                year = int(m.group(1))
                break
        if year is None:
            continue

        item_id: str | None = None
        for cell in meta:
            s = str(cell).strip()
            if not s:
                continue
            if _YEAR_RE.match(s):
                continue
            item_id = s
            break
        if item_id is None:
            continue

        item_ids.add(item_id)
        years.add(year)

        for month_index, cell in enumerate(month_cells, start=1):
            s = str(cell).strip()
            if not s:
                continue
            forecast = _parse_decimal_comma(s)
            rows_out.append(
                {
                    "project": project,
                    "forecast_method": forecast_method,
                    "item_id": item_id,
                    "forecast_date": date(year, month_index, 1),
                    "forecast": forecast,
                }
            )

    if not rows_out:
        return {"rows_inserted": 0, "items": 0, "years": 0}

    out_df = pd.DataFrame(rows_out)

    con = get_connection()
    try:
        if overwrite_project_date_range:
            min_date = out_df["forecast_date"].min()
            max_date = out_df["forecast_date"].max()
            con.execute(
                """
                DELETE FROM forecasts
                WHERE project = ?
                  AND forecast_method = ?
                  AND forecast_date BETWEEN ? AND ?
                """,
                [project, forecast_method, min_date, max_date],
            )

        con.register("forecasts_df", out_df)
        con.execute(
            """
            INSERT INTO forecasts
            SELECT project, forecast_method, item_id, forecast_date, forecast
            FROM forecasts_df
            """
        )
    finally:
        con.close()

    return {
        "rows_inserted": int(len(out_df)),
        "items": int(len(item_ids)),
        "years": int(len(years)),
        "min_date": str(out_df["forecast_date"].min()),
        "max_date": str(out_df["forecast_date"].max()),
        "forecast_method": forecast_method,
        "project": project,
    }

if __name__ == "__main__":
    sales_csv = DATA_DIR / "sales_history.csv"
    forecasts_csv = DATA_DIR / "forecasts.csv"

    import_sales_history(sales_csv)
    import_forecasts(forecasts_csv)
    print("Imported sales history and forecasts.")
