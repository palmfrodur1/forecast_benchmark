# ingest.py
import pandas as pd
import os
from pathlib import Path
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

if __name__ == "__main__":
    sales_csv = DATA_DIR / "sales_history.csv"
    forecasts_csv = DATA_DIR / "forecasts.csv"

    import_sales_history(sales_csv)
    import_forecasts(forecasts_csv)
    print("Imported sales history and forecasts.")
