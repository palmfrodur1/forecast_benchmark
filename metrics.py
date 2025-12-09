# metrics.py
import pandas as pd
from db import get_connection, init_db

def mape(y_true, y_pred):
    df = pd.DataFrame({"y": y_true, "yhat": y_pred}).dropna()
    # Exclude zero actuals from MAPE
    df = df[df["y"] != 0]
    if df.empty:
        return None
    return (abs((df["y"] - df["yhat"]) / df["y"]).mean()) * 100.0

def smape(y_true, y_pred):
    df = pd.DataFrame({"y": y_true, "yhat": y_pred}).dropna()
    denom = (df["y"].abs() + df["yhat"].abs())
    nonzero = denom != 0
    df = df[nonzero]
    if df.empty:
        return None
    return (2.0 * (df["y"] - df["yhat"]).abs() / denom[nonzero]).mean() * 100.0

def wape(y_true, y_pred):
    df = pd.DataFrame({"y": y_true, "yhat": y_pred}).dropna()
    denom = df["y"].abs().sum()
    if denom == 0:
        return None
    return ( (df["y"] - df["yhat"]).abs().sum() / denom ) * 100.0

def recompute_all_metrics():
    """Join history + forecasts on date and write metrics into forecast_metrics."""
    init_db()
    con = get_connection()

    # Join on project, item, date
    joined = con.execute("""
        SELECT
            s.project,
            s.item_id,
            f.forecast_method,
            s.sale_date   AS date,
            s.sales       AS sales,
            f.forecast    AS forecast
        FROM sales_history s
        JOIN forecasts f
          ON s.project = f.project
         AND s.item_id = f.item_id
         AND s.sale_date = f.forecast_date
        ORDER BY s.project, s.item_id, f.forecast_method, s.sale_date
    """).fetchdf()

    if joined.empty:
        print("No overlapping history/forecast dates – nothing to compute.")
        con.close()
        return

    rows = []
    grouped = joined.groupby(["project", "item_id", "forecast_method"])

    for (project, item_id, method), g in grouped:
        y = g["sales"].astype(float)
        yhat = g["forecast"].astype(float)

        metrics = {
            "MAPE":  mape(y, yhat),
            "sMAPE": smape(y, yhat),
            "WAPE":  wape(y, yhat),
        }
        n_points = int(len(g))

        for name, value in metrics.items():
            if value is None:
                continue
            rows.append(
                {
                    "project": project,
                    "item_id": item_id,
                    "forecast_method": method,
                    "metric_name": name,
                    "metric_value": float(value),
                    "n_points": n_points,
                }
            )

    # Write back to DuckDB
    con.execute("DELETE FROM forecast_metrics;")  # simple full refresh
    if rows:
        df_metrics = pd.DataFrame(rows)
        con.register("metrics_df", df_metrics)
        con.execute("""
            INSERT INTO forecast_metrics
            (project, item_id, forecast_method, metric_name, metric_value, n_points)
            SELECT project, item_id, forecast_method, metric_name, metric_value, n_points
            FROM metrics_df
        """)
        print(f"Inserted {len(rows)} metric rows.")
    else:
        print("No metrics to insert.")

    con.close()

if __name__ == "__main__":
    recompute_all_metrics()
