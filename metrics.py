# metrics.py
from __future__ import annotations

from datetime import date

import pandas as pd

from db import get_connection, init_db


def _prepare_eval_frame(
    y_true,
    y_pred,
    *,
    abs_actual_threshold: float = 0.0,
) -> pd.DataFrame:
    df = pd.DataFrame({"y": y_true, "yhat": y_pred}).dropna()
    if df.empty:
        return df

    # Business rule: negative sales are treated as 0 in error calculations.
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0.0)
    df["yhat"] = pd.to_numeric(df["yhat"], errors="coerce")
    df["y"] = df["y"].clip(lower=0.0)

    df = df.dropna(subset=["yhat"])
    if df.empty:
        return df

    thr = float(abs_actual_threshold or 0.0)
    if thr > 0:
        df = df[df["y"].abs() >= thr]

    return df

def mape(y_true, y_pred, *, abs_actual_threshold: float = 0.0):
    df = _prepare_eval_frame(y_true, y_pred, abs_actual_threshold=abs_actual_threshold)
    # Exclude zero actuals from MAPE
    df = df[df["y"] != 0]
    if df.empty:
        return None
    return (abs((df["y"] - df["yhat"]) / df["y"]).mean()) * 100.0

def smape(y_true, y_pred, *, abs_actual_threshold: float = 0.0):
    df = _prepare_eval_frame(y_true, y_pred, abs_actual_threshold=abs_actual_threshold)
    denom = (df["y"].abs() + df["yhat"].abs())
    nonzero = denom != 0
    df = df[nonzero]
    if df.empty:
        return None
    return (2.0 * (df["y"] - df["yhat"]).abs() / denom[nonzero]).mean() * 100.0

def wape(y_true, y_pred, *, abs_actual_threshold: float = 0.0):
    df = _prepare_eval_frame(y_true, y_pred, abs_actual_threshold=abs_actual_threshold)
    denom = df["y"].abs().sum()
    if denom == 0:
        return None
    return ( (df["y"] - df["yhat"]).abs().sum() / denom ) * 100.0


def rmse(y_true, y_pred, *, abs_actual_threshold: float = 0.0):
    df = _prepare_eval_frame(y_true, y_pred, abs_actual_threshold=abs_actual_threshold)
    if df.empty:
        return None
    err = (df["yhat"] - df["y"]).astype(float)
    return float((err.pow(2).mean()) ** 0.5)


def bias(y_true, y_pred, *, abs_actual_threshold: float = 0.0):
    """Mean error (forecast - actual). Positive means over-forecast."""
    df = _prepare_eval_frame(y_true, y_pred, abs_actual_threshold=abs_actual_threshold)
    if df.empty:
        return None
    err = (df["yhat"] - df["y"]).astype(float)
    return float(err.mean())


def bias_pct(y_true, y_pred, *, abs_actual_threshold: float = 0.0):
    """Percent bias (relative bias), based on totals.

    Defined as 100 * sum(forecast - actual) / sum(actual).
    Positive means over-forecast.
    """
    df = _prepare_eval_frame(y_true, y_pred, abs_actual_threshold=abs_actual_threshold)
    if df.empty:
        return None
    denom = float(df["y"].sum())
    if denom == 0:
        return None
    num = float((df["yhat"] - df["y"]).sum())
    return (num / denom) * 100.0

def recompute_all_metrics(
    *,
    project: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    abs_actual_threshold: float = 0.0,
):
    """Join history + forecasts on date and write metrics into forecast_metrics.

    Metrics are computed over the overlapping dates between `sales_history.sale_date`
    and `forecasts.forecast_date`.

    If provided, `start_date`/`end_date` filter the evaluation period (inclusive).
    If provided, `project` scopes both computation and the metrics overwrite.
    """
    init_db()
    con = get_connection()

    where_clauses: list[str] = []
    params: list[object] = []
    if project:
        where_clauses.append("s.project = ?")
        params.append(project)
    if start_date is not None:
        where_clauses.append("s.sale_date >= ?")
        params.append(start_date)
    if end_date is not None:
        where_clauses.append("s.sale_date <= ?")
        params.append(end_date)

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    joined = con.execute(
        f"""
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
        {where_sql}
        ORDER BY s.project, s.item_id, f.forecast_method, s.sale_date
        """,
        params,
    ).fetchdf()

    if joined.empty:
        # Overwrite semantics: metrics table represents the last computation.
        if project:
            con.execute("DELETE FROM forecast_metrics WHERE project = ?;", [project])
        else:
            con.execute("DELETE FROM forecast_metrics;")

        print("No overlapping history/forecast dates – nothing to compute.")
        con.close()
        return

    rows = []
    grouped = joined.groupby(["project", "item_id", "forecast_method"])

    for (project, item_id, method), g in grouped:
        df_eval = _prepare_eval_frame(
            g["sales"],
            g["forecast"],
            abs_actual_threshold=abs_actual_threshold,
        )
        if df_eval.empty:
            continue

        y = df_eval["y"].astype(float)
        yhat = df_eval["yhat"].astype(float)

        metrics = {
            "MAPE": mape(y, yhat, abs_actual_threshold=0.0),
            "sMAPE": smape(y, yhat, abs_actual_threshold=0.0),
            "WAPE": wape(y, yhat, abs_actual_threshold=0.0),
            "RMSE": rmse(y, yhat, abs_actual_threshold=0.0),
            "Bias": bias(y, yhat, abs_actual_threshold=0.0),
            "BiasPct": bias_pct(y, yhat, abs_actual_threshold=0.0),
        }
        n_points = int(len(df_eval))

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
    if project:
        con.execute("DELETE FROM forecast_metrics WHERE project = ?;", [project])
    else:
        con.execute("DELETE FROM forecast_metrics;")
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
