# db.py
import duckdb
from pathlib import Path

DB_PATH = Path(__file__).parent / "benchmark.duckdb"

def get_connection():
    con = duckdb.connect(str(DB_PATH))
    return con

def init_db():
    con = get_connection()

    con.execute("""
        CREATE TABLE IF NOT EXISTS sales_history (
            project         VARCHAR,
            item_id         VARCHAR,
            sale_date       DATE,
            sales           DOUBLE
        );
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS forecasts (
            project          VARCHAR,
            forecast_method  VARCHAR,
            item_id          VARCHAR,
            forecast_date    DATE,
            forecast         DOUBLE
        );
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS forecast_metrics (
            project          VARCHAR,
            item_id          VARCHAR,
            forecast_method  VARCHAR,
            metric_name      VARCHAR,   -- 'MAPE', 'sMAPE', 'WAPE'
            metric_value     DOUBLE,
            n_points         INTEGER,
            last_updated     TIMESTAMP DEFAULT current_timestamp
        );
    """)

    con.close()

if __name__ == "__main__":
    init_db()
    print("DB initialized.")
