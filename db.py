# db.py
import os
import duckdb
from pathlib import Path

# Previous local default (kept for reference):
# _DEFAULT_DB_PATH = Path(__file__).parent / "benchmark.duckdb"
# DB_PATH = Path(os.getenv("BENCHMARK_DB_PATH", str(_DEFAULT_DB_PATH)))

# Default to the DuckDB file inside the docker named volume mounted at /duckdb
_DEFAULT_DB_PATH = Path("/duckdb_workspace/sog.duckdb")
DB_PATH = Path(os.getenv("BENCHMARK_DB_PATH", str(_DEFAULT_DB_PATH)))

def get_connection():
    con = duckdb.connect(str(DB_PATH))
    return con


def get_connection_for_path(db_path: str | Path, *, read_only: bool = False):
    """Connect to a specific DuckDB file path.

    This is useful when the DB file is not the default `DB_PATH`.
    """
    return duckdb.connect(str(Path(db_path)), read_only=read_only)


def get_sog_volume_connection(*, read_only: bool = True):
    """Connect to the SOG DuckDB stored in the docker named volume.

    Run your code inside a container with the named volume mounted:
    `-v duckdb_workspace:/duckdb`

    Then this function will open: `/duckdb/sog.duckdb`.
    """
    return get_connection_for_path("/duckdb/sog.duckdb", read_only=read_only)

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
