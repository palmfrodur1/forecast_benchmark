"""Database helpers.

This project primarily uses DuckDB today.

We also include a MySQL connection helper so you can point new scripts at a MySQL
instance without rewriting connection code.

Important: the Streamlit app and SQL queries are currently DuckDB-oriented
(`split_part`, `strpos`, `::VARCHAR`, DuckDB DataFrame registration, etc.).
If you set BENCHMARK_DB_BACKEND=mysql, you should expect to update query SQL
and some DuckDB-specific behaviors across the project.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import duckdb

try:
    import sqlalchemy as sa  # type: ignore
except Exception:  # pragma: no cover
    sa = None


_DOTENV_LOADED = False


def load_env() -> None:
    """Load environment variables from a local .env file if python-dotenv is installed.

    This keeps local development simple (no need to `source .env`) and is safe
    because `.env` is git-ignored in this repo.
    """

    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True

    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return

    # Repo root is the same folder as this file.
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)


class DBBackend(str, Enum):
    DUCKDB = "duckdb"
    MYSQL = "mysql"


def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip()
    return v if v else default


def get_db_backend() -> DBBackend:
    """Return selected backend.

Controlled by env var `BENCHMARK_DB_BACKEND` (duckdb|mysql). Defaults to duckdb.
"""
    raw = (_env("BENCHMARK_DB_BACKEND", "duckdb") or "duckdb").lower()
    try:
        return DBBackend(raw)
    except Exception:
        return DBBackend.DUCKDB


# -----------------------------
# DuckDB configuration
# -----------------------------


def _default_duckdb_path() -> Path:
    """Choose a sensible default DuckDB path.

Preference order:
1) Local repo file `./sog.duckdb` (local dev)
2) Docker named volume mount path `/duckdb/sog.duckdb` (compose + run scripts)
3) Legacy path `/duckdb_workspace/sog.duckdb` (older setups)
"""
    local = Path(__file__).parent / "sog.duckdb"
    if local.exists():
        return local

    for candidate in (Path("/duckdb/sog.duckdb"), Path("/duckdb_workspace/sog.duckdb")):
        if candidate.exists():
            return candidate
    # If nothing exists yet, default to the modern container path.
    return Path("/duckdb/sog.duckdb")


@dataclass(frozen=True)
class DuckDBConfig:
    path: Path


def get_duckdb_config() -> DuckDBConfig:
    path = Path(_env("BENCHMARK_DB_PATH", str(_default_duckdb_path())) or str(_default_duckdb_path()))
    return DuckDBConfig(path=path)


def get_duckdb_connection(*, read_only: bool = False):
    cfg = get_duckdb_config()
    return duckdb.connect(str(cfg.path), read_only=read_only)


def get_connection_for_path(db_path: str | Path, *, read_only: bool = False):
    """Connect to a specific DuckDB file path."""
    return duckdb.connect(str(Path(db_path)), read_only=read_only)


def get_sog_volume_connection(*, read_only: bool = True):
    """Convenience: connect to the DuckDB file in the docker named volume."""
    return get_connection_for_path("/duckdb/sog.duckdb", read_only=read_only)


# -----------------------------
# MySQL configuration
# -----------------------------


@dataclass(frozen=True)
class MySQLConfig:
    host: str
    port: int
    user: str
    password: str
    database: str


def get_mysql_config() -> MySQLConfig:
    """Build MySQL config from env vars.

Env vars:
- MYSQL_HOST (default: 192.168.1.50)
- MYSQL_PORT (default: 4406)
- MYSQL_USER (default: root)
- MYSQL_PASSWORD (no default)
- MYSQL_DATABASE (default: forecast_benchmark)

Note: We intentionally do NOT hardcode passwords in the repo.
"""

    host = _env("MYSQL_HOST", "192.168.1.50") or "192.168.1.50"
    port_raw = _env("MYSQL_PORT", "4406") or "4406"
    user = _env("MYSQL_USER", "root") or "root"
    password = _env("MYSQL_PASSWORD", "") or ""
    database = _env("MYSQL_DATABASE", "forecast_benchmark") or "forecast_benchmark"
    try:
        port = int(port_raw)
    except Exception:
        port = 4406

    return MySQLConfig(host=host, port=port, user=user, password=password, database=database)


def get_mysql_engine(*, database: str | None = None, echo: bool = False):
    """Return a SQLAlchemy engine for MySQL (requires sqlalchemy + pymysql)."""
    load_env()
    if sa is None:
        raise RuntimeError(
            "MySQL backend requires SQLAlchemy. Install deps: `pip install sqlalchemy pymysql` "
            "or add them to requirements.txt."
        )
    cfg = get_mysql_config()
    if not cfg.password:
        raise RuntimeError(
            "MYSQL_PASSWORD is not set. Create a local .env (ignored by git) and set MYSQL_PASSWORD."
        )
    url = sa.engine.URL.create(
        drivername="mysql+pymysql",
        username=cfg.user,
        password=cfg.password,
        host=cfg.host,
        port=cfg.port,
        database=(database or cfg.database),
    )
    return sa.create_engine(url, echo=echo, pool_pre_ping=True, future=True)


def get_mysql_connection():
    """Return a SQLAlchemy connection to MySQL."""
    engine = get_mysql_engine()
    return engine.connect()


def init_mysql_db() -> None:
    """Create the project's tables in MySQL.

This creates the same logical tables as DuckDB init_db(), plus `item_features`
which is used by the Streamlit app.

Target schema/database is MYSQL_DATABASE (default: forecast_benchmark).
"""
    if sa is None:
        raise RuntimeError("init_mysql_db requires sqlalchemy to be installed")

    engine = get_mysql_engine()
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                """
                CREATE TABLE IF NOT EXISTS sales_history (
                    project   VARCHAR(255) NOT NULL,
                    item_id   VARCHAR(255) NOT NULL,
                    sale_date DATE         NOT NULL,
                    sales     DOUBLE       NULL,
                    INDEX idx_sales_project_item_date (project, item_id, sale_date)
                )
                """
            )
        )

        conn.execute(
            sa.text(
                """
                CREATE TABLE IF NOT EXISTS forecasts (
                    project         VARCHAR(255) NOT NULL,
                    forecast_method VARCHAR(255) NOT NULL,
                    item_id         VARCHAR(255) NOT NULL,
                    forecast_date   DATE         NOT NULL,
                    forecast        DOUBLE       NULL,
                    INDEX idx_fc_project_item_date (project, item_id, forecast_date),
                    INDEX idx_fc_project_method (project, forecast_method)
                )
                """
            )
        )

        conn.execute(
            sa.text(
                """
                CREATE TABLE IF NOT EXISTS forecast_metrics (
                    project         VARCHAR(255) NOT NULL,
                    item_id         VARCHAR(255) NOT NULL,
                    forecast_method VARCHAR(255) NOT NULL,
                    metric_name     VARCHAR(64)  NOT NULL,
                    metric_value    DOUBLE       NULL,
                    n_points        INT          NULL,
                    last_updated    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_fm_project_item (project, item_id),
                    INDEX idx_fm_method (forecast_method)
                )
                """
            )
        )

        conn.execute(
            sa.text(
                """
                CREATE TABLE IF NOT EXISTS item_features (
                    project   VARCHAR(255) NOT NULL,
                    item_id   VARCHAR(255) NOT NULL,
                    name      VARCHAR(255) NULL,
                    item_type VARCHAR(255) NULL,
                    flavour   VARCHAR(255) NULL,
                    size      VARCHAR(255) NULL,
                    UNIQUE KEY uq_item_features (project, item_id)
                )
                """
            )
        )


# -----------------------------
# Public API (existing imports)
# -----------------------------


def get_connection():
    """Return the project's primary DB connection.

Default backend is DuckDB.

If BENCHMARK_DB_BACKEND=mysql is set, this function intentionally raises because
the rest of the codebase expects DuckDB-specific SQL and APIs.
"""
    backend = get_db_backend()
    if backend == DBBackend.DUCKDB:
        return get_duckdb_connection()

    raise RuntimeError(
        "BENCHMARK_DB_BACKEND=mysql is set, but the app currently assumes DuckDB. "
        "Use get_mysql_engine()/get_mysql_connection() in new scripts, or refactor SQL to be MySQL-compatible."
    )


def init_db():
    """Initialize tables in the default DuckDB database."""
    con = get_duckdb_connection()

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS sales_history (
            project         VARCHAR,
            item_id         VARCHAR,
            sale_date       DATE,
            sales           DOUBLE
        );
    """
    )

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS forecasts (
            project          VARCHAR,
            forecast_method  VARCHAR,
            item_id          VARCHAR,
            forecast_date    DATE,
            forecast         DOUBLE
        );
    """
    )

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS forecast_metrics (
            project          VARCHAR,
            item_id          VARCHAR,
            forecast_method  VARCHAR,
            metric_name      VARCHAR,   -- 'MAPE', 'sMAPE', 'WAPE'
            metric_value     DOUBLE,
            n_points         INTEGER,
            last_updated     TIMESTAMP DEFAULT current_timestamp
        );
    """
    )

    con.close()

if __name__ == "__main__":
    init_db()
    print("DB initialized.")
