"""One-time migration: DuckDB -> MySQL.

Copies the core tables from the project's DuckDB database into the MySQL schema
used by app_mysql.py.

Default behavior TRUNCATEs the target MySQL tables before inserting, to avoid
creating duplicates if you run the script more than once.

Usage (local MySQL):
  export MYSQL_HOST=127.0.0.1
  export MYSQL_PORT=4406
  export MYSQL_USER=root
  export MYSQL_PASSWORD=...  # required
  export MYSQL_DATABASE=forecast_benchmark

  python migrate_duckdb_to_mysql.py --duckdb-path ./sog.duckdb

Notes:
- If DuckDB doesn't contain item_features, that table is skipped.
- Requires: sqlalchemy, pymysql (already in requirements.txt)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Iterable

import sqlalchemy as sa

from db import get_connection_for_path, get_duckdb_config, get_mysql_engine, init_mysql_db


@dataclass(frozen=True)
class TableSpec:
    name: str
    # Preferred column order for export/import.
    columns: list[str]
    # Columns that should be treated as DATE in MySQL.
    date_columns: set[str]
    # Columns that must be non-null for MySQL inserts.
    required_columns: set[str]
    # Columns to omit from insert (e.g., MySQL defaults).
    optional_columns: set[str]


TABLES: list[TableSpec] = [
    TableSpec(
        name="sales_history",
        columns=["project", "item_id", "sale_date", "sales"],
        date_columns={"sale_date"},
        required_columns={"project", "item_id", "sale_date"},
        optional_columns=set(),
    ),
    TableSpec(
        name="forecasts",
        columns=["project", "forecast_method", "item_id", "forecast_date", "forecast"],
        date_columns={"forecast_date"},
        required_columns={"project", "forecast_method", "item_id", "forecast_date"},
        optional_columns=set(),
    ),
    TableSpec(
        name="forecast_metrics",
        columns=[
            "project",
            "item_id",
            "forecast_method",
            "metric_name",
            "metric_value",
            "n_points",
            "last_updated",
        ],
        date_columns=set(),
        required_columns={"project", "item_id", "forecast_method", "metric_name"},
        optional_columns={"last_updated"},
    ),
    TableSpec(
        name="item_features",
        columns=["project", "item_id", "name", "item_type", "flavour", "size"],
        date_columns=set(),
        required_columns={"project", "item_id"},
        optional_columns=set(),
    ),
]


def _duckdb_table_columns(con, table: str) -> set[str]:
    try:
        rows = con.execute(f"PRAGMA table_info('{table}')").fetchall()
    except Exception:
        return set()
    # pragma table_info: cid, name, type, notnull, dflt_value, pk
    cols: set[str] = set()
    for r in rows:
        try:
            cols.add(str(r[1]))
        except Exception:
            continue
    return cols


def _normalize_value(value: Any, *, column: str, date_columns: set[str]) -> Any:
    if value is None:
        return None

    if column in date_columns:
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        if isinstance(value, datetime):
            return value.date()
        # Attempt to parse ISO-ish strings.
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
            except Exception:
                return value

    # Keep timestamps as datetimes where possible.
    if isinstance(value, datetime):
        return value

    # DuckDB may hand back Decimal-like numbers; MySQL DOUBLE can accept float.
    if isinstance(value, (int, float, str)):
        return value

    # Fallback: let SQLAlchemy/dbapi attempt conversion.
    return value


def _iter_duckdb_rows(con, sql: str, *, chunk_size: int) -> Iterable[tuple[list[str], list[tuple[Any, ...]]]]:
    cur = con.execute(sql)
    cols = [d[0] for d in (cur.description or [])]
    while True:
        batch = cur.fetchmany(chunk_size)
        if not batch:
            break
        yield cols, batch


def _truncate_mysql_table(conn: sa.Connection, table: str) -> None:
    conn.execute(sa.text(f"TRUNCATE TABLE {table}"))


def _mysql_insert_many(
    conn: sa.Connection,
    *,
    table: str,
    columns: list[str],
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return
    cols_sql = ", ".join(columns)
    params_sql = ", ".join([f":{c}" for c in columns])
    stmt = sa.text(f"INSERT INTO {table} ({cols_sql}) VALUES ({params_sql})")
    conn.execute(stmt, rows)


def _count_duckdb_rows(con, table: str) -> int:
    try:
        return int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    except Exception:
        return 0


def _count_mysql_rows(conn: sa.Connection, table: str) -> int:
    try:
        return int(conn.execute(sa.text(f"SELECT COUNT(*) FROM {table}")).scalar_one())
    except Exception:
        return 0


def migrate(
    *,
    duckdb_path: str,
    truncate: bool,
    chunk_size: int,
    tables: list[str] | None,
    limit: int | None,
) -> None:
    init_mysql_db()
    engine = get_mysql_engine()

    duck = get_connection_for_path(duckdb_path, read_only=True)

    try:
        available_tables = {r[0] for r in duck.execute("SHOW TABLES").fetchall()}
    except Exception:
        available_tables = set()

    selected_specs = [t for t in TABLES if (tables is None or t.name in tables)]

    print(f"DuckDB: {duckdb_path}")
    print("MySQL: connected")

    with engine.begin() as conn:
        for spec in selected_specs:
            if spec.name not in available_tables:
                # Some legacy DuckDB files won't have item_features.
                print(f"- {spec.name}: skipped (not present in DuckDB)")
                continue

            duck_cols = _duckdb_table_columns(duck, spec.name)
            if not duck_cols:
                print(f"- {spec.name}: skipped (could not read schema)")
                continue

            export_cols = [c for c in spec.columns if c in duck_cols and c not in spec.optional_columns]
            # If an optional column exists, include it.
            for c in spec.optional_columns:
                if c in duck_cols:
                    export_cols.append(c)

            if not export_cols:
                print(f"- {spec.name}: skipped (no matching columns)")
                continue

            duck_count = _count_duckdb_rows(duck, spec.name)
            print(f"- {spec.name}: exporting {duck_count} rows")

            if truncate:
                _truncate_mysql_table(conn, spec.name)

            sql = f"SELECT {', '.join(export_cols)} FROM {spec.name}"
            # Filter out rows that would violate MySQL NOT NULL constraints.
            req = [c for c in spec.required_columns if c in export_cols]
            if req:
                where = " AND ".join([f"{c} IS NOT NULL" for c in req])
                sql += f" WHERE {where}"
            if limit is not None and int(limit) > 0:
                sql += f" LIMIT {int(limit)}"
            inserted = 0

            for cols, batch in _iter_duckdb_rows(duck, sql, chunk_size=chunk_size):
                out_rows: list[dict[str, Any]] = []
                for tup in batch:
                    row: dict[str, Any] = {}
                    for c, v in zip(cols, tup):
                        row[c] = _normalize_value(v, column=c, date_columns=spec.date_columns)
                    # Defensive: skip rows that still violate required columns
                    # after normalization.
                    if any(row.get(rc) is None for rc in spec.required_columns if rc in cols):
                        continue
                    out_rows.append(row)

                _mysql_insert_many(conn, table=spec.name, columns=cols, rows=out_rows)
                inserted += len(out_rows)

            mysql_count = _count_mysql_rows(conn, spec.name)
            print(f"  inserted: {inserted} (mysql now has {mysql_count} rows)")

    duck.close()


def main() -> None:
    default_duck = str(get_duckdb_config().path)

    p = argparse.ArgumentParser(description="Migrate forecast_benchmark data from DuckDB to MySQL")
    p.add_argument("--duckdb-path", default=default_duck, help=f"Path to DuckDB file (default: {default_duck})")
    p.add_argument(
        "--no-truncate",
        action="store_true",
        help="Do not truncate MySQL tables before inserting (may create duplicates)",
    )
    p.add_argument("--chunk-size", type=int, default=10_000, help="Rows per batch insert")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max rows per table to copy (for quick validation)",
    )
    p.add_argument(
        "--tables",
        nargs="*",
        default=None,
        help="Optional subset of tables to migrate (e.g., sales_history forecasts)",
    )

    args = p.parse_args()

    migrate(
        duckdb_path=str(args.duckdb_path),
        truncate=(not bool(args.no_truncate)),
        chunk_size=int(args.chunk_size),
        tables=args.tables,
        limit=(None if args.limit is None else int(args.limit)),
    )


if __name__ == "__main__":
    main()
