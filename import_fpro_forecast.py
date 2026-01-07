from __future__ import annotations

import argparse
from pathlib import Path

from db import init_db
from ingest import import_forecasts_horizontal_months


def main() -> None:
    p = argparse.ArgumentParser(description="One-time import of a horizontal monthly forecast file into forecasts")
    p.add_argument("--csv-path", required=True, help="Path to the semicolon-delimited forecast CSV")
    p.add_argument("--project", default="Kjoris", help="Project name to write into forecasts")
    p.add_argument("--forecast-method", default="FPRO", help="forecast_method value to set")
    p.add_argument(
        "--no-overwrite",
        action="store_true",
        help="If set, do not delete existing rows in the imported date range",
    )

    args = p.parse_args()

    csv_path = Path(args.csv_path).expanduser().resolve()
    if not csv_path.exists():
        raise SystemExit(f"File not found: {csv_path}")

    init_db()

    data = csv_path.read_bytes()
    stats = import_forecasts_horizontal_months(
        data,
        project=args.project,
        forecast_method=args.forecast_method,
        overwrite_project_date_range=(not args.no_overwrite),
    )

    print("Import complete:")
    for k in sorted(stats.keys()):
        print(f"- {k}: {stats[k]}")


if __name__ == "__main__":
    main()
