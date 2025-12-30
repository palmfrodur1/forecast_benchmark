#!/usr/bin/env bash
set -euo pipefail

# Runs the Streamlit app against the DuckDB file inside the named Docker volume.
# Assumes the volume is mounted at /duckdb inside the container.

docker run --rm \
  -p 8501:8501 \
  -v duckdb_workspace:/duckdb \
  -v "$PWD":/app \
  -w /app \
  -e BENCHMARK_DB_PATH=/duckdb/sog.duckdb \
  python:3.13-slim sh -lc "pip -q install -r requirements.txt && streamlit run app.py --server.address=0.0.0.0 --server.port=8501"
