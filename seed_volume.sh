#!/usr/bin/env bash
set -euo pipefail

# Seed the named docker volume (duckdb_workspace) with ./sog.duckdb
# This is useful on a fresh machine where the volume exists but is empty.

if [[ ! -f "sog.duckdb" ]]; then
  echo "ERROR: expected ./sog.duckdb in $(pwd)"
  exit 2
fi

docker run --rm \
  -v duckdb_workspace:/duckdb \
  -v "$PWD":/seed:ro \
  alpine:3.20 sh -lc '
    if [ -f /duckdb/sog.duckdb ]; then
      echo "duckdb_workspace already contains /duckdb/sog.duckdb"
    else
      cp /seed/sog.duckdb /duckdb/sog.duckdb && echo "Seeded /duckdb/sog.duckdb into duckdb_workspace"
    fi
  '
