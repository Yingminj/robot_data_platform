#!/usr/bin/env bash
set -Eeuo pipefail

psql --set ON_ERROR_STOP=1 --username "${POSTGRES_USER}" --dbname "${POSTGRES_DB}" <<SQL
SELECT 'CREATE DATABASE ${MLFLOW_DB} OWNER ${POSTGRES_USER}'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '${MLFLOW_DB}')\gexec
SQL

