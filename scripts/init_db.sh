#!/usr/bin/env bash
set -euo pipefail

docker exec -i gary-postgres psql -U gary -d gary_db < src/gary/sql/001_init.sql
