#!/usr/bin/env bash
set -euo pipefail

API_BASE_URL="${1:-http://127.0.0.1:8000}"

echo "Health check..."
curl -fsS "$API_BASE_URL/health"
echo

echo "Readiness check..."
curl -fsS "$API_BASE_URL/ready"
echo

echo "Smoke checks passed (health + ready)."
