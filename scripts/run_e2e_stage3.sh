#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT/.env"
OUTPUT_DIR="${E2E_OUTPUT_DIR:-$ROOT/execution_histories/e2e_smoke}"

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

mkdir -p "$OUTPUT_DIR"
cd "$ROOT"

python scripts/run_e2e_smoke.py
