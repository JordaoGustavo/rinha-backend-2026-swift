#!/usr/bin/env bash
set -euo pipefail
BASE_URL="https://raw.githubusercontent.com/zanfranceschi/rinha-de-backend-2026/main/resources"
mkdir -p resources
curl -fsSL "$BASE_URL/references.json.gz" -o resources/references.json.gz
curl -fsSL "$BASE_URL/mcc_risk.json"      -o resources/mcc_risk.json
curl -fsSL "$BASE_URL/normalization.json"  -o resources/normalization.json
echo "Resources downloaded."
