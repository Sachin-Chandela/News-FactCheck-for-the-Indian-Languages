#!/usr/bin/env bash
# project1/scripts/infer.sh
# Run claim inference using the full MuRIL + NLI fusion pipeline.
#
# Usage:
#   bash project1/scripts/infer.sh                          # interactive REPL
#   bash project1/scripts/infer.sh --claim "your claim"     # single JSON output

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CLAIM=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --claim) CLAIM="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Project 1 — Claim Inference"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for f in "$ROOT/models/muril/config.json" "$ROOT/models/fact_index.faiss" "$ROOT/models/claims.pkl"; do
    [ -f "$f" ] || { echo "❌ Missing: $f"; echo "   Run train.sh and build_index.sh first."; exit 1; }
done

cd "$ROOT"

if [ -n "$CLAIM" ]; then
    python - <<EOF
import json
from project1.src.inference import predict_claim
result = predict_claim("$CLAIM")
print(json.dumps(result, indent=2, ensure_ascii=False))
EOF
else
    python project1/src/inference.py
fi
