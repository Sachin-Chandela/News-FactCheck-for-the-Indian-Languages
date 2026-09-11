#!/usr/bin/env bash
# project1/scripts/build_index.sh
# Encode all claims and build the FAISS retriever index.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Project 1 — Build FAISS Retriever"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

DATA="$ROOT/data/final_dataset.csv"
[ -f "$DATA" ] || { echo "❌ $DATA not found. Run preprocess.sh first."; exit 1; }

mkdir -p "$ROOT/models"
cd "$ROOT"
python project1/src/build_retriever.py

echo "✅ Index saved: models/fact_index.faiss + models/claims.pkl"
