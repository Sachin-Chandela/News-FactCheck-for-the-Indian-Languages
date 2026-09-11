#!/usr/bin/env bash
# project1/scripts/preprocess.sh
# Merge raw CSVs, normalise labels, and save final_dataset.csv

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Project 1 — Preprocess Data"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for f in "$ROOT/data/facts_merged_final.csv" "$ROOT/data/factcheck_dataset_modified.csv"; do
    [ -f "$f" ] || { echo "❌ Missing: $f"; exit 1; }
done

cd "$ROOT"
python project1/src/preprocess.py

echo "✅ Saved: data/final_dataset.csv"
