#!/usr/bin/env bash
# project1/scripts/train.sh
# Fine-tune MuRIL on [claim][SEP][evidence] pairs with weighted CE loss.
#
# Usage: bash project1/scripts/train.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Project 1 — Train MuRIL Classifier"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

DATA="$ROOT/data/final_dataset.csv"
[ -f "$DATA" ] || { echo "❌ $DATA not found. Run preprocess.sh first."; exit 1; }

mkdir -p "$ROOT/models/muril"
cd "$ROOT"
python project1/src/train_classifier.py

echo ""
echo "✅ Model saved: models/muril/"
