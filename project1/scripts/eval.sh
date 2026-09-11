#!/usr/bin/env bash
# project1/scripts/eval.sh
# Evaluate MuRIL on the validation split (runs automatically after training).
# For a custom held-out test set, pass --test_json path/to/test.json
#
# Usage:
#   bash project1/scripts/eval.sh
#   bash project1/scripts/eval.sh --test_json data/my_test.json

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

TEST_JSON=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --test_json) TEST_JSON="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Project 1 — Evaluation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

MODEL="$ROOT/models/muril/config.json"
[ -f "$MODEL" ] || { echo "❌ MuRIL model not found. Run train.sh first."; exit 1; }

cd "$ROOT"

if [ -n "$TEST_JSON" ]; then
    echo "📋 Evaluating on custom test set: $TEST_JSON"
    python - <<EOF
import json, torch
from sklearn.metrics import classification_report
from project1.src.inference import predict_claim

with open("$TEST_JSON") as f:
    data = json.load(f)

y_true, y_pred = [], []
label_map = {"true": 0, "false": 1, "misleading": 2, "other": 3}

for entry in data:
    result = predict_claim(entry["claim"])
    y_pred.append(label_map.get(result["verdict"], 3))
    y_true.append(entry["label"])

print(classification_report(y_true, y_pred,
    target_names=["true","false","misleading","other"], zero_division=0))
EOF
else
    echo "📋 Re-running training eval (prints val metrics from last checkpoint)..."
    python project1/src/train_classifier.py
fi
