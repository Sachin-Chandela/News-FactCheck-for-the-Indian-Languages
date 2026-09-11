#!/usr/bin/env bash
# project2/scripts/eval.sh
# Run the 4-model ablation benchmark.
#
# Modes:
#   --mode simple   Uses english_reference field from test_set.json
#                   (no Llama needed — fast, ~5 min)
#   --mode full     Runs live Llama-3.2 translation for each entry
#                   (requires ~8 GB VRAM, ~30–60 min depending on test size)
#
# Usage:
#   bash project2/scripts/eval.sh --mode simple
#   bash project2/scripts/eval.sh --mode full

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MODE="simple"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ "$MODE" != "simple" && "$MODE" != "full" ]]; then
    echo "❌ Invalid mode: $MODE. Use 'simple' or 'full'."
    exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Project 2 — Ablation Benchmark Evaluation"
echo "  Mode: $MODE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

TEST="$ROOT/data/test_set.json"
[ -f "$TEST" ] || { echo "❌ Test set not found: $TEST"; exit 1; }

# Check fine-tuned models exist
for m in roberta_medical pubmed_medical; do
    [ -f "$ROOT/models/$m/config.json" ] || {
        echo "⚠️  Warning: models/$m/ not found."
        echo "   Run train_nli.sh to fine-tune before evaluating fine-tuned configs."
    }
done

cd "$ROOT"

if [ "$MODE" = "full" ]; then
    echo "🌍 Full pipeline evaluation (live Llama-3.2 translation)..."
    echo "   Note: claims are pre-translated once to avoid reloading Llama."
    python project2/src/final_pipeline_evaluator.py
else
    echo "🧪 Simple evaluation (english_reference from test_set.json)..."
    python project2/src/research_evaluator.py
fi

echo ""
echo "✅ Evaluation complete."
