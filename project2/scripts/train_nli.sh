#!/usr/bin/env bash
# project2/scripts/train_nli.sh
# Fine-tune RoBERTa and PubMedBERT on SciFact for NLI-based medical fact verification.
#
# Usage: bash project2/scripts/train_nli.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Project 2 — Fine-tune NLI Models on SciFact"
echo "  Models: roberta-base + PubMedBERT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

SCIFACT="$ROOT/data/SciFact"
for f in claims_train.csv claims_validation.csv corpus_train.csv; do
    [ -f "$SCIFACT/$f" ] || {
        echo "❌ Missing: $SCIFACT/$f"
        echo "   Place SciFact CSV files in data/SciFact/ before running."
        exit 1
    }
done

echo "✅ SciFact data found."
echo ""
echo "🚀 Training roberta-base..."
echo "   → will save to: models/roberta_medical/"
echo ""
echo "🚀 Training PubMedBERT..."
echo "   → will save to: models/pubmed_medical/"
echo ""

mkdir -p "$ROOT/models/roberta_medical" "$ROOT/models/pubmed_medical"
cd "$ROOT"
python project2/src/finetune.py

echo ""
echo "✅ Fine-tuning complete."
echo "   models/roberta_medical/"
echo "   models/pubmed_medical/"
