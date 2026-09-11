#!/usr/bin/env bash
# project2/scripts/build_index.sh
# Download PubHealth dataset and build the medical FAISS index.
# Runtime: ~5–15 minutes on GPU, ~40–60 minutes on CPU.
#
# Usage: bash project2/scripts/build_index.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Project 2 — Build PubHealth Medical FAISS Index"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ⚠️  Requires: HuggingFace login + internet access"
echo "  ⚠️  Will download health_fact dataset (~200 MB)"
echo ""

mkdir -p "$ROOT/index"
cd "$ROOT"
python project2/src/medical_indexer.py

echo ""
echo "✅ Index saved:"
echo "   index/health_claims.index"
echo "   index/mapping.json"
