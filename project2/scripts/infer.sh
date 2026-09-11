#!/usr/bin/env bash
# project2/scripts/infer.sh
# Run the full 3-stage ShieldHealth pipeline on a single Hinglish claim.
#
# Stages: Llama translation → FAISS retrieval → PubMedBERT NLI verdict
#
# Usage:
#   bash project2/scripts/infer.sh                               # interactive REPL
#   bash project2/scripts/infer.sh --claim "Haldi se cancer thik hota hai"

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CLAIM=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --claim) CLAIM="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Project 2 — ShieldHealth Claim Inference"
echo "  Pipeline: Llama → FAISS → PubMedBERT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

INDEX="$ROOT/index/health_claims.index"
[ -f "$INDEX" ] || {
    echo "❌ Medical index not found: $INDEX"
    echo "   Run: bash project2/scripts/build_index.sh"
    exit 1
}

cd "$ROOT"

if [ -n "$CLAIM" ]; then
    python - <<EOF
from project2.src.integrated_checker import MedicalFactChecker

checker = MedicalFactChecker()
claim = "$CLAIM"

eng = checker.translate_hinglish(claim)
print(f"\n✨ English: {eng}")

matches, scores = checker.retrieve_evidence(eng, k=3)
print("\n📚 Top 3 Matches:")
for i, (m, s) in enumerate(zip(matches, scores)):
    verdict = checker.get_verdict(eng, m['explanation'])
    print(f"  [{i+1}] Score={s:.3f} | Verdict: {verdict}")
    print(f"       Claim: {m['claim'][:80]}...")
EOF
else
    python project2/src/integrated_checker.py
fi
