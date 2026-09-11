#!/usr/bin/env bash
# project2/scripts/demo.sh
# Launch the ShieldHealth Gradio web demo.
#
# Usage:
#   bash project2/scripts/demo.sh                 # local at :7860
#   bash project2/scripts/demo.sh --share         # public Gradio tunnel
#   bash project2/scripts/demo.sh --port 8080     # custom port

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SHARE=false
PORT=7860
while [[ $# -gt 0 ]]; do
    case "$1" in
        --share) SHARE=true; shift ;;
        --port)  PORT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Project 2 — ShieldHealth Gradio Demo"
echo "  URL : http://127.0.0.1:$PORT"
[ "$SHARE" = true ] && echo "  Public sharing : ENABLED"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

INDEX="$ROOT/index/health_claims.index"
[ -f "$INDEX" ] || {
    echo "⚠️  Warning: $INDEX not found."
    echo "   Run build_index.sh first. The UI will error on first query."
}

cd "$ROOT"
GRADIO_SERVER_PORT="$PORT" GRADIO_SHARE="$SHARE" python project2/src/GUI.py

