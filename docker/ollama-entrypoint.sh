#!/bin/sh
set -eu

MODEL_NAME="${RAG_OLLAMA_MODEL:-qwen3.5:4b}"

ollama serve &
OLLAMA_PID=$!

cleanup() {
  kill "$OLLAMA_PID" 2>/dev/null || true
  wait "$OLLAMA_PID" 2>/dev/null || true
}

trap cleanup INT TERM EXIT

READY=0
for _ in $(seq 1 90); do
  if ollama list >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 1
done

if [ "$READY" -ne 1 ]; then
  echo "Ollama service did not become ready in time" >&2
  exit 1
fi

ollama pull "$MODEL_NAME"

wait "$OLLAMA_PID"
