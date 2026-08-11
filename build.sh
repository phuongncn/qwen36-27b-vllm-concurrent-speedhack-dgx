#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_IMAGE="${BASE_IMAGE:-vllm-sm121:latest}"
V3_IMAGE="${V3_IMAGE:-vllm-qwen35-v3:latest}"
WINNER_IMAGE="${WINNER_IMAGE:-vllm-qwen35-frspec-x16-v1:latest}"
SKIP_V3="${SKIP_V3:-0}"

case "$SKIP_V3" in 0|1) ;; *) echo "SKIP_V3 must be 0 or 1" >&2; exit 2 ;; esac

if [ "$SKIP_V3" = 0 ]; then
    docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || {
        echo "Missing SM121 vLLM base image: $BASE_IMAGE" >&2
        echo "Build one first; see README.md -> Prerequisites." >&2
        exit 1
    }
    docker build \
        --build-arg "VLLM_BASE=$BASE_IMAGE" \
        -t "$V3_IMAGE" \
        -f "$ROOT_DIR/docker/Dockerfile.v3" \
        "$ROOT_DIR"
else
    docker image inspect "$V3_IMAGE" >/dev/null 2>&1 || {
        echo "SKIP_V3=1 but V3_IMAGE is missing: $V3_IMAGE" >&2
        exit 1
    }
fi

docker build \
    --build-arg "VLLM_BASE=$V3_IMAGE" \
    -t "$WINNER_IMAGE" \
    -f "$ROOT_DIR/docker/Dockerfile" \
    "$ROOT_DIR"

echo "Built: $WINNER_IMAGE"
echo "Run: MODEL_DIR=/path/to/model $ROOT_DIR/run.sh agentbulk"
