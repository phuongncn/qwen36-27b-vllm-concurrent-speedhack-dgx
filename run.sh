#!/usr/bin/env bash
# Fast ThinkingCap Qwen3.6-27B AutoRound launcher for NVIDIA GB10 / DGX Spark.
# 2026-08-11 winner: FR-Spec 98k + x16 LM head + MTP3 + prefix-cache align.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${MODEL_DIR:-$SCRIPT_DIR/ThinkingCap-Qwen3.6-27B-int4-AutoRound-v1}"
IMAGE="${VLLM_IMAGE:-vllm-qwen35-frspec-x16-v1:latest}"
PORT="${PORT:-8000}"
CONTAINER="vllm-thinkingcap-$PORT"
SERVED_NAME="ThinkingCap-Qwen3.6-27B-int4-AutoRound-v1"
VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-$HOME/.cache/vllm}"

usage() {
    echo "Usage: $0 [agentcode|agentbulk|status|logs|stop]"
    echo "  agentcode  256k context, one active sequence, lowest interactive latency"
    echo "  agentbulk  131k context, up to 32 active sequences, maximum throughput"
    echo "Environment overrides: PORT, MODEL_DIR, VLLM_IMAGE, THINKING=true|false"
    echo "Experimental: DGX_SPARK_DRAFT_VOCAB, MTP_TOKENS, LONG_PREFILL_TOKEN_THRESHOLD"
}

ACTION="${1:-}"
if [ -z "$ACTION" ]; then
    echo ""
    echo "ThinkingCap Qwen3.6-27B AutoRound GB10 optimized"
    echo "  1. agentcode — 256k context, c1 latency"
    echo "  2. agentbulk — 131k context, up to c32"
    echo "  3. status"
    echo "  4. logs"
    echo "  5. stop"
    echo ""
    read -r -p "Select [1-5]: " choice
    case "$choice" in
        1) ACTION=agentcode ;;
        2) ACTION=agentbulk ;;
        3) ACTION=status ;;
        4) ACTION=logs ;;
        5) ACTION=stop ;;
        *) echo "Invalid selection" >&2; exit 2 ;;
    esac
fi

case "$ACTION" in
    status)
        docker ps -a --filter "name=^${CONTAINER}$" \
            --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
        exit 0
        ;;
    logs)
        docker logs --tail 100 -f "$CONTAINER"
        exit 0
        ;;
    stop)
        if docker ps -aq --filter "name=^${CONTAINER}$" | grep -q .; then
            docker rm -f "$CONTAINER" >/dev/null
            echo "Stopped $CONTAINER"
        else
            echo "$CONTAINER is not present"
        fi
        exit 0
        ;;
    agentcode)
        PROFILE=agentcode
        MAX_MODEL_LEN=262144
        MAX_NUM_SEQS=1
        MAX_BATCHED_TOKENS=4096
        GPU_MEMORY_UTILIZATION=0.80
        ;;
    agentbulk)
        PROFILE=agentbulk
        MAX_MODEL_LEN=131072
        MAX_NUM_SEQS=32
        MAX_BATCHED_TOKENS=32768
        GPU_MEMORY_UTILIZATION=0.80
        ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

[ -f "$MODEL_DIR/config.json" ] || {
    echo "Missing model config: $MODEL_DIR/config.json" >&2
    exit 1
}
[ -f "$MODEL_DIR/model.safetensors.index.json" ] || {
    echo "Missing model weights/index in: $MODEL_DIR" >&2
    exit 1
}
docker image inspect "$IMAGE" >/dev/null 2>&1 || {
    echo "Missing Docker image: $IMAGE" >&2
    exit 1
}

THINKING="${THINKING:-true}"
case "$THINKING" in
    true|false) ;;
    *) echo "THINKING must be true or false" >&2; exit 2 ;;
esac

# The production winner uses a 98k+tail draft-only vocabulary subset.  The full
# target LM head is unchanged, and setting VLLM_IMAGE to the V3 fallback simply
# makes these environment variables inert.
DGX_SPARK_DRAFT_VOCAB="${DGX_SPARK_DRAFT_VOCAB:-98304}"
case "$DGX_SPARK_DRAFT_VOCAB" in
    ''|*[!0-9]*) echo "DGX_SPARK_DRAFT_VOCAB must be a non-negative integer" >&2; exit 2 ;;
esac
if [ "$DGX_SPARK_DRAFT_VOCAB" -gt 248320 ]; then
    echo "DGX_SPARK_DRAFT_VOCAB must be <= 248320" >&2
    exit 2
fi
DGX_SPARK_DRAFT_VOCAB_TAIL="${DGX_SPARK_DRAFT_VOCAB_TAIL:-512}"
case "$DGX_SPARK_DRAFT_VOCAB_TAIL" in
    ''|*[!0-9]*) echo "DGX_SPARK_DRAFT_VOCAB_TAIL must be a non-negative integer" >&2; exit 2 ;;
esac
if [ "$DGX_SPARK_DRAFT_VOCAB_TAIL" -gt 248320 ]; then
    echo "DGX_SPARK_DRAFT_VOCAB_TAIL must be <= 248320" >&2
    exit 2
fi
DOCKER_EXPERIMENT_ENV=()
if [ -n "$DGX_SPARK_DRAFT_VOCAB" ]; then
    DOCKER_EXPERIMENT_ENV+=(
        -e "DGX_SPARK_DRAFT_VOCAB=$DGX_SPARK_DRAFT_VOCAB"
        -e "DGX_SPARK_DRAFT_VOCAB_TAIL=$DGX_SPARK_DRAFT_VOCAB_TAIL"
    )
fi

VLLM_EXPERIMENT_ARGS=()
if [ -n "${LONG_PREFILL_TOKEN_THRESHOLD:-}" ]; then
    case "$LONG_PREFILL_TOKEN_THRESHOLD" in
        ''|*[!0-9]*) echo "LONG_PREFILL_TOKEN_THRESHOLD must be a non-negative integer" >&2; exit 2 ;;
        *) VLLM_EXPERIMENT_ARGS+=(--long-prefill-token-threshold "$LONG_PREFILL_TOKEN_THRESHOLD") ;;
    esac
fi

PREFIX_CACHING="${PREFIX_CACHING:-true}"
case "$PREFIX_CACHING" in
    true)
        # Qwen3.5's hybrid GDN/attention stack supports prefix caching only in
        # align mode; the model explicitly rejects Mamba cache mode "all".
        VLLM_EXPERIMENT_ARGS+=(--enable-prefix-caching --mamba-cache-mode align)
        ;;
    false) ;;
    *) echo "PREFIX_CACHING must be true or false" >&2; exit 2 ;;
esac

MTP_TOKENS="${MTP_TOKENS:-3}"
case "$MTP_TOKENS" in
    1|2|3|4|5) ;;
    *) echo "MTP_TOKENS must be one of 1,2,3,4,5" >&2; exit 2 ;;
esac

if docker ps -aq --filter "name=^${CONTAINER}$" | grep -q .; then
    echo "Replacing existing $CONTAINER"
    docker rm -f "$CONTAINER" >/dev/null
fi

mkdir -p "$VLLM_CACHE_DIR" "$HOME/.cache/huggingface"

echo "Starting $PROFILE on port $PORT"
echo "  model=$MODEL_DIR"
echo "  context=$MAX_MODEL_LEN max_seqs=$MAX_NUM_SEQS batched_tokens=$MAX_BATCHED_TOKENS"
echo "  MTP=$MTP_TOKENS attention=TRITON_ATTN KV=FP8 vision=on image=$IMAGE"
echo "  prefix_cache=$PREFIX_CACHING long_prefill_threshold=${LONG_PREFILL_TOKEN_THRESHOLD:-default}"

docker run -d \
    --name "$CONTAINER" \
    --gpus all \
    --net=host \
    --ipc=host \
    --shm-size=16g \
    -v "$MODEL_DIR:/model:ro" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -v "$VLLM_CACHE_DIR:/root/.cache/vllm" \
    -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
    "${DOCKER_EXPERIMENT_ENV[@]}" \
    "$IMAGE" serve /model \
    --served-model-name "$SERVED_NAME" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --tensor-parallel-size 1 \
    --quantization autoround \
    --attention-backend TRITON_ATTN \
    --kv-cache-dtype fp8 \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
    --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_TOKENS}" \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --default-chat-template-kwargs "{\"enable_thinking\":$THINKING}" \
    --trust-remote-code \
    --block-size 128 \
    --load-format instanttensor \
    "${VLLM_EXPERIMENT_ARGS[@]}" \
    -O3 >/dev/null

echo -n "Waiting for health"
deadline=$((SECONDS + 900))
while ! curl -sf --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null; do
    if ! docker ps -q --filter "name=^${CONTAINER}$" | grep -q .; then
        echo ""
        echo "Container exited during startup" >&2
        docker logs "$CONTAINER" --tail 100 >&2 || true
        exit 1
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
        echo ""
        echo "Timed out waiting for server" >&2
        docker logs "$CONTAINER" --tail 100 >&2 || true
        exit 1
    fi
    echo -n "."
    sleep 5
done

echo ""
echo "Ready: http://127.0.0.1:$PORT/v1 ($PROFILE)"
echo "Benchmark: python3 $SCRIPT_DIR/benchmarks/ab_benchmark.py --label live --output $SCRIPT_DIR/results/live.json --concurrency 1,4,8 --repeats 3"
