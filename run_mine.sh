#!/usr/bin/env bash
set -euo pipefail

export VLLM_CACHE_ROOT=/gscratch/xlab/tsor13/.cache/vllm
# VLLM_CACHE_ROOT=$HOME/.cache/vllm
export VLLM_PORT=8011


# Port for vLLM (reused sequentially across runs)
VLLM_PORT="${VLLM_PORT:-8010}"

# Models and splits to iterate
MODELS=(
  # "google/gemma-3-1b-it"
  # "google/gemma-3-4b-it"
  # "google/gemma-3-12b-it"
  # "google/gemma-3-27b-it"

  # "tsor13/chatv1"
  # "tsor13/explicit-it-v1"
  "tsor13/chatv1"
  "tsor13/explicitv1"
)
SPLITS=("curated" "wildchat")

# --- helpers ---------------------------------------------------------------

VLLM_PID=""

stop_vllm() {
  if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    kill "$VLLM_PID" || true
    # give it a moment to release the port
    wait "$VLLM_PID" 2>/dev/null || true
    VLLM_PID=""
  fi
}

start_vllm() {
  local model="$1"
  local port="$2"
  local served_name="$3"
  local log_file="$4"

  # Start server
  uv run vllm serve "$model" \
    --port "$port" \
    --served-model-name "$served_name" \
    --trust-remote-code \
    >"$log_file" 2>&1 &
  VLLM_PID=$!

  # Wait for readiness (poll /v1/models)
  echo "Waiting for vLLM ($served_name) to become ready on :$port ..."
  for _ in {1..120}; do
    if curl -s "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "vLLM ready."
      return 0
    fi
    # bail early if process died
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "vLLM exited early; see log: $log_file"
      return 1
    fi
    sleep 1
  done
  echo "Timed out waiting for vLLM to start; see log: $log_file"
  return 1
}

run_one() {
  local model="$1"
  local split="$2"

  # Use the basename as a nice alias (e.g., gemma-3-12b-it)
  local alias="${model##*/}"

  local eval_dir="results/${split}/${alias}"
  local log_dir="logs/${split}/${alias}"
  mkdir -p "$eval_dir" "$log_dir"

  echo
  echo "===================="
  echo "Model: $model"
  echo "Split: $split"
  echo "Eval:  $eval_dir"
  echo "Logs:  $log_dir"
  echo "===================="

    # --- vLLM phase ---
    start_vllm "$model" "$VLLM_PORT" "$alias" "$log_dir/vllm.log"

    # vLLM inference (IMPORTANT: use served-model-name = $alias)
    echo "[inference:vllm] $alias | $split"
    uv run python src/inference.py \
    --mode vllm \
    --model "$alias" \
    --data "$split" \
    --eval-dir "$eval_dir" \
    --sampling regenerate \
    --num-generations 10 \
    | tee "$log_dir/inference_vllm.log" >/dev/null

  # # --- vLLM phase ---
  # start_vllm "$model" "$VLLM_PORT" "$alias" "$log_dir/vllm.log"

  # # vLLM inference
  # echo "[inference:vllm] $model | $split"
  # uv run python src/inference.py \
  #   --mode vllm \
  #   --model "$model" \
  #   --data "$split" \
  #   --eval-dir "$eval_dir" \
  #   --sampling regenerate \
  #   --num-generations 10 \
  #   | tee "$log_dir/inference_vllm.log" >/dev/null

  # Stop vLLM before transformers mode
  stop_vllm
  sleep 2

  # --- transformers phase ---
  echo "[inference:transformers] $model | $split"
  uv run python src/inference.py \
    --mode transformers \
    --model "$model" \
    --data "$split" \
    --eval-dir "$eval_dir" \
    --sampling regenerate \
    --num-generations 10 \
    | tee "$log_dir/inference_transformers.log" >/dev/null

  # --- post-processing ---
  echo "[partition] $model | $split"
  uv run python src/partition.py \
    --eval-dir "$eval_dir" \
    --alg classifier \
    | tee "$log_dir/partition.log" >/dev/null

  echo "[score] $model | $split"
  uv run python src/score.py \
    --eval-dir "$eval_dir" \
    --patience 0.8 \
    | tee "$log_dir/score.log" >/dev/null

  echo "[summarize] $model | $split"
  uv run python src/summarize.py \
    --eval-dir "$eval_dir" \
    | tee "$log_dir/summarize.log" >/dev/null

  echo "✅ Done: $model | $split"
}

# Ensure we always clean up the server on exit/error
trap 'stop_vllm' EXIT

# --- main loop -------------------------------------------------------------
for model in "${MODELS[@]}"; do
  for split in "${SPLITS[@]}"; do
    run_one "$model" "$split"
  done
done

echo "🎉 All runs complete."

# export MODEL_NAME=google/gemma-3-1b-it
# export SPLIT=curated # etiher curated or wildchat
# 
# export VLLM_PORT=8010
# # Start VLLM server
# uv run vllm serve $MODEL_NAME --port $VLLM_PORT --served-model-name $MODEL_NAME > vllm.log 2>&1 &
# 
# uv run python src/inference.py \
#   --mode vllm \
#   --model $MODEL_NAME \
#   --data $SPLIT \
#   --eval-dir results/$SPLIT/$MODEL_NAME \
#   --sampling regenerate \
#   --num-generations 10
# 
# pkill -f vllm
# 
# uv run python src/inference.py \
#   --mode transformers \
#   --model $MODEL_NAME \
#   --data $SPLIT \
#   --eval-dir results/$SPLIT/$MODEL_NAME \
#   --sampling regenerate \
#   --num-generations 10
# 
# uv run python src/partition.py \
#   --eval-dir results/$SPLIT/$MODEL_NAME \
#   --alg classifier
# 
# uv run python src/score.py \
#   --eval-dir results/$SPLIT/$MODEL_NAME \
#   --patience 0.8
# 
# uv run python src/summarize.py --eval-dir results/$SPLIT/$MODEL_NAME
# 