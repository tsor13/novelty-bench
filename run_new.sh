# # export MODEL_NAME=google/gemma-3-1b-it
# # export MODEL_NAME=google/gemma-3-1b-pt
# export MODEL_NAME=qwen/Qwen3-0.6B
# export SPLIT=curated
# export SAMPLING=regenerate # or "fewshot"
# # export SAMPLING=fewshot
# # export FEWSHOT_FILE=src/fewshot/gemma.jsonl # or None
# export FEWSHOT_FILE=None
# export TEMPLATE_MODE=default # or "colon" or "explicit-description" or "explicit-assistant"
# # export TEMPLATE_MODE=colon

export RESULTS_DIR="new_results/$MODEL_NAME/$SPLIT/$TEMPLATE_MODE/$SAMPLING/"

echo "Results directory: $RESULTS_DIR"


uv run python src/inference.py \
  --mode transformers \
  --model $MODEL_NAME \
  --data $SPLIT \
  --eval-dir $RESULTS_DIR \
  --sampling $SAMPLING \
  --fewshot-file $FEWSHOT_FILE \
  --template-mode $TEMPLATE_MODE \
  --num-generations 10

uv run python src/partition.py \
  --eval-dir $RESULTS_DIR \
  --alg classifier

uv run python src/score.py \
  --eval-dir $RESULTS_DIR \
  --patience 0.8

uv run python src/summarize.py --eval-dir $RESULTS_DIR

echo "Done"