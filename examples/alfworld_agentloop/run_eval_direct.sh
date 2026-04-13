#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/storage/v-jinpewang/az_workspace/zhanglin/reproduction/lwb/model/Qwen2.5-1.5B-Instruct}"
ALFWORLD_DATA_ROOT="${ALFWORLD_DATA_ROOT:-/storage/v-jinpewang/az_workspace/zhanglin/reproduction/lwb/data}"
SPLIT="${SPLIT:-valid_unseen}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/alfworld_direct_eval}"

export VLLM_USE_V1=1
export TOKENIZERS_PARALLELISM=true

python examples/alfworld_agentloop/eval_direct.py \
  --model-path "${MODEL_PATH}" \
  --alfworld-data-root "${ALFWORLD_DATA_ROOT}" \
  --split "${SPLIT}" \
  --rollout-name vllm \
  --max-steps 30 \
  --batch-size 8 \
  --num-workers 2 \
  --gpus-per-node 1 \
  --output-dir "${OUTPUT_DIR}" \
  --save-traces
