#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

MODEL_PATH=${MODEL_PATH:-/storage/v-jinpewang/az_workspace/zhanglin/reproduction/lwb/model/Qwen2.5-1.5B-Instruct}
ALFWORLD_DATA_ROOT=${ALFWORLD_DATA_ROOT:-/storage/v-jinpewang/az_workspace/zhanglin/reproduction/lwb/data}
DATA_DIR=${DATA_DIR:-/storage/v-jinpewang/az_workspace/zhanglin/reproduction/lwb/data/alfworld_prompt_grpo}
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/train.parquet}
VAL_FILE=${VAL_FILE:-${DATA_DIR}/valid_unseen.parquet}
AGENT_LOOP_CONFIG=${AGENT_LOOP_CONFIG:-${REPO_ROOT}/examples/alfworld_agentloop/config/agent_loops.yaml}

MAX_STEPS=${MAX_STEPS:-50}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}
ROLLOUT_LOGPROB_MB_PER_GPU=${ROLLOUT_LOGPROB_MB_PER_GPU:-8}
REF_LOGPROB_MB_PER_GPU=${REF_LOGPROB_MB_PER_GPU:-8}
AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-2}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-150}

mkdir -p "${DATA_DIR}"

if [[ ! -f "${TRAIN_FILE}" || ! -f "${VAL_FILE}" ]]; then
  python "${REPO_ROOT}/examples/alfworld_agentloop/preprocess_alfworld_direct.py" \
    --alfworld-data-root "${ALFWORLD_DATA_ROOT}" \
    --output-dir "${DATA_DIR}" \
    --splits train valid_unseen \
    --agent-name alfworld_prompt_grpo_agent \
    --max-steps "${MAX_STEPS}"
fi

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.compute_mean_std_cross_steps=True \
  algorithm.use_kl_in_reward=False \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size=32 \
  data.max_prompt_length=2048 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  +actor_rollout_ref.model.override_config.attn_implementation=eager \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.use_invalid_action_penalty=True \
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.fsdp_config.dtype=float16 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOGPROB_MB_PER_GPU}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.dtype=float16 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.prompt_length=2048 \
  actor_rollout_ref.rollout.response_length=512 \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.temperature=0.4 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.skip_tokenizer_init=True \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.data_parallel_size=1 \
  actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_NUM_WORKERS}" \
  actor_rollout_ref.rollout.agent.default_agent_loop=alfworld_prompt_grpo_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_CONFIG}" \
  actor_rollout_ref.rollout.val_kwargs.n=2 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOGPROB_MB_PER_GPU}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.fsdp_config.dtype=float16 \
  reward.reward_model.enable=False \
  reward.reward_manager.name=naive \
  trainer.project_name='verl_alfworld_prompt_grpo' \
  trainer.experiment_name='qwen2_5_1_5b_prompt_agentloop_grpo' \
  trainer.logger='["console"]' \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.save_freq=20 \
  trainer.test_freq=10 \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.val_before_train=False \
  data.shuffle=True \
  "$@"
