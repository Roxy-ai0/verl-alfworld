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
PROJECT_NAME=${PROJECT_NAME:-verl_alfworld_prompt_grpo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2_5_1_5b_prompt_agentloop_grpo}
LOGGER_BACKENDS=${LOGGER_BACKENDS:-'["console","wandb"]'}
WANDB_TAGS=${WANDB_TAGS:-'["alfworld","prompt_grpo","agent_loop"]'}

MAX_STEPS=${MAX_STEPS:-50}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
ROLLOUT_LOGPROB_MB_PER_GPU=${ROLLOUT_LOGPROB_MB_PER_GPU:-2}
REF_LOGPROB_MB_PER_GPU=${REF_LOGPROB_MB_PER_GPU:-2}
AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-2}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-150}
ROLLOUT_N=${ROLLOUT_N:-4}
VAL_ROLLOUT_N=${VAL_ROLLOUT_N:-2}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-10}
RESET_RAY=${RESET_RAY:-1}
RAY_INCLUDE_DASHBOARD=${RAY_INCLUDE_DASHBOARD:-false}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-${SLURM_CPUS_PER_TASK:-32}}
RAY_OBJECT_STORE_MEMORY=${RAY_OBJECT_STORE_MEMORY:-4294967296}
RAY_TMP_DIR=${RAY_TMP_DIR:-/tmp/ray_alfworld_prompt}
DATA_CACHE_DIR=${DATA_CACHE_DIR:-${DATA_DIR}/rlhf_cache}
HF_HOME=${HF_HOME:-${DATA_CACHE_DIR}/hf_home}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${DATA_CACHE_DIR}/datasets}
FILTER_OVERLONG_PROMPTS_WORKERS=${FILTER_OVERLONG_PROMPTS_WORKERS:-1}
BALANCE_BATCH=${BALANCE_BATCH:-False}
AGENT_LOOP_NAME=${AGENT_LOOP_NAME:-alfworld_stepwise_prompt_grpo_agent}
AGENT_LOOP_MANAGER_CLASS=${AGENT_LOOP_MANAGER_CLASS:-verl.experimental.alfworld.stepwise_agent_loop_manager.ALFWorldStepwiseAgentLoopManager}

mkdir -p "${DATA_DIR}"
mkdir -p "${RAY_TMP_DIR}"
mkdir -p "${DATA_CACHE_DIR}"
mkdir -p "${HF_HOME}"
mkdir -p "${HF_DATASETS_CACHE}"

export HF_HOME
export HF_DATASETS_CACHE

if [[ "${RESET_RAY}" == "1" ]]; then
  echo "[ALFWorld stepwise] Stopping existing Ray runtime to avoid stale non-stepwise workers."
  ray stop -f || true
else
  echo "[ALFWorld stepwise] RESET_RAY=0, reusing current Ray runtime."
fi

echo "[ALFWorld stepwise] Launch configuration:"
echo "  agent_loop=${AGENT_LOOP_NAME}"
echo "  agent_loop_manager=${AGENT_LOOP_MANAGER_CLASS}"
echo "  rollout.prompt_length=2048"
echo "  rollout.response_length=512"
echo "  rollout.n=${ROLLOUT_N}"
echo "  validate.n=${VAL_ROLLOUT_N}"
echo "  ray.include_dashboard=${RAY_INCLUDE_DASHBOARD}"
echo "  ray.num_cpus=${RAY_NUM_CPUS}"
echo "  ray.object_store_memory=${RAY_OBJECT_STORE_MEMORY}"
echo "  ray.tmp_dir=${RAY_TMP_DIR}"
echo "  data.cache_dir=${DATA_CACHE_DIR}"
echo "  HF_HOME=${HF_HOME}"
echo "  HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
echo "  data.filter_overlong_prompts_workers=${FILTER_OVERLONG_PROMPTS_WORKERS}"
echo "  trainer.balance_batch=${BALANCE_BATCH}"
echo "[ALFWorld stepwise] Expected post-start sanity checks:"
echo "  - num_turns/mean ~= 2"
echo "  - response_length/mean <= 512"
echo "  - prompt_length/mean <= 2048"

if [[ ! -f "${TRAIN_FILE}" || ! -f "${VAL_FILE}" ]]; then
  python "${REPO_ROOT}/examples/alfworld_agentloop/preprocess_alfworld_direct.py" \
    --alfworld-data-root "${ALFWORLD_DATA_ROOT}" \
    --output-dir "${DATA_DIR}" \
    --splits train valid_unseen \
    --agent-name "${AGENT_LOOP_NAME}" \
    --max-steps "${MAX_STEPS}"
fi

python -m verl.trainer.main_ppo \
  +ray_kwargs.ray_init.include_dashboard=${RAY_INCLUDE_DASHBOARD} \
  ray_kwargs.ray_init.num_cpus=${RAY_NUM_CPUS} \
  +ray_kwargs.ray_init.object_store_memory=${RAY_OBJECT_STORE_MEMORY} \
  +ray_kwargs.ray_init._temp_dir="${RAY_TMP_DIR}" \
  algorithm.adv_estimator=grpo \
  algorithm.compute_mean_std_cross_steps=True \
  algorithm.use_kl_in_reward=False \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  +data.cache_dir="${DATA_CACHE_DIR}" \
  ++data.filter_overlong_prompts_workers="${FILTER_OVERLONG_PROMPTS_WORKERS}" \
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
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.temperature=0.4 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.skip_tokenizer_init=True \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.data_parallel_size=1 \
  actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_NUM_WORKERS}" \
  actor_rollout_ref.rollout.agent.default_agent_loop="${AGENT_LOOP_NAME}" \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_CONFIG}" \
  +actor_rollout_ref.rollout.agent.agent_loop_manager_class="${AGENT_LOOP_MANAGER_CLASS}" \
  actor_rollout_ref.rollout.val_kwargs.n="${VAL_ROLLOUT_N}" \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOGPROB_MB_PER_GPU}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.fsdp_config.dtype=float16 \
  reward.reward_model.enable=False \
  reward.reward_manager.name=naive \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.logger="${LOGGER_BACKENDS}" \
  trainer.wandb_tags="${WANDB_TAGS}" \
  trainer.balance_batch="${BALANCE_BATCH}" \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.save_freq_unit=epoch \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.test_freq_unit=epoch \
  trainer.log_freq=1 \
  trainer.log_freq_unit=step \
  trainer.progress_bar_unit=step \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.val_before_train=False \
  data.shuffle=True \
  "$@"
