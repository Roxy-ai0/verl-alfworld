# ALFWorld Direct Prompt Baseline

This directory contains a minimal `veRL / Agent Loop` baseline for **text-only ALFWorld**:

- `Direct Prompt = current task + current observation + current admissible actions`
- no trajectory history in the prompt
- no reflection memory
- fixed output format:

```text
<think> ... </think>
<action> ... </action>
```

## Files

- `config/agent_loops.yaml`: registers the custom `alfworld_direct_agent`
- `preprocess_alfworld_direct.py`: optional parquet builder
- `eval_direct.py`: standalone evaluation script
- `run_eval_direct.sh`: one-command evaluation entry

Core implementation lives under `verl/experimental/alfworld/`.

## Environment

Install text-only ALFWorld and veRL dependencies in your conda env:

```bash
conda create -n verl-alfworld python=3.10 -y
conda activate verl-alfworld

pip install -U pip setuptools wheel
pip install -e .
pip install vllm==0.8.3 flash-attn --no-build-isolation
pip install alfworld textworld datasets
```

If your ALFWorld assets are not downloaded yet:

```bash
export ALFWORLD_DATA=/storage/v-jinpewang/az_workspace/zhanglin/reproduction/lwb/data
alfworld-download
```

The script expects the official ALFWorld layout:

```text
$ALFWORLD_DATA/json_2.1.1/valid_seen/.../traj_data.json
$ALFWORLD_DATA/json_2.1.1/valid_seen/.../game.tw-pddl
$ALFWORLD_DATA/json_2.1.1/valid_unseen/.../traj_data.json
$ALFWORLD_DATA/json_2.1.1/valid_unseen/.../game.tw-pddl
```

## Run Evaluation

Directly scan the ALFWorld directory and evaluate:

```bash
bash examples/alfworld_agentloop/run_eval_direct.sh
```

Or explicitly:

```bash
python examples/alfworld_agentloop/eval_direct.py \
  --model-path /storage/v-jinpewang/az_workspace/zhanglin/reproduction/lwb/model/Qwen2.5-1.5B-Instruct \
  --alfworld-data-root /storage/v-jinpewang/az_workspace/zhanglin/reproduction/lwb/data \
  --split valid_unseen \
  --rollout-name vllm \
  --output-dir outputs/alfworld_direct_eval \
  --save-traces
```

Outputs:

- `outputs/alfworld_direct_eval/<split>_summary.json`
- `outputs/alfworld_direct_eval/<split>_traces.jsonl` when `--save-traces` is enabled

## Optional: Build Parquet

If you want a veRL-style parquet dataset for later reuse:

```bash
python examples/alfworld_agentloop/preprocess_alfworld_direct.py \
  --alfworld-data-root /storage/v-jinpewang/az_workspace/zhanglin/reproduction/lwb/data \
  --output-dir data/alfworld_direct \
  --splits valid_seen valid_unseen
```

Then evaluate from parquet:

```bash
python examples/alfworld_agentloop/eval_direct.py \
  --model-path /storage/v-jinpewang/az_workspace/zhanglin/reproduction/lwb/model/Qwen2.5-1.5B-Instruct \
  --parquet data/alfworld_direct/valid_unseen.parquet \
  --split valid_unseen \
  --rollout-name vllm \
  --output-dir outputs/alfworld_direct_eval \
  --save-traces
```
