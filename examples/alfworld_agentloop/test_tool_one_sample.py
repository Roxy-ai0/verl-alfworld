# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import asyncio
import json
import os
from uuid import uuid4

import ray
import torch

import eval_direct as eval_lib
from eval_tool import make_config
from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager, DictConfigWrap
from verl.experimental.alfworld.dataset import discover_episodes
from verl.experimental.alfworld.env import ALFWorldTextEnv
from verl.experimental.alfworld.prompt import (
    ALFWORLD_ENV_STEP_TOOL,
    ALFWORLD_TOOL_SYSTEM_PROMPT,
    build_tool_calling_prompt_step0,
)
from verl.experimental.alfworld.tool_agent_loop import ALFWorldToolCallingAgentLoop
from verl.experimental.alfworld.tool_utils import validate_env_step_tool_call
from verl.utils import omega_conf_to_dataclass
from verl.utils.dataset.rl_dataset import get_dataset_class


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one ALFWorld tool-calling step on a single sample.")
    parser.add_argument("--model-path", default=eval_lib.DEFAULT_MODEL_PATH)
    parser.add_argument("--alfworld-data-root", default=eval_lib.DEFAULT_ALFWORLD_DATA_ROOT)
    parser.add_argument("--split", default="valid_unseen", choices=["train", "valid_seen", "valid_unseen"])
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--agent-loop-config", default="examples/alfworld_agentloop/config/agent_loops.yaml")
    parser.add_argument("--rollout-name", default="vllm")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--tool-parser-format", default="hermes")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--gpus-per-node", type=int, default=1)
    parser.add_argument("--prompt-length", type=int, default=2048)
    parser.add_argument("--response-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    return parser.parse_args()


def load_one_episode(args: argparse.Namespace):
    episodes = discover_episodes(
        data_root=args.alfworld_data_root,
        split=args.split,
        limit=args.sample_index + 1,
    )
    if args.sample_index >= len(episodes):
        raise IndexError(f"sample-index {args.sample_index} out of range for split {args.split}")
    return episodes[args.sample_index]


def build_sampling_params(config) -> dict:
    return {
        "temperature": config.actor_rollout_ref.rollout.val_kwargs.temperature,
        "top_p": config.actor_rollout_ref.rollout.val_kwargs.top_p,
        "top_k": config.actor_rollout_ref.rollout.val_kwargs.top_k,
        "repetition_penalty": 1.0,
        "logprobs": config.actor_rollout_ref.rollout.calculate_log_probs,
    }


async def run_one_step(args: argparse.Namespace) -> None:
    config = make_config(args)
    episode = load_one_episode(args)
    env = ALFWorldTextEnv(gamefile=episode.gamefile, max_steps=args.max_steps)

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"CUDA_VISIBLE_DEVICES = {visible_devices}")
    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count() = {torch.cuda.device_count()}")

    ray.init(
        num_gpus=args.gpus_per_node,
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "VLLM_USE_V1": "1",
                "VLLM_LOGGING_LEVEL": "INFO",
                "NCCL_DEBUG": "WARN",
            }
        },
        ignore_reinit_error=True,
    )
    print(f"Ray available resources: {ray.available_resources()}")

    agent_loop_manager = eval_lib.init_agent_loop_manager(config)
    servers = list(zip(agent_loop_manager.server_addresses, agent_loop_manager.server_handles, strict=True))
    server_manager = AsyncLLMServerManager(
        config,
        servers,
        load_balancer_handle=agent_loop_manager.global_load_balancer,
    )

    model_config = omega_conf_to_dataclass(config.actor_rollout_ref.model)
    dataset_cls = get_dataset_class(config.data)
    tool_loop = ALFWorldToolCallingAgentLoop(
        trainer_config=DictConfigWrap(config=config),
        server_manager=server_manager,
        tokenizer=model_config.tokenizer,
        processor=model_config.processor,
        dataset_cls=dataset_cls,
        data_config=DictConfigWrap(config.data),
        max_steps=args.max_steps,
        history_window=0,
    )

    try:
        state = env.reset()
        prompt_text = build_tool_calling_prompt_step0(
            current_observation=state.observation,
            admissible_actions=state.admissible_actions,
        )
        messages = [
            {"role": "system", "content": ALFWORLD_TOOL_SYSTEM_PROMPT.strip()},
            {"role": "user", "content": prompt_text},
        ]
        prompt_ids = await tool_loop.apply_chat_template(messages, tools=tool_loop.tool_schemas)
        output = await tool_loop.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=build_sampling_params(config),
        )
        response_ids = output.token_ids[: tool_loop.response_length]
        raw_response_text = await tool_loop._decode_tokens(response_ids, skip_special_tokens=False)
        response_content, tool_calls = await tool_loop._extract_tool_calls(response_ids)
        action, parse_meta = validate_env_step_tool_call(tool_calls, state.admissible_actions)

        print(f"Task ID: {episode.task_id}")
        print(f"Gamefile: {episode.gamefile}")
        print(f"Task Description: {episode.task_description}")
        print()
        print("=== Current Observation ===")
        print(state.observation)
        print()
        print("=== Admissible Actions ===")
        print(json.dumps(state.admissible_actions, ensure_ascii=False, indent=2))
        print()
        print("=== System Message ===")
        print(ALFWORLD_TOOL_SYSTEM_PROMPT.strip())
        print()
        print("=== User Prompt ===")
        print(prompt_text)
        print()
        print("=== Tool Schema ===")
        print(json.dumps(ALFWORLD_ENV_STEP_TOOL, ensure_ascii=False, indent=2))
        print()
        print("=== Model Raw Output ===")
        print(raw_response_text)
        print()
        print("=== Parsed Tool Content ===")
        print(response_content)
        print()
        print("=== Parsed Action ===")
        print(action)
        print()
        print("=== Parse Meta ===")
        print(json.dumps(eval_lib.to_serializable(parse_meta), ensure_ascii=False, indent=2))
        print()
        print(f"invalid = {bool(parse_meta.get('invalid', False))}")
        print(f"fallback_used = {bool(parse_meta.get('fallback_used', False))}")

        if not bool(parse_meta.get("invalid", False)):
            next_state = env.step(action)
            print()
            print("=== Step Result ===")
            print(f"reward = {next_state.reward}")
            print(f"done = {next_state.done}")
            print(f"won = {next_state.won}")
            print("new_observation:")
            print(next_state.observation)
        else:
            print()
            print("=== Step Result ===")
            print("Tool call invalid; skipped env.step(action).")
    finally:
        env.close()
        ray.shutdown()


def main() -> None:
    args = parse_args()
    asyncio.run(run_one_step(args))


if __name__ == "__main__":
    main()
