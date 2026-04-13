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

import logging
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput

from .env import ALFWorldTextEnv
from .parser import parse_action
from .prompt import build_direct_prompt

logger = logging.getLogger(__name__)


@register("alfworld_direct_agent")
class ALFWorldDirectAgentLoop(AgentLoopBase):
    def __init__(self, *args, max_steps: int = 30, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_steps = max_steps
        self.response_length = self.rollout_config.response_length

    async def _decode(self, token_ids: list[int]) -> str:
        return await self.loop.run_in_executor(None, lambda: self.tokenizer.decode(token_ids, skip_special_tokens=True))

    async def _build_fallback_prompt_ids(self, task_description: str, error_message: str) -> list[int]:
        messages = [
            {
                "role": "user",
                "content": (
                    "ALFWorld direct prompt episode failed before rollout.\n"
                    f"Task: {task_description}\n"
                    f"Error: {error_message}"
                ),
            }
        ]
        return await self.apply_chat_template(messages)

    async def _build_fallback_response_ids(self) -> list[int]:
        return await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.encode(
                "<think>environment error</think>\n<action>look</action>",
                add_special_tokens=False,
            ),
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        extra_info = kwargs.get("extra_info", {}) or {}
        task_description = str(extra_info["task_description"])
        gamefile = str(extra_info["gamefile"])
        max_steps = int(extra_info.get("max_steps", self.max_steps))

        request_id = uuid4().hex
        env = ALFWorldTextEnv(gamefile=gamefile, max_steps=max_steps)
        episode_trace: list[dict[str, Any]] = []
        last_prompt_ids: list[int] = []
        last_response_ids: list[int] = []
        last_logprobs: list[float] | None = None
        last_observation = ""
        invalid_action_count = 0
        total_generate_time = 0.0
        total_num_preempted = None
        success = 0.0
        final_reward = 0.0
        error_message = None

        try:
            state = env.reset()
            last_observation = state.observation

            for step_idx in range(max_steps):
                if not state.admissible_actions:
                    error_message = "No admissible actions returned by ALFWorld."
                    break

                prompt_text = build_direct_prompt(
                    task_description=task_description,
                    current_observation=state.observation,
                    admissible_actions=state.admissible_actions,
                )
                messages = [{"role": "user", "content": prompt_text}]
                prompt_ids = await self.apply_chat_template(messages)

                timing = {}
                with simple_timer("generate_sequences", timing):
                    output: TokenOutput = await self.server_manager.generate(
                        request_id=request_id,
                        prompt_ids=prompt_ids,
                        sampling_params=sampling_params,
                    )
                total_generate_time += timing.get("generate_sequences", 0.0)
                if total_num_preempted is None:
                    total_num_preempted = output.num_preempted if output.num_preempted is not None else -1
                else:
                    total_num_preempted += output.num_preempted if output.num_preempted is not None else 0

                response_ids = output.token_ids[: self.response_length]
                response_text = await self._decode(response_ids)
                action, parse_meta = parse_action(response_text, state.admissible_actions)
                invalid_action_count += int(parse_meta["fallback_used"])

                next_state = env.step(action)
                last_observation = next_state.observation
                final_reward = next_state.reward
                success = 1.0 if next_state.won or (next_state.done and next_state.reward > 0) else 0.0

                episode_trace.append(
                    {
                        "step": step_idx,
                        "observation": state.observation,
                        "admissible_actions": state.admissible_actions,
                        "prompt_text": prompt_text,
                        "response_text": response_text,
                        "parsed_action": action,
                        "reward": next_state.reward,
                        "done": next_state.done,
                        "won": next_state.won,
                        **parse_meta,
                    }
                )

                last_prompt_ids = prompt_ids
                last_response_ids = response_ids
                last_logprobs = output.log_probs[: len(response_ids)] if output.log_probs else None
                state = next_state

                if state.done:
                    break

            reward_extra_info = {
                "acc": success,
                "success": bool(success),
                "num_env_steps": len(episode_trace),
                "invalid_action_count": invalid_action_count,
                "final_reward": final_reward,
                "error": error_message,
            }

            if not last_prompt_ids:
                last_prompt_ids = await self._build_fallback_prompt_ids(
                    task_description=task_description,
                    error_message=error_message or "No model generation was produced.",
                )
            if not last_response_ids:
                last_response_ids = await self._build_fallback_response_ids()

            metrics = AgentLoopMetrics(
                generate_sequences=total_generate_time,
                num_preempted=total_num_preempted if total_num_preempted is not None else -1,
            )
            return AgentLoopOutput(
                prompt_ids=last_prompt_ids,
                response_ids=last_response_ids,
                response_mask=[1] * len(last_response_ids),
                response_logprobs=last_logprobs,
                reward_score=success,
                num_turns=max(len(episode_trace) * 2, 1),
                metrics=metrics,
                extra_fields={
                    "episode_trace": episode_trace,
                    "reward_extra_info": reward_extra_info,
                    "final_observation": last_observation,
                    "alfworld_task_description": task_description,
                    "alfworld_gamefile": gamefile,
                },
            )
        except Exception as exc:
            logger.exception("ALFWorld direct prompt episode failed: %s", exc)
            fallback_prompt_ids = await self._build_fallback_prompt_ids(task_description=task_description, error_message=str(exc))
            fallback_response_ids = await self._build_fallback_response_ids()
            return AgentLoopOutput(
                prompt_ids=fallback_prompt_ids,
                response_ids=fallback_response_ids,
                response_mask=[1] * len(fallback_response_ids),
                reward_score=0.0,
                num_turns=1,
                metrics=AgentLoopMetrics(generate_sequences=total_generate_time, num_preempted=-1),
                extra_fields={
                    "episode_trace": episode_trace,
                    "reward_extra_info": {
                        "acc": 0.0,
                        "success": False,
                        "num_env_steps": len(episode_trace),
                        "invalid_action_count": invalid_action_count,
                        "final_reward": final_reward,
                        "error": str(exc),
                    },
                    "final_observation": last_observation,
                    "alfworld_task_description": task_description,
                    "alfworld_gamefile": gamefile,
                },
            )
        finally:
            env.close()
