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

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput

from .direct_agent_loop import ALFWorldDirectAgentLoop
from .env import ALFWorldTextEnv
from .parser import parse_action
from .prompt import build_prompt_grpo_prompt
from .task_category import infer_alfworld_task_category

logger = logging.getLogger(__name__)


@register("alfworld_prompt_grpo_agent")
class ALFWorldPromptGRPOAgentLoop(ALFWorldDirectAgentLoop):
    def __init__(
        self,
        *args,
        max_steps: int = 30,
        history_window: int = 5,
        success_reward: float = 10.0,
        failure_reward: float = 0.0,
        invalid_action_penalty: float = 0.1,
        tool_extension_mode: str = "reserved",
        **kwargs,
    ):
        super().__init__(*args, max_steps=max_steps, **kwargs)
        self.history_window = history_window
        self.success_reward = success_reward
        self.failure_reward = failure_reward
        self.invalid_action_penalty = invalid_action_penalty
        self.tool_extension_mode = tool_extension_mode

    def _get_recent_history(self, episode_trace: list[dict[str, Any]], history_window: int) -> list[dict[str, str]]:
        recent_trace = episode_trace[-history_window:] if history_window > 0 else []
        return [
            {
                "observation": str(step.get("observation", "")),
                "action": str(step.get("parsed_action", "")),
            }
            for step in recent_trace
        ]

    async def _append_followup_user_turn(
        self,
        messages: list[dict[str, str]],
        prompt_ids: list[int],
        response_mask: list[int],
        response_logprobs: list[float],
        prompt_text: str,
    ) -> None:
        add_messages = [{"role": "user", "content": prompt_text}]
        messages.extend(add_messages)
        user_turn_ids = await self.apply_chat_template(add_messages, remove_system_prompt=True)
        prompt_ids.extend(user_turn_ids)
        response_mask.extend([0] * len(user_turn_ids))
        if response_logprobs:
            response_logprobs.extend([0.0] * len(user_turn_ids))

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        extra_info = kwargs.get("extra_info", {}) or {}
        task_description = str(extra_info["task_description"])
        gamefile = str(extra_info["gamefile"])
        max_steps = int(extra_info.get("max_steps", self.max_steps))
        history_window = int(extra_info.get("history_window", self.history_window))
        success_reward_value = float(extra_info.get("success_reward", self.success_reward))
        failure_reward_value = float(extra_info.get("failure_reward", self.failure_reward))
        invalid_action_penalty = float(extra_info.get("invalid_action_penalty", self.invalid_action_penalty))

        request_id = uuid4().hex
        env = ALFWorldTextEnv(gamefile=gamefile, max_steps=max_steps)
        messages: list[dict[str, str]] = []
        full_sequence_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        assistant_step_spans: list[dict[str, Any]] = []
        episode_trace: list[dict[str, Any]] = []
        last_observation = ""
        invalid_action_count = 0
        total_generate_time = 0.0
        total_num_preempted = None
        success = 0.0
        final_reward = 0.0
        episode_reward = failure_reward_value
        error_message = None
        task_category = infer_alfworld_task_category(gamefile=gamefile, task_description=task_description)

        try:
            state = env.reset()
            last_observation = state.observation

            for step_idx in range(max_steps):
                if not state.admissible_actions:
                    error_message = "No admissible actions returned by ALFWorld."
                    break

                recent_history = self._get_recent_history(episode_trace, history_window)
                prompt_text = build_prompt_grpo_prompt(
                    task_description=task_description,
                    trajectory_history=recent_history,
                    current_observation=state.observation,
                    admissible_actions=state.admissible_actions,
                    step_count=len(episode_trace),
                )

                if step_idx == 0:
                    messages = [{"role": "user", "content": prompt_text}]
                    full_sequence_ids = await self.apply_chat_template(messages)
                else:
                    await self._append_followup_user_turn(
                        messages=messages,
                        prompt_ids=full_sequence_ids,
                        response_mask=response_mask,
                        response_logprobs=response_logprobs,
                        prompt_text=prompt_text,
                    )
                    if len(response_mask) >= self.response_length:
                        error_message = "Response length budget exhausted before the next environment step."
                        break

                timing = {}
                with simple_timer("generate_sequences", timing):
                    output: TokenOutput = await self.server_manager.generate(
                        request_id=request_id,
                        prompt_ids=full_sequence_ids,
                        sampling_params=sampling_params,
                    )
                total_generate_time += timing.get("generate_sequences", 0.0)
                if total_num_preempted is None:
                    total_num_preempted = output.num_preempted if output.num_preempted is not None else -1
                else:
                    total_num_preempted += output.num_preempted if output.num_preempted is not None else 0

                remaining_response_budget = self.response_length - len(response_mask)
                response_ids = output.token_ids[:remaining_response_budget]
                if not response_ids:
                    error_message = "Response length budget exhausted during generation."
                    break

                response_text = await self._decode(response_ids)
                action, parse_meta = parse_action(
                    response_text,
                    state.admissible_actions,
                    require_think_tags=True,
                )
                invalid = bool(parse_meta.get("invalid", False))
                invalid_action_count += int(invalid)

                assistant_start = len(response_mask)
                full_sequence_ids.extend(response_ids)
                response_mask.extend([1] * len(response_ids))
                if output.log_probs:
                    response_logprobs.extend(output.log_probs[: len(response_ids)])
                elif response_logprobs:
                    response_logprobs.extend([0.0] * len(response_ids))
                assistant_end = len(response_mask) - 1

                messages.append({"role": "assistant", "content": response_text})

                next_state = env.step(action)
                last_observation = next_state.observation
                final_reward = next_state.reward
                success = 1.0 if next_state.won or (next_state.done and next_state.reward > 0) else 0.0

                assistant_step_spans.append(
                    {
                        "step": step_idx,
                        "start": assistant_start,
                        "end": assistant_end,
                        "invalid": invalid,
                    }
                )
                episode_trace.append(
                    {
                        "step": step_idx,
                        "observation": state.observation,
                        "admissible_actions": state.admissible_actions,
                        "prompt_text": prompt_text,
                        "response_text": response_text,
                        "parsed_action": action,
                        "history_length": len(recent_history),
                        "history_window": history_window,
                        "reward": next_state.reward,
                        "done": next_state.done,
                        "won": next_state.won,
                        **parse_meta,
                    }
                )

                state = next_state
                if state.done:
                    break
                if len(response_mask) >= self.response_length:
                    error_message = "Response length budget exhausted before episode termination."
                    break

            episode_reward = success_reward_value if success > 0 else failure_reward_value
            rm_scores = [0.0] * len(response_mask)
            step_reward_mask = [0] * len(response_mask)
            step_ids = [0] * len(response_mask)
            step_scores: list[float] = []
            step_invalid_flags: list[int] = []
            step_invalid_reasons: list[str | None] = []

            for span in assistant_step_spans:
                step_index = int(span["step"])
                step_invalid = int(bool(span["invalid"]))
                step_score = episode_reward - invalid_action_penalty * step_invalid
                rm_scores[span["end"]] = step_score
                step_reward_mask[span["end"]] = 1
                for token_idx in range(span["start"], span["end"] + 1):
                    step_ids[token_idx] = step_index + 1

                episode_trace[step_index]["step_score"] = step_score
                step_scores.append(step_score)
                step_invalid_flags.append(step_invalid)
                step_invalid_reasons.append(episode_trace[step_index].get("invalid_reason"))

            mean_step_score = sum(step_scores) / len(step_scores) if step_scores else 0.0
            invalid_rate = sum(step_invalid_flags) / len(step_invalid_flags) if step_invalid_flags else 0.0
            reward_extra_info = {
                "acc": success,
                "success": bool(success),
                "num_env_steps": len(episode_trace),
                "invalid_action_count": invalid_action_count,
                "final_reward": final_reward,
                "episode_reward": episode_reward,
                "mean_step_score": mean_step_score,
                "invalid_rate": invalid_rate,
                "history_window": history_window,
                "task_category": task_category,
                "error": error_message,
            }

            if not full_sequence_ids:
                full_sequence_ids = await self._build_fallback_prompt_ids(
                    task_description=task_description,
                    error_message=error_message or "No model generation was produced.",
                )
            if not response_mask:
                fallback_response_ids = await self._build_fallback_response_ids()
                response_mask = [1] * len(fallback_response_ids)
                response_logprobs = []
                rm_scores = [0.0] * len(fallback_response_ids)
                step_reward_mask = [0] * len(fallback_response_ids)
                step_ids = [0] * len(fallback_response_ids)
                full_sequence_ids = full_sequence_ids + fallback_response_ids

            response_ids = full_sequence_ids[-len(response_mask) :]
            prompt_ids = full_sequence_ids[: len(full_sequence_ids) - len(response_mask)]
            reward_extra_info["prompt_length"] = len(prompt_ids)
            reward_extra_info["response_length"] = len(response_ids)
            metrics = AgentLoopMetrics(
                generate_sequences=total_generate_time,
                num_preempted=total_num_preempted if total_num_preempted is not None else -1,
            )
            return AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_mask=response_mask,
                response_logprobs=response_logprobs or None,
                reward_score=success,
                rm_scores=rm_scores,
                step_reward_mask=step_reward_mask,
                step_ids=step_ids,
                num_turns=max(len(episode_trace) * 2, 1),
                metrics=metrics,
                extra_fields={
                    "episode_trace": episode_trace,
                    "step_scores": step_scores,
                    "step_invalid_flags": step_invalid_flags,
                    "step_invalid_reasons": step_invalid_reasons,
                    "reward_extra_info": reward_extra_info,
                    "final_observation": last_observation,
                    "alfworld_task_description": task_description,
                    "alfworld_gamefile": gamefile,
                    "agent_loop_mode": "prompt_based",
                    "tool_extension_mode": self.tool_extension_mode,
                },
            )
        except Exception as exc:
            logger.exception("ALFWorld prompt GRPO episode failed: %s", exc)
            fallback_prompt_ids = await self._build_fallback_prompt_ids(task_description=task_description, error_message=str(exc))
            fallback_response_ids = await self._build_fallback_response_ids()
            return AgentLoopOutput(
                prompt_ids=fallback_prompt_ids,
                response_ids=fallback_response_ids,
                response_mask=[1] * len(fallback_response_ids),
                reward_score=0.0,
                rm_scores=[0.0] * len(fallback_response_ids),
                step_reward_mask=[0] * len(fallback_response_ids),
                step_ids=[0] * len(fallback_response_ids),
                num_turns=1,
                metrics=AgentLoopMetrics(generate_sequences=total_generate_time, num_preempted=-1),
                extra_fields={
                    "episode_trace": episode_trace,
                    "step_scores": [],
                    "step_invalid_flags": [],
                    "step_invalid_reasons": [],
                    "reward_extra_info": {
                        "acc": 0.0,
                        "success": False,
                        "num_env_steps": len(episode_trace),
                        "invalid_action_count": invalid_action_count,
                        "final_reward": final_reward,
                        "episode_reward": 0.0,
                        "mean_step_score": 0.0,
                        "invalid_rate": 0.0,
                        "history_window": history_window,
                        "task_category": task_category,
                        "prompt_length": len(fallback_prompt_ids),
                        "response_length": len(fallback_response_ids),
                        "error": str(exc),
                    },
                    "final_observation": last_observation,
                    "alfworld_task_description": task_description,
                    "alfworld_gamefile": gamefile,
                    "agent_loop_mode": "prompt_based",
                    "tool_extension_mode": self.tool_extension_mode,
                },
            )
        finally:
            env.close()
