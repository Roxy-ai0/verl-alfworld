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


@register("alfworld_stepwise_prompt_grpo_agent")
class ALFWorldStepwisePromptGRPOAgentLoop(ALFWorldDirectAgentLoop):
    """ALFWorld GRPO agent loop with verl-agent style step-wise samples.

    Training returns one sample per environment step:
    - prompt_ids: the single-turn prompt for that step only
    - response_ids: the model response for that step only
    - rm_scores: episode-level reward placed on the last response token

    Validation keeps one sample per episode so existing eval/validation logic remains stable.
    """

    def __init__(
        self,
        *args,
        max_steps: int = 30,
        history_window: int = 5,
        success_reward: float = 10.0,
        failure_reward: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, max_steps=max_steps, **kwargs)
        self.history_window = history_window
        self.success_reward = success_reward
        self.failure_reward = failure_reward

    def _get_recent_history(self, episode_trace: list[dict[str, Any]], history_window: int) -> list[dict[str, str]]:
        recent_trace = episode_trace[-history_window:] if history_window > 0 else []
        return [
            {
                "observation": str(step.get("observation", "")),
                "action": str(step.get("parsed_action", "")),
            }
            for step in recent_trace
        ]

    async def _generate_step(
        self,
        *,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
    ) -> tuple[TokenOutput, float]:
        timing = {}
        with simple_timer("generate_sequences", timing):
            output: TokenOutput = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
            )
        return output, timing.get("generate_sequences", 0.0)

    def _build_reward_extra_info(
        self,
        *,
        success: float,
        final_reward: float,
        episode_reward: float,
        invalid_action_count: int,
        num_env_steps: int,
        history_window: int,
        task_category: str,
        error_message: str | None,
        prompt_length: int,
        response_length: int,
    ) -> dict[str, Any]:
        invalid_rate = invalid_action_count / max(num_env_steps, 1)
        return {
            "acc": success,
            "success": bool(success),
            "num_env_steps": num_env_steps,
            "invalid_action_count": invalid_action_count,
            "final_reward": final_reward,
            "episode_reward": episode_reward,
            "mean_step_score": episode_reward,
            "invalid_rate": invalid_rate,
            "history_window": history_window,
            "task_category": task_category,
            "prompt_length": prompt_length,
            "response_length": response_length,
            "error": error_message,
        }

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput | list[AgentLoopOutput]:
        validate = bool(kwargs.get("validate", False))
        extra_info = kwargs.get("extra_info", {}) or {}
        task_description = str(extra_info["task_description"])
        gamefile = str(extra_info["gamefile"])
        max_steps = int(extra_info.get("max_steps", self.max_steps))
        history_window = int(extra_info.get("history_window", self.history_window))
        success_reward_value = float(extra_info.get("success_reward", self.success_reward))
        failure_reward_value = float(extra_info.get("failure_reward", self.failure_reward))
        rollout_uid = str(kwargs.get("uid", uuid4().hex))
        data_source = str(kwargs.get("data_source", "alfworld_text"))

        env = ALFWorldTextEnv(gamefile=gamefile, max_steps=max_steps)
        request_id = uuid4().hex
        task_category = infer_alfworld_task_category(gamefile=gamefile, task_description=task_description)

        episode_trace: list[dict[str, Any]] = []
        step_records: list[dict[str, Any]] = []
        last_prompt_ids: list[int] = []
        last_response_ids: list[int] = []
        last_logprobs: list[float] | None = None
        last_observation = ""
        total_generate_time = 0.0
        total_num_preempted = None
        invalid_action_count = 0
        final_reward = 0.0
        success = 0.0
        error_message = None
        traj_uid = uuid4().hex

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
                messages = [{"role": "user", "content": prompt_text}]
                prompt_ids = await self.apply_chat_template(messages)

                output, step_generate_time = await self._generate_step(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                )
                total_generate_time += step_generate_time
                if total_num_preempted is None:
                    total_num_preempted = output.num_preempted if output.num_preempted is not None else -1
                else:
                    total_num_preempted += output.num_preempted if output.num_preempted is not None else 0

                response_ids = output.token_ids[: self.response_length]
                if not response_ids:
                    error_message = "Response length budget exhausted during generation."
                    break

                response_text = await self._decode(response_ids)
                action, parse_meta = parse_action(
                    response_text,
                    state.admissible_actions,
                    require_think_tags=True,
                )
                is_action_valid = not bool(parse_meta.get("invalid", False))
                invalid_action_count += int(not is_action_valid)

                next_state = env.step(action)
                last_observation = next_state.observation
                final_reward = next_state.reward
                success = 1.0 if next_state.won or (next_state.done and next_state.reward > 0) else 0.0

                step_record = {
                    "step": step_idx,
                    "prompt_ids": prompt_ids,
                    "prompt_text": prompt_text,
                    "response_ids": response_ids,
                    "response_text": response_text,
                    "response_logprobs": output.log_probs[: len(response_ids)] if output.log_probs else None,
                    "observation": state.observation,
                    "admissible_actions": state.admissible_actions,
                    "parsed_action": action,
                    "history_length": len(recent_history),
                    "is_action_valid": is_action_valid,
                    **parse_meta,
                }
                step_records.append(step_record)
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

                last_prompt_ids = prompt_ids
                last_response_ids = response_ids
                last_logprobs = step_record["response_logprobs"]
                state = next_state
                if state.done:
                    break

            episode_reward = success_reward_value if success > 0 else failure_reward_value

            if validate:
                if not last_prompt_ids:
                    last_prompt_ids = await self._build_fallback_prompt_ids(
                        task_description=task_description,
                        error_message=error_message or "No model generation was produced.",
                    )
                if not last_response_ids:
                    last_response_ids = await self._build_fallback_response_ids()
                reward_extra_info = self._build_reward_extra_info(
                    success=success,
                    final_reward=final_reward,
                    episode_reward=episode_reward,
                    invalid_action_count=invalid_action_count,
                    num_env_steps=len(episode_trace),
                    history_window=history_window,
                    task_category=task_category,
                    error_message=error_message,
                    prompt_length=len(last_prompt_ids),
                    response_length=len(last_response_ids),
                )
                return AgentLoopOutput(
                    prompt_ids=last_prompt_ids,
                    response_ids=last_response_ids,
                    response_mask=[1] * len(last_response_ids),
                    response_logprobs=last_logprobs,
                    reward_score=episode_reward,
                    num_turns=max(len(episode_trace) * 2, 1),
                    metrics=AgentLoopMetrics(
                        generate_sequences=total_generate_time,
                        num_preempted=total_num_preempted if total_num_preempted is not None else -1,
                    ),
                    extra_fields={
                        "episode_trace": episode_trace,
                        "reward_extra_info": reward_extra_info,
                        "final_observation": last_observation,
                        "alfworld_task_description": task_description,
                        "alfworld_gamefile": gamefile,
                    },
                )

            if not step_records:
                fallback_prompt_ids = await self._build_fallback_prompt_ids(
                    task_description=task_description,
                    error_message=error_message or "No model generation was produced.",
                )
                fallback_response_ids = await self._build_fallback_response_ids()
                step_records.append(
                    {
                        "step": 0,
                        "prompt_ids": fallback_prompt_ids,
                        "prompt_text": "",
                        "response_ids": fallback_response_ids,
                        "response_text": await self._decode(fallback_response_ids),
                        "response_logprobs": None,
                        "observation": last_observation,
                        "admissible_actions": [],
                        "parsed_action": "look",
                        "history_length": 0,
                        "is_action_valid": False,
                        "invalid": True,
                        "invalid_reason": error_message or "no_generation",
                    }
                )

            outputs: list[AgentLoopOutput] = []
            metrics = AgentLoopMetrics(
                generate_sequences=total_generate_time,
                num_preempted=total_num_preempted if total_num_preempted is not None else -1,
            )
            for step_record in step_records:
                response_ids = step_record["response_ids"]
                rm_scores = [0.0] * len(response_ids)
                rm_scores[-1] = episode_reward
                reward_extra_info = self._build_reward_extra_info(
                    success=success,
                    final_reward=final_reward,
                    episode_reward=episode_reward,
                    invalid_action_count=invalid_action_count,
                    num_env_steps=len(step_records),
                    history_window=history_window,
                    task_category=task_category,
                    error_message=error_message,
                    prompt_length=len(step_record["prompt_ids"]),
                    response_length=len(response_ids),
                )
                reward_extra_info["step_index"] = int(step_record["step"])
                reward_extra_info["history_length"] = int(step_record["history_length"])

                outputs.append(
                    AgentLoopOutput(
                        prompt_ids=step_record["prompt_ids"],
                        response_ids=response_ids,
                        response_mask=[1] * len(response_ids),
                        response_logprobs=step_record["response_logprobs"],
                        reward_score=episode_reward,
                        rm_scores=rm_scores,
                        num_turns=2,
                        metrics=metrics,
                        extra_fields={
                            "reward_extra_info": reward_extra_info,
                            "uid": rollout_uid,
                            "traj_uid": traj_uid,
                            "data_source": data_source,
                            "is_action_valid": bool(step_record["is_action_valid"]),
                            "episode_step_index": int(step_record["step"]),
                            "history_length": int(step_record["history_length"]),
                            "alfworld_task_description": task_description,
                            "alfworld_gamefile": gamefile,
                        },
                    )
                )

            return outputs
        except Exception as exc:
            logger.exception("ALFWorld stepwise prompt GRPO episode failed: %s", exc)
            fallback_prompt_ids = await self._build_fallback_prompt_ids(task_description=task_description, error_message=str(exc))
            fallback_response_ids = await self._build_fallback_response_ids()
            reward_extra_info = self._build_reward_extra_info(
                success=0.0,
                final_reward=final_reward,
                episode_reward=failure_reward_value,
                invalid_action_count=invalid_action_count,
                num_env_steps=len(episode_trace),
                history_window=history_window,
                task_category=task_category,
                error_message=str(exc),
                prompt_length=len(fallback_prompt_ids),
                response_length=len(fallback_response_ids),
            )
            fallback_output = AgentLoopOutput(
                prompt_ids=fallback_prompt_ids,
                response_ids=fallback_response_ids,
                response_mask=[1] * len(fallback_response_ids),
                reward_score=failure_reward_value,
                rm_scores=[0.0] * len(fallback_response_ids[:-1]) + [failure_reward_value],
                num_turns=2 if not validate else 1,
                metrics=AgentLoopMetrics(generate_sequences=total_generate_time, num_preempted=-1),
                extra_fields={
                    "reward_extra_info": reward_extra_info,
                    "uid": rollout_uid,
                    "traj_uid": traj_uid,
                    "data_source": data_source,
                    "is_action_valid": False,
                    "episode_step_index": 0,
                    "history_length": 0,
                    "alfworld_task_description": task_description,
                    "alfworld_gamefile": gamefile,
                },
            )
            return fallback_output if validate else [fallback_output]
        finally:
            env.close()
