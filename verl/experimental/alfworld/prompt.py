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

import json


ALFWORLD_TEMPLATE_DIRECT = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your task is to: {task_description}

Your current observation is: {current_observation}
Your admissible actions are: [{admissible_actions}].

Now it's your turn to take an action.
You must choose exactly one action from the admissible actions.
First, reason briefly inside <think> </think>.
Then output exactly one admissible action inside <action> </action>.
Do not output anything else.
"""


ALFWORLD_TOOL_SYSTEM_PROMPT = """
You are an expert agent operating in the ALFRED Embodied Environment.
You must interact with the environment only by calling the tool `env_step`.
The `action` argument must be a single action string chosen from the current admissible actions.
Do not invent actions. Do not output a plain action string by itself.
Do not output a final answer.
Your reply must contain exactly one tool call to `env_step`.
Learn and follow this exact tool-call format for every reply:
<tool_call>
{"name": "env_step", "arguments": {"action": "<one admissible action>"}}
</tool_call>
For example, if the chosen action is `look`, then the reply should be:
<tool_call>
{"name": "env_step", "arguments": {"action": "look"}}
</tool_call>
Any reply without a tool call is invalid.
"""


ALFWORLD_ENV_STEP_TOOL = {
    "type": "function",
    "function": {
        "name": "env_step",
        "description": "Execute exactly one ALFWorld action.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Choose one action from the current admissible actions.",
                }
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
}


ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""


ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""


ALFWORLD_TOOL_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your task is to: {task_description}
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason briefly about the current situation.
Then call the tool `env_step` with exactly one admissible action.
Do not answer with only the action text.
Your assistant reply must contain exactly one tool call to `env_step`.
"""


ALFWORLD_TOOL_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason briefly about the current situation.
Then call the tool `env_step` with exactly one admissible action for current step.
Do not answer with only the action text.
Your assistant reply must contain exactly one tool call to `env_step`.
"""


def normalize_observation(observation: str) -> str:
    text = " ".join((observation or "").strip().split())
    marker = "your task is to:"
    lower_text = text.lower()
    marker_pos = lower_text.find(marker)
    if marker_pos >= 0:
        text = text[:marker_pos].strip()
    return text or "nothing"


def build_direct_prompt(task_description: str, current_observation: str, admissible_actions: list[str]) -> str:
    actions_text = ", ".join(json.dumps(action, ensure_ascii=False) for action in admissible_actions)
    return ALFWORLD_TEMPLATE_DIRECT.format(
        task_description=(task_description or "").strip(),
        current_observation=normalize_observation(current_observation),
        admissible_actions=actions_text,
    ).strip()


def _format_action_history(trajectory_history: list[dict[str, str]]) -> str:
    if not trajectory_history:
        return "None yet."

    formatted_entries = []
    for idx, item in enumerate(trajectory_history, start=1):
        observation = normalize_observation(str(item.get("observation", "")))
        action = " ".join(str(item.get("action", "")).strip().split()) or "N/A"
        formatted_entries.append(f"[{idx}] Observation: {observation}\n[{idx}] Action: {action}")
    return "\n".join(formatted_entries)


def build_react_prompt(
    task_description: str,
    trajectory_history: list[dict[str, str]],
    current_observation: str,
    admissible_actions: list[str],
    step_count: int | None = None,
) -> str:
    actions_text = ", ".join(json.dumps(action, ensure_ascii=False) for action in admissible_actions)
    history = trajectory_history or []
    total_step_count = len(history) if step_count is None else max(int(step_count), 0)
    return ALFWORLD_TEMPLATE.format(
        task_description=(task_description or "").strip(),
        step_count=total_step_count,
        history_length=len(history),
        action_history=_format_action_history(history),
        current_step=total_step_count + 1,
        current_observation=normalize_observation(current_observation),
        admissible_actions=actions_text,
    ).strip()


def build_prompt_grpo_prompt(
    task_description: str,
    trajectory_history: list[dict[str, str]],
    current_observation: str,
    admissible_actions: list[str],
    step_count: int | None = None,
) -> str:
    actions_text = ", ".join(json.dumps(action, ensure_ascii=False) for action in admissible_actions)
    history = trajectory_history or []
    total_step_count = len(history) if step_count is None else max(int(step_count), 0)
    current_observation_text = normalize_observation(current_observation)

    if total_step_count <= 0:
        return ALFWORLD_TEMPLATE_NO_HIS.format(
            current_observation=current_observation_text,
            admissible_actions=actions_text,
        ).strip()

    return ALFWORLD_TEMPLATE.format(
        task_description=(task_description or "").strip(),
        step_count=total_step_count,
        history_length=len(history),
        action_history=_format_action_history(history),
        current_step=total_step_count + 1,
        current_observation=current_observation_text,
        admissible_actions=actions_text,
    ).strip()


def build_tool_calling_prompt_step0(
    task_description: str,
    current_observation: str,
    admissible_actions: list[str],
) -> str:
    actions_text = ", ".join(json.dumps(action, ensure_ascii=False) for action in admissible_actions)
    return ALFWORLD_TOOL_TEMPLATE_NO_HIS.format(
        task_description=(task_description or "").strip(),
        current_observation=normalize_observation(current_observation),
        admissible_actions=actions_text,
    ).strip()


def build_tool_calling_prompt_stepk(
    task_description: str,
    trajectory_history: list[dict[str, str]],
    current_observation: str,
    admissible_actions: list[str],
    step_count: int | None = None,
) -> str:
    actions_text = ", ".join(json.dumps(action, ensure_ascii=False) for action in admissible_actions)
    history = trajectory_history or []
    total_step_count = len(history) if step_count is None else max(int(step_count), 0)
    return ALFWORLD_TOOL_TEMPLATE.format(
        task_description=(task_description or "").strip(),
        step_count=total_step_count,
        history_length=len(history),
        action_history=_format_action_history(history),
        current_step=total_step_count + 1,
        current_observation=normalize_observation(current_observation),
        admissible_actions=actions_text,
    ).strip()
