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
from typing import Any

from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.tools.schemas import OpenAIFunctionToolSchema

from .parser import choose_fallback_action, normalize_action


def build_openai_tool_schemas(tool_schemas: list[dict[str, Any]]) -> list[OpenAIFunctionToolSchema]:
    return [OpenAIFunctionToolSchema.model_validate(tool_schema) for tool_schema in tool_schemas]


def build_env_step_fallback_response(action: str) -> str:
    return (
        "<tool_call>"
        + json.dumps({"name": "env_step", "arguments": {"action": action}}, ensure_ascii=False)
        + "</tool_call>"
    )


def validate_env_step_tool_call(
    tool_calls: list[FunctionCall],
    admissible_actions: list[str],
    *,
    expected_tool_name: str = "env_step",
) -> tuple[str, dict[str, Any]]:
    if not admissible_actions:
        raise ValueError("admissible_actions must be non-empty")

    fallback_action = choose_fallback_action(admissible_actions)
    normalized_actions = {normalize_action(action): action for action in admissible_actions}
    selected_tool_call = tool_calls[0] if tool_calls else None

    meta: dict[str, Any] = {
        "tool_call_count": len(tool_calls),
        "tool_name": None if selected_tool_call is None else selected_tool_call.name,
        "tool_arguments_raw": None if selected_tool_call is None else selected_tool_call.arguments,
        "raw_action": None,
        "parsed_action": None,
        "action_found": False,
        "action_is_string": False,
        "action_is_admissible": False,
        "invalid": False,
        "invalid_reason": None,
        "invalid_reasons": [],
        "fallback_used": False,
        "fallback_action": None,
    }

    if selected_tool_call is None:
        meta["invalid"] = True
        meta["invalid_reason"] = "missing_tool_call"
        meta["invalid_reasons"] = ["missing_tool_call"]
        meta["fallback_used"] = True
        meta["fallback_action"] = fallback_action
        return fallback_action, meta

    if selected_tool_call.name != expected_tool_name:
        meta["invalid"] = True
        meta["invalid_reason"] = "wrong_tool_name"
        meta["invalid_reasons"] = ["wrong_tool_name"]
        meta["fallback_used"] = True
        meta["fallback_action"] = fallback_action
        return fallback_action, meta

    try:
        arguments = json.loads(selected_tool_call.arguments)
    except Exception:
        meta["invalid"] = True
        meta["invalid_reason"] = "bad_arguments"
        meta["invalid_reasons"] = ["bad_arguments"]
        meta["fallback_used"] = True
        meta["fallback_action"] = fallback_action
        return fallback_action, meta

    if not isinstance(arguments, dict):
        meta["invalid"] = True
        meta["invalid_reason"] = "bad_arguments"
        meta["invalid_reasons"] = ["bad_arguments"]
        meta["fallback_used"] = True
        meta["fallback_action"] = fallback_action
        return fallback_action, meta

    if "action" not in arguments:
        meta["invalid"] = True
        meta["invalid_reason"] = "missing_action"
        meta["invalid_reasons"] = ["missing_action"]
        meta["fallback_used"] = True
        meta["fallback_action"] = fallback_action
        return fallback_action, meta

    meta["action_found"] = True
    action = arguments["action"]
    meta["raw_action"] = action

    if not isinstance(action, str):
        meta["invalid"] = True
        meta["invalid_reason"] = "non_string_action"
        meta["invalid_reasons"] = ["non_string_action"]
        meta["fallback_used"] = True
        meta["fallback_action"] = fallback_action
        return fallback_action, meta

    meta["action_is_string"] = True
    normalized_action = normalize_action(action)
    if normalized_action not in normalized_actions:
        meta["invalid"] = True
        meta["invalid_reason"] = "action_not_admissible"
        meta["invalid_reasons"] = ["action_not_admissible"]
        meta["fallback_used"] = True
        meta["fallback_action"] = fallback_action
        return fallback_action, meta

    meta["action_is_admissible"] = True
    meta["parsed_action"] = normalized_actions[normalized_action]
    return normalized_actions[normalized_action], meta
