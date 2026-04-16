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

import re


ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.IGNORECASE | re.DOTALL)
THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)


def normalize_action(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def choose_fallback_action(admissible_actions: list[str]) -> str:
    if not admissible_actions:
        raise ValueError("admissible_actions must be non-empty")

    for action in admissible_actions:
        if normalize_action(action) == "look":
            return action
    return admissible_actions[0]


def parse_action(
    output_text: str,
    admissible_actions: list[str],
    *,
    require_think_tags: bool = False,
) -> tuple[str, dict]:
    if not admissible_actions:
        raise ValueError("admissible_actions must be non-empty")

    normalized_actions = {normalize_action(action): action for action in admissible_actions}
    text = output_text or ""
    has_think = THINK_RE.search(text) is not None
    matches = ACTION_RE.findall(text)
    has_action = bool(matches)
    raw_action = matches[-1].strip() if matches else ""
    raw_action = raw_action.strip("`").strip().strip("\"'")
    normalized = normalize_action(raw_action)
    is_admissible = normalized in normalized_actions

    invalid_reasons: list[str] = []
    if require_think_tags and not has_think:
        invalid_reasons.append("missing_think")
    if not has_action:
        invalid_reasons.append("missing_action")
    elif not is_admissible:
        invalid_reasons.append("action_not_admissible")

    parse_error = (not has_action) or (not is_admissible)

    if is_admissible:
        return normalized_actions[normalized], {
            "raw_action": raw_action,
            "parse_error": parse_error,
            "fallback_used": False,
            "think_found": has_think,
            "action_found": has_action,
            "action_is_admissible": True,
            "invalid": bool(invalid_reasons),
            "invalid_reason": invalid_reasons[0] if invalid_reasons else None,
            "invalid_reasons": invalid_reasons,
        }

    fallback_action = choose_fallback_action(admissible_actions)
    return fallback_action, {
        "raw_action": raw_action,
        "parse_error": parse_error,
        "fallback_used": True,
        "fallback_action": fallback_action,
        "think_found": has_think,
        "action_found": has_action,
        "action_is_admissible": False,
        "invalid": bool(invalid_reasons),
        "invalid_reason": invalid_reasons[0] if invalid_reasons else None,
        "invalid_reasons": invalid_reasons,
    }
