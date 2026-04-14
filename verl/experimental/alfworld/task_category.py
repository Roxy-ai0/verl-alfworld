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

from typing import Any


TASK_GROUPS = {
    "pick_and_place": [
        "pick_and_place_simple",
        "pick_and_place_with_movable_recep",
    ],
    "pick_two_obj_and_place": [
        "pick_two_obj_and_place",
    ],
    "look_at_obj_in_light": [
        "look_at_obj_in_light",
    ],
    "pick_clean_then_place_in_recep": [
        "pick_clean_then_place_in_recep",
    ],
    "pick_heat_then_place_in_recep": [
        "pick_heat_then_place_in_recep",
    ],
    "pick_cool_then_place_in_recep": [
        "pick_cool_then_place_in_recep",
    ],
}

UNKNOWN_TASK_CATEGORY = "unknown"

TASK_ALIAS_TO_GROUP = {
    alias: group_name
    for group_name, aliases in TASK_GROUPS.items()
    for alias in aliases
}


def infer_alfworld_task_category(
    task_id: str = "",
    gamefile: str = "",
    task_description: str = "",
    task_type: str = "",
) -> str:
    candidate_fields = [
        str(task_id or "").lower(),
        str(gamefile or "").lower(),
        str(task_description or "").lower(),
        str(task_type or "").lower(),
    ]

    for field in candidate_fields:
        for alias, group_name in TASK_ALIAS_TO_GROUP.items():
            if alias in field:
                return group_name

    return UNKNOWN_TASK_CATEGORY


def build_category_success_summary(episode_results: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    summary = {
        category: {"success": 0, "total": 0, "success_rate": 0.0}
        for category in [*TASK_GROUPS.keys(), UNKNOWN_TASK_CATEGORY]
    }

    for result in episode_results:
        category = str(result.get("task_category", UNKNOWN_TASK_CATEGORY))
        if category not in summary:
            category = UNKNOWN_TASK_CATEGORY

        summary[category]["total"] += 1
        summary[category]["success"] += int(bool(result.get("success", False)))

    for category_stats in summary.values():
        total = category_stats["total"]
        category_stats["success_rate"] = category_stats["success"] / total if total > 0 else 0.0

    return summary
