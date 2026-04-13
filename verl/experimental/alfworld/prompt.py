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
