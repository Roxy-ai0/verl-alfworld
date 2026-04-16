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

from verl.experimental.alfworld.parser import parse_action


def test_parse_action_requires_think_tag():
    action, meta = parse_action(
        "<action>look</action>",
        ["look", "inventory"],
        require_think_tags=True,
    )

    assert action == "look"
    assert meta["fallback_used"] is False
    assert meta["invalid"] is True
    assert meta["invalid_reason"] == "missing_think"


def test_parse_action_reports_missing_action():
    action, meta = parse_action(
        "<think>I should inspect the room.</think>",
        ["look", "inventory"],
        require_think_tags=True,
    )

    assert action == "look"
    assert meta["fallback_used"] is True
    assert meta["invalid"] is True
    assert meta["invalid_reason"] == "missing_action"


def test_parse_action_reports_non_admissible_action():
    action, meta = parse_action(
        "<think>I will try something.</think><action>open fridge</action>",
        ["look", "inventory"],
        require_think_tags=True,
    )

    assert action == "look"
    assert meta["fallback_used"] is True
    assert meta["invalid"] is True
    assert meta["invalid_reason"] == "action_not_admissible"
