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


def test_parse_action_exact_match():
    action, meta = parse_action(
        "<think>go there</think>\n<action>go to fridge 1</action>",
        ["look", "go to fridge 1"],
    )
    assert action == "go to fridge 1"
    assert meta["fallback_used"] is False


def test_parse_action_fallback_to_look():
    action, meta = parse_action(
        "<think>invalid</think>\n<action>fly to moon</action>",
        ["open cabinet 1", "look", "inventory"],
    )
    assert action == "look"
    assert meta["fallback_used"] is True
