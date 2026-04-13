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

from dataclasses import dataclass
from typing import Any

import textworld
import textworld.gym

from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos


@dataclass
class ALFWorldState:
    observation: str
    admissible_actions: list[str]
    reward: float
    done: bool
    won: bool
    info: dict[str, Any]


def _extract_single(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def _extract_info_field(info: dict[str, Any], key: str, default: Any) -> Any:
    value = info.get(key, default)
    return _extract_single(value)


class ALFWorldTextEnv:
    def __init__(self, gamefile: str, *, max_steps: int = 30, domain_randomization: bool = False):
        self.gamefile = gamefile
        self.max_steps = max_steps
        self.domain_randomization = domain_randomization
        self._env = None
        self._env_id = None

    def _build_env(self):
        wrappers = [AlfredDemangler(shuffle=self.domain_randomization), AlfredInfos]
        request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
        env_id = textworld.gym.register_games(
            [self.gamefile],
            request_infos,
            batch_size=1,
            asynchronous=True,
            max_episode_steps=self.max_steps,
            wrappers=wrappers,
        )
        env = textworld.gym.make(env_id)
        self._env_id = env_id
        self._env = env
        return env

    def reset(self) -> ALFWorldState:
        env = self._env or self._build_env()
        observations, info = env.reset()
        observation = str(_extract_single(observations))
        admissible_actions = list(_extract_info_field(info, "admissible_commands", []))
        won = bool(_extract_info_field(info, "won", False))
        return ALFWorldState(
            observation=observation,
            admissible_actions=admissible_actions,
            reward=0.0,
            done=False,
            won=won,
            info={"gamefile": _extract_info_field(info, "extra.gamefile", self.gamefile), **info},
        )

    def step(self, action: str) -> ALFWorldState:
        if self._env is None:
            raise RuntimeError("Environment has not been reset.")

        observations, scores, dones, infos = self._env.step([action])
        observation = str(_extract_single(observations))
        reward = float(_extract_single(scores))
        done = bool(_extract_single(dones))
        admissible_actions = list(_extract_info_field(infos, "admissible_commands", []))
        won = bool(_extract_info_field(infos, "won", False))

        return ALFWorldState(
            observation=observation,
            admissible_actions=admissible_actions,
            reward=reward,
            done=done,
            won=won,
            info=infos,
        )

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
