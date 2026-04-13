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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _get_templated_task_desc(traj_data: dict[str, Any]) -> str:
    from alfworld.agents.utils.misc import get_templated_task_desc

    return get_templated_task_desc(traj_data)


@dataclass
class ALFWorldEpisode:
    split: str
    task_id: str
    gamefile: str
    traj_data_path: str
    task_description: str
    task_type: str
    scene_num: int | None
    random_seed: int | None


def resolve_split_dir(data_root: str | Path, split: str) -> Path:
    data_root = Path(data_root).expanduser()
    candidate = data_root / "json_2.1.1" / split
    if candidate.exists():
        return candidate

    candidate = data_root / split
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Cannot locate ALFWorld split directory for split={split!r} under data_root={str(data_root)!r}. "
        "Expected either '<data_root>/json_2.1.1/<split>' or '<data_root>/<split>'."
    )


def extract_task_description(traj_data: dict[str, Any], prefer_human_annotation: bool = True) -> str:
    if prefer_human_annotation:
        annotations = traj_data.get("turk_annotations", {}).get("anns", [])
        if annotations:
            task_desc = annotations[0].get("task_desc")
            if task_desc:
                return str(task_desc).strip()
    return _get_templated_task_desc(traj_data).strip()


def discover_episodes(
    data_root: str | Path,
    split: str,
    *,
    prefer_human_annotation: bool = True,
    limit: int | None = None,
) -> list[ALFWorldEpisode]:
    split_dir = resolve_split_dir(data_root, split)
    episodes: list[ALFWorldEpisode] = []

    for traj_path in sorted(split_dir.rglob("traj_data.json")):
        gamefile = traj_path.parent / "game.tw-pddl"
        if not gamefile.exists():
            continue

        with traj_path.open("r", encoding="utf-8") as f:
            traj_data = json.load(f)

        task_description = extract_task_description(traj_data, prefer_human_annotation=prefer_human_annotation)
        scene = traj_data.get("scene", {})
        episodes.append(
            ALFWorldEpisode(
                split=split,
                task_id=str(traj_data.get("task_id", traj_path.parent.name)),
                gamefile=str(gamefile),
                traj_data_path=str(traj_path),
                task_description=task_description,
                task_type=str(traj_data.get("task_type", "")),
                scene_num=scene.get("scene_num"),
                random_seed=scene.get("random_seed"),
            )
        )

        if limit is not None and len(episodes) >= limit:
            break

    return episodes


def build_episode_record(
    episode: ALFWorldEpisode,
    *,
    agent_name: str = "alfworld_direct_agent",
    max_steps: int = 30,
    alfworld_data_root: str | Path | None = None,
) -> dict[str, Any]:
    extra_info = asdict(episode)
    extra_info["max_steps"] = max_steps
    if alfworld_data_root is not None:
        extra_info["alfworld_data_root"] = str(Path(alfworld_data_root).expanduser())

    return {
        "data_source": "alfworld_text",
        "agent_name": agent_name,
        "prompt": [
            {
                "role": "user",
                "content": "Run one direct-prompt ALFWorld episode from the provided metadata.",
            }
        ],
        "ability": "agent",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": extra_info,
    }
