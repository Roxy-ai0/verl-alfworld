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

import argparse
from pathlib import Path

from datasets import Dataset

from verl.experimental.alfworld.dataset import build_episode_record, discover_episodes


DEFAULT_ALFWORLD_DATA_ROOT = "/storage/v-jinpewang/az_workspace/zhanglin/reproduction/lwb/data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ALFWorld direct-prompt parquet files for verl Agent Loop.")
    parser.add_argument("--alfworld-data-root", default=DEFAULT_ALFWORLD_DATA_ROOT)
    parser.add_argument("--output-dir", default="data/alfworld_direct")
    parser.add_argument("--splits", nargs="+", default=["valid_seen", "valid_unseen"])
    parser.add_argument("--agent-name", default="alfworld_direct_agent")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        episodes = discover_episodes(args.alfworld_data_root, split, limit=args.limit)
        records = [
            build_episode_record(
                episode,
                agent_name=args.agent_name,
                max_steps=args.max_steps,
                alfworld_data_root=args.alfworld_data_root,
            )
            for episode in episodes
        ]
        if not records:
            raise RuntimeError(f"No ALFWorld episodes found for split={split!r}")

        output_path = output_dir / f"{split}.parquet"
        Dataset.from_list(records).to_parquet(str(output_path))
        print(f"[saved] split={split} count={len(records)} path={output_path}")


if __name__ == "__main__":
    main()
