from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from verl.experimental.alfworld.task_category import TASK_GROUPS, UNKNOWN_TASK_CATEGORY


def _flatten_numeric_sequences(values: Any) -> list[float]:
    flattened: list[float] = []
    if values is None:
        return flattened

    if isinstance(values, np.ndarray):
        values = values.tolist()

    if isinstance(values, (list, tuple)):
        for value in values:
            flattened.extend(_flatten_numeric_sequences(value))
        return flattened

    if isinstance(values, torch.Tensor):
        if values.numel() == 1:
            return [float(values.detach().item())]
        return [float(v) for v in values.detach().cpu().flatten().tolist()]

    if isinstance(values, np.generic):
        return [float(values.item())]

    if isinstance(values, (bool, int, float)):
        return [float(values)]

    return flattened


def _to_numeric_array(values: Any) -> np.ndarray:
    flattened = _flatten_numeric_sequences(values)
    if not flattened:
        return np.array([], dtype=np.float32)
    return np.asarray(flattened, dtype=np.float32)


def _to_object_list(values: Any) -> list[Any]:
    if values is None:
        return []
    if isinstance(values, np.ndarray):
        return values.tolist()
    if isinstance(values, list):
        return values
    if isinstance(values, tuple):
        return list(values)
    return [values]


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size > 0 else 0.0


def _safe_std(values: np.ndarray) -> float:
    return float(np.std(values)) if values.size > 0 else 0.0


def _safe_max(values: np.ndarray) -> float:
    return float(np.max(values)) if values.size > 0 else 0.0


def _safe_min(values: np.ndarray) -> float:
    return float(np.min(values)) if values.size > 0 else 0.0


def _get_reward_array(reward_extra_infos_dict: dict[str, Any], *keys: str) -> np.ndarray:
    for key in keys:
        if key in reward_extra_infos_dict:
            array = _to_numeric_array(reward_extra_infos_dict[key])
            if array.size > 0:
                return array
    return np.array([], dtype=np.float32)


def extract_alfworld_episode_metrics(
    reward_extra_infos_dict: dict[str, Any],
    *,
    split: str,
    batch: Any = None,
) -> dict[str, float]:
    success = _get_reward_array(reward_extra_infos_dict, "acc", "success")
    final_reward = _get_reward_array(reward_extra_infos_dict, "final_reward")
    invalid_action_count = _get_reward_array(reward_extra_infos_dict, "invalid_action_count")
    invalid_rate = _get_reward_array(reward_extra_infos_dict, "invalid_rate")
    num_env_steps = _get_reward_array(reward_extra_infos_dict, "num_env_steps")
    prompt_length = _get_reward_array(reward_extra_infos_dict, "prompt_length")
    response_length = _get_reward_array(reward_extra_infos_dict, "response_length")
    episode_reward = _get_reward_array(reward_extra_infos_dict, "episode_reward")
    mean_step_score = _get_reward_array(reward_extra_infos_dict, "mean_step_score")

    metrics = {
        f"{split}/success_rate": _safe_mean(success),
        f"{split}/avg_final_reward": _safe_mean(final_reward),
        f"{split}/avg_steps": _safe_mean(num_env_steps),
        f"{split}/prompt_length_mean": _safe_mean(prompt_length),
        f"{split}/prompt_length_max": _safe_max(prompt_length),
        f"{split}/prompt_length_min": _safe_min(prompt_length),
        f"{split}/response_length_mean": _safe_mean(response_length),
        f"{split}/response_length_max": _safe_max(response_length),
        f"{split}/response_length_min": _safe_min(response_length),
        f"{split}/episode_reward_mean": _safe_mean(episode_reward),
        f"{split}/episode_reward_max": _safe_max(episode_reward),
        f"{split}/episode_reward_min": _safe_min(episode_reward),
        f"{split}/invalid_action_count_mean": _safe_mean(invalid_action_count),
        f"{split}/invalid_action_count_max": _safe_max(invalid_action_count),
        f"{split}/score_mean": _safe_mean(mean_step_score),
        f"{split}/score_std": _safe_std(mean_step_score),
    }

    if invalid_rate.size > 0:
        metrics[f"{split}/invalid_action_rate"] = _safe_mean(invalid_rate)
    elif invalid_action_count.size > 0 and num_env_steps.size > 0:
        denom = np.maximum(num_env_steps, 1.0)
        metrics[f"{split}/invalid_action_rate"] = float(np.mean(invalid_action_count / denom))
    else:
        metrics[f"{split}/invalid_action_rate"] = 0.0

    if batch is not None:
        advantages = batch.batch.get("advantages", None)
        response_mask = batch.batch.get("response_mask", None)
        if advantages is not None and response_mask is not None:
            valid_advantages = advantages[response_mask.bool()]
            if valid_advantages.numel() > 0:
                metrics[f"{split}/advantage_mean"] = float(valid_advantages.mean().item())
                metrics[f"{split}/advantage_std"] = float(valid_advantages.std(unbiased=False).item())
            else:
                metrics[f"{split}/advantage_mean"] = 0.0
                metrics[f"{split}/advantage_std"] = 0.0

        step_scores = batch.non_tensor_batch.get("step_scores", None)
        if step_scores is not None:
            step_score_values = _to_numeric_array(step_scores)
            if step_score_values.size > 0:
                metrics[f"{split}/score_mean"] = _safe_mean(step_score_values)
                metrics[f"{split}/score_std"] = _safe_std(step_score_values)

    return metrics


def extract_eval_category_metrics(reward_extra_infos_dict: dict[str, Any]) -> dict[str, float]:
    task_categories = [str(v) for v in _to_object_list(reward_extra_infos_dict.get("task_category", []))]
    success = _get_reward_array(reward_extra_infos_dict, "acc", "success")
    if not task_categories or success.size == 0:
        return {f"eval/category_success/{category}": 0.0 for category in TASK_GROUPS}

    category_to_values = {category: [] for category in TASK_GROUPS}
    for idx, category in enumerate(task_categories):
        normalized = category if category in category_to_values else UNKNOWN_TASK_CATEGORY
        if normalized == UNKNOWN_TASK_CATEGORY:
            continue
        if idx < success.size:
            category_to_values[normalized].append(float(success[idx]))

    metrics = {}
    for category in TASK_GROUPS:
        vals = np.asarray(category_to_values[category], dtype=np.float32)
        metrics[f"eval/category_success/{category}"] = float(np.mean(vals)) if vals.size > 0 else 0.0
    return metrics


def extract_train_loss_alias_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    aliases: dict[str, float] = {}
    mapping = {
        "actor/loss": "train/loss/actor",
        "actor/pg_loss": "train/loss/policy",
        "actor/kl_loss": "train/loss/kl",
        "actor/entropy": "train/loss/entropy",
        "actor/lr": "train/lr",
    }
    for source_key, target_key in mapping.items():
        value = metrics.get(source_key)
        if value is not None:
            aliases[target_key] = float(value)
    return aliases


@dataclass
class _RunningStats:
    total: float = 0.0
    count: int = 0
    min_value: float | None = None
    max_value: float | None = None

    def update(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        self.total += float(values.sum())
        self.count += int(values.size)
        current_min = float(values.min())
        current_max = float(values.max())
        self.min_value = current_min if self.min_value is None else min(self.min_value, current_min)
        self.max_value = current_max if self.max_value is None else max(self.max_value, current_max)

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count > 0 else 0.0


@dataclass
class ALFWorldWandbHelper:
    tracking: Any
    configured: bool = False
    running_stats: dict[str, _RunningStats] = field(default_factory=dict)

    @staticmethod
    def _step_metric_name(metric_name: str) -> str:
        if metric_name.startswith("train/"):
            return f"train_step/{metric_name[len('train/'):]}"
        if metric_name.startswith("eval/"):
            return f"eval_step/{metric_name[len('eval/'):]}"
        return metric_name

    @staticmethod
    def _with_axis_metrics(metrics: dict[str, Any], *, global_step: int, epoch: int) -> dict[str, Any]:
        payload = dict(metrics)
        payload["trainer/global_step"] = global_step
        payload["trainer/epoch"] = epoch
        payload["trainer/episode"] = epoch
        return payload

    def build_step_metrics(self, metrics: dict[str, Any], *, global_step: int, epoch: int) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in metrics.items():
            if key.startswith("train/") or key.startswith("eval/"):
                payload[self._step_metric_name(key)] = value
            else:
                payload[key] = value
        return self._with_axis_metrics(payload, global_step=global_step, epoch=epoch)

    def build_epoch_metrics(self, metrics: dict[str, Any], *, global_step: int, epoch: int) -> dict[str, Any]:
        return self._with_axis_metrics(metrics, global_step=global_step, epoch=epoch)

    def _wandb(self):
        if self.tracking is None:
            return None
        if hasattr(self.tracking, "get_backend"):
            return self.tracking.get_backend("wandb")
        if hasattr(self.tracking, "logger"):
            return self.tracking.logger.get("wandb")
        return None

    def configure(self) -> None:
        wandb = self._wandb()
        if wandb is None or self.configured:
            return

        wandb.define_metric("trainer/global_step")
        wandb.define_metric("trainer/epoch")
        wandb.define_metric("trainer/episode")
        for prefix, step_metric in {
            "train": "trainer/epoch",
            "eval": "trainer/epoch",
            "train_step": "trainer/global_step",
            "eval_step": "trainer/global_step",
        }.items():
            wandb.define_metric(f"{prefix}/*", step_metric=step_metric)

        summary_metrics = {
            "train/success_rate": "max",
            "eval/success_rate": "max",
            "train/invalid_action_rate": "min",
            "eval/invalid_action_rate": "min",
            "train/avg_final_reward": "max",
            "eval/avg_final_reward": "max",
            "train/prompt_length_mean": "mean",
            "eval/prompt_length_mean": "mean",
            "train/response_length_mean": "mean",
            "eval/response_length_mean": "mean",
        }
        for metric_name, summary in summary_metrics.items():
            wandb.define_metric(metric_name, step_metric="trainer/epoch", summary=summary)
            wandb.define_metric(self._step_metric_name(metric_name), step_metric="trainer/global_step", summary=summary)

        for prefix, step_metric in [
            ("train", "trainer/epoch"),
            ("eval", "trainer/epoch"),
            ("train_step", "trainer/global_step"),
            ("eval_step", "trainer/global_step"),
        ]:
            for metric_name in [
                "episode_reward_mean",
                "episode_reward_max",
                "episode_reward_min",
                "score_mean",
                "score_std",
                "avg_steps",
                "invalid_action_count_mean",
                "invalid_action_count_max",
                "prompt_length_max",
                "prompt_length_min",
                "response_length_max",
                "response_length_min",
            ]:
                wandb.define_metric(f"{prefix}/{metric_name}", step_metric=step_metric)

        for category in TASK_GROUPS:
            wandb.define_metric(f"eval/category_success/{category}", step_metric="trainer/epoch", summary="last")
            wandb.define_metric(
                f"eval_step/category_success/{category}", step_metric="trainer/global_step", summary="last"
            )

        self.configured = True

    def update_summary(self, split: str, reward_extra_infos_dict: dict[str, Any], metrics: dict[str, Any]) -> None:
        wandb = self._wandb()
        if wandb is None or getattr(wandb, "run", None) is None:
            return

        self.update_raw_summary(split, reward_extra_infos_dict)

        success_metric_key = f"{split}/success_rate"
        if success_metric_key in metrics:
            current = float(metrics[success_metric_key])
            split_best_key = f"{split}/best_success_rate"
            prev = wandb.run.summary.get(split_best_key)
            wandb.run.summary[split_best_key] = current if prev is None else max(prev, current)
            prev_global = wandb.run.summary.get("best_success_rate")
            wandb.run.summary["best_success_rate"] = current if prev_global is None else max(prev_global, current)
            wandb.run.summary["success_rate/best"] = wandb.run.summary["best_success_rate"]

        invalid_metric_key = f"{split}/invalid_action_rate"
        if invalid_metric_key in metrics:
            current = float(metrics[invalid_metric_key])
            split_min_key = f"{split}/min_invalid_action_rate"
            prev_split = wandb.run.summary.get(split_min_key)
            wandb.run.summary[split_min_key] = current if prev_split is None else min(prev_split, current)
            prev = wandb.run.summary.get("min_invalid_action_rate")
            wandb.run.summary["min_invalid_action_rate"] = current if prev is None else min(prev, current)
            wandb.run.summary["invalid_action_rate/min"] = wandb.run.summary["min_invalid_action_rate"]

        if split == "eval":
            for category in TASK_GROUPS:
                key = f"eval/category_success/{category}"
                if key in metrics:
                    wandb.run.summary[key] = float(metrics[key])

    def update_raw_summary(self, split: str, reward_extra_infos_dict: dict[str, Any]) -> None:
        wandb = self._wandb()
        if wandb is None or getattr(wandb, "run", None) is None:
            return

        reward_stats = self.running_stats.setdefault(f"{split}/episode_reward", _RunningStats())
        prompt_stats = self.running_stats.setdefault(f"{split}/prompt_length", _RunningStats())
        response_stats = self.running_stats.setdefault(f"{split}/response_length", _RunningStats())

        episode_reward = _get_reward_array(reward_extra_infos_dict, "episode_reward")
        prompt_length = _get_reward_array(reward_extra_infos_dict, "prompt_length")
        response_length = _get_reward_array(reward_extra_infos_dict, "response_length")

        reward_stats.update(episode_reward)
        prompt_stats.update(prompt_length)
        response_stats.update(response_length)

        if reward_stats.count > 0:
            wandb.run.summary[f"{split}/episode_reward/max"] = reward_stats.max_value
            wandb.run.summary[f"{split}/episode_reward/min"] = reward_stats.min_value
            wandb.run.summary[f"{split}/episode_reward/mean"] = reward_stats.mean
            wandb.run.summary["episode_reward/max"] = reward_stats.max_value
            wandb.run.summary["episode_reward/min"] = reward_stats.min_value
            wandb.run.summary["episode_reward/mean"] = reward_stats.mean

        if prompt_stats.count > 0:
            wandb.run.summary[f"{split}/prompt_length/max"] = prompt_stats.max_value
            wandb.run.summary[f"{split}/prompt_length/min"] = prompt_stats.min_value
            wandb.run.summary[f"{split}/prompt_length/mean"] = prompt_stats.mean
            wandb.run.summary["prompt_length/max"] = prompt_stats.max_value
            wandb.run.summary["prompt_length/min"] = prompt_stats.min_value

        if response_stats.count > 0:
            wandb.run.summary[f"{split}/response_length/max"] = response_stats.max_value
            wandb.run.summary[f"{split}/response_length/min"] = response_stats.min_value
            wandb.run.summary[f"{split}/response_length/mean"] = response_stats.mean
            wandb.run.summary["response_length/max"] = response_stats.max_value
            wandb.run.summary["response_length/min"] = response_stats.min_value
