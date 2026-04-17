import asyncio
from typing import Any

import hydra
import numpy as np
import ray

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopOutput,
    AgentLoopWorker,
    DictConfigWrap,
    RolloutTraceConfig,
    _agent_loop_registry,
    get_trajectory_info,
    rollout_trace_attr,
)
from verl.protocol import DataProto


class ALFWorldStepwiseAgentLoopWorker(AgentLoopWorker):
    async def generate_sequences(self, batch: DataProto) -> DataProto:
        config = self.rollout_config
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        validate = bool(batch.meta_info.get("validate", False))
        if validate:
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature

        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        max_samples_per_worker = RolloutTraceConfig.get_instance().max_samples_per_step_per_worker
        if max_samples_per_worker is not None:
            unique_sample_indices = np.unique(index)
            if max_samples_per_worker < len(unique_sample_indices):
                selected_samples = set(
                    np.random.choice(unique_sample_indices, max_samples_per_worker, replace=False).tolist()
                )
                traced_indices = set(i for i in range(len(batch)) if index[i] in selected_samples)
            else:
                traced_indices = set(range(len(batch)))
        else:
            traced_indices = set(range(len(batch)))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), validate
        )

        tasks = []
        for i in range(len(batch)):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            tasks.append(
                asyncio.create_task(
                    self._run_agent_loop_stepwise(
                        sampling_params,
                        trajectory_info[i],
                        agent_name=str(kwargs["agent_name"]),
                        trace=i in traced_indices,
                        **kwargs,
                    )
                )
            )

        results = await asyncio.gather(*tasks)
        outputs = [item for group in results for item in group]
        output = self._postprocess(outputs, input_non_tensor_batch=None, validate=validate)
        output.meta_info["replace_training_batch"] = not validate
        return output

    async def _run_agent_loop_stepwise(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        trace: bool = True,
        **kwargs,
    ) -> list:
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
            trace=trace,
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )

            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=DictConfigWrap(config=self.config),
                server_manager=self.server_manager,
                tokenizer=self.tokenizer,
                processor=self.processor,
                dataset_cls=self.dataset_cls,
                data_config=DictConfigWrap(self.config.data),
            )

            result = await agent_loop.run(sampling_params, validate=trajectory["validate"], **kwargs)
            outputs = result if isinstance(result, list) else [result]
            return [
                await self._agent_loop_postprocess(output, trajectory["validate"], **kwargs)
                for output in outputs
            ]


class ALFWorldStepwiseAgentLoopManager(AgentLoopManager):
    agent_loop_workers_class = ray.remote(ALFWorldStepwiseAgentLoopWorker)
