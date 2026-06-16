"""Fully async GRPO entrypoint with separate actor/ref/rollout resource pools.

This keeps VERL's fully_async_policy architecture:

  rollouter -> MessageQueue -> trainer

and adds the split placement we need for local experiments:

  actor/update: trainer pool, usually one A800
  ref forward: optional dedicated ref pool, usually GPU3 A4000
  rollout: standalone vLLM replicas, usually GPUs4-7 A4000

The trainer also uses the async ref prefetch path added in
``fully_async_policy.fully_async_trainer`` so ref forward for the next batch can
overlap with actor update on the current batch.
"""

import asyncio
import os
import socket
import threading
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.experimental.fully_async_policy.message_queue import MessageQueue, MessageQueueClient
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.single_controller.ray import ResourcePoolManager
from verl.trainer.ppo.utils import Role, need_critic, need_reference_policy
from verl.utils.device import auto_set_device
from verl.utils.fs import copy_to_local


def _select_config_value(config, *paths):
    for path in paths:
        value = OmegaConf.select(config, path)
        if value is not None:
            return value
    return None


def create_split_role_worker_mapping(config):
    from verl.experimental.separation.engine_workers import DetachActorWorker
    from verl.single_controller.ray import RayWorkerGroup
    from verl.workers.engine_workers import TrainingWorker

    train_role = Role.ActorRollout if config.async_training.use_trainer_do_validate else Role.Actor
    role_worker_mapping = {
        train_role: ray.remote(DetachActorWorker),
        Role.Critic: ray.remote(TrainingWorker),
    }
    if need_reference_policy(config):
        role_worker_mapping[Role.RefPolicy] = ray.remote(DetachActorWorker)
    return role_worker_mapping, RayWorkerGroup


def create_split_resource_pool_manager(config, roles: list[Role]) -> ResourcePoolManager:
    """Create trainer-side resource pools with optional dedicated ref pool."""
    resource_pool_spec = {}
    mapping = {}

    train_roles = [Role.Actor, Role.ActorRollout, Role.Critic]
    if any(role in roles for role in train_roles):
        if config.trainer.n_gpus_per_node <= 0:
            raise ValueError("trainer.n_gpus_per_node must be greater than 0")
        if config.trainer.nnodes <= 0:
            raise ValueError("trainer.nnodes must be greater than 0")
        resource_pool_spec["actor_pool"] = [config.trainer.n_gpus_per_node] * config.trainer.nnodes
        for role in train_roles:
            if role in roles:
                mapping[role] = "actor_pool"

    if Role.RefPolicy in roles:
        ref_gpus = _select_config_value(
            config,
            "fully_async_split.ref_n_gpus_per_node",
            "split_sync.ref_n_gpus_per_node",
        )
        if ref_gpus is None:
            mapping[Role.RefPolicy] = "actor_pool"
        else:
            ref_nodes = _select_config_value(config, "fully_async_split.ref_nnodes", "split_sync.ref_nnodes") or 1
            if ref_gpus <= 0:
                raise ValueError("fully_async_split.ref_n_gpus_per_node must be greater than 0")
            if ref_nodes <= 0:
                raise ValueError("fully_async_split.ref_nnodes must be greater than 0")
            resource_pool_spec["ref_pool"] = [ref_gpus] * ref_nodes
            mapping[Role.RefPolicy] = "ref_pool"

    return ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)


@ray.remote(num_cpus=1)
class SplitFullyAsyncTaskRunner:
    def __init__(self):
        self.components = {}
        self.shutdown_event = threading.Event()

    def run(self, config):
        print("[SPLIT FULLY ASYNC] Starting fully async split GRPO training...")
        self._initialize_components(config)
        self._run_training_loop()

    def _initialize_components(self, config) -> None:
        print(f"[SPLIT FULLY ASYNC] TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        print("[SPLIT FULLY ASYNC] Initializing tokenizer/processor...")
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        self.components["tokenizer"] = tokenizer
        self.components["processor"] = processor
        self.components["config"] = config

        print("[SPLIT FULLY ASYNC] Creating split worker mapping/resource pools...")
        role_worker_mapping, ray_worker_group_cls = create_split_role_worker_mapping(config)
        self.components["role_worker_mapping"] = role_worker_mapping
        self.components["ray_worker_group_cls"] = ray_worker_group_cls

        print("[SPLIT FULLY ASYNC] Creating trainer...")
        self._create_trainer(config)

        print("[SPLIT FULLY ASYNC] Creating rollouter...")
        self._setup_hybrid_worker_group(config)
        self._create_rollouter(config)

        print("[SPLIT FULLY ASYNC] Connecting trainer and rollouter...")
        ray.get(self.components["trainer"].set_rollouter.remote(self.components["rollouter"]))

        total_train_steps = ray.get(self.components["rollouter"].get_total_train_steps.remote())
        print(f"[SPLIT FULLY ASYNC] total_train_steps={total_train_steps}")
        ray.get(self.components["trainer"].set_total_train_steps.remote(total_train_steps))

        max_queue_size = ray.get(self.components["rollouter"].get_max_queue_size.remote())
        print(f"[SPLIT FULLY ASYNC] Creating MessageQueue max_queue_size={max_queue_size}")
        message_queue = MessageQueue.remote(config, max_queue_size)
        message_queue_client = MessageQueueClient(message_queue)
        self.components["message_queue"] = message_queue
        self.components["message_queue_client"] = message_queue_client

        ray.get(self.components["rollouter"].set_message_queue_client.remote(message_queue_client))
        ray.get(self.components["trainer"].set_message_queue_client.remote(message_queue_client))

        ray.get(self.components["trainer"].load_checkpoint.remote())
        ray.get(self.components["rollouter"].load_checkpoint.remote())

        print("[SPLIT FULLY ASYNC] Initial parameter sync...")
        ray.get(self.components["trainer"]._fit_update_weights.remote())

        if config.trainer.get("val_before_train", True):
            ray.get(self.components["trainer"]._fit_validate.remote(True))

        print("[SPLIT FULLY ASYNC] Initialization complete")

    def _create_trainer(self, config) -> None:
        trainer_role_mapping = {
            role: worker_cls
            for role, worker_cls in self.components["role_worker_mapping"].items()
            if role != Role.Rollout
        }
        resource_pool_manager = create_split_resource_pool_manager(config, roles=list(trainer_role_mapping.keys()))
        trainer = FullyAsyncTrainer.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=trainer_role_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            device_name=config.trainer.device,
        )
        ray.get(trainer.init_workers.remote())
        self.components["trainer"] = trainer

    def _setup_hybrid_worker_group(self, config) -> None:
        if config.async_training.use_trainer_do_validate:
            trainer_wg = ray.get(self.components["trainer"].get_actor_wg.remote())
            self.components["hybrid_worker_group"] = trainer_wg
            print(
                "[SPLIT FULLY ASYNC] Hybrid trainer worker group extracted "
                f"(world_size={getattr(trainer_wg, 'world_size', '?')})"
            )

    def _create_rollouter(self, config) -> None:
        rollouter = FullyAsyncRollouter.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
        )
        if "hybrid_worker_group" in self.components:
            ray.get(rollouter.set_hybrid_worker_group.remote(self.components["hybrid_worker_group"]))
        ray.get(rollouter.init_workers.remote())
        ray.get(rollouter.set_max_required_samples.remote())
        self.components["rollouter"] = rollouter

    def _run_training_loop(self):
        print("[SPLIT FULLY ASYNC] Starting rollouter/trainer loops...")
        rollouter_future = self.components["rollouter"].fit.remote()
        trainer_future = self.components["trainer"].fit.remote()
        futures = [rollouter_future, trainer_future]

        try:
            while futures:
                done_futures, remaining_futures = ray.wait(futures, num_returns=1, timeout=None)
                for future in done_futures:
                    try:
                        ray.get(future)
                        print("[SPLIT FULLY ASYNC] One component completed successfully")
                    except Exception as exc:
                        print(f"[SPLIT FULLY ASYNC] Component failed: {exc}")
                        for remaining in remaining_futures:
                            ray.cancel(remaining)
                        raise
                futures = remaining_futures
        finally:
            asyncio.run(self.components["message_queue_client"].clear_queue())
            print("[SPLIT FULLY ASYNC] Training completed or interrupted")


@hydra.main(config_path="../verl/verl/experimental/fully_async_policy/config", config_name="fully_async_ppo_trainer", version_base=None)
def main(config):
    from time import time

    from verl.trainer.main_ppo import run_ppo

    if not hasattr(config, "async_training"):
        raise RuntimeError("must set async_training config")

    start_time = time()
    auto_set_device(config)
    config.actor_rollout_ref.rollout.nnodes = config.rollout.nnodes
    config.actor_rollout_ref.rollout.n_gpus_per_node = config.rollout.n_gpus_per_node
    config = migrate_legacy_reward_impl(config)
    run_ppo(config, task_runner_class=SplitFullyAsyncTaskRunner)
    print(f"total time: {time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
