"""Synchronous GRPO entrypoint with trainer and rollout on separate GPUs.

This runner keeps the synchronous ``main_ppo_sync`` training loop unchanged, but
initializes the vLLM rollout server in standalone mode. The actor/ref worker
group is placed by ``trainer.n_gpus_per_node``; the rollout server is placed by
``actor_rollout_ref.rollout.nnodes`` and ``n_gpus_per_node``.
"""

import os
import socket
from functools import partial
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf

try:
    import transfer_queue as tq
except ImportError:
    from verl.utils.transferqueue_utils import tq

from verl.checkpoint_engine import CheckpointEngineManager, CheckpointEngineWorker
from verl.experimental.reward_loop import RewardLoopManager, migrate_legacy_reward_impl
from verl.protocol import BatchData
from verl.experimental.teacher_loop import MultiTeacherModelManager
from verl.trainer.distillation import is_distillation_enabled
from verl.trainer.main_ppo import run_ppo
from verl.trainer.main_ppo_sync import AgentLoopManagerTQ, PPOTrainer
from verl.trainer.ppo.utils import Role, need_critic, need_reference_policy, need_teacher_policy
from verl.utils.config import omega_conf_to_dataclass, validate_config
from verl.utils.device import auto_set_device
from verl.utils.import_utils import load_class_from_fqn
from verl.workers.config import CriticConfig, DistillationConfig
from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker, TrainingWorkerConfig
from verl.workers.rollout.llm_server import LLMServerManager
from verl.workers.utils.losses import value_loss
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayWorkerGroup,
    ResourcePoolManager,
    create_colocated_worker_cls,
)
from verl.single_controller.ray.base import split_resource_pool


class RefFanoutWorkerGroup:
    """Fan out ref logprob over independent single-GPU ref replicas."""

    def __init__(self, worker_groups, remote_method_name: str):
        self.worker_groups = worker_groups
        self.remote_method_name = remote_method_name

    @property
    def world_size(self):
        return sum(wg.world_size for wg in self.worker_groups)

    @property
    def workers(self):
        return [worker for wg in self.worker_groups for worker in wg.workers]

    def execute_all_sync(self, method_name: str, *args, **kwargs):
        output = []
        for wg in self.worker_groups:
            output.extend(wg.execute_all_sync(method_name, *args, **kwargs))
        return output

    def compute_ref_log_prob(self, batch):
        chunks = BatchData(batch).chunk(len(self.worker_groups))
        refs = []
        for wg, chunk in zip(self.worker_groups, chunks, strict=True):
            if wg.world_size != 1:
                raise ValueError("RefFanoutWorkerGroup currently expects single-rank ref replicas.")
            refs.extend(wg.execute_all_async(self.remote_method_name, chunk))
        return BatchData(ray.get(refs)).concat()


class SplitSyncPPOTrainer(PPOTrainer):
    """PPO/GRPO trainer with standalone rollout replicas."""

    def init_workers(self):
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        lora_rank = self.config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = self.config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or self.config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        if self.use_reference_policy and self.ref_in_actor:
            raise NotImplementedError("split-sync runner is intended for full-parameter GRPO, not LoRA ref-in-actor.")

        actor_pool = self.resource_pool_manager.get_resource_pool(Role.Actor)
        actor_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.Actor],
            config=self.config.actor_rollout_ref,
            distillation_config=self.config.get("distillation"),
            role=str(Role.Actor),
        )
        self.resource_pool_to_cls[actor_pool][str(Role.Actor)] = actor_cls

        if self.use_reference_policy:
            ref_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            ref_replica_size = OmegaConf.select(self.config, "split_sync.ref_replica_size")
            self.ref_fanout_pools = None
            self.ref_fanout_cls = None
            if ref_replica_size is not None and ref_replica_size < ref_pool.world_size:
                if ref_replica_size <= 0:
                    raise ValueError("split_sync.ref_replica_size must be greater than 0")
                self.ref_fanout_pools = split_resource_pool(ref_pool, int(ref_replica_size))
                self.ref_fanout_cls = ref_cls
            else:
                self.resource_pool_to_cls[ref_pool][str(Role.RefPolicy)] = ref_cls

        if self.use_critic:
            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)
            critic_cfg.engine.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
            critic_cfg.engine.max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
            worker_cfg = TrainingWorkerConfig(
                model_type="value_model",
                model_config=critic_cfg.model_config,
                engine_config=critic_cfg.engine,
                optimizer_config=critic_cfg.optim,
                checkpoint_config=critic_cfg.checkpoint,
            )
            critic_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=worker_cfg)
            self.resource_pool_to_cls[critic_pool][str(Role.Critic)] = critic_cls

        all_wg = {}
        wg_kwargs = {"device_name": self.config.trainer.device}
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = RayWorkerGroup(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            all_wg.update(wg_dict.spawn(prefix_set=class_dict.keys()))

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.reset()
            value_loss_ = partial(value_loss, config=critic_cfg)
            self.critic_wg.set_loss_fn(value_loss_)

        self.actor_rollout_wg = all_wg[str(Role.Actor)]
        self.actor_rollout_wg.init_model()

        if self.use_reference_policy:
            if self.ref_fanout_pools is not None:
                ref_role_name = str(Role.RefPolicy)
                ref_wgs = []
                for idx, ref_pool in enumerate(self.ref_fanout_pools):
                    worker_dict_cls = create_colocated_worker_cls(class_dict={ref_role_name: self.ref_fanout_cls})
                    wg_dict = RayWorkerGroup(
                        resource_pool=ref_pool,
                        ray_cls_with_init=worker_dict_cls,
                        name_prefix=f"refrep{idx}_",
                        **wg_kwargs,
                    )
                    ref_wg = wg_dict.spawn(prefix_set={ref_role_name})[ref_role_name]
                    ref_wg.init_model()
                    ref_wgs.append(ref_wg)
                self.ref_policy_wg = RefFanoutWorkerGroup(
                    worker_groups=ref_wgs,
                    remote_method_name=f"{ref_role_name}_compute_ref_log_prob",
                )
            else:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()

        resource_pool = (
            self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            if self.config.reward.reward_model.enable
            else None
        )
        self.reward_loop_manager = RewardLoopManager(config=self.config, rm_resource_pool=resource_pool)

        if self.use_teacher_policy:
            teacher_resource_pool = self.resource_pool_manager.get_resource_pool(Role.TeacherModel)
            self.teacher_model_manager = MultiTeacherModelManager(
                config=self.config,
                resource_pool=teacher_resource_pool,
            )
            self.distillation_config: DistillationConfig = omega_conf_to_dataclass(self.config.distillation)
        else:
            self.teacher_model_manager = None
            self.distillation_config = None

        self.llm_server_manager = LLMServerManager.create(config=self.config)

        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            agent_loop_manager_cls = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            agent_loop_manager_cls = AgentLoopManagerTQ
        self.async_rollout_manager = agent_loop_manager_cls.create(
            config=self.config,
            llm_client=self.llm_server_manager.get_client(),
            teacher_client=self.teacher_model_manager.get_client() if self.use_teacher_policy else None,
            reward_loop_worker_handles=self.reward_loop_manager.reward_loop_workers,
            replay_buffer=self.replay_buffer,
        )

        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            trainer=self.actor_rollout_wg,
            replicas=self.llm_server_manager.get_replicas(),
        )

        self._log_worker_visible_devices()
        self.checkpoint_manager.sleep_replicas()

    def _log_worker_visible_devices(self):
        try:
            actor_visible = self.actor_rollout_wg.execute_all_sync("get_cuda_visible_devices")
            print(f"[split-sync] actor CUDA_VISIBLE_DEVICES={actor_visible}")
        except Exception as exc:
            print(f"[split-sync] failed to query actor CUDA_VISIBLE_DEVICES: {exc}")

        if self.use_reference_policy:
            try:
                ref_visible = self.ref_policy_wg.execute_all_sync("get_cuda_visible_devices")
                print(f"[split-sync] ref CUDA_VISIBLE_DEVICES={ref_visible}")
            except Exception as exc:
                print(f"[split-sync] failed to query ref CUDA_VISIBLE_DEVICES: {exc}")

        try:
            workers = []
            for replica in self.llm_server_manager.get_replicas():
                workers.extend(replica.workers)
            rollout_wg = RayWorkerGroup(
                worker_handles=workers,
                ray_cls_with_init=RayClassWithInitArgs(cls=ray.remote(CheckpointEngineWorker)),
            )
            rollout_visible = rollout_wg.execute_all_sync("get_cuda_visible_devices")
            print(f"[split-sync] rollout CUDA_VISIBLE_DEVICES={rollout_visible}")
        except Exception as exc:
            print(f"[split-sync] failed to query rollout CUDA_VISIBLE_DEVICES: {exc}")


@ray.remote(num_cpus=1)
class SplitSyncTaskRunner:
    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_training_workers(self, config):
        self.role_worker_mapping[Role.Actor] = ray.remote(ActorRolloutRefWorker)
        self.mapping[Role.Actor] = "actor_pool"

        if need_reference_policy(config):
            self.role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            ref_pool_gpus = OmegaConf.select(config, "split_sync.ref_n_gpus_per_node")
            self.mapping[Role.RefPolicy] = "ref_pool" if ref_pool_gpus is not None else "actor_pool"

        if need_critic(config):
            self.role_worker_mapping[Role.Critic] = ray.remote(TrainingWorker)
            self.mapping[Role.Critic] = "actor_pool"

    def init_resource_pool_mgr(self, config):
        resource_pool_spec = {
            "actor_pool": [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }

        ref_pool_gpus = OmegaConf.select(config, "split_sync.ref_n_gpus_per_node")
        if need_reference_policy(config) and ref_pool_gpus is not None:
            ref_pool_nodes = OmegaConf.select(config, "split_sync.ref_nnodes", default=config.trainer.nnodes)
            if ref_pool_gpus <= 0:
                raise ValueError("split_sync.ref_n_gpus_per_node must be greater than 0")
            if ref_pool_nodes <= 0:
                raise ValueError("split_sync.ref_nnodes must be greater than 0")
            resource_pool_spec["ref_pool"] = [ref_pool_gpus] * ref_pool_nodes

        if config.reward.reward_model.enable_resource_pool:
            if config.reward.reward_model.n_gpus_per_node <= 0:
                raise ValueError("config.reward.reward_model.n_gpus_per_node must be greater than 0")
            if config.reward.reward_model.nnodes <= 0:
                raise ValueError("config.reward.reward_model.nnodes must be greater than 0")
            resource_pool_spec["reward_pool"] = [
                config.reward.reward_model.n_gpus_per_node
            ] * config.reward.reward_model.nnodes
            self.mapping[Role.RewardModel] = "reward_pool"
        else:
            config.reward.reward_model.nnodes = config.trainer.nnodes
            config.reward.reward_model.n_gpus_per_node = config.trainer.n_gpus_per_node
            self.mapping[Role.RewardModel] = "actor_pool"

        distillation_config = config.get("distillation")
        if is_distillation_enabled(distillation_config):
            if distillation_config.n_gpus_per_node <= 0:
                raise ValueError("config.distillation.n_gpus_per_node must be greater than 0")
            if distillation_config.nnodes <= 0:
                raise ValueError("config.distillation.nnodes must be greater than 0")
            resource_pool_spec["teacher_pool"] = [
                distillation_config.n_gpus_per_node
            ] * distillation_config.nnodes
            self.mapping[Role.TeacherModel] = "teacher_pool"

        self.resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)

    def run(self, config):
        print(f"SplitSyncTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        tq.init(config.transfer_queue)
        trainer = None
        try:
            self.add_training_workers(config)
            self.init_resource_pool_mgr(config)
            trainer = SplitSyncPPOTrainer(
                config=config,
                role_worker_mapping=self.role_worker_mapping,
                resource_pool_manager=self.resource_pool_manager,
            )
            trainer.init_workers()
            trainer.fit()
        finally:
            if trainer:
                trainer.replay_buffer.close()
            tq.close()


@hydra.main(config_path="../verl/verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    config.transfer_queue.enable = True
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )
    run_ppo(config, task_runner_class=SplitSyncTaskRunner)


if __name__ == "__main__":
    main()
