import os
from collections import OrderedDict

import ray


@ray.remote(num_gpus=1)
class GPUProbe:
    def __init__(self, role: str):
        self.role = role

    def info(self):
        ctx = ray.get_runtime_context()
        return {
            "role": self.role,
            "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "accelerator_ids": ctx.get_accelerator_ids(),
        }


def main():
    ray.init(ignore_reinit_error=True)
    layout = OrderedDict(
        [
            ("actor", 1),
            ("ref", int(os.environ.get("PROBE_REF_GPUS", "3"))),
            ("rollout", int(os.environ.get("PROBE_ROLLOUT_GPUS", "4"))),
        ]
    )
    actors = []
    for role, count in layout.items():
        for idx in range(count):
            actors.append(GPUProbe.remote(f"{role}:{idx}"))
    for item in ray.get([actor.info.remote() for actor in actors]):
        print(item)
    ray.shutdown()


if __name__ == "__main__":
    main()
