import datetime
import os

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from experiments.amplitudes.experiment import AmplitudeExperiment
from experiments.amplitudes.experimentxl import AmplitudeXLExperiment
from experiments.eventgen.processes import ttbarExperiment
from experiments.tagging.experiment import TopTaggingExperiment
from experiments.tagging.finetuneexperiment import TopTaggingFineTuneExperiment
from experiments.tagging.jetclassexperiment import JetClassTaggingExperiment
from experiments.tagging.toptagxlexperiment import TopTagXLExperiment


def _check_recipe_is_swept():
    """Refuse to train on a recipe whose `???` keys were never filled in.

    `???` does NOT behave as hydra's mandatory-value marker here. The per-model recipes
    inherit `tag_gts_and_friends_default` (and through it `tag_default`), which DEFINES
    `batchsize: 512` and `lr: 1e-3`; OmegaConf merge treats a child `???` as "no value
    supplied", so the parent's value survives and the run trains at 512 / 1e-3 with no
    error. 25 recipes currently carry the marker, and only the 8 `jc_*` ones say so in a
    comment -- the `top_*` and `xl_*` ones call the keys "REQUIRED", which they are not.
    A wrong-but-plausible row in the results table is worse than a failed launch, and the
    comment that already existed did not propagate, so this is a check rather than a note.

    Skipped silently when hydra is not managing the run (find_lr, bperf and the tests
    compose configs directly, and none of them trains). A key overridden on the command
    line counts as filled, since that is a legitimate way to supply it.
    """
    from hydra.core.hydra_config import HydraConfig

    try:
        hc = HydraConfig.get()
        choice = hc.runtime.choices.get("training")
        root = next(s.path for s in hc.runtime.config_sources if s.provider == "main")
        overridden = {o.split("=", 1)[0] for o in hc.overrides.task}
        path = os.path.join(root, "training", f"{choice}.yaml")
        with open(path) as fh:
            lines = fh.readlines()
    except Exception:
        return  # not under `hydra.main`, or the recipe is not a file we can find

    unfilled = [
        key
        for line in lines
        if (key := line.split(":", 1)[0].strip())
        and not line.startswith(("#", " ", "\t"))
        and line.split(":", 1)[-1].strip() == "???"
        and f"training.{key}" not in overridden
    ]
    if unfilled:
        raise SystemExit(
            f"\n{path}\n  still has unswept keys: {', '.join(unfilled)}.\n"
            f"  `???` is NOT enforced by hydra here -- the shared default would supply "
            f"batchsize=512, lr=1e-3\n  and this run would look legitimate. Fill them from "
            f"utils/find_lr.py, or pass e.g.\n  `training.{unfilled[0]}=<value>` on the "
            f"command line if that is what you meant."
        )


@hydra.main(config_path="config_quick", config_name="toptagging", version_base=None)
def main(cfg):
    _check_recipe_is_swept()
    if torch.cuda.is_available() and cfg.gpus == -1:
        world_size = torch.cuda.device_count()
    elif torch.cuda.is_available() and cfg.gpus >= 1:
        world_size = cfg.gpus
    else:
        world_size = 1

    if world_size > 1:
        print(
            "Warning: Running with multi-GPU is not fully supported in this repo; we do not recommend using this feature."
        )
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        os.environ.setdefault("NCCL_DEBUG", "WARN")
        os.environ.setdefault("NCCL_IB_DISABLE", "1")
        os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
        os.environ.setdefault("OMP_NUM_THREADS", "1")

        _set_common_env(world_size)
        mp.spawn(ddp_worker, nprocs=world_size, args=(cfg,))
    else:
        # no GPU or only one GPU -> run on main process
        ddp_worker(rank=0, cfg=cfg)


def ddp_worker(rank, cfg):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)

    if world_size > 1:
        # set up communication between processes
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(minutes=30),
        )
        torch.cuda.set_device(rank)

    if cfg.exp_type == "toptagging":
        constructor = TopTaggingExperiment
    elif cfg.exp_type == "toptaggingft":
        constructor = TopTaggingFineTuneExperiment
    elif cfg.exp_type == "toptagxl":
        constructor = TopTagXLExperiment
    elif cfg.exp_type == "jctagging":
        constructor = JetClassTaggingExperiment
    elif cfg.exp_type == "amplitudes":
        constructor = AmplitudeExperiment
    elif cfg.exp_type == "amplitudesxl":
        constructor = AmplitudeXLExperiment
    elif cfg.exp_type == "ttbar":
        constructor = ttbarExperiment
    else:
        raise ValueError(f"exp_type {cfg.exp_type} not implemented")

    exp = constructor(cfg, rank, world_size)
    exp()

    if world_size > 1:
        dist.barrier(device_ids=[rank])
        dist.destroy_process_group()


def _set_common_env(world_size):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(_find_free_port()))
    os.environ["WORLD_SIZE"] = str(world_size)


def _find_free_port():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


if __name__ == "__main__":
    main()
