"""Batch-size-only finder: fill the `batchsize:` key of many recipes in one pass.

    python utils/find_bs.py                          # all 8 GT hybrids, top tagging
    python utils/find_bs.py --models CGENNLGATrGraphGPS LorentzNetLGATrSlimGraphGPS
    python utils/find_bs.py --overrides data.dataset=mini   # CPU/smoke passthrough

Exists for the 2026-08-16 table-wide lr decision (lr: 1e-3 in every top_<hybrid>.yaml,
provenance in the yamls and docs/cgenn-compile.md): with the lr fixed, the only number
the finder still owes each recipe is the GPU-fit batch size, and running the full
300-step lr ramp per model to get it wastes an H100 hour. This runs ONLY the doubling
search (find_lr's own `find_max_batch_size`, worst-case constructed probe batch, two
full training steps per rung) and prints one paste-ready line per model.

Faithful-posture rules inherited from find_lr, deliberately:
- each model is composed WITH ITS OWN training recipe when one exists (optimizer state
  is part of the memory) and with `compile` as the model yaml ships it (compiled
  retention differs from eager by up to ~2x on the GPS rows -- sizing eager would hand
  training a batch chosen under a memory profile it never runs at);
- the default cap is BS_CEILING (=512), the campaign's update-starvation ceiling --
  the search answers "what fits UP TO the size we would actually train at", pass
  --bs-max to ask the pure-fit question instead.

CUDA only in effect: on CPU the search is a no-op and reports the configured size.
"""

import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Dynamo warns once PER TRACED CALL SITE PER COMPILE about lru_cache-wrapped functions
# (lgatr/primitives/{bilinear,linear}.py -- cached `.to(...)` of constant basis tensors,
# the benign pure case; compiled-vs-eager TOL and DET gates pin the soundness). Over a
# multi-model sizing pass that is thousands of identical lines drowning the numbers this
# tool exists to print. "once" keeps a single instance visible rather than hiding it.
warnings.filterwarnings("once", message=".*functools.lru_cache.*")

import hydra
import torch

from experiments.logger import LOGGER
from utils.find_lr import BS_CEILING, RECIPE_PREFIX, build_experiment, find_max_batch_size

HYBRIDS = [
    "PlainGraphTrans", "PlainGraphGPS",
    "ParticleNetParTGraphTrans", "ParticleNetParTGraphGPS",
    "CGENNLGATrGraphTrans", "CGENNLGATrGraphGPS",
    "LorentzNetLGATrSlimGraphTrans", "LorentzNetLGATrSlimGraphGPS",
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="*", default=HYBRIDS,
                    help="model stems (tag_ prefix optional; default: the 8 GT hybrids)")
    ap.add_argument("--task", default="toptagging", help="hydra config name / task")
    ap.add_argument("--config-path", default="config")
    ap.add_argument("--bs-start", type=int, default=16)
    ap.add_argument("--bs-max", type=int, default=BS_CEILING,
                    help=f"search cap (default {BS_CEILING}, the campaign ceiling)")
    ap.add_argument("--bs-safety", type=float, default=1.0)
    ap.add_argument("--overrides", nargs="*", default=[],
                    help="extra hydra overrides applied to every model (e.g. data.dataset=mini)")
    args = ap.parse_args(argv)

    prefix = RECIPE_PREFIX.get(args.task)
    config_dir = os.path.abspath(args.config_path)
    results = {}
    for raw in args.models:
        stem = raw[4:] if raw.startswith("tag_") else raw
        overrides = [f"model=tag_{stem}", "save=false", *args.overrides]
        recipe = f"{prefix}_{stem}" if prefix else None
        recipe_path = recipe and os.path.join(args.config_path, "training", f"{recipe}.yaml")
        if recipe_path and os.path.isfile(recipe_path):
            overrides.append(f"training={recipe}")
        else:
            recipe_path = None
        LOGGER.info(f"=== {stem}: sizing under {overrides}")
        exp = None
        try:
            with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
                cfg = hydra.compose(config_name=args.task, overrides=overrides)
            cfg.train = cfg.evaluate = cfg.plot = cfg.save = False
            exp = build_experiment(cfg)
            bs = find_max_batch_size(exp, args.bs_start, args.bs_max, args.bs_safety)
            results[stem] = (bs, recipe_path)
        except Exception as err:
            LOGGER.error(f"{stem}: batch-size search FAILED: {err}")
            results[stem] = (None, recipe_path)
        finally:
            del exp
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            torch._dynamo.reset()  # drop this model's compiled graphs before the next row

    LOGGER.info("=" * 64)
    for stem, (bs, recipe_path) in results.items():
        if bs is None:
            LOGGER.info(f"FIND_BS  model={stem}  batchsize=FAILED (see log above)")
        else:
            where = f"  ->  {recipe_path}  (paste: batchsize: {bs})" if recipe_path else ""
            LOGGER.info(f"FIND_BS  model={stem}  batchsize={bs}{where}")
    LOGGER.info("=" * 64)
    return results


if __name__ == "__main__":
    main()
