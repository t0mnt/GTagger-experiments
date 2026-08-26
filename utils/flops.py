#!/usr/bin/env python3
"""Forward FLOPs and parameter count for one model + arbitrary config overrides.

    python utils/flops.py tag_PlainGraphGPS
    python utils/flops.py tag_PlainGraphGPS model.net.use_rwse=true
    python utils/flops.py tag_cgenn --task jctagging

Same measurement as the results-table cell and tests/experiments/test_tag_flops.py:
ONE jet of TopTaggingExperiment.FLOPS_JET_SIZE constituents, batch size 1, eager, with
flash geometric products routed to a traceable arm. Prices an ablation column without
training anything -- FLOPs is a property of the forward pass, not of the weights -- and
without re-evaluating a finished run, which on JetClass costs hours.

Needs the task's data present, since the batch comes from the real loader.
"""
import argparse, os, sys, warnings

warnings.filterwarnings("ignore")
# run as `python utils/flops.py`, so sys.path[0] is utils/ and `experiments` is not
# importable -- put the repo root first, as run.py gets for free by living there.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="e.g. tag_PlainGraphGPS")
    ap.add_argument("overrides", nargs="*", help="hydra overrides, e.g. model.net.use_rwse=true")
    ap.add_argument("--task", default="toptagging",
                    choices=["toptagging", "jctagging", "toptagxl"])
    ap.add_argument("--dataset", default=None,
                    help="data.dataset for toptagging (default mini: FLOPs does not depend on it)")
    ap.add_argument("--config-dir", default=None)
    args = ap.parse_args()

    import hydra
    import experiments.logger
    from experiments.tagging.experiment import TopTaggingExperiment
    from experiments.tagging.jetclassexperiment import JetClassTaggingExperiment

    experiments.logger.LOGGER.disabled = True
    cdir = args.config_dir or os.path.join(ROOT, "config")

    ov = [f"model={args.model}", "save=false", "training.batchsize=8", *args.overrides]
    if args.task == "toptagging":
        ov.append(f"data.dataset={args.dataset or 'mini'}")
    elif args.dataset:
        ap.error("--dataset applies to toptagging only")

    with hydra.initialize_config_dir(config_dir=cdir, version_base=None):
        cfg = hydra.compose(config_name=args.task, overrides=ov)
        Exp = JetClassTaggingExperiment if args.task == "jctagging" else TopTaggingExperiment
        exp = Exp(cfg)
    exp._init(); exp.init_physics(); exp.init_model()
    exp.init_data(); exp._init_dataloader(); exp._init_loss()

    flops = exp._count_flops(exp.test_loader)
    params = sum(p.numel() for p in exp.model.parameters() if p.requires_grad)
    extra = (" " + " ".join(args.overrides)) if args.overrides else ""
    if flops is None:
        print(f"{args.model}{extra}  [{args.task}]  FLOPs=FAILED  params={params}")
        sys.exit(1)
    n = getattr(exp, "_flops_jet_n", exp.FLOPS_JET_SIZE)
    tag = "" if n == exp.FLOPS_JET_SIZE else "  [WARNING: no jet reached FLOPS_JET_SIZE]"
    print(f"{args.model}{extra}  [{args.task}]  "
          f"FLOPs={flops:.4e}  params={params}  (n={n}){tag}")


if __name__ == "__main__":
    main()
