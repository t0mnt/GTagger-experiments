#!/usr/bin/env python3
"""kNN-graph diameter for a tagger's local branch -- i.e. where GNN stacking stops
being effective, an issue transformers do not have.

    python utils/graph_diameter.py tag_PlainGraphGPS
    python utils/graph_diameter.py tag_PlainGraphGPS model.net.knn_k=8
    python utils/graph_diameter.py tag_PlainGraphTrans --jets 1024

A message-passing round moves information exactly one hop. Once the round count
reaches the graph's diameter every node has seen every other node in its jet, and every
further round is re-mixing what is already there -- it cannot deliver new information,
whatever it does to the representation. Attention has no such ceiling: it is one hop by
construction, so stacking more of it keeps composing higher-order functions of the whole
jet. That asymmetry is why "depth" is not one number for a GPS-style hybrid: the local
branch saturates at the diameter while the global branch does not.

Which makes the diameter a design constant worth measuring rather than assuming. A jet
kNN graph is not the sparse, large-diameter object the GraphGPS benchmarks are built on
(ZINC molecules are degree ~2; the LRGB suite is explicitly long-range): it is k=16 over
~50 constituents packed into a small dR patch, and it mixes in a handful of hops. This
script builds the graph the model itself builds -- by spying on the module's own knn(),
so the metric, the k, the self-exclusion and the padding mask are whatever that config
actually uses -- and reports the hop distribution.

Read it against the model's num_layers: rounds beyond the max diameter are provably
information-free for every jet in the sample. That does not by itself prove they are
harmful (the local branch is suppressible -- BatchNorm gamma -> 0 in norm_local zeroes it
exactly), but it does bound what they can possibly contribute, and it is the number to
quote when choosing how deep the local branch should go.
"""
import argparse, os, sys, warnings

warnings.filterwarnings("ignore")
# run as `python utils/graph_diameter.py`, so sys.path[0] is utils/ and `experiments` is
# not importable -- put the repo root first, as run.py gets for free by living there.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _minimal_files(task, user_overrides):
    """One file per split for the streaming tasks, unless the caller asked otherwise.

    The experiment builds train, val AND test loaders before a single batch can be read,
    so the shipped ranges open 121 files (jctagging) or 675 (toptagxl) to sample one
    batch -- minutes of wall clock, and enough of a login node to draw a resource
    warning. The diameter is a property of the graph builder plus the jets' multiplicity,
    so one file per split measures the same distribution at ~1/100 the setup. Any
    explicit data.*_files_range override wins, for when a wider sample IS the point.
    """
    ranges = {
        "jctagging": {"train": [0, 1], "test": [100, 101], "val": [120, 121]},
        "toptagxl": {"train": [0, 1], "test": [500, 501], "val": [625, 626]},
    }.get(task, {})
    return [
        f"data.{split}_files_range=[{lo},{hi}]"
        for split, (lo, hi) in ranges.items()
        if not any(o.startswith(f"data.{split}_files_range") for o in user_overrides)
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="tag_PlainGraphGPS")
    ap.add_argument("overrides", nargs="*", help="hydra overrides, e.g. model.net.knn_k=8")
    ap.add_argument("--task", default="toptagging",
                    choices=["toptagging", "jctagging", "toptagxl"])
    ap.add_argument("--jets", type=int, default=256, help="jets to sample (one batch)")
    ap.add_argument("--dataset", default="mini", help="data.dataset for toptagging")
    ap.add_argument("--config-dir", default=None)
    args = ap.parse_args()

    import hydra, torch, numpy as np
    import experiments.logger
    from experiments.tagging.experiment import TopTaggingExperiment
    from experiments.tagging.jetclassexperiment import JetClassTaggingExperiment

    experiments.logger.LOGGER.disabled = True

    # Spy on the module's own knn() rather than rebuilding the graph here: the metric
    # (deltaR vs minkowski), k, the self-exclusion and the padding mask then come from
    # the real config, and a graph this script built itself could silently differ.
    captured = {}
    targets = []
    for modname in ("plaingraphgps", "plaingraphtrans", "particlenetpartgraphgps",
                    "lorentznetlgatrslimgraphgps", "lorentznetlgatrslimgraphtrans",
                    "cgennlgatrgraphgps", "CGENNLGATrGraphTransHybrid"):
        try:
            mod = __import__(f"experiments.baselines.{modname}", fromlist=["knn"])
        except Exception:
            continue
        if hasattr(mod, "knn"):
            targets.append(mod)
    if not targets:
        sys.exit("no baseline module exposing knn() could be imported")

    def make_spy(real):
        def spy(x, k, metric="deltaR", mask=None, **kw):
            idx = real(x, k, metric=metric, mask=mask, **kw)
            # keep the FIRST graph of the forward: models that rebuild per layer
            # (ParticleNet-ParT GPS) would otherwise report only their last one.
            captured.setdefault("idx", idx)
            captured.setdefault("mask", mask)
            return idx
        return spy

    for mod in targets:
        mod.knn = make_spy(mod.knn)

    cdir = args.config_dir or os.path.join(ROOT, "config")
    ov = [f"model={args.model}", "save=false", "model.compile=false",
          f"evaluation.batchsize={args.jets}", *args.overrides]
    if args.task == "toptagging":
        ov.append(f"data.dataset={args.dataset}")
    else:
        shrunk = _minimal_files(args.task, args.overrides)
        ov += shrunk
        if shrunk:
            print(f"[note] sampling one file per split ({len(shrunk)} ranges narrowed) -- "
                  f"pass data.<split>_files_range to widen")

    with hydra.initialize_config_dir(config_dir=cdir, version_base=None):
        cfg = hydra.compose(config_name=args.task, overrides=ov)
        Exp = JetClassTaggingExperiment if args.task == "jctagging" else TopTaggingExperiment
        exp = Exp(cfg)
    exp._init(); exp.init_physics(); exp.init_model()
    exp.init_data(); exp._init_dataloader(); exp._init_loss()

    batch = next(iter(exp.test_loader))
    batch = batch.to(exp.device) if hasattr(batch, "to") else batch
    with torch.no_grad():
        exp._get_ypred_and_label(batch)
    if "idx" not in captured:
        sys.exit(f"{args.model} never called knn() -- it has no static kNN local branch")

    idx, mask = captured["idx"].cpu(), captured["mask"].cpu()
    B, P, K = idx.shape
    A = torch.zeros(B, P, P, dtype=torch.bool)
    A.scatter_(2, idx, True)
    A = A | A.transpose(1, 2)                       # hops are undirected for reachability
    m = mask.bool()
    A &= m.unsqueeze(1) & m.unsqueeze(2)            # padded rows/cols are not nodes
    A |= torch.eye(P, dtype=torch.bool).expand(B, -1, -1) & m.unsqueeze(2)

    n_real = m.sum(1)
    reach, diam = A.clone(), torch.full((B,), -1)
    for hop in range(1, P + 1):
        covered = ((reach.sum(-1) == n_real.view(-1, 1)) | ~m).all(1)
        diam[covered & (diam < 0)] = hop
        if (diam >= 0).all():
            break
        reach = (reach.float() @ A.float()) > 0

    keep = (n_real > 1) & (diam > 0)                # 1-constituent jets have no hops
    d, n = diam[keep].numpy(), n_real[keep].numpy()
    if not len(d):
        sys.exit("no multi-constituent jets in the sampled batch")

    # how many LOCAL rounds this config runs: a GPS stack applies its MPNN once per
    # layer, a GraphTrans stack only inside its gnn_blocks (its transformer half is not
    # message passing and is not diameter-bound).
    layers, net = None, exp.model.net
    for key in ("gnn_blocks", "layers", "blocks"):
        mod = getattr(net, key, None)
        if mod is not None and hasattr(mod, "__len__"):
            layers = len(mod)
            break
    if layers is None:
        layers = getattr(net, "num_layers", None) or getattr(net, "num_blocks", None)

    print(f"{args.model} [{args.task}]  jets={len(d)}  k={K}")
    print(f"  constituents  mean {n.mean():5.1f}   min {n.min():3d}   max {n.max():3d}")
    print(f"  DIAMETER      mean {d.mean():5.2f}   min {d.min():3d}   max {d.max():3d}   "
          f"median {np.median(d):.0f}   p95 {np.percentile(d, 95):.0f}")
    vals, cnts = np.unique(d, return_counts=True)
    print("  hops          " + "  ".join(f"{int(v)}:{int(c)}" for v, c in zip(vals, cnts)))
    if layers:
        spare = max(0, layers - int(d.max()))
        tail = (f"{spare} beyond the max diameter -- information-free for EVERY jet sampled"
                if spare else "none beyond the max diameter -- every round can still deliver")
        print(f"  local rounds  {layers}  -> {tail}")


if __name__ == "__main__":
    main()
