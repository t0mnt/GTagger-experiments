#!/usr/bin/env python3
"""Which branch a GPS layer actually leans on -- measured on activations, not weights.

    python utils/branch_balance.py tag_PlainGraphGPS
    python utils/branch_balance.py tag_PlainGraphGPS --ckpt runs/.../models/model_run0.pt
    python utils/branch_balance.py tag_PlainGraphGPS --ckpt <path> --task jctagging

A GraphGPS layer fuses its two branches by SUM -- ``h = h_local + h_attn`` -- and each
branch arrives through its own norm, so the branches' relative magnitude at that sum is
what decides which one the layer is using. Reading it off the norm's gamma is a good
proxy and cheap (no data needed), but it assumes both branches' normalised outputs carry
comparable structure. This measures the thing itself, on a live forward, over the REAL
nodes only.

Measured as the SPREAD ACROSS NODES, not the RMS -- an earlier version of this script
used RMS and it was wrong in a way worth recording. A norm emits ``gamma * x_hat + beta``,
so its RMS is ``sqrt(gamma^2 + beta^2)``: a branch whose output has collapsed to a learned
constant still reads large. Only the per-node variation reaches the sum as information,
and beta is subtracted straight back out by the next layer's norm (every layer but the
last). On a JetClass PlainGraphGPS checkpoint the RMS ratio read 1.1-1.3 across the stack
while the gamma ratio read 3-9; solving sqrt(gamma^2 + beta^2) showed both branches were
bias-dominated (beta ~1.4 against gamma ~0.04-0.35) and the gamma reading had been right.
The ``const/sig`` columns exist so that regime is visible rather than inferred.

Companion to utils/graph_diameter.py. The diameter says where the local branch runs out
of new information to deliver; this says whether the model noticed. A local branch still
carrying full weight at a depth past the diameter is doing something other than
delivering information -- that gap is the interesting measurement, and it needs both
numbers to state.

Hooks MaskedNorm directly, which is what makes the masking exact: MaskedNorm takes the
node mask as its SECOND forward argument, so the hook reads the same mask the layer used
rather than reconstructing one that might disagree at the padding.

Without --ckpt this reports an untrained network (all gamma = 1), which is the baseline
the trained profile should be read against, and a way to check the script itself. Read
only the RATIO there: the forward runs in eval mode, so a MaskedNorm with norm='batch'
uses running stats, and an untrained net's (0, 1) stats make it a near-identity -- the
absolute spread then just tracks the residual stream doubling at each sum fusion (0.50 ->
386 over ten layers), which says nothing about a trained model.
"""
import argparse, os, sys, warnings

warnings.filterwarnings("ignore")
# run as `python utils/branch_balance.py`, so sys.path[0] is utils/ and `experiments` is
# not importable -- put the repo root first, as run.py gets for free by living there.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _minimal_files(task, user_overrides):
    """One file per split for the streaming tasks, unless the caller asked otherwise.

    The experiment builds train, val AND test loaders before a single batch can be read,
    so the shipped ranges open 121 files (jctagging) or 675 (toptagxl) to sample one
    batch -- minutes of wall clock, and enough of a login node to draw a resource
    warning. The weights come from --ckpt and the batch is a 256-jet sample either way,
    so one file per split measures the same thing at ~1/100 the setup. Any explicit
    data.*_files_range override wins, for when a wider sample IS the point.
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
    ap.add_argument("overrides", nargs="*", help="hydra overrides, e.g. model.net.num_layers=6")
    ap.add_argument("--ckpt", default=None, help="models/model_run0.pt from a finished run")
    ap.add_argument("--task", default="toptagging",
                    choices=["toptagging", "jctagging", "toptagxl"])
    ap.add_argument("--jets", type=int, default=256, help="jets to average over (one batch)")
    ap.add_argument("--dataset", default="mini", help="data.dataset for toptagging")
    ap.add_argument("--config-dir", default=None)
    args = ap.parse_args()

    import hydra, torch
    import experiments.logger
    from experiments.tagging.experiment import TopTaggingExperiment
    from experiments.tagging.jetclassexperiment import JetClassTaggingExperiment

    experiments.logger.LOGGER.disabled = True
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

    if args.ckpt:
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["model"]
        missing, unexpected = exp.model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            # loud, not fatal: a shape/name drift silently loading 'most' of a checkpoint
            # would report the profile of a half-trained network as if it were trained.
            print(f"[warning] state_dict mismatch: {len(missing)} missing, "
                  f"{len(unexpected)} unexpected (first: {(missing + unexpected)[:2]})")

    # every GPS layer that owns both branch norms, in stack order
    layers = [m for m in exp.model.modules()
              if hasattr(m, "norm_local") and hasattr(m, "norm_attn")]
    if not layers:
        sys.exit(f"{args.model} has no GPS layer with norm_local/norm_attn branches")

    stats = {}

    def hook(tag, i):
        def fn(_mod, inp, out):
            # MaskedNorm.forward(h, mask_bool) -- inp[1] is the layer's own node mask, so
            # the average is over real nodes and never over padding.
            mask = inp[1] if len(inp) > 1 and inp[1] is not None else None
            vals = out[mask] if mask is not None and mask.dtype == torch.bool else out.reshape(-1, out.shape[-1])
            v = vals.float()
            # SPREAD ACROSS NODES, not RMS. A norm emits gamma*x_hat + beta, so its RMS is
            # sqrt(gamma^2 + beta^2) -- a branch collapsed to a constant still reads large.
            # Only the per-node variation reaches the sum as information, and beta is
            # subtracted out by the next layer's norm anyway (except in the last layer).
            stats.setdefault((i, tag), []).append(
                (
                    v.std(dim=0, unbiased=False).pow(2).mean().sqrt().item(),  # signal
                    v.mean(dim=0).pow(2).mean().sqrt().item(),                 # constant
                )
            )
        return fn

    handles = []
    for i, layer in enumerate(layers):
        handles.append(layer.norm_local.register_forward_hook(hook("local", i)))
        handles.append(layer.norm_attn.register_forward_hook(hook("attn", i)))

    exp.model.eval()
    batch = next(iter(exp.test_loader))
    batch = batch.to(exp.device) if hasattr(batch, "to") else batch
    try:
        with torch.no_grad():
            exp._get_ypred_and_label(batch)
    finally:
        for h in handles:
            h.remove()

    def gamma(mod):
        w = getattr(getattr(mod, "norm", mod), "weight", None)
        return w.abs().mean().item() if w is not None else float("nan")

    src = args.ckpt or "UNTRAINED (gamma = 1 baseline)"
    print(f"{args.model} [{args.task}]  {len(layers)} layers  {args.jets} jets\n  {src}")
    def avg(i, tag, j):
        vals = [s[j] for s in stats.get((i, tag), [])]
        return sum(vals) / len(vals) if vals else float("nan")

    print(f"{'layer':>5} {'sig_local':>10} {'sig_attn':>9} {'sig_ratio':>10} "
          f"{'gamma_ratio':>12} {'const/sig_L':>12} {'const/sig_A':>12}")
    for i, layer in enumerate(layers):
        sl, sa = avg(i, "local", 0), avg(i, "attn", 0)
        cl, ca = avg(i, "local", 1), avg(i, "attn", 1)
        gl, ga = gamma(layer.norm_local), gamma(layer.norm_attn)
        print(f"{i:5d} {sl:10.4f} {sa:9.4f} {sl / sa if sa else float('nan'):10.3f} "
              f"{gl / ga if ga else float('nan'):12.3f} "
              f"{cl / sl if sl else float('nan'):12.2f} {ca / sa if sa else float('nan'):12.2f}")
    print("  sig_* is the per-node SPREAD of each branch's output -- the part that reaches\n"
          "  the sum as information. sig_ratio is the measurement; gamma_ratio is the\n"
          "  weight-only proxy for it, and they should track closely.\n"
          "  const/sig is how far each branch has collapsed toward a constant. Above ~1 the\n"
          "  branch is mostly a learned offset, which the NEXT layer's norm subtracts out --\n"
          "  so a high value means that layer is contributing little to what follows.")


if __name__ == "__main__":
    main()
