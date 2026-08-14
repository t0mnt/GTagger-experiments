"""Does torch.compile cost or save VRAM? Per model, measured, at a FIXED batch size.

The question this answers is one the repo had an unstated assumption about. `utils/bperf.py`
sizes each row EAGER and runs both states at that size, on the reasoning that "eager is the
memory-hungrier state, so an eager-sized batch is the CONSERVATIVE shared choice". On
2026-08-13 CGENNLGATrGraphGPS falsified that: sized eager at 256 (88.6 GiB peak on a
worst-case batch, survived), the COMPILED state OOM'd on an ordinary batch with 91.2 GiB
allocated. Compiled wanted MORE.

The other four rows of that sweep are NOT evidence to the contrary. The doubling search
overshoots and falls back a rung, so they landed at 57-69% of the card -- 29 to 40 GiB of
slack, against a ~2.7 GiB effect. GPS was the only row positioned to reveal anything. Hence
this: a FIXED batch size for every row, so the per-model eager-vs-compiled ratio is the only
thing that varies.

WHY IT CAN GO EITHER WAY. Eager autograd's live set is a sawtooth over time: each op saves
what its own backward needs and every other intermediate is freed as soon as its last
forward consumer runs, so the peak is the maximum INSTANTANEOUS live set. AOTAutograd
instead picks one cut through the joint graph, and everything on that cut is held
SIMULTANEOUSLY from end-of-forward until the backward consumes it. Compiled wins when the
cut is narrower than the sawtooth's peak (the common case) and loses when the model has many
wide intermediates that eager would have staggered. The default
`activation_memory_budget=None` is torch's 1.0 -- minimize recompute, i.e. SAVE EVERYTHING --
which is the maximal-retention end of the dial.

That the partitioner REPLACES rather than preserves eager's profile is already on record
here: docs/cgenn-compile.md measures `gp_impl: sparse` retaining 84.84 MB eager and 252.76 MB
compiled -- 3.0x MORE -- while einsum goes 293.44 -> 253.63 MB, i.e. less. The partitioner
flattened all three impls to ~253 MB regardless of where eager sat.

GPU ONLY, and that is the point: VRAM is CUDA. The CPU proxy is unusable twice over -- the
peak column of a CPU profile is a proxy the compile log already flags as unreliable, and the
compiled CPU path dies with a native `double free or corruption` inside lgatr's sparse-table
construction (a CPU-inductor artifact; CUDA compiles fine).

Usage:
    python utils/vram_compile_matrix.py                      # every row, bs=64, full dataset
    python utils/vram_compile_matrix.py --models CGENN       # substring filter, as bperf
    python utils/vram_compile_matrix.py --bs 128             # ratios are batch-dependent
    python utils/vram_compile_matrix.py --dataset mini       # ~30x faster to load, P_max 135
                                                             # vs the full set's 160, so read
                                                             # the RATIO and not the absolute
"""

import argparse
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:  # `python utils/...` puts utils/ on sys.path, not the root
    sys.path.insert(0, str(REPO))

WARMUP = 3  # the compile log is emphatic: first-call numbers are not steady state
SAME_BAND = 0.05  # |ratio - 1| within this reads as "about the same"


def measure(row_overrides, knob, state, bs, dataset):
    """Peak VRAM and retention for ONE (row, state), in this process. Prints a RESULT line."""
    import hydra
    import torch

    from utils.find_lr import _probe_lengths, _worst_case_batch, build_experiment

    if not torch.cuda.is_available():
        raise SystemExit(
            "no CUDA device: this measures VRAM, which does not exist on CPU. The CPU proxy "
            "is unusable twice over -- the compile log flags a CPU profile's peak column as "
            "unreliable, and the compiled CPU path dies with a native `double free or "
            "corruption` in lgatr's sparse-table construction. Run it on the GPU "
            "(docs/oscar-vram.sbatch).")

    with hydra.initialize_config_dir(config_dir=str(REPO / "config"), version_base=None):
        cfg = hydra.compose(
            config_name="toptagging",
            overrides=[*row_overrides, f"{knob}={'true' if state == 'compiled' else 'false'}",
                       "save=false", f"training.batchsize={bs}", f"data.dataset={dataset}"],
        )
    exp = build_experiment(cfg)
    exp.model.train()
    lengths = _probe_lengths(getattr(exp, "data_train", None))

    def make_batch():
        """A FRESH batch per step. `embed_tagging_data` mutates the caller's `ptr` IN PLACE,
        so a batch is CONSUMED by a step; reusing one across two steps double-counts the
        spurion offsets and dies with an IndexError. Deterministic, so every step still gets
        an identical batch -- see utils/find_lr.py::find_max_batch_size."""
        batch = None if lengths is None else _worst_case_batch(exp, bs, lengths)
        return next(iter(exp.train_loader)) if batch is None else batch

    def step():
        loss, _ = exp._batch_loss(make_batch())
        exp.optimizer.zero_grad(set_to_none=True)
        exp.scaler.scale(loss).backward()
        exp.scaler.step(exp.optimizer)
        exp.scaler.update()

    for _ in range(WARMUP):  # compilation, autotune, allocator growth, optimizer state
        step()

    # Retention and peak on SEPARATE steps, deliberately. The retention step runs under
    # saved_tensors_hooks, and whether ENTERING that context invalidates dynamo guards --
    # recompiling inside the measured step and inflating a compiled peak with build-time
    # allocations, in the exact direction this tool expects to find -- is a torch-version
    # property: measured NO on 2.13 (unique_graphs flat across the context, hooks fire on
    # the compiled path), unmeasured on the pinned 2.8 CUDA build. One extra step removes
    # the question instead of depending on the answer, and the peak step is hook-free and
    # one step more settled besides.
    held = {}
    with torch.autograd.graph.saved_tensors_hooks(
        lambda t: (held.__setitem__(id(t), t.numel() * t.element_size()), t)[1], lambda t: t
    ):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    step()
    torch.cuda.synchronize()
    print(f"RESULT\t{state}\t{torch.cuda.max_memory_allocated()}\t{sum(held.values())}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None, help="substring filter, as bperf")
    ap.add_argument("--bs", type=int, default=64,
                    help="FIXED batchsize for every row and both states (default 64: fits "
                         "every row eager on a 93 GiB H100). Ratios are batch-dependent -- a "
                         "fixed per-step overhead amortizes -- so re-run near your real "
                         "regime for a row you care about")
    ap.add_argument("--dataset", default="full", help="full | mini (default full)")
    ap.add_argument("--timeout", type=int, default=1800, help="per-case seconds")
    # worker mode: one subprocess per (row, state). Separate processes are not tidiness --
    # bperf's own note applies, that a shared process leaves dynamo cache entries holding
    # each model alive for the next measurement, which is exactly what this reads.
    ap.add_argument("--row", help=argparse.SUPPRESS)
    ap.add_argument("--state", choices=("eager", "compiled"), help=argparse.SUPPRESS)
    args = ap.parse_args()

    from utils.bperf import MATRIX

    if args.row:
        name, base, knob, _ = next(r for r in MATRIX if r[0] == args.row)
        measure(base, knob, args.state, args.bs, args.dataset)
        return

    rows = [r for r in MATRIX if not args.models or any(m in r[0] for m in args.models)]
    if not rows:
        raise SystemExit(f"--models {args.models} matched no row. "
                         f"Available: {[r[0] for r in MATRIX]}")

    results = {}
    for name, *_ in rows:
        for state in ("eager", "compiled"):
            cmd = [sys.executable, __file__, "--row", name, "--state", state,
                   "--bs", str(args.bs), "--dataset", args.dataset]
            print(f"[vram] {name} {state} ...", flush=True)
            try:
                p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                                   timeout=args.timeout)
                line = next((l for l in (p.stdout + p.stderr).splitlines()
                             if l.startswith("RESULT")), None)
                if line:
                    _, _, peak, retained = line.split("\t")
                    results[(name, state)] = (int(peak), int(retained))
                    print(f"[vram]   peak {int(peak) / 2**30:6.2f} GiB, "
                          f"retained {int(retained) / 2**30:6.2f} GiB", flush=True)
                else:
                    tail = (p.stdout + p.stderr).strip().splitlines()[-3:]
                    print(f"[vram]   FAILED: {' | '.join(tail)}", flush=True)
            except subprocess.TimeoutExpired:
                print(f"[vram]   TIMEOUT after {args.timeout}s", flush=True)

    GIB = 2**30
    print(f"\n## compiled-vs-eager VRAM, bs={args.bs}, dataset={args.dataset}, "
          f"{WARMUP} warm-up steps\n")
    print("| row | peak eager | peak compiled | peak ratio | retained eager | "
          "retained compiled | verdict |")
    print("|---|---|---|---|---|---|---|")
    for name, *_ in rows:
        e, c = results.get((name, "eager")), results.get((name, "compiled"))
        if not (e and c):
            print(f"| {name} | {'-' if not e else f'{e[0]/GIB:.2f} GiB'} | "
                  f"{'-' if not c else f'{c[0]/GIB:.2f} GiB'} | - | - | - | INCOMPLETE |")
            continue
        ratio = c[0] / e[0]
        verdict = ("**MORE compiled**" if ratio > 1 + SAME_BAND else
                   "less compiled" if ratio < 1 - SAME_BAND else "about the same")
        print(f"| {name} | {e[0]/GIB:.2f} GiB | {c[0]/GIB:.2f} GiB | {ratio:.3f}x | "
              f"{e[1]/GIB:.2f} GiB | {c[1]/GIB:.2f} GiB | {verdict} |")
    print(f"\nPeak sets the batch size; retention is the partitioner's cut and is shown "
          f"because it is the mechanism. A row reading MORE is a row whose bperf batchsize "
          f"must not be trusted, and a candidate for `activation_memory_budget` (see "
          f"docs/cgenn-compile.md).")


if __name__ == "__main__":
    main()
