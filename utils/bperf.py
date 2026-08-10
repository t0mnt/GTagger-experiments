"""beta-PERF: the cluster eager-vs-compiled (and gp_impl) it/s matrix.

NOT a training reimplementation -- it is a DRIVER. Each row shells out to the repo's
real entry point (``run.py``) with ordinary hydra overrides, so every measured number
comes from the exact code path the campaign uses. It only chooses the overrides, reads
the run's own timing lines, and tabulates.

Method (docs/cgenn-compile.md, "ignore the first timing estimate"): each run does
``--iters`` steps; it/s is measured between the "Finished iteration LO" and "Finished
iteration HI" log lines, so compile warm-up -- which lands in the first few iterations
-- is excluded by construction. Validation is pushed past the window so it cannot
pollute the timing. GPU strongly recommended; on CPU the numbers are not the campaign
decision input and the header says so.

WHY THE RUNS ARE SEEDED. ``seed`` defaults to ``null`` in ``config/default.yaml`` and
``base_experiment`` seeds only ``if cfg.seed is not None`` -- so WITHOUT ``seed=`` the two
states of a row shuffle differently and walk different batches. That matters more than it
sounds: padded length varies 87-110 per batch on top-tagging and attention cost is O(P^2),
so batch order is the dominant per-step noise term. A fixed seed makes the comparison
PAIRED -- both states see the identical batch sequence -- cancelling that term instead of
averaging over it. Correct here because this is a throughput benchmark with ``save=false``,
not a campaign row (campaign trials deliberately vary their seeds).

WHAT PAIRING DOES NOT CANCEL, worth knowing before trusting a 3%% margin:
  * thermal / clock state -- the two states run back-to-back, so the second starts on a
    warmer card. A BIAS, not noise; pairing does not touch it.
  * order effect -- states always run false-then-true, so drift favours the same state on
    every row.
Both shrink if the window starts past the thermal ramp. For decision-grade numbers use
``--iters 310 --window 100 300``, and if a row lands inside the margin, re-run it with the
states swapped before believing the sign.

One-shot instrument: cleanup.md schedules its deletion once the numbers are recorded in
the compile log and the knob flips are committed.

Usage:
    python utils/bperf.py                       # full matrix, table only
    python utils/bperf.py --models tag_cgenn    # substring filter
    python utils/bperf.py --iters 210 --window 10 200
    python utils/bperf.py --find-batchsize      # size each row first (GPU); otherwise the
                                                # yaml value is used, which is the unswept
                                                # 512 fallback until you sweep
    python utils/bperf.py --apply               # ALSO edit the production yamls
                                                # (only flips with >3%% margin)
"""

import argparse
import datetime
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent

# (row name, base overrides, the knob that turns compile on, production yaml to edit)
MATRIX = [
    ("tag_cgenn/einsum", ["model=tag_cgenn", "model.net.gp_impl=einsum"], "model.compile", "config/model/tag_cgenn.yaml"),
    ("tag_cgenn/matmul", ["model=tag_cgenn", "model.net.gp_impl=matmul"], "model.compile", "config/model/tag_cgenn.yaml"),
    ("tag_cgenn/sparse", ["model=tag_cgenn", "model.net.gp_impl=sparse"], "model.compile", "config/model/tag_cgenn.yaml"),
    ("tag_lorentznet", ["model=tag_lorentznet"], "model.compile", "config/model/tag_lorentznet.yaml"),
    ("tag_slim", ["model=tag_slim"], "model.net.compile", "config/model/tag_slim.yaml"),
    ("tag_lgatr", ["model=tag_lgatr"], "model.net.compile", "config/model/tag_lgatr.yaml"),
    ("CGENNLGATrGraphTrans", ["model=tag_CGENNLGATrGraphTrans"], "model.compile", "config/model/tag_CGENNLGATrGraphTrans.yaml"),
    ("CGENNLGATrGraphGPS", ["model=tag_CGENNLGATrGraphGPS"], "model.compile", "config/model/tag_CGENNLGATrGraphGPS.yaml"),
    ("LNetSlimGraphTrans", ["model=tag_LorentzNetLGATrSlimGraphTrans"], "model.compile", "config/model/tag_LorentzNetLGATrSlimGraphTrans.yaml"),
    ("LNetSlimGraphGPS", ["model=tag_LorentzNetLGATrSlimGraphGPS"], "model.compile", "config/model/tag_LorentzNetLGATrSlimGraphGPS.yaml"),
    ("tag_ParT", ["model=tag_ParT"], "model.compile", "config/model/tag_ParT.yaml"),
    ("tag_particlenet", ["model=tag_particlenet"], "model.compile", "config/model/tag_particlenet.yaml"),
    ("tag_transformer", ["model=tag_transformer"], "model.compile", "config/model/tag_transformer.yaml"),
    ("PlainGraphTrans", ["model=tag_PlainGraphTrans"], "model.compile", "config/model/tag_PlainGraphTrans.yaml"),
    ("PlainGraphGPS", ["model=tag_PlainGraphGPS"], "model.compile", "config/model/tag_PlainGraphGPS.yaml"),
    ("PNParTGraphTrans", ["model=tag_ParticleNetParTGraphTrans"], "model.compile", "config/model/tag_ParticleNetParTGraphTrans.yaml"),
    ("PNParTGraphGPS", ["model=tag_ParticleNetParTGraphGPS"], "model.compile", "config/model/tag_ParticleNetParTGraphGPS.yaml"),
]

ITER_RE = re.compile(r"Finished iteration (\d+) after ([0-9.]+)s")

# Rows --apply must never flip. EMPTY as of 2026-08-10: every knob-bearing model now
# compiles and survives a real backward, so beta-PERF decides all of them on speed.
#
# History, kept because the reasoning is what matters if a row ever regresses:
#   * PlainGraphTrans crashed on the backward because the kNN k-cap made k symbolic --
#     fixed by the static-k compile twin (plaingraphtrans.knn, `compiled_knn`).
#   * LNetSlimGraphGPS crashed because AOT saved the GPS layer's channel-first <->
#     channel-last transpose as a graph output, and inductor cannot stride-order a VIEW
#     output whose strides carry a symbolic dim -- fixed by a scoped
#     torch._functorch.config.patch(recompute_views=True) in its wrapper.
#   * the three PairEmbed-twin models sat here for TRAINING-numerics reasons until the
#     weighted pair-BN made them faithful (train delta <= 3.2e-15).
# If a model lands here again, the bar is the same: a CORRECTNESS reason no walltime
# number can overrule -- not "we did not measure it yet".
NO_APPLY = set()


def find_batchsize(row_overrides, knob, config_path, bs_start, bs_max, safety):
    """Largest power-of-two batchsize that survives a full training step, for THIS row.

    Reuses ``utils.find_lr.find_max_batch_size`` rather than reimplementing the doubling
    search, so the two tools cannot drift apart -- it runs a real fwd + bwd + optimizer
    step at each candidate, so the measured memory is what training actually uses.

    Sized ONCE per row and then applied to BOTH states, which is what keeps the
    eager-vs-compiled comparison paired. Sizing per state would compare two different
    batch sizes and measure the wrong thing. The search returns a power of two, and
    inductor's memory delta is nowhere near a factor of two, so the two states land on the
    same rung in practice. If a compiled row ever OOMs at the eager-sized batch, that is
    itself the finding and the row reports it.

    ``knob=false`` IS PASSED EXPLICITLY, not left to the yaml. Eleven production configs now
    ship ``compile: true``, so a bare ``model=tag_ParT`` sizes a COMPILED model -- which
    contradicts the paragraph above, spends an inductor build per row inside the driver
    process, and leaves dynamo cache entries holding those models alive for every
    subsequent row's search.

    Everything is torn down between rows for the same reason: the timing runs are
    subprocesses and start clean, but the searches all share THIS process, so a row that
    left 4 GB resident would hand the next row a smaller ceiling and silently undersize it.

    CUDA only -- find_max_batch_size returns the configured batchsize unchanged on CPU.
    """
    import gc

    import hydra
    import torch
    from utils.find_lr import build_experiment, find_max_batch_size

    with hydra.initialize_config_dir(config_dir=str(REPO / config_path), version_base=None):
        cfg = hydra.compose(config_name="toptagging",
                            overrides=[*row_overrides, f"{knob}=false", "save=false"])
    exp = build_experiment(cfg)
    try:
        return int(find_max_batch_size(exp, bs_start, bs_max, safety))
    finally:
        del exp
        try:
            torch._dynamo.reset()
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_once(overrides, iters, window, config_path, timeout, seed):
    cmd = [
        sys.executable, "run.py",
        "--config-path", config_path, "--config-name", "toptagging",
        *overrides,
        "save=false",
        f"seed={seed}",  # makes the two states of a row PAIRED -- see the module docstring
        "training.epochs=null", f"training.iterations={iters}",
        f"training.validate_every_n_steps={iters + 1}",  # keep validation out of the window
    ]
    try:
        out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                             timeout=timeout).stdout
    except subprocess.TimeoutExpired as e:
        return None, f"TIMEOUT after {timeout}s", (e.stdout or "")[-2000:]
    marks = {int(n): float(t) for n, t in ITER_RE.findall(out)}
    lo, hi = window
    if lo not in marks or hi not in marks:
        return None, f"missing timing lines (have {sorted(marks)})", out[-2000:]
    return (hi - lo) / (marks[hi] - marks[lo]), None, None


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None, help="substring filter on row names")
    ap.add_argument("--iters", type=int, default=110)
    ap.add_argument("--window", type=int, nargs=2, default=(10, 100), metavar=("LO", "HI"),
                    help="steady-state window; both bounds must be iterations the run "
                         "actually logs (torch logs 1, 10, 100, 1000, ... plus validation steps)")
    ap.add_argument("--config-path", default="config", help="hydra config dir (default: production)")
    ap.add_argument("--timeout", type=int, default=3600, help="per-run seconds")
    ap.add_argument("--margin", type=float, default=0.03,
                    help="minimum relative win before a flip is recommended (default 3%%)")
    ap.add_argument("--find-batchsize", action="store_true",
                    help="size each row with find_lr's OOM doubling search and use that "
                         "batchsize for BOTH states, instead of whatever the yaml carries "
                         "(which is the unswept 512 fallback until you sweep). GPU only.")
    ap.add_argument("--bs-start", type=int, default=16)
    ap.add_argument("--bs-max", type=int, default=16384)
    ap.add_argument("--bs-safety", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234,
                    help="fixed seed for BOTH states of every row; without it the runs "
                         "shuffle differently and the comparison is unpaired")
    ap.add_argument("--apply", action="store_true",
                    help="edit the production yaml compile knobs per the verdicts")
    args = ap.parse_args()

    rows = [r for r in MATRIX if not args.models or any(m in r[0] for m in args.models)]
    results, recs = [], []
    for name, base, knob, yaml_path in rows:
        row_base = list(base)
        if args.find_batchsize:
            try:
                bs = find_batchsize(row_base, knob, args.config_path,
                                    args.bs_start, args.bs_max, args.bs_safety)
                row_base = row_base + [f"training.batchsize={bs}"]
                print(f"[bperf] {name} batchsize={bs} (sized once eager, used for both "
                      f"states)", flush=True)
            except Exception as e:
                print(f"[bperf] {name} batchsize search FAILED ({type(e).__name__}: {e}); "
                      f"falling back to the config value", flush=True)
        pair = {}
        for state in ("false", "true"):
            print(f"[bperf] {name} {knob}={state} ...", flush=True)
            its, err, tail = run_once(row_base + [f"{knob}={state}"], args.iters,
                                      tuple(args.window), args.config_path, args.timeout,
                                      args.seed)
            pair[state] = its
            if err:
                print(f"[bperf]   FAILED: {err}\n{tail}", flush=True)
        e, c = pair["false"], pair["true"]
        speedup = (c / e) if (e and c) else None
        want_true = speedup is not None and speedup >= 1 + args.margin
        want_false = speedup is not None and speedup <= 1 - args.margin
        verdict = ("compile: true" if want_true else
                   "compile: false" if want_false else
                   "within margin -- keep current" if speedup else "INCOMPLETE")
        if name in NO_APPLY and want_true:
            verdict += " -- NOT APPLIED (semantics/backward, see cgenn-compile.md)"
        results.append((name, e, c, speedup, verdict))
        if speedup and (want_true or want_false) and not (name in NO_APPLY and want_true):
            recs.append((yaml_path, knob, "true" if want_true else "false"))
        print(f"[bperf] {name}: eager={e and f'{e:.2f}'} it/s, "
              f"compiled={c and f'{c:.2f}'} it/s -> {verdict}", flush=True)

    lines = [
        f"\n## beta-PERF {datetime.datetime.now():%Y-%m-%d %H:%M} "
        f"(tree={args.config_path}, iters={args.iters}, window={tuple(args.window)}, "
        f"cuda={'yes' if _cuda() else 'NO -- not decision-grade'})",
        "", "| row | eager it/s | compiled it/s | speedup | verdict |", "|---|---|---|---|---|",
    ]
    for name, e, c, s, v in results:
        lines.append(f"| {name} | {e and f'{e:.2f}' or '-'} | {c and f'{c:.2f}' or '-'} "
                     f"| {s and f'{s:.3f}x' or '-'} | {v} |")
    lines += ["", "Recommended one-liners (production tree):"]
    lines += [f"- `{y}`: set `{k.split('.')[-1]}: {v}`" for y, k, v in recs] or ["- none"]
    report = "\n".join(lines)
    print(report)
    (REPO / "bperf_results.md").open("a").write(report + "\n")

    if args.apply and recs:
        for yaml_path, knob, val in recs:
            p = REPO / yaml_path
            t = p.read_text()
            key = knob.split(".")[-1]
            new, n = re.subn(rf"^(\s*{key}:\s*)(true|false)", rf"\g<1>{val}", t, count=1, flags=re.M)
            if n == 1 and new != t:
                p.write_text(new)
                print(f"[bperf] applied {key}: {val} -> {yaml_path}")
            elif n == 1:
                print(f"[bperf] {yaml_path} already at {key}: {val}")
            else:
                print(f"[bperf] SKIP {yaml_path}: no unique '{key}:' line found")


if __name__ == "__main__":
    main()
