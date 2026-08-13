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
Both shrink if the window starts past the thermal ramp.

WHICH WINDOWS ARE EVEN POSSIBLE. The it/s comes from the run's own
"Finished iteration N after Ts" lines, and ``base_experiment`` emits those only at
``step in [0, 9, 99, 999, 9999, 99999]`` -- N in {1, 10, 100, 1000, 10000, 100000} -- or
every ``validate_every_n_steps``, which this driver deliberately pushes past the end of
the run so a validation pass cannot land inside the timing window (the logged time is raw
wall clock since training start; it has no train-only component to parse instead). So the
window bounds must BOTH be powers-of-ten marks <= --iters. ``--window 100 300`` cannot
work at any --iters, and earlier revisions of this docstring recommended exactly that;
``check_window`` now rejects it in a second rather than after the runs.

  * cheap screen:      --iters 110  --window 10 100      (the default; 90 steps from 10)
  * decision-grade:    --iters 1010 --window 100 1000    (900 steps, all past the ramp)

If a row lands inside the margin, re-run it with the states swapped before believing the
sign.

One-shot instrument: cleanup.md schedules its deletion once the numbers are recorded in
the compile log and the knob flips are committed.

Usage:
    python utils/bperf.py                       # full matrix, table only
    python utils/bperf.py --models tag_cgenn    # substring filter
    python utils/bperf.py --iters 1010 --window 100 1000
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
# Run as a SCRIPT (`python utils/bperf.py`), sys.path[0] is utils/, not the repo root,
# so find_batchsize()'s `from utils.find_lr import ...` raises ModuleNotFoundError --
# caught only as "batchsize search FAILED", after which every row silently falls back
# to the yaml's unswept 512 and CGENN OOMs a 93 GB H100. Importing bperf from the repo
# root hides this, which is exactly how it got shipped.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

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

# What `training.batchsize` resolves to for a recipe nobody has swept: the value inherited
# from the parent config, NOT a value anyone chose for the model. Named here because the
# --find-batchsize failure path has to be able to say what it is refusing to run at.
UNSWEPT_FALLBACK_BS = 512

# The iterations base_experiment logs a timing line for, absent a validation pass:
# `step in [0, 9, 99, 999, 9999, 99999]`. run_once() pushes validate_every_n_steps past
# the end of the run, so these are the ONLY marks available -- see the module docstring.
TORCH_MARKS = [1, 10, 100, 1000, 10000, 100000]


def check_window(window, iters):
    """Reject an unachievable --window before spending the runs on it.

    A window bound that is never logged makes every row fail with 'missing timing lines'
    -- after the full matrix has already run. On a GPU with --find-batchsize that is hours
    for zero numbers, so this is a fail-fast, not a nicety.
    """
    lo, hi = window
    marks = [m for m in TORCH_MARKS if m <= iters]
    bad = [b for b in (lo, hi) if b not in marks]
    if bad or lo >= hi:
        raise SystemExit(
            f"--window {lo} {hi} is not measurable at --iters {iters}. The run only logs a "
            f"timing line at {marks} (base_experiment logs at steps [0, 9, 99, 999, ...], "
            f"and validation is pushed past the end of the run on purpose so it cannot "
            f"pollute the window). Pick both bounds from that list, lo < hi -- e.g. "
            f"`--iters 110 --window 10 100` to screen, `--iters 1010 --window 100 1000` "
            f"for a decision.")

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

    That "nowhere near a factor of two" is an ASSUMPTION, and the CGENN work has since put a
    dent in it: eager retention differs by ~6x between gp_impls where compiled retention is
    equal to within 1 MB (docs/cgenn-compile.md, CORRECTION), because AOT's partitioner
    re-decides what to save. Eager is the memory-hungrier state, so an eager-sized batch is
    the CONSERVATIVE shared choice and the pairing stays sound -- but it also means these
    it/s are measured at a batch nobody trains at, since find_lr sizes on the SHIPPED
    (compiled) config. Read the ratio, not the absolute, and do not rank gp_impls by it/s
    from this driver.

    ``knob=false`` IS PASSED EXPLICITLY, not left to the yaml. EVERY model config that carries
    a ``compile`` key ships it TRUE (20 of the 36 in ``config/model``: 18 tag_, 2 amp_; the
    other 16 have no such key and none anywhere says false), so a bare ``model=tag_ParT``
    would size a COMPILED model -- which
    contradicts the paragraph above, spends an inductor build per row inside the driver
    process, and leaves dynamo cache entries holding those models alive for every
    subsequent row's search.

    Everything is torn down between rows for the same reason: the timing runs are
    subprocesses and start clean, but the searches all share THIS process, so a row that
    left 4 GB resident would hand the next row a smaller ceiling and silently undersize it.

    Inherits find_max_batch_size's default probe batch, which is CONSTRUCTED at +5 sd of
    the batch total rather than drawn at random. That lowers the chosen rung slightly
    versus the old random probe; harmless here, because this driver reports a RATIO and
    both states get the same batch, and strictly better than OOMing partway through a
    row's timing run.

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
    """One timed run of one state. Returns (it_per_s, error, tail).

    BOTH STREAMS ARE READ, and that is not defensive tidying -- it is the difference
    between this script working and not. `base_experiment._init_logger` attaches a bare
    `logging.StreamHandler()`, whose default stream is **stderr**, so every line this
    function exists to parse -- including "Finished iteration N after Ts" -- arrives on
    stderr and NONE of it on stdout. Reading `.stdout` alone made every row report
    `missing timing lines (have [])` and the whole matrix INCOMPLETE, on GPU as much as on
    CPU. Verified: `bperf_results.md` had never recorded a single measured number.
    """
    cmd = [
        sys.executable, "run.py",
        "--config-path", config_path, "--config-name", "toptagging",
        *overrides,
        "save=false",
        f"seed={seed}",  # makes the two states of a row PAIRED -- see the module docstring
        "training.epochs=null", f"training.iterations={iters}",
        f"training.validate_every_n_steps={iters + 1}",  # keep validation out of the window
        # EVALUATION OFF. config/default.yaml ships `evaluate: true`, and without this every
        # run followed its {iters} timed training steps with a full test+val pass -- 1578
        # batches at the evaluation batchsize, for numbers this driver reads none of. It
        # dominated the sweep: on the 2026-08-12 H100 run the two rows that COMPLETED took
        # 125 and 119 minutes while the row that CRASHED before evaluation took 41, which is
        # how the cost was spotted. Timing comes from the "Finished iteration N" lines during
        # TRAINING, so dropping evaluation cannot move a measured number.
        # (`plot` needs no override: base_experiment gates it behind `save`, already false.)
        "evaluate=false",
    ]
    try:
        p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout)
        out = (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired as e:
        def _txt(s):
            return s.decode(errors="replace") if isinstance(s, bytes) else (s or "")
        return None, f"TIMEOUT after {timeout}s", (_txt(e.stdout) + _txt(e.stderr))[-2000:]
    marks = {int(n): float(t) for n, t in ITER_RE.findall(out)}
    lo, hi = window
    if not marks:
        return None, ("no timing lines at all -- the run almost certainly died before "
                      "training; read the tail"), out[-3000:]
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
    check_window(tuple(args.window), args.iters)   # before any run, not after all of them

    rows = [r for r in MATRIX if not args.models or any(m in r[0] for m in args.models)]
    if not rows:
        raise SystemExit(
            f"--models {args.models} matched no row. Available: "
            f"{[r[0] for r in MATRIX]}")
    results, recs = [], []
    for name, base, knob, yaml_path in rows:
        row_base = list(base)
        if args.find_batchsize:
            # FATAL, not a fallback. This used to print a warning and continue at the yaml
            # value -- which is the unswept 512 -- and that has now cost two runs:
            # a ModuleNotFoundError (see the sys.path note at the top of this file) and an
            # IndexError in the probe, each turning a 30-second failure into hours of H100
            # time that OOM'd row by row. There is no useful "continue" here: --find-batchsize
            # is an explicit request to SIZE the rows, and a size the driver could not measure
            # is not a size. Wanting the yaml value is spelled by omitting the flag.
            try:
                bs = find_batchsize(row_base, knob, args.config_path,
                                    args.bs_start, args.bs_max, args.bs_safety)
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise SystemExit(
                    f"[bperf] {name}: --find-batchsize failed ({type(e).__name__}: {e}).\n"
                    f"Stopping instead of running the matrix at the config batchsize -- for "
                    f"an unswept recipe that is {UNSWEPT_FALLBACK_BS}, which OOMs the CGENN "
                    f"rows on a 93 GB H100. Fix the search, or drop --find-batchsize to "
                    f"accept the yaml value deliberately.") from e
            row_base = row_base + [f"training.batchsize={bs}"]
            print(f"[bperf] {name} batchsize={bs} (sized once eager, used for both "
                  f"states)", flush=True)
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
            # ANCHOR THE INDENT from the knob path rather than collapsing both conventions
            # to a bare "compile". `model.compile` is the WRAPPER knob and sits at column 0;
            # `model.net.compile` is the net's own and sits indented under `net:`. The old
            # pattern used `\s*`, which matches either, and `count=1` then took whichever
            # came FIRST in the file -- so a config carrying both would have had the wrong
            # knob flipped, silently, on a driver whose --apply edits PRODUCTION yamls.
            # Verified across config/model: every wrapper knob is at indent 0 and every net
            # knob at indent 1. Latent today (no file carries both) and cheap to close.
            indent = "" if knob.count(".") == 1 else "[ \t]+"
            pat = rf"^({indent}{key}:[ \t]*)(true|false)\b"
            # findall BEFORE substituting, because re.subn(count=1) returns n <= 1 and so
            # cannot report multiplicity: the branch this replaces printed "no unique
            # '{key}:' line found" but could only ever fire on ZERO matches. The uniqueness
            # it claimed to check was never checked.
            hits = re.findall(pat, t, flags=re.M)
            level = "top level" if not indent else "net level"
            if len(hits) != 1:
                print(f"[bperf] SKIP {yaml_path}: expected exactly one '{key}:' at "
                      f"{level}, found {len(hits)}")
                continue
            new, _ = re.subn(pat, rf"\g<1>{val}", t, count=1, flags=re.M)
            if new != t:
                p.write_text(new)
                print(f"[bperf] applied {key}: {val} -> {yaml_path} ({level})")
            else:
                print(f"[bperf] {yaml_path} already at {key}: {val}")


if __name__ == "__main__":
    main()
