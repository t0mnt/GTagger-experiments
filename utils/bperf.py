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

One-shot instrument: cleanup.md schedules its deletion once the numbers are recorded in
the compile log and the knob flips are committed.

Usage:
    python utils/bperf.py                       # full matrix, table only
    python utils/bperf.py --models tag_cgenn    # substring filter
    python utils/bperf.py --iters 210 --window 10 200
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

# Rows whose compile knob changes MORE than kernel fusion, so speed must not decide them.
# What remains after the weighted pair-BN landed (todo 4b-quater, done): only the two
# backward-crashers, which raise InductorError at the first loss.backward() regardless of
# how fast the forward is. test_nonequi_compile.test_compile_true_is_backward_verified
# also fails on a bare flip of either, by design.
#
# The three PairEmbed-twin models used to be here for TRAINING-numerics reasons and no
# longer are: the twins now weight the pair-BN statistics by the eager reference multiset
# (train delta <= 3.2e-15), so speed is once again the only open question for them. The
# GPS pair still ships false, but that is exactly the performance call --apply is allowed
# to make: they are in the sweep precisely so beta-PERF can decide them.
NO_APPLY = {"PlainGraphTrans", "LNetSlimGraphGPS"}


def run_once(overrides, iters, window, config_path, timeout):
    cmd = [
        sys.executable, "run.py",
        "--config-path", config_path, "--config-name", "toptagging",
        *overrides,
        "save=false",
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
    ap.add_argument("--apply", action="store_true",
                    help="edit the production yaml compile knobs per the verdicts")
    args = ap.parse_args()

    rows = [r for r in MATRIX if not args.models or any(m in r[0] for m in args.models)]
    results, recs = [], []
    for name, base, knob, yaml_path in rows:
        pair = {}
        for state in ("false", "true"):
            print(f"[bperf] {name} {knob}={state} ...", flush=True)
            its, err, tail = run_once(base + [f"{knob}={state}"], args.iters,
                                      tuple(args.window), args.config_path, args.timeout)
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
