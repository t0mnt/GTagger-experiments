#!/usr/bin/env python
"""Collect everything needed to diagnose non-finite-gradient incidents into one zip.

Pure standard library (torch is optional, only for the GPU/precision report). Works from
any shell, PowerShell included:

    python utils/collect_nanlogs.py                      # auto-discover under runs/ and logs/
    python utils/collect_nanlogs.py runs/<exp>/<run> ... # or name the run directories
    python utils/collect_nanlogs.py --copy-ckpt ...      # also copy the last checkpoint
                                                         # saved BEFORE the first skip

Per run directory it gathers: config.yaml, the key lines of out_*.log (model, frames,
environment, budget, every "Skipping iteration" line with its gamma values, dumps,
abort), the models/ listing, the requirement pins of the saved source snapshot, and a
per-run summary (first/last skip step, count, longest consecutive streak, outcome).
Writes nanlogs/ and nanlogs.zip next to where it is run.
"""
import argparse
import time
import glob
import os
import re
import shutil
import sys
import zipfile

KEY = re.compile(
    r"Instantiated model|Frames approach|env:|Epoch budget|Starting to train|Constructed dataloaders|"
    r"Skipping iteration|non-finite|consecutive non-finite|Loading model|Early stopping|"
    r"Finished training|Exiting with error|Traceback|Error|slurm|job id|Creating|Using .* files|"
    r"torch|cuda|cudnn|lloca|lgatr|version|preserve_variance|mass_reg|gamma_max|Validation|val_",
    re.IGNORECASE,
)
SKIP = re.compile(r"Skipping iteration (\d+).*?\[(\d+) consecutive\]")


def summarize_log(path):
    """Key lines of one log plus its first 40 lines (job header: command line, interpreter and
    package paths, versions) and the skip statistics."""
    first = last = None
    count = 0
    maxrun = 0
    gammas = []
    aborted = False
    dumps = []
    keylines = []
    with open(path, errors="replace") as fh:
        for i, line in enumerate(fh):
            if i < 40 or KEY.search(line):
                keylines.append(line.rstrip("\n"))
            m = SKIP.search(line)
            if m:
                step, run = int(m.group(1)), int(m.group(2))
                first = step if first is None else first
                last = step
                count += 1
                maxrun = max(maxrun, run)
                g = re.search(r"gamma_mean=(\S+) gamma_max=(\S+)", line)
                if g:
                    gammas.append((step, g.group(1), g.group(2)))
            if "consecutive non-finite gradient norms" in line:
                aborted = True
            d = re.search(r"non-finite batch written to (\S+)", line)
            if d:
                dumps.append(d.group(1))
    return dict(first=first, last=last, count=count, maxrun=maxrun, aborted=aborted,
                dumps=dumps, gammas=gammas, keylines=keylines)


def collect_run(run_dir, out_root, copy_ckpt):
    name = run_dir.strip("/\\").replace("\\", "/").replace("/", "__")
    out = os.path.join(out_root, name)
    os.makedirs(out, exist_ok=True)
    report = [f"run_dir: {run_dir}"]
    for cfg in ("config.yaml", "config_0.yaml"):
        p = os.path.join(run_dir, cfg)
        if os.path.exists(p):
            shutil.copy(p, out)
    logs = sorted(glob.glob(os.path.join(run_dir, "out_*.log")))
    first_skip = None
    for lg in logs:
        s = summarize_log(lg)
        with open(os.path.join(out, os.path.basename(lg) + ".keylines.txt"), "w") as fh:
            fh.write("\n".join(s["keylines"]) + "\n")
        report.append(
            f"{os.path.basename(lg)}: skips={s['count']} first={s['first']} last={s['last']} "
            f"longest_streak={s['maxrun']} aborted={s['aborted']} dumps={s['dumps']}"
        )
        for step, gm, gx in s["gammas"][:20]:
            report.append(f"    skip at {step}: gamma_mean={gm} gamma_max={gx}")
        if s["first"] is not None and (first_skip is None or s["first"] < first_skip):
            first_skip = s["first"]
    models_dir = os.path.join(run_dir, "models")
    ckpts = []
    if os.path.isdir(models_dir):
        with open(os.path.join(out, "models.txt"), "w") as fh:
            for f in sorted(os.listdir(models_dir)):
                p = os.path.join(models_dir, f)
                fh.write(f"{f}\t{os.path.getsize(p)}\t{os.path.getmtime(p):.0f}\n")
                m = re.match(r"model_run(\d+)_it(\d+)\.pt$", f)
                if m:
                    ckpts.append((int(m.group(2)), p))
    before = [c for c in sorted(ckpts) if first_skip is not None and c[0] < first_skip]
    if before:
        report.append(f"last checkpoint before first skip ({first_skip}): {before[-1][1]}")
        if copy_ckpt:
            shutil.copy(before[-1][1], out)
            report.append("    copied")
    elif first_skip is not None:
        report.append(f"no checkpoint saved before the first skip at {first_skip}")
    src = os.path.join(run_dir, "source.zip")
    if os.path.exists(src):
        try:
            with zipfile.ZipFile(src) as z:
                names = z.namelist()
                req = [n for n in names if n.endswith("requirements.txt")]
                if req:
                    txt = z.read(req[0]).decode(errors="replace")
                    pins = [l for l in txt.splitlines() if re.search(r"lloca|lgatr|torch", l, re.I)]
                    with open(os.path.join(out, "pins.txt"), "w") as fh:
                        fh.write("\n".join(pins) + "\n")
                report.append(
                    "source snapshot: lloca-2.0 migration doc "
                    + ("PRESENT" if any(n.endswith("docs/lloca2-migration.md") for n in names) else "absent")
                    + " (present => code from after the 2.0 migration)"
                )
        except zipfile.BadZipFile:
            report.append("source.zip unreadable")
    for d in glob.glob(os.path.join(run_dir, "nonfinite_batch_step*.pt")):
        report.append(f"dump in run dir: {d} ({os.path.getsize(d)} bytes)")
    with open(os.path.join(out, "summary.txt"), "w") as fh:
        fh.write("\n".join(report) + "\n")
    print("\n".join(report))
    print()


def _sha_head(path, nbytes=4 << 20):
    """sha256 of the first nbytes of a file: a cheap fingerprint that still changes when a file
    is re-downloaded with different content (full hashes of 100 MB ROOT files are too slow)."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(nbytes))
    return h.hexdigest()[:16]


def data_report(out_root):
    """Fingerprint the datasets the tagging configs point at: file counts, bytes, mtimes and a
    head-hash of a few files per folder. Two collections taken before and after a scratch wipe
    can then be diffed line by line."""
    import hashlib

    lines = []
    roots = []
    for cfg in glob.glob("config/*.yaml"):
        with open(cfg, errors="replace") as fh:
            for line in fh:
                m = re.match(r"\s*data_dir:\s*(\S+)", line)
                if m and m.group(1) not in roots:
                    roots.append(m.group(1))
    for root in roots:
        if not os.path.isdir(root):
            lines.append(f"{root}: MISSING")
            continue
        for dirpath, dirnames, files in os.walk(root):
            dirnames.sort()
            data = sorted(f for f in files if f.endswith((".root", ".npz", ".npy", ".h5", ".hdf5", ".parquet", ".pt")))
            if not data:
                continue
            paths = [os.path.join(dirpath, f) for f in data]
            sizes = [os.path.getsize(p) for p in paths]
            mt = [os.path.getmtime(p) for p in paths]
            sample = paths[:: max(1, len(paths) // 3)][:3]
            heads = " ".join(f"{os.path.basename(p)}:{_sha_head(p)}" for p in sample)
            lines.append(
                f"{dirpath}: {len(paths)} files, {sum(sizes) / 1e9:.2f} GB, "
                f"mtime {time.strftime('%Y-%m-%d', time.localtime(min(mt)))}..{time.strftime('%Y-%m-%d', time.localtime(max(mt)))}, "
                f"head-sha {heads}"
            )
    for cfg in sorted(glob.glob("experiments/tagging/miniweaver/configs*/*.yaml")):
        with open(cfg, "rb") as fh:
            lines.append(f"{cfg}: sha256 {hashlib.sha256(fh.read()).hexdigest()[:16]}")
    with open(os.path.join(out_root, "data.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print()


def environment_report(out_root):
    """Everything about the software stack that a run log does NOT record (the per-run `env:`
    line lists environment variables only). Full pip freeze, interpreter location (a venv or a
    container image's site-packages), torch/CUDA/cuDNN/driver versions, GPU, the precision
    switches, SDPA backends, repo commit. Diff two of these to see what a wipe changed."""
    import platform
    import subprocess

    lines = [
        f"date {time.strftime('%Y-%m-%d %H:%M:%S')}  host {platform.node()}",
        f"python {sys.version.split()[0]}  executable {sys.executable}  prefix {sys.prefix}",
        f"VIRTUAL_ENV={os.environ.get('VIRTUAL_ENV')}  CONDA_PREFIX={os.environ.get('CONDA_PREFIX')}",
    ]
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        lines.append(f"repo HEAD {head}{' (dirty: ' + str(len(dirty.splitlines())) + ' files)' if dirty else ''}")
    except Exception as e:
        lines.append(f"git unavailable: {e}")
    try:
        import importlib.metadata as md

        for pkg in ("torch", "lloca", "lgatr", "xformers", "torch-geometric", "triton", "numpy", "hydra-core", "omegaconf"):
            try:
                dist = md.distribution(pkg)
                lines.append(f"{pkg}=={dist.version}  @ {dist.locate_file('')}")
            except md.PackageNotFoundError:
                lines.append(f"{pkg}: not installed")
    except Exception as e:  # pragma: no cover
        lines.append(f"metadata unavailable: {e}")
    try:
        import torch

        lines.append(f"torch.version.cuda={torch.version.cuda} cudnn={torch.backends.cudnn.version()} torch.file={torch.__file__}")
        lines.append(f"cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                cap = torch.cuda.get_device_capability(i)
                lines.append(f"gpu{i}={torch.cuda.get_device_name(i)} sm_{cap[0]}{cap[1]}")
        lines.append(
            f"float32_matmul_precision={torch.get_float32_matmul_precision()} "
            f"cudnn.allow_tf32={torch.backends.cudnn.allow_tf32} "
            f"matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32} "
            f"cudnn.benchmark={torch.backends.cudnn.benchmark} "
            f"cudnn.deterministic={torch.backends.cudnn.deterministic}"
        )
        try:
            lines.append(
                f"sdpa flash={torch.backends.cuda.flash_sdp_enabled()} "
                f"mem_efficient={torch.backends.cuda.mem_efficient_sdp_enabled()} "
                f"math={torch.backends.cuda.math_sdp_enabled()}"
            )
        except Exception:
            pass
    except Exception as e:
        lines.append(f"torch report unavailable: {e}")
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        lines.append("nvidia-smi: " + (smi.stdout.strip().replace("\n", " | ") or smi.stderr.strip()[:200]))
    except Exception as e:
        lines.append(f"nvidia-smi unavailable: {type(e).__name__}")
    for k in ("PYTORCH_CUDA_ALLOC_CONF", "TORCHINDUCTOR_CACHE_DIR", "TRITON_LIBCUDA_PATH", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "LD_LIBRARY_PATH", "CUDA_HOME", "APPTAINER_CONTAINER", "SINGULARITY_CONTAINER"):
        if os.environ.get(k) is not None:
            lines.append(f"env {k}={os.environ[k]}")
    with open(os.path.join(out_root, "environment.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    try:
        import subprocess as sp

        freeze = sp.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=120).stdout
        with open(os.path.join(out_root, "pip_freeze.txt"), "w") as fh:
            fh.write(freeze)
        lines.append(f"pip freeze: {len(freeze.splitlines())} packages -> pip_freeze.txt")
    except Exception as e:
        lines.append(f"pip freeze unavailable: {type(e).__name__}")
    print("\n".join(lines))
    print()


def discover(all_runs=False):
    """Run directories to collect: those whose out log carries a non-finite event, or, with
    --all-runs, every run directory that has an out log (so clean runs can be compared with
    the incident runs: same config/keyline/models.txt record, skips=0)."""
    runs = set()
    for lg in glob.glob("runs/**/out_*.log", recursive=True) + glob.glob("logs/**/*.out", recursive=True):
        if all_runs and lg.startswith("runs"):
            runs.add(os.path.dirname(lg))
            continue
        try:
            with open(lg, errors="replace") as fh:
                if any("Skipping iteration" in l or "non-finite" in l for l in fh):
                    runs.add(os.path.dirname(lg))
        except OSError:
            pass
    return sorted(runs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="*", help="run directories (default: auto-discover)")
    ap.add_argument("--copy-ckpt", action="store_true", help="copy the last checkpoint before the first skip")
    ap.add_argument("--out", default="nanlogs")
    ap.add_argument(
        "--all-runs",
        action="store_true",
        help="also collect every run under runs/ that has NO non-finite event (config, key lines, "
        "models.txt; no dumps), so clean runs can be compared with the incident runs",
    )
    args = ap.parse_args()
    runs = args.run_dirs or discover(all_runs=args.all_runs)
    if not runs:
        print("no run directories given and none with non-finite events found under runs/ or logs/")
        return 1
    os.makedirs(args.out, exist_ok=True)
    environment_report(args.out)
    data_report(args.out)
    for r in runs:
        collect_run(r, args.out, args.copy_ckpt)
    for lg in glob.glob("logs/**/*.out", recursive=True):
        s = summarize_log(lg)
        # with --all-runs every SLURM log is kept (skips=0 included): after a scratch wipe the
        # SLURM logs are often the only record of a run whose run_dir is gone
        if s["count"] or args.all_runs:
            with open(os.path.join(args.out, "slurm_" + os.path.basename(lg) + ".keylines.txt"), "w") as fh:
                fh.write("\n".join(s["keylines"]) + "\n")
    zip_path = args.out + ".zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(args.out):
            for f in files:
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, os.path.dirname(args.out) or "."))
    print(f"wrote {zip_path} ({os.path.getsize(zip_path) // 1024} KB) -- upload this file")
    return 0


if __name__ == "__main__":
    sys.exit(main())
