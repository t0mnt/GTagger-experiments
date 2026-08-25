#!/usr/bin/env python3
"""Emit `LABEL=path/roc.txt` lines for roc_overlay.py, labelling by what actually differs.

    python utils/roc_labels.py runs/topt_local_debug | tee /tmp/curves.txt
    python utils/roc_overlay.py -o roc.pdf $(cat /tmp/curves.txt)

Run dirs are named after the model CLASS, so several architecturally different runs
share a name: tag_cgenn's flash and sparse arms are both `CGENN_*`, a use_rwse ablation
is another `PlainGraphGPS_*`, and so on. Labelling by class (optionally plus one hand-
picked knob) silently merges them and the overlay then draws whichever the glob reached
first, with no indication that anything was dropped.

So: group the dirs by class, diff their saved configs, and put whatever differs into the
label. VOLATILE keys -- run identity, seed, paths, job ids, timings -- are excluded,
because those differ between repeat trials of the SAME configuration, which must collapse
to one label rather than N.
"""
import argparse, glob, os, re, sys
from omegaconf import OmegaConf

VOLATILE = re.compile(
    r"(^|\.)(run_idx|run_name|run_dir|base_dir|exp_name|seed|slurm_job_id|warm_start_idx"
    r"|mlflow\.|use_mlflow|save$|train$|evaluate$|plot$)"
)


def flat(cfg, prefix=""):
    out = {}
    for k, v in cfg.items():
        key = f"{prefix}{k}"
        if hasattr(v, "items"):
            out.update(flat(v, key + "."))
        else:
            out[key] = str(v)
    return out


def short(key):
    """Trim `model.net.gp_impl` -> `gp_impl`, keeping enough to stay unambiguous."""
    return key.split(".")[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir")
    ap.add_argument("--all-trials", action="store_true",
                    help="emit every trial; default keeps one per distinct configuration")
    args = ap.parse_args()

    entries = []
    for d in sorted(glob.glob(os.path.join(args.runs_dir, "*/"))):
        rocs = sorted(glob.glob(os.path.join(d, "plots_*", "roc.txt")))
        if not rocs:
            continue
        cfg_path = next((p for p in (os.path.join(d, "config_0.yaml"),
                                     os.path.join(d, "config.yaml")) if os.path.exists(p)), None)
        if cfg_path is None:
            print(f"# NO CONFIG, skipped: {os.path.basename(d.rstrip('/'))}", file=sys.stderr)
            continue
        cls = os.path.basename(d.rstrip("/")).rsplit("_", 1)[0]
        entries.append((cls, d, rocs[-1], flat(OmegaConf.load(cfg_path))))

    by_cls = {}
    for cls, d, roc, cfg in entries:
        by_cls.setdefault(cls, []).append((d, roc, cfg))

    seen, n_dropped = set(), 0
    for cls in sorted(by_cls):
        group = by_cls[cls]
        keys = set().union(*(c.keys() for _, _, c in group))
        differing = sorted(
            k for k in keys
            if not VOLATILE.search(k) and len({c.get(k) for _, _, c in group}) > 1
        )
        if differing:
            print(f"# {cls}: {len(group)} run dirs differ in {[short(k) for k in differing]}",
                  file=sys.stderr)
        for d, roc, cfg in group:
            bits = ", ".join(f"{short(k)}={cfg.get(k)}" for k in differing)
            label = f"{cls} ({bits})" if bits else cls
            if not args.all_trials and label in seen:
                n_dropped += 1
                continue
            seen.add(label)
            print(f"{label}={roc}")
    if n_dropped:
        print(f"# kept one trial per configuration; {n_dropped} repeat trial(s) not emitted",
              file=sys.stderr)


if __name__ == "__main__":
    main()
