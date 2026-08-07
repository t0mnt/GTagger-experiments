#!/usr/bin/env python
"""Collect the per-run LaTeX table rows into one comparison table across models.

Every ``run.py`` invocation logs one line per evaluated split:

    table test: <Model> & <frames> & <iters>[N trials] & <params> & <acc> & <auc>
                & <rej03> & <rej05> & <rej08> & <time>s & <flops> & <kNN> \\

For warm-started runs that line already carries ``mean +- std`` over the trials in
that run directory. This script walks a ``runs/`` tree, takes the latest such line
per run directory (highest ``run_idx`` = most trials), de-duplicates by
(model, frames, kNN) — so ablation variants of the same model keep separate rows,
and a true re-run of an identical variant supersedes older ones by the log file's
mtime (NOT by directory order: run-dir names end in a random suffix, so path order
says nothing about recency) — and assembles them into a single LaTeX ``tabular``.

    python utils/aggregate_table.py                       # scans runs/, split=test
    python utils/aggregate_table.py --runs runs/topt_local_debug --split test --out table.tex
"""

import argparse
import os
import re
from glob import glob

COLUMNS = {
    # toptagging / toptagxl (binary): the TaggingExperiment row format
    "toptagging": (
        r"model & frames & iters & params & accuracy & AUC & "
        r"$1/\epsilon_B$(0.3) & (0.5) & (0.8) & time & FLOPs & kNN"
    ),
    "toptagxl": (
        r"model & frames & iters & params & accuracy & AUC & "
        r"$1/\epsilon_B$(0.3) & (0.5) & (0.8) & time & FLOPs & kNN"
    ),
    # jctagging (10-class): per-class rejections. Column set MUST match
    # TaggingExperiment._log_table_row's output exactly (... & time & FLOPs & kNN) --
    # a legend narrower than the row silently mislabels every column after the gap.
    "jctagging": (
        r"model & frames & iters & params & accuracy & AUC(ovo) & "
        r"$1/\epsilon_B$: HBB & HCC & HGG & H4Q & HQQL & TBQQ & TBL & WQQ & ZQQ & "
        r"time & FLOPs & kNN"
    ),
}


def latest_row(run_dir, split):
    """Return (row, mtime) for the `table <split>:` row from the highest-index log
    in `run_dir`; mtime is the modification time of the log the row came from
    (the recency stamp used for cross-run-dir dedup)."""
    logs = sorted(
        glob(os.path.join(run_dir, "out_*.log")),
        key=lambda p: int(re.search(r"out_(\d+)\.log$", p).group(1)),
    )
    pattern = re.compile(rf"table {re.escape(split)}:\s*(.*?)\s*\\\\\s*$")
    row, src = None, None
    for log in logs:  # later logs accumulate more trials -> keep the last match
        try:
            with open(log) as f:
                for line in f:
                    m = pattern.search(line)
                    if m:
                        row, src = m.group(1), log
        except OSError:
            continue
    if row is None:
        return None, 0.0
    try:
        mtime = os.path.getmtime(src)
    except OSError:
        mtime = 0.0
    return row, mtime


def _is_aggregated(row):
    """True if this row already carries an in-run mean +- std (the warm-start accumulator)."""
    return "\\pm" in row or re.search(r"\[\d+ trials\]", row) is not None


def _consolidate(key, entries):
    """Collapse one variant's run dirs into a single (mtime, row, dirs) table entry.

    entries: list of (mtime, row, run_dir) for the SAME (exp_type, model, frames, kNN).
    """
    entries = sorted(entries, key=lambda e: e[0])
    if len(entries) == 1:
        return entries[-1][0], entries[-1][1], [entries[-1][2]]
    if any(_is_aggregated(r) for _, r, _ in entries):
        # mixed mechanisms, or several dirs each already holding their own trials --
        # merging would double-count, so fall back to the historical newest-wins rule
        newest = entries[-1]
        print(f"[note] {key}: {len(entries)} run dirs, at least one already aggregated in-run; "
              f"keeping the newest ({newest[2]}) rather than merging")
        return newest[0], newest[1], [newest[2]]

    cols = [[c.strip() for c in r.split("&")] for _, r, _ in entries]
    ncols = len(cols[0])
    if any(len(c) != ncols for c in cols):
        newest = entries[-1]
        print(f"[note] {key}: run dirs disagree on column count; keeping the newest ({newest[2]})")
        return newest[0], newest[1], [newest[2]]

    n = len(entries)
    out = []
    for j in range(ncols):
        vals = [c[j] for c in cols]
        if all(v == vals[0] for v in vals):          # model / frames / params / kNN
            out.append(vals[0])
            continue
        nums, unit = [], ""
        for v in vals:                                # train_time carries a trailing "s"
            m = re.fullmatch(r"(-?[\d.]+(?:[eE][-+]?\d+)?)([a-zA-Z%]*)", v)
            if m is None:
                nums = None
                break
            nums.append(float(m.group(1)))
            unit = m.group(2)
        if nums is None:
            out.append(vals[-1])                      # not numeric -> newest run's value
            continue
        mean = sum(nums) / n
        var = sum((x - mean) ** 2 for x in nums) / (n - 1)
        # match the per-run formatter's precision so merged and in-run rows look alike
        dec = max((len(v.split(".")[1].rstrip("s")) if "." in v else 0) for v in vals)
        fmt = f".{dec}f" if "e" not in vals[0].lower() else ".3e"
        out.append(f"${format(mean, fmt)} \\pm {format(var ** 0.5, fmt)}${unit}")

    out[2] = f"{out[2]} [{n} trials]"                 # iters column carries the trial count
    dirs = [d for _, _, d in entries]
    print(f"[note] {key}: merged {n} trials from {', '.join(os.path.basename(x) for x in dirs)}")
    return entries[-1][0], " & ".join(out), dirs


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--runs", default="runs", help="root directory to scan (default: runs)")
    ap.add_argument("--split", default="test", help="which split's row to collect (default: test)")
    ap.add_argument("--out", default=None, help="optional path to write the .tex table")
    args = ap.parse_args()

    run_dirs = sorted(
        {
            os.path.dirname(p)
            for p in glob(os.path.join(args.runs, "**", "out_*.log"), recursive=True)
        }
    )
    # key on (model, frames, kNN) -- the variant axes that are table columns -- so
    # ablation runs of the SAME model (identity vs learnedpd frames, deltaR vs
    # minkowski kNN) each keep their own row. On a true re-run of the identical
    # variant the row with the NEWEST log mtime wins (run-dir names carry a random
    # suffix, so lexicographic path order says nothing about which run is current).
    rows = {}  # key -> (mtime, row, run_dir)
    for d in run_dirs:
        row, mtime = latest_row(d, args.split)
        if row is None:
            continue
        etype = "unknown"
        try:
            with open(os.path.join(d, "config.yaml")) as f:
                m = re.search(r"^exp_type:\s*(\S+)", f.read(), re.M)
                if m:
                    etype = m.group(1)
        except OSError:
            pass
        cells = [c.strip() for c in row.split("&")]
        model = cells[0]
        frames = cells[1] if len(cells) > 1 else ""
        knn = cells[-1] if len(cells) > 2 else ""
        # different tasks (toptagging / toptagxl / jctagging) report different metric
        # columns -> group into SEPARATE tables keyed by the run's exp_type
        key = (etype, model, frames, knn)
        rows.setdefault(key, []).append((mtime, row, d))

    # Trials of the same variant are INDEPENDENT run dirs (see GUIDE section 8) -- three
    # plain submissions, no warm start, no shared directory. Group them here and form
    # mean +- std across the group, so the raw per-trial values survive to this point and
    # the statistic is chosen at presentation time rather than frozen during training.
    # A run whose own row is already `[N trials] mean +- std` came from the in-run
    # accumulator (fresh-trial warm starts into one dir); that path still works, so such a
    # group is NOT merged -- the newest row wins, exactly as before.
    rows = {k: _consolidate(k, v) for k, v in rows.items()}

    if not rows:
        print(f"No 'table {args.split}:' rows found under {args.runs}/")
        return

    etypes = sorted({k[0] for k in rows})
    tables = []
    for et in etypes:
        body = " \\\\\n".join(rows[k][1] for k in sorted(rows) if k[0] == et) + " \\\\"
        first = next(rows[k][1] for k in sorted(rows) if k[0] == et)
        ncols = first.count("&") + 1
        colspec = "l l r r " + "c " * max(0, ncols - 7) + "r r l"
        legend = COLUMNS.get(et, " & ".join(["col"] * ncols))
        tables.append(
            f"% task: {et}\n"
            "% columns: " + legend + "\n"
            "\\begin{tabular}{" + colspec.strip() + "}\n"
            "\\hline\n" + body + "\n\\hline\n"
            "\\end{tabular}\n"
        )
    table = "\n".join(tables)
    print(table)
    if args.out:
        with open(args.out, "w") as f:
            f.write(table)
        print(f"[wrote {args.out}]")


if __name__ == "__main__":
    main()
