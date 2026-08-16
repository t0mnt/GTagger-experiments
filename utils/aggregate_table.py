#!/usr/bin/env python
"""Collect the per-run LaTeX table rows into one comparison table across models.

Every ``run.py`` invocation logs one line per evaluated split:

    table test: <Model> & <frames> & <iters>[N trials] & <params> & <acc> & <auc>
                & <rej03> & <rej05> & <rej08> & <time>s & <flops> & <kNN> \\

For warm-started runs that line already carries ``mean +- std`` over the trials in
that run directory (the CANONICAL trial mechanism for campaign rows — GUIDE.md §8).
This script walks a ``runs/`` tree, takes the latest such line per run directory
(highest ``run_idx`` = most trials), groups directories by (task, model, frames, kNN)
— so ablation variants of the same model keep separate rows — and consolidates each
group into one row. Single-trial rows in a group pool into ``mean +- std [n trials]``
at parse time; the pooling REFUSES (keeps the newest by log mtime, with a note) when
inference could lie: mixed with an in-run-aggregated row (double-counting),
disagreeing iters/params/FLOPs cells (different experiments sharing a key), or
identical metrics across dirs (pinned-seed clones). Recency is the log file's mtime,
NOT directory order: run-dir names end in a random suffix, so path order says
nothing about recency. Output is a single LaTeX ``tabular`` per task.

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
        r"model & frames & iters & jets & params & accuracy & AUC & "
        r"$1/\epsilon_B$(0.3) & (0.5) & (0.8) & time(h) & FLOPs & kNN"
    ),
    "toptagxl": (
        r"model & frames & iters & jets & params & accuracy & AUC & "
        r"$1/\epsilon_B$(0.3) & (0.5) & (0.8) & time(h) & FLOPs & kNN"
    ),
    # jctagging (10-class): per-class rejections. Column set MUST match
    # TaggingExperiment._log_table_row's output exactly (... & time & FLOPs & kNN) --
    # a legend narrower than the row silently mislabels every column after the gap.
    "jctagging": (
        r"model & frames & iters & jets & params & accuracy & AUC(ovo) & "
        r"$1/\epsilon_B$: HBB & HCC & HGG & H4Q & HQQL & TBQQ & TBL & WQQ & ZQQ & "
        r"time(h) & FLOPs & kNN"
    ),
}


def training_batchsize(run_dir):
    """`training.batchsize` from a run's config.yaml, or None.

    Scoped to the `training:` block on purpose: the dump also carries
    `evaluation.batchsize` (2048 in the tagging configs), and a bare
    `^\s*batchsize:` search matches whichever comes first. Regex rather than
    OmegaConf because this module is stdlib-only -- that is what lets it run on a
    login node's system python without the container.
    """
    try:
        with open(os.path.join(run_dir, "config.yaml")) as f:
            txt = f.read()
    except OSError:
        return None
    blk = re.search(r"^training:\n((?:[ \t]+.*\n)*)", txt, re.M)
    if blk is None:
        return None
    m = re.search(r"^\s+batchsize:\s*(\d+)", blk.group(1), re.M)
    return int(m.group(1)) if m else None


def seconds_to_hours(cell):
    """Rewrite a train-time cell from seconds to hours, pooled or not.

    `9415s` -> `2.61h`; `$9415.0 \\pm 12.3$s` -> `$2.61 \\pm 0.00$h`. Anything that
    matches neither form is returned untouched rather than mangled.
    """
    m = re.fullmatch(r"\$(-?[\d.]+) \\pm (-?[\d.]+)\$s", cell)
    if m:
        return f"${float(m.group(1)) / 3600:.2f} \\pm {float(m.group(2)) / 3600:.2f}$h"
    m = re.fullmatch(r"(-?[\d.]+)s", cell)
    if m:
        return f"{float(m.group(1)) / 3600:.2f}h"
    return cell


def _sole(values):
    """The single value in a set, or None if it is empty, unknown, or contradictory."""
    if not values:
        return None
    clean = {v for v in values if v is not None}
    return clean.pop() if len(clean) == 1 else None


def augment_row(row, batchsize):
    """Add a `jets seen` cell after `iters`, and put the train time in hours.

    Applied at RENDER time, after pooling, so no column index used by the invariant
    check or the averaging shifts. jets = iters x batchsize, which is the fair-budget
    number (equal epochs, not equal steps): a batch-512 row and a batch-128 row with
    4x the iterations have seen the SAME data, and the iters column alone hides that.
    Falls back to `n/a` when the batchsize is unknown or the trials disagree on it,
    rather than guessing.
    """
    cells = [c.strip() for c in row.split("&")]
    if len(cells) >= 3:  # ... & time & FLOPs & kNN -- time is always third from the end
        cells[-3] = seconds_to_hours(cells[-3])
    jets = "n/a"
    m = re.match(r"^\s*(\d+)", cells[2]) if len(cells) > 2 else None
    if m is not None and batchsize:
        jets = f"{int(m.group(1)) * batchsize / 1e6:.1f}M"
    cells.insert(3, jets)
    return " & ".join(cells)


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

    # invariant cells must agree before pooling: iters (2), params (3), FLOPs (-2).
    # A mismatch means these dirs are DIFFERENT experiments that happen to share the
    # table key (budget/width ablations) -- averaging them would fabricate an ensemble.
    for idx, name in ((2, "iters"), (3, "params"), (ncols - 2, "FLOPs")):
        if len({c[idx] for c in cols}) > 1:
            newest = entries[-1]
            print(f"[warning] {key}: run dirs disagree on {name} "
                  f"({', '.join(sorted({c[idx] for c in cols}))}); NOT pooling -- these are "
                  f"different experiments sharing a table key; keeping the newest ({newest[2]})")
            return newest[0], newest[1], [newest[2]]

    n = len(entries)
    out = []
    averaged = []                                     # column indices that actually differed
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
        averaged.append(j)
        mean = sum(nums) / n
        var = sum((x - mean) ** 2 for x in nums) / (n - 1)
        # match the per-run formatter's precision so merged and in-run rows look alike
        dec = max((len(v.split(".")[1].rstrip("s")) if "." in v else 0) for v in vals)
        fmt = f".{dec}f" if "e" not in vals[0].lower() else ".3e"
        out.append(f"${format(mean, fmt)} \\pm {format(var ** 0.5, fmt)}${unit}")

    # identical metrics across dirs = a pinned seed produced clones (only train time, if
    # anything, differs). Pooling would stamp [n trials] +- 0.000 on what is one trial.
    if not [j for j in averaged if j != ncols - 3]:
        newest = entries[-1]
        print(f"[warning] {key}: {n} run dirs with IDENTICAL metrics -- same pinned seed? "
              f"NOT pooling (that would be fake precision); keeping one ({newest[2]}). "
              f"Unset `seed` (or vary it per submission) and re-run the extra trials.")
        return newest[0], newest[1], [newest[2]]

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
    batchsizes = {}  # key -> {training.batchsize}; a set, so disagreement is visible
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
        batchsizes.setdefault(key, set()).add(training_batchsize(d))

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
        keys = [k for k in sorted(rows) if k[0] == et]
        # one batchsize per key, or None -> augment_row writes `n/a` rather than guessing
        shown = [augment_row(rows[k][1], _sole(batchsizes.get(k))) for k in keys]
        body = " \\\\\n".join(shown) + " \\\\"
        ncols = shown[0].count("&") + 1
        # model frames | iters jets params | metrics... | time flops | knn
        colspec = "l l r r r " + "c " * max(0, ncols - 8) + "r r l"
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
