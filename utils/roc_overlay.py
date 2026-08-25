#!/usr/bin/env python3
"""Overlay the ROC curves of several runs on one axes -- the multi-model figure the
per-run plots cannot produce.

Each run writes plots_<idx>/roc.txt (two columns: fpr, tpr) when the run had
`evaluation.save_roc=true`. Point this at those files:

    python roc_overlay.py -o roc_overlay.pdf \
        "PlainGraphTrans=runs/.../PlainGraphTrans_6377/plots_0/roc.txt" \
        "PlainGraphGPS=runs/.../PlainGraphGPS_7101/plots_0/roc.txt"

Emits both panels the field uses: 1/eps_B vs eps_S (log y) and SIC.
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

FONTSIZE, FONTSIZE_LEGEND = 14, 11
CYCLE = ["#0343DE", "#A52A2A", "darkorange", "black", "#2E8B57", "#8B008B",
         "#B8860B", "#4682B4", "#C71585", "#008B8B", "#556B2F", "#FF4500"]
# Separating deltaR from minkowski doubles the curve count, so colour alone runs out:
# after one pass through CYCLE the linestyle changes too. A pair that shares a colour
# is then still distinguishable in print and in greyscale.
DASHES = ["-", "--", ":", "-."]


def style(i):
    return dict(color=CYCLE[i % len(CYCLE)], linestyle=DASHES[(i // len(CYCLE)) % len(DASHES)])


def load(path):
    a = np.loadtxt(path)
    fpr, tpr = a[:, 0], a[:, 1]
    # 1/fpr is +inf at the first point; drop non-positive fpr rather than let
    # matplotlib silently swallow it (the per-run plotter warns on exactly this).
    keep = fpr > 0
    return fpr[keep], tpr[keep]


def auc_from_roc(fpr, tpr):
    o = np.argsort(fpr)
    return float(np.trapezoid(tpr[o], fpr[o])) if hasattr(np, "trapezoid") else float(np.trapz(tpr[o], fpr[o]))


def rej_at(fpr, tpr, eps_s):
    """1/eps_B at a fixed signal efficiency -- the table's own working points."""
    o = np.argsort(tpr)
    f = np.interp(eps_s, tpr[o], fpr[o])
    return float("inf") if f <= 0 else 1.0 / f


def average_by_label(entries):
    """Average the curves that share a label, so a 3-trial model draws ONE curve.

    Averages 1/fpr on a common eps_S (=tpr) grid, NOT fpr. aggregate_table.py reports
    the mean of 1/eps_B across trials, and mean(1/x) != 1/mean(x) -- averaging fpr would
    give a curve whose rejections disagree with the table it sits beside, which is the
    whole reason this option exists. Picking one trial instead is worse still: on
    PlainGraphGPS the per-trial rej(0.3) spread is +-102 and the single-trial figure
    inverted the deltaR-vs-minkowski conclusion the pooled table draws.
    """
    grid = np.linspace(0.005, 1.0, 4000)
    groups = {}
    for label, fpr, tpr in entries:
        groups.setdefault(label, []).append((fpr, tpr))
    out = []
    for label, curves in groups.items():
        if len(curves) == 1:
            out.append((label, *curves[0]))
            continue
        rejs = []
        for fpr, tpr in curves:
            o = np.argsort(tpr)
            rejs.append(np.interp(grid, tpr[o], 1.0 / fpr[o]))
        mean_rej = np.mean(rejs, axis=0)
        out.append((f"{label} [{len(curves)} trials]", 1.0 / mean_rej, grid))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("curves", nargs="*", help="LABEL=path/to/roc.txt")
    # Labels contain spaces (roc_labels.py emits "CGENN (gp_impl=flash, ...)"), so a
    # bare $(cat curves.txt) word-splits every one of them into garbage. Reading the
    # file here removes the shell from the path entirely.
    p.add_argument("--list", dest="list_file", metavar="FILE",
                   help="read LABEL=path lines from FILE ('#' comments and blanks ignored)")
    p.add_argument("-o", "--out", default="roc_overlay.pdf")
    p.add_argument("--eps", type=float, nargs="*", default=[0.3, 0.5, 0.8])
    p.add_argument("--average", action="store_true",
                   help="curves sharing a label are averaged into one (use with "
                        "roc_labels.py --all-trials, so the figure matches the pooled table)")
    args = p.parse_args()

    specs = list(args.curves)
    if args.list_file:
        with open(args.list_file) as fh:
            specs += [ln.strip() for ln in fh
                      if ln.strip() and not ln.lstrip().startswith("#")]
    if not specs:
        raise SystemExit("no curves given: pass LABEL=path arguments or --list FILE")

    entries = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"expected LABEL=path, got {spec!r}")
        # rsplit, not split: roc_labels.py puts the differing config keys INTO the
        # label ("CGENN (gp_impl=flash, ...)"), so the first "=" is inside the label.
        label, path = spec.rsplit("=", 1)
        if not os.path.exists(path):
            raise SystemExit(f"missing: {path}")
        fpr, tpr = load(path)
        entries.append((label, fpr, tpr))

    if args.average:
        entries = average_by_label(entries)

    print(f"{'model':28s} {'AUC':>7s} " + " ".join(f"{'1/eB('+str(e)+')':>11s}" for e in args.eps))
    for label, fpr, tpr in entries:
        cells = " ".join(f"{rej_at(fpr, tpr, e):11.0f}" for e in args.eps)
        print(f"{label:28s} {auc_from_roc(fpr, tpr):7.4f} {cells}")

    with PdfPages(args.out) as pdf:
        # panel 1 -- physicists' ROC
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        for i, (label, fpr, tpr) in enumerate(entries):
            ax.plot(tpr, 1 / fpr, label=label, lw=1.6, **style(i))
        rnd = np.linspace(1e-3, 1, 200)
        ax.plot(rnd, 1 / rnd, "k--", lw=1, label="random")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\epsilon_S$", fontsize=FONTSIZE)
        ax.set_ylabel(r"$1/\epsilon_B$", fontsize=FONTSIZE)
        ax.set_xlim(0, 1)
        ax.legend(frameon=False, fontsize=FONTSIZE_LEGEND)
        fig.savefig(pdf, bbox_inches="tight", format="pdf")
        plt.close(fig)

        # panel 2 -- the ParticleNet/ParT convention: eps_B on a log y axis vs eps_S,
        # AUC in the legend. Same data as panel 1, inverted axis; reviewers in this
        # field expect one or the other, so both are emitted.
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        for i, (label, fpr, tpr) in enumerate(entries):
            ax.plot(tpr, fpr, label=f"{label} (AUC = {auc_from_roc(fpr, tpr):.4f})",
                    lw=1.4, **style(i))
        ax.set_yscale("log")
        ax.set_xlabel(r"True positive rate ($\epsilon_S$)", fontsize=FONTSIZE)
        ax.set_ylabel(r"False positive rate ($\epsilon_B$)", fontsize=FONTSIZE)
        ax.set_xlim(0, 1)
        ax.legend(frameon=True, fontsize=FONTSIZE_LEGEND, loc="upper left")
        fig.savefig(pdf, bbox_inches="tight", format="pdf")
        plt.close(fig)

        # panel 3 -- significance improvement
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        for i, (label, fpr, tpr) in enumerate(entries):
            ax.plot(tpr, tpr / np.sqrt(fpr), label=label, lw=1.6, **style(i))
        ax.plot(rnd, rnd**0.5, "k--", lw=1, label="random")
        ax.set_xlabel(r"$\epsilon_S$", fontsize=FONTSIZE)
        ax.set_ylabel(r"$\epsilon_S/\sqrt{\epsilon_B}$", fontsize=FONTSIZE)
        ax.set_xlim(0, 1)
        ax.legend(frameon=False, fontsize=FONTSIZE_LEGEND)
        fig.savefig(pdf, bbox_inches="tight", format="pdf")
        plt.close(fig)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
