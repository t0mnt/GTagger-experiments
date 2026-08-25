#!/usr/bin/env python3
"""AUC-vs-FLOPs Pareto scatter (the 'better ^' figure).

    python pareto.py pareto.tsv -o pareto.pdf [--x params]

Input is a TAB-separated file with a header-less body of
    model <TAB> auc <TAB> flops <TAB> params <TAB> group
and '#' comment lines. Groups get distinct marker/colour so hybrids read apart
from baselines.
"""
import argparse
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

FONTSIZE = 13
STYLE = {"baseline": ("#7B68EE", "o"), "hybrid": ("#2E8B57", "s"), "plain": ("darkorange", "^")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("table")
    p.add_argument("-o", "--out", default="pareto.pdf")
    p.add_argument("--x", choices=["flops", "params"], default="flops")
    p.add_argument("--ymin", type=float, default=None)
    args = p.parse_args()

    rows = []
    for line in open(args.table):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, auc, flops, params, group = line.split("\t")
        rows.append((name, float(auc), float(flops), int(params), group))
    if not rows:
        raise SystemExit("no data rows -- fields must be TAB separated")

    xi = 2 if args.x == "flops" else 3
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    for g, (c, m) in STYLE.items():
        pts = [r for r in rows if r[4] == g]
        if pts:
            ax.scatter([r[xi] for r in pts], [r[1] for r in pts], c=c, marker=m, s=52,
                       label=g, zorder=3, edgecolors="none")
    for name, auc, flops, params, group in rows:
        ax.annotate(name, ((flops, params)[xi - 2], auc), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8.5)

    ax.set_xscale("log")
    ax.set_xlabel({"flops": "FLOPs", "params": "Parameters"}[args.x], fontsize=FONTSIZE)
    ax.set_ylabel("AUC", fontsize=FONTSIZE)
    if args.ymin is not None:
        ax.set_ylim(bottom=args.ymin)
    ax.grid(alpha=0.25, zorder=0)
    ax.legend(frameon=False, fontsize=11, loc="lower right")

    # the "better" arrow: up and to the left
    ax.annotate("better", xy=(0.055, 0.955), xytext=(0.20, 0.955), xycoords="axes fraction",
                textcoords="axes fraction", color="red", fontsize=11, va="center",
                arrowprops=dict(arrowstyle="->", color="red", lw=1.4))

    fig.savefig(args.out, bbox_inches="tight", format="pdf")
    print(f"wrote {args.out}  ({len(rows)} models)")


if __name__ == "__main__":
    main()
