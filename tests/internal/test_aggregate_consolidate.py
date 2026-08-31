"""Pins for aggregate_table's pooling guards (5cec76a) and its grouping key.

The guards were exercised synthetically in the session that wrote them but never
committed as tests — this file closes that gap (2026-08-16 audit). Row layout used
here: model & frames & iters & params & acc & rej & time(s) & flops & knn, i.e. the
invariant cells sit at 2/3/ncols-2 and train time at ncols-3, as _consolidate assumes.

The second half covers the grouping key end to end (through `main`), added when
`exp_name` joined it: the guards were working exactly as designed and the table was
still wrong, because the bucket they were guarding held two different experiments.
"""

import os
import sys
from unittest import mock

from utils.aggregate_table import _consolidate, main, run_expname

ROW_A = "m & f & 48k & 366k & 0.9410 & 1700 & 120.0s & 3.2G & deltaR"
ROW_B = "m & f & 48k & 366k & 0.9420 & 1750 & 121.0s & 3.2G & deltaR"

# a full-width toptagging row, for the end-to-end keying tests below: _consolidate
# reads invariants at 2/3/ncols-2 and augment_row rewrites ncols-3, so these need the
# real column count, not the abbreviated ROW_A layout.
FULL = ("PlainGraphGPS & IdentityFrames & 47320 & {params} & {acc} & 0.9858 & "
        "1500 & 380 & 55 & {t}s & {flops} & deltaR")


def _mkrun(root, exp, name, params=2326927, acc="0.9401", flops="9.796e+08", t=3600):
    d = os.path.join(root, "runs", exp, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "config.yaml"), "w") as f:
        f.write(f"exp_type: toptagging\nexp_name: {exp}\ntraining:\n  batchsize: 512\n")
    row = FULL.format(params=params, acc=acc, flops=flops, t=t)
    with open(os.path.join(d, "out_0.log"), "w") as f:
        f.write(f"noise\ntable test: {row} \\\\\n")
    return d


def _aggregate(runs_root, capsys):
    with mock.patch.object(sys, "argv", ["aggregate_table.py", "--runs", runs_root]):
        main()
    return capsys.readouterr().out


def test_pools_disagreeing_trials_with_mean_pm_std():
    mtime, row, dirs = _consolidate("k", [(1.0, ROW_A, "d1"), (2.0, ROW_B, "d2")])
    assert "[2 trials]" in row.split("&")[2]
    acc = row.split("&")[4]
    assert "\\pm" in acc and "0.9415" in acc  # mean of .9410/.9420, sample std
    assert dirs == ["d1", "d2"]


def test_refuses_on_disagreeing_invariants():
    other_params = ROW_B.replace("366k", "400k")
    mtime, row, dirs = _consolidate("k", [(1.0, ROW_A, "d1"), (2.0, other_params, "d2")])
    assert row == other_params and dirs == ["d2"]  # newest by mtime, unpooled
    assert "[2 trials]" not in row


def test_refuses_identical_metrics_as_pinned_seed_clones():
    only_time_differs = ROW_A.replace("120.0s", "125.0s")
    mtime, row, dirs = _consolidate("k", [(1.0, ROW_A, "d1"), (2.0, only_time_differs, "d2")])
    assert row == only_time_differs and dirs == ["d2"]
    assert "\\pm" not in row and "[2 trials]" not in row


def test_refuses_mix_with_in_run_aggregated_row():
    aggregated = ROW_A.replace("0.9410", "$0.9410 \\pm 0.0007$")
    mtime, row, dirs = _consolidate("k", [(1.0, aggregated, "d1"), (2.0, ROW_B, "d2")])
    assert row == ROW_B and dirs == ["d2"]  # newest-wins, no double counting


def test_newest_is_by_mtime_not_list_order():
    mtime, row, dirs = _consolidate(
        "k", [(2.0, ROW_B.replace("366k", "400k"), "newer"), (1.0, ROW_A, "older")]
    )
    assert dirs == ["newer"]  # invariant guard keeps the mtime-newest despite list order


# --- exp_name in the grouping key -------------------------------------------------
# Regression for the failure that motivated it: `exp_name=topt_L4` gave the ablation its
# own run DIRECTORY, but the key was (exp_type, model, frames, kNN), so the ablation and
# the campaign landed in one bucket -- the params guard refused to pool and printed the
# newest run under the plain model name. Fourteen table rows were wrong that way.


def test_expname_from_config_beats_parent_dir(tmp_path):
    d = _mkrun(str(tmp_path), "topt_L4", "PlainGraphGPS_4444")
    assert run_expname(d) == "topt_L4"


def test_expname_falls_back_to_parent_dir_without_config(tmp_path):
    d = _mkrun(str(tmp_path), "topt_L4", "PlainGraphGPS_4444")
    os.remove(os.path.join(d, "config.yaml"))
    assert run_expname(d) == "topt_L4"


def test_ablation_does_not_clobber_the_campaign_row(tmp_path, capsys):
    """The exact 2026-08-31 failure: a depth ablation kept its own dir but not its own row."""
    root = str(tmp_path)
    for name, acc, t in (("a_1", "0.9401", 3600), ("a_2", "0.9399", 3610), ("a_3", "0.9403", 3620)):
        _mkrun(root, "topt_local_debug", name, acc=acc, t=t)
    _mkrun(root, "topt_L4", "b_1", params=937615, acc="0.9400", flops="3.919e+08", t=3630)
    out = _aggregate(os.path.join(root, "runs"), capsys)

    assert "NOT pooling" not in out  # the params guard must not fire across exp_names
    assert "[3 trials]" in out       # campaign trials still pool with each other
    assert "PlainGraphGPS [topt_local_debug]" in out and "2326927" in out
    assert "PlainGraphGPS [topt_L4]" in out and "937615" in out


def test_separates_variants_that_differ_only_in_expname(tmp_path, capsys):
    """knn_k changes no keyed cell and no param count -- exp_name is the ONLY separator."""
    root = str(tmp_path)
    _mkrun(root, "topt_local_debug", "a_1", acc="0.9401", flops="9.796e+08")
    _mkrun(root, "topt_k4", "b_1", acc="0.9384", flops="3.897e+08", t=3630)
    out = _aggregate(os.path.join(root, "runs"), capsys)

    assert "9.796e+08" in out and "3.897e+08" in out  # both survive; neither wins a tiebreak
    assert "PlainGraphGPS [topt_k4]" in out


def test_single_campaign_root_renders_unannotated(tmp_path, capsys):
    """A one-root scan (the paper table) must look exactly as it did before this change."""
    root = str(tmp_path)
    for name, acc, t in (("a_1", "0.9401", 3600), ("a_2", "0.9399", 3610), ("a_3", "0.9403", 3620)):
        _mkrun(root, "topt_local_debug", name, acc=acc, t=t)
    out = _aggregate(os.path.join(root, "runs", "topt_local_debug"), capsys)

    assert "[topt_local_debug]" not in out  # nothing to disambiguate -> no annotation
    assert "PlainGraphGPS & IdentityFrames & 47320 [3 trials]" in out
