"""Pins for aggregate_table._consolidate's pooling guards (5cec76a).

The guards were exercised synthetically in the session that wrote them but never
committed as tests — this file closes that gap (2026-08-16 audit). Row layout used
here: model & frames & iters & params & acc & rej & time(s) & flops & knn, i.e. the
invariant cells sit at 2/3/ncols-2 and train time at ncols-3, as _consolidate assumes.
"""

from utils.aggregate_table import _consolidate

ROW_A = "m & f & 48k & 366k & 0.9410 & 1700 & 120.0s & 3.2G & deltaR"
ROW_B = "m & f & 48k & 366k & 0.9420 & 1750 & 121.0s & 3.2G & deltaR"


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
