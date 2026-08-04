"""The results-table legend must frame the row the way `_log_table_row` emits it.

A legend that disagrees with the row does not fail loudly: LaTeX renders and every
column after the disagreement is silently mislabeled. That happened: JetClass emitted
two `table <split>:` rows of different widths (a hand-rolled one without FLOPs and the
base class's with), `aggregate_table.latest_row` keeps the LAST match, so the column
headed `kNN` carried the FLOPs value and the kNN string fell into an unnamed column.

These tests pin the two halves of the contract that failed, without hardcoding each
task's metric count (which is legitimately task-specific and would make the test a
copy of the legend it checks):
  1. exactly one emitter, so "which row wins" is never a question;
  2. every legend opens and closes with the fixed columns the emitter always writes.
"""

import re
from pathlib import Path

import pytest

from utils.aggregate_table import COLUMNS

REPO = Path(__file__).resolve().parents[2]

# experiments/tagging/experiment.py::_log_table_row always emits, in order:
#   model & frames & iters & params & <task metrics...> & time & flops & knn
LEADING = ["model", "frames", "iters", "params"]
TRAILING = ["time", "flops", "knn"]


def _cells(legend):
    return [c.strip().lower() for c in legend.split("&")]


@pytest.mark.parametrize("task", sorted(COLUMNS))
def test_legend_has_the_fixed_leading_columns(task):
    assert _cells(COLUMNS[task])[: len(LEADING)] == LEADING, (
        f"{task}: legend must start with {LEADING} -- that is what _log_table_row writes "
        f"before the task metrics."
    )


@pytest.mark.parametrize("task", sorted(COLUMNS))
def test_legend_has_the_fixed_trailing_columns(task):
    tail = _cells(COLUMNS[task])[-len(TRAILING) :]
    assert tail == TRAILING, (
        f"{task}: legend ends with {tail}, but _log_table_row always writes "
        f"{TRAILING} after the task metrics. A missing trailing column shifts every "
        f"label after it -- this is the JetClass FLOPs-under-the-kNN-header defect."
    )


@pytest.mark.parametrize(
    "path",
    [
        "experiments/tagging/experiment.py",
        "experiments/tagging/jetclassexperiment.py",
        "experiments/tagging/toptagxlexperiment.py",
    ],
)
def test_single_table_row_emitter(path):
    """Only the shared `_log_table_row` may emit a `table <split>:` line.

    A second emitter is not a harmless duplicate log line: the aggregator keeps the
    LAST match, so whichever emitter runs second silently wins.
    """
    source = (REPO / path).read_text()
    inline = re.findall(r'"table \{title\}:', source)
    expected = 1 if path.endswith("tagging/experiment.py") else 0
    assert len(inline) == expected, (
        f"{path} has {len(inline)} inline `table <split>:` emitters, expected {expected}. "
        f"Extra metric columns belong in the `metric_fmts` passed to _log_table_row, not "
        f"in a second LOGGER.info -- the aggregator keeps the last match."
    )
