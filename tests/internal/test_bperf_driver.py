"""beta-PERF driver gates: the parts that decide whether hours of GPU time are wasted.

The driver itself needs a GPU to produce a number, but nothing here does. Everything gated
below is a *pre-flight* property -- the row table, the window check, the fail-fast on a
failed sizing search, and the regex `--apply` uses to edit PRODUCTION yamls. Those are
exactly the parts whose failures are expensive: they surface after the matrix has run, or
not at all.
"""

import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location("bperf", REPO / "utils" / "bperf.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bperf = _load()


# --------------------------------------------------------------- the row table

def test_every_row_points_at_a_real_yaml_with_exactly_one_knob_line():
    """What `--apply` edits. A knob at the wrong indent is silently SKIPPED, not applied.

    `model.compile` is the wrapper knob (column 0), `model.net.compile` the net's own
    (indented under `net:`), and the driver picks the pattern from the dotted depth. If a
    config ever moves its knob, --apply prints a SKIP that is easy to miss in a long log and
    the production yaml keeps the old posture.
    """
    problems = []
    for name, _base, knob, yaml_path in bperf.MATRIX:
        path = REPO / yaml_path
        if not path.exists():
            problems.append(f"{name}: missing {yaml_path}")
            continue
        key = knob.split(".")[-1]
        indent = "" if knob.count(".") == 1 else "[ \t]+"
        hits = re.findall(rf"^({indent}{key}:[ \t]*)(true|false)\b", path.read_text(), flags=re.M)
        if len(hits) != 1:
            problems.append(f"{name}: {yaml_path} has {len(hits)} '{knob}' lines at the "
                            f"expected indent, want exactly 1")
    assert not problems, "\n".join(problems)


def test_row_names_are_unique_and_substring_filtering_is_documented_accurately():
    """`--models` is substring matching, so a name that is a prefix of another over-selects.

    todo.md tells the reader `--models CGENN` is the two hybrids only and that the three
    gp_impl rows need `--models tag_cgenn CGENN`. That is a claim about this table; pin it.
    """
    names = [r[0] for r in bperf.MATRIX]
    assert len(names) == len(set(names)), f"duplicate row names: {names}"

    def select(*filters):
        return [n for n in names if any(f in n for f in filters)]

    assert select("CGENN") == ["CGENNLGATrGraphTrans", "CGENNLGATrGraphGPS"]
    assert select("tag_cgenn", "CGENN") == [
        "tag_cgenn/einsum", "tag_cgenn/matmul", "tag_cgenn/sparse",
        "CGENNLGATrGraphTrans", "CGENNLGATrGraphGPS",
    ]


# ------------------------------------------------------------- the window check

@pytest.mark.parametrize("iters,window", [(110, (10, 100)), (1010, (100, 1000))])
def test_the_two_documented_windows_are_measurable(iters, window):
    bperf.check_window(window, iters)


@pytest.mark.parametrize("iters,window,why", [
    (110, (100, 300), "300 is not a power-of-ten mark"),
    (110, (10, 1000), "1000 exceeds --iters"),
    (1010, (1000, 100), "lo >= hi"),
    (1010, (100, 100), "lo == hi"),
    (110, (0, 100), "0 is not a mark"),
])
def test_unmeasurable_windows_are_rejected_before_any_run(iters, window, why):
    """Fail-fast: an unachievable window otherwise costs the WHOLE matrix before it shows."""
    with pytest.raises(SystemExit):
        bperf.check_window(window, iters)


# ------------------------------------------- a failed sizing search must be fatal

def test_a_failed_batchsize_search_stops_the_run(monkeypatch, capsys):
    """The bug that cost two runs: --find-batchsize failing was a WARNING, not an error.

    The driver then ran the whole matrix at the yaml value -- the unswept 512 -- which OOMs
    the CGENN rows. It happened once with a ModuleNotFoundError (the sys.path note at the top
    of bperf.py) and again with an IndexError in find_lr's probe. Both times the log said
    "batchsize search FAILED ... falling back to the config value" and kept going.

    There is no useful fallback: --find-batchsize is a request to SIZE the rows, and wanting
    the yaml value is spelled by omitting the flag.
    """
    def boom(*a, **kw):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(bperf, "find_batchsize", boom)
    monkeypatch.setattr(
        "sys.argv",
        ["bperf.py", "--models", "tag_cgenn/einsum", "--find-batchsize", "--iters", "110"],
    )
    # if the fallback ever comes back, run_once is reached and this fires instead
    monkeypatch.setattr(bperf, "run_once", lambda *a, **kw: pytest.fail(
        "the driver continued past a failed --find-batchsize and started timing runs"))

    with pytest.raises(SystemExit) as excinfo:
        bperf.main()
    message = str(excinfo.value)
    assert "find-batchsize failed" in message
    assert str(bperf.UNSWEPT_FALLBACK_BS) in message, (
        "the message must name the fallback batchsize it is refusing to run at")


def test_the_matrix_still_runs_when_sizing_is_not_requested(monkeypatch):
    """The flag is opt-in: without it the yaml value is a DELIBERATE choice, not a fallback."""
    monkeypatch.setattr(bperf, "find_batchsize", lambda *a, **kw: pytest.fail(
        "sized a row without --find-batchsize"))
    monkeypatch.setattr(bperf, "run_once", lambda *a, **kw: (12.5, None, None))
    monkeypatch.setattr("sys.argv", ["bperf.py", "--models", "tag_cgenn/einsum"])
    written = []
    monkeypatch.setattr(bperf, "REPO", REPO)  # report is appended to bperf_results.md
    monkeypatch.setattr(Path, "open", lambda self, *a, **kw: _Sink(written))
    bperf.main()
    assert written, "no report was produced"


class _Sink:
    def __init__(self, log):
        self.log = log

    def write(self, text):
        self.log.append(text)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
