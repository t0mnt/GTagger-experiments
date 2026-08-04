"""The downloader and the experiment configs must agree about what is on disk.

These live in different files that nothing forces to move together: the collector
decides what to fetch and where to put it, while `config/{jctagging,toptagxl}.yaml`
decides which files the loader asks for. When they drift, nothing raises -- the loader
resolves fewer files than requested and trains on a silently smaller dataset (the
"Using N of M requested files" line is INFO-level).
"""

import importlib.util
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location("collect_data", REPO / "data" / "collect_data.py")
collect_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(collect_data)


def _cfg(name, tree="config"):
    return yaml.safe_load((REPO / tree / f"{name}.yaml").read_text())


def test_toptagxl_ranges_match_the_collector_exactly():
    """`_organize_toptagxl` sorts files by these ranges; the loader then requests them.

    Unlike JetClass the numbering here is global and continuous across splits, so the
    two sides must be identical, not merely compatible.
    """
    data = _cfg("toptagxl")["data"]
    for split, key in (("train", "train_files_range"), ("test", "test_files_range"), ("val", "val_files_range")):
        folder, (lo, hi) = collect_data.TOPTAGXL_EXPECTED[split]
        assert [lo, hi] == list(data[key]), (
            f"toptagxl {split}: collector expects {folder}/<class>_{lo:03d}..{hi - 1:03d} but "
            f"config/toptagxl.yaml {key} is {data[key]}. The organizer files by the collector's "
            f"range and the loader reads by the config's -- a mismatch means missing files."
        )


def test_jetclass_config_requests_no_more_than_the_collector_downloads():
    """JetClass configs deliberately request a SUBSET (e.g. one val file per class).

    So the contract is containment, not equality: whatever the config asks for must fit
    inside what the collector actually fetches, per class.
    """
    data = _cfg("jctagging")["data"]
    n_classes = 10  # JetClass (Pythia) has 10 jet classes, one .root series each
    for split, key in (("train", "train_files_range"), ("test", "test_files_range"), ("val", "val_files_range")):
        _folder, expected_total = collect_data.JETCLASS_EXPECTED[split]
        lo, hi = data[key]
        requested_per_class = hi - lo
        available_per_class = expected_total // n_classes
        assert requested_per_class <= available_per_class, (
            f"jctagging {split}: config requests {requested_per_class} files/class "
            f"({key}={data[key]}) but the collector only downloads {available_per_class} "
            f"({expected_total} total). The loader would resolve fewer files than requested "
            f"and train on less data, at INFO level only."
        )


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_jetclass_extract_subdir_is_consistent_with_the_expected_folder(split):
    """Extraction target vs completeness-check target.

    These differ ON PURPOSE -- the ten train part-tars are packed flat (so they extract
    INTO train_100M and merge), while the single val/test tars each carry their own
    directory (so they extract into Pythia and land in val_5M/test_20M). Verified
    against the real archives. The invariant that must hold either way: the folder the
    completeness check counts is at or below the folder the tars extract into.
    """
    subdir, _files = collect_data.JETCLASS[split]
    expected_folder, _n = collect_data.JETCLASS_EXPECTED[split]
    assert expected_folder.startswith(subdir), (
        f"jetclass {split}: tars extract into '{subdir}' but the completeness check counts "
        f"'{expected_folder}', which is not inside it -- one of the two is wrong."
    )


@pytest.mark.parametrize("task", ["toptagxl", "jctagging"])
def test_quick_configs_request_files_that_can_exist(task):
    """config_quick trims the file RANGE, but the numbering is not free to invent.

    A quick range outside what the collector downloads resolves to zero files and dies
    at miniweaver's bare `assert len(new_files) > 0` -- which is how
    config_quick/toptagxl.yaml shipped broken (train [622,623] under a 000-499 train
    split, plus a data_dir typo) while config/ was fine and this file only read config/.
    """
    quick, full = _cfg(task, "config_quick")["data"], _cfg(task)["data"]
    assert quick["data_dir"] == full["data_dir"], (
        f"config_quick/{task}.yaml data_dir {quick['data_dir']!r} != "
        f"config/{task}.yaml {full['data_dir']!r}; the collector only builds the latter."
    )
    for key in ("train_files_range", "test_files_range", "val_files_range"):
        qlo, qhi = quick[key]
        flo, fhi = full[key]
        assert flo <= qlo < qhi <= fhi, (
            f"config_quick/{task}.yaml {key}={quick[key]} falls outside the files the "
            f"collector actually provides for that split ({full[key]}); it would resolve "
            f"zero files."
        )
