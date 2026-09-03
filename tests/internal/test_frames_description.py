"""The frames cell must separate variants that no other table column separates.

`aggregate_table` keys a row on `(exp_type, model, frames, kNN, exp_name)`. Everything a
framesnet config can change that is NOT one of those five is invisible to the key, so two
runs collapse into one row and the newer silently wins. `is_global` is the case that bit:
it swaps the per-particle frames for one event-averaged frame -- worth ~0.5% accuracy on
top tagging -- while leaving the class name, the parameter count and the kNN string
identical. `exp_name` keeps the rows apart, but only if the launch remembers to set it,
and the rendered cell still reads `LearnedPDFrames` for both.

These tests pin the annotation and the assumption that makes it safe: it fires only where
the shipped configs are non-global, so no already-recorded row changes shape.
"""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from lloca.framesnet.equi_frames import LearnedPDFrames, LearnedSO13Frames
from lloca.framesnet.nonequi_frames import IdentityFrames, RandomFrames

from experiments.tagging.experiment import TaggingExperiment

REPO = Path(__file__).resolve().parents[2]
LEARNED_YAMLS = sorted(
    p
    for d in ("config", "config_quick")
    for p in (REPO / d / "model" / "framesnet").glob("learned*.yaml")
)


def describe(framesnet):
    """`_frames_description` reads only `self.model.framesnet`, so stub the rest away."""
    exp = SimpleNamespace(model=SimpleNamespace(framesnet=framesnet))
    return TaggingExperiment._frames_description(exp)


def bare(cls, **attrs):
    """A framesnet instance without its `__init__` -- the real one wants a built network."""
    fn = object.__new__(cls)
    for k, v in attrs.items():
        setattr(fn, k, v)
    return fn


@pytest.mark.parametrize("cls", [LearnedPDFrames, LearnedSO13Frames])
def test_global_learned_frames_are_annotated(cls):
    local = describe(bare(cls, is_global=False))
    glob = describe(bare(cls, is_global=True))
    assert local == cls.__name__
    assert glob == f"{cls.__name__}(global)"
    assert local != glob, (
        "a global and a per-particle run of the same framesnet must not share a frames "
        "cell -- nothing else in the row distinguishes them, so aggregate_table would "
        "merge them and keep whichever log was written last."
    )


def test_random_frames_keep_their_transform_type_only():
    """RandomFrames is global BY CONSTRUCTION; annotating it would rewrite published rows."""
    fn = bare(RandomFrames, is_global=True, transform_type="rotation")
    assert describe(fn) == "RandomFrames(rotation)"


def test_identity_frames_are_unannotated():
    """IdentityFrames hardcodes is_global=True and has no transform_type -- the baseline
    row every task's table leads with, so its cell has to stay the bare class name."""
    assert describe(bare(IdentityFrames, is_global=True)) == "IdentityFrames"


@pytest.mark.parametrize(
    "path", LEARNED_YAMLS, ids=lambda p: f"{p.parent.parent.parent.name}/{p.name}"
)
def test_shipped_learned_configs_are_not_global(path):
    """The annotation is a deviation marker: it only leaves recorded rows untouched while
    every shipped learned framesnet is per-particle. If one flips to global, the rows it
    already produced were written unannotated and this assumption has to be revisited."""
    m = re.search(r"^is_global:\s*(\S+)", path.read_text(), re.M)
    assert m is not None, f"{path} has no is_global key"
    assert m.group(1) == "false", (
        f"{path} ships is_global: {m.group(1)}; rows recorded before this change carry an "
        f"unannotated frames cell and would no longer match."
    )
