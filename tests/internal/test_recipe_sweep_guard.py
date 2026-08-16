"""Gate for `run.py::_check_recipe_is_swept`.

KEEP while any recipe still carries `???`. The guard exists because that marker does NOT
behave as hydra's mandatory value: the per-model recipes inherit `tag_gts_and_friends_default`
-> `tag_default`, which DEFINES `batchsize: 512` and `lr: 1e-3`, and OmegaConf merge treats a
child `???` as "no value supplied", so the parent's survives. An unswept recipe therefore
trains at 512 / 1e-3 and produces a plausible-looking results row instead of an error.

25 recipes carry the marker today. Only the 8 `jc_*` ones warn about this in a comment; the
`top_*` and `xl_*` ones call the keys "REQUIRED". A comment was already the fix and it did not
propagate, hence a check.

FIRES     an unswept recipe raises SystemExit rather than training
CLI       a key supplied on the command line counts as filled
FILLED    a swept recipe (top_particlenet) passes
NOHYDRA   composing without hydra managing the run is silently skipped (find_lr, bperf, tests)
MARKER    the `???` inventory matches what the guard would catch, so neither drifts
"""

import os
from pathlib import Path

import hydra
import pytest
from hydra.core.hydra_config import HydraConfig

from run import _check_recipe_is_swept

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "config"


def _with_hydra(overrides):
    """Compose under a real HydraConfig, the way `hydra.main` would."""
    with hydra.initialize_config_dir(config_dir=str(CONFIG), version_base=None):
        cfg = hydra.compose(
            config_name="toptagging", overrides=overrides, return_hydra_config=True
        )
        HydraConfig.instance().set_config(cfg)
        try:
            yield
        finally:
            HydraConfig.instance().cfg = None


def _run(overrides):
    gen = _with_hydra(overrides)
    next(gen)
    try:
        _check_recipe_is_swept()
    finally:
        gen.close()


def test_unswept_recipe_is_refused():
    """FIRES: the whole point — 512/1e-3 in the results table is worse than a dead launch."""
    with pytest.raises(SystemExit) as exc:
        _run(["model=tag_PlainGraphGPS", "training=top_PlainGraphGPS", "save=false"])
    msg = str(exc.value)
    assert "top_PlainGraphGPS.yaml" in msg
    assert "batchsize" in msg and "lr" in msg


def test_command_line_values_count_as_filled():
    """CLI: overriding on the command line is a legitimate way to supply the value."""
    _run([
        "model=tag_PlainGraphGPS",
        "training=top_PlainGraphGPS",
        "training.batchsize=256",
        "training.lr=3e-4",
        "save=false",
    ])


def test_partial_override_still_refuses_the_rest():
    """CLI: filling one key must not excuse the other.

    Since the 2026-08-16 table-wide lr decision every top_<hybrid>.yaml ships
    lr: 1e-3, so the live unswept key on this recipe is batchsize -- the CLI
    supplies lr and the guard must still refuse for batchsize."""
    with pytest.raises(SystemExit) as exc:
        _run([
            "model=tag_PlainGraphGPS",
            "training=top_PlainGraphGPS",
            "training.lr=1e-3",
            "save=false",
        ])
    listed = str(exc.value).split("unswept keys: ")[1].split(".")[0]
    assert listed == "batchsize", listed  # not the key already supplied on the command line


def test_swept_recipe_passes():
    """FILLED: a real published recipe must launch."""
    _run(["model=tag_particlenet", "training=top_particlenet", "save=false"])


def test_no_hydra_context_is_silently_skipped():
    """NOHYDRA: find_lr, bperf and the tests compose directly and none of them trains."""
    HydraConfig.instance().cfg = None
    assert not HydraConfig.initialized()
    _check_recipe_is_swept()  # must not raise


def test_marker_inventory_matches_what_the_guard_catches():
    """MARKER: keeps the guard's parser and the actual recipe files from drifting apart.

    Anything the guard would catch must be a `top_/jc_/xl_` per-model recipe, and every
    recipe carrying a bare `???` must be one the guard catches — a marker it cannot see is
    a silent 512/1e-3 run waiting to happen.
    """
    caught, present = set(), set()
    for path in sorted((CONFIG / "training").glob("*.yaml")):
        lines = path.read_text().splitlines(keepends=True)
        keys = [
            key
            for line in lines
            if (key := line.split(":", 1)[0].strip())
            and not line.startswith(("#", " ", "\t"))
            and line.split(":", 1)[-1].strip() == "???"
        ]
        if keys:
            caught.add(path.name)
        if any(l.strip().endswith("???") and not l.strip().startswith("#") for l in lines):
            present.add(path.name)
    assert caught == present, f"markers the guard cannot see: {present - caught}"
    assert caught, "no recipe carries `???` any more — delete this guard and its gate"
    assert all(n.split("_")[0] in ("top", "jc", "xl") for n in caught), sorted(caught)
    print(f"guard covers {len(caught)} unswept recipes: {sorted(caught)}")


def test_find_lr_names_the_recipe_for_every_model():
    """find_lr's `FIND_LR ... -> <recipe>` pointer must resolve for every model.

    It is what a 25-model sweep is transcribed from (`grep FIND_LR`), so a dropped pointer
    is a value typed into the wrong file, or hunted for by hand. The derivation keys off the
    MODEL CONFIG name; it used to key off the net's CLASS name, which agrees for the 8
    hybrids (tag_PlainGraphGPS -> PlainGraphGPS -> top_PlainGraphGPS.yaml) and disagrees for
    all 8 baselines (tag_cgenn -> CGENN -> top_CGENN.yaml, which does not exist), silently
    printing no pointer at all -- including for tag_cgenn, whose recipe is one of the unswept
    25 and whose gp_impl posture is still open.

    Mirrors find_lr's lookup rather than importing it: the real one needs a live HydraConfig.
    """
    import yaml

    missing = []
    for model_cfg in sorted((CONFIG / "model").glob("tag_*.yaml")):
        stem = model_cfg.stem[len("tag_") :]
        net = (yaml.safe_load(model_cfg.read_text()) or {}).get("net") or {}
        cls = net.get("_target_", "").rsplit(".", 1)[-1]
        for prefix in ("top", "jc", "xl"):
            real = CONFIG / "training" / f"{prefix}_{stem}.yaml"
            if not real.is_file():
                continue  # no recipe for this (model, task) pair -- nothing to point at
            found = next(
                (c for c in (stem, cls) if (CONFIG / "training" / f"{prefix}_{c}.yaml").is_file()),
                None,
            )
            if found is None:
                missing.append(f"{model_cfg.name} @ {prefix}: {real.name} exists but is unreachable")
    assert not missing, "\n".join(missing)
