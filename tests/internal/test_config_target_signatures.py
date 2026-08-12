"""Every key in a model config must be accepted by the class it instantiates.

Written after shipping the failure it catches. Six configs gained a `use_amp: false` key
on the reasoning that AMP-off should be stated where it matters most; four of the six
wrappers do not TAKE that parameter, so hydra raised

    TypeError: CGENNLGATrGraphGPSWrapper.__init__() got an unexpected keyword argument 'use_amp'

and the absence of the key turned out to be a requirement rather than an oversight.

Two things made that expensive to find. Composition does NOT catch it -- `hydra.compose`
and `OmegaConf.to_container(resolve=True)` both succeed, because the mismatch only surfaces
at INSTANTIATE. And the only instantiate-level gate over production configs,
`test_lgatr_migration_parity.test_production_manifest`, covers just the six lgatr-family
models, so `tag_cgenn` and `tag_lorentznet` carried the same broken key with nothing to
report it.

This is the cheap general version: a static signature check, no model construction, so it
runs over all 36 configs in well under a second and does not care how big the nets are.
Nested keys (`net:`, `framesnet:`) are hydra sub-configs with their own targets and are left
to hydra's own recursion; a target taking **kwargs is exempt because it accepts anything.
"""

import importlib
import inspect
import logging.handlers  # noqa: F401  (experiments.logger assumes it is imported)
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIG_DIRS = [REPO / "config" / "model", REPO / "config_quick" / "model"]
HYDRA_KEYS = {"_target_", "_partial_", "_recursive_", "_convert_", "_args_", "defaults"}

CONFIGS = sorted(
    (d / f.name for d in CONFIG_DIRS if d.is_dir() for f in sorted(d.glob("*.yaml"))),
    key=lambda p: (p.parts[-3], p.name),
)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: f"{p.parts[-3]}/{p.name}")
def test_every_key_is_accepted_by_its_target(path):
    cfg = yaml.safe_load(path.read_text()) or {}
    target = cfg.get("_target_")
    if target is None:
        pytest.skip("no _target_ (a defaults-only or group config)")

    module, name = target.rsplit(".", 1)
    cls = getattr(importlib.import_module(module), name)
    sig = inspect.signature(cls.__init__)
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        pytest.skip(f"{name} takes **kwargs, so any key is accepted")

    rejected = [
        key
        for key, value in cfg.items()
        if key not in HYDRA_KEYS
        and not isinstance(value, dict)  # nested sub-config: hydra recurses with its own target
        and key not in sig.parameters
    ]
    assert not rejected, (
        f"{path.parts[-3]}/{path.name} sets {rejected}, which "
        f"{name}.__init__ does not accept -- hydra will raise at instantiate time, and "
        f"composition will NOT catch it. Accepted: {sorted(sig.parameters)[:12]}..."
    )
