"""BIT pins for the four GT-hybrid eager forwards -- the rewrite program's reference.

WHY THIS EXISTS. The CGENN improvement roadmap (docs/cgenn-compile.md, 2026-08-14) plans
BIT-class source rewrites -- layout unification, blade-contraction batching -- through the
shared `experiments/baselines/cgenn/` package and the GPS glue. The recorded BIT pins for
the hybrids were deleted by the post-merge wipe (da497a9; only tag_cgenn's gate was
restored), so at HEAD nothing machine-checks "this rewrite changed no eager bit" for the
models the program touches. This file is that check, in test_cgenn_compile's exact idiom.

RECORD BEFORE THE FIRST REWRITE, in the environment the suite runs in (fixtures are
bit-level and do NOT transfer across torch versions):

    CGENN_COMPILE=record python -m pytest tests/experiments/test_hybrid_bit_pin.py -q

With no fixtures recorded every test SKIPS, so the default suite is unaffected until the
rewrite period starts. TOL-class rewrites (blade batching changes contraction order) are
NOT judged here -- for those, re-record after review and gate at tolerance in their own
test, stating the class change in the commit.

Scope: eager, eval mode, default knobs (the shipped gp_impl; compile stays off -- BIT is
an eager property, the compiled path is TOL by charter). state_dict hash pins init RNG
order and parameter identity; the forward pins the arithmetic.
"""

import hashlib
import os
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "tests" / "fixtures" / "hybrid_bit"
RECORD = os.environ.get("CGENN_COMPILE") == "record"

torch.set_num_threads(1)  # run-context-independent arithmetic (test_cgenn_compile's lesson)

MODELS = [
    "tag_CGENNLGATrGraphTrans",
    "tag_CGENNLGATrGraphGPS",
    "tag_LorentzNetLGATrSlimGraphTrans",
    "tag_LorentzNetLGATrSlimGraphGPS",
]


def _build(model, float64):
    import logging.handlers  # noqa: F401
    import hydra
    import experiments.logger
    from experiments.tagging.experiment import TopTaggingExperiment

    experiments.logger.LOGGER.disabled = True
    overrides = ["save=false", "training.batchsize=4", "data.dataset=mini",
                 f"model={model}", f"use_float64={'true' if float64 else 'false'}"]
    with hydra.initialize_config_dir(config_dir=str(REPO / "config_quick"), version_base=None):
        cfg = hydra.compose(config_name="toptagging", overrides=overrides)
    torch.manual_seed(0)
    exp = TopTaggingExperiment(cfg)
    exp._init(); exp.init_physics(); exp.init_model(); exp.init_data()
    exp._init_dataloader(); exp._init_loss()
    exp.model.eval()
    return exp


def _sd_hash(model):
    h = hashlib.sha256()
    for k in sorted(model.state_dict()):
        v = model.state_dict()[k]
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("prec", ["fp32", "fp64"])
def test_bit_eager_vs_pin(model, prec):
    """torch.equal against the recording, no tolerance -- plus the state_dict hash, so a
    reordered init RNG draw fails as loudly as changed arithmetic."""
    path = FIX / f"{model}_{prec}.pt"
    exp = _build(model, float64=(prec == "fp64"))
    torch.manual_seed(1)
    data = next(iter(exp.train_loader))
    with torch.no_grad():
        y = exp._get_ypred_and_label(data.clone())[0].detach().clone()

    if RECORD:
        FIX.mkdir(parents=True, exist_ok=True)
        torch.save({"y": y, "sd_hash": _sd_hash(exp.model),
                    "torch": torch.__version__}, path)
        pytest.skip(f"recorded {path.name}")
    if not path.exists():
        pytest.skip("no hybrid BIT pins recorded (CGENN_COMPILE=record before the "
                    "first BIT-class rewrite; see module docstring)")

    ref = torch.load(path, weights_only=False)
    assert _sd_hash(exp.model) == ref["sd_hash"], (
        f"{model}/{prec}: state_dict changed vs the pin -- init RNG order, a renamed "
        f"module, or a new parameter/buffer. A BIT-class rewrite must not do any of that.")
    assert torch.equal(y, ref["y"]), (
        f"{model}/{prec}: eager forward is no longer bit-identical to the pin recorded on "
        f"torch {ref['torch']} (this env: {torch.__version__}). If the torch version "
        f"changed, re-record; if it did not, the rewrite changed arithmetic and is not "
        f"BIT-class.")
