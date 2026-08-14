"""Does every model actually TRAIN? The post-merge confirmation gate.

Run before the campaign:
    CGENN_COMPILE_GATES=1 python -m pytest tests/experiments/test_training_smoke.py -q -s

Every other gate in this repo measures a single forward. This one runs real optimizer
steps and asserts the two things that matter before spending GPU-months:

1. **It trains**: finite losses across N steps, no crash.
2. **Gradient actually flows**: a non-degenerate fraction of parameters receives a
   NONZERO gradient.

(2) is the assertion with teeth, and it is deliberately measured on the PRODUCTION
config, not `config_quick`. Measured 2026-08-09: `tag_PlainGraphGPS` gets nonzero
gradient on 230/230 parameters under `config/`, but **1/54** under `config_quick` --
the quick tree's `dim: 16` narrows the SAN head to 16->8->4 and at init all four
pre-activations at the last hidden layer are negative, so the ReLU emits exactly zero
and severs the backward. A benign small-width initialization accident in a test-only
config, but it means a quick-config training smoke silently proves nothing for that
model. `tag_PlainGraphTrans` is 71/71 on the same tree, so the head narrowing is the
discriminator, not the model family.

Also note: "parameters moved" is NOT evidence of learning. AdamW's decoupled weight
decay moves a parameter whose gradient is exactly zero, which is precisely how the
PlainGraphGPS case first looked healthy. Assert on gradients.

DELIBERATELY OUT OF SCOPE: EMA semantics and checkpoint round-trip. Those are broken by
pre-existing defects that live on `main` (see docs/audit-ledger.md), not by anything in
this branch, and asserting them here would turn a green-light gate into a red one for
reasons unrelated to the code under test.
"""

import os

import pytest
import torch

# Same one-liner test_cgenn_compile.py and test_lgatr_migration_parity.py already carry, and
# here it is load-bearing rather than hygienic: with CGENN_SMOKE_COMPILE=1, tag_cgenn dies
# with `Fatal Python error: Aborted` inside AOT's compiled backward at the default thread
# count, and PASSES with this line (measured both ways, and reproduced at 3fe5197 so it
# predates the sparse-GP work -- docs/audit-ledger.md). CPU inductor's threaded C++ kernels
# are what fall over; the arithmetic and gradients are fine, and the campaign's GPU inductor
# emits Triton, which does not take this path.
torch.set_num_threads(1)

MODELS = [
    "tag_ParT", "tag_particlenet", "tag_transformer", "tag_MIParT",
    "tag_PlainGraphTrans", "tag_PlainGraphGPS",
    "tag_ParticleNetParTGraphTrans", "tag_ParticleNetParTGraphGPS",
    "tag_cgenn", "tag_lorentznet",
    "tag_CGENNLGATrGraphTrans", "tag_CGENNLGATrGraphGPS",
    "tag_LorentzNetLGATrSlimGraphTrans", "tag_LorentzNetLGATrSlimGraphGPS",
    "tag_lgatr", "tag_slim",
]

RUN = os.environ.get("CGENN_COMPILE_GATES") == "1"

# Default OFF: building inductor kernels for every compiled model and running 8 optimizer
# steps each in ONE process exhausted resources and killed the interpreter (a harness
# flaw that looked like a model crash), which is why this gate forces eager.
#
# Set CGENN_SMOKE_COMPILE=1 to keep each model's SHIPPED knob instead. Use it with `-k` to
# select a few models at a time, and use it on GPU: every compile gate in this repo runs on
# CPU, so inductor's CUDA backend (Triton kernels) is a code path nothing has exercised.
#     CGENN_COMPILE_GATES=1 CGENN_SMOKE_COMPILE=1 \
#         python -m pytest tests/experiments/test_training_smoke.py -q -s -k "tag_ParT"
KEEP_COMPILE = os.environ.get("CGENN_SMOKE_COMPILE") == "1"

# Default 8: past ParT's warmup_steps=5, so the trimmer's post-warm-up path runs.
# CGENN_SMOKE_STEPS raises it for GPU soak runs -- the CGENN-GPS stride-guard crash
# (docs/cgenn-compile.md, 2026-08-14) was batch-SHAPE-dependent and fired on a later batch,
# a class an 8-step smoke can miss; with CGENN_SMOKE_COMPILE=1 on the card, e.g.
# CGENN_SMOKE_STEPS=100 turns this gate into the compiled-training soak per model.
STEPS = int(os.environ.get("CGENN_SMOKE_STEPS", 8))
MIN_GRAD_FRACTION = 0.5
# Structurally-unused parameters are legitimate: a scalar readout never consumes the
# vector half of the final lgatr linear, so those `weight_v` tensors are outside the
# autograd graph by construction (grad is None, not zero). The 50% bar accommodates
# that without accommodating a severed backward.


def _build_production(model):
    """Build from the PRODUCTION config tree -- see the module docstring for why."""
    import logging.handlers  # noqa: F401
    import pathlib
    import hydra
    import experiments.logger
    from experiments.tagging.experiment import TopTaggingExperiment

    experiments.logger.LOGGER.disabled = True
    root = pathlib.Path(__file__).resolve().parents[2]
    # Force EAGER unless CGENN_SMOKE_COMPILE=1 (see KEEP_COMPILE above). This gate answers
    # "does the MODEL train?"; whether the compile stack survives a training step is a
    # different question, already answered by test_nonequi_compile.test_compiled_backward.
    overrides = ["save=false", "training.batchsize=4", "data.dataset=mini",
                 f"model={model}"]
    # MIParT deliberately has NO compile knob (operator decision), so overriding it
    # raises; lgatr/slim keep theirs at net level. Everything else uses the wrapper knob.
    compile_off = {"tag_MIParT": None,
                   "tag_lgatr": "model.net.compile=false",
                   "tag_slim": "model.net.compile=false"}.get(model, "model.compile=false")
    if compile_off and not KEEP_COMPILE:
        overrides.append(compile_off)
    # EXTRA overrides, so the go/no-go can be pointed at the configuration the campaign
    # will actually run. The shipped configs all carry `framesnet: identity`, so without
    # this the smoke test cannot cover learned frames -- and learned frames are exactly
    # where the worst bug of this program lived (the PairEmbed twins computed pair
    # features grad-enabled where eager uses no_grad; under a LEARNED framesnet the
    # backward through sqrt(0) NaN'd the framesnet on step one).
    #     CGENN_SMOKE_OVERRIDES="model/framesnet=learnedpd" \
    #     CGENN_COMPILE_GATES=1 pytest tests/experiments/test_training_smoke.py -q -s \
    #         -k "tag_ParT or tag_particlenet or tag_transformer or Plain or ParticleNetParT"
    # Frames apply to the LLoCa-canonicalized family ONLY: the equivariant models
    # (CGENN/LorentzNet hybrids, tag_lgatr, tag_slim) are equivariant by construction and
    # their wrappers require IdentityFrames, so select with -k rather than running all 16.
    overrides += os.environ.get("CGENN_SMOKE_OVERRIDES", "").split()
    with hydra.initialize_config_dir(config_dir=str(root / "config"), version_base=None):
        cfg = hydra.compose(config_name="toptagging", overrides=overrides)
    torch.manual_seed(0)
    exp = TopTaggingExperiment(cfg)
    exp._init()
    exp.init_physics()
    exp.init_model()
    exp.init_data()
    exp._init_dataloader()
    exp._init_loss()
    return exp


@pytest.mark.skipif(not RUN, reason="training smoke: set CGENN_COMPILE_GATES=1")
@pytest.mark.parametrize("model", MODELS)
def test_model_trains(model):
    exp = _build_production(model)
    exp.model.train()
    torch.manual_seed(1)
    data = next(iter(exp.train_loader))
    opt = torch.optim.AdamW(exp.model.parameters(), lr=1e-4)

    import gc
    losses, grad_fraction = [], 0.0
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        y_pred, label = exp._get_ypred_and_label(data.clone())[:2]
        loss = exp.loss(y_pred, label)
        loss.backward()
        if step == 0:
            params = list(exp.model.parameters())
            live = sum(1 for p in params
                       if p.grad is not None and p.grad.abs().max() > 0)
            grad_fraction = live / max(len(params), 1)
        opt.step()
        losses.append(loss.item())

    print(f"GATE-TRAINSMOKE[{model}] loss {losses[0]:.4f} -> {losses[-1]:.4f} "
          f"| nonzero-grad params {grad_fraction:.0%}")

    assert all(torch.isfinite(torch.tensor(losses))), (
        f"{model}: non-finite loss during training: {losses}")
    del opt, exp
    gc.collect()
    assert grad_fraction >= MIN_GRAD_FRACTION, (
        f"{model}: only {grad_fraction:.0%} of parameters receive a nonzero gradient "
        f"-- the backward is severed somewhere (a dead activation, a stray no_grad, or "
        f"a detached path). NOTE: parameter MOVEMENT would not have caught this, since "
        f"AdamW's weight decay moves zero-gradient parameters.")
