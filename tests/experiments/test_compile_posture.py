"""Permanent guards for the compile twins. KEEP — these outlive the port instruments.

WHY THIS FILE EXISTS. The compile twins (`PairEmbed.compiled_dense`,
`compiled_attention`, `PlainGraphTrans.compiled_knn`, the weighted pair-BN, the scoped
`recompute_views`) are PERMANENT model code: a second implementation of the same maths,
selected by a flag. That is the duplicated-code hazard the CGENN dedup removed elsewhere,
and it is tolerable here only because gates pin the two implementations together.

Those gates were written in `test_nonequi_compile.py` — which `cleanup.md` schedules for
DELETION as a port instrument. Port instruments and regression guards ended up in the same
file, so running the wipe would strip the drift protection off code that STAYS. This file
carves out the part that must not be deleted:

  * `BACKWARD_VERIFIED` + `test_compile_true_is_backward_verified` — runs in the DEFAULT
    suite (no env gate) and is the thing standing between a one-character yaml edit and a
    campaign that dies at its first optimizer step. Every eval/no_grad gate can be green
    while the joint forward+backward graph refuses to lower; that is exactly what
    tag_PlainGraphTrans and tag_LorentzNetLGATrSlimGraphGPS did.
  * `test_train_mode_differential` — the guard that caught the worst bug of the program:
    the twins were exact in eval and wrong in TRAIN, because the pair BatchNorm saw the
    padded grid instead of the real pairs. Eval-only gates cannot see that class.

DELIBERATELY FIXTURE-FREE. `test_nonequi_compile`'s version loads recorded weights from
`tests/fixtures/nonequi_compile/`, which the same wipe deletes. Here both sides are built
from a fixed seed and the second is loaded from the first's `state_dict`, so the two
implementations are compared against each other rather than against a recording. That is
all this property needs — it is a twin-vs-eager equality, not a regression pin.
"""

import re

import pytest
import torch
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Models VERIFIED to survive a real `loss.backward()` under `compile: true`.
# Membership is NECESSARY but not SUFFICIENT for shipping true -- a model can be
# backward-verified and still ship false on a performance posture.
# Earned by measurement (test_nonequi_compile.test_compiled_backward while it exists;
# afterwards, by running a compiled training step by hand and recording the numbers here).
BACKWARD_VERIFIED = {
    "tag_cgenn",
    "tag_lorentznet",
    "tag_CGENNLGATrGraphTrans",
    "tag_CGENNLGATrGraphGPS",
    "tag_LorentzNetLGATrSlimGraphTrans",
    "tag_LorentzNetLGATrSlimGraphGPS",   # scoped recompute_views; 65/65 finite
    "tag_particlenet",
    "tag_transformer",
    "tag_ParT",                          # 217/217 finite
    "tag_ParticleNetParTGraphTrans",     # 85/85 finite
    "tag_ParticleNetParTGraphGPS",       # 64/64 finite
    "tag_PlainGraphTrans",               # static-k kNN twin; 71/71 finite
    "tag_PlainGraphGPS",                 # 54/54 finite (nonzero=1 is the documented
                                         # config_quick dead-ReLU artifact; eager agrees)
}

# Twins that reach the eager statistics through a WEIGHTED BatchNorm rather than by being
# handed the identical tensor, so train-mode agreement is TOL-class (float reassociation)
# instead of bit-exact. Everything else must be bit-zero.
TWIN_TOL_MODELS = {
    "tag_ParT",
    "tag_ParticleNetParTGraphTrans",
    "tag_ParticleNetParTGraphGPS",
}
TRAIN_TOL = 1e-10

COMPILE_MODELS = [
    "tag_ParT", "tag_particlenet", "tag_transformer",
    "tag_PlainGraphTrans", "tag_PlainGraphGPS",
    "tag_ParticleNetParTGraphTrans", "tag_ParticleNetParTGraphGPS",
]


def test_compile_true_is_backward_verified():
    """POSTURE: no production config may ship `compile: true` unless the model is in
    BACKWARD_VERIFIED. Cheap (reads yaml only) and ungated, so it runs on every suite.

    Scope: the WRAPPER knob. The nested `net.compile` belongs to third-party nets that
    compile themselves (tag_lgatr, tag_slim, the pelicans); this repo has no twins or
    fixtures for those, so they can never earn membership here. tag_lgatr's knob was
    measured out-of-band instead (dynamo 2 frames / 2 ok, backward 405/405 finite).
    """
    offenders = []
    for cfg in sorted((REPO / "config" / "model").glob("tag_*.yaml")):
        if re.search(r"^compile:\s*true\b", cfg.read_text(), flags=re.M):
            if cfg.stem not in BACKWARD_VERIFIED:
                offenders.append(cfg.stem)
    assert not offenders, (
        f"ships compile: true but not in BACKWARD_VERIFIED: {offenders}. Every compile "
        f"gate is eval/no_grad, so a model can pass all of them and still raise "
        f"InductorError at the first loss.backward(). Earn membership by running a "
        f"compiled training step and recording the gradient count above.")


def _build(model):
    import logging.handlers  # noqa: F401
    import hydra
    import experiments.logger
    from experiments.tagging.experiment import TopTaggingExperiment

    experiments.logger.LOGGER.disabled = True
    with hydra.initialize_config_dir(config_dir=str(REPO / "config_quick"), version_base=None):
        cfg = hydra.compose(config_name="toptagging",
                            overrides=["save=false", "training.batchsize=4",
                                       "data.dataset=mini", f"model={model}",
                                       "use_float64=true"])
    torch.manual_seed(0)
    exp = TopTaggingExperiment(cfg)
    exp._init(); exp.init_physics(); exp.init_model(); exp.init_data()
    exp._init_dataloader(); exp._init_loss()
    return exp


def _pre_compile(model, net):
    """Exactly the flags the wrappers' compile knobs set."""
    if model == "tag_ParT" and getattr(net, "pair_embed", None) is not None:
        net.pair_embed.sparse_eval = False
        net.pair_embed.compiled_dense = True
    if model == "tag_PlainGraphTrans":
        net.compiled_knn = True
    if model in ("tag_ParticleNetParTGraphTrans", "tag_ParticleNetParTGraphGPS"):
        for m in net.modules():
            if hasattr(m, "compiled_attention"):
                m.compiled_attention = True
            if hasattr(m, "compiled_dense"):
                m.compiled_dense = True


def _kill_dropout(model):
    """Zero every stochastic path, in all THREE forms it takes in this tree: nn.Dropout.p,
    nn.MultiheadAttention.dropout (a float attribute), and the local ParT port's own
    Attention.dropout (also a float attribute, different class, and shipping 0.1 on eight
    blocks in production). Missing the third reads as a ~5e-2 'divergence' that is purely
    RNG desynchronization."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
        if isinstance(getattr(m, "dropout", None), float):
            m.dropout = 0.0
        if isinstance(getattr(m, "drop_prob", None), float):
            m.drop_prob = 0.0


def _bn_buffers(model):
    return {n: b.detach().clone() for n, b in model.named_buffers()
            if n.endswith(("running_mean", "running_var"))}


@pytest.mark.parametrize("model", COMPILE_MODELS)
def test_train_mode_differential(model):
    """TRAIN: the compile knob must not change training numerics.

    BN running buffers are checked alongside outputs because they are persistent state: a
    compiled-trained checkpoint carries them into any later eager evaluation or finetune.
    """
    ref, out = None, {}
    for flags in (False, True):
        exp = _build(model)
        if ref is None:
            ref = {k: v.detach().clone() for k, v in exp.model.state_dict().items()}
        exp.model.load_state_dict(ref, strict=True)
        _kill_dropout(exp.model)
        if flags:
            _pre_compile(model, exp.model.net)
        torch.manual_seed(1)
        data = next(iter(exp.train_loader))
        exp.model.train()
        with torch.no_grad():
            torch.manual_seed(0)
            y = exp._get_ypred_and_label(data.clone())[0].detach().clone()
            for _ in range(2):                      # let the BN running buffers move
                torch.manual_seed(0)
                exp._get_ypred_and_label(data.clone())
        out[flags] = (y, _bn_buffers(exp.model))

    dy = (out[True][0] - out[False][0]).abs().max().item()
    b0, b1 = out[False][1], out[True][1]
    db = max((b1[k] - b0[k]).abs().max().item() for k in b0) if b0 else 0.0
    print(f"POSTURE-TRAIN[{model}] max|dy|={dy:.3e} max|d(BN)|={db:.3e}")

    if model in TWIN_TOL_MODELS:
        rel = dy / (1 + out[False][0].abs().max().item())
        assert rel < TRAIN_TOL and db < TRAIN_TOL, (
            f"{model}: the weighted-BN twin no longer reproduces the eager training "
            f"statistics (rel={rel:.3e}, max|d(BN)|={db:.3e}). The statistics weight must "
            f"be exactly the eager reference multiset, with the REAL batch dim in it.")
    else:
        assert dy == 0.0 and db == 0.0, (
            f"{model}: the compile knob changed training numerics "
            f"(max|dy|={dy:.3e}, max|d(BN)|={db:.3e}) -- it must only change kernels.")
