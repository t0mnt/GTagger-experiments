"""Stage-4 compile gates: the non-equivariant family (docs/cgenn-compile.md).

Covers the padded-dense rows: ParT (the in-repo lloca port), ParticleNet, the baseline
transformer, the Plain hybrids and the ParticleNet-ParT hybrids (the latter pair re-runs
kNN per block from UPDATED coordinates — dynamic graph recomputation — but via dense
top-k, which traces; the gates adjudicate). MIParT is BIT/hash-pinned only (regression
pin); compile support for it was not pursued (operator decision), so it is excluded from
the compile-gate parametrization.

Fixtures (recorded at pre-gate HEAD, record-before-edit discipline):
    NONEQUI_COMPILE=record python -m pytest tests/experiments/test_nonequi_compile.py -q

Gates mirror the strict Stage-1 bars: BIT (torch.equal fp32+fp64), content hashes, and
env-gated (CGENN_COMPILE_GATES=1) TOL/DET/BREAKS (0 on a cold build, except the
documented data-dependent class below)/RECOMP (no growth across a production-regime
sweep; every sweep shape keeps max padded length above every quick-tree kNN k, max 7).

Documented break class (the only non-zero bar): the GraphGPS pair's masked BatchNorm
normalizes over the REAL nodes only (``out[mask_bool] = norm(h[mask_bool])``,
plaingraphgps.MaskedNorm; 'batch' is the GraphGPS-official norm). Boolean advanced
indexing lowers to aten.nonzero, whose OUTPUT SHAPE depends on tensor data — a
data-dependent-by-design break, same class as the Stage-3 CGENN-hybrid kNN edge builds
(hoisted eager there; normalized-by-design here), not a fixable tracing artifact. The
bar pins the exact break-event count and requires every break reason to be that class,
so any new break of any other kind still fails the gate.
"""

import json
import os
import re

import pytest
import torch

from tests.experiments.test_cgenn_compile import (
    REPO,
    _batch_fields,
    _content_hash,
    _fixed_batch,
    _forward,
    _rebuild,
)

FIX = REPO / "tests" / "fixtures" / "nonequi_compile"
RECORD = os.environ.get("NONEQUI_COMPILE") == "record"
RUN_COMPILE_GATES = os.environ.get("CGENN_COMPILE_GATES") == "1"

MODELS = [
    "tag_ParT",
    "tag_particlenet",
    "tag_MIParT",
    "tag_transformer",
    "tag_PlainGraphTrans",
    "tag_PlainGraphGPS",
    "tag_ParticleNetParTGraphTrans",
    "tag_ParticleNetParTGraphGPS",
]

# compile-gate scope: everything except MIParT (BIT/hash regression pin only -- no
# compile support, operator decision; its wrapper exposes no compile knob either)
COMPILE_MODELS = [m for m in MODELS if m != "tag_MIParT"]

# documented data-dependent break class (module docstring): masked BatchNorm over real
# nodes in the GraphGPS pair. Event counts are pinned exactly (deterministic on the
# fixed fixture batch); every break reason must be this class -- any other break fails.
# The counts are a property of the torch version too (measured on torch 2.13): a torch
# upgrade that changes dynamo's break accounting will fail this gate ON PURPOSE -- the
# upgrade re-pins these numbers consciously, with the reason-class assert as the guard
# that only the documented class is present in the new census.
BREAK_BARS = {
    "tag_PlainGraphGPS": 11,
    "tag_ParticleNetParTGraphGPS": 7,
}
BREAK_CLASS = "Dynamic shape operator"


def _build(model, float64, extra_overrides=()):
    import logging.handlers  # noqa: F401
    import hydra
    import experiments.logger
    from experiments.tagging.experiment import TopTaggingExperiment

    experiments.logger.LOGGER.disabled = True
    overrides = ["save=false", "training.batchsize=4", "data.dataset=mini",
                 f"model={model}", f"use_float64={'true' if float64 else 'false'}",
                 *extra_overrides]
    with hydra.initialize_config_dir(config_dir=str(REPO / "config_quick"), version_base=None):
        cfg = hydra.compose(config_name="toptagging", overrides=overrides)
    torch.manual_seed(0)
    exp = TopTaggingExperiment(cfg)
    exp._init()
    exp.init_physics()
    exp.init_model()
    exp.init_data()
    exp._init_dataloader()
    exp._init_loss()
    exp.model.eval()
    return exp


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("prec", ["fp32", "fp64"])
def test_bit_eager_vs_fixtures(model, prec):
    """BIT: eager outputs bit-identical to the recording.

    SCOPE -- read this before treating a green BIT as proof of port fidelity. The
    fixtures were recorded at pre-GATE HEAD, which for `tag_ParT` was already AFTER the
    yaml switched `net._target_` from `lloca.backbone.particletransformer` to the in-repo
    port. So this gate pins the port against ITSELF: it proves the compile work did not
    change ParT, and proves nothing about whether the port matches the library it was
    copied from. Same reasoning applies to any future in-repo port recorded this way.

    That separate question was answered directly, out-of-band (2026-08-10), by building
    tag_ParT from the merge-base tree (lloca's ParticleTransformer) and from HEAD (the
    port), assigning both identical parameters from a seeded generator, and running the
    same batch: `max|dy| = 0.000e+00`, identical parameter-name set, 2,141,005 params
    each. The port reproduces lloca's ParT exactly on the shipped configuration.

    It is exact because every divergence the port carries is inert there: `use_amp:
    false` neutralizes `autocast('cuda')` -> `autocast(x.device.type)`; `for_inference` is
    unset so the softmax/sigmoid branch never fires; and `framesnet: identity` means the
    frame-trimming fix (the port trims frames alongside particles, where lloca prepares
    them BEFORE the trimmer permutes/truncates) has nothing to realign. Under LEARNED
    frames with `trim: true` the port and lloca genuinely differ -- deliberately, with the
    port being the correct one -- so an ablation that turns frames on is NOT covered by
    the equality above.
    """
    path = FIX / f"{model}_{prec}.pt"
    exp = _build(model, float64=(prec == "fp64"))
    if RECORD:
        data = _fixed_batch(exp)
        pack = {"batch": _batch_fields(data), "y": _forward(exp, data),
                "sd": exp.model.state_dict()}
        FIX.mkdir(parents=True, exist_ok=True)
        torch.save(pack, path)
        return
    if not path.exists():
        pytest.skip("no nonequi_compile fixtures recorded")
    ref = torch.load(path, weights_only=False)
    exp.model.load_state_dict(ref["sd"], strict=True)
    y = _forward(exp, _rebuild(ref["batch"]))
    assert torch.equal(y, ref["y"]), (
        f"{model}/{prec}: eager output changed vs the recorded fixture "
        f"(max|diff|={(y - ref['y']).abs().max().item():.3e})")


def test_fixture_content_hashes():
    hash_file = FIX / "content_hashes.json"
    if RECORD:
        hashes = {f.name: _content_hash(torch.load(f, weights_only=False))
                  for f in sorted(FIX.glob("*.pt"))}
        hash_file.write_text(json.dumps(hashes, indent=1, sort_keys=True))
        return
    if not hash_file.exists():
        pytest.skip("no fixtures")
    stored = json.loads(hash_file.read_text())
    for fname, expected in sorted(stored.items()):
        live = _content_hash(torch.load(FIX / fname, weights_only=False))
        assert live == expected, f"{fname}: fixture content changed"




# per-model pre-compile adjustments mirroring the wrappers' compile knobs: ParT's default
# sparse pair path gathers real pairs via nonzero (data-dependent by design); compiled
# ParT runs the dense twin (same function, 2.2e-15 -- see ParTWrapper). The PN-ParT pair
# routes its identity-frames nn.MHA calls through the sdpa_plain_attention twin (the
# bool-kpm + float-bias warn in nn.MHA's preamble is a dynamo-skipped fn -> one break
# per block; see particlenettransformer.sdpa_plain_attention).
def _pre_compile(model, net):
    if model in ("tag_ParT",) and getattr(net, "pair_embed", None) is not None:
        net.pair_embed.sparse_eval = False
        net.pair_embed.compiled_dense = True
    if model == "tag_PlainGraphTrans":
        net.compiled_knn = True  # static-k kNN twin (symbolic k breaks inductor's stride order)
    if model in ("tag_ParticleNetParTGraphTrans", "tag_ParticleNetParTGraphGPS"):
        for m in net.modules():
            if hasattr(m, "compiled_attention"):
                m.compiled_attention = True
            if hasattr(m, "compiled_dense"):
                m.compiled_dense = True  # PairEmbed all-pairs twin (tril_indices pins seq_len)


# Models whose twin reaches the eager statistics through a WEIGHTED BatchNorm rather
# than by feeding the identical tensor, so train-mode agreement is TOL-class (float
# reassociation) instead of bit-exact. Everything else must be bit-zero.
#
# History worth keeping: these three used to diverge by 6.5e-01 / 1.5e-02 / 4.4e-04 with
# BN running buffers off by 15.0 / 1.1 / 1.1, because the twin fed BN the full padded
# grid where eager feeds the lower-triangular real pairs. Fixed by weighting the
# statistics (particletransformer._weighted_batchnorm1d); they now sit at <= 3.2e-15.
TWIN_TOL_MODELS = {
    "tag_ParT",
    "tag_ParticleNetParTGraphTrans",
    "tag_ParticleNetParTGraphGPS",
}
TRAIN_TOL = 1e-10


# Models VERIFIED to survive a real `loss.backward()` under `compile: true`.
# This list is the gate for the whole tree, not just this file's family: every
# production config that ships `compile: true` must appear here (asserted below), and
# membership is earned by test_compiled_backward actually running the backward.
#
# Membership is NECESSARY but not SUFFICIENT for shipping true: a model can be
# backward-verified and still ship false on a performance posture (see the GPS pair,
# whose backward is verified here but whose graphs split at the masked BatchNorm).
#
# WHY THIS EXISTS: `dynamo.explain`, TOL, DET, BIT and the train-mode differential are
# all no_grad/eval measurements, so they compile the INFERENCE graph. With autograd on,
# AOTAutograd emits the joint fwd+bwd graph and inductor can fail to lower it -- which is
# exactly what tag_PlainGraphTrans and tag_LorentzNetLGATrSlimGraphGPS did (InductorError,
# "cannot determine truth value of Relational", from symbolic shapes under dynamic=True).
# Both shipped `compile: true` and would have died at the first training step.
BACKWARD_VERIFIED = {
    "tag_cgenn",
    "tag_lorentznet",
    "tag_CGENNLGATrGraphTrans",
    "tag_CGENNLGATrGraphGPS",
    "tag_LorentzNetLGATrSlimGraphTrans",
    "tag_particlenet",
    "tag_transformer",
    # added once the weighted pair-BN made the twins train-faithful; measured
    # 217/217, 85/85, 64/64 parameters with nonzero finite gradients under compile
    "tag_ParT",
    "tag_ParticleNetParTGraphTrans",
    "tag_ParticleNetParTGraphGPS",
    # added once the static-k kNN twin removed the InductorError that used to kill this
    # model at its first loss.backward(): 71/71 nonzero finite grads under compile
    "tag_PlainGraphTrans",
}


def test_compile_true_is_backward_verified():
    """POSTURE: no production config may ship `compile: true` unless the model is in
    BACKWARD_VERIFIED. Cheap (reads yaml), runs in the default suite; the expensive
    half that earns membership is test_compiled_backward below.

    SCOPE, and how the rest is covered. This checks the WRAPPER knob only. The nested
    `net.compile` belongs to third-party nets that compile themselves (tag_lgatr,
    tag_slim, the pelicans), which this file has no fixtures or twins for -- so they can
    never earn BACKWARD_VERIFIED membership here, and a naive widening of the regex would
    just fail the gate forever.

    They are not unchecked, though. `tag_slim` and the pelicans carry `net.compile: true`
    from `main`, i.e. they predate this branch. `tag_lgatr` does NOT -- its knob is new
    here (operator-adopted, matching upstream tagging-guide practice), which makes it
    exactly the round-4 shape: a compile default whose backward nothing exercised, since
    test_training_smoke forces `net.compile=false` for lgatr/slim by default. So it was
    measured directly, from the shipped config with no overrides at all: dynamo compiled
    2 frames (total 2, ok 2), inductor produced 4 entries, and the backward gave 405/405
    finite gradients. `CGENN_SMOKE_COMPILE=1 pytest tests/experiments/test_training_smoke.py
    -k "tag_lgatr or tag_slim"` re-runs that as 8 real optimizer steps (100% / 98%
    nonzero-grad) and is the reproducible form.
    """
    offenders = []
    for cfg in sorted((REPO / "config" / "model").glob("tag_*.yaml")):
        text = cfg.read_text()
        # only the wrapper-level knob -- see the docstring for how net.compile is covered
        if re.search(r"^compile:\s*true\b", text, flags=re.M):
            if cfg.stem not in BACKWARD_VERIFIED:
                offenders.append(cfg.stem)
    assert not offenders, (
        f"ships compile: true but not in BACKWARD_VERIFIED: {offenders}. Every "
        f"compile gate is eval/no_grad, so a model can pass all of them and still "
        f"raise InductorError at the first loss.backward(); earn membership by "
        f"running test_compiled_backward (CGENN_COMPILE_GATES=1) first.")


@pytest.mark.skipif(not RUN_COMPILE_GATES, reason="compile smoke gates: set CGENN_COMPILE_GATES=1")
@pytest.mark.parametrize("model", sorted(BACKWARD_VERIFIED))
def test_compiled_backward(model):
    """BACKWARD: the compiled net must survive a real training step, not just a forward.

    This is the gate the whole program was missing: `_forward` is `no_grad`-wrapped
    everywhere, so nothing ever built a joint forward+backward graph.
    """
    exp = _build(model, float64=True, extra_overrides=("model.compile=true",))
    exp.model.train()
    data = _fixed_batch(exp)
    y, label = exp._get_ypred_and_label(data.clone())[:2]
    exp.loss(y, label).backward()
    grads = [p.grad for p in exp.model.parameters() if p.grad is not None]
    assert grads, f"BACKWARD[{model}]: compiled step produced no gradients at all"
    assert all(torch.isfinite(g).all() for g in grads), (
        f"BACKWARD[{model}]: non-finite gradients from the compiled step")
    print(f"GATE-BACKWARD[{model}] compiled training step OK ({len(grads)} grads, finite)")


def _kill_dropout(model):
    """Zero every stochastic path, in all THREE forms it takes in this tree.

    A train-mode differential is meaningless unless dropout is fully off, and "fully" is
    the tricky part -- probability lives in three different places here:
      1. `nn.Dropout` modules (`.p`),
      2. `nn.MultiheadAttention.dropout` -- a float ATTRIBUTE, not a module,
      3. the local ParT port's own `Attention.dropout` -- also a float attribute, and the
         one that is easiest to miss because it looks like (2) but is a different class.
    Production ParT ships 0.1 on eight `net.blocks.*.attn`, so missing (3) makes eager and
    compiled draw different masks and reports a ~5e-2 relative "divergence" that is purely
    RNG desynchronization. Measured: tag_ParT train-mode compiled-vs-eager reads 4.7e-02
    with (3) missed and 8.8e-16 with it zeroed. `DropPath.drop_prob` is included for the
    same reason, though it currently ships at 0.

    The flags-only comparison below happens to be insensitive to (3) -- the twin changes
    PairEmbed only, so both sides consume the RNG identically -- but anything that adds
    real dynamo to the comparison is NOT, which is exactly when someone would chase a
    phantom bug.
    """
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
        if isinstance(m, torch.nn.MultiheadAttention):
            m.dropout = 0.0
        # float-attribute dropouts on custom classes (ParT's Attention, DropPath)
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

    The TOL/DET/BIT gates all run under ``.eval()`` and ``no_grad`` (see _forward), so
    they constrain inference only. This gate closes that blind spot: it applies exactly
    the flags the wrapper's compile knob applies (no dynamo needed -- the divergence is
    flag-induced, and dynamo itself is numerics-preserving) and compares TRAIN-mode
    outputs and BatchNorm running buffers against the eager build.

    Most models must be bit-identical. The three in TWIN_TOL_MODELS reach the eager
    statistics through a WEIGHTED BatchNorm rather than by being handed the identical
    tensor, so they are held to TRAIN_TOL instead of to zero -- what is left is float
    reassociation, not a different statistic. BN running buffers are checked alongside the
    outputs because they are persistent state: a compiled-trained checkpoint carries them
    into any later eager evaluation, export or finetune.
    """
    ref = torch.load(FIX / f"{model}_fp64.pt", weights_only=False)
    out = {}
    for flags in (False, True):
        exp = _build(model, float64=True)
        exp.model.load_state_dict(ref["sd"], strict=True)
        _kill_dropout(exp.model)
        if flags:
            _pre_compile(model, exp.model.net)
        data = _fixed_batch(exp)
        exp.model.train()
        torch.manual_seed(0)
        with torch.no_grad():
            y = exp._get_ypred_and_label(data.clone())[0].detach().clone()
            for _ in range(2):  # let the BN running buffers move
                torch.manual_seed(0)
                exp._get_ypred_and_label(data.clone())
        out[flags] = (y, _bn_buffers(exp.model))
    dy = (out[True][0] - out[False][0]).abs().max().item()
    b_eager, b_flags = out[False][1], out[True][1]
    db = max((b_flags[k] - b_eager[k]).abs().max().item() for k in b_eager) if b_eager else 0.0
    print(f"GATE-TRAIN[{model}] train-mode max|dy|={dy:.3e} max|d(BN buffers)|={db:.3e}")

    if model in TWIN_TOL_MODELS:
        scale = 1 + abs(out[False][0]).max().item()
        rel = dy / scale
        assert rel < TRAIN_TOL and db < TRAIN_TOL, (
            f"TRAIN[{model}]: weighted-BN twin no longer reproduces the eager training "
            f"statistics (rel={rel:.3e}, max|d(BN)|={db:.3e}); the statistics weight must "
            f"be exactly the eager reference multiset -- tril of real pairs for ParT, tril "
            f"of all positions for the PN-ParT pair, with the REAL batch dim in the weight "
            f"(a broadcast-only weight undercounts w.sum() by bsz).")
    else:
        assert dy == 0.0 and db == 0.0, (
            f"TRAIN[{model}]: compile knob changed training numerics "
            f"(max|dy|={dy:.3e}, max|d(BN buffers)|={db:.3e}); the eval-mode TOL gate "
            f"cannot see this")


@pytest.mark.skipif(not RUN_COMPILE_GATES, reason="compile smoke gates: set CGENN_COMPILE_GATES=1")
@pytest.mark.parametrize("model", COMPILE_MODELS)
def test_tol_det_compiled_vs_eager(model):
    """TOL: compiled net vs eager <= 1e-10 rel (fp64 CPU). DET: compiled twice, torch.equal."""
    ref = torch.load(FIX / f"{model}_fp64.pt", weights_only=False)
    exp = _build(model, float64=True)
    exp.model.load_state_dict(ref["sd"], strict=True)
    y_eager = _forward(exp, _rebuild(ref["batch"]))
    _pre_compile(model, exp.model.net)
    exp.model.net = torch.compile(exp.model.net, dynamic=True)
    y1 = _forward(exp, _rebuild(ref["batch"]))
    y2 = _forward(exp, _rebuild(ref["batch"]))
    rel = (y1 - y_eager).abs().max() / (1 + y_eager.abs().max())
    print(f"GATE-TOL[{model}] compiled-vs-eager rel={rel:.3e}")
    assert rel < 1e-10, f"TOL[{model}]: {rel:.3e} >= 1e-10"
    assert torch.equal(y1, y2), f"DET[{model}]: compiled forward not deterministic"


@pytest.mark.skipif(not RUN_COMPILE_GATES, reason="compile smoke gates: set CGENN_COMPILE_GATES=1")
@pytest.mark.parametrize("model", COMPILE_MODELS)
def test_breaks_and_recomp(model):
    """BREAKS: 0 graph breaks over a COLD net (documented-class bar for the GPS pair).
    RECOMP: no growth across the sweep."""
    import torch._dynamo as dynamo
    ref = torch.load(FIX / f"{model}_fp64.pt", weights_only=False)
    exp = _build(model, float64=True)
    exp.model.load_state_dict(ref["sd"], strict=True)

    data = _rebuild(ref["batch"])
    captured = {}
    orig_forward = exp.model.net.forward

    def spy(*a, **k):
        captured["args"], captured["kwargs"] = a, k
        return orig_forward(*a, **k)

    exp.model.net.forward = spy
    _forward(exp, data)
    exp.model.net.forward = orig_forward
    exp_cold = _build(model, float64=True)
    exp_cold.model.load_state_dict(ref["sd"], strict=True)
    _pre_compile(model, exp_cold.model.net)
    explanation = dynamo.explain(exp_cold.model.net)(*captured["args"], **captured["kwargs"])
    report = str(explanation)
    report = re.sub(r"0x[0-9a-fA-F]+", "0x...", report)
    report = re.sub(r"(___check_(?:type|obj)_id\([^,]+, )\d+\)", r"\1...)", report)
    # dynamo numbers its per-frame globals dicts by how many frames it has seen in
    # the PROCESS, so this id shifts with test ORDER and churned ~150 noise lines per
    # gated run -- which would bury a real change. Normalized; no assertion reads it.
    report = re.sub(r"__builtins_dict___\d+", "__builtins_dict___N", report)
    # dynamo's per-phase compile TIMINGS: a name followed by one or MORE
    # comma-separated floats. The single-float form was normalized from the start;
    # split-graph models compile in several attempts and emit the multi-float form,
    # which churned every run until this was widened (same signal-burying class as
    # the globals-dict id below).
    report = re.sub(r"^([\w.]+)(?:, \d+\.\d+)+$", r"\1, ...", report, flags=re.M)
    (FIX / f"dynamo_explain_{model}.txt").write_text(report)
    print(f"GATE-BREAKS[{model}] graph_break_count =", explanation.graph_break_count)
    # anti-vacuous: something must actually have been traced -- a silently disabled
    # dynamo (TORCHDYNAMO_DISABLE=1 or a fallback-suppressing regression) would
    # otherwise pass every 0-bar below with zero graphs
    assert explanation.graph_count >= 1, f"BREAKS[{model}]: dynamo traced nothing"
    bar = BREAK_BARS.get(model, 0)
    assert explanation.graph_break_count == bar, (
        f"BREAKS[{model}]: {explanation.graph_break_count} != pinned {bar}\n{report[:2000]}")
    if bar:
        # every (deduped) break reason must be the documented data-dependent class
        reasons = re.findall(r"^    Reason: (.+)$", report, flags=re.M)
        assert reasons and all(r == BREAK_CLASS for r in reasons), (
            f"BREAKS[{model}]: non-documented break class present: {reasons}")

    dynamo.reset()
    from torch._dynamo.utils import counters as dyn_counters
    dyn_counters.clear()
    exp2 = _build(model, float64=True)
    exp2.model.load_state_dict(ref["sd"], strict=True)
    _pre_compile(model, exp2.model.net)
    exp2.model.net = torch.compile(exp2.model.net, dynamic=True)
    ptr = ref["batch"]["ptr"]
    graph_counts = []
    for keep in [[1, 3, 20, 40], [2, 5, 30, 25], [6, 9, 25, 14]]:
        d2 = _rebuild(ref["batch"])
        rows = []
        for j, n in enumerate(keep):
            n = min(n, int(ptr[j + 1] - ptr[j]))
            rows.extend(range(int(ptr[j]), int(ptr[j]) + n))
        idx = torch.tensor(rows, dtype=torch.long)
        for key in ("x", "scalars", "batch"):
            d2[key] = d2[key].index_select(0, idx)
        counts = [min(k, int(ptr[j + 1] - ptr[j])) for j, k in enumerate(keep)]
        d2.ptr = torch.tensor([0] + list(torch.tensor(counts).cumsum(0)), dtype=torch.long)
        d2.batch = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts))
        _forward(exp2, d2)
        graph_counts.append(
            sum(v for k, v in dyn_counters["stats"].items() if k == "unique_graphs"))
    print(f"GATE-RECOMP[{model}] unique_graphs per sweep shape = {graph_counts}")
    # strict for every model, including the documented-break pair: their nonzero-split
    # subgraphs take the real-node count as an unbacked dynamic dim from the first
    # build (measured [10,10,10] / [8,8,8]), so shape sweeps must not grow any of them.
    # graph_counts[0] >= 1 is the anti-vacuous half: a no-op dynamo compiles nothing
    # and would trivially satisfy the equality.
    assert graph_counts[0] >= 1, f"RECOMP[{model}]: nothing compiled ({graph_counts})"
    assert graph_counts[2] == graph_counts[0], (
        f"RECOMP[{model}]: re-specializing per shape ({graph_counts})")
