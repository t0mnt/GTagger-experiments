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
    """BIT: eager outputs bit-identical to the recording."""
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
    if model in ("tag_ParticleNetParTGraphTrans", "tag_ParticleNetParTGraphGPS"):
        for m in net.modules():
            if hasattr(m, "compiled_attention"):
                m.compiled_attention = True
            if hasattr(m, "compiled_dense"):
                m.compiled_dense = True  # PairEmbed all-pairs twin (tril_indices pins seq_len)


# Models whose compile knob changes TRAINING semantics, not just kernel fusion: the
# PairEmbed twin feeds the padded PxP grid to a BatchNorm-first embed where the eager
# sparse path feeds real pairs only. Eval is exact (running stats); train is not, and BN
# running buffers are persistent state that would carry into checkpoints. Every model
# listed here MUST ship compile: false -- asserted below, so a bare flip fails the gate.
TWIN_TRAIN_DIVERGENT = {
    "tag_ParT",
    "tag_ParticleNetParTGraphTrans",
    "tag_ParticleNetParTGraphGPS",
}


def _kill_dropout(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
        if isinstance(m, torch.nn.MultiheadAttention):
            m.dropout = 0.0


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

    Clean models must be bit-identical. The documented twin models must (a) still
    diverge -- if one goes clean, the twin was changed and the docs/posture need
    revisiting -- and (b) ship ``compile: false``, which is the assertion that actually
    prevents a compiled-trained checkpoint from ever being produced.
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

    if model in TWIN_TRAIN_DIVERGENT:
        assert dy > 0 or db > 0, (
            f"TRAIN[{model}]: listed as twin-divergent but now bit-identical -- if the "
            f"twin was made train-exact (masked pair-BN), remove it from "
            f"TWIN_TRAIN_DIVERGENT and revisit the compile posture")
        cfg = (REPO / "config" / "model" / f"{model}.yaml").read_text()
        assert re.search(r"^compile:\s*false\b", cfg, flags=re.M), (
            f"TRAIN[{model}]: ships a compile knob that changes TRAINING numerics "
            f"(max|dy|={dy:.3e}, max|d(BN)|={db:.3e}) -- production config must keep "
            f"compile: false until the pair-BN is made mask-aware")
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
    report = re.sub(r"^([\w.]+), \d+\.\d+$", r"\1, ...", report, flags=re.M)
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
