"""Stage-3 gates: the CGENN-hybrid port of the compile fix family (docs/cgenn-compile.md).

Covers the two CGENN hybrids (tag_CGENNLGATrGraphTrans, tag_CGENNLGATrGraphGPS); the GPS
hybrid imports its CGENN stack from CGENNLGATrGraphTransHybrid.py, so one file's rewrites
serve both — and both get their own fixtures and gates here.

Fixtures (record BEFORE any rewrite lands — same record-before-edit discipline as Stage 1):
    CGENN_HYBRID_COMPILE=record python -m pytest tests/experiments/test_cgenn_hybrid_compile.py -q

Gates mirror test_cgenn_compile.py:
  BIT      eager forward vs recorded fixtures — torch.equal, fp32 AND fp64 (gp_impl pinned
           to the einsum reference path once the knob exists; before that, the only path).
  TOL-IMPL gp_impl variants vs the einsum reference — reassociation-scale bars (activates
           automatically once the hybrid net grows the gp_impl kwarg).
  TOL/DET/BREAKS/RECOMP  compiled-net gates, env-gated via CGENN_COMPILE_GATES=1.
The lgatr144 parity fixtures remain a second, independent guard over hybrid behavior.
"""

import inspect
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

FIX = REPO / "tests" / "fixtures" / "cgenn_hybrid_compile"
RECORD = os.environ.get("CGENN_HYBRID_COMPILE") == "record"
RUN_COMPILE_GATES = os.environ.get("CGENN_COMPILE_GATES") == "1"

MODELS = ["tag_CGENNLGATrGraphTrans", "tag_CGENNLGATrGraphGPS"]
GP_IMPLS = ["matmul", "sparse"]


def _has_gp_impl():
    from experiments.baselines.CGENNLGATrGraphTransHybrid import CGENNLGATrGraphTrans
    return "gp_impl" in inspect.signature(CGENNLGATrGraphTrans.__init__).parameters


def _ref_overrides():
    # pin the BIT-reference contraction once the knob exists; before that there is only one path
    return ["model.net.gp_impl=einsum"] if _has_gp_impl() else []


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
    """BIT: hybrid eager outputs bit-identical to the pre-port recording."""
    path = FIX / f"{model}_{prec}.pt"
    exp = _build(model, float64=(prec == "fp64"), extra_overrides=_ref_overrides())
    if RECORD:
        data = _fixed_batch(exp)
        pack = {"batch": _batch_fields(data), "y": _forward(exp, data),
                "sd": exp.model.state_dict()}
        FIX.mkdir(parents=True, exist_ok=True)
        torch.save(pack, path)
        return
    if not path.exists():
        pytest.skip("no cgenn_hybrid_compile fixtures recorded")
    ref = torch.load(path, weights_only=False)
    exp.model.load_state_dict(ref["sd"], strict=True)
    y = _forward(exp, _rebuild(ref["batch"]))
    assert torch.equal(y, ref["y"]), (
        f"{model}/{prec}: eager output not bit-identical to the pre-port fixture "
        f"(max|diff|={(y - ref['y']).abs().max().item():.3e}) — rewrite bug, do not relax")


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


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("impl", GP_IMPLS)
def test_impl_tol_vs_reference(model, impl):
    """TOL-IMPL: hybrid gp_impl variants vs the einsum reference (fp64, rel <= 1e-10)."""
    if not _has_gp_impl():
        pytest.skip("hybrid gp_impl knob not landed yet")
    path = FIX / f"{model}_fp64.pt"
    if not path.exists():
        pytest.skip("no fixtures")
    ref = torch.load(path, weights_only=False)
    exp = _build(model, float64=True, extra_overrides=_ref_overrides())
    exp.model.load_state_dict(ref["sd"], strict=True)
    y_ref = _forward(exp, _rebuild(ref["batch"]))
    exp2 = _build(model, float64=True, extra_overrides=[f"model.net.gp_impl={impl}"])
    exp2.model.load_state_dict(ref["sd"], strict=True)
    y = _forward(exp2, _rebuild(ref["batch"]))
    rel = (y - y_ref).abs().max() / (1 + y_ref.abs().max())
    print(f"GATE-TOL-IMPL {model} {impl} rel={rel:.3e}")
    assert rel < 1e-10, f"TOL-IMPL {model}/{impl}: {rel:.3e} >= 1e-10"


@pytest.mark.skipif(not RUN_COMPILE_GATES, reason="compile smoke gates: set CGENN_COMPILE_GATES=1")
@pytest.mark.parametrize("model", MODELS)
def test_tol_det_compiled_vs_eager(model):
    """TOL: compiled hybrid net vs eager <= 1e-10 rel (fp64 CPU). DET: compiled twice."""
    ref = torch.load(FIX / f"{model}_fp64.pt", weights_only=False)
    exp = _build(model, float64=True, extra_overrides=_ref_overrides())
    exp.model.load_state_dict(ref["sd"], strict=True)
    y_eager = _forward(exp, _rebuild(ref["batch"]))
    exp.model.net = torch.compile(exp.model.net, dynamic=True)
    y1 = _forward(exp, _rebuild(ref["batch"]))
    y2 = _forward(exp, _rebuild(ref["batch"]))
    rel = (y1 - y_eager).abs().max() / (1 + y_eager.abs().max())
    print(f"GATE-TOL[{model}] compiled-vs-eager rel={rel:.3e}")
    assert rel < 1e-10, f"TOL[{model}]: {rel:.3e} >= 1e-10"
    assert torch.equal(y1, y2), f"DET[{model}]: compiled forward not deterministic"


@pytest.mark.skipif(not RUN_COMPILE_GATES, reason="compile smoke gates: set CGENN_COMPILE_GATES=1")
@pytest.mark.parametrize("model", MODELS)
def test_breaks_and_recomp(model):
    """BREAKS: explain over a COLD hybrid net with the edges HOISTED — the wrapper builds
    them via net.build_edges outside the compiled region, exactly like tag_cgenn's
    permanently-eager wrapper edges — so the strict Stage-1 bar applies: 0 graph breaks.
    Port history: 24 breaks (pre-port) -> 3 (fix family eliminated) -> 0 (edge hoist).
    RECOMP: strict bar again — unique_graphs must not grow across the production-regime
    sweep (padded length P with P-1 >= k; a small-P batch would legitimately compile one
    extra kNN regime, k_actual = min(k, P-1) being a real branch that changes topk's
    shape semantics — verified via guard_fail_fn, docs/cgenn-compile.md Stage-3 log. The
    sweep stays in the production regime; quick-config k = 4)."""
    import torch._dynamo as dynamo
    ref = torch.load(FIX / f"{model}_fp64.pt", weights_only=False)
    ov = _ref_overrides()
    exp = _build(model, float64=True, extra_overrides=ov)
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
    exp_cold = _build(model, float64=True, extra_overrides=ov)
    exp_cold.model.load_state_dict(ref["sd"], strict=True)
    explanation = dynamo.explain(exp_cold.model.net)(*captured["args"], **captured["kwargs"])
    report = str(explanation)
    report = re.sub(r"0x[0-9a-fA-F]+", "0x...", report)
    report = re.sub(r"(___check_(?:type|obj)_id\([^,]+, )\d+\)", r"\1...)", report)
    report = re.sub(r"^([\w.]+), \d+\.\d+$", r"\1, ...", report, flags=re.M)
    (FIX / f"dynamo_explain_{model}.txt").write_text(report)
    print(f"GATE-BREAKS[{model}] graph_break_count =", explanation.graph_break_count)
    assert explanation.graph_break_count == 0, f"graph breaks:\n{report[:2000]}"

    dynamo.reset()
    from torch._dynamo.utils import counters as dyn_counters
    dyn_counters.clear()
    exp2 = _build(model, float64=True, extra_overrides=ov)
    exp2.model.load_state_dict(ref["sd"], strict=True)
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
    assert graph_counts[2] == graph_counts[0], (
        f"RECOMP[{model}]: re-specializing per shape inside the production kNN regime "
        f"({graph_counts})")
