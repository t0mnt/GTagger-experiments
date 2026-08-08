"""Stage-4 compile gates: the non-equivariant family (docs/cgenn-compile.md).

Covers the padded-dense rows: ParT (the in-repo lloca port), ParticleNet, MIParT, the
baseline transformer, the Plain hybrids and the ParticleNet-ParT hybrids (the latter pair
re-runs kNN per block from UPDATED coordinates — dynamic graph recomputation — but via
dense top-k, which traces; the gates adjudicate).

Fixtures (recorded at pre-gate HEAD, record-before-edit discipline):
    NONEQUI_COMPILE=record python -m pytest tests/experiments/test_nonequi_compile.py -q

Gates mirror the strict Stage-1 bars: BIT (torch.equal fp32+fp64), content hashes, and
env-gated (CGENN_COMPILE_GATES=1) TOL/DET/BREAKS (0, cold build)/RECOMP (no growth across
a production-regime sweep; every sweep shape keeps max padded length above the quick
kNN k = 4).
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


@pytest.mark.skipif(not RUN_COMPILE_GATES, reason="compile smoke gates: set CGENN_COMPILE_GATES=1")
@pytest.mark.parametrize("model", MODELS)
def test_tol_det_compiled_vs_eager(model):
    """TOL: compiled net vs eager <= 1e-10 rel (fp64 CPU). DET: compiled twice, torch.equal."""
    ref = torch.load(FIX / f"{model}_fp64.pt", weights_only=False)
    exp = _build(model, float64=True)
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
    """BREAKS: 0 graph breaks over a COLD net. RECOMP: no growth across the sweep."""
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
    exp2 = _build(model, float64=True)
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
        f"RECOMP[{model}]: re-specializing per shape ({graph_counts})")
