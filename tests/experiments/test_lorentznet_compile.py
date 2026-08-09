"""Stage-2 gates: torch.compile support for tag_lorentznet (docs/cgenn-compile.md).

The migration runbook's readiness note held: the LGEB stack is compile-clean as-is —
edges are built in the (permanently eager) wrapper, the net interface is all-tensor, and
there are no cached properties, 3-operand einsums, tensor-valued repeats, bool-mask
scatters, or in-trace ``.item()`` calls. Stage 2 is therefore gate-running, no rewrites.

Fixtures (recorded at pre-gate HEAD, same discipline as Stages 1/3):
    LORENTZNET_COMPILE=record python -m pytest tests/experiments/test_lorentznet_compile.py -q

Gates mirror test_cgenn_compile.py: BIT (torch.equal fp32+fp64 vs recording), fixture
content hashes, and env-gated (CGENN_COMPILE_GATES=1) TOL/DET/BREAKS/RECOMP with the
same bars as Stage 1 (0 breaks on a cold build; <= 2 unique graphs across the sweep).
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

FIX = REPO / "tests" / "fixtures" / "lorentznet_compile"
RECORD = os.environ.get("LORENTZNET_COMPILE") == "record"
RUN_COMPILE_GATES = os.environ.get("CGENN_COMPILE_GATES") == "1"


def _build(float64, extra_overrides=()):
    import logging.handlers  # noqa: F401
    import hydra
    import experiments.logger
    from experiments.tagging.experiment import TopTaggingExperiment

    experiments.logger.LOGGER.disabled = True
    overrides = ["save=false", "training.batchsize=4", "data.dataset=mini",
                 "model=tag_lorentznet", f"use_float64={'true' if float64 else 'false'}",
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


@pytest.mark.parametrize("prec", ["fp32", "fp64"])
def test_bit_eager_vs_fixtures(prec):
    """BIT: eager outputs bit-identical to the recording (no rewrites landed = must hold)."""
    path = FIX / f"{prec}.pt"
    exp = _build(float64=(prec == "fp64"))
    if RECORD:
        data = _fixed_batch(exp)
        pack = {"batch": _batch_fields(data), "y": _forward(exp, data),
                "sd": exp.model.state_dict()}
        FIX.mkdir(parents=True, exist_ok=True)
        torch.save(pack, path)
        return
    if not path.exists():
        pytest.skip("no lorentznet_compile fixtures recorded")
    ref = torch.load(path, weights_only=False)
    exp.model.load_state_dict(ref["sd"], strict=True)
    y = _forward(exp, _rebuild(ref["batch"]))
    assert torch.equal(y, ref["y"]), (
        f"{prec}: eager output changed vs the recorded fixture "
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
def test_tol_det_compiled_vs_eager():
    """TOL: compiled net vs eager <= 1e-10 rel (fp64 CPU). DET: compiled twice, torch.equal."""
    ref = torch.load(FIX / "fp64.pt", weights_only=False)
    exp = _build(float64=True)
    exp.model.load_state_dict(ref["sd"], strict=True)
    y_eager = _forward(exp, _rebuild(ref["batch"]))
    exp.model.net = torch.compile(exp.model.net, dynamic=True)
    y1 = _forward(exp, _rebuild(ref["batch"]))
    y2 = _forward(exp, _rebuild(ref["batch"]))
    rel = (y1 - y_eager).abs().max() / (1 + y_eager.abs().max())
    print(f"GATE-TOL[lorentznet] compiled-vs-eager rel={rel:.3e}")
    assert rel < 1e-10, f"TOL: {rel:.3e} >= 1e-10"
    assert torch.equal(y1, y2), "DET: compiled forward not deterministic across calls"


@pytest.mark.skipif(not RUN_COMPILE_GATES, reason="compile smoke gates: set CGENN_COMPILE_GATES=1")
def test_breaks_and_recomp():
    """BREAKS: 0 graph breaks over a COLD net. RECOMP: <= 2 compiles across a (B, P) sweep."""
    import torch._dynamo as dynamo
    ref = torch.load(FIX / "fp64.pt", weights_only=False)
    exp = _build(float64=True)
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
    exp_cold = _build(float64=True)
    exp_cold.model.load_state_dict(ref["sd"], strict=True)
    explanation = dynamo.explain(exp_cold.model.net)(*captured["args"], **captured["kwargs"])
    report = str(explanation)
    report = re.sub(r"0x[0-9a-fA-F]+", "0x...", report)
    report = re.sub(r"(___check_(?:type|obj)_id\([^,]+, )\d+\)", r"\1...)", report)
    # dynamo numbers its per-frame globals dicts by how many frames it has seen in
    # the PROCESS, so this id shifts with test ORDER and churned ~150 noise lines per
    # gated run -- which would bury a real change. Normalized; no assertion reads it.
    report = re.sub(r"__builtins_dict___\d+", "__builtins_dict___N", report)
    report = re.sub(r"^([\w.]+), \d+\.\d+$", r"\1, ...", report, flags=re.M)
    (FIX / "dynamo_explain.txt").write_text(report)
    print("GATE-BREAKS[lorentznet] graph_break_count =", explanation.graph_break_count)
    assert explanation.graph_break_count == 0, f"graph breaks:\n{report[:2000]}"

    dynamo.reset()
    from torch._dynamo.utils import counters as dyn_counters
    dyn_counters.clear()
    exp2 = _build(float64=True)
    exp2.model.load_state_dict(ref["sd"], strict=True)
    exp2.model.net = torch.compile(exp2.model.net, dynamic=True)
    ptr = ref["batch"]["ptr"]
    for keep in [[1, 3, 20, 40], [2, 5, 30, 25], [4, 4, 4, 4]]:
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
    n_compiles = sum(v for k, v in dyn_counters["stats"].items() if k == "unique_graphs")
    print("GATE-RECOMP[lorentznet] unique_graphs =", n_compiles)
    assert n_compiles <= 2, f"RECOMP: {n_compiles} unique graphs (> 2) across the sweep"
