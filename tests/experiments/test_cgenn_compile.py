"""Stage-1 gates for CGENN torch.compile support (docs/cgenn-compile.md).

Fixtures (record BEFORE any rewrite lands — the whole point is eager-vs-recorded bit-identity):
    CGENN_COMPILE=record python -m pytest tests/experiments/test_cgenn_compile.py -q
writes tests/fixtures/cgenn_compile/{fp32,fp64}.pt (a fixed mini-dataset batch + eager outputs
of the tag_cgenn model) plus content_hashes.json (canonical content hashes — torch.save file
bytes are process-dependent at identical content, so hashes are computed over sorted-key
tensor bytes, the same contract as the lgatr parity fixtures).

Gates (check mode, no env var):
  BIT    eager forward vs recorded fixtures — torch.equal, fp32 AND fp64, zero tolerance.
         The §2 rewrites are pure data movement and the §einsum rewrite is measured
         bit-identical; a BIT failure is a rewrite bug. Never relax to allclose.
  TOL    compiled net vs eager net — relative <= 1e-10 (fp64, CPU).
  DET    compiled net twice — torch.equal.
  BREAKS torch._dynamo.explain over a COLD (freshly built) net — 0 graph breaks (report
         committed next to the fixtures as dynamo_explain.txt). Cold matters: explain
         after an eager warm-up cannot see first-call-only breaks (cached_property RLock).
  RECOMP forward sweep over (B, P) shapes with dynamic=True — <= 2 compilations.
  (SUITE = the repo's normal pytest run with the knob off; this file is part of it.)
Compile gates are skipped on CPU test runs unless CGENN_COMPILE_GATES=1 (they are the
dedicated smoke; compile on CPU is slow but valid).
"""

import hashlib
import json
import os
import re
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "tests" / "fixtures" / "cgenn_compile"
RECORD = os.environ.get("CGENN_COMPILE") == "record"
RUN_COMPILE_GATES = os.environ.get("CGENN_COMPILE_GATES") == "1"

torch.set_num_threads(1)  # run-context-independent arithmetic (same lesson as lgatr parity)


def _build(float64, extra_overrides=()):
    import logging.handlers  # noqa: F401
    import hydra
    import experiments.logger
    from experiments.tagging.experiment import TopTaggingExperiment

    experiments.logger.LOGGER.disabled = True
    overrides = ["save=false", "training.batchsize=4", "data.dataset=mini",
                 "model=tag_cgenn", f"use_float64={'true' if float64 else 'false'}",
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


def _fixed_batch(exp):
    torch.manual_seed(1)
    return next(iter(exp.train_loader))


def _batch_fields(data):
    return {k: data[k].clone() for k in data.keys() if torch.is_tensor(data[k])}


def _rebuild(fields):
    from torch_geometric.data import Data
    d = Data()
    for k, v in fields.items():
        d[k] = v.clone()
    return d


def _forward(exp, data):
    with torch.no_grad():
        return exp._get_ypred_and_label(data.clone())[0].detach().clone()


def _content_hash(obj):
    h = hashlib.sha256()

    def feed(x, path):
        h.update(path.encode())
        if isinstance(x, dict):
            for k in sorted(x, key=str):
                feed(x[k], f"{path}/{k}")
        elif isinstance(x, (list, tuple)):
            for i, v in enumerate(x):
                feed(v, f"{path}[{i}]")
        elif torch.is_tensor(x):
            h.update(str(x.dtype).encode())
            h.update(str(tuple(x.shape)).encode())
            h.update(x.detach().cpu().contiguous().numpy().tobytes())
        else:
            h.update(repr(x).encode())

    feed(obj, "")
    return h.hexdigest()


REF_IMPL = ["model.net.gp_impl=einsum"]  # the BIT-reference path; the yaml default is the
# campaign posture (sparse) and is TOL-class, so reference gates pin einsum explicitly


@pytest.mark.parametrize("prec", ["fp32", "fp64"])
def test_bit_eager_vs_fixtures(prec):
    """BIT: eager outputs bit-identical to the pre-rewrite recording. torch.equal, no tolerance."""
    path = FIX / f"{prec}.pt"
    exp = _build(float64=(prec == "fp64"), extra_overrides=REF_IMPL)
    if RECORD:
        data = _fixed_batch(exp)
        pack = {"batch": _batch_fields(data), "y": _forward(exp, data),
                "sd": exp.model.state_dict()}
        FIX.mkdir(parents=True, exist_ok=True)
        torch.save(pack, path)
        return
    if not path.exists():
        pytest.skip("no cgenn_compile fixtures recorded")
    ref = torch.load(path, weights_only=False)
    exp.model.load_state_dict(ref["sd"], strict=True)
    y = _forward(exp, _rebuild(ref["batch"]))
    assert torch.equal(y, ref["y"]), (
        f"{prec}: eager output not bit-identical to the pre-rewrite fixture "
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
        print(f"{live}  {fname}")
        assert live == expected, f"{fname}: fixture content changed"


GP_IMPLS = ["matmul", "sparse"]


@pytest.mark.parametrize("impl", GP_IMPLS)
@pytest.mark.parametrize("prec", ["fp32", "fp64"])
def test_impl_tol_vs_reference(impl, prec):
    """TOL-IMPL: gp_impl variants vs the einsum reference — reassociation-scale only.

    matmul (dense GEMM) and sparse (quasigroup gather over the 256 nonzero cayley
    entries) reorder the same arithmetic, so they are TOL-class by design (R2 bars).
    BIT stays owned by the default einsum path, which these variants never touch.
    """
    path = FIX / f"{prec}.pt"
    if not path.exists():
        pytest.skip("no cgenn_compile fixtures recorded")
    ref = torch.load(path, weights_only=False)
    f64 = prec == "fp64"
    exp = _build(float64=f64, extra_overrides=REF_IMPL)
    exp.model.load_state_dict(ref["sd"], strict=True)
    y_ref = _forward(exp, _rebuild(ref["batch"]))
    exp2 = _build(float64=f64, extra_overrides=[f"model.net.gp_impl={impl}"])
    exp2.model.load_state_dict(ref["sd"], strict=True)
    y = _forward(exp2, _rebuild(ref["batch"]))
    rel = (y - y_ref).abs().max() / (1 + y_ref.abs().max())
    bar = 1e-10 if f64 else 1e-5
    print(f"GATE-TOL-IMPL {impl}/{prec} rel={rel:.3e}")
    assert rel < bar, f"TOL-IMPL {impl}/{prec}: {rel:.3e} >= {bar}"


def _compiled_net(exp):
    return torch.compile(exp.model.net, dynamic=True)


@pytest.mark.skipif(not RUN_COMPILE_GATES, reason="compile smoke gates: set CGENN_COMPILE_GATES=1")
@pytest.mark.parametrize("impl", ["einsum", *GP_IMPLS])
def test_tol_det_compiled_vs_eager(impl):
    """TOL: compiled vs eager <= 1e-10 rel (fp64 CPU), per gp_impl. DET: compiled twice."""
    ref = torch.load(FIX / "fp64.pt", weights_only=False)
    exp = _build(float64=True, extra_overrides=[f"model.net.gp_impl={impl}"])
    exp.model.load_state_dict(ref["sd"], strict=True)
    y_eager = _forward(exp, _rebuild(ref["batch"]))
    exp.model.net = _compiled_net(exp)
    y1 = _forward(exp, _rebuild(ref["batch"]))
    y2 = _forward(exp, _rebuild(ref["batch"]))
    rel = (y1 - y_eager).abs().max() / (1 + y_eager.abs().max())
    print(f"GATE-TOL[{impl}] compiled-vs-eager rel={rel:.3e}")
    assert rel < 1e-10, f"TOL[{impl}]: {rel:.3e} >= 1e-10"
    assert torch.equal(y1, y2), f"DET[{impl}]: compiled forward not deterministic across calls"


@pytest.mark.skipif(not RUN_COMPILE_GATES, reason="compile smoke gates: set CGENN_COMPILE_GATES=1")
@pytest.mark.parametrize("impl", ["einsum", *GP_IMPLS])
def test_breaks_and_recomp(impl):
    """BREAKS: 0 graph breaks over the net. RECOMP: <= 2 compiles across a (B, P) sweep.

    Parametrized over gp_impl; the committed explain artifact stays owned by the einsum
    reference path (the other impls assert the same 0-break bar without artifact churn).
    """
    import torch._dynamo as dynamo
    ref = torch.load(FIX / "fp64.pt", weights_only=False)
    ov = [f"model.net.gp_impl={impl}"]
    exp = _build(float64=True, extra_overrides=ov)
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
    # explain must see a COLD model: lazily-materialized state (e.g. functools.cached_property
    # fills the instance dict through an RLock on first touch) is invisible to explain once any
    # eager forward has warmed the instance -- exactly how six first-call-only RLock graph
    # breaks hid behind a clean explain report while RECOMP counted their fragments. The model
    # code now materializes those at init; the cold rebuild keeps this gate honest anyway.
    exp_cold = _build(float64=True, extra_overrides=ov)
    exp_cold.model.load_state_dict(ref["sd"], strict=True)
    explanation = dynamo.explain(exp_cold.model.net)(*captured["args"], **captured["kwargs"])
    # str(explanation) embeds repr()s of live objects, whose heap addresses differ every
    # process -- so the committed artifact came back 637 lines "changed" after every run,
    # which is how a tracked report stops being read and starts being `git checkout`ed.
    # Normalize the addresses: the file then diffs only when the GRAPH changes, which is
    # the thing it exists to record.
    report = str(explanation)
    report = re.sub(r"0x[0-9a-fA-F]+", "0x...", report)  # repr() heap addresses
    # ___check_type_id / ___check_obj_id guards embed id() of a class or module object
    report = re.sub(r"(___check_(?:type|obj)_id\([^,]+, )\d+\)", r"\1...)", report)
    # dynamo numbers its per-frame globals dicts by how many frames it has seen in
    # the PROCESS, so this id shifts with test ORDER and churned ~150 noise lines per
    # gated run -- which would bury a real change. Normalized; no assertion reads it.
    report = re.sub(r"__builtins_dict___\d+", "__builtins_dict___N", report)
    # trailing compile-time table: wall-clock seconds, non-deterministic by nature
    report = re.sub(r"^([\w.]+), \d+\.\d+$", r"\1, ...", report, flags=re.M)
    if impl == "einsum":
        (FIX / "dynamo_explain.txt").write_text(report)
    print(f"GATE-BREAKS[{impl}] graph_break_count =", explanation.graph_break_count)
    assert explanation.graph_break_count == 0, f"graph breaks:\n{report[:2000]}"

    dynamo.reset()
    from torch._dynamo.utils import counters as dyn_counters
    dyn_counters.clear()  # measurement isolation: counters survive dynamo.reset(), and the
    # explain() call above compiles its own segments -- without this the RECOMP number
    # counts them too (observed 21 with a break-free net)
    exp2 = _build(float64=True, extra_overrides=ov)
    exp2.model.load_state_dict(ref["sd"], strict=True)
    exp2.model.net = torch.compile(exp2.model.net, dynamic=True)
    ptr = ref["batch"]["ptr"]
    for keep in [[1, 3, 20, 40], [2, 5, 30, 25], [4, 4, 4, 4]]:
        rows = []
        for j, n in enumerate(keep):
            n = min(n, int(ptr[j + 1] - ptr[j]))
            rows.extend(range(int(ptr[j]), int(ptr[j]) + n))
        idx = torch.tensor(rows, dtype=torch.long)
        d2 = _rebuild(ref["batch"])
        for key in ("x", "scalars", "batch"):
            d2[key] = d2[key].index_select(0, idx)
        counts = [min(k, int(ptr[j + 1] - ptr[j])) for j, k in enumerate(keep)]
        d2.ptr = torch.tensor([0] + list(torch.tensor(counts).cumsum(0)), dtype=torch.long)
        d2.batch = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts))
        _forward(exp2, d2)
    n_compiles = sum(v for k, v in dyn_counters["stats"].items() if k == "unique_graphs")
    print(f"GATE-RECOMP[{impl}] unique_graphs =", n_compiles)
    assert n_compiles <= 2, f"RECOMP[{impl}]: {n_compiles} unique graphs (> 2) across the sweep"
