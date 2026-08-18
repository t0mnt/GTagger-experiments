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
    """BIT: eager outputs bit-identical to the pre-rewrite recording. torch.equal, no tolerance.

    CPU-TIER by contract, same as the hybrid BIT pins: the fixtures' y is a CPU
    recording and BIT-identity is a same-device statement, so on a GPU node this gate
    (and its RECORD path -- a GPU re-record would silently flip the fixture's device)
    skips. Every other gate in this battery is device-generic and runs everywhere."""
    if torch.cuda.is_available():
        pytest.skip("cgenn_compile BIT fixtures are CPU recordings; BIT is same-device "
                    "-- run this gate in a CPU session")
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


GP_IMPLS = ["matmul", "sparse", "flash"]


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
    # fp32 bar re-set with gather-commute hoisting (FLASH-3 step 1, measured): the
    # einsum-vs-blockdiag reassociation seed is unchanged (~1e-7/layer at fp32), but
    # its amplification through four nonlinear layers depends on the activations'
    # operating points, which the hoist (TOL-class) moved — readings went 4e-6-class
    # to 3.3e-5 for matmul/sparse while CGENN_HOIST=0 reverts them and the fp64
    # comparison, the actual correctness arbiter, stays ~1e-13 under its unchanged
    # 1e-10 bar. 1e-4 still fails loudly on logic bugs (those read 1e-2..1e0).
    bar = 1e-10 if f64 else 1e-4
    print(f"GATE-TOL-IMPL {impl}/{prec} rel={rel:.3e}")
    assert rel < bar, f"TOL-IMPL {impl}/{prec}: {rel:.3e} >= {bar}"


# ------------------------------------------------------------------ backward gates
# Every gate above runs under torch.no_grad() (`_forward`). That was sufficient while all
# three gp_impls were plain autograd-composed arithmetic -- pinning the forward pinned the
# backward with it. It stopped being sufficient the moment gp_impl=sparse started routing
# through a hand-written torch.autograd.Function (experiments/baselines/cgenn/sparse_gp.py):
# its backward is CODE, not a derivative of the gated forward, and every gate in this file
# would stay green with the gradients completely wrong.
#
# The Function's OWN gates -- gradcheck, bit-identity and retention against the exact
# expression it replaced -- are in tests/experiments/test_sparse_gp.py, which is KEEP and
# fixture-free. Only the integration-scale ones live here, because they need this file's
# fixtures and hydra build, and this file is a port instrument the cleanup.md wipe deletes.


def _grads(exp, data):
    """name -> parameter gradient of a fixed, seedless scalar functional of the output."""
    exp.model.zero_grad(set_to_none=True)
    y = exp._get_ypred_and_label(data.clone())[0]
    # distinct weight per output element: no gradient component can cancel out of the
    # comparison the way a plain .sum() lets equal-and-opposite rows do
    w = torch.linspace(-1, 1, y.numel(), dtype=y.dtype, device=y.device).reshape(y.shape)
    (y * w).sum().backward()
    return {n: p.grad.detach().clone() for n, p in exp.model.named_parameters()
            if p.grad is not None}


@pytest.mark.parametrize("impl", GP_IMPLS)
def test_backward_tol_vs_reference(impl):
    """BACKWARD-TOL: whole-model parameter gradients vs the einsum reference, fp64.

    Bar is 1e-8, two orders looser than the forward's 1e-10, and deliberately so: the
    measured worst case sits at ~2.6e-10 for BOTH matmul and sparse, on the same parameter
    (net.CGLs.0.phi_x.0.linear_left.weight). matmul is untouched by the sparse work, so
    that agreement is the signal -- the ~6 digits are the model's own backward
    conditioning, not anything this file's subject introduced. fp32 is not gated: it reads
    ~2.5e-3 for both impls, which is too loose to catch anything.
    """
    path = FIX / "fp64.pt"
    if not path.exists():
        pytest.skip("no cgenn_compile fixtures recorded")
    ref = torch.load(path, weights_only=False)

    def built(ov):
        e = _build(float64=True, extra_overrides=ov)
        e.model.load_state_dict(ref["sd"], strict=True)
        return e

    g_ref = _grads(built(REF_IMPL), _rebuild(ref["batch"]))
    g = _grads(built([f"model.net.gp_impl={impl}"]), _rebuild(ref["batch"]))
    assert set(g) == set(g_ref), f"BACKWARD-TOL[{impl}]: different parameters got gradients"
    worst, where = 0.0, ""
    for n in g_ref:
        rel = float((g[n] - g_ref[n]).abs().max() / (1 + g_ref[n].abs().max()))
        if rel > worst:
            worst, where = rel, n
    print(f"GATE-BACKWARD-TOL[{impl}] worst rel={worst:.3e} at {where}")
    assert worst < 1e-8, f"BACKWARD-TOL[{impl}]: {worst:.3e} >= 1e-8 at {where}"


def _saved_bytes(exp, data):
    held = {}

    def pack(t):
        held[id(t)] = t.numel() * t.element_size()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        exp._get_ypred_and_label(data.clone())
    return sum(held.values())


def test_sparse_retains_less_than_dense():
    """The claim the Function exists to make, as a gate.

    The eager three-liner retained TWO (B, N, 16, 16) tensors per layer where einsum/matmul
    retain one; the custom Function saves only its inputs. Measured here: 84.8 MB vs
    293.4 MB, a 3.46x reduction; the bar is 0.5, so a regression that merely reinstates
    parity fails.

    This gate is EAGER -- `_build` does not compile. Under torch.compile the partitioner
    equalizes all three impls to within 1 MB, so this ratio says nothing about the campaign
    posture, and the 1.50x H100 peak it was once thought to explain remains unexplained.
    """
    path = FIX / "fp64.pt"
    if not path.exists():
        pytest.skip("no cgenn_compile fixtures recorded")
    ref = torch.load(path, weights_only=False)
    got = {}
    for impl in ["einsum", *GP_IMPLS]:
        exp = _build(float64=True, extra_overrides=[f"model.net.gp_impl={impl}"])
        exp.model.load_state_dict(ref["sd"], strict=True)
        got[impl] = _saved_bytes(exp, _rebuild(ref["batch"]))
        print(f"GATE-SAVED[{impl}] {got[impl] / 2**20:.3f} MB retained for backward")
    ratio = got["sparse"] / got["einsum"]
    print(f"GATE-SAVED sparse/einsum = {ratio:.3f}")
    assert ratio < 0.5, (
        f"sparse retains {ratio:.2f}x the einsum path -- the autograd Function in "
        f"experiments/baselines/cgenn/sparse_gp.py is no longer doing its job")


@pytest.mark.skipif(not RUN_COMPILE_GATES, reason="compile smoke gates: set CGENN_COMPILE_GATES=1")
@pytest.mark.parametrize("impl", ["einsum", *GP_IMPLS])
def test_compiled_backward(impl):
    """BACKWARD: a compiled net must survive a real training step, not just a forward.

    Separate from the gate above because it exercises a different thing: dynamo/AOT
    tracing THROUGH the custom Function into a joint forward+backward graph. A Function
    that traces fine under no_grad can still graph-break (or silently fall back) once
    autograd is live, and the compiled backward is what the campaign actually runs.
    """
    exp = _build(float64=True, extra_overrides=[f"model.net.gp_impl={impl}"])
    exp.model.net = torch.compile(exp.model.net, dynamic=True)
    exp.model.train()
    grads = _grads(exp, _fixed_batch(exp))
    assert grads, f"BACKWARD[{impl}]: compiled step produced no gradients at all"
    assert all(torch.isfinite(v).all() for v in grads.values()), (
        f"BACKWARD[{impl}]: non-finite gradients from the compiled step")
    print(f"GATE-BACKWARD[{impl}] compiled training step OK ({len(grads)} grads, finite)")


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
    # dynamo's per-phase compile TIMINGS: a name followed by one or MORE
    # comma-separated floats. The single-float form was normalized from the start;
    # split-graph models compile in several attempts and emit the multi-float form,
    # which churned every run until this was widened (same signal-burying class as
    # the globals-dict id below).
    report = re.sub(r"^([\w.]+)(?:, \d+\.\d+)+$", r"\1, ...", report, flags=re.M)
    if impl == "einsum" and not torch.cuda.is_available():
        # CPU-tier artifact, like the BIT fixtures: the committed report records the CPU
        # reference graph. Writing it on a GPU node records the (different) CUDA graph
        # AND dirties the cluster checkout -- which silently ABORTED the git pull at the
        # head of round-trip #3, so that whole round-trip ran the previous tree (caught
        # because its numbers replicated round-trip #2 to ~1%). The runbook now also
        # hard-resets the cluster checkout, belt and braces.
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


@pytest.mark.skipif(not RUN_COMPILE_GATES, reason="compile smoke gates: set CGENN_COMPILE_GATES=1")
def test_degree_zero_node_compiled_vs_eager():
    """Ragged-edge differential (round-trip #5 audit): a SINGLE-CONSTITUENT jet.

    Its node is real but has ZERO fully-connected edges -- degree-0 receiver, degree-0
    sender -- the shape where the padded scatter-write (compiled), segment_reduce
    (eager), the clamp(min=1) mean divisor, and the hoisted gathers all meet. The
    RECOMP sweep above pushes a 1-constituent jet through the compiled FORWARD only;
    this gate compares eager-vs-compiled forward AND parameter gradients at fp64.
    Low-multiplicity jets are data-reachable, so this is a correctness gate, not an
    exotic: fwd <= 1e-10 (the battery's model-level fp64 bar -- GPU reassociation
    is larger than CPU's), grads <= 1e-8 (the taxonomy bar)."""
    exp = _build(float64=True)
    data = _fixed_batch(exp)
    ptr = data["ptr"]
    keep = torch.cat([torch.arange(ptr[0], ptr[0] + 1), torch.arange(ptr[1], ptr[-1])])
    fields = {k: data[k][keep].clone() for k in ("x", "scalars")}
    sizes = torch.tensor([1] + [int(ptr[i + 1] - ptr[i]) for i in range(1, len(ptr) - 1)])
    fields["ptr"] = torch.cat([torch.zeros(1, dtype=torch.long), sizes.cumsum(0)])
    fields["batch"] = torch.repeat_interleave(torch.arange(len(sizes)), sizes)
    fields["label"] = data["label"].clone()
    crafted = _rebuild(fields)

    def fwd_bwd(d):
        exp.model.zero_grad(set_to_none=True)
        y = exp._get_ypred_and_label(d.clone())[0]
        w = torch.linspace(-1, 1, y.numel(), dtype=y.dtype, device=y.device).reshape(y.shape)
        (y * w).sum().backward()
        g = torch.cat([p.grad.flatten() for p in exp.model.parameters()
                       if p.grad is not None])
        return y.detach().clone(), g.clone()

    exp.model.train()
    y_e, g_e = fwd_bwd(crafted)
    exp.model.net = torch.compile(exp.model.net, dynamic=True)
    y_c, g_c = fwd_bwd(crafted)
    fd = (y_e - y_c).abs().max().item()
    gd = (g_e - g_c).abs().max().item()
    print(f"GATE-DEG0 fwd max|diff|={fd:.3e} grad max|diff|={gd:.3e}")
    assert torch.isfinite(y_c).all() and torch.isfinite(g_c).all(), "DEG0: non-finite"
    assert fd < 1e-10, f"DEG0 fwd: {fd:.3e} >= 1e-10"
    assert gd < 1e-8, f"DEG0 grads: {gd:.3e} >= 1e-8"


@pytest.mark.skipif(not RUN_COMPILE_GATES, reason="compile smoke gates: set CGENN_COMPILE_GATES=1")
def test_regional_compile_vs_eager(monkeypatch):
    """REGIONAL (flash round-trip #6): submodule-wise compile vs eager, flash impl.

    CGENN_REGIONAL=1 + compile=true compiles the plain-nn Sequential MLPs as units
    and keeps orchestration (and the flash custom op) eager -- the experiment's
    premise is that no joint graph contains the opaque op. This gate pins the
    mechanism on CPU: some units actually compile, forward matches eager at fp64
    TOL, gradients exist, are finite, and match eager."""
    monkeypatch.setenv("CGENN_REGIONAL", "1")
    exp = _build(float64=True,
                 extra_overrides=["model.net.gp_impl=flash", "model.compile=true"])
    monkeypatch.delenv("CGENN_REGIONAL")
    exp_e = _build(float64=True,
                   extra_overrides=["model.net.gp_impl=flash", "model.compile=false"])
    exp_e.model.load_state_dict(exp.model.state_dict(), strict=True)
    data = _fixed_batch(exp)

    from torch._dynamo.eval_frame import OptimizedModule
    n_units = sum(isinstance(m, OptimizedModule) for m in exp.model.net.modules())
    # nn.Module.compile() wraps in place (no OptimizedModule node); detect via the
    # _compiled_call_impl it installs instead, on any torch that uses it.
    n_units = max(n_units, sum(getattr(m, "_compiled_call_impl", None) is not None
                               for m in exp.model.net.modules()))
    # 5 on the quick tree (n_layers=1: phi_h/theta_h/psi_x/chi_x + head); the
    # production net has 4 CGLs (~17 units) -- the bound is a floor, not the count.
    assert n_units >= 4, f"REGIONAL: only {n_units} compiled units found (want >= 4 MLPs)"

    def fwd_bwd(e):
        e.model.train()
        e.model.zero_grad(set_to_none=True)
        y = e._get_ypred_and_label(data.clone())[0]
        w = torch.linspace(-1, 1, y.numel(), dtype=y.dtype, device=y.device).reshape(y.shape)
        (y * w).sum().backward()
        g = torch.cat([p.grad.flatten() for p in e.model.parameters()
                       if p.grad is not None])
        return y.detach().clone(), g.clone()

    y_r, g_r = fwd_bwd(exp)
    y_e, g_e = fwd_bwd(exp_e)
    fd = (y_r - y_e).abs().max().item()
    gd = (g_r - g_e).abs().max().item()
    print(f"GATE-REGIONAL units={n_units} fwd max|diff|={fd:.3e} grad max|diff|={gd:.3e}")
    assert torch.isfinite(y_r).all() and torch.isfinite(g_r).all(), "REGIONAL: non-finite"
    assert fd < 1e-10, f"REGIONAL fwd: {fd:.3e} >= 1e-10"
    assert gd < 1e-8, f"REGIONAL grads: {gd:.3e} >= 1e-8"
