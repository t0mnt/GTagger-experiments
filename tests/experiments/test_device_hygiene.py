"""GPU-class device bugs, caught on a CPU-only runner.

Motivation (docs/cgenn-compile.md, final audit): the CGENN hybrid's
``CliffordAlgebra.b()`` built its blade-index tensors on the DEFAULT device (CPU --
nothing in this repo calls ``torch.set_default_device``) and used them to index a
GPU-resident buffer. Bit-identical and free on a CPU runner, a per-forward host->device
transfer on a GPU. Every gate in the repo runs on CPU, so nothing could see it, and the
fix reached only one of two duplicated copies of the class.

Two complementary nets, neither of which needs a GPU:

1. ``test_no_device_implicit_tensor_in_forward`` (dynamic) intercepts torch's tensor
   factories during a REAL forward and fails on any call that omits ``device=``,
   reporting the exact call site. Zero false positives -- it sees only executed code.
2. ``test_no_default_device_tensor_creation`` (static) is the cheap always-on scan of
   the same property, so a new offender fails even in code the mini batch doesn't reach.

Both are proven non-vacuous: reverting the ``b()`` fix makes each of them fail on
exactly that line, which also demonstrates that ``b()`` really is on the live forward
path (via ``MVLayerNorm -> norm()``).

A fake-CUDA forward under ``FakeTensorMode`` was tried FIRST and abandoned, recorded
here so nobody retries it: fake tensors do enforce device agreement (verified: binary
ops, matmul and index_select all raise ``FakeTensorDeviceMismatchError``), but they
cannot traverse these models at all -- ``to_dense_batch`` needs real counts and raises
``GuardOnDataDependentSymNode`` even with a ``ShapeEnv``. Note also that advanced
indexing (``gpu[cpu_idx]``) is legal in torch and raises nowhere, which is precisely
why the b() bug degraded performance rather than crashing.
"""

import ast
import os

import pytest
import torch

from tests.experiments.test_cgenn_compile import REPO, _fixed_batch
from tests.experiments.test_nonequi_compile import _build

RUN_SLOW = os.environ.get("CGENN_COMPILE_GATES") == "1"

MODELS = [
    "tag_ParT", "tag_particlenet", "tag_transformer", "tag_MIParT",
    "tag_PlainGraphTrans", "tag_PlainGraphGPS",
    "tag_ParticleNetParTGraphTrans", "tag_ParticleNetParTGraphGPS",
    "tag_cgenn", "tag_lorentznet",
    "tag_CGENNLGATrGraphTrans", "tag_CGENNLGATrGraphGPS",
    "tag_LorentzNetLGATrSlimGraphTrans", "tag_LorentzNetLGATrSlimGraphGPS",
]


_TRACED_FACTORIES = ["tensor", "zeros", "ones", "empty", "arange", "eye", "full",
                     "rand", "randn", "randint"]


@pytest.mark.skipif(not RUN_SLOW, reason="device hygiene sweep: set CGENN_COMPILE_GATES=1")
@pytest.mark.parametrize("model", MODELS)
def test_no_device_implicit_tensor_in_forward(model):
    """DYNAMIC: run a real forward and record every tensor factory called WITHOUT
    ``device=``, with its call site.

    Strictly better than the static scan for precision -- it sees only code that
    actually executes, so unreachable helpers and __init__-time table building cannot
    produce false positives, and it needs no allowlist. (A fake-CUDA forward was tried
    first and abandoned: FakeTensorMode cannot traverse these models at all, because
    `to_dense_batch` needs real counts and raises GuardOnDataDependentSymNode.)

    Construction is deliberately excluded by clearing the record after the model is
    built: __init__-time tensors become buffers/parameters that `.to(device)` moves.
    """
    import traceback

    hits, orig = {}, {n: getattr(torch, n) for n in _TRACED_FACTORIES}

    def make(fn):
        def wrapper(*a, **k):
            if k.get("device") is None:
                # attribute ONLY when our code is the DIRECT creator: stack[-1] is this
                # wrapper, so stack[-2] is the immediate caller. Walking further up
                # blames our call site for a third-party library's internal allocation
                # (e.g. xformers' BlockDiagonalMask.from_seqlens builds its own tensors
                # from a python list; placement is xformers' business, not ours).
                stack = traceback.extract_stack()
                fr = stack[-2] if len(stack) >= 2 else None
                if fr and "/experiments/" in fr.filename and "/test_" not in fr.filename:
                    rel = fr.filename.split("GTagger-experiments/")[-1]
                    hits.setdefault(f"{rel}:{fr.lineno}", (fr.line or "").strip())
            return fn(*a, **k)
        return wrapper

    for n in _TRACED_FACTORIES:
        setattr(torch, n, make(orig[n]))
    try:
        exp = _build(model, float64=False)
        data = _fixed_batch(exp)
        hits.clear()  # measure the FORWARD only, not construction
        with torch.no_grad():
            exp._get_ypred_and_label(data.clone())
    finally:
        for n in _TRACED_FACTORIES:
            setattr(torch, n, orig[n])

    print(f"GATE-DEVICE[{model}] per-forward device-implicit factory calls = {len(hits)}")
    assert not hits, (
        f"{model}: tensor(s) created per-forward without device= (CPU by default -- a "
        f"host->device transfer every forward on GPU, or a mismatch):\n  "
        + "\n  ".join(f"{k}  |  {v}" for k, v in sorted(hits.items())))


# Tensor factories that silently default to CPU when `device=` is omitted.
# The `*_like` family is deliberately EXCLUDED: it inherits device (and dtype) from its
# argument, so those calls are device-correct by construction.
_FACTORIES = {"tensor", "zeros", "ones", "empty", "arange", "eye", "full", "linspace",
              "rand", "randn", "randint"}

# Enclosing functions exempt from the per-forward rule, each with the reason it cannot
# run inside a forward. Verified by call-site inspection, not assumed.
_NON_FORWARD_FNS = {
    "construct_gmt": "called only from CliffordAlgebra.__init__ (blade table build)",
    "_compute_lookup": "__init__-time path-index lookup",
    "sparse_gp_tables": "__init__-time sparse table build",
    "output_blades": "no call site in the repo",
    "random": "equivariance-test utility; no forward call site",
    "random_vector": "equivariance-test utility; reached only via versor/rotor",
    "versor": "equivariance-test utility; no forward call site",
    "rotor": "equivariance-test utility; no forward call site",
    "geometric_product_paths": "__init__-time grade-path table",
}
# files whose tensors must all be device-explicit (model code that runs per forward)
_SCANNED = [
    "experiments/baselines/CGENNLGATrGraphTransHybrid.py",
    "experiments/baselines/cgenn/cliffordalgebra.py",
    "experiments/baselines/cgennlgatrgraphgps.py",
    "experiments/baselines/particletransformer.py",
    "experiments/baselines/particlenettransformer.py",
    "experiments/baselines/particlenetpartgraphgps.py",
    "experiments/baselines/plaingraphtrans.py",
    "experiments/baselines/plaingraphgps.py",
    "experiments/baselines/lorentznet.py",
    "experiments/baselines/lorentznetlgatrslimgraphtrans.py",
    "experiments/baselines/lorentznetlgatrslimgraphgps.py",
]


def _factory_calls_without_device(path):
    """(line, source) for torch.<factory>(...) calls lacking a device= kwarg."""
    src = (REPO / path).read_text()
    tree = ast.parse(src)
    lines = src.splitlines()
    # map: function def line ranges for __init__, which is allowed to build on CPU
    exempt = {"__init__", "reset_parameters", *_NON_FORWARD_FNS}
    init_ranges = [(n.lineno, n.end_lineno) for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name in exempt]
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr in _FACTORIES
                and isinstance(f.value, ast.Name) and f.value.id == "torch"):
            continue
        if any(kw.arg == "device" for kw in node.keywords):
            continue
        if any(lo <= node.lineno <= hi for lo, hi in init_ranges):
            continue  # __init__-time: becomes a buffer/param, moved by .to()
        out.append((node.lineno, lines[node.lineno - 1].strip()))
    return out


def test_no_default_device_tensor_creation():
    """No per-forward tensor factory may omit `device=` in model code.

    Nothing sets a default device, so an omitted `device=` means CPU regardless of
    where the model lives -- at best a silent host->device transfer every forward, at
    worst a device-mismatch crash on GPU. __init__-time construction is exempt (those
    become buffers/parameters that `.to()` moves).
    """
    offenders = []
    for path in _SCANNED:
        for lineno, src in _factory_calls_without_device(path):
            offenders.append(f"{path}:{lineno}: {src}")
    assert not offenders, (
        "per-forward tensor creation without device= (CPU by default, so this is a "
        "host->device transfer or a mismatch on GPU):\n  " + "\n  ".join(offenders))


# ---------------------------------------------------------------------------------
# DDP-class bugs: the same "only on the real training rig" shape as the device bugs
# above. base_experiment._init_model REPLACES `self.model.net` with
# DistributedDataParallel(net) whenever world_size > 1, and run.py sets
# world_size = torch.cuda.device_count() -- so this is the DEFAULT on a multi-GPU node.
# Every gate here runs single-process on CPU and cannot see it.
#
# Found by audit, not by a gate: this branch added two wrapper->net reach-ins that DDP
# breaks. `edges = self.net.build_edges(...)` (both CGENN-LGATr wrappers) raises
# AttributeError, and `if hasattr(self.net, "trimmer"): self.net.trimmer.tick()`
# (ParTWrapper) silently evaluates False -- so ParT's SequenceTrimmer would never warm
# up and never trim, changing training behaviour with no error at all. The silent one is
# the dangerous shape, and a crash-only test would have missed it.
# ---------------------------------------------------------------------------------


def test_wrappers_reach_into_net_through_inner_net():
    """STATIC: no wrapper may touch `self.net.<attr>` at RUNTIME without inner_net().

    __init__ is exempt: it runs before DDP wrapping, so `self.net.compile(...)` there is
    fine. Everything else runs after, where `self.net` may be a DDP object that proxies
    nothing.
    """
    import ast as _ast

    path = REPO / "experiments" / "tagging" / "wrappers.py"
    tree = _ast.parse(path.read_text())
    offenders = []
    for cls in [n for n in tree.body if isinstance(n, _ast.ClassDef)]:
        for fn in [n for n in cls.body if isinstance(n, _ast.FunctionDef)]:
            if fn.name == "__init__":
                continue
            for node in _ast.walk(fn):
                if (
                    isinstance(node, _ast.Attribute)
                    and isinstance(node.value, _ast.Attribute)
                    and node.value.attr == "net"
                    and isinstance(node.value.value, _ast.Name)
                    and node.value.value.id == "self"
                ):
                    offenders.append(
                        f"{path.name}:{node.lineno}: {cls.name}.{fn.name} -> self.net.{node.attr}"
                    )
    assert not offenders, (
        "wrapper reaches into self.net outside __init__ without inner_net(). Under DDP "
        "`self.net` is a DistributedDataParallel that proxies NO attribute access, so "
        "this either raises AttributeError or -- worse -- silently no-ops behind a "
        "hasattr() guard. Route it through experiments.tagging.wrappers.inner_net:\n  "
        + "\n  ".join(offenders)
    )


def test_inner_net_sees_through_ddp():
    """FUNCTIONAL: inner_net() actually recovers attributes through a real DDP wrap.

    Pairs with the static test above: that one says "you used the helper", this one says
    "the helper works". Uses a single-process gloo group, so it needs no GPU.
    """
    from experiments.tagging.wrappers import inner_net

    class _Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2, 2)
            self.trimmer = torch.nn.Identity()

        def build_edges(self, *a):
            return "edges"

        def forward(self, x):
            return self.lin(x)

    inner = _Inner()
    assert inner_net(inner) is inner, "inner_net must be a no-op on a bare module"

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = "29591"
    already = torch.distributed.is_initialized()
    if not already:
        torch.distributed.init_process_group("gloo", rank=0, world_size=1)
    try:
        ddp = torch.nn.parallel.DistributedDataParallel(inner)
        # the properties that made the bug possible, asserted so nobody "simplifies"
        # inner_net away on the belief that DDP forwards attributes
        assert not hasattr(ddp, "trimmer"), "DDP started proxying attributes; re-audit"
        assert not hasattr(ddp, "build_edges")
        assert inner_net(ddp) is inner
        assert hasattr(inner_net(ddp), "trimmer")
        assert inner_net(ddp).build_edges() == "edges"
    finally:
        if not already:
            torch.distributed.destroy_process_group()
