"""FLASH PLAN v2, Step 3 CPU gates: the custom-op WIRING, verified without a GPU.

The CUDA kernels themselves compile only at first launch (GPU gate day covers them);
everything else -- schema, fake tensors, autograd registration, the CPU composite,
and its parity against the shipped GP -- is gated here so the operator's GPU
round-trip can only fail inside kernel code, never in wiring.
"""

import pytest
import torch

from experiments.baselines.cgenn.cliffordalgebra import CliffordAlgebra, sparse_gp_tables
from experiments.baselines.cgenn.flash_kernels_p1m3 import fcgp
from experiments.baselines.cgenn.sparse_gp import sparse_gp_expression

torch.set_num_threads(1)
BAR = 1e-13


@pytest.fixture(scope="module")
def tables():
    alg = CliffordAlgebra((1.0, -1.0, -1.0, -1.0)).to(torch.float64)
    pidx = alg.geometric_product_paths.nonzero().T.contiguous()
    spath, spval, _ = sparse_gp_tables(alg, pidx)
    return alg, spath, spval.to(torch.float64)


def _case(B=6, N=5, M=4, dtype=torch.float64, seed=0):
    torch.manual_seed(seed)
    return (torch.randn(B, N, 16, dtype=dtype), torch.randn(B, N, 16, dtype=dtype),
            torch.randn(M, N, 35, dtype=dtype))


def test_opcheck_cpu():
    """torch.library.opcheck: schema, fake kernel, autograd registration -- the
    machine-checkable parts of custom-op hygiene."""
    x, y, w = _case()
    torch.library.opcheck(fcgp, (x.requires_grad_(True), y.requires_grad_(True),
                                 w.requires_grad_(True)))


def test_cpu_forward_matches_expression(tables):
    alg, spath, spval = tables
    x, y, w = _case(B=7, N=11, M=6)
    ref = sparse_gp_expression(x, y, w, alg.gp_k_idx, spath, spval)
    got = fcgp(x, y, w)
    rel = ((got - ref).abs().max() / (1 + ref.abs().max())).item()
    print(f"GATE-FLASH-OP-FWD rel={rel:.3e}")
    assert rel < BAR


def test_cpu_grads_match_expression(tables):
    alg, spath, spval = tables
    x0, y0, w0 = _case(B=5, N=7, M=4, seed=1)
    go = torch.randn(5, 4, 16, dtype=torch.float64)
    grads = {}
    for name, fn in (
        ("expr", lambda a, b, c: sparse_gp_expression(a, b, c, alg.gp_k_idx, spath, spval)),
        ("op", fcgp),
    ):
        a, b, c = (t.clone().requires_grad_(True) for t in (x0, y0, w0))
        (fn(a, b, c) * go).sum().backward()
        grads[name] = (a.grad, b.grad, c.grad)
    for nm, got, want in zip(("dL/dx", "dL/dy", "dL/dw"), grads["op"], grads["expr"]):
        rel = ((got - want).abs().max() / (1 + want.abs().max())).item()
        print(f"GATE-FLASH-OP-GRAD {nm} rel={rel:.3e}")
        assert rel < BAR, nm


def test_compile_traces_the_op_without_breaks():
    import torch._dynamo as dynamo

    x, y, w = _case(dtype=torch.float32)
    explanation = dynamo.explain(lambda a, b, c: fcgp(a, b, c).square().sum())(x, y, w)
    dynamo.reset()
    assert explanation.graph_break_count == 0, str(explanation)[:1500]


def test_compile_joint_forward_backward():
    """torch.compile through a TRAINING step of the op -- AOT traces the registered
    backward into the joint graph. GPU round-trip #4 caught that a raw Triton launch
    inside the backward dies on fake tensors during that trace; the backward is now
    its own opaque op (fcgp_bwd). This CPU joint-compile pins the WIRING (the
    CUDA-branch-under-fake scenario itself is GPU-tier, covered by
    test_cgenn_compile's compiled-backward gate)."""
    import torch._dynamo as dynamo

    def f(a, b, c):
        return fcgp(a, b, c).square().sum()

    x0, y0, w0 = _case(dtype=torch.float64, seed=2)
    eager = [t.clone().requires_grad_(True) for t in (x0, y0, w0)]
    f(*eager).backward()
    dynamo.reset()
    comp = [t.clone().requires_grad_(True) for t in (x0, y0, w0)]
    torch.compile(f, dynamic=True)(*comp).backward()
    dynamo.reset()
    for nm, a, b in zip(("dL/dx", "dL/dy", "dL/dw"), comp, eager):
        rel = ((a.grad - b.grad).abs().max() / (1 + b.grad.abs().max())).item()
        assert rel < 1e-13, f"compiled joint {nm}: rel={rel:.3e}"


def test_backward_op_has_a_fake_kernel():
    """The bwd op must be traceable with fake tensors (the exact round-trip #4
    failure mode): calling it under FakeTensorMode must hit register_fake, not the
    real body."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        x = torch.empty(6, 5, 16)
        y = torch.empty(6, 5, 16)
        w = torch.empty(4, 5, 35)
        go = torch.empty(6, 4, 16)
        from experiments.baselines.cgenn.flash_kernels_p1m3 import fcgp_bwd

        gx, gy, gw = fcgp_bwd(x, y, w, go)
    assert gx.shape == x.shape and gy.shape == y.shape and gw.shape == w.shape


def test_kernels_present_when_triton_is():
    from experiments.baselines.cgenn import flash_kernels_p1m3 as mod

    if mod._HAS_TRITON:
        assert hasattr(mod, "_fcgp_fwd_kernel") and hasattr(mod, "_fcgp_bwd_kernel")
    else:  # pragma: no cover
        pytest.skip("triton not installed; CPU composite only")


def test_kernel_bodies_use_only_portable_triton_syntax():
    """Tripwire from GPU round-trips #1-2: the NGC container's Triton frontend has no
    tuple()/comprehension/generator support inside @jit, and module globals must be
    constexpr. Pin the kernels to the fully-explicit flash-clifford style: no tuple()
    calls, no comprehensions, no starred call args, no bare NB/NP global reads. The
    one non-explicit pattern allowed is the tuple RETURN of the generated bodies
    (flash-kingdon's proven trick), which lives in the generated module, not here."""
    import ast
    from pathlib import Path

    src = Path("experiments/baselines/cgenn/flash_kernels_p1m3.py").read_text()
    tree = ast.parse(src)
    kernels = [n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name.startswith("_fcgp")]
    assert len(kernels) == 2
    for fn in kernels:
        arg_names = {a.arg for a in fn.args.args}
        for node in ast.walk(fn):
            assert not isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp,
                                         ast.DictComp)), f"{fn.name}: comprehension"
            assert not isinstance(node, ast.Starred), f"{fn.name}: starred arg"
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in ("tuple", "list", "dict", "zip", "map"), (
                    f"{fn.name}: python builtin {node.func.id}() inside @jit")
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                assert not (node.id in ("NB", "NP") and node.id not in arg_names), (
                    f"{fn.name}: bare global {node.id} inside @jit")
