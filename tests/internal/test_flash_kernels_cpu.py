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


def _kernel_jit_deps():
    """(kernel, {co_name: JITFunction}) for both kernels; asserts each has >= 1 dep
    so the binding-name gates below can never pass vacuously."""
    from experiments.baselines.cgenn import flash_kernels_p1m3 as mod

    if not mod._HAS_TRITON:
        pytest.skip("triton not installed; CPU composite only")
    from triton.runtime.jit import JITFunction

    out = []
    for kernel in (mod._fcgp_fwd_kernel, mod._fcgp_bwd_kernel):
        deps = {name: kernel.fn.__globals__[name]
                for name in kernel.fn.__code__.co_names
                if isinstance(kernel.fn.__globals__.get(name), JITFunction)}
        assert deps, f"{kernel.fn.__name__}: expected a generated-body subfunction call"
        out.append((kernel, deps))
    return out


def test_jit_subfunction_bindings_match_their_def_names():
    """GPU round-trip #6 rule: a JITFunction a kernel calls must be bound under the
    wrapped function's OWN __name__. Inductor's user-defined-kernel embedding re-emits
    each dependency's `src` -- whose def line carries the original name -- and its
    JITFunction branch (unlike ConstexprFunction) writes no alias for a mismatched
    binding, so `_fwd_body = triton.jit(_wgp_fwd)` compiles at raw launch but dies at
    ast_to_ttir (NameError) the moment inductor sees the kernel via triton_op. CUDA-only
    failure mode, machine-checked here on CPU."""
    for kernel, deps in _kernel_jit_deps():
        for name, jitfn in deps.items():
            assert name == jitfn.fn.__name__, (
                f"{kernel.fn.__name__} calls '{name}' but the emitted def is "
                f"'{jitfn.fn.__name__}' -- inductor's closure embedding will NameError")


def test_inductor_closure_embedding_resolves_all_calls():
    """The same rule, end-to-end through torch's actual emitter: the transitive-closure
    source inductor would generate for each kernel must contain a def for every
    JITFunction name the kernel calls (this is the exact text ast_to_ttir re-parses in
    the compile worker)."""
    import ast as ast_mod

    try:  # moved between releases: _inductor.utils (2.8 era) -> codegen.wrapper (2.13)
        from torch._inductor.codegen.wrapper import (
            user_defined_triton_kernel_transitive_closure_source_code as closure_src,
        )
    except ImportError:
        try:
            from torch._inductor.utils import (
                user_defined_triton_kernel_transitive_closure_source_code as closure_src,
            )
        except ImportError:
            pytest.skip("inductor closure emitter not importable on this torch")

    for kernel, deps in _kernel_jit_deps():
        src = closure_src(kernel)
        defs = {n.name for n in ast_mod.walk(ast_mod.parse(src))
                if isinstance(n, ast_mod.FunctionDef)}
        missing = set(deps) - defs
        assert not missing, (
            f"{kernel.fn.__name__}: emitted closure lacks defs for {sorted(missing)}; "
            f"defs present: {sorted(defs)}")


def test_launch_cfg_env_hook(monkeypatch):
    """The round-trip #8 tuning hook: unset env == exactly the old launch (no
    warps/stages kwargs at all -- pinning triton's version-dependent defaults
    explicitly would change the schedule on an upgrade); set env parses and
    validates; garbage refuses loudly instead of silently launching a default."""
    from experiments.baselines.cgenn import flash_kernels_p1m3 as fk

    monkeypatch.delenv("CGENN_FLASH_FWD_CFG", raising=False)
    assert fk._launch_cfg("CGENN_FLASH_FWD_CFG", 64, None, None) == (64, None, None)
    assert fk._launch_kwargs(None, None) == {}

    monkeypatch.setenv("CGENN_FLASH_FWD_CFG", "128,8,2")
    assert fk._launch_cfg("CGENN_FLASH_FWD_CFG", 64, None, None) == (128, 8, 2)
    assert fk._launch_kwargs(8, 2) == {"num_warps": 8, "num_stages": 2}

    for bad in ("128", "0,4,2", "96,4,2", "64,3,2", "64,4,0", "a,b,c"):
        monkeypatch.setenv("CGENN_FLASH_FWD_CFG", bad)
        with pytest.raises(ValueError):
            fk._launch_cfg("CGENN_FLASH_FWD_CFG", 64, None, None)


def test_flash_tune_grid_and_parse():
    """flash_tune must at least import and expose a sane sweep grid on CPU (the
    sweep itself is CUDA-only and exits saying so)."""
    import utils.flash_tune as ft
    with pytest.raises(SystemExit) as exc:
        ft.main([])
    assert "CUDA" in str(exc.value)
