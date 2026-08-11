"""Gates for `experiments/baselines/cgenn/sparse_gp.py` — the sparse geometric product's
custom `torch.autograd.Function`.

KEEP-permanently, and self-contained on purpose. The subject is shipped model code with a
HAND-WRITTEN backward: unlike the rest of the CGENN compile work, its gradients are not a
derivative of a gated forward, so nothing else in the tree can catch a wrong one. The
model-level gates live in `test_cgenn_compile.py`, which `cleanup.md` schedules for
deletion as a port instrument — the same "permanent guards and port instruments in one
file" mistake that `test_compile_posture.py` was carved out of. So these build their own
algebra, need no fixtures, no hydra and no dataset, and run in about six seconds.

Each gate compares against the exact three-line expression the Function replaced:

    pair = x.unsqueeze(-1) * y[..., gp_k_idx]
    w    = weight[..., sp_path] * sp_val
    out  = einsum("bnij,nij->bnj", pair, w)          # or "bnij,mnij->bmj" for the fc layer

BIT     forward is bit-identical (torch.equal) — the premise of the whole change
GRAD    gradients agree with plain autograd through that expression to <= 1e-13 (fp64)
CHECK   gradcheck vs a numeric jacobian, and gradgradcheck (backward is differentiable)
SAVED   the Function retains < 1/8 of what the expression retained — its reason to exist
MASKED  all of the above again on a path set that actually exercises `sp_val == 0`, the
        one branch the shipped Lorentz algebra never reaches

and one that stands alone, because nothing else in the tree measures it:

RECOMP-BWD  a compiled TRAINING step must not re-specialize per batch shape. Every other
        RECOMP measurement here runs under `no_grad` and so never sees the joint graph.
"""

import pytest
import torch

torch.set_num_threads(1)  # run-context-independent arithmetic (same lesson as lgatr parity)

NAMES = {False: "gp", True: "fcgp"}


def _case(fc, B=7, N=16, M=16, dtype=torch.float64):
    """(function under test, eager equivalent, inputs) for one of the two layer shapes."""
    from experiments.baselines.cgenn.cliffordalgebra import CliffordAlgebra, sparse_gp_tables
    from experiments.baselines.cgenn.sparse_gp import sparse_geometric_product

    alg = CliffordAlgebra((1.0, -1.0, -1.0, -1.0)).to(dtype)
    pidx = alg.geometric_product_paths.nonzero().T.contiguous()
    spath, spval, sel = (t.to(dtype) if t.is_floating_point() else t
                         for t in sparse_gp_tables(alg, pidx))
    nb, P = alg.n_blades, pidx.shape[1]
    torch.manual_seed(0)
    args = (torch.randn(B, N, nb, dtype=dtype), torch.randn(B, N, nb, dtype=dtype),
            torch.randn(*((M, N, P) if fc else (N, P)), dtype=dtype))

    def under_test(x, y, w):
        return sparse_geometric_product(x, y, w, alg, spath, spval, sel)

    def eager(x, y, w):
        pair = x.unsqueeze(-1) * y[..., alg.gp_k_idx]
        if fc:
            return torch.einsum("bnij,mnij->bmj", pair, w[:, :, spath] * spval)
        return torch.einsum("bnij,nij->bnj", pair, w[:, spath] * spval)

    return under_test, eager, args


@pytest.mark.parametrize("fc", [False, True], ids=["gp", "fcgp"])
def test_gradcheck(fc):
    """CHECK: the hand-written backward vs a numeric jacobian, fp64.

    gradgradcheck as well, which is not ceremony: it is what proves the backward is itself
    composed of differentiable ops (no in-place fold onto a saved tensor, no detach), i.e.
    that anything downstream needing double backward still works. lgatr's equivalent
    folds signs in-place and documents why that is safe; this one has no in-place op at
    all, and this gate is how that stays true.
    """
    fn, _, args = _case(fc, B=2, N=3, M=4)
    args = tuple(a.requires_grad_(True) for a in args)
    assert torch.autograd.gradcheck(fn, args, raise_exception=True)
    assert torch.autograd.gradgradcheck(fn, args, raise_exception=True)
    print(f"GATE-GRADCHECK[{NAMES[fc]}] analytic == numeric, double-backward OK")


@pytest.mark.parametrize("fc", [False, True], ids=["gp", "fcgp"])
def test_matches_the_expression_it_replaced(fc):
    """BIT + GRAD: same forward BITS, same gradients as plain autograd.

    The sharp instrument for this change. The model-level gates are integration-scale (the
    model's own gradient conditioning costs ~6 digits before anything of ours is reached);
    here the comparison is against the exact expression the Function replaced, so the
    forward must be bit-zero and the gradients roundoff-scale.
    """
    fn, eager, args = _case(fc)
    assert torch.equal(fn(*args), eager(*args)), (
        f"{NAMES[fc]}: forward is NOT bit-identical to the expression it replaced. That "
        f"identity is the premise of the change -- the einsum is meant to move INSIDE the "
        f"Function unchanged, so only what autograd retains differs.")
    g = torch.randn(eager(*args).shape, dtype=torch.float64)
    out = {}
    for name, f in (("eager", eager), ("Function", fn)):
        a = tuple(t.clone().requires_grad_(True) for t in args)
        f(*a).backward(g)
        out[name] = [t.grad for t in a]
    for nm, got, want in zip(("dL/dx", "dL/dy", "dL/dw"), out["Function"], out["eager"]):
        rel = ((got - want).abs().max() / (1 + want.abs().max())).item()
        print(f"GATE-GRAD[{NAMES[fc]}] {nm} rel={rel:.3e}")
        assert rel < 1e-13, f"{nm}: {rel:.3e} >= 1e-13 -- backward disagrees with autograd"


@pytest.mark.parametrize("fc", [False, True], ids=["gp", "fcgp"])
def test_retains_less_than_the_expression_it_replaced(fc):
    """SAVED: the reason the Function exists, as a gate.

    The eager form saves TWO (B, N, 16, 16) tensors for backward -- `y[..., gp_k_idx]`,
    which the mul needs, and `pair`, which the einsum needs. That is why the sparse path
    did 16x fewer MACs than the dense forms and still measured a 1.50x HIGHER GPU peak: a
    memory regression no flops-shaped gate can see, which is how it reached the campaign
    posture. The Function saves only its inputs, two (B, N, 16) tensors.

    Bar is 1/8 against a measured ~1/16 -- tight enough that reinstating even ONE of the
    two retained intermediates fails, loose enough not to trip on bookkeeping.
    """
    fn, eager, args = _case(fc, B=512, dtype=torch.float32)
    args = tuple(a.requires_grad_(True) for a in args)
    saved = {}
    for name, f in (("eager", eager), ("Function", fn)):
        held = {}

        def pack(t, held=held):
            held[id(t)] = t.numel() * t.element_size()
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            f(*args)
        saved[name] = sum(held.values())
        print(f"GATE-SAVED[{NAMES[fc]}] {name:8s} {saved[name] / 2**20:8.3f} MB retained")
    ratio = saved["Function"] / saved["eager"]
    print(f"GATE-SAVED[{NAMES[fc]}] Function/eager = {ratio:.4f}")
    assert ratio < 0.125, (
        f"{NAMES[fc]}: the Function retains {ratio:.3f}x the eager expression. It exists "
        f"to retain ~1/16 -- something is being saved for backward again.")


@pytest.mark.parametrize("fc", [False, True], ids=["gp", "fcgp"])
def test_masked_grade_paths(fc):
    """The one branch the shipped algebra never exercises: `sp_val == 0`.

    `sparse_gp_tables` zeroes the weight for any (i, j) whose grade triple `product_paths`
    masks, and CLAMPS its path index to 0 so the gather stays in bounds. Under the Lorentz
    metric every one of the 256 blade pairs lands on an allowed triple, so no entry is ever
    masked and that clamp is dead code in production -- which is exactly the kind of branch
    that is wrong when someone finally needs it (the failure mode is the clamped entries
    dumping their gradient onto path 0, silently, in dL/dweight).

    So this builds a DELIBERATELY reduced path set, keeping every other allowed triple, and
    demands the same two things as the unmasked case: bit-identical forward and
    autograd-matching gradients.
    """
    from experiments.baselines.cgenn.cliffordalgebra import CliffordAlgebra, sparse_gp_tables
    from experiments.baselines.cgenn.sparse_gp import sparse_geometric_product

    dtype = torch.float64
    alg = CliffordAlgebra((1.0, -1.0, -1.0, -1.0)).to(dtype)
    full = alg.geometric_product_paths.nonzero().T.contiguous()
    pidx = full[:, ::2].contiguous()  # drop every other grade path
    spath, spval, sel = (t.to(dtype) if t.is_floating_point() else t
                         for t in sparse_gp_tables(alg, pidx))
    assert (spval == 0).any(), "the reduced path set masked nothing -- test is vacuous"
    assert (spath[spval == 0] == 0).all(), "masked entries should carry the clamped index 0"
    assert (spath[spval != 0] == 0).any(), (
        "no LIVE entry shares the clamped index 0, so a clobber would not be visible here")

    B, N, M, nb, P = 7, 16, 16, alg.n_blades, pidx.shape[1]
    torch.manual_seed(0)
    args = tuple(t.requires_grad_(True) for t in (
        torch.randn(B, N, nb, dtype=dtype), torch.randn(B, N, nb, dtype=dtype),
        torch.randn(*((M, N, P) if fc else (N, P)), dtype=dtype)))

    def eager(x, y, w):
        pair = x.unsqueeze(-1) * y[..., alg.gp_k_idx]
        if fc:
            return torch.einsum("bnij,mnij->bmj", pair, w[:, :, spath] * spval)
        return torch.einsum("bnij,nij->bnj", pair, w[:, spath] * spval)

    def fn(x, y, w):
        return sparse_geometric_product(x, y, w, alg, spath, spval, sel)

    assert torch.equal(fn(*args), eager(*args)), f"{NAMES[fc]}: masked forward differs"
    g = torch.randn(eager(*args).shape, dtype=dtype)
    out = {}
    for name, f in (("eager", eager), ("Function", fn)):
        a = tuple(t.detach().clone().requires_grad_(True) for t in args)
        f(*a).backward(g)
        out[name] = [t.grad for t in a]
    for nm, got, want in zip(("dL/dx", "dL/dy", "dL/dw"), out["Function"], out["eager"]):
        rel = ((got - want).abs().max() / (1 + want.abs().max())).item()
        print(f"GATE-MASKED[{NAMES[fc]}] {nm} rel={rel:.3e}")
        assert rel < 1e-13, f"masked {nm}: {rel:.3e} -- clamped paths leak into the gradient"


@pytest.mark.parametrize("fc", [False, True], ids=["gp", "fcgp"])
def test_compiled_training_step_does_not_respecialize(fc):
    """RECOMP-BACKWARD: a compiled TRAINING step must not recompile per batch shape.

    The gate that did not exist, and the bug it would have caught is the reason it does.
    Every RECOMP measurement in this tree runs under `no_grad`, so none of them sees the
    JOINT graph -- and the first version of this backward used two 3-operand einsums, which
    hand their contraction path to opt_einsum, whose search reads concrete sizes. Measured:
    1, 2, 3, 4 unique graphs over four distinct batch shapes, against 1 for `gp_impl=einsum`
    on the same sweep. Jet multiplicity varies every batch, so that is a recompile per step
    for the whole campaign, and every existing gate was green.

    The specs in sparse_gp.py are all binary now. This is what keeps them that way.
    """
    import torch._dynamo as dynamo
    from torch._dynamo.utils import counters

    fn, _, (x0, y0, w0) = _case(fc, B=4, dtype=torch.float32)
    w = w0.clone().requires_grad_(True)

    def step(B):
        torch.manual_seed(B)
        a = torch.randn(B, *x0.shape[1:]).requires_grad_(True)
        b = torch.randn(B, *y0.shape[1:]).requires_grad_(True)
        compiled(a, b, w).square().sum().backward()

    dynamo.reset()
    counters.clear()
    compiled = torch.compile(fn, dynamic=True)
    seen = []
    for B in (4, 7, 11, 13):
        step(B)
        seen.append(counters["stats"]["unique_graphs"])
    print(f"GATE-RECOMP-BWD[{NAMES[fc]}] unique_graphs per shape: {seen}")
    dynamo.reset()
    assert seen[-1] <= 2, (
        f"{NAMES[fc]}: {seen} -- the compiled training step re-specializes per batch shape. "
        f"Check for an einsum with THREE OR MORE operands in sparse_gp.py: opt_einsum's path "
        f"search reads concrete sizes and pins the graph to them.")


@pytest.mark.parametrize("fc", [False, True], ids=["gp", "fcgp"])
def test_compiled_path_bypasses_the_function(fc):
    """The Function is EAGER-ONLY, and this is what keeps it that way.

    Measured at model level (tag_cgenn, fp64, 5 warm-up + 10 timed): eager, the Function is
    0.88x the time and 0.17x the retained memory of the expression -- a clear win. Compiled,
    it is 1.84x the time and 1.00x the memory -- a pure loss, because AOTAutograd's
    partitioner already reaches that retention on its own and the Function only adds its
    recompute on top. `tag_cgenn.yaml` ships `compile: true`, so the compiled row is the one
    the campaign runs; `sparse_geometric_product` therefore branches on
    `torch.compiler.is_compiling()`.

    That branch is one line and deleting it looks like a simplification. It is worth 1.84x
    on the shipped posture, and NO other gate here can see it: every one of them either runs
    eager, or checks a property (gradients, retention, graph count) that both paths satisfy.
    """
    import torch._dynamo as dynamo

    fn, eager, args = _case(fc, B=6, N=8, M=6, dtype=torch.float32)
    from experiments.baselines.cgenn import sparse_gp

    calls = []
    real = sparse_gp.SparseGeometricProduct.forward

    def counting(*a):
        calls.append(1)
        return real(*a)

    sparse_gp.SparseGeometricProduct.forward = staticmethod(counting)
    try:
        fn(*args)
        assert calls, "eager call did NOT enter the Function -- the branch is inverted"
        eager_calls = len(calls)
        dynamo.reset()
        compiled = torch.compile(fn, dynamic=True)
        out_c = compiled(*args)
        assert len(calls) == eager_calls, (
            f"{NAMES[fc]}: the COMPILED path entered SparseGeometricProduct "
            f"({len(calls) - eager_calls} calls). It must not -- the Function is 1.84x "
            f"slower than the plain expression once AOT partitions the joint graph.")
    finally:
        sparse_gp.SparseGeometricProduct.forward = staticmethod(real)
        dynamo.reset()
    assert torch.equal(out_c, eager(*args)), (
        f"{NAMES[fc]}: compiled output differs from the eager expression")
    print(f"GATE-EAGERONLY[{NAMES[fc]}] compiled path skips the Function, output bit-equal")
