"""FLASH PLAN v2, Step 2 gates: the generated Cl(1,3) weighted GP vs the shipped GP.

The generated module is the mathematical content of the future Triton kernels,
verified BEFORE any kernel exists (the plan's risk-concentration mitigation):

TOL-FWD-2   elementwise/dim-2 semantics vs `sparse_gp_expression` (einsum path), fp64
TOL-FWD-3   fc/dim-3 semantics (sum over input features) vs `sparse_gp_expression`
            (the shipped blockdiag path), fp64
TOL-GRAD    generated gradient function vs autograd through the generated forward
CROSS-GRAD  autograd through the generated forward vs autograd through the shipped
            expression (ties the generated backward to the reference arithmetic)
CHECK       gradcheck on the torch wrapper
Skips cleanly when kingdon was never installed? No -- the generated module is
COMMITTED and torch-only; these gates always run. Only regeneration needs kingdon.
"""

import pytest
import torch

from experiments.baselines.cgenn.cliffordalgebra import CliffordAlgebra, sparse_gp_tables
from experiments.baselines.cgenn.flash_ref_p1m3 import wgp, wgp_grads
from experiments.baselines.cgenn.sparse_gp import sparse_gp_expression

torch.set_num_threads(1)
BAR = 1e-13


@pytest.fixture(scope="module")
def tables():
    alg = CliffordAlgebra((1.0, -1.0, -1.0, -1.0)).to(torch.float64)
    pidx = alg.geometric_product_paths.nonzero().T.contiguous()
    spath, spval, _ = sparse_gp_tables(alg, pidx)
    return alg, spath, spval.to(torch.float64)


def _rel(a, b):
    return ((a - b).abs().max() / (1 + b.abs().max())).item()


def test_forward_dim2_matches_expression(tables):
    alg, spath, spval = tables
    torch.manual_seed(0)
    B, N = 7, 11
    x = torch.randn(B, N, 16, dtype=torch.float64)
    y = torch.randn(B, N, 16, dtype=torch.float64)
    weight = torch.randn(N, 35, dtype=torch.float64)
    ref = sparse_gp_expression(x, y, weight, alg.gp_k_idx, spath, spval)
    got = wgp(x, y, weight.expand(B, N, 35))
    rel = _rel(got, ref)
    print(f"GATE-FLASH-FWD2 rel={rel:.3e}")
    assert rel < BAR


def test_forward_dim3_fc_matches_expression(tables):
    alg, spath, spval = tables
    torch.manual_seed(1)
    B, N, M = 5, 7, 6
    x = torch.randn(B, N, 16, dtype=torch.float64)
    y = torch.randn(B, N, 16, dtype=torch.float64)
    weight = torch.randn(M, N, 35, dtype=torch.float64)
    ref = sparse_gp_expression(x, y, weight, alg.gp_k_idx, spath, spval)  # blockdiag path
    got = wgp(x[:, None], y[:, None], weight[None].expand(B, M, N, 35)).sum(dim=2)
    rel = _rel(got, ref)
    print(f"GATE-FLASH-FWD3 rel={rel:.3e}")
    assert rel < BAR


def test_generated_grads_match_autograd(tables):
    torch.manual_seed(2)
    shape = (9, 4)
    x = torch.randn(*shape, 16, dtype=torch.float64, requires_grad=True)
    y = torch.randn(*shape, 16, dtype=torch.float64, requires_grad=True)
    w = torch.randn(*shape, 35, dtype=torch.float64, requires_grad=True)
    go = torch.randn(*shape, 16, dtype=torch.float64)

    (wgp(x, y, w) * go).sum().backward()
    gx, gy, gw = wgp_grads(x.detach(), y.detach(), w.detach(), go)
    for name, got, want in (("dL/dx", gx, x.grad), ("dL/dy", gy, y.grad), ("dL/dw", gw, w.grad)):
        rel = _rel(got, want)
        print(f"GATE-FLASH-GRAD {name} rel={rel:.3e}")
        assert rel < BAR, name


def test_cross_grads_match_expression(tables):
    alg, spath, spval = tables
    torch.manual_seed(3)
    B, N = 6, 5
    x0 = torch.randn(B, N, 16, dtype=torch.float64)
    y0 = torch.randn(B, N, 16, dtype=torch.float64)
    w0 = torch.randn(N, 35, dtype=torch.float64)
    go = torch.randn(B, N, 16, dtype=torch.float64)

    grads = {}
    for name, fn in (
        ("expr", lambda a, b, c: sparse_gp_expression(a, b, c, alg.gp_k_idx, spath, spval)),
        ("gen", lambda a, b, c: wgp(a, b, c.expand(B, N, 35))),
    ):
        a, b, c = (t.clone().requires_grad_(True) for t in (x0, y0, w0))
        (fn(a, b, c) * go).sum().backward()
        grads[name] = (a.grad, b.grad, c.grad)
    for nm, got, want in zip(("dL/dx", "dL/dy", "dL/dw"), grads["gen"], grads["expr"]):
        rel = _rel(got, want)
        print(f"GATE-FLASH-XGRAD {nm} rel={rel:.3e}")
        assert rel < BAR, nm


def test_gradcheck():
    torch.manual_seed(4)
    x = torch.randn(3, 16, dtype=torch.float64, requires_grad=True)
    y = torch.randn(3, 16, dtype=torch.float64, requires_grad=True)
    w = torch.randn(3, 35, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(wgp, (x, y, w))


def test_generated_file_is_current():
    """Regeneration is deterministic-enough to diff: if the committed module drifts
    from what the pinned generator emits, fail loudly (someone edited generated code
    or bumped a dependency without re-running the generator + gates)."""
    kingdon = pytest.importorskip("kingdon")  # regeneration needs kingdon; parity gates above never skip
    import importlib.util
    import io
    from contextlib import redirect_stdout
    from pathlib import Path

    from utils import flash_gen

    committed = Path(flash_gen.OUT_PATH).read_text()
    tmp = Path(flash_gen.OUT_PATH).with_suffix(".regen.tmp")
    orig = flash_gen.OUT_PATH
    try:
        flash_gen.OUT_PATH = str(tmp)
        with redirect_stdout(io.StringIO()):
            flash_gen.main()
        assert tmp.read_text() == committed, (
            "committed flash_ref_p1m3.py != regenerated output -- re-run "
            "utils/flash_gen.py and its gates, or revert the manual edit")
    finally:
        flash_gen.OUT_PATH = orig
        tmp.unlink(missing_ok=True)
