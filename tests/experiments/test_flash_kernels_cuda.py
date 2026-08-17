"""FLASH PLAN v2, Step 3 GPU gates: the Triton fcgp kernels on the campaign card.

Run (inside the container, repo venv active):

    python -m pytest tests/experiments/test_flash_kernels_cuda.py -q -s

Skips without CUDA. Gates: forward + backward parity vs the shipped
`sparse_gp_expression` at fp64 (TOL 1e-13) and fp32 (1e-5); run-to-run DETERMINISM
of forward and all three gradients (bit-equal repeat calls -- the property the
two-stage weight-grad reduction exists for, where flash-clifford's atomic_add
reference is nondeterministic); and a CUDA-event microbenchmark vs the shipped
blockdiag contraction at the campaign shapes, printed for the step-4 race record
(pinned matmul precision -- the sparse_gp_race lesson).
"""

import pytest
import torch

from experiments.baselines.cgenn.cliffordalgebra import CliffordAlgebra, sparse_gp_tables
from experiments.baselines.cgenn.flash_kernels_p1m3 import fcgp
from experiments.baselines.cgenn.sparse_gp import sparse_gp_expression

if not torch.cuda.is_available():
    pytest.skip("CUDA required (GPU gate day)", allow_module_level=True)

torch.set_float32_matmul_precision("highest")


@pytest.fixture(scope="module")
def tables64():
    # tables are built on CPU (exactly as model __init__ does) and the three tensors
    # moved -- sparse_gp_tables itself is CPU-only by design (GPU round-trip finding)
    alg = CliffordAlgebra((1.0, -1.0, -1.0, -1.0)).to(torch.float64)
    pidx = alg.geometric_product_paths.nonzero().T.contiguous()
    spath, spval, _ = sparse_gp_tables(alg, pidx)
    return alg.gp_k_idx.cuda(), spath.cuda(), spval.to(torch.float64).cuda()


def _case(B, N, M, dtype, seed=0):
    torch.manual_seed(seed)
    mk = lambda *s: torch.randn(*s, dtype=dtype, device="cuda")
    return mk(B, N, 16), mk(B, N, 16), mk(M, N, 35)


@pytest.mark.parametrize("dtype,bar", [(torch.float64, 1e-13), (torch.float32, 1e-5)])
def test_parity_forward_and_grads(tables64, dtype, bar):
    kidx, spath, spval = tables64
    spval_d = spval.to(dtype)
    x0, y0, w0 = _case(37, 11, 6, dtype)
    go = torch.randn(37, 6, 16, dtype=dtype, device="cuda")
    grads = {}
    for name, fn in (
        ("expr", lambda a, b, c: sparse_gp_expression(a, b, c, kidx, spath, spval_d)),
        ("op", fcgp),
    ):
        a, b, c = (t.clone().requires_grad_(True) for t in (x0, y0, w0))
        out = fn(a, b, c)
        (out * go).sum().backward()
        grads[name] = (out.detach(), a.grad, b.grad, c.grad)
    for nm, got, want in zip(("fwd", "dL/dx", "dL/dy", "dL/dw"), grads["op"], grads["expr"]):
        rel = ((got - want).abs().max() / (1 + want.abs().max())).item()
        print(f"GATE-FLASH-CUDA[{dtype}] {nm} rel={rel:.3e}")
        assert rel < bar, nm


def test_run_to_run_determinism():
    x0, y0, w0 = _case(4096, 19, 11, torch.float32, seed=3)
    go = torch.randn(4096, 11, 16, dtype=torch.float32, device="cuda")

    def once():
        a, b, c = (t.clone().requires_grad_(True) for t in (x0, y0, w0))
        out = fcgp(a, b, c)
        (out * go).sum().backward()
        return out.detach().clone(), a.grad.clone(), b.grad.clone(), c.grad.clone()

    r1, r2 = once(), once()
    for nm, t1, t2 in zip(("fwd", "dL/dx", "dL/dy", "dL/dw"), r1, r2):
        assert torch.equal(t1, t2), f"{nm} differs across identical runs -- NOT deterministic"
    print("GATE-FLASH-DET forward + all three gradients bit-equal across runs")


@pytest.mark.parametrize("B,N,M", [(13000, 11, 11), (200000, 11, 11), (200000, 19, 11)])
def test_benchmark_vs_blockdiag(tables64, B, N, M):
    kidx, spath, spval = tables64
    spval32 = spval.to(torch.float32)
    x0, y0, w0 = _case(B, N, M, torch.float32, seed=4)

    def bench(fn, reps=30):
        for _ in range(5):
            a, b, c = (t.clone().requires_grad_(True) for t in (x0, y0, w0))
            fn(a, b, c).sum().backward()
        torch.cuda.synchronize()
        t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
        t0.record()
        for _ in range(reps):
            a, b, c = (t.clone().requires_grad_(True) for t in (x0, y0, w0))
            fn(a, b, c).sum().backward()
        t1.record()
        torch.cuda.synchronize()
        return t0.elapsed_time(t1) / reps

    t_ref = bench(lambda a, b, c: sparse_gp_expression(a, b, c, alg.gp_k_idx, spath32, spval32))
    t_op = bench(fcgp)
    print(f"BENCH-FLASH B={B} N={N} M={M}: blockdiag {t_ref:.2f} ms  flash {t_op:.2f} ms  "
          f"ratio {t_op / t_ref:.2f}x  (adopt bar per race discipline: < 0.90x everywhere)")
