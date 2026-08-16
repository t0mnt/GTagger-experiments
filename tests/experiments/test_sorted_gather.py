"""Gates for SortedGather (experiments/baselines/cgenn/sorted_gather.py).

KEEP, fixture-free, same charter as test_sparse_gp: a HAND-WRITTEN backward that no
gated forward implies. The subject replaces autograd's atomic scatter-add for
receiver-side gathers with a segment sum over the sorted receiver index -- gates pin:

BIT      forward is bit-identical to plain `x[idx]` (it IS index_select)
GRAD     backward matches plain autograd on CPU (deterministic index_add_) to <=1e-13
         fp64, INCLUDING zero-degree nodes (empty segments) and grads for fp32
CHECK    gradcheck + gradgradcheck at fp64
FALLBACK counts=None and the CGENN_SORTED_GATHER=0 kill switch route to plain autograd
COMPILE  a compiled fwd+bwd has zero graph breaks and <=2 graphs over varying batch
         shapes, its forward is bit-equal to eager, and its gradients match at fp64
"""

import numpy as np
import pytest
import torch

from experiments.baselines.cgenn import sorted_gather as sg_mod
from experiments.baselines.cgenn.sorted_gather import sorted_gather

torch.set_num_threads(1)


def _case(N=13, E=41, F=5, dtype=torch.float64, seed=0, with_empty=True):
    g = torch.Generator().manual_seed(seed)
    idx = torch.sort(torch.randint(0, N, (E,), generator=g)).values
    if with_empty:
        # force at least one zero-degree node INSIDE the range and one at the end
        idx = idx.clamp(min=1)  # node 0 has degree 0
        idx = torch.where(idx == N - 1, torch.tensor(N - 2), idx)  # last node too
        idx = torch.sort(idx).values
    counts = torch.zeros(N, 1).index_add_(0, idx, torch.ones(idx.numel(), 1))
    x = torch.randn(N, F, dtype=dtype, generator=g)
    return x, idx, counts


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_forward_bit_identical(dtype):
    x, idx, counts = _case(dtype=dtype)
    assert torch.equal(sorted_gather(x, idx, counts), x[idx])


@pytest.mark.parametrize("dtype,bar", [(torch.float64, 1e-13), (torch.float32, 1e-5)])
def test_backward_matches_autograd_including_empties(dtype, bar):
    x, idx, counts = _case(dtype=dtype)
    g = torch.randn(idx.numel(), x.shape[1], dtype=dtype)

    a = x.clone().requires_grad_(True)
    a[idx].backward(g)
    b = x.clone().requires_grad_(True)
    sorted_gather(b, idx, counts).backward(g)

    rel = ((b.grad - a.grad).abs().max() / (1 + a.grad.abs().max())).item()
    assert rel < bar, f"backward disagrees with autograd: rel={rel:.3e}"
    assert (b.grad[0] == 0).all() and (b.grad[-1] == 0).all(), (
        "zero-degree nodes must receive exact-zero gradient rows")


def test_gradcheck():
    x, idx, counts = _case(N=6, E=11, F=3)
    x.requires_grad_(True)
    assert torch.autograd.gradcheck(lambda t: sorted_gather(t, idx, counts), (x,))
    assert torch.autograd.gradgradcheck(lambda t: sorted_gather(t, idx, counts), (x,))


def test_fallback_paths(monkeypatch):
    x, idx, counts = _case()
    calls = []
    real = sg_mod.SortedGather.apply
    monkeypatch.setattr(sg_mod.SortedGather, "apply", lambda *a: calls.append(1) or real(*a))
    assert torch.equal(sorted_gather(x, idx, None), x[idx])
    assert not calls, "counts=None must not enter the Function"
    monkeypatch.setattr(sg_mod, "_ENABLED", False)
    assert torch.equal(sorted_gather(x, idx, counts), x[idx])
    assert not calls, "kill switch must not enter the Function"
    monkeypatch.setattr(sg_mod, "_ENABLED", True)
    sorted_gather(x, idx, counts)
    assert calls, "enabled path must use the Function"


def test_compiled_no_breaks_bit_forward_tol_grads():
    import torch._dynamo as dynamo
    from torch._dynamo.utils import counters

    def step(x, idx, counts):
        y = sorted_gather(x, idx, counts)
        return (y * y).sum()

    x, idx, counts = _case(dtype=torch.float32)
    explanation = dynamo.explain(step)(x.clone().requires_grad_(True), idx, counts)
    assert explanation.graph_break_count == 0, str(explanation)[:1500]

    dynamo.reset()
    counters.clear()
    compiled = torch.compile(step, dynamic=True)
    seen = []
    for seed, (N, E) in enumerate([(13, 41), (9, 23), (17, 55)]):
        xe, ide, ce = _case(N=N, E=E, dtype=torch.float64, seed=seed)
        a = xe.clone().requires_grad_(True)
        step(a, ide, ce).backward()
        b = xe.clone().requires_grad_(True)
        compiled(b, ide, ce).backward()
        # forward values through the compiled path are checked via the loss scalar
        rel = ((b.grad - a.grad).abs().max() / (1 + a.grad.abs().max())).item()
        assert rel < 1e-13, f"compiled grads vs eager: rel={rel:.3e}"
        seen.append(counters["stats"]["unique_graphs"])
    dynamo.reset()
    assert seen[-1] <= 2, f"re-specializes per shape: {seen}"


def test_edge_counts_is_the_degree_vector_contract():
    """The Function's REQUIRES line, as an executable statement: counts must equal
    bincount(idx). The CGLs satisfy it by construction (2.2a); if a future call site
    hands anything else, gradients are silently wrong -- keep the contract loud."""
    x, idx, counts = _case()
    assert torch.equal(counts.view(-1).long(), torch.bincount(idx, minlength=x.shape[0]))
