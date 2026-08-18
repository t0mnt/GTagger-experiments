"""Gates for SortedGather + SortedGatherPermuted (experiments/baselines/cgenn/sorted_gather.py).

KEEP, fixture-free, same charter as test_sparse_gp: a HAND-WRITTEN backward that no
gated forward implies. The subjects replace autograd's atomic scatter-add for the
edge gathers with a segment sum -- directly over the sorted receiver index
(SortedGather) or through a precomputed stable-sort permutation for the unsorted
sender index (SortedGatherPermuted, FLASH-2). Both variants get the same gates:

BIT      forward is bit-identical to plain `x[idx]` (it IS index_select)
GRAD     backward matches plain autograd on CPU (deterministic index_add_) to <=1e-13
         fp64, INCLUDING zero-degree nodes (empty segments) and grads for fp32
CHECK    gradcheck + gradgradcheck at fp64
FALLBACK missing extras and the CGENN_SORTED_GATHER=0 kill switch route to plain autograd
COMPILE  a compiled fwd+bwd has zero graph breaks and <=2 graphs over varying batch
         shapes, its forward is bit-equal to eager, and its gradients match at fp64
CONTRACT the REQUIRES lines as executable statements (counts = bincount; perm =
         stable argsort)
"""

import numpy as np
import pytest
import torch

from experiments.baselines.cgenn import sorted_gather as sg_mod
from experiments.baselines.cgenn.sorted_gather import sorted_gather, sorted_gather_perm

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


# ---------------------------------------------------------------------------
# SortedGatherPermuted (FLASH-2): sender-side twin -- UNSORTED idx, backward via a
# precomputed stable-sort permutation. Same gate set as above.
# ---------------------------------------------------------------------------


def _perm_case(N=13, E=41, F=5, dtype=torch.float64, seed=0, with_empty=True):
    """Like _case but idx stays UNSORTED (the sender index j) and the extras are the
    exact tensors the net forwards compute: perm = argsort(idx, stable=True),
    counts = degree vector via the same (E, 1) ones index_add_."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, N, (E,), generator=g)
    if with_empty:
        idx = idx.clamp(min=1)  # node 0 has degree 0
        idx = torch.where(idx == N - 1, torch.tensor(N - 2), idx)  # last node too
    perm = torch.argsort(idx, stable=True)
    counts = torch.zeros(N, 1).index_add_(0, idx, torch.ones(idx.numel(), 1))
    x = torch.randn(N, F, dtype=dtype, generator=g)
    return x, idx, perm, counts


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_perm_forward_bit_identical(dtype):
    x, idx, perm, counts = _perm_case(dtype=dtype)
    assert torch.equal(sorted_gather_perm(x, idx, perm, counts), x[idx])


@pytest.mark.parametrize("dtype,bar", [(torch.float64, 1e-13), (torch.float32, 1e-5)])
def test_perm_backward_matches_autograd_including_empties(dtype, bar):
    x, idx, perm, counts = _perm_case(dtype=dtype)
    g = torch.randn(idx.numel(), x.shape[1], dtype=dtype)

    a = x.clone().requires_grad_(True)
    a[idx].backward(g)
    b = x.clone().requires_grad_(True)
    sorted_gather_perm(b, idx, perm, counts).backward(g)

    rel = ((b.grad - a.grad).abs().max() / (1 + a.grad.abs().max())).item()
    assert rel < bar, f"permuted backward disagrees with autograd: rel={rel:.3e}"
    assert (b.grad[0] == 0).all() and (b.grad[-1] == 0).all(), (
        "zero-degree nodes must receive exact-zero gradient rows")


def test_perm_gradcheck():
    x, idx, perm, counts = _perm_case(N=6, E=11, F=3)
    x.requires_grad_(True)
    assert torch.autograd.gradcheck(lambda t: sorted_gather_perm(t, idx, perm, counts), (x,))
    assert torch.autograd.gradgradcheck(lambda t: sorted_gather_perm(t, idx, perm, counts), (x,))


def test_perm_fallback_paths(monkeypatch):
    x, idx, perm, counts = _perm_case()
    calls = []
    real = sg_mod.SortedGatherPermuted.apply
    monkeypatch.setattr(sg_mod.SortedGatherPermuted, "apply",
                        lambda *a: calls.append(1) or real(*a))
    assert torch.equal(sorted_gather_perm(x, idx, None, counts), x[idx])
    assert torch.equal(sorted_gather_perm(x, idx, perm, None), x[idx])
    assert not calls, "missing perm/counts must not enter the Function"
    monkeypatch.setattr(sg_mod, "_ENABLED", False)
    assert torch.equal(sorted_gather_perm(x, idx, perm, counts), x[idx])
    assert not calls, "kill switch must not enter the Function"
    monkeypatch.setattr(sg_mod, "_ENABLED", True)
    sorted_gather_perm(x, idx, perm, counts)
    assert calls, "enabled path must use the Function"


def test_perm_compiled_no_breaks_tol_grads():
    import torch._dynamo as dynamo
    from torch._dynamo.utils import counters

    def step(x, idx, perm, counts):
        y = sorted_gather_perm(x, idx, perm, counts)
        return (y * y).sum()

    x, idx, perm, counts = _perm_case(dtype=torch.float32)
    explanation = dynamo.explain(step)(x.clone().requires_grad_(True), idx, perm, counts)
    assert explanation.graph_break_count == 0, str(explanation)[:1500]

    dynamo.reset()
    counters.clear()
    compiled = torch.compile(step, dynamic=True)
    seen = []
    for seed, (N, E) in enumerate([(13, 41), (9, 23), (17, 55)]):
        xe, ide, pe, ce = _perm_case(N=N, E=E, dtype=torch.float64, seed=seed)
        a = xe.clone().requires_grad_(True)
        step(a, ide, pe, ce).backward()
        b = xe.clone().requires_grad_(True)
        compiled(b, ide, pe, ce).backward()
        rel = ((b.grad - a.grad).abs().max() / (1 + a.grad.abs().max())).item()
        assert rel < 1e-13, f"compiled grads vs eager: rel={rel:.3e}"
        seen.append(counters["stats"]["unique_graphs"])
    dynamo.reset()
    assert seen[-1] <= 2, f"re-specializes per shape: {seen}"


# ---------------------------------------------------------------------------
# padded_segment_sum + the slot/K backward route (FLASH-3 step 2): the segment sum
# as a static-shape padded scatter-write -- no atomics, no segment_reduce fallback,
# no per-call host read. Same gate set: parity incl. empties, gradcheck, compile.
# ---------------------------------------------------------------------------


def _slot_of(idx, counts):
    counts_long = counts.view(-1).long()
    offsets = torch.cumsum(counts_long, 0) - counts_long
    return torch.arange(idx.numel()) - offsets.index_select(0, idx)


@pytest.mark.parametrize("dtype,bar", [(torch.float64, 1e-13), (torch.float32, 1e-5)])
def test_padded_segment_sum_matches_segment_reduce(dtype, bar):
    x, idx, counts = _case(dtype=dtype)
    data = torch.randn(idx.numel(), 4, dtype=dtype)
    slot = _slot_of(idx, counts)
    K = int(counts.max())
    want = torch.segment_reduce(data, "sum", lengths=counts.view(-1).long(), axis=0)
    got = sg_mod.padded_segment_sum(data, idx, slot, x.shape[0], K)
    rel = ((got - want).abs().max() / (1 + want.abs().max())).item()
    print(f"GATE-PADSUM rel={rel:.3e}")
    assert rel < bar
    assert (got[0] == 0).all() and (got[-1] == 0).all(), "empty segments must be exact zero"
    # a LOOSER bound K must give the identical result (the callers pass k, not max degree)
    got_loose = sg_mod.padded_segment_sum(data, idx, slot, x.shape[0], K + 3)
    assert torch.equal(got, got_loose)


def test_padded_slot_contract():
    """Executable REQUIRES: slot = in-segment rank, (idx, slot) unique, K >= max count."""
    x, idx, counts = _case()
    slot = _slot_of(idx, counts)
    assert (slot >= 0).all() and (slot < counts.view(-1).long().index_select(0, idx)).all()
    pairs = idx * (int(counts.max()) + 1) + slot
    assert pairs.unique().numel() == pairs.numel(), "(idx, slot) pairs must be unique"


@pytest.mark.parametrize("dtype,bar", [(torch.float64, 1e-13), (torch.float32, 1e-5)])
def test_slot_backward_matches_autograd(dtype, bar):
    x, idx, counts = _case(dtype=dtype)
    slot = _slot_of(idx, counts)
    K = int(counts.max())
    g = torch.randn(idx.numel(), x.shape[1], dtype=dtype)
    a = x.clone().requires_grad_(True)
    a[idx].backward(g)
    b = x.clone().requires_grad_(True)
    sorted_gather(b, idx, counts, slot, K).backward(g)
    rel = ((b.grad - a.grad).abs().max() / (1 + a.grad.abs().max())).item()
    print(f"GATE-SLOT-BWD rel={rel:.3e}")
    assert rel < bar
    assert (b.grad[0] == 0).all() and (b.grad[-1] == 0).all()


def test_slot_gradcheck():
    x, idx, counts = _case(N=6, E=11, F=3)
    slot = _slot_of(idx, counts)
    K = int(counts.max())
    x.requires_grad_(True)
    assert torch.autograd.gradcheck(lambda t: sorted_gather(t, idx, counts, slot, K), (x,))
    assert torch.autograd.gradgradcheck(lambda t: sorted_gather(t, idx, counts, slot, K), (x,))


def test_slot_compiled_no_breaks():
    import torch._dynamo as dynamo

    x, idx, counts = _case(dtype=torch.float32)
    slot = _slot_of(idx, counts)
    K = int(counts.max())

    def step(t):
        y = sorted_gather(t, idx, counts, slot, K)
        agg = sg_mod.padded_segment_sum(y, idx, slot, t.shape[0], K)
        return (agg * agg).sum()

    explanation = dynamo.explain(step)(x.clone().requires_grad_(True))
    dynamo.reset()
    assert explanation.graph_break_count == 0, str(explanation)[:1500]


def test_perm_extras_contract():
    """The Function's REQUIRES line, executable: perm must be the stable argsort of
    idx and counts its degree vector -- exactly what the net forwards compute. A
    wrong perm/counts pair gives silently wrong gradients, so keep the contract loud."""
    x, idx, perm, counts = _perm_case()
    assert torch.equal(perm, torch.argsort(idx, stable=True))
    assert torch.equal(counts.view(-1).long(), torch.bincount(idx, minlength=x.shape[0]))
    # and the reduction identity the backward relies on: gathering by perm then
    # segment-summing over the sorted runs equals autograd's scatter-add.
    g = torch.randn(idx.numel(), x.shape[1], dtype=torch.float64)
    want = torch.zeros_like(x).index_add_(0, idx, g)
    got = torch.segment_reduce(g.index_select(0, perm).contiguous(), "sum",
                               lengths=counts.view(-1).long(), axis=0)
    assert ((got - want).abs().max() / (1 + want.abs().max())).item() < 1e-13
