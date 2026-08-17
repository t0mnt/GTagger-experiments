"""SortedGather: deterministic segment-sum backward for RECEIVER-side edge gathers.

WHY (docs/cgenn-compile.md, round 4 + final ledger resolution): the top kernel on both
H100 profiles (`triton_poi_fused_index_put_mul_new_zeros`, 27.9% of CUDA on tag_cgenn,
22.7% on CGENN-GPS) is not the aggregation -- it is the BACKWARD of the edge gathers:
autograd differentiates `x[idx]` into an atomic scatter-add of edge gradients into node
slots. For the RECEIVER index that scatter is needless work: receivers arrive SORTED
from both edge builders (machine-checked in tests/experiments/test_edge_builders.py) and
their per-node degree is already threaded through every CGL as `edge_counts` (Phase
2.2a). A scatter over a sorted index IS a segment sum, so the backward here is one
`torch.segment_reduce` call -- the same op, invariant, and lengths source as the 2.2b
forward aggregation, with the same two wins: no atomics (deterministic gradients where
autograd's scatter-add was nondeterministic run-to-run) and no Triton scatter kernel.
Sender-side gathers (`x[j]`) keep plain autograd -- senders are not sorted.

CLASS: forward is BIT-identical to `x[idx]` (it IS index_select), so the hybrid BIT
pins stand un-re-recorded. Gradients are the same sum reassociated -- and the atomics
they replace never produced two identical runs anyway, so this is a strict determinism
upgrade, not a comparability break. Gates: tests/experiments/test_sorted_gather.py
(BIT fwd, grads vs plain autograd at fp64, empty segments, gradcheck, compile
breaks/recomp); the model-level BACKWARD-TOL and compile batteries run with this
active by default.

KILL SWITCH: CGENN_SORTED_GATHER=0 reverts to plain `x[idx]` (the shield pattern) --
one env var, no code change, for the GPU gate day. Read ONCE at import: set it in the
environment before the process launches (as the runbook does), not mid-process.
"""

import os

import torch

_ENABLED = os.environ.get("CGENN_SORTED_GATHER", "1") != "0"


class SortedGather(torch.autograd.Function):
    """y = x.index_select(0, idx) with a segment-sum backward.

    REQUIRES: idx sorted ascending and counts[k] == (idx == k).sum() -- exactly the
    receiver/edge_counts invariant the CGLs already carry. Violating it returns wrong
    gradients silently, which is why `sorted_gather` only engages when the caller
    hands it the counts it already computed for the mean aggregation.
    """

    generate_vmap_rule = True

    @staticmethod
    def forward(x, idx, counts):
        return x.index_select(0, idx)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, idx, counts = inputs
        ctx.save_for_backward(counts)

    @staticmethod
    def backward(ctx, grad_out):
        (counts,) = ctx.saved_tensors
        lengths = counts.view(-1).to(torch.int64)
        # sum over each receiver's contiguous run of edges = the scatter-add autograd
        # would do, deterministically; zero-degree nodes get exact-zero gradient rows
        # (segment_reduce's sum identity -- pinned for empties by the 2.2b gates).
        gx = torch.segment_reduce(grad_out.contiguous(), "sum", lengths=lengths, axis=0)
        return gx, None, None


def sorted_gather(x, idx, counts):
    """`x[idx]` for a SORTED idx, deterministic backward. Falls back to plain
    autograd when counts is None (call sites without the hoisted degrees) or the
    kill switch is set."""
    if counts is None or not _ENABLED:
        return x[idx]
    return SortedGather.apply(x, idx, counts)


class SortedGatherPermuted(torch.autograd.Function):
    """`x[idx]` for an UNSORTED idx, with a deterministic segment-sum backward via a
    precomputed stable-sort permutation (FLASH-2: the send-side twin of SortedGather).

    scatter_add(grad, idx) == segment_sum(grad[perm]) over the sorted runs of
    idx[perm]: within a segment the addends are the same set whatever their order, so
    the value is exact-equal in expectation class and DETERMINISTIC by construction
    (fixed perm + torch.segment_reduce), where autograd's scatter atomics were not.
    REQUIRES: perm = argsort(idx, stable=True) and counts = bincount(idx) -- computed
    once per forward next to the receiver degrees and threaded the same way.
    """

    generate_vmap_rule = True

    @staticmethod
    def forward(x, idx, perm, counts):
        return x.index_select(0, idx)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, idx, perm, counts = inputs
        ctx.save_for_backward(perm, counts)

    @staticmethod
    def backward(ctx, grad_out):
        perm, counts = ctx.saved_tensors
        lengths = counts.view(-1).to(torch.int64)
        g = grad_out.index_select(0, perm).contiguous()
        gx = torch.segment_reduce(g, "sum", lengths=lengths, axis=0)
        return gx, None, None, None


def sorted_gather_perm(x, idx, perm, counts):
    """`x[idx]` for an unsorted idx with the permutation-based deterministic backward.
    Falls back to plain autograd when the permutation/counts were not threaded or the
    kill switch is set (same CGENN_SORTED_GATHER switch as the receiver side)."""
    if perm is None or counts is None or not _ENABLED:
        return x[idx]
    return SortedGatherPermuted.apply(x, idx, perm, counts)
