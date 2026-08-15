"""The edge builders' RECEIVER-SORTEDNESS invariant, machine-checked.

Phase 2.2b (docs/cgenn-compile.md) aggregates messages with torch.segment_reduce over
contiguous runs, which is only correct because both edge builders emit receivers in
nondecreasing order:

  * kNN (generate_edges_vectorized): receivers are arange-expanded per node and the
    validity filter preserves order;
  * fully-connected (CGENNWrapper.forward and generate_edges_vectorized's k=None path):
    `pair.nonzero(as_tuple=True)` returns row-major order.

Structural today -- but a future edit (symmetrization, shuffling, a different builder)
could silently break it, and the failure mode is WRONG AGGREGATION, not a crash. This
test fails loudly instead. Fixture-free, CPU, adversarial masks (empty jets, singleton
jets, full jets, ragged sizes).
"""

import torch

from experiments.baselines.CGENNLGATrGraphTransHybrid import generate_edges_vectorized


def _masks():
    torch.manual_seed(0)
    B, P = 16, 37
    cases = []
    # ragged random occupancy, including tiny jets
    for frac in (0.2, 0.6, 1.0):
        m = torch.zeros(B, P, dtype=torch.bool)
        for b in range(B):
            n = max(1, int(P * frac * torch.rand(1).item()))
            m[b, :n] = True
        cases.append(m)
    # extremes: singleton jets and a full batch
    m = torch.zeros(B, P, dtype=torch.bool)
    m[:, 0] = True
    cases.append(m)
    cases.append(torch.ones(B, P, dtype=torch.bool))
    return cases


def _points(mask):
    B, P = mask.shape
    return torch.randn(B, P, 2)


def test_knn_receivers_sorted():
    for k in (3, 8, 40):  # k > P-1 exercises the k_actual clamp
        for mask in _masks():
            e = generate_edges_vectorized(mask, _points(mask), k, mask.shape[1], "cpu")
            recv = e[0]
            assert (recv[1:] >= recv[:-1]).all(), (
                f"kNN receivers not sorted at k={k}: segment_reduce aggregation "
                f"(Phase 2.2b) would be silently wrong")


def test_fully_connected_receivers_sorted():
    for mask in _masks():
        e = generate_edges_vectorized(mask, _points(mask), None, mask.shape[1], "cpu")
        recv = e[0]
        assert (recv[1:] >= recv[:-1]).all(), (
            "fully-connected receivers not sorted: segment_reduce aggregation "
            "(Phase 2.2b) would be silently wrong")


def test_degrees_match_segment_lengths():
    """The hoisted degree counts (2.2a) must equal bincount over receivers -- the same
    tensor 2.2b casts to segment lengths. Ties the two hoists together."""
    for mask in _masks():
        e = generate_edges_vectorized(mask, _points(mask), 8, mask.shape[1], "cpu")
        recv = e[0]
        N = mask.numel()
        counts = torch.zeros(N, 1).index_add_(0, recv, torch.ones(recv.numel(), 1))
        assert torch.equal(counts.view(-1).long(), torch.bincount(recv, minlength=N))
        assert int(counts.sum()) == recv.numel()
