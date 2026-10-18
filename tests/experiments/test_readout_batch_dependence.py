"""Pins the two batch-composition properties of the tag_cgenn readout.

The readout keeps the official-repo quirk (cgenn.py, `torch.mean(h, dim=1)` over the PADDED
node axis; frozen as parity, see docs/cgenn-compile.md "Untouchable"). Two consequences,
both pinned so that a change to either is deliberate:

1. At a FIXED padded width, a jet's score does not depend on which other jets share its
   batch (eval mode: BatchNorm uses running stats, the graph is per-jet, the readout divisor
   is the width). This is what makes evaluation reproducible with shuffle=False loaders.
2. Changing the padded width changes the score of the same jet. That is the quirk itself;
   this assert documents its magnitude and flips the day the readout becomes mask-aware,
   which then has to be a stated modeling change with its own validation runs.
"""

import torch
from torch_geometric.data import Batch

from tests.experiments.test_cgenn_compile import _build, _forward


def _jets(loader, n):
    jets = []
    for batch in loader:
        jets.extend(batch.to_data_list())
        if len(jets) >= n:
            return jets[:n]
    return jets


def test_score_independent_of_batch_mates_at_fixed_width():
    exp = _build(float64=True)
    jets = _jets(exp.train_loader, 6)
    widest = max(j.num_nodes for j in jets)
    anchor = next(j for j in jets if j.num_nodes == widest)  # sets the padded width itself
    others = [j for j in jets if j is not anchor]
    y_a = _forward(exp, Batch.from_data_list([anchor, others[0], others[1]]))[0]
    y_b = _forward(exp, Batch.from_data_list([others[2], anchor, others[3]]))[1]
    torch.testing.assert_close(y_a, y_b, rtol=1e-12, atol=1e-12)


def test_score_depends_on_padded_width():
    exp = _build(float64=True)
    jets = sorted(_jets(exp.train_loader, 8), key=lambda j: j.num_nodes)
    small, big = jets[0], jets[-1]
    assert big.num_nodes > small.num_nodes, "mini set has no width spread; pick other jets"
    y_narrow = _forward(exp, Batch.from_data_list([small, small]))[0]      # width = small
    y_wide = _forward(exp, Batch.from_data_list([small, big]))[0]          # width = big
    dev = (y_wide - y_narrow).abs().max().item()
    assert dev > 1e-6, (
        "the padded-width readout dependence is gone: if the readout became mask-aware on "
        "purpose, retire this test with the modeling change that did it")
