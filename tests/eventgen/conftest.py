import pytest
import torch


@pytest.fixture(autouse=True)
def _seed():
    # Every eventgen test draws unseeded random events, and several checks are
    # numerically seed-sensitive (near-massless draws blow up ~1/M^2 jacobian and
    # velocity terms past tolerance -- observed as rare flakes in test_transforms
    # AND test_coordinates). One autouse fixture here pins the RNG for the whole
    # directory instead of per-file copies.
    torch.manual_seed(0)
