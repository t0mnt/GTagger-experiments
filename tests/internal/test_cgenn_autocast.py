"""Gates for the vendored fp32 precision islands in experiments/baselines/cgenn/autocast.py.

Three things need pinning, and the first is the one that lets this land ahead of any AMP
fixtures:

1. AMP OFF is bit-identical. Every shipped config is `use_amp: false`, so decorating the
   Clifford primitives must be a no-op today. If this fails, the decorators changed eager
   numerics and every recorded BIT fixture is invalid.
2. The islands actually hold under bf16 autocast, with an UNGUARDED control (`naive_amp`)
   that must diverge -- otherwise test 2 could pass vacuously on a path autocast never
   touches, which is exactly what happens to b()/q() (see test_invariants_are_naturally_safe).
3. The copy still matches the lgatr it was vendored from. autocast.py is a copy rather than
   an import so the package stays transplantable to DavidRuhe/clifford-group-equivariant-
   neural-networks; a copy that silently drifts is worse than an import.
"""

import pytest
import torch

from experiments.baselines.cgenn.autocast import minimum_autocast_precision, naive_amp
from experiments.baselines.cgenn.cliffordalgebra import CliffordAlgebra
from experiments.baselines.cgenn.fcgp import (
    FullyConnectedSteerableGeometricProductLayer as FCGP,
)

torch.set_num_threads(1)  # CPU inductor + OpenMP aborts CGENN otherwise (audit ledger)


@pytest.fixture
def gp_and_input():
    torch.manual_seed(0)
    alg = CliffordAlgebra((1.0, -1.0, -1.0, -1.0))
    alg.gp_impl = "einsum"
    layer = FCGP(alg, 12, 8, normalization_init=None).eval()
    return alg, layer, torch.randn(64, 12, 16)


def test_amp_off_is_bit_identical(gp_and_input):
    """The load-bearing safety property: no autocast region -> the decorator is a passthrough."""
    _, layer, x = gp_and_input
    with torch.no_grad():
        a, b = layer(x), layer(x)
    assert torch.equal(a, b)
    assert a.dtype is torch.float32, "no autocast region must leave the dtype alone"


def test_island_holds_under_bf16_and_the_control_diverges(gp_and_input):
    """Guarded == fp32 exactly; unguarded must NOT, or the test proves nothing."""
    _, layer, x = gp_and_input
    with torch.no_grad():
        ref = layer(x)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            guarded = layer(x)
            with naive_amp():
                unguarded = layer(x)

    guarded_dev = (guarded.float() - ref).abs().max().item()
    unguarded_dev = (unguarded.float() - ref).abs().max().item()

    assert guarded.dtype is torch.float32
    assert guarded_dev == 0.0, f"island leaked: {guarded_dev:.3e}"
    # Control. Measured 4.5e-2 on this shape; assert only that bf16 is materially wrong here,
    # so the bound survives a different seed or shape.
    assert unguarded_dev > 1e-3, (
        f"unguarded bf16 deviated by only {unguarded_dev:.3e} -- if bf16 is harmless on this "
        "path the guard proves nothing and this test is vacuous"
    )


def test_invariants_are_naturally_safe(gp_and_input):
    """b()/q() do not need the guard, and the test says so rather than implying otherwise.

    After the signed-dot rewrite they are elementwise mul + sum, and autocast only casts the
    matmul/einsum/conv family -- so they run fp32 under bf16 autocast either way. The
    decorator is kept as defence against a future rewrite reintroducing an einsum, but this
    test records that it is currently doing nothing, so nobody reads a passing guard test as
    evidence that it was needed.
    """
    alg, _, x = gp_and_input
    with torch.no_grad():
        ref = alg.q(x)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            with naive_amp():
                unguarded = alg.q(x)
    assert (unguarded.float() - ref).abs().max().item() == 0.0


def test_copy_matches_lgatr():
    """The vendored decorator behaves identically to the lgatr one it was copied from."""
    lgatr_autocast = pytest.importorskip("lgatr.utils.autocast")

    def f(t):
        return t @ t.T

    ours = minimum_autocast_precision(torch.float32, output="high")(f)
    theirs = lgatr_autocast.minimum_autocast_precision(torch.float32, output="high")(f)

    torch.manual_seed(0)
    t = torch.randn(32, 32)
    with torch.no_grad():
        assert torch.equal(ours(t), theirs(t))  # outside autocast
        with torch.autocast("cpu", dtype=torch.bfloat16):
            o, th = ours(t), theirs(t)
    assert o.dtype is th.dtype
    assert torch.equal(o, th)
