"""Grade structure of the dense CGENN stream: blades no product can reach are EXACTLY zero.

rotorch (tBuLi/rotorch) stores only the blades a layer can produce and asserts the output
*type* of every layer in its tests. GTagger's stream is a dense (tokens, channels, 16) tensor,
so the same fact has to be asserted on the values: the embedding of a four-momentum is a pure
vector, and the first message product of vectors reaches grades {0, 2} plus the vector
left-linear and the scalar bias, so grades 3 and 4 are structurally empty there. (The first
node update already multiplies the aggregated message, which carries grade 2, with itself, so
bivector x bivector reaches grade 4 inside layer one -- nothing is asserted past the message
product.) Any impl that leaks mass into an unreachable blade -- a wrong sign
table, a mis-sliced hoisted weight, a kernel writing the wrong slot -- fails here with exact
arithmetic, since every such entry is a product with at least one structural zero.

These are exact-zero asserts, not tolerances: 0 * finite = 0 and sums of zeros are 0 in IEEE
arithmetic, so a nonzero is a bug, not rounding. Runs on CPU for every gp_impl (flash uses its
gated CPU composite there).
"""

import pytest
import torch

from tests.experiments.test_cgenn_compile import _build, _fixed_batch, _forward


def _grade_mask(algebra, grades):
    g = algebra.bbo_grades.long()
    return torch.zeros_like(g, dtype=torch.bool).index_fill_(
        0, torch.cat([torch.nonzero(g == k).flatten() for k in grades]), True)


@pytest.mark.parametrize("impl", ["einsum", "sparse", "flash"])
def test_unreachable_blades_are_exactly_zero(impl):
    exp = _build(float64=False, extra_overrides=[f"model.net.gp_impl={impl}"])
    net = exp.model.net
    alg = net.algebra
    seen = {}

    def tap(name):
        def hook(mod, inp, out):
            seen[name] = (out[1] if isinstance(out, tuple) else out).detach()
        return hook

    net.embedding_x.register_forward_hook(tap("embed"))
    # the hoisted message path (CGENN_HOIST=1, the default) calls phi_x[0] and phi_x[1]
    # directly rather than the Sequential, so tap the layer norm that both paths end in
    net.CGLs[0].phi_x[1].register_forward_hook(tap("phi_x_1"))
    _forward(exp, _fixed_batch(exp))

    embed, phi = seen["embed"], seen["phi_x_1"]
    assert embed.shape[-1] == 16 and phi.shape[-1] == 16
    # the embedding is a per-channel scalar mix of a vector: only grade 1 is populated
    assert torch.equal(embed[..., ~_grade_mask(alg, [1])], torch.zeros_like(embed[..., ~_grade_mask(alg, [1])]))
    # vector x vector reaches grades {0, 2}; the left linear adds grade 1 and the bias grade 0
    assert torch.equal(phi[..., _grade_mask(alg, [3, 4])], torch.zeros_like(phi[..., _grade_mask(alg, [3, 4])]))
    # and the reachable blades are actually populated (the test is not vacuous)
    assert embed[..., _grade_mask(alg, [1])].abs().sum() > 0
    assert phi[..., _grade_mask(alg, [0, 1, 2])].abs().sum() > 0
