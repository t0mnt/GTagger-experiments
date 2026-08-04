"""Components that exist twice must stay numerically identical.

Several building blocks live in two places: a vendored reference copy under
`experiments/baselines/{cgenn,particlenet}.py` that a BASELINE row uses, and an inlined
copy inside a hybrid that a HYBRID row uses. Nothing links the two, so a fix applied to
one silently leaves the other behind -- which is exactly what happened to the CGENN
MVLayerNorm weight-decay fix: it landed in the vendored file while both hybrids kept
importing the inlined 2-d gain, so the hybrid rows trained under a different
regularization than their own reference row.

These tests give the two copies identical weights and identical inputs and demand
identical outputs. They are cheap (no data, tiny tensors) and they fail loudly the first
time someone patches one copy only.
"""

import importlib

import pytest
import torch

H = importlib.import_module("experiments.baselines.CGENNLGATrGraphTransHybrid")
V_linear = importlib.import_module("experiments.baselines.cgenn.linear")
V_mvsilu = importlib.import_module("experiments.baselines.cgenn.mvsilu")
V_mvln = importlib.import_module("experiments.baselines.cgenn.mvlayernorm")
V_norm = importlib.import_module("experiments.baselines.cgenn.normalization")
V_gp = importlib.import_module("experiments.baselines.cgenn.gp")
V_fcgp = importlib.import_module("experiments.baselines.cgenn.fcgp")
V_alg = importlib.import_module("experiments.baselines.cgenn.cliffordalgebra")
H_pn = importlib.import_module("experiments.baselines.particlenettransformer")
V_pn = importlib.import_module("experiments.baselines.particlenet")

METRIC = (1.0, -1.0, -1.0, -1.0)


def _algebras():
    return H.CliffordAlgebra(METRIC), V_alg.CliffordAlgebra(METRIC)


def _assert_same_output(hybrid, vendored, inputs):
    """Copy vendored weights into the hybrid copy (reshaping across deliberate rank
    differences such as the 1-d vs 2-d norm gain) and compare outputs."""
    hp, vp = dict(hybrid.named_parameters()), dict(vendored.named_parameters())
    assert set(hp) == set(vp), (
        f"parameter names diverged: hybrid-only {sorted(set(hp) - set(vp))}, "
        f"vendored-only {sorted(set(vp) - set(hp))}"
    )
    with torch.no_grad():
        for name in hp:
            assert hp[name].numel() == vp[name].numel(), (
                f"{name}: hybrid {tuple(hp[name].shape)} vs vendored {tuple(vp[name].shape)} "
                f"differ in SIZE, not just rank -- the copies have diverged."
            )
            hp[name].copy_(vp[name].reshape(hp[name].shape))
    hybrid.eval()
    vendored.eval()
    with torch.no_grad():
        a, b = hybrid(*inputs), vendored(*inputs)
    if isinstance(a, tuple):
        for x, y in zip(a, b):
            assert torch.equal(x, y), f"outputs differ, max |delta| {(x - y).abs().max():.3e}"
    else:
        assert torch.equal(a, b), f"outputs differ, max |delta| {(a - b).abs().max():.3e}"


@pytest.mark.parametrize(
    "name,build_hybrid,build_vendored",
    [
        ("MVLinear", lambda a: H.MVLinear(a[0], 3, 4), lambda a: V_linear.MVLinear(a[1], 3, 4)),
        ("MVSiLU", lambda a: H.MVSiLU(a[0], 3), lambda a: V_mvsilu.MVSiLU(a[1], 3)),
        ("MVLayerNorm", lambda a: H.MVLayerNorm(a[0], 3), lambda a: V_mvln.MVLayerNorm(a[1], 3)),
        (
            "NormalizationLayer",
            lambda a: H.NormalizationLayer(a[0], 3),
            lambda a: V_norm.NormalizationLayer(a[1], 3),
        ),
        (
            "SteerableGeometricProductLayer",
            lambda a: H.SteerableGeometricProductLayer(a[0], 3),
            lambda a: V_gp.SteerableGeometricProductLayer(a[1], 3),
        ),
        (
            "FullyConnectedSteerableGeometricProductLayer",
            lambda a: H.FullyConnectedSteerableGeometricProductLayer(a[0], 3, 3),
            lambda a: V_fcgp.FullyConnectedSteerableGeometricProductLayer(a[1], 3, 3),
        ),
    ],
)
def test_cgenn_primitive_parity(name, build_hybrid, build_vendored):
    algebras = _algebras()
    torch.manual_seed(1)
    hybrid = build_hybrid(algebras)
    torch.manual_seed(1)
    vendored = build_vendored(algebras)
    torch.manual_seed(2)
    x = torch.randn(5, 3, 16)
    _assert_same_output(hybrid, vendored, (x,))


CGENN_PAIRS = [
    ("MVLinear", lambda a: H.MVLinear(a[0], 3, 4), lambda a: V_linear.MVLinear(a[1], 3, 4)),
    ("MVSiLU", lambda a: H.MVSiLU(a[0], 3), lambda a: V_mvsilu.MVSiLU(a[1], 3)),
    ("MVLayerNorm", lambda a: H.MVLayerNorm(a[0], 3), lambda a: V_mvln.MVLayerNorm(a[1], 3)),
    ("NormalizationLayer", lambda a: H.NormalizationLayer(a[0], 3),
     lambda a: V_norm.NormalizationLayer(a[1], 3)),
    ("SteerableGeometricProductLayer", lambda a: H.SteerableGeometricProductLayer(a[0], 3),
     lambda a: V_gp.SteerableGeometricProductLayer(a[1], 3)),
    ("FullyConnectedSteerableGeometricProductLayer",
     lambda a: H.FullyConnectedSteerableGeometricProductLayer(a[0], 3, 3),
     lambda a: V_fcgp.FullyConnectedSteerableGeometricProductLayer(a[1], 3, 3)),
]


@pytest.mark.parametrize("name,build_hybrid,build_vendored", CGENN_PAIRS)
def test_cgenn_primitive_initialisation_parity(name, build_hybrid, build_vendored):
    """Same seed in, same INITIAL weights out.

    The forward-parity test above copies vendored weights into the hybrid, so it is blind
    to an initialisation that drifted -- and init is a real divergence: it changes where
    training starts, silently, for one row of the table only.
    """
    algebras = _algebras()
    torch.manual_seed(1)
    hybrid = build_hybrid(algebras)
    torch.manual_seed(1)
    vendored = build_vendored(algebras)
    hp, vp = dict(hybrid.named_parameters()), dict(vendored.named_parameters())
    assert set(hp) == set(vp)
    for pname in sorted(hp):
        a, b = hp[pname].detach(), vp[pname].detach()
        assert a.numel() == b.numel(), (
            f"{name}.{pname}: hybrid {tuple(a.shape)} vs vendored {tuple(b.shape)}"
        )
        assert torch.equal(a.reshape(-1), b.reshape(-1)), (
            f"{name}.{pname} INITIALISES differently: max |delta| "
            f"{(a.reshape(-1) - b.reshape(-1)).abs().max():.3e}. The two copies have drifted."
        )


@pytest.mark.parametrize("op", ["geometric_product", "norm", "norms", "qs", "alpha", "beta", "q"])
def test_clifford_algebra_op_parity(op):
    """Same multivector in, same result out -- one shared input, not two draws."""
    alg_h, alg_v = _algebras()
    torch.manual_seed(0)
    mv = torch.randn(4, 16)
    args = (mv, mv) if op == "geometric_product" else (mv,)
    a, b = getattr(alg_h, op)(*args), getattr(alg_v, op)(*args)
    if isinstance(a, (list, tuple)):
        a = torch.cat([t.reshape(t.shape[0], -1) for t in a], -1)
        b = torch.cat([t.reshape(t.shape[0], -1) for t in b], -1)
    assert torch.equal(a, b), f"{op} differs, max |delta| {(a - b).abs().max():.3e}"


def test_cayley_table_parity():
    alg_h, alg_v = _algebras()
    assert torch.equal(alg_h.cayley, alg_v.cayley)
    assert alg_h.grades.tolist() == alg_v.grades.tolist()
    assert alg_h.n_subspaces == alg_v.n_subspaces


def test_edgeconv_block_parity():
    """The hybrid's EdgeConv must reduce to vendored ParticleNet when transport is off."""
    kwargs = dict(k=4, in_feat=6, out_feats=(8, 8))
    torch.manual_seed(5)
    hybrid = H_pn.EdgeConvBlock(**kwargs)
    torch.manual_seed(5)
    vendored = V_pn.EdgeConvBlock(**kwargs)
    torch.manual_seed(11)
    points, features = torch.randn(3, 3, 9), torch.randn(3, 6, 9)
    _assert_same_output(hybrid, vendored, (points, features))
