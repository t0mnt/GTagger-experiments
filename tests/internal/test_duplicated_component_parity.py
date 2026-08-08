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
V_cgenn = importlib.import_module("experiments.baselines.cgenn.cgenn")
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


@pytest.mark.parametrize("layer_type", ["fc", "gpmlp"])
@pytest.mark.parametrize("use_invariants_to_update", [True, False])
def test_cglayer_parity(layer_type, use_invariants_to_update):
    """The whole message-passing layer, not just its primitives.

    The primitive tests above pin MVLinear/MVSiLU/... one at a time; they would not catch a
    rewired forward (a dropped residual, a swapped concat order, a different aggregation).
    This builds both CGLayers at the widths the two backbones actually wire -- all-equal
    hidden channels, node_attr from the input widths -- and demands identical outputs from
    identical weights, for both layer types and both update modes.

    The hybrid dropped the vendored `use_invariant_network` flag. That flag's False branch
    sets in_features_h=0 and passes h=None; no config in this repo sets it, so the branch is
    dead and the live path is the one compared here.
    """
    algebras = _algebras()
    CX, CH, IN_X, IN_H = 4, 6, 2, 5
    kwargs = dict(
        layer_type=layer_type,
        use_invariants_to_update=use_invariants_to_update,
        edge_attr_x=3 * IN_X,
        edge_attr_h=0,
        node_attr_x=IN_X,
        node_attr_h=IN_H,
    )
    torch.manual_seed(3)
    hybrid = H.CGLayer(algebras[0], CX, CX, CX, CH, CH, CH, **kwargs)
    torch.manual_seed(3)
    vendored = V_cgenn.CGLayer(algebras[1], CX, CX, CX, CH, CH, CH, **kwargs)
    torch.manual_seed(9)
    n_nodes, n_edges = 12, 30
    inputs = (
        torch.randn(n_nodes, CH),
        torch.randn(n_nodes, CX, 16),
        torch.randint(0, n_nodes, (2, n_edges)),
        torch.randn(n_nodes, IN_H),          # node_attr_h
        torch.randn(n_nodes, IN_X, 16),      # node_attr_x
        None,                                # edge_attr_h
        torch.randn(n_edges, 3 * IN_X, 16),  # edge_attr_x
    )
    _assert_same_output(hybrid, vendored, inputs)


def test_gps_hybrids_reuse_the_graphtrans_classes():
    """Each family has TWO rows. They must share one class object, not two copies.

    Every parity result here is proven on the GraphTrans module; it transfers to the GraphGPS
    row only because the GPS file imports the same objects rather than redefining them. If a
    GPS variant ever grows its own copy, the topology comparison stops being controlled and
    every test in this file silently covers half the table.
    """
    import experiments.baselines.cgennlgatrgraphgps as cgenn_gps
    import experiments.baselines.particlenetpartgraphgps as pn_gps

    for gps, source, names in (
        (pn_gps, H_pn, ["EdgeConvBlock", "PairEmbed"]),
        (cgenn_gps, H, ["CGLayer", "CliffordAlgebra"]),
    ):
        for name in names:
            assert getattr(gps, name) is getattr(source, name), (
                f"{gps.__name__}.{name} is not the same object as {source.__name__}.{name}: "
                f"the GraphGPS row has its own copy and can drift from its GraphTrans partner."
            )


def test_pairwise_features_are_a_permutation_of_llocas_at_the_configured_width():
    """pair_input_dim=4 is what every config uses, and there the two orderings are the same
    four features permuted -- identical model class ahead of BatchNorm + 1x1 Conv. At width 1
    they differ in CONTENT (lndelta here, lnm2 in lloca), which is the trap this pins."""
    from lloca.backbone.particletransformer import pairwise_lv_fts_pp as lloca_fts

    torch.manual_seed(0)
    xi = torch.randn(64, 4, 30).abs() + 0.1
    xj = torch.randn(64, 4, 30).abs() + 0.1
    hybrid = H_pn.pairwise_lv_fts(xi, xj, num_outputs=4)
    lloca = lloca_fts(xi, xj, num_outputs=4)
    # lloca [lnm2, lnkt, lnz, lndelta] -> hybrid [lnkt, lnz, lndelta, lnm2]
    assert torch.equal(hybrid, lloca[:, [1, 2, 3, 0]]), (
        f"the two ParT pairwise feature sets are no longer a pure permutation at width 4: "
        f"max |delta| {(hybrid - lloca[:, [1, 2, 3, 0]]).abs().max():.3e}"
    )
    assert not torch.equal(
        H_pn.pairwise_lv_fts(xi, xj, num_outputs=1), lloca_fts(xi, xj, num_outputs=1)
    ), "width 1 now agrees; update the note in pairwise_lv_fts, which says it does not"


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
    """The hybrid's EdgeConv must reduce to the ported ParticleNet when transport is off.

    `experiments/baselines/particlenet.py` is no longer a stock-weaver copy: it is the LLoCa
    port that `tag_particlenet` runs, so this is now a comparison between two live models
    rather than against a museum piece. Identity frames and 3-d coords switch off both the
    transport and the hybrid's phi-wrapping, leaving the pure EdgeConv arithmetic.
    """
    from lloca.framesnet.frames import Frames
    from lloca.reps.tensorreps import TensorReps

    torch.manual_seed(5)
    hybrid = H_pn.EdgeConvBlock(k=4, in_feat=6, out_feats=(8, 8))
    torch.manual_seed(5)
    ported = V_pn.EdgeConvBlock(k=4, in_reps=TensorReps("6x0n"), out_feats=(8, 8))
    torch.manual_seed(11)
    points, features = torch.randn(3, 3, 9), torch.randn(3, 6, 9)
    frames = Frames(is_identity=True, device=points.device, dtype=points.dtype, shape=(3 * 9,))
    hp, vp = dict(hybrid.named_parameters()), dict(ported.named_parameters())
    assert set(hp) == set(vp), f"parameter names diverged: {sorted(set(hp) ^ set(vp))}"
    with torch.no_grad():
        for name in hp:
            hp[name].copy_(vp[name].reshape(hp[name].shape))
    hybrid.eval()
    ported.eval()
    with torch.no_grad():
        a = hybrid(points, features, knn_metric="deltaR", mask=None, frames=None)
        b = ported(points, features, frames=frames)
    assert torch.equal(a, b), f"outputs differ, max |delta| {(a - b).abs().max():.3e}"


def test_ported_particlenet_transport_is_byte_faithful_to_installed_lloca():
    """The port must stay a port.

    `experiments/baselines/particlenet.py` was generated from `lloca.backbone.particlenet` and
    carries exactly two deliberate additions (the for_inference sigmoid, and knn_metric). The
    tensorial message passing itself -- `change_local_frame`, the two `get_graph_feature`
    variants and `knn` -- is copied verbatim, and it is the part whose silent divergence would
    be invisible in results while invalidating "this is LLoCa-ParticleNet".

    Compare ASTs rather than text so formatting, comments and lint fixes are free, but any
    change to the arithmetic on either side fails. If upstream deliberately changed one of
    these, port the change and update the row in docs/diffs.md in the same commit.
    """
    import ast
    import inspect

    from lloca.backbone import particlenet as L_pn

    for name in ("change_local_frame", "get_graph_feature_v1", "get_graph_feature_v2", "knn"):
        ours = ast.dump(ast.parse(inspect.getsource(getattr(V_pn, name))))
        theirs = ast.dump(ast.parse(inspect.getsource(getattr(L_pn, name))))
        assert ours == theirs, (
            f"{name}() has drifted from the installed lloca. The port is only meaningful while "
            f"the transport math is identical -- diff the two files, port the change "
            f"deliberately, and record it in docs/diffs.md."
        )


def test_ported_particlenet_deltaR_matches_installed_lloca_end_to_end():
    """Whole-model bit-parity on the default path, so the published row cannot move silently."""
    from lloca.backbone import particlenet as L_pn
    from lloca.framesnet.frames import Frames

    kwargs = dict(
        input_dims=6,
        hidden_reps_list=["6x0n", "8x0n"],
        num_classes=2,
        conv_params=[(4, (8, 8)), (4, (8, 8))],
        fc_params=[(16, 0.0)],
    )
    torch.manual_seed(5)
    ported = V_pn.ParticleNet(**kwargs)
    torch.manual_seed(5)
    lloca = L_pn.ParticleNet(**kwargs)
    torch.manual_seed(11)
    points, features = torch.randn(3, 3, 9), torch.randn(3, 6, 9)
    frames = Frames(is_identity=True, device=points.device, dtype=points.dtype, shape=(3 * 9,))
    ported.eval()
    lloca.eval()
    with torch.no_grad():
        # v is supplied but must be IGNORED at the deltaR default -- that is the point
        a = ported(points, features, frames=frames, v=torch.randn(3, 4, 9))
        b = lloca(points, features, frames=frames)
    assert torch.equal(a, b), f"outputs differ, max |delta| {(a - b).abs().max():.3e}"


def test_minkowski_changes_the_graph_and_deltaR_does_not():
    """The new metric must actually do something, and only when asked."""
    from lloca.framesnet.frames import Frames

    kwargs = dict(
        input_dims=6,
        hidden_reps_list=["6x0n", "8x0n"],
        num_classes=2,
        conv_params=[(4, (8, 8)), (4, (8, 8))],
        fc_params=[(16, 0.0)],
    )
    torch.manual_seed(5)
    dr = V_pn.ParticleNet(**kwargs)
    torch.manual_seed(5)
    mink = V_pn.ParticleNet(knn_metric="minkowski", **kwargs)
    torch.manual_seed(11)
    points, features = torch.randn(3, 3, 9), torch.randn(3, 6, 9)
    v = torch.randn(3, 4, 9)
    frames = Frames(is_identity=True, device=points.device, dtype=points.dtype, shape=(3 * 9,))
    dr.eval()
    mink.eval()
    with torch.no_grad():
        a = dr(points, features, frames=frames, v=v)
        b = mink(points, features, frames=frames, v=v)
        c = mink(points, features, frames=frames, v=None)  # no momenta -> falls back to deltaR
    assert not torch.equal(a, b), "knn_metric=minkowski produced the deltaR graph"
    assert torch.equal(a, c), "minkowski without v must fall back to the deltaR graph exactly"

    with pytest.raises(ValueError, match="knn_metric"):
        V_pn.ParticleNet(knn_metric="euclidean", **kwargs)


def test_edgeconv_block_matches_the_live_lloca_baseline():
    """Close the chain against the library itself: hybrid == ported == installed lloca.

    The `tag_particlenet` row now runs the in-repo port, so the two tests above compare it to
    the hybrid and to lloca respectively. This one keeps the third edge of the triangle, so a
    change that moved BOTH in-repo files together -- the failure the other two cannot see --
    still fails. Under identity frames and an unmasked plain-L2 kNN the hybrid's EdgeConv must
    be BIT-identical to lloca's, which is what makes "the hybrid's GNN stage is ParticleNet" a
    checkable claim rather than a description. The hybrid's documented extensions (phi-wrapped
    deltaR, padding-aware kNN, explicit self-loop removal, k capping) are all switched off by
    these arguments.
    """
    from lloca.backbone import particlenet as L_pn
    from lloca.framesnet.frames import Frames
    from lloca.reps.tensorreps import TensorReps

    torch.manual_seed(5)
    hybrid = H_pn.EdgeConvBlock(k=4, in_feat=6, out_feats=(8, 8))
    torch.manual_seed(5)
    lloca = L_pn.EdgeConvBlock(k=4, in_reps=TensorReps("6x0n"), out_feats=(8, 8))
    torch.manual_seed(11)
    # 3-d coords so both take the gram-matrix branch (the hybrid wraps phi only for 2-d
    # eta-phi input, a deliberate divergence tested elsewhere)
    points, features = torch.randn(3, 3, 9), torch.randn(3, 6, 9)
    frames = Frames(is_identity=True, device=points.device, dtype=points.dtype, shape=(3 * 9,))
    hp, lp = dict(hybrid.named_parameters()), dict(lloca.named_parameters())
    assert set(hp) == set(lp), f"parameter names diverged: {sorted(set(hp) ^ set(lp))}"
    with torch.no_grad():
        for name in hp:
            hp[name].copy_(lp[name].reshape(hp[name].shape))
    hybrid.eval()
    lloca.eval()
    with torch.no_grad():
        a = hybrid(points, features, knn_metric="deltaR", mask=None, frames=None)
        b = lloca(points, features, frames=frames)
    assert torch.equal(a, b), f"outputs differ, max |delta| {(a - b).abs().max():.3e}"


def test_local_part_pins_the_library_identity_path():
    """The in-repo ParT (tag_ParT's executed class) must stay forward-bit-identical to
    lloca.backbone.particletransformer on the identity-frames path -- the shipped row.
    The local file deliberately diverges on per-particle frames (the trimmer fix, see its
    module docstring), so this pins exactly the path that must never drift silently."""
    import torch
    from lloca.backbone.particletransformer import ParticleTransformer as LibParT
    from lloca.framesnet.nonequi_frames import IdentityFrames

    from experiments.baselines.particletransformer import ParticleTransformer as LocParT

    kw = dict(input_dim=7, num_classes=2, attn_reps="8x0n+2x1n", trim=True,
              use_pre_activation_pair=False, pair_input_dim=4,
              cls_block_params=dict(dropout=0, attn_dropout=0, activation_dropout=0),
              version=1)
    torch.manual_seed(7)
    lib = LibParT(**kw)
    torch.manual_seed(7)
    loc = LocParT(**kw)
    assert list(lib.state_dict()) == list(loc.state_dict())
    assert all(torch.equal(a, b)
               for (_, a), (_, b) in zip(lib.state_dict().items(), loc.state_dict().items()))
    lib.eval()
    loc.eval()
    torch.manual_seed(1)
    B, P = 3, 17
    x = torch.randn(B, 7, P)
    v = torch.randn(B, 4, P)
    mask = (torch.rand(B, 1, P) > 0.2).float()
    mask[..., :3] = 1
    fr = IdentityFrames()(fourmomenta=v.permute(0, 2, 1).reshape(-1, 4), scalars=None,
                          ptr=torch.arange(0, (B + 1) * P, P))
    with torch.no_grad():
        a = lib(x, fr, v=v, mask=mask)
        b = loc(x, fr, v=v, mask=mask)
    assert torch.equal(a, b), f"identity-path drift, max |delta| {(a - b).abs().max():.3e}"
