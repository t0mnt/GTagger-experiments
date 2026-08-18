

import functools
import itertools
import math
import operator
import torch
from torch import nn

# CGENN machinery: imported from the single source of truth rather than duplicated here.
# The copies this replaces had already drifted twice -- the b() device fix and the
# _as_int_grades coercion each reached only one of the two -- which is why the import is
# the right end state. BIT (docs/cgenn-compile.md, Stage 3) proves the
# swap changed nothing.
from experiments.baselines.cgenn.cliffordalgebra import (
    CliffordAlgebra,
    ShortLexBasisBladeOrder,
    construct_gmt,
    sparse_gp_tables,
)
from experiments.baselines.cgenn.linear import MVLinear
from experiments.baselines.cgenn.normalization import NormalizationLayer
from experiments.baselines.cgenn.mvsilu import MVSiLU
from experiments.baselines.cgenn.mvlayernorm import MVLayerNorm
from experiments.baselines.cgenn.gp import SteerableGeometricProductLayer
from experiments.baselines.cgenn.sorted_gather import (
    padded_segment_sum,
    sorted_gather,
    sorted_gather_perm,
)
from experiments.baselines.cgenn import fcgp as fcgp_mod
from experiments.baselines.cgenn.fcgp import FullyConnectedSteerableGeometricProductLayer
from experiments.baselines.cgenn.utils import unsqueeze_like

from lgatr import (
    LGATr,
    embed_vector,
    extract_scalar,
    get_num_spurions,
    get_spurions,
)


# Inspired by https://github.com/pygae/clifford
# copied from the itertools docs











EPS = 1e-6








def cgenn_gain_and_bias_names(module):
    """Fully-qualified names of CGENN's activation/normalization gains and biases.

    These are identity-initialised shape parameters of a nonlinearity or a norm -- MVSiLU's
    ``sigmoid(a * norms + b)`` gate (a=1, b=0) and the two normalization gains -- not weights.
    Decaying them expresses a prior toward deleting the nonlinearity rather than toward a
    simpler function: at a=0 the MVSiLU gate collapses to the constant sigmoid(b). They are
    missed by both of the optimizer's structural rules (ndim>1, and named ``.a``/``.b`` rather
    than ``.bias``), so they are declared here instead. The official CGENN top-tagging recipe
    runs Adam with no weight decay at all, so exempting them restores the reference behaviour
    for exactly these parameters while this repo's weight_decay still applies to real weights.

    Computed by walking the module tree, never hardcoded, so adding a block cannot silently
    drop a parameter from the exemption.
    """
    names = set()
    for mod_name, mod in module.named_modules():
        cls = type(mod).__name__
        if cls == "MVSiLU":
            attrs = ("a", "b")
        elif cls in ("NormalizationLayer", "MVLayerNorm"):
            attrs = ("a",)
        else:
            continue
        for attr in attrs:
            param = getattr(mod, attr, None)
            if param is not None:
                names.add(f"{mod_name}.{attr}" if mod_name else attr)
    return names


def get_invariants(algebra, input):
    norms = algebra.qs(input, grades=algebra.grades_list[1:])
    return torch.cat([input[..., :1], *norms], dim=-1)

def psi(p):
    """`\psi(p) = Sgn(p) \cdot \log(|p| + 1)`"""
    return torch.sign(p) * torch.log(torch.abs(p) + 1)

def unsorted_segment_sum(data, segment_ids, num_segments):
    r"""Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
    Adapted from https://github.com/vgsatorras/egnn.
    """
    result = data.new_zeros((num_segments, data.size(1)))
    result.index_add_(0, segment_ids, data)
    return result

def unsorted_segment_mean(data, segment_ids, num_segments, counts=None, slot=None, K=None):
    r"""Custom PyTorch op to replicate TensorFlow's `unsorted_segment_mean`.
    Adapted from https://github.com/vgsatorras/egnn.

    `counts`: optional precomputed (num_segments, 1) receiver-degree tensor -- see the
    package twin (experiments/baselines/cgenn/cgenn.py) for the full rationale. Hoisted
    once per CGENN block instead of rebuilt here by a full (E, C) ones scatter per call;
    bit-identical (exact small integers, same divisor values broadcast).
    """
    if counts is not None:
        # Phase 2.2b segment-sum swap -- full rationale in the package twin
        # (experiments/baselines/cgenn/cgenn.py). Sorted receivers (machine-checked by
        # tests/experiments/test_edge_builders.py) make this bit-equal on CPU and
        # deterministic on CUDA; adopted on the H100 profile's 25-27% scatter share.
        # FLASH-3 step 2: with slot/K threaded, the sum runs as the padded
        # scatter-write instead (package twin holds the rationale). TOL.
        if slot is not None:
            result = padded_segment_sum(data, segment_ids, slot, num_segments, K)
        else:
            lengths = counts.view(-1).to(torch.int64)
            result = torch.segment_reduce(data, "sum", lengths=lengths, axis=0)
        return result / counts.clamp(min=1)
    result = data.new_zeros((num_segments, data.size(1)))
    result.index_add_(0, segment_ids, data)
    counts = data.new_zeros((num_segments, data.size(1)))
    counts.index_add_(0, segment_ids, torch.ones_like(data))
    return result / counts.clamp(min=1)

def _pairwise_deltaR(points_part):
    """ΔR = sqrt(Δη² + Δφ²) with circular φ wrap."""
    eta_diff = points_part[:, None, 0] - points_part[None, :, 0]
    phi_diff = torch.abs(points_part[:, None, 1] - points_part[None, :, 1])
    phi_diff = torch.min(phi_diff, 2 * math.pi - phi_diff)
    return torch.sqrt(eta_diff**2 + phi_diff**2 + 1e-8)


def _pairwise_minkowski(p4_part):
    """
    Lorentz-invariant distance: sqrt(|Δp²| + ε)
    where Δp² = ΔE² - Δpx² - Δpy² - Δpz²  (signature +,-,-,-)
    p4_part: (N, 4) tensor in (E, px, py, pz) order
    """
    diff = p4_part[:, None, :] - p4_part[None, :, :]   # (N, N, 4)
    # Minkowski quadratic form with metric (+,-,-,-)
    mink = (diff[..., 0] ** 2
            - diff[..., 1] ** 2
            - diff[..., 2] ** 2
            - diff[..., 3] ** 2)
    return torch.sqrt(torch.abs(mink) + 1e-8)


def generate_edges_vectorized(mask, points, k, M, device,
                              metric="deltaR", fourmomenta=None):
    """Directed, fully-batched kNN edges. Each real particle connects to its k
    nearest real neighbours: edge (i -> j) means j is a neighbour of i, with i the
    receiver (aggregation index in CGENN) and j the sender. No symmetrization, no
    per-jet Python loop. Assumes M == P (the dense particle count).

    Returns COO edge_index (2, E), node ids offset by b*M into the flat B*M space,
    rows = [receivers ; senders].
    """
    B, P = mask.shape
    mask_bool = mask.bool()

    if k is None:  # fully connected within each jet, no self-loops
        pair = mask_bool[:, :, None] & mask_bool[:, None, :]
        pair = pair & ~torch.eye(P, dtype=torch.bool, device=device)[None]
        b, i, j = pair.nonzero(as_tuple=True)
        return torch.stack([b * M + i, b * M + j])

    # ---- pairwise distance for the whole batch: (B, P, P) ----
    # distances follow the input dtype: a .float() downcast here would silently
    # defeat use_float64 runs (near-tied intervals re-rank under transforms)
    if metric == "minkowski":
        # Δp² = m_i² + m_j² - 2<p_i,p_j>, via a gram matrix (no (B,P,P,4) tensor)
        p4 = fourmomenta
        sig = p4.new_tensor([1.0, -1.0, -1.0, -1.0])              # (E, px, py, pz) metric
        msq = (p4 * p4 * sig).sum(-1)                             # (B, P)
        gram = torch.bmm(p4 * sig, p4.transpose(1, 2))           # (B, P, P)
        dist = torch.sqrt((msq[:, :, None] + msq[:, None, :] - 2 * gram).abs() + 1e-8)
    else:  # deltaR with phi wrap
        eta, phi = points[..., 0], points[..., 1]
        deta = eta[:, :, None] - eta[:, None, :]
        dphi = (phi[:, :, None] - phi[:, None, :]).abs()
        dphi = torch.minimum(dphi, 2 * math.pi - dphi)
        dist = torch.sqrt(deta ** 2 + dphi ** 2 + 1e-8)

    # forbid self-loops and senders that are padded particles
    eye = torch.eye(P, dtype=torch.bool, device=device)[None]
    dist = dist.masked_fill(eye | (~mask_bool)[:, None, :], float("inf"))

    k_actual = min(k, P - 1)
    nbr = dist.topk(k_actual, dim=-1, largest=False).indices      # (B, P, k) senders per receiver

    offset = (torch.arange(B, device=device) * M)[:, None, None]
    recv = (torch.arange(P, device=device)[None, :, None] + offset).expand(B, P, k_actual).reshape(-1)
    send = (nbr + offset).reshape(-1)

    # keep edges with both endpoints real (padded senders in sparse jets just get fewer
    # edges). Also drop self-loops: on jets with n_real<=k, topk fills from the tied +inf
    # pool (self + padded); padded fills fail the realness filter but a self fill has both
    # endpoints real and would survive as a spurious, tie-break-dependent i->i message --
    # the same sparse-jet leak fixed in the Plain/LorentzNet static-kNN nbr_masks.
    valid = mask_bool.reshape(-1)
    keep = valid[recv] & valid[send] & (recv != send)
    return torch.stack([recv[keep], send[keep]])

class CGLayer(nn.Module):
    def __init__(
        self,
        algebra,
        in_features_x,
        hidden_features_x,
        out_features_x,
        in_features_h,
        hidden_features_h,
        out_features_h,
        edge_attr_x=3,
        edge_attr_h=0,
        node_attr_x=2,
        node_attr_h=2,
        aggregation="mean",
        use_invariants_to_update=True,
        residual=False,
        normalization_init=None,
        layer_type="fc",
    ):
        super().__init__()
        self.edge_attr_x = edge_attr_x
        self.algebra = algebra
        invariants_h = out_features_x * self.algebra.n_subspaces
        f_in_h = 3 * in_features_h
        self.phi_h = nn.Sequential(
            nn.Linear(
                f_in_h + edge_attr_h + invariants_h,
                hidden_features_h,
                bias=False,
            ),
            nn.BatchNorm1d(hidden_features_h),
            nn.ReLU(),
            nn.Linear(hidden_features_h, hidden_features_h),
            nn.ReLU(),
        )
        f_in_x = 3 * in_features_x
        if layer_type == "fc":
            self.phi_x = nn.Sequential(
                FullyConnectedSteerableGeometricProductLayer(
                    self.algebra,
                    edge_attr_x + f_in_x,
                    hidden_features_x,
                    normalization_init=normalization_init,
                ),
                MVLayerNorm(self.algebra, hidden_features_x),
            )
            self.theta_x = nn.Sequential(
                FullyConnectedSteerableGeometricProductLayer(
                    self.algebra,
                    node_attr_x + in_features_x + hidden_features_x,
                    out_features_x,
                    normalization_init=normalization_init,
                ),
                MVLayerNorm(self.algebra, out_features_x),
            )
        elif layer_type == "gpmlp":
            self.phi_x = nn.Sequential(
                MVLinear(self.algebra, edge_attr_x + f_in_x, hidden_features_x),
                MVSiLU(self.algebra, hidden_features_x),
                SteerableGeometricProductLayer(
                    self.algebra,
                    hidden_features_x,
                    normalization_init=normalization_init,
                ),
                MVLayerNorm(self.algebra, hidden_features_x),
            )
            self.theta_x = nn.Sequential(
                MVLinear(
                    self.algebra,
                    node_attr_x + in_features_x + hidden_features_x,
                    out_features_x,
                ),
                MVSiLU(self.algebra, out_features_x),
                SteerableGeometricProductLayer(
                    self.algebra,
                    out_features_x,
                    normalization_init=normalization_init,
                ),
                MVLayerNorm(self.algebra, out_features_x),
            )
        else:
            raise ValueError(f"Unknown layer type {layer_type}.")
        self.theta_h = nn.Sequential(
            nn.Linear(
                node_attr_h
                + algebra.n_subspaces * hidden_features_x
                + in_features_h
                + hidden_features_h,
                hidden_features_h,
            ),
            nn.BatchNorm1d(hidden_features_h),
            nn.ReLU(),
            nn.Linear(hidden_features_h, out_features_h),
        )
        if aggregation == "mean":
            self.aggregation = unsorted_segment_mean
        elif aggregation == "sum":
            self.aggregation = unsorted_segment_sum
        else:
            raise ValueError(f"Unknown aggregation {aggregation}")
        self.use_invariants_to_update = use_invariants_to_update
        self.residual = residual
        self.layer_type = layer_type
        self.out_features_x = out_features_x
        self.in_features_x = in_features_x
        self.in_features_h = in_features_h
        self.out_features_h = out_features_h
        self.psi_x = nn.Sequential(
            nn.Linear(hidden_features_h, hidden_features_h),
            nn.ReLU(),
            nn.Linear(hidden_features_h, out_features_x * self.algebra.n_subspaces),
        )
        self.chi_x = nn.Sequential(
            nn.Linear(hidden_features_h, hidden_features_h),
            nn.ReLU(),
            nn.Linear(hidden_features_h, out_features_x * self.algebra.n_subspaces),
        )
        self.aggregation = aggregation

    def reduce(self, input, segment_ids, num_segments, counts=None, slot=None, K=None):
        if self.aggregation == "mean":
            red = unsorted_segment_mean(input, segment_ids, num_segments=num_segments,
                                        counts=counts, slot=slot, K=K)
        elif self.aggregation == "sum":
            red = unsorted_segment_sum(input, segment_ids, num_segments=num_segments)
        else:
            raise ValueError(f"Invalid aggregation function {self.aggregation}.")
        return red

    def message_x(self, x_i, x_j, edge_attr_x=None):
        x_diff = x_i - x_j
        input = [x_i, x_j, x_diff]
        if edge_attr_x is not None:
            input.append(edge_attr_x)
        input = torch.cat(input, dim=1)
        return self.phi_x(input)

    def _message_x_hoisted(self, x, x_i, x_j, i, j, edge_attr_x,
                           edge_counts, send_perm, send_counts,
                           agg_slot=None, agg_k=None):
        # gather-commute hoisting (FLASH-3 step 1): phi_x = [FCGP, MVLayerNorm], and
        # the FCGP's linear_right/linear_left halves of the message are LINEAR in the
        # gathered concat, so they run at NODE level and get gathered afterward
        # (fcgp.message_right_left holds the identity and the TOL class statement).
        # The quadratic GP keeps the plain edge-level concat, built exactly as
        # message_x builds it. CGENN_HOIST=0 restores the original path.
        input = [x_i, x_j, x_i - x_j]
        if edge_attr_x is not None:
            input.append(edge_attr_x)
        input = torch.cat(input, dim=1)
        fcgp, norm = self.phi_x[0], self.phi_x[1]
        right, left = fcgp.message_right_left(
            x, i, j, edge_attr_x, edge_counts, send_perm, send_counts,
            agg_slot, agg_k)
        return norm(fcgp(input, input_right=right, left=left))

    def message_h(self, h_i, h_j, invariants_ij, edge_attr_h=None):
        input = [invariants_ij, h_i, h_j, h_i - h_j]
        if edge_attr_h is not None:
            input.append(edge_attr_h)
        input = torch.cat(input, dim=1)
        return self.phi_h(input)

    def update_x(self, x, x_red, node_attr_x):
        if node_attr_x is not None:
            input = torch.cat([x, x_red, node_attr_x], dim=1)
        else:
            input = torch.cat([x, x_red], dim=1)
        return self.theta_x(input)

    def update_h(self, h, h_red, invariants_i, node_attr_h):
        if node_attr_h is not None:
            input = torch.cat([h, h_red, invariants_i, node_attr_h], dim=1)
        else:
            input = torch.cat(
                [
                    h,
                    h_red,
                    invariants_i,
                ],
                dim=1,
            )
        return self.theta_h(input)

    def forward(self, h, x, edges, node_attr_h, node_attr_x, edge_attr_h, edge_attr_x,
                edge_counts=None, send_perm=None, send_counts=None,
                agg_slot=None, agg_k=None):
        i, j = edges
        # receiver gathers: BIT-identical forward, deterministic segment-sum backward
        # (sorted_gather; i is sorted and edge_counts is its degree vector -- 2.2a/2.2b's
        # own invariant). Sender gathers x[j]/h[j]: same forward, backward routed through
        # the stable-sort permutation (sorted_gather_perm, FLASH-2) -- deterministic
        # segment sum instead of atomic scatter-add. Falls back to plain autograd when
        # the extras are None. agg_slot/agg_k (FLASH-3 step 2) route every receiver-side
        # segment sum through the padded scatter-write instead of segment_reduce.
        x_i = sorted_gather(x, i, edge_counts, agg_slot, agg_k)
        x_j = sorted_gather_perm(x, j, send_perm, send_counts)
        if fcgp_mod._HOIST and self.layer_type == "fc":
            m_x = self._message_x_hoisted(x, x_i, x_j, i, j, edge_attr_x,
                                          edge_counts, send_perm, send_counts,
                                          agg_slot, agg_k)
        else:
            m_x = self.message_x(x_i, x_j, edge_attr_x)
        m_invariants = get_invariants(self.algebra, m_x).flatten(1)
        if h is not None:
            h_j = sorted_gather_perm(h, j, send_perm, send_counts)
            m_h = self.message_h(sorted_gather(h, i, edge_counts, agg_slot, agg_k),
                                 h_j, m_invariants, edge_attr_h)
        else:
            m_h = None
        if self.use_invariants_to_update:
            weights = self.psi_x(m_h).view(
                len(m_h), self.out_features_x, self.algebra.n_subspaces
            )
            weights = weights.index_select(2, self.algebra.blade_subspace_idx)
            m_x = m_x * torch.sigmoid(weights)
        x_red = self.reduce(m_x.flatten(1), i, num_segments=x.size(0),
                            counts=edge_counts).view(
            len(x), *m_x.shape[1:]
        )
        if m_h is not None:
            h_red = self.reduce(m_h, i, num_segments=h.size(0), counts=edge_counts)
        else:
            h_red = None
        x_u = self.update_x(x, x_red, node_attr_x)
        u_invariants = get_invariants(self.algebra, x).flatten(1)
        if h_red is not None:
            h_u = self.update_h(h, h_red, u_invariants, node_attr_h)
        if self.use_invariants_to_update:
            weights = self.chi_x(h_u).view(
                len(h_u), self.out_features_x, self.algebra.n_subspaces
            )
            weights = weights.index_select(2, self.algebra.blade_subspace_idx)
            x_u = x_u * torch.sigmoid(weights)
        if self.residual and self.in_features_h == self.out_features_h:
            h = h_u + h
        else:
            h = h_u
        if self.residual and self.in_features_x == self.out_features_x:
            x = x_u + x
        else:
            x = x_u
        return h, x

class CGENNBackbone(nn.Module):
    def __init__(
        self,
        in_features_h: int = 2,
        hidden_features_h: int = 72,
        in_features_x: int = 1,
        hidden_features_x: int = 8,
        n_layers=4,
        use_invariants_to_update=True,
        normalization_init=None,
        residual=False,
        aggregation="mean",
        layer_type="fc",
        gp_impl="einsum",
    ):
        super().__init__()
        self.in_features_h = in_features_h
        self.hidden_features_h = hidden_features_h
        self.in_features_x = in_features_x
        self.hidden_features_x = hidden_features_x
        self.algebra = CliffordAlgebra((1.0, -1.0, -1.0, -1.0))
        # geometric-product contraction for the weighted GP layers: einsum (BIT reference)
        # | matmul (dense GEMM) | sparse (quasigroup gather; lgatr 2.0 sparse_gp posture).
        # matmul/sparse are TOL-class -- see docs/cgenn-compile.md and baselines/cgenn.
        if gp_impl not in ("einsum", "matmul", "sparse", "flash"):
            raise ValueError(f"gp_impl must be einsum|matmul|sparse|flash, got {gp_impl!r}")
        self.algebra.gp_impl = gp_impl
        self.n_layers = n_layers
        self.embedding_h = nn.Linear(in_features_h, hidden_features_h)
        self.embedding_x = MVLinear(
            self.algebra, in_features_x, hidden_features_x, subspaces=False
        )
        self.CGLs = nn.ModuleList(
            [
                CGLayer(
                    self.algebra,
                    hidden_features_x,
                    hidden_features_x,
                    hidden_features_x,
                    hidden_features_h,
                    hidden_features_h,
                    hidden_features_h,
                    # edge_attr_x = 3 copies (diff, i, j) of the raw input mv channels
                    edge_attr_x=1+ 2 * in_features_x,
                    use_invariants_to_update=use_invariants_to_update,
                    normalization_init=normalization_init,
                    residual=residual,
                    aggregation=aggregation,
                    layer_type=layer_type,
                    node_attr_h=in_features_h,
                    node_attr_x=in_features_x,
                )
                for i in range(n_layers)
            ]
        )

    def forward(
        self,
        h,
        x,
        edges,
        node_attr_h=None,
        node_attr_x=None,
        edge_attr_h=None,
        edge_attr_x=None,
        knn_k=None,
    ):
        h = self.embedding_h(h)
        x = self.embedding_x(x)
        # Receiver degrees once per block -- see the package twin (cgenn.py, CGENN.forward)
        # for the full rationale. One (E, 1) scatter replacing a full (E, C) ones scatter
        # inside every mean reduce; bit-identical, pins verify.
        i_recv = edges[0]
        edge_counts = x.new_zeros(x.size(0), 1).index_add_(
            0, i_recv, x.new_ones(i_recv.size(0), 1))
        # Sender-side twin (FLASH-2): stable-sort permutation + degrees of the unsorted
        # sender index, once per forward -- see the package twin (cgenn.py) for rationale.
        j_send = edges[1]
        send_perm = torch.argsort(j_send, stable=True)
        send_counts = x.new_zeros(x.size(0), 1).index_add_(
            0, j_send, x.new_ones(j_send.size(0), 1))
        # FLASH-3 step 2: per-edge in-segment rank + static degree bound (see the
        # package twin) -- kNN graphs bound receiver degree by k structurally, so the
        # model passes min(k, P-1) with no host read. None keeps segment_reduce.
        if knn_k is not None:
            counts_long = edge_counts.view(-1).long()
            offsets = torch.cumsum(counts_long, 0) - counts_long
            agg_slot = (torch.arange(i_recv.size(0), device=i_recv.device)
                        - offsets.index_select(0, i_recv))
            agg_k = knn_k  # int (or SymInt under dynamic compile) -- never int()-cast
        else:
            agg_slot, agg_k = None, None
        for i in range(self.n_layers):
            h, x = self.CGLs[i](
                h,
                x,
                edges,
                node_attr_x=node_attr_x,
                node_attr_h=node_attr_h,
                edge_attr_x=edge_attr_x,
                edge_attr_h=edge_attr_h,
                edge_counts=edge_counts,
                send_perm=send_perm,
                send_counts=send_counts,
                agg_slot=agg_slot,
                agg_k=agg_k,
            )
        return h, x

class CGENNLGATrGraphTrans(nn.Module):
    """Hybrid CGENN -> L-GATr model with complementary raw+learned features"""
    def __init__(
        self,
        in_s_channels: int,
        hidden_mv_channels: int,
        hidden_s_channels: int,
        num_classes: int,
        num_blocks: int,
        num_heads: int,
        k: int = None,
        cgenn_layers: int = 2,
        cgenn_hidden_h: int = 72,
        cgenn_hidden_x: int = 8,
        cgenn_aggregation: str = "mean",
        cgenn_residual: bool = False,  # official CGENN top-tagging default (tag_cgenn row)
        cgenn_layer_type: str = "fc",
        cgenn_normalization_init=None,  # official CGENN: no NormalizationLayer (tag_cgenn row)
        gp_impl: str = "einsum",  # einsum (BIT ref) | matmul | sparse -- docs/cgenn-compile.md
        concat_original: bool = True,
        use_explicit_edge_features: bool = True,
        beam_spurion: str = "xyplane",
        add_time_spurion: bool = True,
        beam_mirror: bool = True,
        knn_metric: str = "deltaR",
        activation: str = "gelu",
        multi_query: bool = False,
        increase_hidden_channels_attention: int = 2,
        increase_hidden_channels_mlp: int = 2,
        num_hidden_layers_mlp: int = 1,
        norm_elementwise_affine: bool = True,  # v2-native; parity pins set False
        primitives: dict | None = None,        # e.g. {'sparse_gp': False} in parity tier 1
        head_scale: bool = False,
        dropout_prob: float = None,
        checkpoint_blocks: bool = False,
    ):
        super().__init__()
        self.algebra = CliffordAlgebra((1.0, -1.0, -1.0, -1.0))
        self.hidden_mv_channels = hidden_mv_channels
        self.in_s_channels = in_s_channels
        self.concat_original = concat_original
        self.use_explicit_edge_features = use_explicit_edge_features
        if not use_explicit_edge_features:
            # The CGENN stage's CGLayers expect the edge multivectors and re-injected node
            # attributes (edge_attr_x / node_attr_x / node_attr_h nonzero); the forward would
            # pass None -> a shape RuntimeError on the first layer. Fail loudly instead. (The
            # GraphGPS sibling supports the toggle by zeroing those dims at construction.)
            raise NotImplementedError(
                "CGENNLGATrGraphTrans supports only use_explicit_edge_features=True; the CGENN "
                "stage is constructed with the static edge/node attributes always enabled. Use "
                "CGENNLGATrGraphGPS if you need the no-edge-feature ablation."
            )
        self.spurion_kwargs = {
            "beam_spurion": beam_spurion,
            "add_time_spurion": add_time_spurion,
            "beam_mirror": beam_mirror,
        }
        num_spurions = get_num_spurions(
            beam_spurion, add_time_spurion, beam_mirror=beam_mirror
        )
        self.num_spurions = num_spurions
        self.k = k
        self.knn_metric = knn_metric
        if knn_metric not in ("deltaR", "minkowski"):
            raise ValueError(f"knn_metric must be 'deltaR' or 'minkowski', got '{knn_metric}'")

        # Spurions are injected as extra mv input channels (1 particle channel +
        # num_spurions spurion channels), so no spurion tokens enter the graph.
        in_mv_channels_cgenn = 1 + num_spurions

        self.cgenn = CGENNBackbone(
            in_features_h=in_s_channels,
            hidden_features_h=cgenn_hidden_h,
            in_features_x=in_mv_channels_cgenn,
            hidden_features_x=cgenn_hidden_x,
            n_layers=cgenn_layers,
            use_invariants_to_update=True,
            normalization_init=cgenn_normalization_init,
            residual=cgenn_residual,
            aggregation=cgenn_aggregation,
            layer_type=cgenn_layer_type,
            gp_impl=gp_impl,
        )

        # concat_original skips the raw particle kinematic channel (ch 0) only. Spurion
        # channels are excluded: they are global constants (zero batch variance) already
        # folded into every CGENN channel via embedding_x, so concatenating them adds no signal.
        mv_bridge_in = cgenn_hidden_x + 1 if concat_original else cgenn_hidden_x
        self.mv_bridge = MVLinear(self.algebra, mv_bridge_in, hidden_mv_channels, subspaces=True)

        s_bridge_in = cgenn_hidden_h + (in_s_channels if concat_original else 0)  # <-- wider
        self.s_bridge = nn.Linear(s_bridge_in, hidden_s_channels)

        self.cls_mv_scalar = nn.Parameter(torch.zeros(1, 1, hidden_mv_channels))
        torch.nn.init.normal_(self.cls_mv_scalar, std=0.02)
        self.cls_s = nn.Parameter(torch.zeros(1, 1, hidden_s_channels))
        torch.nn.init.normal_(self.cls_s, std=0.02)

        attention = dict(
            multi_query=multi_query,
            num_heads=num_heads,
            attn_ratio=increase_hidden_channels_attention,
            head_scale=head_scale,
        )
        mlp = dict(
            nonlinearity=activation,
            mlp_ratio=increase_hidden_channels_mlp,
            # v2 counts ALL layers: v1 num_hidden_layers=N == v2 num_layers_mlp=N+1
            num_layers_mlp=num_hidden_layers_mlp + 1,
        )
        self.net = LGATr(
            norm_elementwise_affine=norm_elementwise_affine,
            **({"primitives": primitives} if primitives is not None else {}),
            num_blocks=num_blocks,
            in_mv_channels=hidden_mv_channels,
            out_mv_channels=num_classes,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=hidden_s_channels,
            out_s_channels=None,
            hidden_s_channels=hidden_s_channels,
            attention=attention,
            mlp=mlp,
            dropout_prob=dropout_prob,
            checkpoint_blocks=checkpoint_blocks,
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        # both CLS tokens are learnable invariants; excluding them from weight
        # decay is the ParT/ViT convention and does not affect equivariance
        return {
            "cls_mv_scalar",
            "cls_s",
        } | cgenn_gain_and_bias_names(self)

    def build_edges(self, v, mask, points):
        """Static kNN edges from raw inputs -- eager by design (data-dependent nonzero).
        The wrapper calls this OUTSIDE the compiled region so the compiled forward is
        break-free; forward falls back to building them itself when edges is None."""
        fourmomenta_flat = v if (self.knn_metric == "minkowski" and self.k is not None) else None
        return generate_edges_vectorized(
            mask, points, self.k, points.shape[1], v.device,
            metric=self.knn_metric, fourmomenta=fourmomenta_flat,
        )

    def forward(self, x, v, mask, points, edges=None):
   # points-first inputs from the wrapper:
        #   x: (B, P, C)   v: (B, P, 4) [E, px, py, pz]   mask: (B, P)   points: (B, P, 2)

        # Stage 1: Multivector embedding
        fourmomenta_ga = v[:, :, None, :]       # (B, P, 1, 4)  # Stage 1; was v[:, :, None, [3, 0, 1, 2]]
        mv = embed_vector(fourmomenta_ga)               # (B, P, 1, 16)
        s = x                                           # (B, P, C)

        # Stage 2: Inject spurions as extra mv channels (not tokens)
        # Each spurion is broadcast to every particle slot as an additional
        # input channel, giving CGENN access to the symmetry-breaking axes
        # without adding any nodes to the graph.
        device = s.device
        if self.num_spurions > 0:
            spurions = get_spurions(**self.spurion_kwargs).to(device=device, dtype=s.dtype)
            # spurions: (num_spurions, 16)  →  (B, P, num_spurions, 16)
            spurion_channels = spurions[None, None, :, :].expand(
                mv.shape[0], mv.shape[1], -1, -1
            )
            mv = torch.cat([mv, spurion_channels], dim=2)   # (B, P, 1+num_spurions, 16)
        # s and mask are unmodified — no spurion tokens

        B, P, _ = s.shape
        M = P

        # Stage 3: Build graph edges (native dtype: see generate_edges_vectorized)
        if edges is None:
            edges = self.build_edges(v, mask, points)

        # Stage 4: Flatten for CGENN over the dense B*P layout (padded slots included),
        # matching official CGENN, whose theta_h BatchNorm also runs over padded nodes.
        # Edges connect real nodes only, so padding reaches only the scalar BN stats.
        total_nodes = B * M
        h_flat = s.reshape(total_nodes, -1)
        x_flat_raw = mv.reshape(total_nodes, -1, 16)  # (B*P, 1+num_spurions, 16)

        if self.use_explicit_edge_features:
            i, j = edges
            particle_diff = x_flat_raw[i, :1] - x_flat_raw[j, :1]
            edge_attr_x = torch.cat([particle_diff, x_flat_raw[i], x_flat_raw[j]], dim=1)

            node_attr_x = x_flat_raw
            node_attr_h = h_flat
        else:
            edge_attr_x = None
            node_attr_x = None
            node_attr_h = None

        # Stage 5: CGENN layers
        h_flat, x_flat = self.cgenn(
            h_flat,
            x_flat_raw,
            edges,
            node_attr_h=node_attr_h,
            node_attr_x=node_attr_x,
            edge_attr_h=None,
            edge_attr_x=edge_attr_x,
            # FLASH-3 step 2: the kNN builder bounds receiver degree by k -- a static
            # python int (no min() with the symbolic P: that would plant a shape guard
            # under compile(dynamic=True), and k alone is already a valid bound).
            # Fully-connected mode (k=None) keeps segment_reduce: its P-1 bound is
            # symbolic and int()-ing it would re-specialize per padded length.
            knn_k=self.k,
        )

        # Reshape back
        h = h_flat.view(B, M, -1)
        x = x_flat.view(B, M, -1, 16)

        # Stage 6: Linear bridge
        if self.concat_original:
            # Skip-connect raw particle kinematics (channel 0) only. Spurion channels
            # (1..num_spurions) are excluded: fixed constants with no per-particle variance,
            # already encoded in every CGENN channel via embedding_x -- repeating them wastes
            # bridge capacity.
            particle_mv = mv[:, :, :1, :]                  # (B, P, 1, 16)
            x = torch.cat([particle_mv, x], dim=2)         # (B, P, 1+hidden_x, 16)
            h = torch.cat([s, h], dim=2) 

        x_bridge = x.reshape(B * M, -1, 16)
        h_bridge = h.reshape(B * M, -1)

        mv_out = self.mv_bridge(x_bridge).view(B, M, -1, 16)
        s_out = self.s_bridge(h_bridge).view(B, M, -1)

        # Stage 7: Add learnable CLS token. Equivariance-safe: only the SCALAR (grade-0)
        # multivector component is learnable -- scalars are Lorentz-invariant, so a
        # learned constant there transforms trivially (the equivariance suite confirms;
        # a learnable VECTOR token would pick a direction and break it).
        cls_mv = torch.zeros(B, 1, self.hidden_mv_channels, 16, device=device, dtype=s_out.dtype)
        cls_mv[..., 0] = self.cls_mv_scalar.expand(B, 1, -1)
        cls_s = self.cls_s.expand(B, -1, -1)
        cls_mask = torch.ones(B, 1, device=device, dtype=torch.bool)
        mv_out = torch.cat([cls_mv, mv_out], dim=1)
        s_out = torch.cat([cls_s, s_out], dim=1)
        mask = torch.cat([cls_mask, mask], dim=1)

        # Stage 8: L-GATr Transformer
        attn_mask = mask[:, None, None, :]
        out_mv, _ = self.net(mv_out, s_out, attn_mask=attn_mask)

        # Stage 9: Classification from CLS token
        cls_out = out_mv[:, 0]
        output = extract_scalar(cls_out)[..., 0]

        return output
     
