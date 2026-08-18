# adapted from https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks
import torch
from torch import nn

from experiments.baselines.cgenn import fcgp as fcgp_mod
from experiments.baselines.cgenn.cliffordalgebra import CliffordAlgebra
from experiments.baselines.cgenn.fcgp import (
    FullyConnectedSteerableGeometricProductLayer,
)
from experiments.baselines.cgenn.gp import SteerableGeometricProductLayer
from experiments.baselines.cgenn.linear import MVLinear
from experiments.baselines.cgenn.mvlayernorm import MVLayerNorm
from experiments.baselines.cgenn.mvsilu import MVSiLU
from experiments.baselines.cgenn.sorted_gather import (
    padded_segment_sum,
    sorted_gather,
    sorted_gather_perm,
)


def get_invariants(algebra, input):
    norms = algebra.qs(input, grades=algebra.grades_list[1:])
    return torch.cat([input[..., :1], *norms], dim=-1)


def psi(p):
    r"""`\psi(p) = Sgn(p) \cdot \log(|p| + 1)`"""
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

    `counts`: optional precomputed (num_segments, 1) receiver-degree tensor. The default
    path rebuilds it here by scattering a full (E, C) tensor of ones -- but the degree is
    identical across channels AND across every reduce call in the step (it depends only on
    `edges`), so callers hoist it once per forward (see CGENN.forward). Bit-identical:
    degrees are small exact integers whatever the summation order, and dividing by a
    (num_segments, 1) broadcast applies the same divisor values elementwise that the
    (num_segments, C) tensor did. The BIT gate + hybrid pins verify this against the
    pre-hoist recordings.
    """
    if counts is not None:
        # Phase 2.2b (docs/cgenn-compile.md): receivers arrive SORTED from both edge
        # builders (structural: arange-expansion / row-major nonzero; machine-checked by
        # tests/experiments/test_edge_builders.py), so the aggregation is a contiguous
        # segment sum -- torch.segment_reduce replaces the index_add_ scatter. Bit-equal
        # to index_add_ on CPU for sorted ids INCLUDING empty segments (padded nodes),
        # and deterministic on CUDA where index_add_'s atomics are not
        # (utils/seg_reduce_probe.py: green on torch 2.13 CPU, public 2.8 CPU, and the
        # NGC build on both A6000 and H100). Adopted after the H100 profile ranked this
        # scatter at 25-27% of CUDA time post-2.2a. `counts` are the hoisted receiver
        # degrees -- exact integers, so the cast is exact and sums to E.
        # FLASH-3 step 2: with slot/K threaded (per-edge in-segment rank + the static
        # degree bound), the sum runs as the padded scatter-write instead --
        # segment_reduce host-reads its lengths on every CUDA call, the attributed
        # source of the profiled sync wall (docs, step 2). TOL vs segment_reduce.
        if slot is not None:
            result = padded_segment_sum(data, segment_ids, slot, num_segments, K)
        else:
            lengths = counts.view(-1).to(torch.int64)
            # unsafe=True (Opus finding, adopted): the default validates lengths with
            # per-call HOST READS -- syncs re-checking an invariant true by
            # construction and pinned by the executable contracts
            # (tests/experiments/test_sorted_gather.py). Same kernel, BIT-identical.
            result = torch.segment_reduce(data, "sum", lengths=lengths, axis=0,
                                          unsafe=True)
        return result / counts.clamp(min=1)
    result = data.new_zeros((num_segments, data.size(1)))
    result.index_add_(0, segment_ids, data)
    counts = data.new_zeros((num_segments, data.size(1)))
    counts.index_add_(0, segment_ids, torch.ones_like(data))
    return result / counts.clamp(min=1)


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
        # the stable-sort permutation (sorted_gather_perm, FLASH-2) -- j is unsorted, but
        # a fixed perm + segment sum is still deterministic where autograd's atomics were
        # not. Both fall back to plain autograd when the extras are None. agg_slot/agg_k
        # (FLASH-3 step 2) route every receiver-side segment sum through the padded
        # scatter-write instead of torch.segment_reduce's host-reading fallback.
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
            weights = self.psi_x(m_h).view(len(m_h), self.out_features_x, self.algebra.n_subspaces)
            weights = weights.index_select(2, self.algebra.blade_subspace_idx)
            m_x = m_x * torch.sigmoid(weights)

        x_red = self.reduce(m_x.flatten(1), i, num_segments=x.size(0),
                            counts=edge_counts, slot=agg_slot, K=agg_k
                            ).view(len(x), *m_x.shape[1:])

        if m_h is not None:
            h_red = self.reduce(m_h, i, num_segments=h.size(0), counts=edge_counts,
                                slot=agg_slot, K=agg_k)
        else:
            h_red = None

        x_u = self.update_x(x, x_red, node_attr_x)
        u_invariants = get_invariants(self.algebra, x).flatten(1)

        if h_red is not None:
            h_u = self.update_h(h, h_red, u_invariants, node_attr_h)

        if self.use_invariants_to_update:
            weights = self.chi_x(h_u).view(len(h_u), self.out_features_x, self.algebra.n_subspaces)

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


class CGENN(nn.Module):
    def __init__(
        self,
        in_features_h: int = 2,
        hidden_features_h: int = 72,
        in_features_x: int = 1,
        hidden_features_x: int = 8,
        decoder_features=64,
        n_outputs=1,
        n_layers=4,
        dropout=0.2,
        use_invariants_to_update=True,
        use_invariant_network=True,
        normalization_init=None,
        residual=False,
        aggregation="mean",
        layer_type="fc",
        gp_impl="einsum",
    ):
        super().__init__()

        if not use_invariant_network:
            in_features_h = 0
            hidden_features_h = 0

        self.in_features_h = in_features_h
        self.hidden_features_h = hidden_features_h
        self.in_features_x = in_features_x
        self.hidden_features_x = hidden_features_x
        self.use_invariant_network = use_invariant_network

        self.algebra = CliffordAlgebra((1.0, -1.0, -1.0, -1.0))
        # geometric-product contraction used by the weighted GP layers (fcgp/gp):
        #   einsum -- the reference two-operand chains, bit-identical to the recorded
        #             fixtures (BIT gate); the default.
        #   matmul -- dense outer product + one GEMM (lgatr 2.0's dense form).
        #   sparse -- quasigroup gather over the 256 nonzero cayley entries, 16x fewer
        #             MACs (lgatr 2.0's sparse_gp, adapted to per-path weights).
        # matmul/sparse reorder the same arithmetic -> TOL-class (docs/cgenn-compile.md);
        # layers read this off the shared algebra instance at construction.
        if gp_impl not in ("einsum", "matmul", "sparse", "flash"):
            raise ValueError(f"gp_impl must be einsum|matmul|sparse|flash, got {gp_impl!r}")
        self.algebra.gp_impl = gp_impl
        self.n_layers = n_layers
        self.embedding_h = nn.Linear(in_features_h, hidden_features_h)
        self.embedding_x = MVLinear(self.algebra, in_features_x, hidden_features_x, subspaces=False)
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
                    use_invariants_to_update=use_invariants_to_update,
                    normalization_init=normalization_init,
                    residual=residual,
                    aggregation=aggregation,
                    layer_type=layer_type,
                    node_attr_h=in_features_h,
                    node_attr_x=in_features_x,
                    edge_attr_x=3 * in_features_x,
                )
                for i in range(n_layers)
            ]
        )
        self.graph_dec = nn.Sequential(
            nn.Linear(
                hidden_features_h + hidden_features_x * self.algebra.n_subspaces,
                decoder_features,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_features, n_outputs),
        )  # head network

    @torch.jit.ignore
    def no_weight_decay(self):
        return cgenn_gain_and_bias_names(self)

    def forward(
        self,
        h,
        x,
        edge_attr_h,
        edge_attr_x,
        node_attr_h,
        node_attr_x,
        edges,
        node_mask,
        knn_k=None,
    ):
        # node_mask arrives DENSE, (batch, n_nodes, 1): the padded-node count is read off
        # this tensor's shape below, so under torch.compile(dynamic=True) it is a symbolic
        # size. The previous python-int n_nodes argument specialized the graph per value --
        # one recompilation for every distinct padded length, which is exactly what the
        # RECOMP gate forbids.
        if not self.use_invariant_network:
            h = None

        if h is not None:
            h = self.embedding_h(h)
        if x is not None:
            x = self.embedding_x(x)

        # Receiver degrees, once per forward: every CGL's mean aggregation divided by a
        # count it rebuilt per call by scattering a full (E, C) ones tensor -- 2 scatters
        # per layer for a value that depends only on `edges`. One (E, 1) scatter here
        # replaces all of them. Bit-identical (exact small integers, then the same divisor
        # values broadcast; see unsorted_segment_mean) -- the BIT gate verifies against
        # the pre-hoist fixtures. Ones-summation order cannot change the result, so this
        # is also degree-count determinism on CUDA for free.
        ref = x if x is not None else h
        i_recv = edges[0]
        edge_counts = ref.new_zeros(ref.size(0), 1).index_add_(
            0, i_recv, ref.new_ones(i_recv.size(0), 1))
        # Sender-side twin (FLASH-2): the stable-sort permutation of the (unsorted) sender
        # index + its degree vector, once per forward, so every CGL's x[j]/h[j] backward is
        # a deterministic gather+segment-sum instead of an atomic scatter-add (the other
        # half of the 27.9% index_put kernel). Forward stays x[idx], so pins stand.
        j_send = edges[1]
        send_perm = torch.argsort(j_send, stable=True)
        send_counts = ref.new_zeros(ref.size(0), 1).index_add_(
            0, j_send, ref.new_ones(j_send.size(0), 1))
        # FLASH-3 step 2: per-edge rank within its (sorted) receiver segment + the
        # static degree bound from the caller (the wrapper knows its builder's bound:
        # fully-connected dense -> n_nodes - 1, a python int, no host read). With these,
        # every receiver-side segment sum runs as the padded scatter-write instead of
        # torch.segment_reduce, whose CUDA path host-reads lengths per call -- the
        # attributed source of the profiled sync wall. knn_k=None keeps segment_reduce.
        if knn_k is not None:
            counts_long = edge_counts.view(-1).long()
            offsets = torch.cumsum(counts_long, 0) - counts_long
            agg_slot = (torch.arange(i_recv.size(0), device=i_recv.device)
                        - offsets.index_select(0, i_recv))
            agg_k = knn_k  # int (or SymInt under dynamic compile) -- never int()-cast:
            # that would re-specialize the graph per value (the RECOMP gate's ban)
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

        invariants = get_invariants(self.algebra, x).flatten(1)

        if h is not None:
            h = torch.cat([h, invariants], dim=1)
        else:
            h = invariants

        h = h * node_mask.reshape(-1, 1)
        h = h.view(
            -1,
            node_mask.shape[1],
            self.hidden_features_h + self.hidden_features_x * self.algebra.n_subspaces,
        )
        h = torch.mean(h, dim=1)  # average over point cloud
        #quirk from official repo kept, this divides by the padded batch max n_nodes, not each jet's true multiplicity so the readout depends on padding
        pred = self.graph_dec(h)
        return pred
