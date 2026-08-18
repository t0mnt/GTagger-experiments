# from https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks
import math
import os

import torch
from torch import nn

from experiments.baselines.cgenn.cliffordalgebra import sparse_gp_tables
from experiments.baselines.cgenn.linear import MVLinear, mv_apply_weight
from experiments.baselines.cgenn.normalization import NormalizationLayer
from experiments.baselines.cgenn.sorted_gather import sorted_gather, sorted_gather_perm
from experiments.baselines.cgenn.sparse_gp import sparse_geometric_product
from experiments.baselines.cgenn.flash_kernels_p1m3 import fcgp as fcgp_flash
from experiments.baselines.cgenn.utils import unsqueeze_like
from .autocast import minimum_autocast_precision

# FLASH-3 step 1 kill switch: CGENN_HOIST=0 reverts the CGL message path to the
# original edge-level linears (the shield pattern -- one env var, no code change).
# Read ONCE at import, like CGENN_SORTED_GATHER.
_HOIST = os.environ.get("CGENN_HOIST", "1") != "0"


class FullyConnectedSteerableGeometricProductLayer(nn.Module):
    def __init__(
        self,
        algebra,
        in_features,
        out_features,
        include_first_order=True,
        normalization_init=0,
    ):
        super().__init__()

        self.algebra = algebra
        self.in_features = in_features
        self.out_features = out_features
        self.include_first_order = include_first_order

        if normalization_init is not None:
            self.normalization = NormalizationLayer(algebra, in_features, normalization_init)
        else:
            self.normalization = nn.Identity()
        self.linear_right = MVLinear(algebra, in_features, in_features, bias=False)
        if include_first_order:
            self.linear_left = MVLinear(algebra, in_features, out_features, bias=True)

        self.product_paths = algebra.geometric_product_paths
        # integer indices of the allowed grade paths: index_put with these is the same values
        # in the same order as the boolean-mask assignment, without rebuilding a mask each
        # forward (and without the runtime nonzero() a bool mask implies under compile)
        self.register_buffer("_path_idx", self.product_paths.nonzero().T.contiguous(),
                             persistent=False)
        self.gp_impl = getattr(algebra, "gp_impl", "einsum")
        sp_path, sp_val, sp_sel = sparse_gp_tables(algebra, self._path_idx)
        self.register_buffer("_sp_path", sp_path, persistent=False)
        self.register_buffer("_sp_val", sp_val, persistent=False)
        self.register_buffer("_sp_sel", sp_sel, persistent=False)
        self.weight = nn.Parameter(torch.empty(out_features, in_features, self.product_paths.sum()))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(
            self.weight,
            std=1 / math.sqrt(self.in_features * (self.algebra.dim + 1)),
        )

    def _get_weight(self):
        weight = torch.zeros(
            self.out_features,
            self.in_features,
            *self.product_paths.size(),
            dtype=self.weight.dtype,
            device=self.weight.device,
        )
        weight[:, :, self._path_idx[0], self._path_idx[1], self._path_idx[2]] = self.weight
        bsi = self.algebra.blade_subspace_idx
        weight_repeated = (
            weight.index_select(-3, bsi).index_select(-2, bsi).index_select(-1, bsi)
        )
        return self.algebra.cayley * weight_repeated

    @minimum_autocast_precision(torch.float32, output="high")
    def message_right_left(self, x, i, j, edge_attr, edge_counts, send_perm, send_counts,
                           agg_slot=None, agg_k=None):
        """Gather-commute hoisting (FLASH-3 step 1): the linear_right / linear_left
        halves of the CGL message, computed at NODE level and gathered, instead of at
        EDGE level after the concat.

        The message input is cat[x_i, x_j, x_i - x_j, e] (both CGL twins build exactly
        this), and the two linears are linear in it, so with the weight sliced along
        the input channels as [W_A | W_B | W_C | W_E]:

            W @ cat[x_i, x_j, x_i - x_j, e]
              = (W_A + W_C) @ x |gathered at i  +  (W_B - W_C) @ x |gathered at j
                + W_E @ e

        The first two matmuls run over N node rows instead of E ~ k*N edge rows --
        the profiled 74-78% mm block's largest members -- and the gathers reuse the
        deterministic SortedGather/SortedGatherPermuted machinery the CGL already
        threads. Weight-slice pre-addition and the gather reorder are reassociation
        only: TOL class (fp64 <= 1e-13 gates in tests/internal/test_hoist_message.py);
        the hybrid BIT pins are re-recorded with this change stated. linear_left's
        bias is added exactly ONCE here, at the edge-level recombination.

        Returns (input_right_pre_norm, left) as (E, ...) tensors; forward() consumes
        them via its input_right= / left= keywords.
        """
        alg = self.algebra
        c = x.shape[1]
        e_ch = self.in_features - 3 * c
        w_r = self.linear_right.weight
        rA = mv_apply_weight(w_r[:, :c] + w_r[:, 2 * c:3 * c], x, alg)
        rB = mv_apply_weight(w_r[:, c:2 * c] - w_r[:, 2 * c:3 * c], x, alg)
        right = (sorted_gather(rA, i, edge_counts, agg_slot, agg_k)
                 + sorted_gather_perm(rB, j, send_perm, send_counts))
        if e_ch:
            right = right + mv_apply_weight(w_r[:, 3 * c:], edge_attr, alg)

        if not self.include_first_order:  # no linear_left exists on this layer
            return right, None
        w_l = self.linear_left.weight
        lA = mv_apply_weight(w_l[:, :c] + w_l[:, 2 * c:3 * c], x, alg)
        lB = mv_apply_weight(w_l[:, c:2 * c] - w_l[:, 2 * c:3 * c], x, alg)
        left = (sorted_gather(lA, i, edge_counts, agg_slot, agg_k)
                + sorted_gather_perm(lB, j, send_perm, send_counts))
        if e_ch:
            left = left + mv_apply_weight(w_l[:, 3 * c:], edge_attr, alg)
        if self.linear_left.bias is not None:
            bias = alg.embed(self.linear_left.bias, self.linear_left.b_dims)
            left = left + unsqueeze_like(bias, left, dim=2)
        return right, left

    @minimum_autocast_precision(torch.float32, output="high")
    def forward(self, input, input_right=None, left=None):
        if input_right is None:
            input_right = self.linear_right(input)
        input_right = self.normalization(input_right)

        if self.gp_impl == "sparse":
            # quasigroup gather (lgatr 2.0 sparse_gp): contract only the 256 nonzero cayley
            # entries -- 16x fewer MACs than the dense forms, no dense weight materialized.
            # Wrapped in a custom autograd Function so the two (B, N, 16, 16) intermediates
            # stay transient instead of being retained for backward (sparse_gp.py).
            product = sparse_geometric_product(
                input, input_right, self.weight,
                self.algebra, self._sp_path, self._sp_val, self._sp_sel)
        elif self.gp_impl == "flash":
            # generated Cl(1,3) Triton contraction (FLASH PLAN v2, step 4): same compact
            # weight, same math -- gated at 3e-16 vs the sparse expression; CUDA runs the
            # kernels, CPU the gated reference composition inside the custom op.
            product = fcgp_flash(input, input_right, self.weight)
        elif self.gp_impl == "matmul":
            # dense outer product + one GEMM (lgatr 2.0 dense form)
            weight = self._get_weight()
            nb = self.algebra.n_blades
            outer = (input.unsqueeze(-1) * input_right.unsqueeze(-2)).flatten(1)
            wf = weight.permute(0, 3, 1, 2, 4).reshape(self.out_features * nb, -1)
            product = (outer @ wf.T).view(-1, self.out_features, nb)
        else:
            # two-operand chain == opt_einsum's path for "bni,mnijk,bnk->bmj" at these
            # shapes ("bnk,bni->bnki" then "bnki,mnijk->bmj"); a 3-operand einsum
            # recomputes that path per call and re-specializes the compiled graph per
            # batch shape (see cliffordalgebra.geometric_product). Bit-identity to the
            # recorded fixtures enforced by the BIT gate -- this is the reference path.
            weight = self._get_weight()
            outer = torch.einsum("bnk,bni->bnki", input_right, input)
            product = torch.einsum("bnki,mnijk->bmj", outer, weight)

        if self.include_first_order:
            if left is None:
                left = self.linear_left(input)
            return (left + product) / math.sqrt(2)
        else:
            return product
