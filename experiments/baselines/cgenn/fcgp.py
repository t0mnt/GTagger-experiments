# from https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks
import math

import torch
from torch import nn

from experiments.baselines.cgenn.cliffordalgebra import sparse_gp_tables
from experiments.baselines.cgenn.linear import MVLinear
from experiments.baselines.cgenn.normalization import NormalizationLayer
from experiments.baselines.cgenn.sparse_gp import sparse_geometric_product
from experiments.baselines.cgenn.flash_kernels_p1m3 import fcgp as fcgp_flash
from .autocast import minimum_autocast_precision


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
    def forward(self, input):
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
            return (self.linear_left(input) + product) / math.sqrt(2)
        else:
            return product
