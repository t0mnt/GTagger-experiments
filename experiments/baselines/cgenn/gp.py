# from https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks
import math

import torch
from torch import nn

from experiments.baselines.cgenn.cliffordalgebra import sparse_gp_tables
from experiments.baselines.cgenn.linear import MVLinear
from experiments.baselines.cgenn.normalization import NormalizationLayer
from experiments.baselines.cgenn.sparse_gp import sparse_geometric_product


class SteerableGeometricProductLayer(nn.Module):
    def __init__(self, algebra, features, include_first_order=True, normalization_init=0):
        super().__init__()

        self.algebra = algebra
        self.features = features
        self.include_first_order = include_first_order

        if normalization_init is not None:
            self.normalization = NormalizationLayer(algebra, features, normalization_init)
        else:
            self.normalization = nn.Identity()
        self.linear_right = MVLinear(algebra, features, features, bias=False)
        if include_first_order:
            self.linear_left = MVLinear(algebra, features, features, bias=True)

        self.product_paths = algebra.geometric_product_paths
        self.register_buffer("_path_idx", self.product_paths.nonzero().T.contiguous(),
                             persistent=False)
        self.gp_impl = getattr(algebra, "gp_impl", "einsum")
        sp_path, sp_val, sp_sel = sparse_gp_tables(algebra, self._path_idx)
        self.register_buffer("_sp_path", sp_path, persistent=False)
        self.register_buffer("_sp_val", sp_val, persistent=False)
        self.register_buffer("_sp_sel", sp_sel, persistent=False)
        self.weight = nn.Parameter(torch.empty(features, self.product_paths.sum()))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, std=1 / (math.sqrt(self.algebra.dim + 1)))

    def _get_weight(self):
        weight = torch.zeros(
            self.features,
            *self.product_paths.size(),
            dtype=self.weight.dtype,
            device=self.weight.device,
        )
        weight[:, self._path_idx[0], self._path_idx[1], self._path_idx[2]] = self.weight
        bsi = self.algebra.blade_subspace_idx
        weight_repeated = (
            weight.index_select(-3, bsi).index_select(-2, bsi).index_select(-1, bsi)
        )
        return self.algebra.cayley * weight_repeated

    def forward(self, input):
        input_right = self.linear_right(input)
        input_right = self.normalization(input_right)

        if self.gp_impl == "sparse":
            # quasigroup gather (lgatr 2.0 sparse_gp) -- see fcgp.forward
            product = sparse_geometric_product(
                input, input_right, self.weight,
                self.algebra, self._sp_path, self._sp_val, self._sp_sel)
        elif self.gp_impl == "matmul":
            # dense outer product + per-feature bmm (lgatr 2.0 dense form)
            weight = self._get_weight()
            nb = self.algebra.n_blades
            outer = input.unsqueeze(-1) * input_right.unsqueeze(-2)
            wf = weight.permute(0, 1, 3, 2).reshape(self.features, nb * nb, nb)
            product = torch.bmm(
                outer.permute(1, 0, 2, 3).reshape(self.features, -1, nb * nb), wf
            ).permute(1, 0, 2)
        else:
            # two-operand chain == opt_einsum's path for "bni,nijk,bnk->bnj" at these
            # shapes ("bnk,bni->bnki" then "bnki,nijk->bnj"); a 3-operand einsum
            # recomputes that path per call and re-specializes the compiled graph per
            # batch shape (see cliffordalgebra.geometric_product). Bit-identity to the
            # recorded fixtures enforced by the BIT gate -- this is the reference path.
            weight = self._get_weight()
            outer = torch.einsum("bnk,bni->bnki", input_right, input)
            product = torch.einsum("bnki,nijk->bnj", outer, weight)

        if self.include_first_order:
            return (
                self.linear_left(input)
                + product
            ) / math.sqrt(2)

        else:
            return product
