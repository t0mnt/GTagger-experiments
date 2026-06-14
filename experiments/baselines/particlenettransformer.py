"""ParticleNet-Transformer hybrid.

A hybrid tagger: a ParticleNet (EdgeConv) graph backbone feeds a
ParticleTransformer-style encoder with pairwise Lorentz interaction features and a
class token. Like the ParT and ParticleNet baselines, this is a non-equivariant
backbone that is made Lorentz-equivariant through LLoCa input canonicalization in
``experiments.tagging.wrappers.ParticleNetParTGraphTransWrapper`` (which inherits
TaggerWrapper); the backbone itself is frame-agnostic.

Adapted from the weaver implementation. The weaver-specific wrapper,
``get_model`` and ``get_loss`` have been removed.

Input convention (channels-first, matches ParT/ParticleNet, NOT the
``(E, px, py, pz)`` convention used elsewhere in this repo):
    points:   (N, 2, P)            kNN coordinates, e.g. (eta, phi)
    features: (N, C, P)            scalar per-particle features
    v:        (N, 4, P)            four-momenta as (px, py, pz, energy)
    mask:     (N, 1, P)            1 for real particles, 0 for padding
"""

import math
import random
import warnings
import copy
import logging
import numpy as np
import torch
import torch.nn as nn
from functools import partial

from lloca.backbone.attention import LLoCaAttention
from lloca.backbone.particlenet import change_local_frame
from lloca.backbone.particletransformer import Block as LLoCaBlock
from lloca.framesnet.frames import Frames
from lloca.reps.tensorreps import TensorReps
from lloca.reps.tensorreps_transform import TensorRepsTransform
from lloca.utils.lorentz import lorentz_eye

_logger = logging.getLogger(__name__)


def knn(x, k, metric='deltaR', mask=None):
    # cap k: a node has at most (num_points - 1) possible neighbours, so this
    # avoids a topk out-of-range crash on small/sparse events (e.g. few-particle
    # jets) where k would otherwise exceed the available points.
    num_points = x.size(-1)
    k = min(k, max(1, num_points - 1))
    if metric == 'minkowski':
        # x: (N, 4, P) in (px, py, pz, E). Neighbours are ranked by the absolute
        # Minkowski interval |(x_i - x_j)^2| with metric diag(-1, -1, -1, +1).
        sig = x.new_tensor([-1.0, -1.0, -1.0, 1.0]).view(1, -1, 1)
        xp = x * sig
        msq = (x * xp).sum(dim=1, keepdim=True)  # (N, 1, P): E^2 - p^2
        gram = torch.matmul(x.transpose(2, 1), xp)  # (N, P, P): <x_i, x_j>_mink
        d2 = msq.transpose(2, 1) + msq - 2 * gram  # (N, P, P): (x_i - x_j)^2_mink
        pairwise_distance = -d2.abs()
        # drop self-loops explicitly (lightlike pairs can also reach |interval|=0)
        eye = torch.eye(num_points, dtype=torch.bool, device=x.device).unsqueeze(0)
        pairwise_distance = pairwise_distance.masked_fill(eye, float('-inf'))
        if mask is not None:
            # never rank a padded particle as a neighbour
            pairwise_distance = pairwise_distance.masked_fill(~mask.bool().unsqueeze(1), float('-inf'))
        idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    else:
        # 'deltaR': L2 distance on the supplied coordinates -- sqrt(d_eta^2 + d_phi^2)
        # for the first layer, or feature-space L2 for the dynamic deeper layers.
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        if mask is not None:
            eye = torch.eye(num_points, dtype=torch.bool, device=x.device).unsqueeze(0)
            pairwise_distance = pairwise_distance.masked_fill(
                eye | ~mask.bool().unsqueeze(1), float('-inf')
            )
            idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
        else:
            idx = pairwise_distance.topk(k=k + 1, dim=-1)[1][:, :, 1:]  # (batch_size, num_points, k)
    return idx


# v1 is faster on GPU
def get_graph_feature_v1(x, k, idx, frames=None, trafo=None):
    batch_size, num_dims, num_points = x.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    fts = x.transpose(2, 1).reshape(-1, num_dims)  # -> (batch_size, num_points, num_dims) -> (batch_size*num_points, num_dims)
    fts = fts[idx, :].view(batch_size, num_points, k, num_dims)  # neighbors: -> (batch_size*num_points*k, num_dims) -> ...
    fts = fts.permute(0, 3, 1, 2).contiguous()  # (batch_size, num_dims, num_points, k)
    x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)
    # LLoCa tensorial message-passing: express neighbour j (in its own local frame)
    # in the centre i's frame before forming the edge feature fts - x (no-op for
    # identity frames / pure-scalar reps). Mirrors lloca.backbone.particlenet.
    if frames is not None:
        fts = change_local_frame(fts, idx, frames, trafo)
    fts = torch.cat((x, fts - x), dim=1)  # ->(batch_size, 2*num_dims, num_points, k)
    return fts


# v2 is faster on CPU
def get_graph_feature_v2(x, k, idx, frames=None, trafo=None):
    batch_size, num_dims, num_points = x.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    fts = x.transpose(0, 1).reshape(num_dims, -1)  # -> (num_dims, batch_size, num_points) -> (num_dims, batch_size*num_points)
    fts = fts[:, idx].view(num_dims, batch_size, num_points, k)  # neighbors: -> (num_dims, batch_size*num_points*k) -> ...
    fts = fts.transpose(1, 0).contiguous()  # (batch_size, num_dims, num_points, k)
    if frames is not None:
        fts = change_local_frame(fts, idx, frames, trafo)

    x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)
    fts = torch.cat((x, fts - x), dim=1)  # ->(batch_size, 2*num_dims, num_points, k)

    return fts


class EdgeConvBlock(nn.Module):
    r"""EdgeConv layer.
    Introduced in "`Dynamic Graph CNN for Learning on Point Clouds
    <https://arxiv.org/pdf/1801.07829>`__".  Can be described as follows:
    .. math::
       x_i^{(l+1)} = \max_{j \in \mathcal{N}(i)} \mathrm{ReLU}(
       \Theta \cdot (x_j^{(l)} - x_i^{(l)}) + \Phi \cdot x_i^{(l)})
    where :math:`\mathcal{N}(i)` is the neighbor of :math:`i`.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    batch_norm : bool
        Whether to include batch normalization on messages.
    """

    def __init__(self, k, in_reps, out_feats, batch_norm=True, activation=True, cpu_mode=False):
        super(EdgeConvBlock, self).__init__()
        self.k = k
        self.batch_norm = batch_norm
        self.activation = activation
        self.num_layers = len(out_feats)
        self.get_graph_feature = get_graph_feature_v2 if cpu_mode else get_graph_feature_v1
        # in_reps types the input features for LLoCa transport (e.g. "32x0n+8x1n"):
        # change_local_frame uses it to rotate neighbours into the centre frame.
        # An int is accepted as a pure-scalar rep ("<n>x0n"), for which the transport
        # is the identity (recovers the plain, non-equivariant EdgeConv).
        in_reps = TensorReps(f"{in_reps}x0n") if isinstance(in_reps, int) else TensorReps(in_reps)
        self.trafo = TensorRepsTransform(in_reps)
        in_feat = in_reps.dim

        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(nn.Conv2d(2 * in_feat if i == 0 else out_feats[i - 1], out_feats[i], kernel_size=1, bias=False if self.batch_norm else True))

        if batch_norm:
            self.bns = nn.ModuleList()
            for i in range(self.num_layers):
                self.bns.append(nn.BatchNorm2d(out_feats[i]))

        if activation:
            self.acts = nn.ModuleList()
            for i in range(self.num_layers):
                self.acts.append(nn.ReLU())

        if in_feat == out_feats[-1]:
            self.sc = None
        else:
            self.sc = nn.Conv1d(in_feat, out_feats[-1], kernel_size=1, bias=False)
            self.sc_bn = nn.BatchNorm1d(out_feats[-1])

        if activation:
            self.sc_act = nn.ReLU()

    def forward(self, points, features, frames=None, knn_metric='deltaR', mask=None):

        topk_indices = knn(points, self.k, metric=knn_metric, mask=mask)
        k = topk_indices.size(-1)  # may be capped below self.k for tiny events
        x = self.get_graph_feature(features, k, topk_indices, frames, self.trafo)

        for conv, bn, act in zip(self.convs, self.bns, self.acts):
            x = conv(x)  # (N, C', P, K)
            if bn:
                x = bn(x)
            if act:
                x = act(x)

        fts = x.mean(dim=-1)  # (N, C, P)

        # shortcut
        if self.sc:
            sc = self.sc(features)  # (N, C_out, P)
            sc = self.sc_bn(sc)
        else:
            sc = features

        return self.sc_act(sc + fts)  # (N, C_out, P)



@torch.jit.script
def delta_phi(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


@torch.jit.script
def delta_r2(eta1, phi1, eta2, phi2):
    return (eta1 - eta2)**2 + delta_phi(phi1, phi2)**2


def to_pt2(x, eps=1e-8):
    pt2 = x[:, :2].square().sum(dim=1, keepdim=True)
    if eps is not None:
        pt2 = pt2.clamp(min=eps)
    return pt2


def to_m2(x, eps=1e-8):
    m2 = x[:, 3:4].square() - x[:, :3].square().sum(dim=1, keepdim=True)
    if eps is not None:
        m2 = m2.clamp(min=eps)
    return m2


def atan2(y, x):
    sx = torch.sign(x)
    sy = torch.sign(y)
    pi_part = (sy + sx * (sy ** 2 - 1)) * (sx - 1) * (-math.pi / 2)
    atan_part = torch.arctan(y / (x + (1 - sx ** 2))) * sx ** 2
    return atan_part + pi_part


def to_ptrapphim(x, return_mass=True, eps=1e-8, for_onnx=False):
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    px, py, pz, energy = x.split((1, 1, 1, 1), dim=1)
    pt = torch.sqrt(to_pt2(x, eps=eps))
    # rapidity = 0.5 * torch.log((energy + pz) / (energy - pz))
    rapidity = 0.5 * torch.log(1 + (2 * pz) / (energy - pz).clamp(min=1e-20))
    phi = (atan2 if for_onnx else torch.atan2)(py, px)
    if not return_mass:
        return torch.cat((pt, rapidity, phi), dim=1)
    else:
        m = torch.sqrt(to_m2(x, eps=eps))
        return torch.cat((pt, rapidity, phi, m), dim=1)


def boost(x, boostp4, eps=1e-8):
    # boost x to the rest frame of boostp4
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    p3 = -boostp4[:, :3] / boostp4[:, 3:].clamp(min=eps)
    b2 = p3.square().sum(dim=1, keepdim=True)
    gamma = (1 - b2).clamp(min=eps)**(-0.5)
    gamma2 = (gamma - 1) / b2
    gamma2.masked_fill_(b2 == 0, 0)
    bp = (x[:, :3] * p3).sum(dim=1, keepdim=True)
    v = x[:, :3] + gamma2 * bp * p3 + x[:, 3:] * gamma * p3
    return v


def p3_norm(p, eps=1e-8):
    return p[:, :3] / p[:, :3].norm(dim=1, keepdim=True).clamp(min=eps)


def pairwise_lv_fts(xi, xj, num_outputs=4, eps=1e-8, for_onnx=False):
    pti, rapi, phii = to_ptrapphim(xi, False, eps=None, for_onnx=for_onnx).split((1, 1, 1), dim=1)
    ptj, rapj, phij = to_ptrapphim(xj, False, eps=None, for_onnx=for_onnx).split((1, 1, 1), dim=1)

    delta = delta_r2(rapi, phii, rapj, phij).sqrt()
    lndelta = torch.log(delta.clamp(min=eps))
    if num_outputs == 1:
        return lndelta

    if num_outputs > 1:
        ptmin = ((pti <= ptj) * pti + (pti > ptj) * ptj) if for_onnx else torch.minimum(pti, ptj)
        lnkt = torch.log((ptmin * delta).clamp(min=eps))
        lnz = torch.log((ptmin / (pti + ptj).clamp(min=eps)).clamp(min=eps))
        outputs = [lnkt, lnz, lndelta]

    if num_outputs > 3:
        xij = xi + xj
        lnm2 = torch.log(to_m2(xij, eps=eps))
        outputs.append(lnm2)

    if num_outputs > 4:
        lnds2 = torch.log(torch.clamp(-to_m2(xi - xj, eps=None), min=eps))
        outputs.append(lnds2)

    # the following features are not symmetric for (i, j)
    if num_outputs > 5:
        xj_boost = boost(xj, xij)
        costheta = (p3_norm(xj_boost, eps=eps) * p3_norm(xij, eps=eps)).sum(dim=1, keepdim=True)
        outputs.append(costheta)

    if num_outputs > 6:
        deltarap = rapi - rapj
        deltaphi = delta_phi(phii, phij)
        outputs += [deltarap, deltaphi]

    assert (len(outputs) == num_outputs)
    return torch.cat(outputs, dim=1)


def build_sparse_tensor(uu, idx, seq_len):
    # inputs: uu (N, C, num_pairs), idx (N, 2, num_pairs)
    # return: (N, C, seq_len, seq_len)
    batch_size, num_fts, num_pairs = uu.size()
    idx = torch.min(idx, torch.ones_like(idx) * seq_len)
    i = torch.cat((
        torch.arange(0, batch_size, device=uu.device).repeat_interleave(num_fts * num_pairs).unsqueeze(0),
        torch.arange(0, num_fts, device=uu.device).repeat_interleave(num_pairs).repeat(batch_size).unsqueeze(0),
        idx[:, :1, :].expand_as(uu).flatten().unsqueeze(0),
        idx[:, 1:, :].expand_as(uu).flatten().unsqueeze(0),
    ), dim=0)
    return torch.sparse_coo_tensor(
        i, uu.flatten(),
        size=(batch_size, num_fts, seq_len + 1, seq_len + 1),
        device=uu.device).to_dense()[:, :, :seq_len, :seq_len]


class PairEmbed(nn.Module):
    def __init__(
            self, pairwise_lv_dim, pairwise_input_dim, dims,
            remove_self_pair=False, use_pre_activation_pair=True, mode='sum',
            normalize_input=True, activation='gelu', eps=1e-8,
            for_onnx=False):
        super().__init__()

        self.pairwise_lv_dim = pairwise_lv_dim
        self.pairwise_input_dim = pairwise_input_dim
        self.is_symmetric = (pairwise_lv_dim <= 5) and (pairwise_input_dim == 0)
        self.remove_self_pair = remove_self_pair
        self.mode = mode
        self.for_onnx = for_onnx
        self.pairwise_lv_fts = partial(pairwise_lv_fts, num_outputs=pairwise_lv_dim, eps=eps, for_onnx=for_onnx)
        self.out_dim = dims[-1]

        if self.mode == 'concat':
            input_dim = pairwise_lv_dim + pairwise_input_dim
            module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
            for dim in dims:
                module_list.extend([
                    nn.Conv1d(input_dim, dim, 1),
                    nn.BatchNorm1d(dim),
                    nn.GELU() if activation == 'gelu' else nn.ReLU(),
                ])
                input_dim = dim
            if use_pre_activation_pair:
                module_list = module_list[:-1]
            self.embed = nn.Sequential(*module_list)
        elif self.mode == 'sum':
            if pairwise_lv_dim > 0:
                input_dim = pairwise_lv_dim
                module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
                for dim in dims:
                    module_list.extend([
                        nn.Conv1d(input_dim, dim, 1),
                        nn.BatchNorm1d(dim),
                        nn.GELU() if activation == 'gelu' else nn.ReLU(),
                    ])
                    input_dim = dim
                if use_pre_activation_pair:
                    module_list = module_list[:-1]
                self.embed = nn.Sequential(*module_list)

            if pairwise_input_dim > 0:
                input_dim = pairwise_input_dim
                module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
                for dim in dims:
                    module_list.extend([
                        nn.Conv1d(input_dim, dim, 1),
                        nn.BatchNorm1d(dim),
                        nn.GELU() if activation == 'gelu' else nn.ReLU(),
                    ])
                    input_dim = dim
                if use_pre_activation_pair:
                    module_list = module_list[:-1]
                self.fts_embed = nn.Sequential(*module_list)
        else:
            raise RuntimeError('`mode` can only be `sum` or `concat`')

    def forward(self, x, uu=None):
        # x: (batch, v_dim, seq_len) -- four-momenta in the local (canonicalized) frames
        # uu: (batch, v_dim, seq_len, seq_len)
        # Returns the interaction bias padded for a prepended CLS token, shape
        # (batch, out_dim=num_heads, P+1, P+1) with a zero CLS row/column. As in the LLoCa
        # ParT the bias is built from the canonicalized local momenta directly.
        assert (x is not None or uu is not None)
        with torch.no_grad():
            if x is not None:
                batch_size, _, seq_len = x.size()
            else:
                batch_size, _, seq_len, _ = uu.size()
            if self.is_symmetric and not self.for_onnx:
                i, j = torch.tril_indices(seq_len, seq_len, offset=-1 if self.remove_self_pair else 0,
                                          device=(x if x is not None else uu).device)
                if x is not None:
                    x = x.unsqueeze(-1).repeat(1, 1, 1, seq_len)
                    xi = x[:, :, i, j]  # (batch, dim, seq_len*(seq_len+1)/2)
                    xj = x[:, :, j, i]
                    x = self.pairwise_lv_fts(xi, xj)
                if uu is not None:
                    # (batch, dim, seq_len*(seq_len+1)/2)
                    uu = uu[:, :, i, j]
            else:
                if x is not None:
                    x = self.pairwise_lv_fts(x.unsqueeze(-1), x.unsqueeze(-2))
                    if self.remove_self_pair:
                        i = torch.arange(0, seq_len, device=x.device)
                        x[:, :, i, i] = 0
                    x = x.view(-1, self.pairwise_lv_dim, seq_len * seq_len)
                if uu is not None:
                    uu = uu.view(-1, self.pairwise_input_dim, seq_len * seq_len)
            if self.mode == 'concat':
                if x is None:
                    pair_fts = uu
                elif uu is None:
                    pair_fts = x
                else:
                    pair_fts = torch.cat((x, uu), dim=1)

        if self.mode == 'concat':
            elements = self.embed(pair_fts)  # (batch, embed_dim, num_elements)
        elif self.mode == 'sum':
            if x is None:
                elements = self.fts_embed(uu)
            elif uu is None:
                elements = self.embed(x)
            else:
                elements = self.embed(x) + self.fts_embed(uu)

        if self.is_symmetric and not self.for_onnx:
            y = torch.zeros(batch_size, self.out_dim, seq_len, seq_len, dtype=elements.dtype, device=elements.device)
            y[:, :, i, j] = elements
            y[:, :, j, i] = elements
        else:
            y = elements.view(-1, self.out_dim, seq_len, seq_len)

        # pad for the prepended CLS token: CLS-CLS / CLS-particle / particle-CLS = 0
        y_padded = torch.zeros(batch_size, self.out_dim, seq_len + 1, seq_len + 1,
                               dtype=y.dtype, device=y.device)
        y_padded[:, :, 1:, 1:] = y
        return y_padded  # (batch, out_dim=num_heads, P+1, P+1)


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # From https://github.com/rwightman/pytorch-image-models/blob/18ec173f95aa220af753358bf860b16b6691edb2/timm/layers/weight_init.py#L8
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


class ParticleNetParTGraphTrans(nn.Module):

    def __init__(self,
                 input_dim,
                 num_classes=None,
                 hidden_reps_list=None,   # per-edge-conv input reps, e.g. ["7x0n","32x0n+8x1n",...]
                 attn_reps="8x0n+2x1n",   # per-head transformer rep; embed_dim = attn_reps.dim * num_heads
                 use_input_concat=True, #change to false if ablation implies so
                 conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],

                 # network configurations
                 pair_input_dim=4,
                 pair_extra_dim=0,
                 remove_self_pair=False,
                 use_pre_activation_pair=True,
                 pair_embed_dims=[64, 64, 64],
                 num_heads=8,
                 num_layers=8,
                 block_params=None,
                 use_fusion=True,
                 use_fts_bn=True,
                 fc_params=[],
                 activation='gelu',
                 bias=True,  # toggle the pairwise additive attention bias (ParT-style U_ij)
                 knn_metric='deltaR',  # first-layer kNN: 'deltaR' (eta-phi L2) or 'minkowski' (4-momenta)
                 #
                 #trim=True,
                 for_inference=False,
                 use_amp=False,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.use_input_concat = use_input_concat

        self.use_fts_bn = use_fts_bn
        if self.use_fts_bn:
            self.bn_fts = nn.BatchNorm1d(input_dim)

        # --- GNN: tensorial EdgeConv stack (LLoCa change_local_frame) ---
        # hidden_reps_list[i] types edge_conv[i]'s input features so the neighbours can be
        # transported into the centre frame. A None entry (or None list) falls back to the
        # pure-scalar rep of the right width, for which the transport is the identity (i.e.
        # the original non-tensorial EdgeConv) -- useful for layer 0 whose input_dim is set
        # by the wrapper. Deeper entries should carry >=1 vector to communicate geometry.
        if hidden_reps_list is None:
            hidden_reps_list = [None] * len(conv_params)
        assert len(hidden_reps_list) == len(conv_params)
        self.edge_convs = nn.ModuleList()
        for idx, layer_param in enumerate(conv_params):
            k, channels = layer_param
            expected = input_dim if idx == 0 else conv_params[idx - 1][1][-1]
            in_reps = hidden_reps_list[idx]
            if in_reps is None:
                in_reps = expected  # pure-scalar fallback (int -> "<expected>x0n")
            reps_dim = (TensorReps(f"{in_reps}x0n") if isinstance(in_reps, int) else TensorReps(in_reps)).dim
            assert reps_dim == expected, f"hidden_reps_list[{idx}].dim={reps_dim} != expected {expected}"
            self.edge_convs.append(EdgeConvBlock(k=k, in_reps=in_reps, out_feats=channels, cpu_mode=for_inference))

        self.use_fusion = use_fusion
        if self.use_fusion:
            in_chn = sum(x[-1] for _, x in conv_params)
            out_chn = int(np.clip((in_chn // 128) * 128, 128, 1024))
            self.fusion_block = nn.Sequential(nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False), nn.BatchNorm1d(out_chn), nn.ReLU())
        else:
            out_chn = conv_params[-1][-1][-1] #if no fusion, output dimension of gnn is the last edgeconv block's

        #self.trimmer = SequenceTrimmer(enabled=trim and not for_inference) | No Trimmer
        self.for_inference = for_inference
        self.use_amp = use_amp

        self.knn_metric = knn_metric
        if knn_metric not in ('deltaR', 'minkowski'):
            raise ValueError(f"knn_metric must be 'deltaR' or 'minkowski', got '{knn_metric}'")

        # --- transformer: LLoCa tensorial attention (transports q/k/v between frames) ---
        self.attn_reps = TensorReps(attn_reps)
        embed_dim = self.attn_reps.dim * num_heads
        self.num_heads = num_heads
        self.head_dim = self.attn_reps.dim
        self.attention = LLoCaAttention(self.attn_reps, num_heads)

        bridge_in_dim = out_chn + input_dim if use_input_concat else out_chn
        self.bridge = nn.Linear(bridge_in_dim, embed_dim)
        self.bridge_norm = nn.LayerNorm(embed_dim)

        default_cfg = dict(embed_dim=embed_dim, num_heads=num_heads, ffn_ratio=4,
                           dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                           activation=activation,
                           scale_fc=True, scale_attn=True, scale_heads=True, scale_resids=True)

        cfg_block = copy.deepcopy(default_cfg)
        if block_params is not None:
            cfg_block.update(block_params)
        _logger.info('cfg_block: %s' % str(cfg_block))

        self.pair_extra_dim = pair_extra_dim
        # `bias` toggles the pairwise additive attention bias: when False, no
        # PairEmbed is built and the transformer runs without interaction features.
        self.pair_embed = PairEmbed(
            pair_input_dim, pair_extra_dim, pair_embed_dims + [num_heads],
            remove_self_pair=remove_self_pair, use_pre_activation_pair=use_pre_activation_pair,
            for_onnx=for_inference) if bias and pair_embed_dims is not None and pair_input_dim + pair_extra_dim > 0 else None
        # The class token is prepended to the particle sequence and goes through every
        # (tensorial) block, GraphTrans-style; it occupies the covariant jet-frame slot in
        # the per-token frames so the readout token stays Lorentz-invariant.
        self.blocks = nn.ModuleList([LLoCaBlock(attention=self.attention, **cfg_block) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)

        if fc_params is not None:
            fcs = []
            in_dim = embed_dim
            for out_dim, drop_rate in fc_params:
                fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate)))
                in_dim = out_dim
            fcs.append(nn.Linear(in_dim, num_classes))
            self.fc = nn.Sequential(*fcs)
        else:
            self.fc = None

        # init
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        trunc_normal_(self.cls_token, std=.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token', }

    def forward(self, points, features, v=None, frames=None, mask=None, uu=None, uu_idx=None, cls_frames=None):

        '''
        points:   (B, 2, P)   kNN coordinates (eta, phi)
        features: (B, C, P)   canonicalized scalar features
        v:        (B, 4, P)   local four-momenta (px, py, pz, E)
        frames:   Frames with matrices (B, P, 4, 4) -- the per-particle local frames
        mask:     (B, 1, P)   1 for real particles, 0 for padding
        cls_frames: Frames (B, 4, 4) -- the covariant jet frame for the prepended CLS
                    token's slot. A prepended token going through the tensorial blocks
                    must live in a covariant frame to stay Lorentz-invariant; the jet
                    frame is that frame. None falls back to identity (e.g. IdentityFrames,
                    where everything is identity anyway).
        '''

        if mask is None:
            mask = (features.abs().sum(dim=1, keepdim=True) != 0)  # (B, 1, P)
        else:
            mask = mask.bool()

        points = points * mask
        features = features * mask
        coord_shift = (mask == 0) * 1e9
        B, _, P = features.size()

        # per-particle frames flattened (B*P, 4, 4) for the EdgeConv change_local_frame
        frames_flat = frames.reshape(-1, 4, 4) if frames is not None else None

        with torch.no_grad():  # extra pairwise feature handling
            if not self.for_inference and uu_idx is not None:
                uu = build_sparse_tensor(uu, uu_idx, P)  # rebuild dense (B, C', P, P)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            fts = self.bn_fts(features) * mask if self.use_fts_bn else features
            mask_knn = mask.squeeze(1)  # (B, P): padded particles excluded from the kNN
            outputs = []
            for idx, conv in enumerate(self.edge_convs):
                # first layer: static graph from the input geometry (eta-phi or, for
                # 'minkowski', the four-momenta); deeper layers: dynamic feature graph.
                # frames_flat drives the tensorial transport of neighbours into the centre.
                if idx == 0 and self.knn_metric == 'minkowski' and v is not None:
                    pts, metric = v + coord_shift, 'minkowski'
                else:
                    pts, metric = (points if idx == 0 else fts) + coord_shift, 'deltaR'
                fts = conv(pts, fts, frames=frames_flat, knn_metric=metric, mask=mask_knn) * mask
                if self.use_fusion:
                    outputs.append(fts)
            if self.use_fusion:
                fts = self.fusion_block(torch.cat(outputs, dim=1)) * mask

            bridge_in = torch.cat([fts, features], dim=1) if self.use_input_concat else fts
            x = bridge_in.permute(0, 2, 1)                  # (B, P, in_dim)
            x = self.bridge_norm(self.bridge(x))            # (B, P, embed_dim)

            attn_mask = None
            if (v is not None or uu is not None) and self.pair_embed is not None:
                attn_mask = self.pair_embed(v, uu)          # (B, num_heads, P+1, P+1)

            # prepend the class token, then run the tensorial blocks over [CLS, particles]
            cls = self.cls_token.expand(B, 1, -1)
            x = torch.cat([cls, x], dim=1)                  # (B, P+1, embed_dim)

            pad = ~mask.squeeze(1)                          # (B, P), True = padded
            padding_mask = torch.cat([torch.zeros_like(pad[:, :1]), pad], dim=1)  # CLS valid

            # per-token frames including the CLS slot (the covariant jet frame, so the
            # prepended token stays invariant); LLoCaAttention transports every token.
            if frames is not None:
                if cls_frames is not None:
                    cls_mat = cls_frames.matrices.to(frames.dtype)
                    cls_is_id = cls_frames.is_identity
                else:
                    cls_mat = lorentz_eye((B,), device=frames.device, dtype=frames.dtype)
                    cls_is_id = True
                seq_mat = torch.cat([cls_mat.unsqueeze(1), frames.matrices], dim=1)  # (B, P+1, 4, 4)
                frames_seq = Frames(
                    seq_mat,
                    is_global=frames.is_global and cls_is_id,
                    is_identity=frames.is_identity and cls_is_id,
                )
                self.attention.prepare_frames(frames_seq)

            for block in self.blocks:
                x = block(x, x_cls=None, padding_mask=padding_mask, attn_mask=attn_mask)

            x_cls = self.norm(x[:, 0])

            if self.fc is None:
                return x_cls
            output = self.fc(x_cls)
            if self.for_inference:
                output = torch.softmax(output, dim=1)

            return output
