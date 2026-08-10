"""LLoCa Particle Transformer (ParT) -- the in-repo copy the tag_ParT row executes.

Port of lloca.backbone.particletransformer (lloca 1.3.6), following the same pattern as
particlenet.py: the repo executes code it can fix, rather than a library copy it cannot.
The module tree and parameter names are identical to the library class, and the shipped
identity-frames row is forward-bit-identical to it (pinned by
tests/internal/test_duplicated_component_parity.py). Deltas vs the library, all found or
validated by the adversarial review:

1. `for_inference` single-logit heads use sigmoid. Softmax over a 1-wide dim is
   identically 1, so an exported top-tagging head (out=1, BCE-trained) would emit a
   constant; sigmoid is the correct inverse link. Inert in training/eval here.
2. Per-particle frames ride the SequenceTrimmer with x/v/mask AND are re-shaped to
   batch form (B, P, 4, 4) before prepare_frames. The library prepares frames on the
   untrimmed, unpermuted order (crash on trimmed batches; wrong frames on
   permutation-only steps), and flat (B*P, 4, 4) frames -- the framesnet wrapper
   convention -- additionally flatten as (H, B, P) inside LLoCaAttention's head
   expansion while q/k/v flatten as (B, H, P): a systematic token/frame misalignment
   for B > 1. With both fixed, the port passes permutation equivariance at 1e-6 (fp32)
   and Lorentz invariance at 3e-15 (fp64) with per-particle frames; the library fails
   both. Identity/global frames take the library's exact path (bit-parity pinned).
3. `Attention._load_from_state_dict` snapshots keys before its legacy in_proj rename
   loop -- the library mutates the dict while iterating it, which raises (interleaved
   checkpoint layouts) or silently skips keys (blocked layouts).
4. A PN-style clamp-before-sqrt in `pairwise_lv_fts` was evaluated and REJECTED: both
   call sites run under torch.no_grad(), so the NaN-backward it would guard against
   cannot occur, and the forward clamp shifts self-pair diagonal features. The raw
   library sqrt is kept.

Original upstream: https://github.com/hqucms/weaver-core (ParT, arXiv:2202.03772), with
LLoCa's tensorial frame transport (Attention wraps a prepare_frames-d LLoCaAttention).
"""

"""
Paper: "Particle Transformer for Jet Tagging" - https://arxiv.org/abs/2202.03772

We have to do two things to build LLoCa-ParT
- Construct a LLoCaAttention module for the whole transformer that preprocesses the frames
  and is passed to each attention block during initialization.
- When evaluating attention, use the LLoCaAttention module.

More comments:
- We also added an extra clamp in to_ptrapphim to avoid numerical issues from log(0). This case
  might not happen in the original ParT, but it can happen with LLoCa for highly boosted frames.
- For simplicity, we use LLoCaAttention only for the self-attention blocks, and use
  default attention (corresponds to scalar messages only) for the class attention blocks.

You can use 'git diff --no-index' to compare this file with the original particletransformer.py file.
"""

# ruff: noqa

import copy
import math
import random
from collections.abc import Callable
from functools import partial
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from lloca.framesnet.frames import Frames
from lloca.reps.tensorreps import TensorReps
from lloca.backbone.attention import LLoCaAttention


@torch.jit.script
def delta_phi(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


@torch.jit.script
def delta_r2(eta1, phi1, eta2, phi2):
    return (eta1 - eta2) ** 2 + delta_phi(phi1, phi2) ** 2


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


def to_ptrapphim(x, return_mass=True, eps=1e-8):
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    px, py, pz, energy = x.split((1, 1, 1, 1), dim=1)
    pt = torch.sqrt(to_pt2(x, eps=eps))
    # rapidity = 0.5 * torch.log((energy + pz) / (energy - pz))
    rapidity = 0.5 * torch.log((1 + (2 * pz) / (energy - pz).clamp(min=1e-20)).clamp(min=1e-20))
    phi = torch.atan2(py, px)
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
    gamma = (1 - b2).clamp(min=eps) ** (-0.5)
    gamma2 = (gamma - 1) / b2
    gamma2.masked_fill_(b2 == 0, 0)
    bp = (x[:, :3] * p3).sum(dim=1, keepdim=True)
    v = x[:, :3] + gamma2 * bp * p3 + x[:, 3:] * gamma * p3
    return v


def p3_norm(p, eps=1e-8):
    return p[:, :3] / p[:, :3].norm(dim=1, keepdim=True).clamp(min=eps)


def to_energy_momentum(x, return_unit_vector=True):
    energy = x[:, 3:4]
    p2 = x[:, :3].square().sum(dim=1, keepdim=True)
    mom = torch.sqrt(p2)
    if return_unit_vector:
        return energy, mom, x[:, :3] / mom.clamp(min=1e-8)
    else:
        return energy, mom


def to_cos_sin_angles(xi, xj, normed_inputs=False, eps=1e-8):
    if normed_inputs:
        ni, nj = xi, xj
    else:
        ni, nj = p3_norm(xi, eps), p3_norm(xj, eps)
    cos = (ni * nj).sum(dim=1, keepdim=True).clamp(min=-1, max=1)
    sin = torch.linalg.cross(ni, nj, dim=1).norm(dim=1, keepdim=True).clamp(min=0, max=1)
    return cos, sin


def pairwise_lv_fts_pp(xi, xj, num_outputs=4, eps=1e-8):
    pti, rapi, phii = to_ptrapphim(xi, False, eps=None).split((1, 1, 1), dim=1)
    ptj, rapj, phij = to_ptrapphim(xj, False, eps=None).split((1, 1, 1), dim=1)

    # modified this for convenience (only lorentz scalars is most conservative)
    xij = xi + xj
    lnm2 = torch.log(to_m2(xij, eps=eps))
    if num_outputs > 0:
        outputs = [lnm2]

    if num_outputs > 1:
        # NOTE (adversarial review): a PN-style clamp-before-sqrt was evaluated here and
        # REJECTED. Both pairwise_lv_fts call sites run under torch.no_grad() (PairEmbed's
        # dense and sparse paths), so the sqrt'(0) NaN-backward the clamp would guard
        # against cannot occur -- and a forward clamp measurably shifts the self-pair
        # diagonal lnkt features (3.7e-3 at the output). The library's raw sqrt is kept.
        delta = delta_r2(rapi, phii, rapj, phij).sqrt()
        lndelta = torch.log(delta.clamp(min=eps))
        ptmin = torch.minimum(pti, ptj)
        lnkt = torch.log((ptmin * delta).clamp(min=eps))
        lnz = torch.log((ptmin / (pti + ptj).clamp(min=eps)).clamp(min=eps))
        outputs += [lnkt, lnz, lndelta]

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

    assert len(outputs) == num_outputs
    return torch.cat(outputs, dim=1)


def pairwise_lv_fts_ee(xi, xj, num_outputs=6, eps=1e-8):
    # outputs: [lnm2, cos_angle, sin_angle, lnkt, lnz, lnjade]
    lnm2 = torch.log(to_m2(xi + xj, eps=eps))
    outputs = [lnm2]

    if num_outputs > 1:
        ei, pi, ni = to_energy_momentum(xi)
        ej, pj, nj = to_energy_momentum(xj)
        cos_angle, sin_angle = to_cos_sin_angles(ni, nj, normed_inputs=True)
        outputs += [cos_angle, sin_angle]

    if num_outputs > 3:
        pmin = torch.minimum(pi, pj)
        lnkt = torch.log((pmin * sin_angle).clamp(min=eps))
        lnz = torch.log((pmin / (pi + pj).clamp(min=eps)).clamp(min=eps))
        outputs += [lnkt, lnz]

    if num_outputs > 5:
        lnjade = torch.log((ei * ej * (1 - cos_angle)).clamp(min=eps))
        outputs.append(lnjade)

    assert len(outputs) == num_outputs
    return torch.cat(outputs, dim=1)


def build_sparse_tensor(uu, idx, seq_len):
    # inputs: uu (N, C, num_pairs), idx (N, 2, num_pairs)
    # return: (N, C, seq_len, seq_len)
    batch_size, num_fts, num_pairs = uu.size()
    idx = torch.min(idx, torch.ones_like(idx) * seq_len)
    i = torch.cat(
        (
            torch.arange(0, batch_size, device=uu.device)
            .repeat_interleave(num_fts * num_pairs)
            .unsqueeze(0),
            torch.arange(0, num_fts, device=uu.device)
            .repeat_interleave(num_pairs)
            .repeat(batch_size)
            .unsqueeze(0),
            idx[:, :1, :].expand_as(uu).flatten().unsqueeze(0),
            idx[:, 1:, :].expand_as(uu).flatten().unsqueeze(0),
        ),
        dim=0,
    )
    return torch.sparse_coo_tensor(
        i,
        uu.flatten(),
        size=(batch_size, num_fts, seq_len + 1, seq_len + 1),
        device=uu.device,
    ).to_dense()[:, :, :seq_len, :seq_len]


def tril_indices(row, col, offset=0, *, dtype=torch.long, device="cpu"):
    return torch.ones(row, col, dtype=dtype, device=device).tril(offset).nonzero().T


class SequenceTrimmer(nn.Module):
    def __init__(self, enabled=False, target=(0.9, 1.02), warmup_steps=5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.enabled = enabled
        self.target = target
        self.warmup_steps = warmup_steps
        self.register_buffer("_counter", torch.LongTensor([0]), persistent=False)
        # python-int mirror of _counter: branching on the buffer is a tensor-valued jump
        # (a graph break under compile); the int carries the same value, the buffer stays
        # for state compat and is kept in sync. The BRANCH reads _warmed -- a bool that
        # flips exactly once -- because dynamo guards python attributes by VALUE, so
        # branching on the incrementing int recompiles every warmup step (found by
        # guard_fail_fn: `_counter_int == 0`, `== 1`, ...).
        self._counter_int = 0
        self._warmed = False

    def tick(self):
        """Advance the warmup counter; call once per forward, outside compile.

        Mirrors the upstream in-forward branch exactly (increment OR start trimming):
        with warmup_steps=5 the first five forwards stay untrimmed and the sixth is the
        first trimmed one -- _warmed flips on the (warmup_steps+1)-th tick, not the
        warmup_steps-th (that off-by-one was caught in the final audit)."""
        if self.enabled and not self._warmed:
            if self._counter_int < self.warmup_steps:
                self._counter_int += 1
                self._counter.add_(1)
            else:
                self._warmed = True

    def forward(self, x, v=None, mask=None, uu=None, extra=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        # uu: (N, C', P, P)
        # extra: (N, C'', P) or None -- rides along through the same permutation and
        # truncation (used for per-particle frame matrices; adversarial-review fix)
        if mask is None:
            mask = torch.ones_like(x[:, :1])
        mask = mask.bool()

        if self.enabled:
            if not self._warmed:
                # warmup bookkeeping lives in tick(), called by the wrapper OUTSIDE the
                # compiled region: dynamo guards python ints by value, so even reading
                # the counter in-graph recompiles every warmup step
                pass
            else:
                x, v, mask, uu, extra = self._trim(x, v, mask, uu, extra)

        return x, v, mask, uu, extra

    @torch.compiler.disable
    def _trim(self, x, v, mask, uu, extra):
        """The post-warm-up trim, eager by design (same hoist-eager pattern as the
        CGENN-hybrid edge build): random.uniform + torch.quantile + tensor-valued
        branching + data-dependent slice lengths would otherwise re-specialize the
        compiled graph every training step (maxlen is re-randomized per batch). The
        gates trace the un-warmed net, so this regime is disabled explicitly rather
        than left to per-step recompile churn (final audit finding). Body is verbatim
        upstream (+ the ``extra`` rider)."""
        if v is not None:
            if not isinstance(v, (list, tuple)):
                v = [v]
        if self.training:
            q = min(1, random.uniform(*self.target))
            maxlen = torch.quantile(mask.float().sum(dim=-1), q).long()
            rand = torch.rand_like(mask.float())
            rand.masked_fill_(~mask, -1)
            perm = rand.argsort(dim=-1, descending=True)  # (N, 1, P)
            mask = torch.gather(mask, -1, perm)
            x = torch.gather(x, -1, perm.expand_as(x))
            if extra is not None:
                with torch.enable_grad():
                    extra = torch.gather(extra, -1, perm.expand_as(extra))
            if v is not None:
                v = [torch.gather(_v, -1, perm.expand_as(_v)) for _v in v]
            if uu is not None:
                uu = torch.gather(uu, -2, perm.unsqueeze(-1).expand_as(uu))
                uu = torch.gather(uu, -1, perm.unsqueeze(-2).expand_as(uu))
        else:
            maxlen = mask.sum(dim=-1).max()
        maxlen = max(maxlen, 1)
        if maxlen < mask.size(-1):
            mask = mask[:, :, :maxlen]
            x = x[:, :, :maxlen]
            if extra is not None:
                with torch.enable_grad():
                    extra = extra[:, :, :maxlen]
            if v is not None:
                v = [_v[:, :, :maxlen] for _v in v]
            if uu is not None:
                uu = uu[:, :, :maxlen, :maxlen]
        if v is not None:
            if len(v) == 1:
                v = v[0]
        return x, v, mask, uu, extra


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.drop = nn.Dropout(drop)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        hidden = self.drop(hidden)
        return self.w3(hidden)


class Embed(nn.Module):
    def __init__(self, input_dim, dims, normalize_input=True, activation="gelu"):
        super().__init__()

        self.input_bn = nn.BatchNorm1d(input_dim) if normalize_input else None
        module_list = []
        for dim in dims:
            module_list.extend(
                [
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, dim),
                    nn.GELU() if activation == "gelu" else nn.ReLU(),
                ]
            )
            input_dim = dim
        self.embed = nn.Sequential(*module_list)

    def forward(self, x):
        if self.input_bn is not None:
            # x: (batch, embed_dim, seq_len)
            x = self.input_bn(x)
            x = x.transpose(1, 2).contiguous()
        # x: (batch, seq_len, embed_dim)
        return self.embed(x)


def _weighted_batchnorm1d(x, bn, w):
    """``nn.BatchNorm1d`` whose train-mode statistics are computed over a WEIGHTED
    subset of the N axis, instead of over every entry.

    This is what makes the compiled all-pairs twin numerically faithful to the eager
    reference rather than merely fast. The eager path feeds BN a packed list of the
    lower-triangular REAL pairs (``mask.tril(offset).nonzero()``); the twin builds the
    full ``seq_len**2`` grid, so unweighted statistics would additionally include the
    mirrored upper triangle and every padded pair. Passing ``w`` = that same tril-of-real
    mask makes the weighted mean/var over the grid *identical* to the unweighted mean/var
    over the eager multiset.

    Traceable by construction: ``w.sum()`` is a scalar TENSOR, not a shape, so unlike the
    eager ``nonzero`` gather this introduces no data-dependent shapes.

    Eval is unchanged (running statistics, elementwise), which is why the TOL gate was
    green even while training diverged.
    """
    # Mirror nn.BatchNorm's own branch: eval uses running statistics only when they
    # exist. With track_running_stats=False there are no buffers and BN keeps computing
    # batch statistics even in eval, so that case falls through to the weighted path.
    if not bn.training and bn.running_mean is not None:
        return F.batch_norm(x, bn.running_mean, bn.running_var, bn.weight, bn.bias,
                            False, 0.0, bn.eps)
    n = w.sum()
    mean = (x * w).sum(dim=(0, 2)) / n
    xc = x - mean.view(1, -1, 1)
    var = ((xc * xc) * w).sum(dim=(0, 2)) / n
    if bn.training and bn.track_running_stats:
        with torch.no_grad():
            bn.num_batches_tracked.add_(1)
            m = (bn.momentum if bn.momentum is not None
                 else 1.0 / bn.num_batches_tracked.to(x.dtype))
            bn.running_mean.mul_(1 - m).add_(m * mean.detach())
            # running_var takes the UNBIASED estimate, exactly as nn.BatchNorm1d does,
            # while the normalization above uses the biased one
            bn.running_var.mul_(1 - m).add_(m * var.detach() * n / (n - 1))
    y = xc / torch.sqrt(var.view(1, -1, 1) + bn.eps)
    if bn.affine:
        y = y * bn.weight.view(1, -1, 1) + bn.bias.view(1, -1, 1)
    return y


def _embed_weighted(seq, x, w):
    """Run an embed Sequential, swapping every BatchNorm1d for its weighted twin.
    Conv/activation layers are pointwise along N, so applying them to the whole grid and
    weighting only the STATISTICS is exact."""
    for mod in seq:
        x = _weighted_batchnorm1d(x, mod, w) if isinstance(mod, nn.BatchNorm1d) else mod(x)
    return x


class PairEmbed(nn.Module):
    def __init__(
        self,
        pairwise_lv_dim,
        pairwise_input_dim,
        dims,
        pairwise_lv_type="pp",
        remove_self_pair=False,
        use_pre_activation_pair=True,
        normalize_input=True,
        activation="gelu",
        eps=1e-8,
        for_onnx=False,
        sparse_eval=None,
    ):
        super().__init__()

        self.pairwise_lv_dim = pairwise_lv_dim
        self.pairwise_input_dim = pairwise_input_dim
        self.remove_self_pair = remove_self_pair
        self.for_onnx = for_onnx
        self.sparse_eval = (not for_onnx) if sparse_eval is None else sparse_eval
        # set (with sparse_eval=False) by the compile knob / gates: routes the dense path
        # through the all-pairs twin below, which traces seqlen-dynamic
        self.compiled_dense = False
        self.out_dim = dims[-1]

        if pairwise_lv_type == "pp":
            self.is_symmetric = (pairwise_lv_dim <= 5) and (pairwise_input_dim == 0)
            self.pairwise_lv_fts = partial(pairwise_lv_fts_pp, num_outputs=pairwise_lv_dim, eps=eps)
        elif pairwise_lv_type == "ee":
            self.is_symmetric = (pairwise_lv_dim <= 6) and (pairwise_input_dim == 0)
            self.pairwise_lv_fts = partial(pairwise_lv_fts_ee, num_outputs=pairwise_lv_dim, eps=eps)
        else:
            raise RuntimeError("Invalid value for `pairwise_lv_type`: " + pairwise_lv_type)

        if pairwise_lv_dim > 0:
            input_dim = pairwise_lv_dim
            module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
            for dim in dims:
                module_list.extend(
                    [
                        nn.Conv1d(input_dim, dim, 1),
                        nn.BatchNorm1d(dim),
                        nn.GELU() if activation == "gelu" else nn.ReLU(),
                    ]
                )
                input_dim = dim
            if use_pre_activation_pair:
                module_list = module_list[:-1]
            self.embed = nn.Sequential(*module_list)

        if pairwise_input_dim > 0:
            input_dim = pairwise_input_dim
            module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
            for dim in dims:
                module_list.extend(
                    [
                        nn.Conv1d(input_dim, dim, 1),
                        nn.BatchNorm1d(dim),
                        nn.GELU() if activation == "gelu" else nn.ReLU(),
                    ]
                )
                input_dim = dim
            if use_pre_activation_pair:
                module_list = module_list[:-1]
            self.fts_embed = nn.Sequential(*module_list)

    def _forward_dense(self, x, uu=None, mask=None):
        # x: (batch, v_dim, seq_len)
        # uu: (batch, v_dim, seq_len, seq_len)
        assert x is not None or uu is not None
        if self.is_symmetric and not self.for_onnx and self.compiled_dense:
            # compiled twin of the tril-compute-then-mirror below: torch.tril_indices
            # takes only python ints, which pins seq_len to a constant under compile
            # (the real cause behind the old blanket @torch.compiler.disable). The pair
            # features are symmetric in (i, j) and the embed is pointwise per pair, so
            # the full-grid build produces the same values (GEMM rows are independent);
            # the eager default (sparse path) and the onnx/dense-eager path are untouched.
            # detach() instead of torch.no_grad(): a grad-mode transition splits the
            # compiled graph (empty-reason breaks), while detach traces clean and is the
            # exact gradient twin of eager's no_grad -- REQUIRED, not cosmetic: under a
            # LEARNED framesnet the local momenta carry grad, and backward through the
            # raw sqrt at delta==0 self-pairs is 0*inf=NaN into the framesnet (final
            # audit finding, reproduced; eager never backprops the pair features)
            x = x.detach() if x is not None else None
            uu = uu.detach() if uu is not None else None
            bsz = (x if x is not None else uu).shape[0]
            slen = (x if x is not None else uu).shape[-1]
            # statistics weight: the eager reference feeds BN the lower-triangular REAL
            # pairs, so weight the grid by exactly that set (see _weighted_batchnorm1d)
            with torch.no_grad():
                offset = -1 if self.remove_self_pair else 0
                if mask is not None:
                    pair = mask.reshape(bsz, 1, slen, 1) * mask.reshape(bsz, 1, 1, slen)
                else:
                    pair = x.new_ones(bsz, 1, slen, slen)
                w = pair.tril(offset).reshape(bsz, 1, slen * slen).to(
                    (x if x is not None else uu).dtype)
            elements = 0
            if x is not None:
                fts = self.pairwise_lv_fts(x.unsqueeze(-1), x.unsqueeze(-2))
                # embed's BatchNorm1d wants (batch, channels, num_pairs)
                elements = elements + _embed_weighted(
                    self.embed, fts.reshape(bsz, -1, slen * slen), w)
            if uu is not None:
                elements = elements + _embed_weighted(
                    self.fts_embed, uu.reshape(bsz, -1, slen * slen), w)
            y = elements.reshape(bsz, self.out_dim, slen, slen)
            if self.remove_self_pair:
                di = torch.arange(slen, device=y.device)
                y = y.clone()
                y[:, :, di, di] = 0
            return y
        with torch.no_grad():
            if x is not None:
                batch_size, _, seq_len = x.size()
            else:
                batch_size, _, seq_len, _ = uu.size()
            if self.is_symmetric:
                tril_indices_fn = tril_indices if self.for_onnx else torch.tril_indices
                i, j = tril_indices_fn(
                    seq_len,
                    seq_len,
                    offset=-1 if self.remove_self_pair else 0,
                    device=(x if x is not None else uu).device,
                )
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
                    x = x.reshape(-1, self.pairwise_lv_dim, seq_len * seq_len)
                if uu is not None:
                    uu = uu.reshape(-1, self.pairwise_input_dim, seq_len * seq_len)

        # with grad
        elements = 0
        if x is not None:
            elements = elements + self.embed(x)
        if uu is not None:
            elements = elements + self.fts_embed(uu)

        if self.is_symmetric:
            y = torch.zeros(
                batch_size,
                self.out_dim,
                seq_len,
                seq_len,
                dtype=elements.dtype,
                device=elements.device,
            )
            y[:, :, i, j] = elements
            y[:, :, j, i] = elements
        else:
            y = elements.reshape(-1, self.out_dim, seq_len, seq_len)
        return y

    def _forward_sparse(self, x, uu=None, mask=None):
        # x: (batch, v_dim, seq_len)
        # uu: (batch, v_dim, seq_len, seq_len)
        assert x is not None or uu is not None
        with torch.no_grad():
            if x is not None:
                batch_size, _, seq_len = x.size()
            else:
                batch_size, _, seq_len, _ = uu.size()

            i0, i1, i2, i3 = (Ellipsis,) * 4
            if mask is not None:
                mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)  # (batch_size, 1, seq_len, seq_len)
                if self.is_symmetric:
                    offset = -1 if self.remove_self_pair else 0
                    i0, _, i2, i3 = mask.float().tril(offset).nonzero(as_tuple=True)
                else:
                    i0, _, i2, i3 = mask.nonzero(as_tuple=True)

            if x is not None:
                x = self.pairwise_lv_fts(x.unsqueeze(-1), x.unsqueeze(-2))
                x = x.permute(0, 2, 3, 1)[i0, i2, i3, :]  # (num_elements, pairwise_lv_dim)
                x = x.T.unsqueeze(0).contiguous()  # (1, pairwise_lv_dim, num_elements)
            if uu is not None:
                uu = uu.permute(0, 2, 3, 1)[i0, i2, i3, :]  # (num_elements, pairwise_input_dim)
                uu = uu.T.unsqueeze(0).contiguous()  # (1, pairwise_input_dim, num_elements)

        # with grad
        elements = 0
        if x is not None:
            elements = elements + self.embed(x)
        if uu is not None:
            elements = elements + self.fts_embed(uu)
        elements = elements.squeeze(0).T  # (num_elements, out_dim)

        y = torch.zeros(
            batch_size,
            seq_len,
            seq_len,
            self.out_dim,
            dtype=elements.dtype,
            device=elements.device,
        )
        y[i0, i2, i3, :] = elements
        if self.is_symmetric:
            y[i0, i3, i2, :] = elements
        y = y.permute(0, 3, 1, 2).contiguous()

        return y

    def forward(self, x, uu=None, mask=None):
        if self.sparse_eval:
            return self._forward_sparse(x, uu=uu, mask=mask)
        else:
            return self._forward_dense(x, uu=uu, mask=mask)


def _canonical_mask(
    mask: torch.Tensor | None,
    mask_name: str,
    other_type: Any | None,
    other_name: str,
    target_type: Any,
    check_other: bool = True,
) -> torch.Tensor | None:
    if mask is not None:
        _mask_dtype = mask.dtype
        _mask_is_float = torch.is_floating_point(mask)
        if _mask_dtype != torch.bool and not _mask_is_float:
            raise AssertionError(f"only bool and floating types of {mask_name} are supported")
        if not _mask_is_float:
            mask = torch.zeros_like(mask, dtype=target_type).masked_fill_(mask, float("-inf"))
    return mask


def _none_or_dtype(input: torch.Tensor | None):
    if input is None:
        return None
    elif isinstance(input, torch.Tensor):
        return input.dtype
    raise RuntimeError("input to _none_or_dtype() must be None or torch.Tensor")


class Attention(torch.nn.Module):
    def __init__(
        self,
        attention,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )

        self.in_proj = torch.nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.attention = attention

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # snapshot: the loop pops/inserts while renaming, and iterating the live
        # dict either raises or silently skips keys (adversarial-review finding;
        # upstream lloca 1.3.6 has the same bug)
        for k in list(state_dict.keys()):
            if k.endswith("in_proj_weight"):
                state_dict[k.replace("_weight", ".weight")] = state_dict.pop(k)
            elif k.endswith("in_proj_bias"):
                state_dict[k.replace("_bias", ".bias")] = state_dict.pop(k)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, tgt_len, _ = query.shape
        _, src_len, _ = key.shape

        # (bsz, src_len)
        key_padding_mask = _canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=_none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )

        # (bsz, num_heads, tgt_len, src_len)
        attn_mask = _canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        # merge key padding and attention masks
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (
                bsz,
                src_len,
            ), (
                f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
            )
            key_padding_mask = (
                key_padding_mask.reshape(bsz, src_len)
                .unsqueeze(1)
                .unsqueeze(2)
                .expand(-1, self.num_heads, -1, -1)
            )
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                assert attn_mask.shape == (
                    bsz,
                    self.num_heads,
                    tgt_len,
                    src_len,
                ), (
                    f"expecting attn_mask shape of {(bsz, self.num_heads, tgt_len, src_len)}, but got {attn_mask.shape}"
                )
                attn_mask = attn_mask + key_padding_mask

        # (bsz, seq_len, num_heads*head_dim)
        q, k, v = F._in_projection_packed(query, key, value, self.in_proj.weight, self.in_proj.bias)

        # -> (bsz, num_heads, src/tgt_len, head_dim)
        q = q.reshape(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = k.reshape(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.reshape(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        dropout_p = self.dropout if self.training else 0.0

        if self.attention is not None:
            # particle attention
            attn_output = self.attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
            )
        else:
            # class token attention
            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)

        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, None


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"


class Block(nn.Module):
    def __init__(
        self,
        attention,
        embed_dim=128,
        num_heads=8,
        ffn_ratio=4,
        dropout=0.1,
        attn_dropout=0.1,
        activation_dropout=0.1,
        activation="gelu",
        layer_scale_init_values=None,
        drop_path_rate=0.0,
        scale_attn_mask=False,
        scale_attn=True,
        scale_fc=True,
        scale_heads=True,
        scale_resids=True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ffn_dim = embed_dim * ffn_ratio

        self.pre_attn_norm = nn.LayerNorm(embed_dim)
        self.attn = Attention(attention, embed_dim, num_heads, dropout=attn_dropout)
        self.post_attn_norm = nn.LayerNorm(embed_dim) if scale_attn else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.ls1 = (
            LayerScale(embed_dim, init_values=layer_scale_init_values)
            if layer_scale_init_values
            else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.pre_fc_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, self.ffn_dim)
        if activation == "swiglu":
            self.fc1_g = nn.Linear(embed_dim, self.ffn_dim)
            self.act = nn.SiLU()
        else:
            self.fc1_g = None
            self.act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.act_dropout = nn.Dropout(activation_dropout)
        self.post_fc_norm = nn.LayerNorm(self.ffn_dim) if scale_fc else nn.Identity()
        self.fc2 = nn.Linear(self.ffn_dim, embed_dim)
        self.ls2 = (
            LayerScale(embed_dim, init_values=layer_scale_init_values)
            if layer_scale_init_values
            else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.c_mask = nn.Parameter(torch.ones(1), requires_grad=True) if scale_attn_mask else None
        self.c_attn = (
            nn.Parameter(torch.ones(num_heads), requires_grad=True) if scale_heads else None
        )
        self.w_resid = (
            nn.Parameter(torch.ones(embed_dim), requires_grad=True) if scale_resids else None
        )

    def forward(self, x, x_cls=None, padding_mask=None, attn_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(batch, seq_len, embed_dim)`
            x_cls (Tensor, optional): class token input to the layer of shape `(batch, 1, embed_dim)`
            padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, seq_len)` where padding
                elements are indicated by ``True``.

        Returns:
            encoded output of shape `(batch, seq_len, embed_dim)`
        """

        if x_cls is not None:
            with torch.no_grad():
                # prepend one element for x_cls: -> (batch, 1+seq_len)
                padding_mask = torch.cat(
                    (torch.zeros_like(padding_mask[:, :1]), padding_mask), dim=1
                )
            # class attention: https://arxiv.org/pdf/2103.17239.pdf
            residual = x_cls
            u = torch.cat((x_cls, x), dim=1)  # (batch, 1+seq_len, embed_dim)
            u = self.pre_attn_norm(u)

            # default attention for convenience (could be more fancy here)
            x = self.attn(
                x_cls,
                u,
                u,
                key_padding_mask=padding_mask,
            )[0]  # (1, batch, embed_dim)
        else:
            if self.c_mask is not None and attn_mask is not None:
                attn_mask = torch.mul(self.c_mask, attn_mask)
            residual = x
            x = self.pre_attn_norm(x)
            x = self.attn(x, x, x, key_padding_mask=padding_mask, attn_mask=attn_mask)[
                0
            ]  # (batch, seq_len, embed_dim)

        if self.c_attn is not None:
            bsz, tgt_len, _ = x.size()
            x = x.reshape(bsz, tgt_len, self.num_heads, self.head_dim)
            x = torch.einsum("bthd,h->btdh", x, self.c_attn)
            x = x.reshape(bsz, tgt_len, self.embed_dim)
        x = self.post_attn_norm(x)
        x = self.dropout(x)
        x = self.drop_path1(self.ls1(x))
        x += residual

        residual = x
        x = self.pre_fc_norm(x)
        if self.fc1_g is None:
            x = self.act(self.fc1(x))
        else:
            x_gate = self.fc1_g(x)
            x = self.fc1(x)
            x = self.act(x_gate) * x
        x = self.act_dropout(x)
        x = self.post_fc_norm(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.drop_path2(self.ls2(x))
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x += residual

        return x


class ParticleTransformer(nn.Module):
    """Particle Transformer (ParT) with local frame transformations."""

    def __init__(
        self,
        input_dim,
        attn_reps,
        num_classes=None,
        # network configurations
        pair_input_type="pp",
        pair_input_dim=None,
        pair_extra_dim=0,
        remove_self_pair=False,
        use_pre_activation_pair=True,
        embed_dims=(128, 512, 128),
        ffn_ratio=4,
        pair_embed_dims=(64, 64, 64),
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params=None,
        fc_params=(),
        activation="gelu",
        # misc
        version=1,
        weight_init="moco",
        fix_init=True,
        trim=True,
        for_inference=False,
        for_segmentation=False,
        use_amp=False,
        checkpoint_blocks=False,
        compile=False,
        compile_mode="default",
        compile_dynamic=False,  # ParT does not rely on dynamic shapes that much
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.trimmer = SequenceTrimmer(enabled=trim and not for_inference)
        self.for_inference = for_inference
        self.for_segmentation = for_segmentation
        self.use_amp = use_amp
        self.checkpoint_blocks = checkpoint_blocks

        attn_reps = TensorReps(attn_reps)
        self.embed_dim = attn_reps.dim * num_heads
        self.attention = LLoCaAttention(attn_reps, num_heads)
        default_cfg = dict(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio,
            dropout=0.1,
            attn_dropout=0.1,
            activation_dropout=0.1,
            activation=activation,
            layer_scale_init_values=None,
            drop_path_rate=0.0,
            scale_attn_mask=False,
            scale_fc=True,
            scale_attn=True,
            scale_heads=True,
            scale_resids=True,
        )
        if version > 1:
            default_cfg.update(
                activation="swiglu",
                scale_fc=False,
                scale_attn=False,
                scale_heads=False,
                scale_resids=False,
            )

        cfg_block = copy.deepcopy(default_cfg)
        if block_params is not None:
            cfg_block.update(block_params)

        cfg_cls_block = copy.deepcopy(default_cfg)
        cfg_cls_block.update({"dropout": 0, "attn_dropout": 0, "activation_dropout": 0})
        if cls_block_params is not None:
            cfg_cls_block.update(cls_block_params)

        self.embed = Embed(
            input_dim,
            embed_dims if len(embed_dims) > 0 else [self.embed_dim],
            activation=activation,
        )

        if pair_input_dim is None:
            pair_input_dim = 4 if pair_input_type == "pp" else 6
        self.pair_extra_dim = pair_extra_dim
        self.pair_embed = (
            PairEmbed(
                pair_input_dim,
                pair_extra_dim,
                (*pair_embed_dims, cfg_block["num_heads"]),
                pairwise_lv_type=pair_input_type,
                remove_self_pair=remove_self_pair,
                use_pre_activation_pair=use_pre_activation_pair,
                for_onnx=for_inference,
            )
            if pair_embed_dims is not None and pair_input_dim + pair_extra_dim > 0
            else None
        )
        self.blocks = nn.ModuleList(
            [Block(attention=self.attention, **cfg_block) for _ in range(num_layers)]
        )
        self.cls_blocks = (
            nn.ModuleList([Block(attention=None, **cfg_cls_block) for _ in range(num_cls_layers)])
            if num_cls_layers > 0
            else None
        )
        self.norm = nn.LayerNorm(self.embed_dim)

        if fc_params is not None:
            fcs = []
            in_dim = self.embed_dim
            for param in fc_params:
                try:
                    out_dim, drop_rate, act = param
                except ValueError:
                    (out_dim, drop_rate), act = param, "relu"
                if act == "swiglu":
                    layer = nn.Sequential(
                        SwiGLUFFN(in_dim, out_dim * 4, out_dim, drop=drop_rate),
                        nn.LayerNorm(out_dim),
                    )
                else:
                    layer = nn.Sequential(
                        nn.Linear(in_dim, out_dim),
                        nn.GELU() if act == "gelu" else nn.ReLU(),
                        nn.Dropout(drop_rate),
                    )
                fcs.append(layer)
                in_dim = out_dim
            fcs.append(nn.Linear(in_dim, num_classes))
            self.fc = nn.Sequential(*fcs)
        else:
            self.fc = None

        # cls tokens
        if not self.for_segmentation and num_cls_layers > 0:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim), requires_grad=True)
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        # weight initialization
        if weight_init is not None:
            self.init_weights(weight_init)
        if fix_init:
            self.fix_init_weight()

        if compile:
            self.__class__ = torch.compile(
                self.__class__, dynamic=compile_dynamic, mode=compile_mode
            )

    def fix_init_weight(self):
        def rescale(param, _layer_id):
            param.div_(math.sqrt(2.0 * _layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.out_proj.weight.data, layer_id + 1)
            rescale(layer.fc2.weight.data, layer_id + 1)

    def init_weights(self, mode: str = "") -> None:
        assert mode in ("timm", "moco")
        if mode == "timm":
            named_apply(init_weights_vit_timm, self)
        elif mode == "moco":
            named_apply(init_weights_vit_moco, self)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "cls_token",
        }

    def _forward_encoder(self, x, v=None, mask=None, uu=None, uu_idx=None, frames=None):
        with torch.no_grad():
            if not self.for_inference:
                if uu_idx is not None:
                    uu = build_sparse_tensor(uu, uu_idx, x.size(-1))
            if frames is None or frames.is_identity or frames.is_global:
                # identity/global frames are token-independent: no need to ride the trimmer
                x, v, mask, uu, _ = self.trimmer(x, v, mask, uu)
            else:
                # per-particle frames MUST ride the trimmer with x/v/mask: the trimmer
                # permutes and truncates the sequence, and preparing frames on the
                # untrimmed order transports each token with another token's frame
                # (adversarial-review finding; upstream lloca 1.3.6 has the same bug --
                # crash on trimmed batches, silent frame misassignment on permuted ones).
                # All frame tensor ops run under enable_grad: the surrounding no_grad is
                # for data prep, but learned framesnets train THROUGH these matrices.
                B, _, P = x.shape
                with torch.enable_grad():
                    fm = frames.matrices.reshape(B, P, 16).transpose(1, 2)
                    # the WHOLE trimmer call, not just the frame rider: under a learned
                    # framesnet x and v are functions of the framesnet's parameters too
                    # (TaggerWrapper transports them), and the trimmer's gathers/slices
                    # under the outer no_grad would DETACH them -- silently, and only
                    # from warm-up step warmup_steps+1 onward, cutting the feature path's
                    # gradient mid-run while the forward stays bit-identical (final-audit
                    # finding: framesnet grads ~100% relative wrong from step 6)
                    x, v, mask, uu, fm = self.trimmer(x, v, mask, uu, extra=fm)
                # BATCH-shaped (B, P', 4, 4): prepare_frames inserts the head dim as
                # (*batch, H, N), so flat (B*P, 4, 4) frames flatten in (H, B, P) order
                # while q/k/v flatten in (B, H, P) -- a systematic token/frame
                # misalignment for B>1 (third inherited library bug; the wrapper's flat
                # framesnet output hits it too, which is why frames are re-shaped here)
                with torch.enable_grad():
                    frames = Frames(
                        matrices=fm.transpose(1, 2).reshape(x.shape[0], -1, 4, 4),
                        is_global=frames.is_global,
                        is_identity=frames.is_identity,
                    )
            padding_mask = ~mask.squeeze(1)  # (batch_size, seq_len)
        if frames is not None:
            self.attention.prepare_frames(frames)

        with torch.autocast(x.device.type, enabled=self.use_amp):
            # input embedding
            x = self.embed(x).masked_fill(
                ~mask.transpose(1, 2), 0
            )  # (batch_size, seq_len, num_fts)
            attn_mask = None
            if (v is not None or uu is not None) and self.pair_embed is not None:
                attn_mask = self.pair_embed(
                    v, uu=uu, mask=mask
                )  # (batch_size, num_heads, seq_len, seq_len)

            # transform
            for block in self.blocks:
                if self.checkpoint_blocks:
                    x = checkpoint(
                        block,
                        x,
                        x_cls=None,
                        padding_mask=padding_mask,
                        attn_mask=attn_mask,
                        use_reentrant=False,
                    )
                else:
                    x = block(x, x_cls=None, padding_mask=padding_mask, attn_mask=attn_mask)

        # x: (batch, seq_len, embed_dim)
        # padding_mask: (batch, seq_len)
        return x, padding_mask

    def _forward_aggregator(self, x, padding_mask):
        with torch.autocast(x.device.type, enabled=self.use_amp):
            if self.cls_blocks is not None:
                # for classification: extract using class token
                cls_tokens = self.cls_token.expand(x.size(0), 1, -1)  # (batch, 1, embed_dim)
                for block in self.cls_blocks:
                    if self.checkpoint_blocks:
                        cls_tokens = checkpoint(
                            block,
                            x,
                            x_cls=cls_tokens,
                            padding_mask=padding_mask,
                            use_reentrant=False,
                        )  # (batch, 1, embed_dim)
                    else:
                        cls_tokens = block(
                            x, x_cls=cls_tokens, padding_mask=padding_mask
                        )  # (batch, 1, embed_dim)
                cls_tokens = cls_tokens.squeeze(1)  # (batch, embed_dim)
            else:
                # for classification: simple average pooling
                mask = ~padding_mask.unsqueeze(1)  # (batch, 1, seq_len)
                x = x.transpose(1, 2).contiguous()  # (batch, embed_dim, seq_len)
                counts = mask.float().sum(-1)  # (batch, 1)
                counts = torch.max(counts, torch.ones_like(counts))  # >=1
                cls_tokens = (x * mask).sum(-1) / counts  # (batch, embed_dim)

            x_cls = self.norm(cls_tokens)  # (batch, embed_dim)
        return x_cls

    def forward(self, x, frames, v=None, mask=None, uu=None, uu_idx=None):
        # x: (batch_size, num_fts, seq_len)
        # v: (batch_size, 4, seq_len) [px,py,pz,energy]
        # mask: (batch_size, 1, seq_len) -- real particle = 1, padded = 0
        # for pytorch: uu (batch_size, C', num_pairs), uu_idx (batch_size, 2, num_pairs)
        # for onnx: uu (batch_size, C', seq_len, seq_len), uu_idx=None
        x, padding_mask = self._forward_encoder(
            x, v=v, mask=mask, uu=uu, uu_idx=uu_idx, frames=frames
        )

        if self.cls_blocks is None and self.fc is None:
            # x: (batch, seq_len, embed_dim)
            # padding_mask: (batch, seq_len)
            return x, padding_mask

        with torch.autocast(x.device.type, enabled=self.use_amp):
            # === for segmentation ===
            if self.for_segmentation:
                x = self.norm(x)
                if self.fc is not None:
                    x = self.fc(x)
                # x: (P, N, C) -> output: (N, C, P)
                output = x.transpose(1, 2).contiguous()
                if self.for_inference:
                    # single-logit heads (top-tagging: out=1, BCE) must use sigmoid --
                    # softmax over a 1-wide dim is identically 1 (see module docstring)
                    output = (torch.sigmoid(output) if output.shape[1] == 1
                              else torch.softmax(output, dim=1))
                # print('output:\n', output)
                return output

            x_cls = self._forward_aggregator(x, padding_mask)
            if self.fc is None:
                return x_cls

            # fc
            output = self.fc(x_cls)
            if self.for_inference:
                # single-logit heads use sigmoid (see module docstring)
                output = (torch.sigmoid(output) if output.shape[1] == 1
                          else torch.softmax(output, dim=1))
            # print('output:\n', output)
            return output


### weight initialization methods ###
def init_weights_vit_timm(module: nn.Module, name: str = "") -> None:
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()


def init_weights_vit_moco(module: nn.Module, name: str = "") -> None:
    """ViT weight initialization, matching moco-v3 impl minus fixed PatchEmbed"""
    if isinstance(module, nn.Linear):
        if "in_proj" in name:
            # treat the weights of Q, K, V separately
            val = math.sqrt(6.0 / float(module.weight.shape[0] // 3 + module.weight.shape[1]))
            nn.init.uniform_(module.weight, -val, val)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()


def named_apply(
    fn: Callable,
    module: nn.Module,
    name="",
    depth_first: bool = True,
    include_root: bool = False,
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module
