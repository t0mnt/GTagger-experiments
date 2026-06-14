"""ParticleNet-ParT GraphGPS hybrid: EdgeConv local + ParT-biased attention global.

The rich-edge member of the GraphGPS family for this repo. Each GPS layer fuses a
local and a global branch following the GraphGPS recipe (paper App. D, Eq. 9-11:
per-branch dropout -> residual -> norm, fuse by SUM, then an FFN with its own
residual + norm; FFN inner width = 2*dim):

  * local branch  -- a tensorial ParticleNet EdgeConv on a DYNAMIC kNN graph, rebuilt
    from the current hidden features each layer (DGCNN-style). The first layer is seeded
    from the input geometry: (eta, phi) for 'deltaR' or the four-momenta for
    'minkowski'. Neighbours are transported into the centre's local frame (LLoCa
    change_local_frame, typed by ``edge_reps``). EdgeConv owns an internal shortcut
    residual, so -- exactly as GraphGPS does for its CustomGatedGCN local model -- the
    GPS layer adds only the external normalization on this branch, not a second residual.

  * global branch -- ParT attention made tensorial with LLoCa: the q/k/v are transported
    between local frames (``LLoCaAttention``, typed by ``attn_reps``) and the ParT pairwise
    interaction bias U_ij (a ``PairEmbed`` of the four-momenta) is added to the attention
    logits. This is GraphGPS's 'BiasedTransformer' slot (``attn_bias``) filled with ParT's
    physics bias, computed once from the (canonicalized) four-momenta and shared across
    layers. The LLoCaAttention transport (no params) is likewise shared across layers.

Compared with ParticleNetParTGraphTrans (a sequential EdgeConv stage -> ParT stage),
this interleaves the two per layer. Like the rest of the GraphGPS family it uses
BatchNorm and a mean-pool readout (no class token) -- so, unlike the GraphTrans variant,
it needs no jet frame: mean-pooling the invariant local features is already invariant.

Made Lorentz-equivariant by LLoCa tensorial message-passing (matching the library's
ParticleNet and ParT), wired up in
``experiments.tagging.wrappers.ParticleNetParTGraphGPSWrapper`` (inherits TaggerWrapper),
which canonicalizes the inputs and passes the per-particle frames into the backbone.

Input convention (channels-first, matching ParT/ParticleNet/PlainGraphGPS):
    points:   (N, 2, P)   kNN coordinates (eta, phi), seeds layer 0 for 'deltaR'
    features: (N, C, P)   scalar per-particle features
    v:        (N, 4, P)   four-momenta as (px, py, pz, energy); pairwise bias + 'minkowski'
    mask:     (N, 1, P)   1 for real particles, 0 for padding
"""

import torch
import torch.nn as nn
from lloca.backbone.attention import LLoCaAttention
from lloca.backbone.particletransformer import Attention as LLoCaAttentionBlock
from lloca.reps.tensorreps import TensorReps

from experiments.baselines.particlenettransformer import EdgeConvBlock, PairEmbed
from experiments.baselines.plaingraphgps import MaskedNorm

_ACT = {"relu": nn.ReLU, "gelu": nn.GELU}


class ParTGPSLayer(nn.Module):
    """One GraphGPS layer with an EdgeConv local branch and ParT-biased attention.

    The EdgeConv (like GraphGPS's GatedGCN) carries its own internal residual, so
    only the external norm is applied to the local branch; the attention branch
    gets the standard external dropout -> residual -> norm (Eq. 10).
    """

    def __init__(self, dim, num_heads, edge_k, attention, edge_reps, edge_mlp_layers=2,
                 ffn_ratio=2, dropout=0.0, attn_dropout=0.0, act="relu", norm="batch",
                 for_inference=False):
        super().__init__()
        Act = _ACT[act]
        self.edge_k = edge_k
        # local: tensorial ParticleNet EdgeConv (in==out -> identity shortcut, i.e. internal
        # residual). edge_reps types the hidden features for the LLoCa neighbour transport.
        self.local = EdgeConvBlock(
            k=edge_k, in_reps=edge_reps, out_feats=[dim] * edge_mlp_layers, cpu_mode=for_inference
        )
        # global: LLoCa tensorial attention (transports q/k/v between frames). `attention`
        # is a shared LLoCaAttention (no params); ParT's pairwise bias is the additive mask.
        self.attn = LLoCaAttentionBlock(attention, dim, num_heads, dropout=attn_dropout)
        self.num_heads = num_heads

        self.norm_local = MaskedNorm(dim, norm)
        self.norm_attn = MaskedNorm(dim, norm)
        self.norm_ffn = MaskedNorm(dim, norm)
        self.drop_attn = nn.Dropout(dropout)
        self.drop_ffn = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_ratio * dim), Act(),
            nn.Dropout(dropout), nn.Linear(ffn_ratio * dim, dim),
        )

    def forward(self, h, coord, knn_metric, key_padding_mask, node_mask, mask_knn,
                frames_flat=None, attn_bias=None):
        # h: (B, P, C); coord: (B, Cc, P) or None (None -> dynamic graph on h);
        # node_mask: (B, P, 1); mask_knn: (B, P); frames_flat: (B*P, 4, 4) for the EdgeConv
        mask_bool = ~key_padding_mask
        h_cf = (h * node_mask).transpose(1, 2)                       # (B, C, P)

        # ---- local branch (Eq. 9): EdgeConv owns its residual -> external norm only ----
        kcoord = coord if coord is not None else h_cf
        kmetric = knn_metric if coord is not None else "deltaR"      # deeper layers: feature-space L2
        h_local = self.local(kcoord, h_cf, frames=frames_flat, knn_metric=kmetric, mask=mask_knn)
        h_local = self.norm_local(h_local.transpose(1, 2), mask_bool)          # (B, P, C)

        # ---- global branch (Eq. 10): LLoCa tensorial ParT-biased attention ----
        # attn_bias is (B, num_heads, P, P); the shared LLoCaAttention was prepared with
        # the per-token frames once per forward, so it transports q/k/v here.
        a = self.attn(h, h, h, key_padding_mask=key_padding_mask, attn_mask=attn_bias)[0]
        h_attn = self.norm_attn(self.drop_attn(a) + h, mask_bool)

        # ---- fuse by SUM, then FFN (Eq. 11) ----
        h = h_local + h_attn
        h = self.norm_ffn(self.drop_ffn(self.ffn(h)) + h, mask_bool)
        return h * node_mask


class ParticleNetParTGraphGPS(nn.Module):

    def __init__(self,
                 input_dim,
                 num_classes=None,
                 # dynamic kNN graph for the EdgeConv local branch
                 knn_k=16,
                 knn_metric="deltaR",       # how layer-0 graph is seeded ('deltaR'/'minkowski')
                 edge_mlp_layers=2,
                 use_fts_bn=True,
                 # ParT pairwise attention bias (global branch)
                 bias=True,                 # toggle the ParT pairwise additive attention bias
                 pair_input_dim=4,
                 pair_extra_dim=0,
                 pair_embed_dims=[64, 64, 64],
                 remove_self_pair=False,
                 use_pre_activation_pair=True,
                 # GPS layers
                 dim=128,
                 num_layers=10,
                 num_heads=8,
                 attn_reps="8x0n+2x1n",     # per-head transformer rep; dim = attn_reps.dim * num_heads
                 edge_reps="64x0n+16x1n",   # EdgeConv hidden rep for the neighbour transport (dim = dim)
                 ffn_ratio=2,
                 dropout=0.0,
                 attn_dropout=0.0,
                 act="relu",
                 norm="batch",
                 # readout
                 head_layers=2,
                 for_inference=False,
                 use_amp=False,
                 **kwargs):
        super().__init__(**kwargs)
        if knn_metric not in ("deltaR", "minkowski"):
            raise ValueError(f"knn_metric must be 'deltaR' or 'minkowski', got '{knn_metric}'")
        self.knn_k = knn_k
        self.knn_metric = knn_metric
        self.num_heads = num_heads
        self.for_inference = for_inference
        self.use_amp = use_amp
        Act = _ACT[act]

        # LLoCa tensorial message-passing reps (shared by every GPS layer, since h keeps
        # the same width). attn_reps types the per-head attention q/k/v; edge_reps types
        # the EdgeConv hidden features. Both must carry >=1 vector to communicate geometry.
        attn_reps = TensorReps(attn_reps)
        edge_reps_t = TensorReps(edge_reps)
        assert attn_reps.dim * num_heads == dim, f"{attn_reps.dim}*{num_heads} != dim {dim}"
        assert edge_reps_t.dim == dim, f"edge_reps.dim {edge_reps_t.dim} != dim {dim}"
        self.attention = LLoCaAttention(attn_reps, num_heads)  # shared transport, no params

        self.bn_fts = nn.BatchNorm1d(input_dim) if use_fts_bn else None
        self.node_encoder = nn.Linear(input_dim, dim)

        # ParT pairwise bias: one shared PairEmbed, last dim = num_heads (one bias per head)
        self.pair_embed = None
        if bias and pair_embed_dims is not None and pair_input_dim + pair_extra_dim > 0:
            self.pair_embed = PairEmbed(
                pair_input_dim, pair_extra_dim, pair_embed_dims + [num_heads],
                remove_self_pair=remove_self_pair,
                use_pre_activation_pair=use_pre_activation_pair, for_onnx=for_inference,
            )

        self.layers = nn.ModuleList([
            ParTGPSLayer(dim, num_heads, knn_k, self.attention, edge_reps, edge_mlp_layers,
                         ffn_ratio, dropout, attn_dropout, act, norm, for_inference)
            for _ in range(num_layers)
        ])

        # SAN-style readout: mean pool -> dim-halving MLP -> logits
        head, d = [], dim
        for _ in range(head_layers):
            head += [nn.Linear(d, d // 2), Act()]
            d //= 2
        head += [nn.Linear(d, num_classes)]
        self.head = nn.Sequential(*head)

    def forward(self, points, features, v=None, frames=None, mask=None):
        if mask is None:
            mask = (features.abs().sum(dim=1, keepdim=True) != 0)
        else:
            mask = mask.bool()
        points = points * mask
        features = features * mask
        mask_p = mask.squeeze(1)                     # (B, P)
        P = features.size(-1)

        # per-particle frames flattened (B*P, 4, 4) for the EdgeConv change_local_frame;
        # the shared LLoCaAttention is prepared once for the (tensorial) attention branch.
        frames_flat = frames.reshape(-1, 4, 4) if frames is not None else None
        if frames is not None:
            self.attention.prepare_frames(frames)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # ParT pairwise interaction bias U_ij, computed once and shared across layers.
            # PairEmbed pads a CLS slot at index 0; we mean-pool (no CLS) so we drop it.
            attn_bias = None
            if self.pair_embed is not None and v is not None:
                attn_bias = self.pair_embed(v)[:, :, 1:, 1:]        # (B, num_heads, P, P)

            fts = self.bn_fts(features) * mask if self.bn_fts is not None else features
            h = self.node_encoder(fts.transpose(1, 2))              # (B, P, dim)

            node_mask = mask_p.unsqueeze(-1).to(h.dtype)            # (B, P, 1)
            key_padding_mask = ~mask_p                              # (B, P), True = ignore
            for i, layer in enumerate(self.layers):
                if i == 0:
                    coord = v if self.knn_metric == "minkowski" and v is not None else points
                    metric = self.knn_metric
                else:
                    coord, metric = None, "deltaR"                  # dynamic feature-space graph
                h = layer(h, coord, metric, key_padding_mask, node_mask, mask_p,
                          frames_flat, attn_bias)

            # masked mean pooling over real particles
            pooled = (h * node_mask).sum(dim=1) / node_mask.sum(dim=1).clamp(min=1.0)
            output = self.head(pooled)
            if self.for_inference:
                output = torch.softmax(output, dim=1)
            return output
