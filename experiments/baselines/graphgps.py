"""Plain GraphGPS hybrid: interleaved static-MPNN + torch-MHA (GraphGPS recipe).

GraphGPS (Rampasek et al., 2022, arXiv:2205.12454) differs from this repo's other
graph-transformer hybrids in *how* the local and global stages are combined: each
GPS layer runs a local message-passing block and a global attention block **in
parallel on the same input** and sums them, rather than stacking a GNN stage and
then a transformer stage (cf. PlainGraphTrans, which is sequential). The precise
per-layer recipe (paper App. D, Eq. 9-11) is

    X_M = Norm( Dropout( MPNN(X) ) + X )        # local branch, own residual+norm
    X_T = Norm( Dropout( Attn(X) ) + X )        # global branch, own residual+norm
    X'  = Norm( Dropout( FFN(X_M + X_T) ) + (X_M + X_T) )   # fuse by SUM, then FFN

with the two branch residuals being exactly what makes summing two sub-networks
and stacking many layers trainable. The FFN inner width is 2*dim (paper figure).

This is the "plain", non-equivariant variant:
  * a STATIC kNN graph (built once per forward, reused by every layer), with a
    selectable metric -- 'deltaR' (L2 on (eta, phi)) or 'minkowski' (|interval|
    on the four-momenta) -- and the usual robustness tweaks (k capped at P-1,
    padded nodes excluded), shared with PlainGraphTrans via ``knn``;
  * a plain static message-passing local branch with the default update function
    (message MLP on [h_i, h_j, e_ij] -> masked mean -> update MLP), NOT GatedGCN;
  * a plain global branch of torch ``nn.MultiheadAttention`` (NOT L-GATr);
  * NO positional/structural encodings (LapPE/RWSE): jet constituents are never
    anonymous nodes, so PE/SE is unmotivated here. The only relative encoding is
    the physically-meaningful, canonicalization-invariant Minkowski edge feature
    log|(p_i + p_j)^2|, fed into the local MPNN messages (toggle ``use_edge_attr``).

Being non-equivariant, it is made Lorentz-equivariant through LLoCa input
canonicalization in PlainGraphGPSWrapper (which inherits TaggerWrapper), exactly like
ParT / ParticleNet / PlainGraphTrans.

Input convention (channels-first, matching ParT/ParticleNet/PlainGraphTrans):
    points:   (N, 2, P)   kNN coordinates (eta, phi), used when knn_metric='deltaR'
    features: (N, C, P)   scalar per-particle features
    v:        (N, 4, P)   four-momenta as (px, py, pz, energy), used for 'minkowski'
    mask:     (N, 1, P)   1 for real particles, 0 for padding
"""

import torch
import torch.nn as nn

from experiments.baselines.plaingraphtrans import gather_neighbors, knn

_ACT = {"relu": nn.ReLU, "gelu": nn.GELU}


def minkowski_edge_attr(v, idx, eps=1e-8):
    """Per-edge Lorentz invariant log|(p_i + p_j)^2| on the static kNN graph.

    v: (B, 4, P) as (px, py, pz, E); idx: (B, P, K). Returns (B, 1, P, K). This is
    the physically-motivated, frame-invariant relative encoding GraphGPS's local
    MPNN consumes here in place of PE/SE.
    """
    nbr = gather_neighbors(v, idx)            # (B, 4, P, K)
    psum = v.unsqueeze(-1) + nbr              # (B, 4, P, K), center broadcast over K
    px, py, pz, E = psum[:, 0], psum[:, 1], psum[:, 2], psum[:, 3]
    m2 = E * E - px * px - py * py - pz * pz   # (B, P, K)
    return m2.abs().clamp(min=eps).log().unsqueeze(1)   # (B, 1, P, K)


class GPSLocalMPNN(nn.Module):
    """Local branch (Eq. 7): a plain static-graph message-passing block.

    message m_ij = MLP([h_i, h_j, e_ij]); aggregate = masked mean over the kNN
    neighbours; update h' = MLP([h_i, agg]). It carries NO internal residual or
    norm -- the GPS layer owns the external dropout -> residual -> norm (Eq. 9).
    """

    def __init__(self, dim, edge_dim=0, act="relu"):
        super().__init__()
        Act = _ACT[act]
        self.message = nn.Sequential(
            nn.Conv2d(2 * dim + edge_dim, dim, 1), Act(),
            nn.Conv2d(dim, dim, 1), Act(),
        )
        self.update = nn.Sequential(
            nn.Conv1d(2 * dim, dim, 1), Act(),
            nn.Conv1d(dim, dim, 1),
        )

    def forward(self, h, idx, nbr_mask, edge_attr=None):
        # h: (B, C, P); idx: (B, P, K); nbr_mask: (B, P, K) bool; edge_attr: (B, E, P, K)
        K = idx.shape[-1]
        nbr = gather_neighbors(h, idx)                       # (B, C, P, K)
        center = h.unsqueeze(-1).expand(-1, -1, -1, K)       # (B, C, P, K)
        msg_in = [center, nbr] + ([edge_attr] if edge_attr is not None else [])
        m = self.message(torch.cat(msg_in, dim=1))           # (B, C, P, K)

        nm = nbr_mask.unsqueeze(1).to(m.dtype)               # (B, 1, P, K)
        m = m * nm
        count = nm.sum(dim=-1).clamp(min=1.0)                # (B, 1, P)
        agg = m.sum(dim=-1) / count                          # (B, C, P): masked mean
        return self.update(torch.cat([h, agg], dim=1))       # (B, C, P)


class GPSLayer(nn.Module):
    """One GraphGPS layer: parallel local-MPNN + global-attention, fused by sum.

    Faithful to the paper's residual/norm placement (Eq. 9-11). LayerNorm is used
    (rather than GraphGPS's default BatchNorm) because it is per-token and so
    padding-safe for the variable-size jets here; it is a documented GPS option.
    """

    def __init__(self, dim, num_heads, edge_dim=0, ffn_ratio=2,
                 dropout=0.0, attn_dropout=0.0, act="relu"):
        super().__init__()
        Act = _ACT[act]
        self.local = GPSLocalMPNN(dim, edge_dim, act)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_dropout, batch_first=True)

        self.norm_local = nn.LayerNorm(dim)
        self.norm_attn = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)
        self.drop_local = nn.Dropout(dropout)
        self.drop_attn = nn.Dropout(dropout)
        self.drop_ffn = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_ratio * dim), Act(),
            nn.Dropout(dropout), nn.Linear(ffn_ratio * dim, dim),
        )

    def forward(self, h, idx, nbr_mask, key_padding_mask, node_mask, edge_attr=None):
        # h: (B, P, C); node_mask: (B, P, 1) float
        # ---- local branch (Eq. 9) ----
        h_cf = (h * node_mask).transpose(1, 2)                       # (B, C, P)
        m_local = self.local(h_cf, idx, nbr_mask, edge_attr).transpose(1, 2)  # (B, P, C)
        h_local = self.norm_local(self.drop_local(m_local) + h)
        # ---- global branch (Eq. 10) ----
        a = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)[0]
        h_attn = self.norm_attn(self.drop_attn(a) + h)
        # ---- fuse by SUM, then FFN (Eq. 11) ----
        h = h_local + h_attn
        h = self.norm_ffn(self.drop_ffn(self.ffn(h)) + h)
        return h * node_mask


class PlainGraphGPS(nn.Module):

    def __init__(self,
                 input_dim,
                 num_classes=None,
                 # static kNN graph
                 knn_k=16,
                 knn_metric="deltaR",
                 use_edge_attr=True,
                 use_fts_bn=True,
                 # GPS layers
                 dim=128,
                 num_layers=10,
                 num_heads=8,
                 ffn_ratio=2,
                 dropout=0.0,
                 attn_dropout=0.0,
                 act="relu",
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
        self.use_edge_attr = use_edge_attr
        self.for_inference = for_inference
        self.use_amp = use_amp
        Act = _ACT[act]

        self.bn_fts = nn.BatchNorm1d(input_dim) if use_fts_bn else None
        self.node_encoder = nn.Linear(input_dim, dim)

        edge_dim = 1 if use_edge_attr else 0
        self.layers = nn.ModuleList([
            GPSLayer(dim, num_heads, edge_dim, ffn_ratio, dropout, attn_dropout, act)
            for _ in range(num_layers)
        ])

        # SAN-style readout: mean pool -> dim-halving MLP -> logits
        head, d = [], dim
        for _ in range(head_layers):
            head += [nn.Linear(d, d // 2), Act()]
            d //= 2
        head += [nn.Linear(d, num_classes)]
        self.head = nn.Sequential(*head)

    def forward(self, points, features, v=None, mask=None):
        if mask is None:
            mask = (features.abs().sum(dim=1, keepdim=True) != 0)
        else:
            mask = mask.bool()
        features = features * mask
        mask_p = mask.squeeze(1)                     # (B, P)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # static kNN graph (built once, reused by every GPS layer)
            if self.knn_metric == "minkowski" and v is not None:
                idx = knn(v, self.knn_k, metric="minkowski", mask=mask_p)
            else:
                idx = knn(points, self.knn_k, metric="deltaR", mask=mask_p)
            nbr_mask = gather_neighbors(mask.float(), idx).squeeze(1) > 0.5   # (B, P, K)

            edge_attr = None
            if self.use_edge_attr and v is not None:
                edge_attr = minkowski_edge_attr(v, idx)                       # (B, 1, P, K)
                edge_attr = edge_attr * nbr_mask.unsqueeze(1).to(edge_attr.dtype)

            fts = self.bn_fts(features) * mask if self.bn_fts is not None else features
            h = self.node_encoder(fts.transpose(1, 2))                        # (B, P, dim)

            node_mask = mask_p.unsqueeze(-1).to(h.dtype)                      # (B, P, 1)
            key_padding_mask = ~mask_p                                        # (B, P), True = ignore
            for layer in self.layers:
                h = layer(h, idx, nbr_mask, key_padding_mask, node_mask, edge_attr)

            # masked mean pooling over real particles
            pooled = (h * node_mask).sum(dim=1) / node_mask.sum(dim=1).clamp(min=1.0)
            output = self.head(pooled)
            if self.for_inference:
                output = torch.softmax(output, dim=1)
            return output
