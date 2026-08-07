"""lloca's LLoCa-ParticleNet with an optional Minkowski layer-0 kNN.

`lloca.backbone.particlenet.ParticleNet` seeds its first EdgeConv graph with a plain squared-L2
`knn(points, k)` and never sees the four-momenta, so `knn_metric=minkowski` -- available on all
eight hybrids -- was simply not expressible for the baseline row. This adds it, by the same
construction the ParticleNet-ParT hybrid uses (`particlenettransformer.py`): rank layer-0
neighbours by the absolute Minkowski interval of the four-momenta, and leave every deeper layer
on the dynamic feature-space graph exactly as before.

**The deltaR path is untouched and bit-identical.** It still calls lloca's own `knn`, not the
hybrid's. That matters: the hybrid's deltaR branch wraps `dphi` into `[0, pi]` before the L2
(azimuth is periodic; a pair across the +/-pi seam is adjacent, not ~2pi apart) while lloca's
plain gram L2 does not. Routing the baseline through the hybrid's helper would therefore have
silently changed the published-reproduction row. That difference between baseline and hybrid is
pre-existing and is left as-is here -- this file only adds a metric that did not exist.
"""

import torch
from lloca.backbone.particlenet import EdgeConvBlock as LLoCaEdgeConvBlock
from lloca.backbone.particlenet import ParticleNet as LLoCaParticleNet
from lloca.backbone.particlenet import knn as lloca_knn

from experiments.baselines.particlenettransformer import knn as metric_knn


class MetricEdgeConvBlock(LLoCaEdgeConvBlock):
    """lloca's EdgeConvBlock with a selectable neighbour metric.

    Only the index computation differs; everything after it is lloca's body verbatim, so a
    `metric="deltaR"` call is the unmodified block. Instances are produced by re-typing the
    parent's blocks in place (see `ParticleNetLLoCa`), which leaves every parameter untouched.
    """

    def forward(self, points, features, frames, metric="deltaR", mask=None):
        if metric == "minkowski":
            topk_indices = metric_knn(points, self.k, metric="minkowski", mask=mask)
        else:
            topk_indices = lloca_knn(points, self.k)  # unchanged -> bit-identical baseline
        x = self.get_graph_feature(features, self.k, topk_indices, frames, self.trafo)

        for conv, bn, act in zip(self.convs, self.bns, self.acts, strict=False):
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


class ParticleNetLLoCa(LLoCaParticleNet):
    """lloca's ParticleNet plus `knn_metric`; `deltaR` reproduces it exactly.

    `v` (four-momenta, (N, 4, P), same (px, py, pz, E) convention the hybrids use) is accepted
    by `forward` and consumed only when `knn_metric="minkowski"`, and only by layer 0 -- deeper
    EdgeConvs rank on their own features, as in every ParticleNet.
    """

    def __init__(self, *args, knn_metric="deltaR", **kwargs):
        super().__init__(*args, **kwargs)
        if knn_metric not in ("deltaR", "minkowski"):
            raise ValueError(f"knn_metric must be 'deltaR' or 'minkowski', got '{knn_metric}'")
        self.knn_metric = knn_metric
        # re-type in place rather than rebuild: the parent already constructed and initialised
        # these blocks, and only the method resolution changes, so parameters and their init
        # draw are bit-identical to a plain lloca ParticleNet built with the same seed.
        for block in self.edge_convs:
            block.__class__ = MetricEdgeConvBlock

    def forward(self, points, features, frames, mask=None, v=None):
        if mask is None:
            mask = features.abs().sum(dim=1, keepdim=True) != 0  # (N, 1, P)
        points = points * mask
        features = features * mask
        coord_shift = (mask == 0) * 1e9
        if self.use_counts:
            counts = mask.float().sum(dim=-1)
            counts = torch.max(counts, torch.ones_like(counts))  # >=1

        fts = self.bn_fts(features) * mask if self.use_fts_bn else features
        outputs = []
        for idx, conv in enumerate(self.edge_convs):
            # layer 0 seeds the graph from the geometry; deeper layers from the features
            if idx == 0 and self.knn_metric == "minkowski" and v is not None:
                pts, metric = v + coord_shift, "minkowski"
            else:
                pts, metric = (points if idx == 0 else fts) + coord_shift, "deltaR"
            fts = conv(pts, fts, frames, metric=metric, mask=mask) * mask
            if self.use_fusion:
                outputs.append(fts)
        if self.use_fusion:
            fts = self.fusion_block(torch.cat(outputs, dim=1)) * mask

        if self.for_segmentation:
            x = fts
        else:
            x = fts.sum(dim=-1) / counts if self.use_counts else fts.mean(dim=-1)

        output = self.fc(x)
        if self.for_inference:
            # single-logit heads (top-tagging here: out_channels=1, BCE) must use sigmoid --
            # softmax over a 1-wide dim is identically 1.0. Multi-class is unchanged.
            output = torch.sigmoid(output) if output.shape[1] == 1 else torch.softmax(output, dim=1)
        return output
