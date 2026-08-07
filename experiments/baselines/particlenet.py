"""LLoCa-ParticleNet — the model the `tag_particlenet` table row runs.

Ported from `lloca.backbone.particlenet` (v1.3.6) so this repo owns the file the baseline
executes, rather than executing a library copy it cannot fix. The port is mechanical: the
tensorial message passing is byte-faithful to lloca's, and
`tests/internal/test_duplicated_component_parity.py` pins it against the installed library so
upstream drift is a test failure rather than a surprise.

Two deliberate additions on top of lloca's file, both off by default:

1. `for_inference` single-logit heads use sigmoid. Softmax over a 1-wide dim is identically
   1.0, so a binary head would return a constant score (AUC 0.5). Multi-class is unchanged.
2. `knn_metric="minkowski"` seeds the layer-0 graph by the absolute Minkowski interval
   |(p_i - p_j)^2| instead of a squared L2 on (phi, eta), which is the Lorentz-invariant graph
   all eight hybrids already expose. lloca's backbone never receives the four-momenta, so this
   was not expressible before; `ParticleNetWrapper` now passes them as `v`.

`knn_metric="deltaR"` (the default) still calls this file's `knn`, which is lloca's verbatim.
It deliberately does NOT route through the hybrid's metric-aware helper even though that helper
is imported here: the hybrid wraps `dphi` into `[0, pi]` before the L2 because azimuth is
periodic, lloca does not, and silently adopting that would change the published-reproduction
row. The difference is documented in docs/diffs.md, not fixed here.

Paper: "ParticleNet: Jet Tagging via Particle Clouds" - https://arxiv.org/abs/1902.08570
Code:  https://github.com/hqucms/weaver-core/blob/main/weaver/nn/model/ParticleNet.py

The three things LLoCa adds over stock weaver (unchanged from lloca's file):
- a `hidden_reps_list` giving the hidden representation of each message-passing layer
- frames threaded through the network so they are available during message passing
- an edge convolution that transports neighbour features from their local frame into the
  receiver's, via TensorRepsTransform / IndexSelectFrames / ChangeOfFrames

`git diff --no-index` against the installed lloca file shows exactly the two additions above.
"""

# ruff: noqa

import torch
import torch.nn as nn

from lloca.framesnet.frames import ChangeOfFrames, IndexSelectFrames
from lloca.reps.tensorreps import TensorReps
from lloca.reps.tensorreps_transform import TensorRepsTransform

from experiments.baselines.particlenettransformer import knn as metric_knn


def change_local_frame(x_j_framej, idx, frames, trafo):
    """Transform features x_j from frame 'j' ('x_j_framej') to frame 'i' ('x_j_framei').

    Parameters
    ----------
    x_j_framej : torch.Tensor
        Input features in local frame 'j' of shape (batch_size, num_dims, num_points, k).
    idx : torch.Tensor
        Indices of the nearest neighbors in the batch of shape (batch_size*num_points*k).
    frames : Frames
        Local frames of reference for the particles, shape (num_points, 4, 4).
    trafo : TensorRepsTransform
        Transformation function to apply to the features.

    Returns
    -------
    torch.Tensor
    """
    # we use batch_size*num_points with repeats of k for idx_i, e.g. for 2 points with 3 batch and k=2,
    # idx_i becomes (0,1,2,3,4,5) -> (0,0,1,1,2,2,3,3,4,4,5,5).
    idx_i = torch.arange(
        x_j_framej.shape[2] * x_j_framej.shape[0], device=x_j_framej.device
    ).repeat_interleave(x_j_framej.shape[-1])  # identity (batch, num_points*k)
    idx_j = idx  # indices from knn (batch, num_points*k)

    frames_i = IndexSelectFrames(frames, idx_i)
    frames_j = IndexSelectFrames(frames, idx_j)
    trafo_j_to_i = ChangeOfFrames(frames_j, frames_i)  # convention: (frames_start, frames_end)

    # reshape and apply trafo
    x_j_framej_2 = x_j_framej.permute(0, 2, 3, 1)  # (batch_size, num_points, k, num_dims)
    pre = x_j_framej_2.reshape(-1, x_j_framej_2.shape[-1])  # (batch_size*num_points*k, num_dims)
    x_j_framei = trafo(pre, trafo_j_to_i)
    x_j_framei = x_j_framei.view(x_j_framej_2.shape).permute(
        0, 3, 1, 2
    )  # (batch_size, num_dims, num_points, k)
    return x_j_framei


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k + 1, dim=-1)[1][:, :, 1:]  # (batch_size, num_points, k)
    return idx


# v1 is faster on GPU
def get_graph_feature_v1(x, k, idx, frames, trafo):
    batch_size, num_dims, num_points = x.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    fts = x.transpose(2, 1).reshape(
        -1, num_dims
    )  # -> (batch_size, num_points, num_dims) -> (batch_size*num_points, num_dims)
    fts = fts[idx, :].view(
        batch_size, num_points, k, num_dims
    )  # neighbors: -> (batch_size*num_points*k, num_dims) -> ...
    fts = fts.permute(0, 3, 1, 2).contiguous()  # (batch_size, num_dims, num_points, k)
    x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)
    fts = change_local_frame(fts, idx, frames, trafo)
    fts = torch.cat((x, fts - x), dim=1)  # ->(batch_size, 2*num_dims, num_points, k)
    return fts


# v2 is faster on CPU
def get_graph_feature_v2(x, k, idx, frames, trafo):
    batch_size, num_dims, num_points = x.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    fts = x.transpose(0, 1).reshape(
        num_dims, -1
    )  # -> (num_dims, batch_size, num_points) -> (num_dims, batch_size*num_points)
    fts = fts[:, idx].view(
        num_dims, batch_size, num_points, k
    )  # neighbors: -> (num_dims, batch_size*num_points*k) -> ...
    fts = fts.transpose(1, 0).contiguous()  # (batch_size, num_dims, num_points, k)
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

    def __init__(
        self,
        k,
        in_reps,
        out_feats,
        batch_norm=True,
        activation=True,
        cpu_mode=False,
    ):
        super(EdgeConvBlock, self).__init__()
        self.k = k
        self.batch_norm = batch_norm
        self.activation = activation
        self.num_layers = len(out_feats)
        self.get_graph_feature = get_graph_feature_v2 if cpu_mode else get_graph_feature_v1
        in_feat = in_reps.dim
        self.trafo = TensorRepsTransform(TensorReps(in_reps))

        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(
                nn.Conv2d(
                    2 * in_feat if i == 0 else out_feats[i - 1],
                    out_feats[i],
                    kernel_size=1,
                    bias=False if self.batch_norm else True,
                )
            )

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

    def forward(self, points, features, frames, metric="deltaR", mask=None):
        if metric == "minkowski":
            # the hybrid's helper, imported rather than copied: it is already pinned against
            # this file's EdgeConv by the duplication parity tests
            topk_indices = metric_knn(points, self.k, metric="minkowski", mask=mask)
        else:
            topk_indices = knn(points, self.k)  # lloca's, verbatim -> deltaR stays bit-identical
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


class ParticleNet(nn.Module):
    """ParticleNet with local frame transformations."""

    def __init__(
        self,
        input_dims,
        hidden_reps_list,
        num_classes,
        conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
        fc_params=[(128, 0.1)],
        use_fusion=True,
        use_fts_bn=True,
        use_counts=True,
        for_inference=False,
        for_segmentation=False,
        knn_metric="deltaR",
        **kwargs,
    ):
        # hidden_reps_list: hidden representation for message-passing at beginning of each layer
        super(ParticleNet, self).__init__(**kwargs)
        hidden_reps_list = [TensorReps(x) for x in hidden_reps_list]
        assert input_dims == hidden_reps_list[0].dim
        assert len(hidden_reps_list) == len(conv_params)

        self.use_fts_bn = use_fts_bn
        if self.use_fts_bn:
            self.bn_fts = nn.BatchNorm1d(hidden_reps_list[0].dim)

        self.use_counts = use_counts
        if knn_metric not in ("deltaR", "minkowski"):
            raise ValueError(f"knn_metric must be 'deltaR' or 'minkowski', got '{knn_metric}'")
        self.knn_metric = knn_metric

        self.edge_convs = nn.ModuleList()
        for idx, layer_param in enumerate(conv_params):
            k, channels = layer_param
            in_reps = hidden_reps_list[idx]
            assert (
                in_reps.dim == conv_params[idx - 1][1][-1] if idx > 0 else hidden_reps_list[0].dim
            )
            self.edge_convs.append(
                EdgeConvBlock(k=k, in_reps=in_reps, out_feats=channels, cpu_mode=for_inference)
            )

        self.use_fusion = use_fusion
        if self.use_fusion:
            in_chn = sum(x[-1] for _, x in conv_params)
            out_chn = max(128, min((in_chn // 128) * 128, 1024))
            self.fusion_block = nn.Sequential(
                nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_chn),
                nn.ReLU(),
            )

        self.for_segmentation = for_segmentation

        fcs = []
        for idx, layer_param in enumerate(fc_params):
            channels, drop_rate = layer_param
            if idx == 0:
                in_chn = out_chn if self.use_fusion else conv_params[-1][1][-1]
            else:
                in_chn = fc_params[idx - 1][0]
            if self.for_segmentation:
                fcs.append(
                    nn.Sequential(
                        nn.Conv1d(in_chn, channels, kernel_size=1, bias=False),
                        nn.BatchNorm1d(channels),
                        nn.ReLU(),
                        nn.Dropout(drop_rate),
                    )
                )
            else:
                fcs.append(
                    nn.Sequential(nn.Linear(in_chn, channels), nn.ReLU(), nn.Dropout(drop_rate))
                )
        if self.for_segmentation:
            fcs.append(nn.Conv1d(fc_params[-1][0], num_classes, kernel_size=1))
        else:
            fcs.append(nn.Linear(fc_params[-1][0], num_classes))
        self.fc = nn.Sequential(*fcs)

        self.for_inference = for_inference

    def forward(self, points, features, frames, mask=None, v=None):
        #         print('points:\n', points)
        #         print('features:\n', features)
        if mask is None:
            mask = features.abs().sum(dim=1, keepdim=True) != 0  # (N, 1, P)
        points *= mask
        features *= mask
        coord_shift = (mask == 0) * 1e9
        if self.use_counts:
            counts = mask.float().sum(dim=-1)
            counts = torch.max(counts, torch.ones_like(counts))  # >=1

        if self.use_fts_bn:
            fts = self.bn_fts(features) * mask
        else:
            fts = features
        outputs = []
        for idx, conv in enumerate(self.edge_convs):
            # layer 0 seeds the graph from the geometry, deeper layers from their own
            # features -- as in every ParticleNet. Only layer 0 can use a Lorentz metric.
            if idx == 0 and self.knn_metric == "minkowski" and v is not None:
                pts, metric = v + coord_shift, "minkowski"
            else:
                pts, metric = (points if idx == 0 else fts) + coord_shift, "deltaR"
            fts = conv(pts, fts, frames, metric=metric, mask=mask) * mask
            if self.use_fusion:
                outputs.append(fts)
        if self.use_fusion:
            fts = self.fusion_block(torch.cat(outputs, dim=1)) * mask

        #         assert(((fts.abs().sum(dim=1, keepdim=True) != 0).float() - mask.float()).abs().sum().item() == 0)

        if self.for_segmentation:
            x = fts
        else:
            if self.use_counts:
                x = fts.sum(dim=-1) / counts  # divide by the real counts
            else:
                x = fts.mean(dim=-1)

        output = self.fc(x)
        if self.for_inference:
            # single-logit heads (top-tagging here: out_channels=1, BCE) must use sigmoid --
            # softmax over a 1-wide dim is identically 1.0, silently making every score
            # constant (AUC 0.5). Multi-class (JetClass) is unchanged.
            output = torch.sigmoid(output) if output.shape[1] == 1 else torch.softmax(output, dim=1)
        # print('output:\n', output)
        return output
