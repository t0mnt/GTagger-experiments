import os

import torch
from lgatr import embed_vector, extract_scalar
from lloca.framesnet.frames import Frames
from lloca.framesnet.nonequi_frames import IdentityFrames
from lloca.reps.tensorreps import TensorReps
from lloca.reps.tensorreps_transform import TensorRepsTransform
from lloca.utils.lorentz import lorentz_eye
from lloca.utils.orthogonalize_4d import orthogonalize_4d
from lloca.utils.utils import (
    get_batch_from_ptr,
    get_edge_attr,
    get_edge_index_from_ptr,
    get_ptr_from_batch,
)
from torch import nn
from torch_geometric.nn.aggr import MeanAggregation
from torch_geometric.utils import scatter, to_dense_batch

from experiments.logger import LOGGER
from experiments.misc import get_attention_mask
from experiments.tagging.embedding import get_tagging_features

# FLASH-3 step 2 A/B switch (round-trip #4 audit): the FC graph is the one place
# where the padded segment sum's K bound (n_nodes-1) is NOT small relative to the
# true degrees, and #4's profile shows the padded/fused aggregation kernel at 16%
# of tag_cgenn CUDA -- while unsafe=True has made the segment_reduce alternative
# sync-free too. CGENN_FC_PADDED=0 passes knn_k=None (segment_reduce receiver
# side) for a one-line A/B without a code edit. Read ONCE at import, shield style.
_FC_PADDED = os.environ.get("CGENN_FC_PADDED", "1") != "0"

# Regional-compilation switch (flash round-trip #6 experiment): CGENN_REGIONAL=1
# makes CGENNWrapper's compile=True compile SUBMODULES instead of the whole net --
# see the comment at the wrapper's compile site. Read at CONSTRUCTION (not import):
# it gates a one-time build decision, never the hot path, and construction-time
# reads keep it monkeypatchable in the gates.
_REGIONAL_LEAVES = (nn.Linear, nn.BatchNorm1d, nn.ReLU, nn.SiLU, nn.Sigmoid,
                    nn.Dropout, nn.Identity)


def _compile_regional(net):
    """Compile every nn.Sequential made purely of plain-nn leaves, in place.

    The structural predicate selects exactly the CGL scalar MLPs (phi_h, theta_h,
    psi_x, chi_x: Linear/BatchNorm1d/ReLU stacks) and any plain output head, and
    excludes everything geometric-algebra shaped (phi_x/theta_x hold FCGP/MVLayerNorm
    modules, which fail the leaf test), so the flash custom op and the equivariant ops
    stay in eager orchestration. Each unit compiles with dynamic=True: its batch dim
    is E or N, both varying per step. Returns the number of compiled units."""
    n = 0
    for mod in net.modules():
        if isinstance(mod, nn.Sequential) and len(mod) > 0 and all(
                isinstance(c, _REGIONAL_LEAVES) for c in mod.children()):
            mod.compile(dynamic=True)
            n += 1
    return n

# every to_dense_batch below is deliberate: this repo uses zero padding over sparse jet
# representations, as the MPNN portion of the GNNs is currently not shaped for the latter


class _activation_memory_budget:
    """Scoped `torch._functorch.config.activation_memory_budget` around a compiled call.

    lgatr 2.0 exposes this on every net (`utils/compile.compile_model`) as the escape hatch
    for activation-memory pressure, so the CGENN wrappers expose it too -- same knob, same
    semantics, adapted to this repo's "compile the net, keep the wrapper eager" posture.

    A fraction in (0, 1]: AOTAutograd's min-cut partitioner is told to keep saved
    activations under `budget` x what the default partition would save, buying the
    difference with recomputation in the backward. 1.0 (torch's default) is "save
    whatever is cheapest"; lower values trade backward FLOPs for peak memory. `None` --
    the shipped default -- leaves torch's global setting untouched, so the knob is fully
    inert until someone sets it.

    Scoped per call rather than set globally, exactly like the `recompute_views` patch in
    LorentzNetLGATrSlimGraphGPSWrapper: the smoke test and the FLOPs harness build many
    models in one process, and a global would leak across them. The partitioner reads the
    config while COMPILING, and compilation happens inside the wrapped call, so a scope
    around the call covers every (re)compilation of that net.

    Reach for it only on a measured OOM, and do not expect `gp_impl: sparse` to have
    relieved the pressure first: its 3.5x retention win is EAGER-only. Under compile --
    which every CGENN config ships -- AOT's partitioner already equalizes retention across
    all three impls, which is both why the sparse-GP autograd Function is routed off the
    compiled path and why this knob is the only lever left there
    (docs/cgenn-compile.md, CORRECTION).
    """

    def __init__(self, budget):
        self.budget = budget
        # a STACK, not a slot: one instance is held per wrapper and entered on every
        # forward, so a single slot would be clobbered by any nesting or concurrency
        self._active = []

    def __enter__(self):
        if self.budget is None:
            return self
        import torch._functorch.config as _fc

        if not hasattr(_fc, "activation_memory_budget"):
            raise ValueError("activation_memory_budget requires torch>=2.4")
        patch = _fc.patch(activation_memory_budget=self.budget)
        patch.__enter__()
        self._active.append(patch)
        return self

    def __exit__(self, *exc):
        if self._active:
            self._active.pop().__exit__(*exc)
        return False


class TaggerWrapper(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        framesnet,
        add_fourmomenta_backbone: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_fourmomenta_backbone = add_fourmomenta_backbone
        self.framesnet = framesnet
        self.trafo_fourmomenta = TensorRepsTransform(TensorReps("1x1n"))
        # subclasses that need a single covariant per-event frame (e.g. for a prepended
        # readout token in a tensorial backbone) set this; forward then fills _jet_frames.
        self.compute_jet_frames = False
        self._jet_frames = None

    def jet_frames(self, fourmomenta, scalars, ptr, is_spurion=None):
        """A single covariant frame per event: the boost into the jet rest frame.

        orthogonalize_4d makes the first of its three reference vectors the timelike axis,
        so we pass the jet four-momentum (the sum of the REAL particle four-momenta) as
        that first vector -- the frame's time axis is then the jet direction (its rest
        frame). The two remaining axes (the spatial orientation, which a single momentum
        leaves undetermined) are fixed by the framesnet's per-particle equivariant
        reference vectors, averaged over the event. The equivectors see the SAME
        with-spurion token list as the per-particle framesnet path: the beam/time
        spurions rank-lift the reference set (a 2-constituent or collinear jet spans
        <3 independent directions on its own -- momenta alone would hand the
        orthogonalizer a coplanar stack and return a non-Lorentz frame), and the CLS
        frame then breaks exactly the symmetries the per-particle frames already break,
        no more. Returns Frames (B, 4, 4), or None for a non-learned (identity)
        framesnet.

        Two guards for degenerate inputs:
        - framesnets with a single learned equivector (LearnedSO2Frames) pad the
          reference stack with the beam axis e_z -- the same trivial axis their own
          frame construction uses, invariant under the SO(2) subgroup they canonicalize;
        - events whose orthogonalized frame still fails the Lorentz condition
          (|L eta L^T - eta| large; possible when spurions are disabled) fall back to
          the identity readout frame for those events only.
        """
        fn = self.framesnet
        if not hasattr(fn, "equivectors"):
            return None  # IdentityFrames etc. -> identity readout frame (handled downstream)
        batch = get_batch_from_ptr(ptr)
        if is_spurion is None:
            is_spurion = torch.zeros(fourmomenta.shape[0], dtype=torch.bool, device=batch.device)
        B = ptr.numel() - 1
        jet_p = scatter(
            fourmomenta[~is_spurion], batch[~is_spurion], dim=0, reduce="sum", dim_size=B
        )  # (B, 4): jet four-momentum from the real particles
        jet_p = fn.mass_regularize(jet_p)
        # set-level equivectors (e.g. pelican) require num_graphs; the main framesnet path
        # passes it, so mirror that here.
        vecs = fn.equivectors(
            fn.mass_regularize(fourmomenta),
            scalars=scalars,
            ptr=ptr,
            num_graphs=B,
        )
        vecs = scatter(vecs, batch, dim=0, reduce="mean", dim_size=B)  # (B, n_vectors, 4)
        refs = [jet_p.unsqueeze(1), vecs[:, :2]]
        n_rot_refs = min(vecs.shape[1], 2)
        if n_rot_refs < 2:
            # e.g. LearnedSO2Frames (n_vectors=1) fixes the missing axes to trivial vectors,
            # so pad with the beam axis e_z (invariant under the SO(2)-about-z subgroup)
            pad = torch.zeros(B, 2 - n_rot_refs, 4, device=jet_p.device, dtype=jet_p.dtype)
            pad[..., -1] = 1.0
            refs.append(pad)
        vecs = torch.cat(refs, dim=1)  # (B, 3, 4): time axis + two rotation references
        # jet_frames uses the 4d orthogonalizer with the framesnet's ortho_kwargs. PD-family
        # framesnets key the coplanar regulator as `eps_reg` (for orthogonalize_3d) but
        # orthogonalize_4d wants `eps_reg_coplanar` -- translate so any framesnet works.
        ortho_kwargs = dict(fn.ortho_kwargs)
        if "eps_reg" in ortho_kwargs:
            ortho_kwargs.setdefault("eps_reg_coplanar", ortho_kwargs.pop("eps_reg"))
        # library `checks` would hard-assert on a degenerate event BEFORE our per-event
        # identity fallback below can handle it -- we do the validation ourselves
        ortho_kwargs["checks"] = False
        trafo = orthogonalize_4d(vecs, **ortho_kwargs)  # (B, 4, 4)
        # identity fallback for events whose reference stack was still too degenerate
        # (generic events deviate by ~1e-6 in fp32; broken ones by O(0.01..1))
        eta = torch.diag(
            torch.tensor([1.0, -1.0, -1.0, -1.0], device=trafo.device, dtype=trafo.dtype)
        )
        dev = (trafo @ eta @ trafo.transpose(-1, -2) - eta).abs().amax(dim=(-2, -1))
        bad = dev > 1e-3
        if bad.any():
            if not getattr(self, "_jet_frames_degenerate_warned", False):
                LOGGER.warning(
                    f"jet_frames: {int(bad.sum())} event(s) with a degenerate reference set "
                    f"(max |L eta L^T - eta| = {dev.max().item():.2e}); using the identity "
                    f"readout frame for those events (warning shown once)"
                )
                self._jet_frames_degenerate_warned = True
            eye = lorentz_eye((1,), device=trafo.device, dtype=trafo.dtype)
            trafo = torch.where(bad[:, None, None], eye, trafo)
        return Frames(trafo.to(fourmomenta.dtype))

    def init_standardization(self, fourmomenta, ptr, reduce_size=None):
        # framesnet equivectors edge_attr standardization (if applicable)
        if hasattr(self.framesnet, "equivectors") and hasattr(
            self.framesnet.equivectors, "init_standardization"
        ):
            fourmomenta_reduced = (
                fourmomenta[:reduce_size] if reduce_size is not None else fourmomenta
            )
            self.framesnet.equivectors.init_standardization(fourmomenta_reduced, ptr)

    def forward(self, embedding):
        # extract embedding
        fourmomenta_withspurions = embedding["fourmomenta"]
        scalars_withspurions = embedding["scalars"]
        global_tagging_features_withspurions = embedding["tagging_features"]
        batch_withspurions = embedding["batch"]
        is_spurion = embedding["is_spurion"]
        ptr_withspurions = embedding["ptr"]
        num_graphs = embedding["num_graphs"]
        nospurion_idxs = (~is_spurion).nonzero(as_tuple=False).squeeze(-1)

        # remove spurions from the data again and recompute attributes
        fourmomenta_nospurions = fourmomenta_withspurions.index_select(0, nospurion_idxs)
        scalars_nospurions = scalars_withspurions.index_select(0, nospurion_idxs)

        batch_nospurions = batch_withspurions.index_select(0, nospurion_idxs)
        ptr_nospurions = get_ptr_from_batch(batch_nospurions)
        B = ptr_nospurions.numel() - 1

        scalars_withspurions = torch.cat(
            [scalars_withspurions, global_tagging_features_withspurions], dim=-1
        )
        frames_spurions, tracker = self.framesnet(
            fourmomenta_withspurions,
            scalars_withspurions,
            ptr=ptr_withspurions,
            return_tracker=True,
            num_graphs=num_graphs,
        )
        matrices = frames_spurions.matrices.index_select(0, nospurion_idxs)
        frames_nospurions = Frames(
            matrices,
            is_global=frames_spurions.is_global,
            det=frames_spurions.det.index_select(0, nospurion_idxs),
            inv=frames_spurions.inv.index_select(0, nospurion_idxs),
            is_identity=frames_spurions.is_identity,
            device=frames_spurions.device,
            dtype=frames_spurions.dtype,
            shape=matrices.shape,
        )

        # optional single covariant per-event (jet) frame. The equivectors get the same
        # with-spurion inputs as the framesnet above (spurions rank-lift degenerate jets and
        # match the per-particle symmetry breaking); the jet momentum uses real particles only.
        if self.compute_jet_frames:
            self._jet_frames = self.jet_frames(
                fourmomenta_withspurions,
                scalars_withspurions,
                ptr_withspurions,
                is_spurion=is_spurion,
            )

        # transform features into local frames
        fourmomenta_local_nospurions = self.trafo_fourmomenta(
            fourmomenta_nospurions, frames_nospurions
        )
        jet_nospurions = scatter(
            fourmomenta_nospurions,
            index=batch_nospurions,
            dim=0,
            reduce="sum",
            dim_size=B,
        ).index_select(0, batch_nospurions)
        jet_local_nospurions = self.trafo_fourmomenta(jet_nospurions, frames_nospurions)
        local_tagging_features_nospurions = get_tagging_features(
            fourmomenta_local_nospurions,
            jet_local_nospurions,
            tagging_features="all",
        )

        features_local_nospurions = torch.cat(
            [scalars_nospurions, local_tagging_features_nospurions], dim=-1
        )
        if self.add_fourmomenta_backbone:
            features_local_nospurions = torch.cat(
                [features_local_nospurions, fourmomenta_local_nospurions], dim=-1
            )

        # change dtype (see embedding.py fourmomenta_float64 option)
        features_local_nospurions = features_local_nospurions.to(scalars_nospurions.dtype)
        frames_nospurions.to(scalars_nospurions.dtype)

        return (
            features_local_nospurions,
            fourmomenta_local_nospurions,
            frames_nospurions,
            ptr_nospurions,
            batch_nospurions,
            tracker,
        )


class AggregatedTaggerWrapper(TaggerWrapper):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.aggregator = MeanAggregation()

    def extract_score(self, features, ptr):
        B = ptr.numel() - 1
        score = self.aggregator(features, ptr=ptr, dim_size=B)
        return score


class GraphNetWrapper(AggregatedTaggerWrapper):
    def __init__(
        self,
        net,
        include_edges,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.include_edges = include_edges
        self.net = net(in_channels=self.in_channels, out_channels=self.out_channels)
        if self.include_edges:
            self.register_buffer("edge_inited", torch.tensor(False))
            self.register_buffer("edge_mean", torch.tensor(0.0))
            self.register_buffer("edge_std", torch.tensor(1.0))

    def forward(self, embedding):
        (
            features_local,
            fourmomenta_local,
            frames,
            ptr,
            batch,
            tracker,
        ) = super().forward(embedding)

        edge_index = get_edge_index_from_ptr(ptr, features_local.shape, remove_self_loops=True)
        if self.include_edges:
            edge_attr = self.get_edge_attr(fourmomenta_local, edge_index).to(features_local.dtype)
        else:
            edge_attr = None
        # network
        outputs = self.net(
            inputs=features_local,
            frames=frames,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        # aggregation
        score = self.extract_score(outputs, ptr)
        return score, tracker, frames

    def get_edge_attr(self, fourmomenta, edge_index):
        edge_attr = get_edge_attr(fourmomenta, edge_index)
        if not self.edge_inited:
            self.edge_mean = edge_attr.mean().detach()
            self.edge_std = edge_attr.std().clamp(min=1e-5).detach()
            self.edge_inited = torch.tensor(True, device=edge_attr.device)
        edge_attr = (edge_attr - self.edge_mean) / self.edge_std
        return edge_attr.unsqueeze(-1)


class TransformerWrapper(AggregatedTaggerWrapper):
    def __init__(
        self,
        net,
        *args,
        use_amp=False,
        attention_backend="xformers",
        mean_aggregation=True,
        compile=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_amp = use_amp
        self.attention_backend = attention_backend
        self.mean_aggregation = mean_aggregation
        self.net = net(in_channels=self.in_channels, out_channels=self.out_channels)

        if attention_backend == "flex":
            compile_flex_attention(package_name="lloca")
        if compile:
            # compile the net only, net-level like tag_lgatr (recorded:
            # docs/cgenn-compile.md, Stage 4)
            self.net.compile(dynamic=True)

    def forward(self, embedding):
        # precompute attention mask to avoid cudaStreamSynchronize
        # from .tolist() in get_xformers_attention_mask
        batch_withspurions = embedding["batch"]
        is_spurion = embedding["is_spurion"]
        nospurion_idxs = (~is_spurion).nonzero(as_tuple=False).squeeze(-1)
        batch_nospurions = batch_withspurions.index_select(0, nospurion_idxs)
        ptr_nospurions = get_ptr_from_batch(batch_nospurions)
        ptr, batch = ptr_nospurions, batch_nospurions
        if not self.mean_aggregation:
            batchsize = len(ptr) - 1
            ptr = ptr.clone()
            ptr[1:] = ptr[1:] + (torch.arange(batchsize, device=ptr.device) + 1)
            batch = get_batch_from_ptr(ptr)
        mask_kwarg = get_attention_mask(
            batch,
            dtype=embedding["scalars"].dtype,
            attention_backend=self.attention_backend,
        )

        (
            features_local,
            _,
            frames,
            ptr,
            batch,
            tracker,
        ) = super().forward(embedding)

        # handle global token
        if self.mean_aggregation:
            is_global = None
        else:
            # append global tokens to batch, ptr, features_local and frames
            # and keep a is_global mask for later extraction
            batchsize = len(ptr) - 1
            global_idxs = ptr[:-1] + torch.arange(batchsize, device=batch.device)
            is_global = torch.zeros(
                features_local.shape[0] + batchsize,
                dtype=torch.bool,
                device=ptr.device,
            )
            is_global[global_idxs] = True
            features_local_buffer = features_local.clone()
            features_local = torch.zeros(
                is_global.shape[0],
                *features_local.shape[1:],
                dtype=features_local.dtype,
                device=features_local.device,
            )
            features_local[~is_global] = features_local_buffer
            is_global_channel = torch.zeros(
                features_local.shape[0],
                1,
                dtype=features_local.dtype,
                device=features_local.device,
            )
            is_global_channel[is_global] = 1
            features_local = torch.cat((features_local, is_global_channel), dim=-1)

            # global token frames are identity
            matrices_new = (
                torch.eye(4, device=frames.device, dtype=frames.dtype)
                .unsqueeze(0)
                .expand(is_global.shape[0], -1, -1)
            ).clone()
            matrices_new[~is_global] = frames.matrices
            det_new = torch.ones(
                is_global.shape[0], device=frames.device, dtype=frames.dtype
            ).clone()
            det_new[~is_global] = frames.det
            inv_new = (
                torch.eye(4, device=frames.device, dtype=frames.dtype)
                .unsqueeze(0)
                .expand(is_global.shape[0], -1, -1)
            ).clone()
            inv_new[~is_global] = frames.inv
            frames = Frames(
                matrices_new,
                is_global=frames.is_global,
                det=det_new,
                inv=inv_new,
                is_identity=frames.is_identity,
                device=frames.device,
                dtype=frames.dtype,
                shape=matrices_new.shape,
            )

            ptr[1:] = ptr[1:] + (torch.arange(batchsize, device=ptr.device) + 1)
            batch = get_batch_from_ptr(ptr)

        # add artificial batch dimension
        features_local = features_local.unsqueeze(0)
        frames = frames.reshape(1, *frames.shape)

        # network
        with torch.autocast(features_local.device.type, enabled=self.use_amp):
            outputs = self.net(inputs=features_local, frames=frames, **mask_kwarg)

        # aggregation
        outputs = outputs[0, ...]
        if self.mean_aggregation:
            score = self.extract_score(outputs, ptr)
        else:
            score = outputs[is_global]
        return score, tracker, frames


class ParticleNetWrapper(AggregatedTaggerWrapper):
    def __init__(
        self,
        net,
        *args,
        compile=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.net = net(input_dims=self.in_channels, num_classes=self.out_channels)
        if compile:
            # compile the net only; dense top-k kNN is shape-static and traces clean
            # (recorded: docs/cgenn-compile.md, Stage 4)
            self.net.compile(dynamic=True)

    def forward(self, embedding):
        (
            features_local,
            fourmomenta_local,
            frames,
            _,
            batch,
            tracker,
        ) = super().forward(embedding)
        # ParticleNet kNN uses L2 on (phi, eta) = dphi/deta at positions 4,5 of the 7-feature
        # local block, which sits AFTER extra scalars. A hardcoded [4,5] is right only for
        # extra_scalars=0 (top-tagging); on JetClass it would cluster layer-0 kNN by PID.
        n_extra = features_local.shape[-1] - 7 - (4 if self.add_fourmomenta_backbone else 0)
        assert n_extra >= 0, (
            f"unexpected feature layout for ParticleNetWrapper: {features_local.shape[-1]} channels"
        )
        phieta_local = features_local[..., [n_extra + 4, n_extra + 5]]
        phieta_local, mask = to_dense_batch(phieta_local, batch)
        features_local, _ = to_dense_batch(features_local, batch)
        phieta_local = phieta_local.transpose(1, 2)
        features_local = features_local.transpose(1, 2)
        # four-momenta reach the backbone only so layer-0 kNN can rank on the Minkowski
        # interval; unused when knn_metric=deltaR (the default), which stays bit-identical
        fourmomenta_local, _ = to_dense_batch(fourmomenta_local, batch)
        fourmomenta_local = fourmomenta_local.transpose(1, 2).contiguous()  # (B, 4, P)
        dense_frames, _ = to_dense_batch(frames.matrices, batch)
        dense_frames[~mask] = (
            torch.eye(4, device=dense_frames.device, dtype=dense_frames.dtype)
            .unsqueeze(0)
            .expand((~mask).sum(), -1, -1)
        )

        frames = Frames(
            dense_frames.view(-1, 4, 4),
            is_global=frames.is_global,
            is_identity=frames.is_identity,
            device=frames.device,
            dtype=frames.dtype,
            shape=frames.matrices.shape,
        )
        mask = mask.unsqueeze(1)

        # network
        score = self.net(
            points=phieta_local,
            features=features_local,
            frames=frames,
            mask=mask,
            v=fourmomenta_local,
        )
        return score, tracker, frames


class LGATrWrapper(nn.Module):
    def __init__(
        self,
        net,
        framesnet,
        out_channels,
        mean_aggregation=False,
        use_amp=False,
        attention_backend="xformers",
    ):
        super().__init__()
        self.use_amp = use_amp
        self.attention_backend = attention_backend
        self.net = net(out_mv_channels=out_channels)
        self.aggregator = MeanAggregation() if mean_aggregation else None

        self.framesnet = framesnet  # not actually used
        assert isinstance(framesnet, IdentityFrames)

        if attention_backend == "flex":
            compile_flex_attention(package_name="lgatr")

    def forward(self, embedding):
        # extract embedding (includes spurions)
        fourmomenta = embedding["fourmomenta"]
        scalars = torch.cat([embedding["scalars"], embedding["tagging_features"]], dim=-1)
        batch = embedding["batch"]
        ptr = embedding["ptr"]
        is_spurion = embedding["is_spurion"]

        # rescale fourmomenta (but not the spurions)
        fourmomenta[~is_spurion] = fourmomenta[~is_spurion] / 20

        # handle global token
        if self.aggregator is None:
            batchsize = len(ptr) - 1
            global_idxs = ptr[:-1] + torch.arange(batchsize, device=batch.device)
            is_global = torch.zeros(
                fourmomenta.shape[0] + batchsize,
                dtype=torch.bool,
                device=ptr.device,
            )
            is_global[global_idxs] = True
            fourmomenta_buffer = fourmomenta.clone()
            fourmomenta = torch.zeros(
                is_global.shape[0],
                *fourmomenta.shape[1:],
                dtype=fourmomenta.dtype,
                device=fourmomenta.device,
            )
            fourmomenta[~is_global] = fourmomenta_buffer
            scalars_buffer = scalars.clone()
            scalars = torch.zeros(
                fourmomenta.shape[0],
                scalars.shape[1] + 1,
                dtype=scalars.dtype,
                device=scalars.device,
            )
            token_idx = torch.nn.functional.one_hot(torch.arange(1, device=scalars.device))
            token_idx = token_idx.repeat(batchsize, 1)
            scalars[~is_global] = torch.cat(
                (
                    scalars_buffer,
                    torch.zeros(
                        scalars_buffer.shape[0],
                        token_idx.shape[1],
                        dtype=scalars.dtype,
                        device=scalars.device,
                    ),
                ),
                dim=-1,
            )
            scalars[is_global] = torch.cat(
                (
                    torch.zeros(
                        token_idx.shape[0],
                        scalars_buffer.shape[1],
                        dtype=scalars.dtype,
                        device=scalars.device,
                    ),
                    token_idx,
                ),
                dim=-1,
            )
            ptr[1:] = ptr[1:] + (torch.arange(batchsize, device=ptr.device) + 1)
            batch = get_batch_from_ptr(ptr)
        else:
            is_global = None

        fourmomenta = fourmomenta.unsqueeze(0).to(scalars.dtype)
        scalars = scalars.unsqueeze(0)

        mask_kwarg = get_attention_mask(
            batch,
            dtype=scalars.dtype,
            attention_backend=self.attention_backend,
        )

        mv = embed_vector(fourmomenta).unsqueeze(-2)
        s = scalars if scalars.shape[-1] > 0 else None

        with torch.autocast(mv.device.type, enabled=self.use_amp):
            mv_outputs, _ = self.net(mv, s, **mask_kwarg)
        out = extract_scalar(mv_outputs)[0, :, :, 0]

        if self.aggregator is not None:
            B = ptr.numel() - 1
            logits = self.aggregator(out, index=batch, dim_size=B)
        else:
            logits = out[is_global]
        return logits, {}, None


class ParTWrapper(TaggerWrapper):
    def __init__(
        self,
        net,
        *args,
        use_amp=False,
        compile=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.net = net(input_dim=self.in_channels, num_classes=self.out_channels, use_amp=use_amp)
        if compile:
            # compiled ParT runs the DENSE pair path: the default sparse path gathers
            # real pairs via nonzero (data-dependent by design, untraceable); the dense
            # twin is the same function at reassociation level (measured 2.2e-15, TOL
            # gate bar 1e-10 -- docs/cgenn-compile.md, Stage 4)
            if hasattr(self.net, "pair_embed") and self.net.pair_embed is not None:
                self.net.pair_embed.sparse_eval = False
                self.net.pair_embed.compiled_dense = True
            self.net.compile(dynamic=True)
            self._compiled = True

    def forward(self, embedding):
        (
            features_local,
            fourmomenta_local,
            frames,
            _,
            batch,
            tracker,
        ) = super().forward(embedding)
        fourmomenta_local = fourmomenta_local.to(features_local.dtype)
        fourmomenta_local = fourmomenta_local[..., [1, 2, 3, 0]]  # need (px, py, pz, E)

        features_local, mask = to_dense_batch(features_local, batch)
        fourmomenta_local, _ = to_dense_batch(fourmomenta_local, batch)
        features_local = features_local.transpose(1, 2)
        fourmomenta_local = fourmomenta_local.transpose(1, 2)

        frames_matrices, _ = to_dense_batch(frames.matrices, batch)
        det, _ = to_dense_batch(frames.det, batch)
        inv, _ = to_dense_batch(frames.inv, batch)
        frames_matrices[~mask] = lorentz_eye(
            frames_matrices[~mask].shape[:-2],
            device=frames.device,
            dtype=frames.dtype,
        )
        frames = Frames(
            matrices=frames_matrices,
            is_global=frames.is_global,
            det=det,
            inv=inv,
            is_identity=frames.is_identity,
            device=frames.device,
            dtype=frames.dtype,
            shape=frames.matrices.shape,
        )

        mask = mask.unsqueeze(1).float()

        # network
        if hasattr(self.net, "trimmer"):
            self.net.trimmer.tick()  # warmup bookkeeping, eager by design (see trimmer)
        if getattr(self, "_compiled", False) or hasattr(self.net, "_orig_mod"):
            # dynamic=True alone never promotes the padded seq dim here: each new batch
            # max-length compiles one more static graph (RECOMP gate found [1,2,3]).
            # maybe_mark_dynamic is the SOFT per-tensor hint: on the shipped
            # identity-frames path nothing specializes, so the net compiles ONE
            # seqlen-dynamic graph (RECOMP [1,1,1]); under a LEARNED framesnet the
            # lloca transport pins the seq dim to a constant, and the hard
            # mark_dynamic turned that into a ConstraintViolationError (final audit
            # finding) -- the soft hint degrades to per-shape specialization instead
            for _t in (features_local, fourmomenta_local, mask):
                torch._dynamo.maybe_mark_dynamic(_t, 2)
            # the Frames object carries three tensors; an unmarked one re-pins the graph
            for _t in (frames.matrices, frames.det, frames.inv):
                torch._dynamo.maybe_mark_dynamic(_t, 1)
        score = self.net(
            x=features_local,
            frames=frames,
            v=fourmomenta_local,
            mask=mask,
        )
        return score, tracker, frames


class MIParTWrapper(ParTWrapper):
    def __init__(self, *args, **kwargs):
        # compile support was not pursued for MIParT (operator decision): its backbone
        # keeps nn.MHA warn-breaks + trimmer counters that were only fixed in the local
        # ParT port, and ParTWrapper's twin flags do not exist on its PairEmbed
        assert not kwargs.get("compile", False), "MIParT has no compile support"
        super().__init__(*args, **kwargs)
        assert isinstance(self.framesnet, IdentityFrames)

    def forward(self, embedding):
        (
            features_local,
            fourmomenta_local,
            frames,
            _,
            batch,
            tracker,
        ) = super(ParTWrapper, self).forward(embedding)
        fourmomenta_local = fourmomenta_local.to(features_local.dtype)
        fourmomenta_local = fourmomenta_local[..., [1, 2, 3, 0]]  # need (px, py, pz, E)

        features_local, mask = to_dense_batch(features_local, batch)
        fourmomenta_local, _ = to_dense_batch(fourmomenta_local, batch)
        features_local = features_local.transpose(1, 2)
        fourmomenta_local = fourmomenta_local.transpose(1, 2)
        mask = mask.unsqueeze(1).float()

        # network
        score = self.net(
            x=features_local,
            v=fourmomenta_local,
            mask=mask,
        )
        return score, tracker, frames


class LorentzNetWrapper(nn.Module):
    def __init__(
        self,
        net,
        framesnet,
        out_channels,
        compile=False,
    ):
        super().__init__()
        self.net = net(n_class=out_channels)
        if compile:
            # compile the net only; the wrapper's edge building stays eager by design
            # (recorded: docs/cgenn-compile.md, Stage 2)
            self.net.compile(dynamic=True)

        self.framesnet = framesnet  # not actually used
        assert isinstance(framesnet, IdentityFrames)

    def forward(self, embedding):
        # extract embedding (includes spurions)
        fourmomenta = embedding["fourmomenta"]
        scalars = torch.cat([embedding["scalars"], embedding["tagging_features"]], dim=-1)
        batch = embedding["batch"]
        ptr = embedding["ptr"]
        is_spurion = embedding["is_spurion"]

        # rescale fourmomenta (but not the spurions)
        fourmomenta[~is_spurion] = fourmomenta[~is_spurion] / 20

        edge_index = get_edge_index_from_ptr(ptr, fourmomenta.shape, remove_self_loops=True)
        fourmomenta = fourmomenta.to(scalars.dtype)
        output = self.net(scalars, fourmomenta, edges=edge_index, batch=batch, ptr=ptr)
        return output, {}, None


class PELICANWrapper(nn.Module):
    def __init__(
        self,
        net,
        framesnet,
        out_channels,
    ):
        super().__init__()
        self.net = net(out_channels=out_channels)

        self.register_buffer("edge_inited", torch.tensor(False))
        self.register_buffer("edge_mean", torch.tensor(0.0))
        self.register_buffer("edge_std", torch.tensor(1.0))

        self.framesnet = framesnet  # not actually used
        assert isinstance(framesnet, IdentityFrames)

    def forward(self, embedding):
        # extract embedding (includes spurions)
        fourmomenta = embedding["fourmomenta"]
        scalars = torch.cat([embedding["scalars"], embedding["tagging_features"]], dim=-1)
        batch = embedding["batch"]
        ptr = embedding["ptr"]
        is_spurion = embedding["is_spurion"]
        num_graphs = embedding["num_graphs"]

        # rescale fourmomenta (but not the spurions)
        fourmomenta[~is_spurion] = fourmomenta[~is_spurion] / 20

        edge_index = get_edge_index_from_ptr(ptr, fourmomenta.shape, remove_self_loops=False)
        fourmomenta = fourmomenta.to(scalars.dtype)
        edge_attr = self.get_edge_attr(fourmomenta, edge_index).to(scalars.dtype)
        output = self.net(
            in_rank2=edge_attr,
            edge_index=edge_index,
            batch=batch,
            in_rank1=scalars,
            num_graphs=num_graphs,
        )
        return output, {}, None

    def get_edge_attr(self, fourmomenta, edge_index):
        edge_attr = get_edge_attr(fourmomenta, edge_index)
        if not self.edge_inited:
            self.edge_mean = edge_attr.mean().detach()
            self.edge_std = edge_attr.std().clamp(min=1e-5).detach()
            self.edge_inited = torch.tensor(True, device=edge_attr.device)
        edge_attr = (edge_attr - self.edge_mean) / self.edge_std
        return edge_attr.unsqueeze(-1)


class PELICANWrapperOfficial(nn.Module):
    def __init__(self, net, framesnet, out_channels):
        super().__init__()
        self.net = net(out_channels=out_channels)
        self.framesnet = framesnet
        assert isinstance(framesnet, IdentityFrames)

    def forward(self, embedding):
        # extract embedding (includes spurions)
        fourmomenta = embedding["fourmomenta"]
        scalars = torch.cat([embedding["scalars"], embedding["tagging_features"]], dim=-1)
        batch = embedding["batch"]
        is_spurion = embedding["is_spurion"]

        # rescale fourmomenta (but not the spurions)
        fourmomenta[~is_spurion] = fourmomenta[~is_spurion] / 20
        fourmomenta = fourmomenta.to(scalars.dtype)
        fourmomenta, mask = to_dense_batch(fourmomenta, batch)
        scalars, _ = to_dense_batch(scalars, batch)
        mask = mask.unsqueeze(-1)

        output = self.net(scalars, fourmomenta, mask=mask)
        return output, {}, None


class CGENNWrapper(nn.Module):
    def __init__(self, net, framesnet, out_channels, compile=False,
                 activation_memory_budget=None):
        super().__init__()
        self.net = net(n_outputs=out_channels)
        if compile:
            if os.environ.get("CGENN_REGIONAL", "0") == "1":
                # REGIONAL compilation (docs, flash round-trip #6 experiment): compile the
                # plain-nn Sequential submodules (phi_h/theta_h/psi_x/chi_x MLPs) as
                # individual units and keep the orchestration -- and with it the flash
                # custom op -- in eager Python. No joint AOT graph ever contains the
                # opaque op, so none of flash-compiled's partition seams exist, while the
                # scalar MLPs still get their fused kernels. Opt-in per-process
                # (CGENN_REGIONAL=1); whole-net compile stays the shipping default.
                n = _compile_regional(self.net)
                LOGGER.info(f"CGENN regional compile: {n} Sequential submodules compiled")
            else:
                # net only -- the wrapper stays eager: pair.nonzero, the spurion rescale
                # and to_dense_batch are data-dependent by design (docs/cgenn-compile.md
                # section 2). dynamic=True: N = B*P and the fully-connected E vary per
                # batch.
                self.net.compile(dynamic=True)
        self.budget = _activation_memory_budget(activation_memory_budget)
        self.framesnet = framesnet
        assert isinstance(framesnet, IdentityFrames)

    def forward(self, embedding):
        # we mimic the CGENN wrapper of
        # https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks/blob/master/models/lorentz_cggnn.py

        # extract embedding (includes spurions)
        fourmomenta = embedding["fourmomenta"]
        scalars = torch.cat([embedding["scalars"], embedding["tagging_features"]], dim=-1)
        batch = embedding["batch"]
        is_spurion = embedding["is_spurion"]

        # rescale fourmomenta (but not the spurions)
        fourmomenta[~is_spurion] = fourmomenta[~is_spurion] / 20
        fourmomenta = fourmomenta.to(scalars.dtype)
        zeros = torch.zeros(scalars.shape[0], 1, device=scalars.device, dtype=scalars.dtype)
        scalars = torch.cat((scalars, zeros), dim=-1)

        # pad to dense tensors
        fourmomenta, mask = to_dense_batch(fourmomenta, batch)
        scalars, _ = to_dense_batch(scalars, batch)
        batch_size, n_nodes, _ = fourmomenta.shape
        # fully-connected (no self-loop) edges among REAL nodes, in the DENSE b*n_nodes+i
        # frame the flattened tensors use. The old get_edge_index_from_ptr produced SPARSE
        # ptr[b]+i edges, which only match under equal jet lengths; with variable sizes ~1/3
        # crossed jet boundaries, scrambling CGENN messages and leaking across batch-mates.
        pair = mask[:, :, None] & mask[:, None, :]
        pair &= ~torch.eye(n_nodes, dtype=torch.bool, device=mask.device)[None]
        b_idx, i_idx, j_idx = pair.nonzero(as_tuple=True)
        edge_index = torch.stack([b_idx * n_nodes + i_idx, b_idx * n_nodes + j_idx])
        fourmomenta = fourmomenta.view(batch_size * n_nodes, -1)
        scalars = scalars.view(batch_size * n_nodes, -1)
        # dense (B, n_nodes, 1): the net reads its padded dim from this tensor -- symbolic
        # under compile(dynamic=True), where a python-int argument would re-specialize the
        # graph per distinct padded length (docs/cgenn-compile.md, RECOMP gate)
        node_mask = mask.unsqueeze(-1)

        x = fourmomenta.unsqueeze(-2)
        i, j = edge_index
        edge_attr_x = torch.cat(
            [
                x[i],
                x[j],
                x[i] - x[j],
            ],
            dim=-2,
        )
        node_attr_x = x
        x = embed_vector(x)
        edge_attr_x = embed_vector(edge_attr_x)
        node_attr_x = embed_vector(node_attr_x)

        h = scalars
        edge_attr_h = None
        node_attr_h = h

        with self.budget:
            out = self.net(
                h=h,
                x=x,
                edge_attr_x=edge_attr_x,
                node_attr_x=node_attr_x,
                edge_attr_h=edge_attr_h,
                node_attr_h=node_attr_h,
                edges=edge_index,
                node_mask=node_mask,
                # FLASH-3 step 2: this builder is fully connected over the DENSE
                # padded layout, so receiver degree <= n_nodes - 1 -- a python int
                # already in hand (no host read). The net turns it into the padded
                # segment-sum bound. NOTE (measured trade, docs step 2): on the FC
                # graph the padded buffer is (n_nodes-1)/avg_degree larger than the
                # edge tensor; CGENN_FC_PADDED=0 keeps segment_reduce (unsafe=True,
                # sync-free) on this one net for the A/B.
                knn_k=(n_nodes - 1) if _FC_PADDED else None,
            )

        return out, {}, None


class LGATrSlimWrapper(nn.Module):
    def __init__(
        self,
        net,
        framesnet,
        out_channels,
        mean_aggregation=False,
        attention_backend="xformers",
        use_amp=False,
    ):
        super().__init__()
        self.use_amp = use_amp
        self.attention_backend = attention_backend
        self.net = net(out_s_channels=out_channels)
        self.aggregator = MeanAggregation() if mean_aggregation else None
        self.framesnet = framesnet  # not actually used
        assert isinstance(framesnet, IdentityFrames)

        if attention_backend == "flex":
            compile_flex_attention(package_name="lgatr")

    def forward(self, embedding):
        # extract embedding (includes spurions)
        fourmomenta = embedding["fourmomenta"]
        scalars = torch.cat([embedding["scalars"], embedding["tagging_features"]], dim=-1)
        batch = embedding["batch"]
        ptr = embedding["ptr"]
        is_spurion = embedding["is_spurion"]

        # rescale fourmomenta (but not the spurions)
        fourmomenta[~is_spurion] = fourmomenta[~is_spurion] / 20

        # handle global token
        if self.aggregator is None:
            batchsize = len(ptr) - 1
            global_idxs = ptr[:-1] + torch.arange(batchsize, device=batch.device)
            is_global = torch.zeros(
                fourmomenta.shape[0] + batchsize,
                dtype=torch.bool,
                device=ptr.device,
            )
            is_global[global_idxs] = True
            fourmomenta_buffer = fourmomenta.clone()
            fourmomenta = torch.zeros(
                is_global.shape[0],
                *fourmomenta.shape[1:],
                dtype=fourmomenta.dtype,
                device=fourmomenta.device,
            )
            fourmomenta[~is_global] = fourmomenta_buffer
            scalars_buffer = scalars.clone()
            scalars = torch.zeros(
                fourmomenta.shape[0],
                scalars.shape[1] + 1,
                dtype=scalars.dtype,
                device=scalars.device,
            )
            token_idx = torch.nn.functional.one_hot(torch.arange(1, device=scalars.device))
            token_idx = token_idx.repeat(batchsize, 1)
            scalars[~is_global] = torch.cat(
                (
                    scalars_buffer,
                    torch.zeros(
                        scalars_buffer.shape[0],
                        token_idx.shape[1],
                        dtype=scalars.dtype,
                        device=scalars.device,
                    ),
                ),
                dim=-1,
            )
            scalars[is_global] = torch.cat(
                (
                    torch.zeros(
                        token_idx.shape[0],
                        scalars_buffer.shape[1],
                        dtype=scalars.dtype,
                        device=scalars.device,
                    ),
                    token_idx,
                ),
                dim=-1,
            )
            ptr[1:] = ptr[1:] + (torch.arange(batchsize, device=ptr.device) + 1)
            batch = get_batch_from_ptr(ptr)
        else:
            is_global = None

        fourmomenta = fourmomenta.unsqueeze(0).to(scalars.dtype)
        scalars = scalars.unsqueeze(0)

        mask_kwarg = get_attention_mask(
            batch,
            dtype=fourmomenta.dtype,
            attention_backend=self.attention_backend,
        )

        v = fourmomenta.unsqueeze(-2)
        s = scalars

        with torch.autocast(v.device.type, enabled=self.use_amp):
            _, out_s = self.net(v, s, **mask_kwarg)
        out = out_s[0, :, :]

        if self.aggregator is not None:
            logits = self.aggregator(out, index=batch)
        else:
            logits = out[is_global]
        return logits, {}, None


class CGENNLGATrGraphTransWrapper(nn.Module):
    def __init__(self, net, framesnet, out_channels, compile=False,
                 activation_memory_budget=None):
        super().__init__()
        self.net = net(num_classes=out_channels)
        if compile:
            # compile the net only; edge building is hoisted in forward (data-dependent
            # nonzero, eager by design -- docs/cgenn-compile.md, Stage 3)
            self.net.compile(dynamic=True)
        self.budget = _activation_memory_budget(activation_memory_budget)
        self.framesnet = framesnet  # not actually used
        assert isinstance(framesnet, IdentityFrames)

    def forward(self, embedding):
        fourmomenta = embedding["fourmomenta"]  # (E, px, py, pz), incl. spurions
        scalars = torch.cat([embedding["scalars"], embedding["tagging_features"]], dim=-1)
        batch = embedding["batch"]
        is_spurion = embedding["is_spurion"]
        keep = ~is_spurion  # channel-spurions in model: drop the tokens
        fourmomenta = fourmomenta[keep]
        scalars = scalars[keep]
        batch = batch[keep]
        fourmomenta = (fourmomenta / 20).to(
            scalars.dtype
        )  # match the equivariant baselines; NO reorder
        px, py, pz = fourmomenta[:, 1], fourmomenta[:, 2], fourmomenta[:, 3]  # (E, px, py, pz)
        pt = torch.sqrt(px * px + py * py).clamp(min=1e-8)
        points = torch.stack([torch.asinh(pz / pt), torch.atan2(py, px)], dim=-1)
        fourmomenta, mask = to_dense_batch(fourmomenta, batch)
        scalars, _ = to_dense_batch(scalars, batch)
        points, _ = to_dense_batch(points, batch)
        # hoist the static kNN edges out of the (possibly compiled) net: identical values
        # in identical order eager -- the edges depend only on these inputs
        edges = self.net.build_edges(fourmomenta, mask, points)
        with self.budget:
            output = self.net(
                scalars,
                fourmomenta,
                mask,
                points,
                edges=edges,
            )
        return output, {}, None


class CGENNLGATrGraphGPSWrapper(nn.Module):
    """Wrapper for the equivariant CGENN-L-GATr GraphGPS hybrid.

    Equivariant by construction (CGENN + L-GATr, symmetry broken only by the model's
    own input spurions), so no LLoCa canonicalization: inherits nn.Module with the
    identity framesnet, exactly like CGENNLGATrGraphTransWrapper. Drops the token
    spurions (the model injects its own as mv channels), rescales by 1/20, and keeps
    the time-first (E, px, py, pz) convention (no reorder).
    """

    def __init__(self, net, framesnet, out_channels, compile=False,
                 activation_memory_budget=None):
        super().__init__()
        self.net = net(num_classes=out_channels)
        if compile:
            # compile the net only; edge building is hoisted in forward (data-dependent
            # nonzero, eager by design -- docs/cgenn-compile.md, Stage 3)
            # FIX for the H100 stride-guard crash (2026-08-14): the first multi-batch
            # compiled TRAINING profile died in this model's compiled backward on
            # inductor's runtime stride guard --
            #   assert_size_stride(permute_134, (s27*s77, 16, 1, 1), (1, 7168, 7168, 1))
            #   AssertionError: expected size 16==16, stride 6656==7168 at dim=1
            # -- a saved permuted multivector VIEW whose stride was specialized to one
            # batch's width while its sizes stayed symbolic: the LNetSlimGraphGPS
            # view-saving class (an upstream inductor defect family; that one was proven
            # by monkeypatch, see docs/cgenn-compile.md). Same scoped recompute_views
            # patch, numerics-preserving by construction. VERIFIED on the H100 same day:
            # 16 compiled steps (8 warm + 8 profiled) over varying mini shapes, crash
            # gone, profile clean. A longer soak before the first long run is cheap
            # insurance: CGENN_SMOKE_STEPS=100 CGENN_SMOKE_COMPILE=1 on the smoke gate.
            # Retirement status (2026-08-15 audit, docs/cgenn-compile.md): the Phase-1
            # rewrites cut the saved permuted-view population 71 -> 45, of which only 4
            # are activation-shaped-symbolic (all in sparse_gp's einsum; respelling it
            # does NOT clear them -- measured). So retirement is decided by the gate-day
            # SHIELD-OFF soak, not structurally: CGENN_RECOMPUTE_VIEWS_SHIELD=0 runs
            # this model (and the LNetSlim-GPS twin) unshielded. Default stays ON.
            self._recompute_views = os.environ.get(
                "CGENN_RECOMPUTE_VIEWS_SHIELD", "1") != "0"
            self.net.compile(dynamic=True)
        self.budget = _activation_memory_budget(activation_memory_budget)
        self.framesnet = framesnet  # not actually used
        assert isinstance(framesnet, IdentityFrames)

    def forward(self, embedding):
        fourmomenta = embedding["fourmomenta"]  # (E, px, py, pz), incl. spurions
        scalars = torch.cat([embedding["scalars"], embedding["tagging_features"]], dim=-1)
        batch = embedding["batch"]
        is_spurion = embedding["is_spurion"]
        keep = ~is_spurion  # channel-spurions in model: drop the tokens
        fourmomenta = fourmomenta[keep]
        scalars = scalars[keep]
        batch = batch[keep]
        fourmomenta = (fourmomenta / 20).to(
            scalars.dtype
        )  # match the equivariant baselines; NO reorder
        px, py, pz = fourmomenta[:, 1], fourmomenta[:, 2], fourmomenta[:, 3]  # (E, px, py, pz)
        pt = torch.sqrt(px * px + py * py).clamp(min=1e-8)
        points = torch.stack([torch.asinh(pz / pt), torch.atan2(py, px)], dim=-1)
        fourmomenta, mask = to_dense_batch(fourmomenta, batch)
        scalars, _ = to_dense_batch(scalars, batch)
        points, _ = to_dense_batch(points, batch)
        # hoist the static kNN edges out of the (possibly compiled) net: identical values
        # in identical order eager -- the edges depend only on these inputs
        edges = self.net.build_edges(fourmomenta, mask, points)
        if getattr(self, "_recompute_views", False):
            # see __init__: candidate fix for the H100 stride-guard crash. Scoped per
            # call like LorentzNetLGATrSlimGraphGPSWrapper's -- the smoke test and the
            # FLOPs harness build many models in one process, so never set it globally.
            import torch._functorch.config as _fc
            with _fc.patch(recompute_views=True), self.budget:
                return self.net(scalars, fourmomenta, mask, points, edges=edges), {}, None
        with self.budget:
            output = self.net(
                scalars,
                fourmomenta,
                mask,
                points,
                edges=edges,
            )
        return output, {}, None


class ParticleNetParTGraphTransWrapper(TaggerWrapper):
    """Wrapper for the ParticleNet-ParT graph-transformer hybrid.

    Non-equivariant, made Lorentz-equivariant by LLoCa tensorial message-passing
    (matching the library), exactly like the other GraphTrans/GPS wrappers: the inputs
    are canonicalized and the per-particle frames are passed into the backbone, which
    transports the EdgeConv neighbours and the attention q/k/v between frames. With
    IdentityFrames the transport is a no-op and this reduces to the plain baseline in the
    global frame; any learned framesnet is supported through the shared TaggerWrapper
    machinery.

    The backbone differs from the rest of the repo only in its conventions: it is
    channels-first (N, C, P), expects four-momenta as (px, py, pz, E) rather than
    (E, px, py, pz), and takes a (N, 1, P) mask. It additionally needs (eta, phi)
    points for the EdgeConv kNN, which we read off the local four-momenta.
    """

    def __init__(self, net, *args, use_amp=False, compile=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_amp = use_amp
        self.net = net(input_dim=self.in_channels, num_classes=self.out_channels, use_amp=use_amp)
        # the prepended class token rides in the covariant jet frame -> request it
        self.compute_jet_frames = True
        if compile:
            # route the identity-frames nn.MHA blocks and the tril PairEmbed through
            # their compiled twins (eager default untouched, TOL-gated -- recorded:
            # docs/cgenn-compile.md, Stage 4)
            for m in self.net.modules():
                if hasattr(m, "compiled_attention"):
                    m.compiled_attention = True
                if hasattr(m, "compiled_dense"):
                    m.compiled_dense = True
            self.net.compile(dynamic=True)

    def forward(self, embedding):
        (
            features_local,
            fourmomenta_local,
            frames,
            _,
            batch,
            tracker,
        ) = super().forward(embedding)
        fourmomenta_local = fourmomenta_local.to(features_local.dtype)
        fourmomenta_local = fourmomenta_local[..., [1, 2, 3, 0]]  # need (px, py, pz, E)

        # (eta, phi) points for the EdgeConv kNN, read off the local four-momenta
        px, py, pz = (
            fourmomenta_local[..., 0],
            fourmomenta_local[..., 1],
            fourmomenta_local[..., 2],
        )
        pt = torch.sqrt(px * px + py * py).clamp(min=1e-8)
        points = torch.stack([torch.asinh(pz / pt), torch.atan2(py, px)], dim=-1)

        features_local, mask = to_dense_batch(features_local, batch)
        fourmomenta_local, _ = to_dense_batch(fourmomenta_local, batch)
        points, _ = to_dense_batch(points, batch)
        features_local = features_local.transpose(1, 2).contiguous()  # (B, C, P)
        fourmomenta_local = fourmomenta_local.transpose(1, 2).contiguous()  # (B, 4, P)
        points = points.transpose(1, 2).contiguous()  # (B, 2, P)

        # densify per-particle local frames to (B, P, 4, 4); padded particles -> identity
        # (masked out anyway). The tensorial backbone transports q-k-v between them (LLoCa).
        frames_matrices, _ = to_dense_batch(frames.matrices, batch)
        frames_matrices[~mask] = lorentz_eye(
            frames_matrices[~mask].shape[:-2], device=frames.device, dtype=frames.dtype
        )
        dense_frames = Frames(
            matrices=frames_matrices,
            is_global=frames.is_global,
            is_identity=frames.is_identity,
        )
        mask = mask.unsqueeze(1).float()  # (B, 1, P)

        # the backbone handles AMP internally via use_amp. cls_frames is the covariant
        # jet frame for the prepended class token (None for IdentityFrames -> identity slot).
        score = self.net(
            points=points,
            features=features_local,
            v=fourmomenta_local,
            frames=dense_frames,
            mask=mask,
            cls_frames=self._jet_frames,
        )
        return score, tracker, frames


class LorentzNetLGATrSlimGraphTransWrapper(nn.Module):
    """Wrapper for the internally-equivariant LorentzNet -> L-GATr-slim hybrid.

    Like CGENNLGATrGraphTransWrapper, the backbone is Lorentz-equivariant by
    construction (LorentzNet GNN + L-GATr-slim, with symmetry broken only by its
    own input-stage spurions), so no LLoCa canonicalization is applied and the
    framesnet must be the identity -- hence we inherit nn.Module directly.

    The backbone differs from the rest of the repo only in conventions: it is
    channels-first (N, C, P), expects four-momenta as (px, py, pz, E) rather than
    (E, px, py, pz), takes a (N, 1, P) mask, and uses (eta, phi) points only when
    knn_metric='deltaR'.
    """

    def __init__(self, net, framesnet, out_channels, compile=False):
        super().__init__()
        self.net = net(num_classes=out_channels)
        if compile:
            # compile the net only (dense top-k kNN inside the net is shape-static and
            # traces clean -- docs/cgenn-compile.md, Stage 2)
            self.net.compile(dynamic=True)
        self.framesnet = framesnet  # not actually used
        assert isinstance(framesnet, IdentityFrames)

    def forward(self, embedding):
        fourmomenta = embedding["fourmomenta"]  # (E, px, py, pz), incl. spurions
        scalars = torch.cat([embedding["scalars"], embedding["tagging_features"]], dim=-1)
        batch = embedding["batch"]
        is_spurion = embedding["is_spurion"]

        # the model injects its own input-stage spurions: drop the token spurions
        keep = ~is_spurion
        fourmomenta = fourmomenta[keep]
        scalars = scalars[keep]
        batch = batch[keep]

        # match the scale of the other equivariant baselines
        fourmomenta = (fourmomenta / 20).to(scalars.dtype)

        # (eta, phi) points for the deltaR kNN option (ignored for minkowski)
        px, py, pz = fourmomenta[:, 1], fourmomenta[:, 2], fourmomenta[:, 3]
        pt = torch.sqrt(px * px + py * py).clamp(min=1e-8)
        points = torch.stack([torch.asinh(pz / pt), torch.atan2(py, px)], dim=-1)

        # the model expects four-momenta as (px, py, pz, E)
        fourmomenta = fourmomenta[:, [1, 2, 3, 0]]

        # densify and switch to the (N, C, P) channels-first convention
        fourmomenta, mask = to_dense_batch(fourmomenta, batch)  # (B, P, 4), (B, P)
        scalars, _ = to_dense_batch(scalars, batch)  # (B, P, C)
        points, _ = to_dense_batch(points, batch)  # (B, P, 2)

        output = self.net(
            scalars.transpose(1, 2).contiguous(),  # x: (B, C, P)
            fourmomenta.transpose(1, 2).contiguous(),  # v: (B, 4, P)
            mask.unsqueeze(1),  # (B, 1, P)
            points.transpose(1, 2).contiguous(),  # (B, 2, P)
        )
        return output, {}, None


class LorentzNetLGATrSlimGraphGPSWrapper(nn.Module):
    """Wrapper for the equivariant LorentzNet-L-GATr-slim GraphGPS hybrid.

    Equivariant by construction (broken only by the model's own input spurions),
    so no LLoCa canonicalization: inherits nn.Module + identity framesnet, exactly
    like LorentzNetLGATrSlimGraphTransWrapper. Drops the token spurions, rescales
    by 1/20, and reorders four-momenta to the (px, py, pz, E) the model expects.
    """

    def __init__(self, net, framesnet, out_channels, compile=False):
        super().__init__()
        self.net = net(num_classes=out_channels)
        if compile:
            # compile the net only (dense top-k kNN inside the net is shape-static and
            # traces clean -- docs/cgenn-compile.md, Stage 2)
            # see forward(): AOT view-saving vs inductor. Same gate-day retirement knob
            # as the CGENN-GPS twin: CGENN_RECOMPUTE_VIEWS_SHIELD=0 soaks unshielded.
            self._recompute_views = os.environ.get(
                "CGENN_RECOMPUTE_VIEWS_SHIELD", "1") != "0"
            self.net.compile(dynamic=True)
        self.framesnet = framesnet  # not actually used
        assert isinstance(framesnet, IdentityFrames)

    def forward(self, embedding):
        fourmomenta = embedding["fourmomenta"]  # (E, px, py, pz), incl. spurions
        scalars = torch.cat([embedding["scalars"], embedding["tagging_features"]], dim=-1)
        batch = embedding["batch"]
        is_spurion = embedding["is_spurion"]

        # the model injects its own input-stage spurions: drop the token spurions
        keep = ~is_spurion
        fourmomenta = fourmomenta[keep]
        scalars = scalars[keep]
        batch = batch[keep]

        # match the scale of the other equivariant baselines
        fourmomenta = (fourmomenta / 20).to(scalars.dtype)

        # (eta, phi) points for the deltaR kNN option (ignored for minkowski)
        px, py, pz = fourmomenta[:, 1], fourmomenta[:, 2], fourmomenta[:, 3]
        pt = torch.sqrt(px * px + py * py).clamp(min=1e-8)
        points = torch.stack([torch.asinh(pz / pt), torch.atan2(py, px)], dim=-1)

        # the model expects four-momenta as (px, py, pz, E)
        fourmomenta = fourmomenta[:, [1, 2, 3, 0]]

        # densify and switch to the (N, C, P) channels-first convention
        fourmomenta, mask = to_dense_batch(fourmomenta, batch)  # (B, P, 4), (B, P)
        scalars, _ = to_dense_batch(scalars, batch)  # (B, P, C)
        points, _ = to_dense_batch(points, batch)  # (B, P, 2)

        _args = (
            scalars.transpose(1, 2).contiguous(),  # x: (B, C, P)
            fourmomenta.transpose(1, 2).contiguous(),  # v: (B, 4, P)
            mask.unsqueeze(1),  # (B, 1, P)
            points.transpose(1, 2).contiguous(),  # (B, 2, P)
        )
        if getattr(self, "_recompute_views", False):
            # AOT's partitioner saves VIEWS across the fwd/bwd boundary because they are
            # cheap. Here that saved view is this model's per-layer channel-first <->
            # channel-last multivector transpose, shape (B, P, 4, V) with strides
            # (16*P, 16, 1, 4), and inductor's stride-ordering for graph outputs that are
            # views compares 16*P against 16 -- i.e. asks "is P > 1?" -- via a code path
            # that is not given the ShapeEnv, so a symbolic P raises
            # "cannot determine truth value of Relational: 1 < s53" while lowering the
            # JOINT graph. recompute_views=True makes the backward recompute the view
            # instead of saving it, so it is no longer a graph output and the path is
            # never entered. Numerics-preserving by construction (a recomputed permute is
            # the same permute); measured identical to the eager forward.
            # Scoped to this call rather than set globally, because the smoke test and the
            # FLOPs harness build many models in one process.
            import torch._functorch.config as _fc
            with _fc.patch(recompute_views=True):
                return self.net(*_args), {}, None
        output = self.net(*_args)
        return output, {}, None


class PlainGraphTransWrapper(TaggerWrapper):
    """Wrapper for the plain graph-transformer (static MPNN + torch-MHA encoder).

    Non-equivariant, made Lorentz-equivariant by LLoCa tensorial message-passing
    (matching the library), exactly like ParticleNetParTGraphTransWrapper: the inputs
    are canonicalized and the per-particle frames are passed into the backbone, which
    transports the MPNN neighbours and the attention q/k/v between frames. Channels-first
    (N, C, P), four-momenta as (px, py, pz, E), a (N, 1, P) mask, and (eta, phi) points
    for the deltaR kNN. The prepended class token rides in the covariant jet frame.
    """

    def __init__(self, net, *args, use_amp=False, compile=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_amp = use_amp
        self.net = net(input_dim=self.in_channels, num_classes=self.out_channels, use_amp=use_amp)
        # the prepended class token rides in the covariant jet frame -> request it
        self.compute_jet_frames = True
        if compile:
            # compile the net only; static kNN + torch-MHA trace clean (recorded:
            # docs/cgenn-compile.md, Stage 4).
            # compiled_knn: keep the kNN k a python int -- the eager cap against a symbolic
            # P is what inductor cannot lower on the BACKWARD graph (see knn()'s docstring).
            # Provably identical wherever P - 1 >= k, which is every regime we run.
            self.net.compiled_knn = True
            self.net.compile(dynamic=True)

    def forward(self, embedding):
        (
            features_local,
            fourmomenta_local,
            frames,
            _,
            batch,
            tracker,
        ) = super().forward(embedding)
        fourmomenta_local = fourmomenta_local.to(features_local.dtype)
        fourmomenta_local = fourmomenta_local[..., [1, 2, 3, 0]]  # need (px, py, pz, E)

        px, py, pz = (
            fourmomenta_local[..., 0],
            fourmomenta_local[..., 1],
            fourmomenta_local[..., 2],
        )
        pt = torch.sqrt(px * px + py * py).clamp(min=1e-8)
        points = torch.stack([torch.asinh(pz / pt), torch.atan2(py, px)], dim=-1)

        features_local, mask = to_dense_batch(features_local, batch)
        fourmomenta_local, _ = to_dense_batch(fourmomenta_local, batch)
        points, _ = to_dense_batch(points, batch)

        # densify per-particle local frames to (B, P, 4, 4); padded particles -> identity.
        # The backbone transports q-k-v between them (no-op for IdentityFrames).
        frames_matrices, _ = to_dense_batch(frames.matrices, batch)
        frames_matrices[~mask] = lorentz_eye(
            frames_matrices[~mask].shape[:-2], device=frames.device, dtype=frames.dtype
        )
        dense_frames = Frames(
            matrices=frames_matrices,
            is_global=frames.is_global,
            is_identity=frames.is_identity,
        )

        score = self.net(
            points=points.transpose(1, 2).contiguous(),  # (B, 2, P)
            features=features_local.transpose(1, 2).contiguous(),  # (B, C, P)
            v=fourmomenta_local.transpose(1, 2).contiguous(),  # (B, 4, P)
            mask=mask.unsqueeze(1).float(),  # (B, 1, P)
            frames=dense_frames,
            cls_frames=self._jet_frames,
        )
        return score, tracker, frames


class PlainGraphGPSWrapper(TaggerWrapper):
    """Wrapper for the plain GraphGPS hybrid (interleaved static-MPNN + torch-MHA).

    Non-equivariant, made Lorentz-equivariant by LLoCa tensorial message-passing
    (matching the library): the inputs are canonicalized and the per-particle frames are
    passed into the backbone, which transports the local-MPNN neighbours and the attention
    q/k/v between frames. Channels-first (N, C, P), four-momenta as (px, py, pz, E), a
    (N, 1, P) mask, and (eta, phi) points for the deltaR kNN. No jet frame is needed -- the
    mean-pool readout over invariant local features is already invariant.
    """

    def __init__(self, net, *args, use_amp=False, compile=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_amp = use_amp
        self.net = net(input_dim=self.in_channels, num_classes=self.out_channels, use_amp=use_amp)
        if compile:
            # compile the net only. KNOWN SPLITS: the masked BatchNorm over real nodes
            # (MaskedNorm, norm='batch') is data-dependent by design -> a pinned number
            # of documented graph splits, not a clean single graph (recorded:
            # docs/cgenn-compile.md, Stage 4)
            self.net.compile(dynamic=True)

    def forward(self, embedding):
        (
            features_local,
            fourmomenta_local,
            frames,
            _,
            batch,
            tracker,
        ) = super().forward(embedding)
        fourmomenta_local = fourmomenta_local.to(features_local.dtype)
        fourmomenta_local = fourmomenta_local[..., [1, 2, 3, 0]]  # need (px, py, pz, E)

        px, py, pz = (
            fourmomenta_local[..., 0],
            fourmomenta_local[..., 1],
            fourmomenta_local[..., 2],
        )
        pt = torch.sqrt(px * px + py * py).clamp(min=1e-8)
        points = torch.stack([torch.asinh(pz / pt), torch.atan2(py, px)], dim=-1)

        features_local, mask = to_dense_batch(features_local, batch)
        fourmomenta_local, _ = to_dense_batch(fourmomenta_local, batch)
        points, _ = to_dense_batch(points, batch)

        # densify per-particle local frames to (B, P, 4, 4); padded particles -> identity.
        # The backbone transports q-k-v between them (no-op for IdentityFrames).
        frames_matrices, _ = to_dense_batch(frames.matrices, batch)
        frames_matrices[~mask] = lorentz_eye(
            frames_matrices[~mask].shape[:-2], device=frames.device, dtype=frames.dtype
        )
        dense_frames = Frames(
            matrices=frames_matrices,
            is_global=frames.is_global,
            is_identity=frames.is_identity,
        )

        score = self.net(
            points=points.transpose(1, 2).contiguous(),  # (B, 2, P)
            features=features_local.transpose(1, 2).contiguous(),  # (B, C, P)
            v=fourmomenta_local.transpose(1, 2).contiguous(),  # (B, 4, P)
            frames=dense_frames,
            mask=mask.unsqueeze(1).float(),  # (B, 1, P)
        )
        return score, tracker, frames


class ParticleNetParTGraphGPSWrapper(TaggerWrapper):
    """Wrapper for the ParticleNet-ParT GraphGPS hybrid (tensorial EdgeConv + ParT attn).

    Lorentz-equivariant by LLoCa tensorial message-passing (matching the library): the
    inputs are canonicalized and the per-particle frames are passed into the backbone,
    which transports neighbours (EdgeConv) and q/k/v (attention) between frames. Like the
    GraphTrans wrapper it is channels-first (N, C, P), four-momenta as (px, py, pz, E), a
    (N, 1, P) mask, and (eta, phi) points seeding the layer-0 deltaR kNN. No jet frame is
    needed -- the mean-pool readout over invariant local features is already invariant.
    """

    def __init__(self, net, *args, use_amp=False, compile=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_amp = use_amp
        self.net = net(input_dim=self.in_channels, num_classes=self.out_channels, use_amp=use_amp)
        if compile:
            # route the identity-frames nn.MHA calls + the tril PairEmbed through their
            # compiled twins (eager default untouched, TOL-gated). KNOWN SPLITS: the
            # masked BatchNorm over real nodes (MaskedNorm, norm='batch') is
            # data-dependent by design -> a pinned number of documented graph splits
            # (recorded: docs/cgenn-compile.md, Stage 4)
            for m in self.net.modules():
                if hasattr(m, "compiled_attention"):
                    m.compiled_attention = True
                if hasattr(m, "compiled_dense"):
                    m.compiled_dense = True
            self.net.compile(dynamic=True)

    def forward(self, embedding):
        (
            features_local,
            fourmomenta_local,
            frames,
            _,
            batch,
            tracker,
        ) = super().forward(embedding)
        fourmomenta_local = fourmomenta_local.to(features_local.dtype)
        fourmomenta_local = fourmomenta_local[..., [1, 2, 3, 0]]  # need (px, py, pz, E)

        px, py, pz = (
            fourmomenta_local[..., 0],
            fourmomenta_local[..., 1],
            fourmomenta_local[..., 2],
        )
        pt = torch.sqrt(px * px + py * py).clamp(min=1e-8)
        points = torch.stack([torch.asinh(pz / pt), torch.atan2(py, px)], dim=-1)

        features_local, mask = to_dense_batch(features_local, batch)
        fourmomenta_local, _ = to_dense_batch(fourmomenta_local, batch)
        points, _ = to_dense_batch(points, batch)

        # densify the per-particle local frames to (B, P, 4, 4); padded particles -> identity
        frames_matrices, _ = to_dense_batch(frames.matrices, batch)
        frames_matrices[~mask] = lorentz_eye(
            frames_matrices[~mask].shape[:-2], device=frames.device, dtype=frames.dtype
        )
        dense_frames = Frames(
            matrices=frames_matrices,
            is_global=frames.is_global,
            is_identity=frames.is_identity,
        )

        score = self.net(
            points=points.transpose(1, 2).contiguous(),  # (B, 2, P)
            features=features_local.transpose(1, 2).contiguous(),  # (B, C, P)
            v=fourmomenta_local.transpose(1, 2).contiguous(),  # (B, 4, P)
            frames=dense_frames,
            mask=mask.unsqueeze(1).float(),  # (B, 1, P)
        )
        return score, tracker, frames


def compile_flex_attention(package_name="lgatr"):
    """Run torch.compile on the flex_attention function.

    However, as of today (Dec 2025, pytorch 2.9.0), torch.compile + flex_attention
    for variable-length sequences only works in a few cases:
    - CPU: Forward pass is supported, but backward pass not (https://github.com/pytorch/pytorch/issues/169224)
      To still let the code run through for tests, we skip torch.compile on CPU.
      This way the code runs through, but is super slow because it materializes the attention matrix.
      Note that we use essentially the same approach for xformers, where we fall back to default torch attention on CPU.
      On the plus side, flex_attention supports arbitrary head_dim if torch.compile is not used.
    - GPU: The docs say that only head dimensions being powers of 2 are supported.
      However, on my system only head_dim=2**n with n>=4 works, i.e. head_dim=16,32,...
      Setting head_dim=2,4,8 gives cryptic errors.
      Moreover, transformers with flex_attention are still significantly slower than
      transformers with xformers attention in our implementation.
    """
    if package_name == "lgatr":
        import lgatr.primitives.attention_backends.flex as flex
    elif package_name == "lloca":
        import lloca.backbone.attention_backends.flex as flex
    else:
        raise ValueError(f"Unknown package {package_name}")

    if torch.cuda.is_available():
        # max-autotune strongly recommended for flex-attention with variable-length sequences,
        # see https://pytorch.org/blog/flexattention-for-inference/
        flex.attention = torch.compile(
            flex.attention,
            dynamic=True,
        )
