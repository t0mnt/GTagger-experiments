"""lloca 2.0 ``preserve_variance`` plumbing: the reference momentum actually reaches
``LLoCaAttention``, the rescaling is live, readout tokens sit in a covariant frame, and the
score stays Lorentz invariant with it on.

Three things this pins that the invariance suites alone would not:
- liveness: ``prepare_frames`` receives ``p_ref`` (a wrapper that forgot to pass it would
  raise on lloca 2.0, but a config that silently swallowed ``preserve_variance`` into
  ``**kwargs`` would not) and the score differs from the ``preserve_variance=False`` run;
- gamma_i = (L_i p_ref)^0 / m_ref >= 1 for every token, and ~1 for the class/global token,
  which sits in the jet rest frame (an identity frame there would carry the lab-frame
  E_jet/m_jet and break invariance under boosts);
- Lorentz invariance of the score with the rescaling on (learnedpd frames, float64).
"""

import pytest
import torch
from lloca.backbone.attention import LLoCaAttention

from tests.experiments.test_tag_equivariance import _build
from tests.helpers.equivariance import check_tagging_invariance

FRAME_OVERRIDES = [
    "model/framesnet=learnedpd",
    "use_float64=true",
    "data.tagging_features=null",
    "data.beam_reference=null",
    "data.add_time_reference=false",
]

# (model, extra overrides, index of the class/global token along the token axis or None)
MODELS = [
    ("tag_PlainGraphTrans", [], 0),
    ("tag_ParticleNetParTGraphTrans", [], 0),
    ("tag_PlainGraphGPS", [], None),
    ("tag_ParticleNetParTGraphGPS", ["model.net.knn_k=9999"], None),
    ("tag_ParT", [], None),
    ("tag_transformer", [], None),
    ("tag_transformer", ["model.mean_aggregation=false"], "global"),
]


@pytest.fixture
def gamma_recorder(monkeypatch):
    """Record every gamma that LLoCaAttention computes (None entries mean it was skipped)."""
    calls = []
    orig = LLoCaAttention._compute_gamma

    def spy(self, frames, p_ref, ptr=None):
        gamma = orig(self, frames, p_ref, ptr=ptr)
        calls.append((gamma.detach().clone(), None if ptr is None else ptr.detach().clone()))
        return gamma

    monkeypatch.setattr(LLoCaAttention, "_compute_gamma", spy)
    return calls


@pytest.mark.parametrize("model,extra,readout", MODELS)
def test_preserve_variance_is_live_and_invariant(model, extra, readout, gamma_recorder):
    exp = _build([f"model={model}", *FRAME_OVERRIDES, *extra])
    data = next(iter(exp.train_loader))
    exp.model.eval()
    with torch.no_grad():
        y_on = exp._get_ypred_and_label(data.clone())[0]

    # 1. liveness: gamma was computed from a real p_ref, once per forward
    assert gamma_recorder, f"{model}: LLoCaAttention never computed gamma (p_ref not plumbed?)"
    gamma, ptr = gamma_recorder[-1]
    assert torch.isfinite(gamma).all()
    assert gamma.min() > 1 - 1e-6, f"{model}: gamma < 1 (got {gamma.min().item():.4f})"
    # boosted top jets: the per-particle rest frames see the jet with E >> m
    assert gamma.max() > 1.05, f"{model}: rescaling inert (max gamma {gamma.max().item():.4f})"

    # 2. the readout token lives in the jet rest frame -> gamma ~ 1 there (mass_regularize
    #    and variance_eps move it by O(1e-3) at most for top jets)
    if readout == 0:
        g_cls = gamma[..., 0]
        assert (g_cls - 1).abs().max() < 1e-2, f"{model}: class token gamma {g_cls}"
    elif readout == "global":
        # packed layout: the global token is the first token of every jet segment
        g_glob = gamma.reshape(-1)[ptr[:-1]]
        assert (g_glob - 1).abs().max() < 1e-2, f"{model}: global token gamma {g_glob}"

    # 3. the flag changes the score (plumbing is not a no-op)
    exp_off = _build([f"model={model}", *FRAME_OVERRIDES, *extra, "model.net.preserve_variance=false"])
    exp_off.model.load_state_dict(exp.model.state_dict())
    exp_off.model.eval()
    with torch.no_grad():
        y_off = exp_off._get_ypred_and_label(data.clone())[0]
    assert (y_on - y_off).abs().max() > 1e-6, f"{model}: preserve_variance has no effect"

    # 4. Lorentz invariance with the rescaling on (learnedpd float64 tolerance, as in
    #    test_lloca_frame_invariance)
    max_dev = check_tagging_invariance(
        exp, data, transform="lorentz", num_checks=5, rtol=1e-2, atol=1e-2
    )
    assert max_dev < 2e-2, f"{model}: not Lorentz invariant with preserve_variance (max dev {max_dev:.2e})"
