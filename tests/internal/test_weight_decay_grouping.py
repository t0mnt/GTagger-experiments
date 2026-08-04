"""Weight-decay grouping must not depend on which code path a model happens to take.

There are two groupings: the base one in `BaseExperiment._init_optimizer` and the
ParT/weaver one in `TaggingExperiment._init_optimizer`, which applies to a hardcoded
list of class-token models. Whether a given model lands in that list is an architectural
detail (does it use a CLS token?), NOT a statement about how it should be regularized --
so the two paths must agree on every parameter class except the CLS token itself.

They did not. The ParT path exempts anything named `*.bias`; base exempted only
`ndim <= 1`. CGENN's `MVLinear.bias` is `(1, C, 1)`, so the *same* hybrid family was
regularized differently in its GraphTrans variant (ParT path, exempt) and its GraphGPS
variant (base path, decayed) -- an asymmetry across the study's primary comparison axis,
and invisible because both models train fine either way.
"""

import pytest
import torch
from hydra import compose, initialize
from hydra.utils import instantiate

# the class names TaggingExperiment._init_optimizer routes through the ParT grouping
PART_PATH = {
    "ParticleTransformer",
    "MIParticleTransformer",
    "ParticleNetParTGraphTrans",
    "LorentzNetLGATrSlimGraphTrans",
    "CGENNLGATrGraphTrans",
    "PlainGraphTrans",
}
MODELS = [
    "tag_cgenn",
    "tag_CGENNLGATrGraphTrans",
    "tag_CGENNLGATrGraphGPS",
    "tag_LorentzNetLGATrSlimGraphTrans",
    "tag_LorentzNetLGATrSlimGraphGPS",
    "tag_PlainGraphTrans",
    "tag_PlainGraphGPS",
    "tag_ParticleNetParTGraphTrans",
    "tag_ParticleNetParTGraphGPS",
]


def _net(model_name):
    with initialize(version_base=None, config_path="../../config_quick"):
        cfg = compose(config_name="toptagging", overrides=[f"model={model_name}"])
        cfg.model.out_channels = 2
        net_cfg = cfg.model.net
        for key, value in (("in_s_channels", 8), ("n_scalar", 8), ("in_features_h", 9)):
            if key in net_cfg:
                net_cfg[key] = value
        if "in_channels" in cfg.model:
            cfg.model.in_channels = 7
        if "hidden_reps_list" in net_cfg and net_cfg.hidden_reps_list:
            net_cfg.hidden_reps_list[0] = "7x0n"
    return instantiate(cfg.model).net, net_cfg._target_.rsplit(".", 1)[-1]


def _decayed(net, cls):
    """Names the model's actual grouping path would put in a decayed group."""
    no_wd = set(net.no_weight_decay()) if hasattr(net, "no_weight_decay") else set()
    out = []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue
        if cls in PART_PATH:
            exempt = param.ndim == 1 or name.endswith(".bias") or name in no_wd
        else:
            exempt = param.ndim <= 1 or name.endswith(".bias")
        if not exempt:
            out.append(name)
    return out


@pytest.mark.parametrize("model_name", MODELS)
def test_no_bias_is_weight_decayed(model_name):
    """A parameter named `*.bias` is a bias whatever its rank."""
    net, cls = _net(model_name)
    decayed_biases = [n for n in _decayed(net, cls) if n.endswith(".bias")]
    assert not decayed_biases, (
        f"{model_name} ({'ParT' if cls in PART_PATH else 'base'} grouping) weight-decays "
        f"{len(decayed_biases)} bias tensors, e.g. {decayed_biases[0]}. Multi-dim biases "
        f"(CGENN's MVLinear.bias is (1, C, 1)) slip past an ndim-only exemption."
    )


@pytest.mark.parametrize(
    "trans,gps",
    [
        ("tag_CGENNLGATrGraphTrans", "tag_CGENNLGATrGraphGPS"),
        ("tag_LorentzNetLGATrSlimGraphTrans", "tag_LorentzNetLGATrSlimGraphGPS"),
        ("tag_PlainGraphTrans", "tag_PlainGraphGPS"),
        ("tag_ParticleNetParTGraphTrans", "tag_ParticleNetParTGraphGPS"),
    ],
)
def test_graphtrans_and_graphgps_agree_on_parameter_classes(trans, gps):
    """The two topologies of one family take DIFFERENT grouping paths by design
    (GraphTrans has a CLS token, GraphGPS mean-pools). They must still treat the same
    kinds of parameter the same way, or the topology comparison carries a hidden
    regularization difference."""
    kinds = {}
    for name in (trans, gps):
        net, cls = _net(name)
        kinds[name] = {
            n.rsplit(".", 1)[-1] for n in _decayed(net, cls)
        }  # last path component: 'weight', 'bias', 'a', ...
    only_trans = kinds[trans] - kinds[gps]
    only_gps = kinds[gps] - kinds[trans]
    assert not (only_trans or only_gps), (
        f"decayed parameter kinds differ between topologies: only in {trans}: "
        f"{sorted(only_trans)}; only in {gps}: {sorted(only_gps)}"
    )
