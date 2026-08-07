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

from experiments.base_experiment import lgatr_norm_gain_names

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
    # H15: both real paths exempt lgatr 2.0's affine norm gains via the same helper;
    # the mirror must track them or it would re-report the disease the paths just cured.
    gains = lgatr_norm_gain_names(net)
    out = []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue
        if cls in PART_PATH:
            exempt = param.ndim == 1 or name.endswith(".bias") or name in no_wd or name in gains
        else:
            exempt = param.ndim <= 1 or name.endswith(".bias") or name in gains
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
    "model_name",
    ["tag_lgatr", "tag_slim", "tag_CGENNLGATrGraphTrans", "tag_CGENNLGATrGraphGPS"],
)
def test_norm_gains_sit_in_no_decay_groups(model_name):
    """H15 end-to-end (the runbook's Gate-B extension): build the REAL experiment optimizer
    -- whichever grouping path the model routes through -- and require every parameter named
    `*.weight_mv` / `*.weight_s` to sit in a `weight_decay=0` group. Those names exist only
    on lgatr 2.0's affine norms; `weight_s` is 1-d and already rank-exempt, but asserting
    both keeps the rule self-documenting (runbook H15). Models whose norms are structurally
    affine-off (the GPS hybrids' bare layer constructions, 2.1) pass vacuously -- which is
    itself the structural claim, so an upstream default flip would surface here.
    """
    import logging.handlers  # noqa: F401
    import hydra
    import experiments.logger
    from experiments.tagging.experiment import TopTaggingExperiment

    experiments.logger.LOGGER.disabled = True
    with initialize(version_base=None, config_path="../../config_quick"):
        cfg = compose(config_name="toptagging", overrides=[f"model={model_name}", "save=false"])
    exp = TopTaggingExperiment(cfg)
    exp._init()
    exp.init_physics()
    exp.init_model()
    exp._init_optimizer()
    wd_of = {}
    for group in exp.optimizer.param_groups:
        for p in group["params"]:
            wd_of[id(p)] = group.get("weight_decay", 0.0)
    gains = [
        (n, p)
        for n, p in exp.model.named_parameters()
        if n.rsplit(".", 1)[-1] in ("weight_mv", "weight_s")
    ]
    decayed = [n for n, p in gains if wd_of.get(id(p), 0.0) != 0.0]
    assert not decayed, (
        f"{model_name}: affine norm gains landed in a decayed group: {decayed} -- "
        f"H15's silent gain-decay disease; both grouping paths must exempt them"
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
