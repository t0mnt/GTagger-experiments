# Should be evaluated on GPU for any model with an attention stage: FlopCounterMode has
# no formula for aten::_scaled_dot_product_flash_attention_for_cpu, so a CPU run prices
# EVERY SDPA call at zero -- that is the transformers, the lgatr family, and all GT
# hybrids (their trunks). Measured: L-GATr-slim 297M CPU vs 333M GPU (todo.md, FLOPs
# provenance note). Attention-free models (ParticleNet, LorentzNet, MIParT, ParT,
# tag_cgenn) are device-independent and can be (re)priced anywhere.
import hydra
import pytest
from torch.utils.flop_counter import FlopCounterMode

import experiments.logger
from experiments.tagging.experiment import TopTaggingExperiment


@pytest.mark.parametrize(
    "framesnet,equivectors",
    [
        ["identity", None],
        ["learnedpd", "equimlp"],
        ["learnedpd", "pelican"],
        ["learnedpd", "lgatr"],
    ],
)
@pytest.mark.parametrize(
    "model_list",
    # ids: name each case after its model, so a single row can be regenerated with
    #   pytest tests/experiments/test_tag_flops.py -s -k "tag_cgenn and identity"
    # (the default ids are positional -- model_list13 -- which makes that impossible).
    [
        ["model=tag_ParT"],
        ["model=tag_particlenet"],
        ["model=tag_ParticleNetParTGraphTrans"],
        ["model=tag_transformer"],
        ["model=tag_graphnet"],
        ["model=tag_graphnet", "model.include_edges=true"],
        ["model=tag_lgatr"],
        ["model=tag_MIParT"],
        ["model=tag_MIParT-L"],
        ["model=tag_lorentznet"],
        ["model=tag_pelican_fair"],
        ["model=tag_slim"],
        ["model=tag_top_transformer"],
        # tag_cgenn is the reference row for BOTH CGENN hybrids, so it belongs here --
        # gp_impl pinned because `flash` routes the geometric product through the
        # Cl(1,3) Triton custom op (flash_kernels_p1m3.py). FlopCounterMode dispatches
        # on ATen and has no flop formula for a custom op, so the fused GP would count
        # as ZERO and the row would silently under-report. `sparse` is the same maths
        # through traceable ops. Same reason the two CGENN hybrids below are pinned.
        ["model=tag_cgenn", "model.net.gp_impl=sparse"],
        ["model=tag_CGENNLGATrGraphTrans", "model.net.gp_impl=sparse"],
        ["model=tag_LorentzNetLGATrSlimGraphTrans"],
        ["model=tag_PlainGraphTrans"],
        ["model=tag_PlainGraphGPS"],
        ["model=tag_ParticleNetParTGraphGPS"],
        ["model=tag_CGENNLGATrGraphGPS", "model.net.gp_impl=sparse"],
        ["model=tag_LorentzNetLGATrSlimGraphGPS"],
    ],
    ids=lambda ml: ml[0].split("=", 1)[1] + ("" if len(ml) == 1 else "-pinned"),
)
def test_tagging(framesnet, model_list, equivectors, jet_size=50):
    experiments.logger.LOGGER.disabled = True  # turn off logging

    # create experiment environment
    with hydra.initialize(config_path="../../config", version_base=None):
        overrides = [
            *model_list,
            f"model/framesnet={framesnet}",
            "save=false",
            "training.batchsize=1",
            "data.dataset=mini",
        ]
        if framesnet != "identity":
            overrides.append(f"model/framesnet/equivectors={equivectors}")
        cfg = hydra.compose(config_name="toptagging", overrides=overrides)
        # FLOPs are compile-independent by construction (docs/cgenn-compile.md, table
        # policy) and FX cannot symbolically trace a dynamo-compiled module -- force the
        # eager build regardless of the production configs' compiled-dynamic defaults.
        # Recursive on purpose: nested knobs (e.g. framesnet.equivectors.net.compile in
        # the pelican equivectors config) produced FX's 'symbolically trace a
        # dynamo-optimized function' failure, which sat misdiagnosed inside the 'known
        # pelican environment failures' set until the final operator round.
        from omegaconf import DictConfig, open_dict
        def _force_eager(node):
            if isinstance(node, DictConfig):
                with open_dict(node):
                    for k in list(node.keys()):
                        if k == "compile" and isinstance(node[k], bool):
                            node[k] = False
                        elif isinstance(node[k], DictConfig):
                            _force_eager(node[k])
        _force_eager(cfg.model)
        exp = TopTaggingExperiment(cfg)
    exp._init()
    exp.init_physics()
    try:
        exp.init_model()
    except Exception as e:
        # environment-dependent model inits (e.g. xformers-pinned attention baselines on
        # a CPU-only runner) are tolerated -- but VISIBLY, as a skip, not a silent PASS
        pytest.skip(f"init_model failed (environment-dependent): {type(e).__name__}: {e}")
    exp.init_data()
    exp._init_dataloader()
    exp._init_loss()

    iterator = iter(exp.train_loader)
    data = next(iterator)
    while data.x.shape[0] < jet_size:
        data = next(iterator)
    data.x = data.x[:jet_size]
    data.scalars = data.scalars[:jet_size]
    data.batch = data.batch[:jet_size]
    data.ptr[-1] = jet_size

    with FlopCounterMode(display=False) as flop_counter:
        try:
            exp._get_ypred_and_label(data)
        except AssertionError as e:
            # An attention backend can be missing at FORWARD time rather than init time:
            # lloca's LGATr equivectors resolve `attention_backend` inside their forward, so
            # an xformers build that is ABI-mismatched with the installed torch fails here,
            # past the init_model guard above. Tolerate that ONE message as a visible skip;
            # every other AssertionError is a real failure and must propagate.
            if "not installed, run 'pip install lloca[" not in str(e):
                raise
            pytest.skip(f"attention backend unavailable in this environment: {e}")
    flops = flop_counter.get_total_flops()
    # requires_grad, to match what the results tables record (base_experiment.py:177 and
    # experiment.py's table row both filter). Without the filter this printed 2030817 for
    # tag_slim against the table's 2014401 -- a 16416 gap that looks like an lgatr version
    # skew and is not one: LGATrSlim freezes four tensors in its last block (out_v_channels
    # is 0, so that block's vector-side weights carry no gradient). Every other model has
    # no frozen parameters, which is why only this one row disagreed.
    num_parameters = sum(p.numel() for p in exp.model.parameters() if p.requires_grad)

    print(
        f"flops(batchsize=1)={flops:.2e}; parameters={num_parameters}",
        model_list,
        framesnet,
        equivectors,
    )
    # print(flop_counter.get_table(depth=5))
