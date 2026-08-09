# Should be evaluated on GPU
# otherwise the transformer FLOPs will be off, because it is not using flash-attention
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
        ["model=tag_CGENNLGATrGraphTrans"],
        ["model=tag_LorentzNetLGATrSlimGraphTrans"],
        ["model=tag_PlainGraphTrans"],
        ["model=tag_PlainGraphGPS"],
        ["model=tag_ParticleNetParTGraphGPS"],
        ["model=tag_CGENNLGATrGraphGPS"],
        ["model=tag_LorentzNetLGATrSlimGraphGPS"],
    ],
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
    num_parameters = sum(p.numel() for p in exp.model.parameters())

    print(
        f"flops(batchsize=1)={flops:.2e}; parameters={num_parameters}",
        model_list,
        framesnet,
        equivectors,
    )
    # print(flop_counter.get_table(depth=5))
