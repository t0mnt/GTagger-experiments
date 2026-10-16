import hydra
import pytest
import torch
from lloca.utils.rand_transforms import (
    rand_lorentz,
    rand_rotation,
    rand_xyrotation,
)

import experiments.logger
from experiments.tagging.experiment import TopTaggingExperiment

BREAKING = [
    "data.beam_reference=null",
    "data.add_time_reference=false",
    "data.tagging_features=null",
]


@pytest.mark.parametrize(
    "rand_trafo,breaking_list",
    [
        [rand_rotation, BREAKING],
        [rand_lorentz, BREAKING],
        [rand_xyrotation, []],
    ],
)
@pytest.mark.parametrize(
    "model_list",
    list(
        [
            ["model=tag_ParT"],
            ["model=tag_transformer"],
            ["model=tag_graphnet"],
            ["model=tag_graphnet", "model.include_edges=true"],
        ]
    ),
)
@pytest.mark.parametrize("framesnet", ["learnedpd", "learnedso13"])
def test_amplitudes(
    rand_trafo,
    model_list,
    framesnet,
    breaking_list,
    iterations=1,
    batchsize=4,
):
    experiments.logger.LOGGER.disabled = True  # turn off logging
    # Deterministic: config_quick sets no seed, so the init, the shuffled batch and the
    # random transforms below all came from whatever RNG state earlier tests left behind.
    # The invariance floor is set by mass regularization (E -> sqrt(|p|^2 + mass_reg^2) is
    # not covariant; with data.mass_reg=null every row is exactly invariant, with the
    # framesnet wiring of 7664162 the learnedpd rows sit at ~1e-7 to 4e-7 max MSE), and one
    # session-order draw of a large boost pushed the graphnet+edges/learnedpd/lorentz row to
    # 1.2e-5 while the same case measured 1.5e-7 and 3.7e-7 standalone.
    torch.manual_seed(0)

    # create experiment environment
    with hydra.initialize(config_path="../../config_quick", version_base=None):
        overrides = [
            *model_list,
            f"model/framesnet={framesnet}",
            f"training.batchsize={batchsize}",
            "save=false",
            *breaking_list,
        ]
        cfg = hydra.compose(config_name="toptagging", overrides=overrides)
        exp = TopTaggingExperiment(cfg)
    exp._init()
    exp.init_physics()
    exp.init_model()
    exp.init_data()
    exp._init_dataloader()
    exp._init_loss()
    exp.model.eval()  # turn off dropout

    def cycle(iterable):
        while True:
            yield from iterable

    mses = []
    iterator = iter(cycle(exp.train_loader))
    for _ in range(iterations):
        data = next(iterator)
        data_augmented = data.clone()

        # original data
        y_pred, _, tracker, _ = exp._get_ypred_and_label(data)

        # augmented data: one independent transform per jet (a batch-global
        # transform could not catch cross-jet leakage)
        mom = data_augmented.x  # sparse (total_particles, 4)
        n_jets = int(data_augmented.batch.max().item()) + 1
        trafo = rand_trafo((n_jets,), dtype=mom.dtype)
        trafo = trafo.index_select(0, data_augmented.batch)
        mom_augmented = torch.einsum("...ij,...j->...i", trafo, mom)
        data_augmented.x = mom_augmented
        y_pred_augmented = exp._get_ypred_and_label(data_augmented)[0]

        mse = (y_pred_augmented - y_pred) ** 2
        mses.append(mse.detach())
    mses = torch.cat(mses, dim=0).clamp(min=1e-20)
    print(
        f"log-mean={mses.log().mean().exp():.2e} max={mses.max().item():.2e}",
        model_list,
        rand_trafo.__name__,
        framesnet,
    )
    for key in tracker.keys():
        print(f"data tracker key={key}, value={tracker[key]}")
    # the actual invariance assertion (this test printed-and-passed for months --
    # an assertion-free test can only fail by crashing; adding this immediately
    # exposed the boost_jet feature-degeneration bug at MSE O(10^2..10^4)).
    # Healthy cases measure <= ~1e-6 max MSE on the mini dataset (fp32 logits);
    # a real symmetry break measures >= 1e-2, usually much larger, so 1e-5 leaves
    # an order of magnitude headroom on both sides.
    assert mses.max().item() < 1e-5, (
        f"invariance violated: max MSE {mses.max().item():.2e} "
        f"({model_list}, {rand_trafo.__name__}, {framesnet})"
    )
