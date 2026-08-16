"""find_lr's training-recipe alignment (2026-08-16 incident).

A bare `-cn toptagging model=tag_particlenet` sweep composed the GT AdamW default
while the model's recipe (top_particlenet -> top_ParT) trains with Ranger, whose
damped early steps put its loss-vs-lr curve ~an order of magnitude right of AdamW's.
Both finder statistics read low together (steepest 1.7-2.2e-4 vs the nine-rerun
Ranger envelope 1.32-1.91e-3), twice, before the cause was found. These tests pin
(a) the alignment decision, (b) the config facts that made it load-bearing.
"""

from pathlib import Path

import hydra
import pytest

from utils.find_lr import _recipe_training_choice

REPO = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------ decision function


def _isfile_for(*names):
    paths = {str(Path("config") / "training" / f"{n}.yaml") for n in names}
    return lambda p: str(Path(p)) in paths


def test_particlenet_aligns_to_its_own_recipe():
    got = _recipe_training_choice(
        "toptagging", ["particlenet", "ParticleNet"], "tag_gts_and_friends_default",
        overrides=["model=tag_particlenet", "save=false"],
        isfile=_isfile_for("top_particlenet"),
    )
    assert got == "top_particlenet"


def test_explicit_training_override_wins():
    got = _recipe_training_choice(
        "toptagging", ["particlenet"], "tag_gts_and_friends_default",
        overrides=["model=tag_particlenet", "training=tag_gts_and_friends_default"],
        isfile=_isfile_for("top_particlenet"),
    )
    assert got is None


def test_already_aligned_is_a_noop():
    got = _recipe_training_choice(
        "toptagging", ["particlenet"], "top_particlenet",
        overrides=[], isfile=_isfile_for("top_particlenet"),
    )
    assert got is None


def test_no_recipe_file_keeps_the_default():
    got = _recipe_training_choice(
        "toptagging", ["SomeHybridWithoutRecipe"], "tag_gts_and_friends_default",
        overrides=[], isfile=_isfile_for(),
    )
    assert got is None


def test_jc_prefix_and_unknown_task():
    assert _recipe_training_choice(
        "jctagging", ["transformer"], "jc_default",
        overrides=[], isfile=_isfile_for("jc_transformer"),
    ) == "jc_transformer"
    assert _recipe_training_choice(
        "amplitudes", ["transformer"], "default",
        overrides=[], isfile=_isfile_for("top_transformer", "jc_transformer"),
    ) is None


def test_stem_order_prefers_the_hydra_model_choice():
    # pointer semantics: the model-config stem wins over the net class name
    got = _recipe_training_choice(
        "toptagging", ["cgenn", "CGENN"], "tag_gts_and_friends_default",
        overrides=[], isfile=_isfile_for("top_cgenn", "top_CGENN"),
    )
    assert got == "top_cgenn"


# ---------------------------------------------------------------- config facts


@pytest.fixture()
def compose_toptagging():
    with hydra.initialize_config_dir(config_dir=str(REPO / "config"), version_base=None):
        yield lambda overrides: hydra.compose(config_name="toptagging", overrides=overrides)


def test_particlenet_recipe_is_ranger(compose_toptagging):
    """The recipe the FIND_LR pointer names trains with Ranger at lr 1e-2 (ParT-paper
    hyperparameters). The suggested lr must be measured under THIS optimizer."""
    cfg = compose_toptagging(["model=tag_particlenet", "training=top_particlenet"])
    assert cfg.training.optimizer == "Ranger"
    assert float(cfg.training.lr) == pytest.approx(1e-2)


def test_bare_compose_defaults_to_adamw(compose_toptagging):
    """The mismatch that made alignment load-bearing: without an override the task
    composes the GT AdamW default for EVERY model, baselines included. If this ever
    changes, the alignment docstring's history needs updating too."""
    cfg = compose_toptagging(["model=tag_particlenet"])
    assert cfg.training.optimizer == "AdamW"


def test_ranger_zero_grad_accepts_set_to_none():
    """Alignment routes Ranger through find_lr's batch-size probe, whose step calls
    `zero_grad(set_to_none=True)` -- weaver's Lookahead wrapper predated that kwarg
    and crashed the first aligned H100 sweep (2026-08-16). Pin the modern signature
    on the wrapper, and that the plain call still works."""
    import torch

    from experiments.ranger import Ranger

    p = torch.nn.Parameter(torch.randn(4))
    opt = Ranger([p], lr=1e-3, betas=(0.95, 0.999), eps=1e-5, alpha=0.5, k=6)
    (p * 2).sum().backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    assert p.grad is None, "set_to_none=True must clear grads to None"
    (p * 2).sum().backward()
    opt.zero_grad()  # legacy call path (base_experiment's training loop)
