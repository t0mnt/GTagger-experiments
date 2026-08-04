"""Config-tree invariants that nothing else enforces.

Each of these guards a seam where two files must agree but no import, test or type
checker connects them, so drift is silent until a cluster run behaves oddly.
"""

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "config"
QUICK = REPO / "config_quick"

HYBRIDS = [
    f"{gnn}{topology}"
    for gnn in ("Plain", "ParticleNetParT", "CGENNLGATr", "LorentzNetLGATrSlim")
    for topology in ("GraphTrans", "GraphGPS")
]
TASK_PREFIX = {"toptagging": "top", "jctagging": "jc", "toptagxl": "xl"}

# Knobs that select a CODE PATH rather than a size. config_quick may legitimately shrink
# widths and depths, but if it flips one of these it is exercising a different branch
# than the real config, and a smoke run stops being predictive of the real run.
PATH_SWITCHING = (
    "use_rwse",
    "use_edge_attr",
    "use_explicit_edge_features",
    "bias",
    "norm",
    "knn_metric",
    "use_input_concat",
    "use_fusion",
)


@pytest.mark.parametrize("model", HYBRIDS)
@pytest.mark.parametrize("task", sorted(TASK_PREFIX))
def test_every_hybrid_has_a_recipe_for_every_task(model, task):
    """train.sbatch derives `training=<prefix>_<Model>` and falls back silently if the
    file is absent -- the run would then use the task default instead of the tuned
    recipe, with no warning at submission time."""
    recipe = CONFIG / "training" / f"{TASK_PREFIX[task]}_{model}.yaml"
    assert recipe.is_file(), (
        f"missing {recipe.relative_to(REPO)}. docs/oscar-train.sbatch only adds "
        f"training= when this file exists, so a submission would silently fall back to "
        f"the {task} default recipe."
    )


@pytest.mark.parametrize("model", HYBRIDS)
@pytest.mark.parametrize("task", sorted(TASK_PREFIX))
def test_hybrid_recipes_leave_exactly_batchsize_and_lr_unset(model, task):
    """`???` is a human-facing marker, not an OmegaConf MISSING that errors.

    An unfilled recipe silently trains at the inherited family fallback (512 / 1e-3),
    which is why experiment.py warns at runtime. A stray `???` on any OTHER key would
    fall back the same way with no warning at all.
    """
    raw = (CONFIG / "training" / f"{TASK_PREFIX[task]}_{model}.yaml").read_text()
    unset = {
        line.split(":", 1)[0].strip()
        for line in raw.splitlines()
        if "???" in line and not line.strip().startswith("#")
    }
    assert unset <= {"batchsize", "lr", "weight_decay"}, (
        f"{TASK_PREFIX[task]}_{model}.yaml leaves {sorted(unset - {'batchsize', 'lr', 'weight_decay'})} "
        f"unset; only batchsize/lr (optionally weight_decay) may be '???'. Anything else "
        f"silently inherits the family default."
    )


@pytest.mark.parametrize("model", HYBRIDS)
def test_quick_config_exercises_the_same_code_paths(model):
    """config_quick is meant to be the SAME model, smaller -- not a different one.

    Size differences (num_layers, hidden widths, mlp_ratio) are the whole point and are
    ignored here. But a path-switching knob that disagrees means a config_quick smoke
    run does not exercise what the real run will.
    """
    full_path = CONFIG / "model" / f"tag_{model}.yaml"
    quick_path = QUICK / "model" / f"tag_{model}.yaml"
    if not quick_path.is_file():
        pytest.skip(f"no config_quick counterpart for tag_{model}")
    full = (yaml.safe_load(full_path.read_text()) or {}).get("net", {}) or {}
    quick = (yaml.safe_load(quick_path.read_text()) or {}).get("net", {}) or {}
    disagreements = {
        knob: (full[knob], quick[knob])
        for knob in PATH_SWITCHING
        if knob in full and knob in quick and full[knob] != quick[knob]
    }
    assert not disagreements, (
        f"tag_{model}: config_quick flips path-switching knob(s) {disagreements} "
        f"(full, quick). Smoke runs would exercise a different branch than the real run."
    )
