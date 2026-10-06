"""`evaluation.save_roc`, its two utilities, and the non-finite batch dump must agree on
what gets written where.

Top tagging and TopTagXL write ONE curve, plots_<idx>/roc.txt. JetClass has no single ROC
(10-class softmax): its per-class rejection cells are read from nine one-vs-QCD curves, and
before `_roc_curves` the base `plot()` reached for results["fpr"] there and KeyError'd --
after the whole training had run, with save_roc the one extra thing the operator had asked
for. The dump had the same shape of failure: a save=false run (exactly the kind that goes
hunting for a NaN) has no run dir, so torch.save raised and the only evidence was lost.

CURVES   the base hook names the binary curve `roc`; JetClass names one `roc_<Class>` per
         non-QCD class, and each pair is exactly the arrays its rejection cell is read from
WRITE    `_save_roc` writes every curve as `<plot_path>/<stem>.txt` in the two-column form
         `utils/roc_overlay.load` reads back
LABELS   `utils/roc_labels.py --roc NAME` picks the per-class file; the eval-only re-run keys
         (`evaluation.save_roc`, `evaluation.eval_set`, `train`, `warm_start_idx`, ...) are
         VOLATILE, so a curve regenerated with `train=false evaluation.save_roc=true` labels
         like its training run rather than as a second configuration -- while a real knob
         (gp_impl) still splits
DUMP     `_dump_nonfinite_batch` under save=false writes to a gitignored directory under the
         cwd and does NOT create the run dir; under save=true it writes into the run dir
"""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from scipy.interpolate import interp1d

from experiments.tagging.experiment import TaggingExperiment
from experiments.tagging.jetclassexperiment import JetClassTaggingExperiment
from utils import roc_labels, roc_overlay

REPO = Path(__file__).resolve().parents[2]
JC_CLASSES = ["ZJetsToNuNu", "HToBB", "HToCC", "HToGG", "HToWW4Q", "HToWW2Q1L",
              "TTBar", "TTBarLep", "WToQQ", "ZToQQ"]


def _bare(cls, **attrs):
    """An experiment instance without its `__init__` -- the real one wants a config tree,
    a device and data; everything under test here reads attributes only."""
    exp = object.__new__(cls)
    for k, v in attrs.items():
        setattr(exp, k, v)
    return exp


def _curve(seed, n=60):
    rng = np.random.default_rng(seed)
    fpr, tpr = np.sort(rng.uniform(0, 1, n)), np.sort(rng.uniform(0, 1, n))
    fpr[0], tpr[0], fpr[-1], tpr[-1] = 0.0, 0.0, 1.0, 1.0
    return fpr, tpr


# ----------------------------------------------------------------------------- CURVES

def test_binary_hook_names_the_one_curve_roc():
    fpr, tpr = _curve(0)
    curves = _bare(TaggingExperiment)._roc_curves({"fpr": fpr, "tpr": tpr, "auc": 0.9})
    assert list(curves) == ["roc"]
    assert curves["roc"][0] is fpr and curves["roc"][1] is tpr


def test_jetclass_hook_names_one_curve_per_non_qcd_class():
    results = {"roc_curves": {"HToBB": _curve(1), "HToCC": _curve(2)}}
    curves = _bare(JetClassTaggingExperiment, class_names=JC_CLASSES)._roc_curves(results)
    assert list(curves) == ["roc_HToBB", "roc_HToCC"]
    assert curves["roc_HToBB"] is results["roc_curves"]["HToBB"]


def test_jetclass_eval_keeps_the_curves_its_rejection_cells_are_read_from():
    """The stored (fpr, tpr) must be THE arrays behind rej*_<i>, not a re-derivation: an
    overlay read off roc_<Class>.txt at the class's eps_S has to reproduce the table cell."""
    rng = np.random.default_rng(3)
    n_per_class = 300
    labels = torch.arange(10).repeat_interleave(n_per_class)
    logits = torch.tensor(rng.normal(size=(10 * n_per_class, 10)), dtype=torch.float32)
    logits[torch.arange(len(labels)), labels] += 1.5  # separable enough for finite rejections
    exp = _bare(
        JetClassTaggingExperiment,
        class_names=JC_CLASSES,
        cfg=OmegaConf.create({"use_mlflow": False, "run_name": "fixture"}),
        model=SimpleNamespace(eval=lambda: None),
        _get_ypred_and_label=lambda batch: (logits, labels, None, None),
        _log_table_row=lambda *args, **kwargs: None,
    )
    metrics = exp._evaluate_single([None], "test", mode="eval")

    assert sorted(metrics["roc_curves"]) == sorted(JC_CLASSES[1:])
    eps = {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.99, 6: 0.5, 7: 0.995, 8: 0.5, 9: 0.5}
    for i, eps_s in eps.items():
        fpr, tpr = metrics["roc_curves"][JC_CLASSES[i]]
        cell = metrics[f"rej{str(eps_s).replace('.', '')}_{i}"]
        assert np.isclose(1 / interp1d(tpr, fpr)(eps_s), cell), (i, cell)

    # val mode keeps the scalars only (nothing to write, nothing to carry around)
    assert "roc_curves" not in exp._evaluate_single([None], "val", mode="val")


# ------------------------------------------------------------------------------ WRITE

def test_save_roc_writes_one_file_per_curve_that_roc_overlay_reads_back(tmp_path):
    results = {"test": {"roc_curves": {"HToBB": _curve(1), "HToCC": _curve(2)}}}
    _bare(JetClassTaggingExperiment, class_names=JC_CLASSES, results=results)._save_roc(str(tmp_path))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["roc_HToBB.txt", "roc_HToCC.txt"]

    fpr, tpr = roc_overlay.load(str(tmp_path / "roc_HToBB.txt"))
    want_fpr, want_tpr = results["test"]["roc_curves"]["HToBB"]
    keep = want_fpr > 0  # load() drops the fpr == 0 rows, where 1/fpr is infinite
    np.testing.assert_allclose(fpr, want_fpr[keep])
    np.testing.assert_allclose(tpr, want_tpr[keep])


def test_binary_save_roc_still_writes_roc_txt(tmp_path):
    fpr, tpr = _curve(0)
    _bare(TaggingExperiment, results={"test": {"fpr": fpr, "tpr": tpr}})._save_roc(str(tmp_path))
    assert [p.name for p in tmp_path.iterdir()] == ["roc.txt"]
    got = np.loadtxt(tmp_path / "roc.txt")
    np.testing.assert_allclose(got[:, 0], fpr)
    np.testing.assert_allclose(got[:, 1], tpr)


# ----------------------------------------------------------------------------- LABELS

def _fake_run(root, name, cfg, stems):
    d = root / name
    (d / "plots_0").mkdir(parents=True)
    OmegaConf.save(OmegaConf.create(cfg), d / "config.yaml")
    for stem in stems:
        np.savetxt(d / "plots_0" / f"{stem}.txt", np.stack(_curve(0), axis=-1))


def _labels(*args):
    proc = subprocess.run(
        [sys.executable, str(REPO / "utils" / "roc_labels.py"), *map(str, args)],
        capture_output=True, text=True, cwd=REPO,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.splitlines(), proc.stderr


TRAINED = {
    "model": {"net": {"_target_": "x.PlainGraphGPS", "knn_metric": "deltaR"}},
    "training": {"batchsize": 512, "lr": 1e-3},
    "evaluation": {"save_roc": False, "eval_set": ["train", "test", "val"]},
    "run_idx": 0, "train": True, "evaluate": True, "warm_start_idx": None, "save": True,
}


def test_roc_option_picks_the_per_class_file(tmp_path):
    _fake_run(tmp_path, "PlainGraphGPS_0001", TRAINED, ["roc_HToBB", "roc_HToCC"])
    assert _labels(tmp_path)[0] == []  # the default looks for roc.txt, and there is none
    out, _ = _labels(tmp_path, "--roc", "roc_HToBB")
    assert out == [f"PlainGraphGPS={tmp_path}/PlainGraphGPS_0001/plots_0/roc_HToBB.txt"]


def test_eval_only_rerun_keys_are_volatile():
    for key in ("evaluation.save_roc", "evaluation.eval_set", "train", "evaluate",
                "warm_start_idx", "run_idx", "slurm_job_id"):
        assert roc_labels.VOLATILE.search(key), key
    for key in ("model.net.gp_impl", "model.net.knn_metric", "training.batchsize",
                "training.lr", "data.dataset", "training.epochs"):
        assert not roc_labels.VOLATILE.search(key), key


def test_regenerated_curve_labels_like_its_training_run_but_a_real_knob_splits(tmp_path):
    # the eval-only re-run (GUIDE section 8) rewrites config.yaml with exactly these changed
    rerun = {**TRAINED, "evaluation": {"save_roc": True, "eval_set": ["test"]},
             "run_idx": 1, "train": False, "warm_start_idx": 0}
    _fake_run(tmp_path, "PlainGraphGPS_0001", TRAINED, ["roc"])
    _fake_run(tmp_path, "PlainGraphGPS_0002", rerun, ["roc"])
    cgenn = {**TRAINED, "model": {"net": {"_target_": "x.CGENN", "gp_impl": "flash"}}}
    _fake_run(tmp_path, "CGENN_0001", cgenn, ["roc"])
    _fake_run(tmp_path, "CGENN_0002", {**cgenn, "model": {"net": {"_target_": "x.CGENN", "gp_impl": "sparse"}}}, ["roc"])

    out, err = _labels(tmp_path, "--all-trials")
    labels = [line.rsplit("=", 1)[0] for line in out]
    assert labels == ["CGENN (gp_impl=flash)", "CGENN (gp_impl=sparse)",
                      "PlainGraphGPS", "PlainGraphGPS"], (out, err)
    assert "PlainGraphGPS" not in err, err  # no "differ in [...]" report for the re-run pair


# ------------------------------------------------------------------------------- DUMP

def _dump_cfg(save, run_dir):
    return OmegaConf.create({"save": save, "run_dir": str(run_dir), "exp_name": "x", "run_name": "M_0001"})


def test_nonfinite_dump_survives_save_false_without_creating_the_run_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "runs" / "x" / "M_0001"  # save=false: never created by the run
    exp = _bare(TaggingExperiment, cfg=_dump_cfg(False, run_dir))
    data = {"x": torch.tensor([1.0, float("nan")]), "label": torch.tensor([1])}
    exp._dump_nonfinite_batch(data, torch.tensor(float("nan")), torch.tensor(float("inf")), step=7)

    dumped = sorted((tmp_path / "nonfinite_batches").iterdir())
    assert [p.name for p in dumped] == ["x_M_0001_step7.pt"]
    blob = torch.load(dumped[0])
    assert blob["step"] == 7
    assert torch.equal(blob["tensors"]["batch.label"], torch.tensor([1]))
    assert torch.isnan(blob["tensors"]["batch.x"][1])
    assert not (tmp_path / "runs").exists(), "a run dir with no config.yaml is a MISSING-config row"


def test_nonfinite_dump_goes_into_the_run_dir_when_the_run_saves(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "M_0001"
    run_dir.mkdir()
    exp = _bare(TaggingExperiment, cfg=_dump_cfg(True, run_dir))
    exp._dump_nonfinite_batch({"x": torch.ones(2)}, torch.tensor(1.0), torch.tensor(float("nan")), step=3)
    assert (run_dir / "nonfinite_batch_step3.pt").is_file()
    assert not (tmp_path / "nonfinite_batches").exists()


def test_nonfinite_dump_is_once_per_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exp = _bare(TaggingExperiment, cfg=_dump_cfg(False, tmp_path / "nowhere"))
    for step in (5, 6):
        exp._dump_nonfinite_batch({"x": torch.ones(1)}, torch.tensor(1.0), torch.tensor(float("nan")), step=step)
    assert [p.name for p in (tmp_path / "nonfinite_batches").iterdir()] == ["x_M_0001_step5.pt"]
