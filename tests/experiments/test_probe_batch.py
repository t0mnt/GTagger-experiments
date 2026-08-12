"""Gates for the constructed probe batch in `utils/find_lr.py`.

KEEP-permanently. `find_max_batch_size` sizes the batch for every job in the campaign, and
it decides on ONE batch per rung. It used to draw that batch at random, which makes it a
MEDIAN batch while a real run meets the worst of ~10^5 draws — the gap `bs_safety<1` was
covering, badly (a halved batch is ~2x headroom for a <=1.3x problem). `_worst_case_indices`
closes it by CONSTRUCTING the probe batch instead.

The construction is pure length arithmetic, so it is fully testable on CPU with no GPU, no
model and no hydra. What the gates hold:

SHAPE     exactly `bs` items, no repeats — a repeated jet would silently shrink the probe
TARGET    the constructed batch reaches the +sigmas target in BOTH `sum n` and `sum n^2`
PMAX      the dataset's longest jet is always in the batch, so P_max hits its cap too
CALIB     the constructed totals bracket the worst batch a real run draws — above it, but
          not by so much that the probe throws GPU away
MONOTONE  heavier `sigmas` gives a heavier batch, and sigmas=0 gives a typical one
GUARD     datasets it cannot build from return None (the caller falls back to a random draw)
COLLATE   the real TopTaggingDataset + the real PyG collate produce a usable Batch

TARGET covers both moments deliberately: memory is linear in total nodes for some models
(kNN edge lists, node features) and quadratic for others (dense edges, block attention), and
an earlier version that targeted only `sum n^2` left `sum n` ~4% short at B=128 — the entire
margin for the linear family. CALIB is what would catch a silent regression in the
calibration itself, and it is two-sided: a probe that over-shoots wastes the card as surely
as one that under-shoots leaves the job to die at step 10.
"""

import logging

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from utils.find_lr import (
    PROBE_SIGMAS,
    _probe_lengths,
    _worst_case_batch,
    _worst_case_indices,
    find_max_batch_size,
)

# top-tagging's own shape (mean 49.2, sd 17.2, max 135), reproduced without the dataset
LENGTHS = np.clip(np.random.default_rng(0).gamma(8.2, 6.0, size=40000), 4, 135).astype(np.int64)
BATCHES = [16, 64, 128, 512, 4096]


@pytest.fixture(autouse=True)
def _audible_logger():
    """Three of these gates read the log, so the log must actually reach caplog.

    Whether it does is otherwise a function of COLLECTION ORDER: any module that disables
    the `main` logger globally silences these. One did, and this file is what found it
    (tests/internal/test_jc_wiring.py, now scoped). Not relying on that staying fixed.
    """
    from experiments.logger import LOGGER

    was_disabled, LOGGER.disabled = LOGGER.disabled, False
    was_global = logging.root.manager.disable  # `logging.disable` has no getter
    logging.disable(logging.NOTSET)
    try:
        yield
    finally:
        LOGGER.disabled = was_disabled
        logging.disable(was_global)


def total(lengths, idx, power=2):
    return float((lengths[idx].astype(np.float64) ** power).sum())


def target(lengths, bs, sigmas=PROBE_SIGMAS, power=2):
    moment = lengths.astype(np.float64) ** power
    return bs * moment.mean() + sigmas * np.sqrt(bs) * moment.std()


@pytest.mark.parametrize("bs", BATCHES)
def test_shape_and_no_repeats(bs):
    """SHAPE: a repeated jet would make the probe lighter than the batch it stands for."""
    idx = _worst_case_indices(LENGTHS, bs)
    assert idx is not None and idx.shape == (bs,)
    assert np.unique(idx).size == bs
    assert idx.min() >= 0 and idx.max() < LENGTHS.size


@pytest.mark.parametrize("bs", BATCHES)
@pytest.mark.parametrize("power", [1, 2])
def test_reaches_the_target(bs, power):
    """TARGET: BOTH moments, not just `sum n^2`.

    Targeting the dominant term alone left `sum n` ~4% short at B=128 — the whole margin
    for a model whose memory is linear in total nodes (kNN edge lists, node features).
    """
    idx = _worst_case_indices(LENGTHS, bs)
    assert total(LENGTHS, idx, power) >= target(LENGTHS, bs, power=power)


@pytest.mark.parametrize("bs", BATCHES)
def test_longest_jet_is_always_in(bs):
    """PMAX: dense terms scale with B*P_max, so the probe must cap P_max too."""
    idx = _worst_case_indices(LENGTHS, bs)
    assert LENGTHS[idx].max() == LENGTHS.max()


@pytest.mark.parametrize("bs", [128, 512, 4096])
@pytest.mark.parametrize("power", [1, 2])
def test_calibrated_against_the_worst_batch_of_a_real_run(bs, power):
    """CALIB: +5 sd is meant to BE the worst batch of a long run, not a round number.

    Two checks, because each catches what the other cannot. The SIMULATED one draws real
    batches and takes the max — no distributional assumption, but only as many batches as
    a test can afford. The EXTRAPOLATED one covers the full 50-epoch run via the
    extreme-value estimate for a sum of i.i.d. terms, which is sound for a SUM (CLT) and
    would NOT be for a maximum — P_max is hard-capped by the dataset and is handled by
    construction instead, not by this arithmetic.
    """
    rng = np.random.default_rng(1)
    draws = LENGTHS[rng.integers(0, LENGTHS.size, size=(20000, bs))].astype(np.float64)
    sums = (draws**power).sum(1)
    built = total(LENGTHS, _worst_case_indices(LENGTHS, bs), power)

    assert built >= sums.max(), "lighter than a batch that turned up in 20k draws"

    steps = 50 * (1_211_000 // bs)  # a 50-epoch top-tagging run
    worst = sums.mean() + np.sqrt(2 * np.log(steps)) * sums.std()
    assert built >= worst, f"{built / worst:.3f} of the worst batch a real run draws"
    assert built <= 1.6 * worst, f"{built / worst:.3f}x — over-conservative, wastes the GPU"


@pytest.mark.parametrize("bs", [128, 512])
def test_monotone_in_sigmas(bs):
    """MONOTONE: sigmas is the knob's meaning; sigmas=0 must reproduce a typical batch."""
    totals = [total(LENGTHS, _worst_case_indices(LENGTHS, bs, s)) for s in (0.0, 2.0, 5.0, 8.0)]
    assert totals == sorted(totals)
    typical = bs * (LENGTHS.astype(np.float64) ** 2).mean()
    assert totals[0] == pytest.approx(typical, rel=0.05)


def test_unbuildable_datasets_return_none():
    """GUARD: returning None is what sends the caller back to the random-batch probe."""
    assert _worst_case_indices(LENGTHS, LENGTHS.size) is None  # no room for the top-k pool
    assert _worst_case_indices(LENGTHS, 0) is None
    assert _worst_case_indices(np.array([5, 5, 5]), 8) is None
    assert _probe_lengths(None) is None
    assert _probe_lengths(object()) is None  # no .data_list -> iterable datasets fall back
    assert _worst_case_indices(np.full(4000, 7), 128) is not None  # zero variance still builds


def test_unreachable_target_still_builds_and_says_so(caplog):
    """GUARD: at most bs//4 swaps, so a wild enough distribution cannot reach the target.

    Nothing like top-tagging -- half the jets at n=1 and half at n=200 -- but the arm must
    degrade to "as heavy as I can make it, and here is a warning" rather than under-probe
    in silence or hand back a short batch.
    """
    wild = np.array([1, 200] * 2000, dtype=np.int64)
    bs = 16
    with caplog.at_level("WARNING"):
        idx = _worst_case_indices(wild, bs)
    assert idx is not None and idx.shape == (bs,) and np.unique(idx).size == bs
    assert total(wild, idx) < target(wild, bs)  # the premise of the warning
    assert any("of the +5 sd target" in r.message for r in caplog.records)


class _FakeExp:
    """Enough of an experiment for `find_max_batch_size`, with a MEMORY MODEL.

    The card holds `ceiling` units; a batch costs `sum n^2`. That is CGENN's dominant term
    and the one the old random probe was blind to. Everything else is the smallest stub the
    search actually touches.
    """

    def __init__(self, lengths, ceiling):
        self.lengths, self.ceiling, self.peak = lengths, ceiling, 0.0
        self.data_train = _FakeDataset(lengths)
        self.train_loader = _FakeLoader(lengths, self.data_train)
        self.model = torch.nn.Linear(1, 1)
        self.optimizer, self.scaler = _FakeOptimizer(), _FakeScaler()
        self.cfg = OmegaConf.create({"training": {"batchsize": 16, "clip_grad_norm": None}})

    def _init_dataloader(self):
        self.train_loader.batch_size = int(self.cfg.training.batchsize)

    def _batch_loss(self, data):
        self.peak = float((np.asarray(data) ** 2).sum())
        if self.peak > self.ceiling:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
        return self.model.weight.sum(), {}


class _FakeDataset:
    """Shaped like TaggingDataset: `_probe_lengths` reads `data_list[i].x.shape[0]`."""

    class _Jet:
        __slots__ = ("x",)

        class _X:
            __slots__ = ("shape",)

            def __init__(self, n):
                self.shape = (n, 4)

        def __init__(self, n):
            self.x = self._X(n)

    def __init__(self, lengths):
        self.data_list = [self._Jet(int(n)) for n in lengths]
        self._lengths = lengths

    def __getitem__(self, i):
        return int(self._lengths[i])


class _FakeLoader:
    """collate_fn returns the batch's lengths; iterating draws a RANDOM batch, as before."""

    def __init__(self, lengths, dataset, seed=0):
        self.lengths, self.dataset, self.batch_size = lengths, dataset, 16
        self.rng = np.random.default_rng(seed)

    collate_fn = staticmethod(lambda items: np.asarray(items, dtype=np.int64))

    def __iter__(self):
        yield self.lengths[self.rng.integers(0, self.lengths.size, size=self.batch_size)]


class _FakeOptimizer:
    def zero_grad(self, set_to_none=True):
        pass


class _FakeScaler:
    def scale(self, loss):
        return loss

    def unscale_(self, opt):
        pass

    def step(self, opt):
        pass

    def update(self):
        pass


def _run_search(monkeypatch, lengths, ceiling, **kw):
    for name, fn in (
        ("is_available", lambda: True),
        ("empty_cache", lambda: None),
        ("reset_peak_memory_stats", lambda: None),
        ("max_memory_allocated", lambda: 0),
    ):
        monkeypatch.setattr(torch.cuda, name, fn)
    exp = _FakeExp(lengths, ceiling)
    return find_max_batch_size(exp, 16, 8192, kw.pop("safety", 1.0), **kw)


def _worst_batch_of_a_run(rng, bs, jet_cap=20_000_000):
    """Heaviest batch of a 50-epoch run, capped at `jet_cap` sampled jets to bound the test.

    Capping only makes the estimate SMALLER, so it weakens `survives` assertions and
    strengthens `should have OOM'd` ones -- it cannot manufacture a pass of the first kind.
    """
    steps = min(50 * (1_211_000 // bs), max(1, jet_cap // bs))
    draws = LENGTHS[rng.integers(0, LENGTHS.size, size=(steps, bs))]
    return float((draws.astype(np.int64) ** 2).sum(1).max())


def test_search_returns_a_batch_that_survives_the_worst_real_batch(monkeypatch):
    """The whole change, end to end: probe -> chosen batch -> does the RUN survive it?

    The card is sized INSIDE the band where the two probes disagree (see the frequency test
    below for why that band is narrow). With the constructed probe the run survives; with a
    typical batch -- which is what a random draw gives you, and what `bs_sigmas=0`
    reproduces -- the search over-shoots by an octave and the run dies. That is the failure
    that was observed for real, at step 10, hours into a job.
    """
    rng = np.random.default_rng(3)
    ceiling = 1.5e6  # between a typical 512-batch (1.39e6) and the run's worst (1.59e6)

    built = _run_search(monkeypatch, LENGTHS, ceiling)
    typical = _run_search(monkeypatch, LENGTHS, ceiling, sigmas=0.0)
    assert built < typical, "this ceiling is meant to separate the two probes"
    assert _worst_batch_of_a_run(rng, built) <= ceiling, "constructed probe still OOMs the run"
    assert _worst_batch_of_a_run(rng, typical) > ceiling, "typical probe should have OOM'd"


def test_never_worse_than_the_old_probe_at_any_card_size(monkeypatch):
    """The property that has to hold for EVERY card, not just the one that shows it off.

    Sweeping the ceiling across an octave: the constructed probe must never pick a bigger
    batch than the old one, and the batch it picks must always survive the worst batch the
    run draws. It also PINS THE FREQUENCY -- the two probes agree most of the time, because
    the search moves in powers of two while the probes differ by only ~1.19x, so just the
    log2(1.19) ~ 25% of card sizes landing in that band change rung at all (measured 28%
    over two octaves). Worth stating plainly: this fix removes a failure mode, it does not
    usually change the number.
    """
    rng = np.random.default_rng(4)
    ceilings = np.geomspace(0.8e6, 1.6e6, 24)  # just over an octave
    differ = 0
    for ceiling in ceilings:
        built = _run_search(monkeypatch, LENGTHS, float(ceiling))
        typical = _run_search(monkeypatch, LENGTHS, float(ceiling), sigmas=0.0)
        assert built <= typical, f"ceiling {ceiling:.2e}: chose a LARGER batch ({built}>{typical})"
        assert _worst_batch_of_a_run(rng, built) <= ceiling, (
            f"ceiling {ceiling:.2e}: chosen bs={built} does not survive the run"
        )
        differ += built != typical
    fraction = differ / len(ceilings)
    assert 0.1 <= fraction <= 0.45, f"{fraction:.0%} of card sizes changed answer"


def test_search_falls_back_to_a_random_batch_without_lengths(monkeypatch, caplog):
    """Iterable datasets (JetClass, TopTagXL) must still work, and must SAY they fell back."""
    exp_lengths = LENGTHS
    with caplog.at_level("INFO"):
        chosen = _run_search(monkeypatch, exp_lengths, 1.35e6, sigmas=PROBE_SIGMAS)
    assert any("constructed at +5" in r.message for r in caplog.records)

    caplog.clear()
    monkeypatch.setattr("utils.find_lr._probe_lengths", lambda ds: None)
    with caplog.at_level("INFO"):
        fallback = _run_search(monkeypatch, exp_lengths, 1.35e6)
    assert any("ONE RANDOM BATCH" in r.message for r in caplog.records)
    assert fallback >= chosen  # a median probe is lighter, so it over-shoots


@pytest.mark.parametrize("ceiling", [0.9e6, 1.1e6, 1.5e6, 2.4e6, 3.9e6])
def test_refine_recovers_the_octave_without_ever_overshooting(monkeypatch, ceiling):
    """REFINE: the discarded octave is the bigger number, so it must be exactly right.

    Two properties, and the second is the one that could ruin a campaign: refining must
    never return a batch that does not fit. The bisection only ever accepts a size it has
    actually probed, so the returned size is always one the card ran a full training step at.
    """
    plain = _run_search(monkeypatch, LENGTHS, ceiling)
    fine = _run_search(monkeypatch, LENGTHS, ceiling, refine=True)

    assert fine >= plain, "refining must never LOSE batch size"
    assert fine < 2 * plain, "refining must stay inside the bracketed octave"

    exp = _FakeExp(LENGTHS, ceiling)  # the returned size must survive its own probe batch
    exp.cfg.training.batchsize = fine
    exp._init_dataloader()
    exp._batch_loss(_worst_case_batch(exp, fine, LENGTHS, PROBE_SIGMAS))


def test_refine_gains_are_worth_the_three_extra_probes(monkeypatch):
    """REFINE: pins the average gain, so a regression that quietly no-ops is visible."""
    gains = []
    for ceiling in np.geomspace(0.9e6, 1.8e6, 12):
        plain = _run_search(monkeypatch, LENGTHS, float(ceiling))
        gains.append(_run_search(monkeypatch, LENGTHS, float(ceiling), refine=True) / plain)
    assert 1.2 <= float(np.mean(gains)) <= 1.6, f"mean gain {np.mean(gains):.2f}x"


def test_search_honours_bs_safety_on_top(monkeypatch):
    """bs_safety still works -- it is the only lever left on datasets with no lengths."""
    full = _run_search(monkeypatch, LENGTHS, 1.35e6)
    half = _run_search(monkeypatch, LENGTHS, 1.35e6, safety=0.5)
    assert half == max(16, full // 2)


def test_probe_lengths_and_collate_on_the_real_dataset(tmp_path):
    """COLLATE: the constructed index list must survive the real dataset + real collater."""
    pyg = pytest.importorskip("torch_geometric")
    from experiments.tagging.dataset import TopTaggingDataset

    rng = np.random.default_rng(2)
    n_jets, pad = 4000, 200
    kin = np.zeros((n_jets, pad, 4), dtype=np.float32)
    lengths = np.clip(rng.gamma(8.2, 6.0, size=n_jets), 4, 135).astype(int)
    for i, n in enumerate(lengths):
        kin[i, :n] = rng.normal(size=(n, 4)).astype(np.float32) + 1.0
    path = tmp_path / "mini.npz"
    np.savez(path, kinematics_train=kin, labels_train=(rng.random(n_jets) > 0.5))

    ds = TopTaggingDataset()
    ds.load_data(str(path), "train")

    got = _probe_lengths(ds)
    assert got is not None and np.array_equal(got, lengths)
    assert _probe_lengths(ds) is got  # cached: the search calls this once per rung

    bs = 128
    idx = _worst_case_indices(got, bs)
    batch = pyg.loader.DataLoader(ds, batch_size=bs).collate_fn([ds[int(i)] for i in idx])
    assert batch.ptr.shape == (bs + 1,)  # what _extract_batch reads
    assert batch.label.shape == (bs,)
    assert batch.x.shape == (int(got[idx].sum()), 4)
    assert torch.diff(batch.ptr).max().item() == int(got.max())
