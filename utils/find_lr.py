"""
    python utils/find_lr.py -cp config -cn jctagging model=tag_transformer save=false
    python utils/find_lr.py -cp config -cn toptagging model=tag_transformer save=false
    python utils/find_lr.py -cp config -cn jctagging model=tag_transformer model/framesnet=learnedpd save=false

The task is selected with `-cn` (toptagging / jctagging / amplitudes / ttbar / ...)
and the dataset within a task with the usual data overrides (e.g. `data.dataset=mini`
for top tagging); the sweep simply cycles that task's training dataloader, so a
larger `+lr_find.num_iter` samples more of the data.

The recommended learning rate is reported as `loss-min / 10` (a robust peak lr for
an annealed / one-cycle schedule); the steepest-descent point is also printed.

Prefer `loss-min / 10`: it is stable in `num_iter` because it tracks the
edge-of-divergence lr that the loss landscape fixes. The steepest-descent point is
NOT -- a longer sweep lets ordinary training progress (not the lr) dominate the loss
drop and biases it toward low lr (davidtvs/pytorch-lr-finder#68). Keep `num_iter`
short (300 is deliberate); if a suggestion looks unstable, lower it, don't raise it.

It reuses the experiment's own `_batch_loss`, optimizer, scaler and dataloader,
so the measured loss-vs-lr curve reflects the lr-scale-determining setup: optimizer
type, betas/eps, param-group ratios (`lr_factor_framesnet`), batchsize and amp. Two
regularizers are deliberately switched off for the sweep -- `weight_decay` (inert over
~300 steps) and gradient clipping (it would cap the step and mask the high-lr
divergence the test needs) -- so the curve is the raw loss-vs-lr response. The base
`training.lr` is *ignored*; only the inter-group lr ratios are preserved.

The optimizer (type/betas) and param-groups come from the chosen *training* config; the task
defaults (e.g. `toptagging`) now select `tag_gts_and_friends_default` (AdamW), so
the GT hybrids sweep correctly with just `model=...`. Because the suggested lr is
optimizer-specific, pass `training=top_<baseline>` to sweep a baseline under its own
optimizer (e.g. `training=top_transformer` for the Lion transformer).
 
The test follows the Leslie-Smith / fastai recipe: exponentially ramp the lr over
a few hundred batches, record an EMA-smoothed training loss, stop early if the
loss diverges, then report `loss-min / 10` (steepest-descent is printed too, but see
the num_iter caveat above).
 
Pass `save=false` so no run directory is created. Tune the sweep on the CLI under
`lr_find.*` (these keys are not in the base config, so the `+` prefix is REQUIRED --
a bare `lr_find.num_iter=…` raises a Hydra ConfigCompositionException):
 
    +lr_find.start_lr=1e-7   lowest lr in the sweep                 (default 1e-7)
    +lr_find.end_lr=1e1      highest lr in the sweep                (default 1e1)
    +lr_find.num_iter=300    number of batches in the sweep         (default 300)
                             (dataset-size independent: 300 is right for JetClass too --
                             the sweep samples the lr ramp, not the dataset, and raising
                             num_iter biases the suggestion low, see the note above)
    +lr_find.beta=0.98       EMA factor for loss smoothing          (default 0.98)
    +lr_find.diverge=5.0     stop when smoothed_loss > diverge*best (default 5.0)
    +lr_find.skip_start=10    points dropped from the start          (default 10)
    +lr_find.skip_end=5      points dropped from the end            (default 5)
    +lr_find.output=lr_finder.png   plot path                       (default lr_finder.png)
    +lr_find.force_knn_metric=deltaR  pin the kNN metric for models that have one (default
                                    'deltaR'; 'keep' uses the model's own, 'minkowski' sweeps that).
                                    The lr scale is metric-independent, so the suggestion still
                                    transfers to a run trained under the model's configured metric.

On a GPU you can also auto-size the batch first, then sweep the lr at that size
(so the suggested lr matches the regime you will train in):

    +lr_find.find_batch_size=true   double the batchsize until CUDA OOM (full step)
    +lr_find.bs_start=16            smallest batchsize tried               (default 16)
    +lr_find.bs_max=16384           largest batchsize tried                (default 16384)
    +lr_find.bs_sigmas=5.0          how heavy the probe batch is, in sd of the batch
                                    total  (default 5.0 ~ the worst batch of any run
                                    length you will actually launch; 0 probes a typical
                                    batch, which is what a random draw gives you)
    +lr_find.bs_refine=true         bisect the octave the doubling search throws away
                                    (default false; 3 more probes, and the answer is no
                                    longer a power of two. Worth up to 2x, ~1.5x on
                                    average -- more than the probe fix below is worth --
                                    but the campaign's convention is a round batch that
                                    is comparable across recipes, hence off by default)
    +lr_find.bs_safety=1.0          fraction of the largest fit to use     (default 1.0,
                                    i.e. the largest fitting power of two; <1 adds
                                    headroom but breaks the power of two)

The probe batch is CONSTRUCTED from the dataset's own jet lengths, not drawn: it sits
`bs_sigmas` sd above a typical batch's total size, with P_max at the dataset maximum.
A random draw is a MEDIAN batch, and a long run meets the worst of ~10^5 draws, which
is what used to make `bs_safety<1` necessary -- see PROBE_SIGMAS in this file for the
measured gap. On datasets with no cheap per-item lengths (JetClass, TopTagXL) the probe
falls back to one random batch and says so in the log; there `bs_safety` still applies.

e.g.  python utils/find_lr.py -cp config -cn toptagging model=tag_LorentzNetLGATrSlimGraphGPS \\
          save=false +lr_find.find_batch_size=true
prints both the GPU-fit batchsize and the suggested lr. (On CPU the batch-size
search is a no-op.) It still measures one step rather than a trajectory, so run it
under `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` -- the same setting the run
itself should use -- and verify the printed batchsize with a short real run before a
long job.
"""

import os
import sys
import time

# living in utils/, the repo root is no longer the script dir (= sys.path[0]);
# put it back so the `experiments` package resolves without needing an install
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from experiments.amplitudes.experiment import AmplitudeExperiment
from experiments.amplitudes.experimentxl import AmplitudeXLExperiment
from experiments.eventgen.processes import ttbarExperiment
from experiments.logger import LOGGER
from experiments.tagging.experiment import TopTaggingExperiment
from experiments.tagging.finetuneexperiment import TopTaggingFineTuneExperiment
from experiments.tagging.jetclassexperiment import JetClassTaggingExperiment
from experiments.tagging.toptagxlexperiment import TopTagXLExperiment

CONSTRUCTORS = {
    "toptagging": TopTaggingExperiment,
    "toptaggingft": TopTaggingFineTuneExperiment,
    "toptagxl": TopTagXLExperiment,
    "jctagging": JetClassTaggingExperiment,
    "amplitudes": AmplitudeExperiment,
    "amplitudesxl": AmplitudeXLExperiment,
    "ttbar": ttbarExperiment,
}

DEFAULTS = dict(
    start_lr=1e-7,
    end_lr=1e1,
    num_iter=300,
    beta=0.98,
    diverge=5.0,
    skip_start=10,
    skip_end=5,
    output="lr_finder.png",  # default is REPLACED below by lr_finder/<model>_bs<N>_lr<lr>.png
    outdir="lr_finder",  # where auto-named plots + npz land (created if absent)
    # pin the kNN graph metric for the sweep when the model exposes a `knn_metric` choice
    # (no-op for models without one). The lr SCALE is fixed by the optimizer/batchsize/amp,
    # not the graph metric, so pinning one metric only makes the suggested lr comparable
    # across models -- the model still TRAINS under its own configured metric. 'deltaR' is the
    # default pin (every model can build it); set 'keep' to sweep each model's own metric.
    force_knn_metric="deltaR",
    # optional GPU batch-size search (CUDA only; no-op on CPU)
    find_batch_size=False,
    bs_start=16,  # smallest batchsize tried
    bs_max=16384,  # largest batchsize tried
    bs_sigmas=5.0,  # how heavy the constructed probe batch is, in sd of the batch total
    bs_refine=False,  # bisect the octave the doubling search discards (3 more probes, no power of two)
    bs_safety=1.0,  # fraction of the largest fitting batchsize to use (1.0 keeps a power of two)
)


def build_experiment(cfg):
    """Construct and partially initialize an experiment (no scheduler, no training).

    Mirrors BaseExperiment.full_run() up to the point where the optimizer and
    scaler exist, which is all the range test needs. Matches the init sequence
    used in the equivariance tests.
    """
    try:
        constructor = CONSTRUCTORS[cfg.exp_type]
    except KeyError as err:
        raise ValueError(f"exp_type {cfg.exp_type} not implemented") from err

    exp = constructor(cfg, rank=0, world_size=1)
    exp._init()  # device, logger, (run dir is skipped when save=false)
    exp.init_physics()  # wire data dims -> model dims
    exp.init_model()
    exp.init_data()
    exp._init_dataloader()
    exp._init_loss()
    exp._init_optimizer()
    exp._init_scaler()

    exp.model.to(exp.device)
    return exp


def _is_oom(err):
    return isinstance(err, torch.cuda.OutOfMemoryError) or "out of memory" in str(err).lower()


# --- the probe batch ---------------------------------------------------------
#
# The batch-size search used to probe each rung with ONE random batch --
# next(iter(train_loader)) -- which is a MEDIAN batch, while a real run sees the worst
# of ~10^5 draws. Jets vary in length (top-tagging: mean 49.2, sd 17.2, max 135), and
# memory follows the batch's TOTAL size: dense terms in B*P_max, pair masks in
# B*P_max^2, edge/attention terms in sum_j n_j^2, kNN in k*sum_j n_j. Measured over the
# real length distribution, the worst batch of a 50-epoch run carries 1.31x (B=128) to
# 1.05x (B=4096) the median's sum n^2, and P_max at the dataset cap rather than the
# ~p50 of a batch maximum. That gap is what `bs_safety=0.5` was covering -- by halving
# the batch, i.e. ~2x headroom for a <=1.3x problem.
#
# So build the probe batch to BE the worst batch instead. `_worst_case_indices` picks
# `bs` real jets that sit `sigmas` sd above a typical batch in BOTH `sum n` and
# `sum n^2`, and that include the longest jets in the dataset so P_max hits its cap too.
# One probe per rung (same cost as before), deterministic (so a 25-recipe sweep is
# reproducible), and it MEASURES the peak rather than modelling it -- which is what keeps
# it correct whatever a given model's memory actually scales with, as long as memory is
# non-decreasing in each jet's length. Every net here qualifies.
#
# Constructed batch vs the heaviest batch of a DIRECTLY SIMULATED 50-epoch run (every one
# of its 473k / 118k / 15k batches drawn), each relative to a random batch's median:
#
#             ------ sum n ------      ----- sum n^2 -----
#     B       built   run's worst      built   run's worst
#     128     1.165      1.150         1.428      1.299
#     512     1.078      1.063         1.187      1.144
#    4096     1.029      1.024         1.053      1.045
#
# P_max needs no such comparison: the longest jet in the dataset is in every constructed
# batch, and no batch can exceed that. (An earlier version targeted `sum n^2` only and
# came in 4% UNDER the run's worst `sum n` at B=128 -- the entire margin for a model whose
# memory is linear in total nodes. Both moments, or neither.)
#
# Iterable datasets (JetClass, TopTagXL) do not expose per-item lengths cheaply; there
# the old random-batch probe still applies and `bs_safety` is still the only lever.
PROBE_SIGMAS = 5.0


def _probe_lengths(dataset):
    """Per-item constituent counts, cached on the dataset. None if not cheaply available."""
    if dataset is None:
        return None
    cached = getattr(dataset, "_probe_lengths", None)
    if cached is not None:
        return cached
    data_list = getattr(dataset, "data_list", None)  # map-style TaggingDataset only
    if data_list is None or len(data_list) == 0:
        return None
    try:
        lengths = np.fromiter(
            (int(d.x.shape[0]) for d in data_list), dtype=np.int64, count=len(data_list)
        )
    except (AttributeError, TypeError, ValueError, IndexError):
        return None
    try:
        dataset._probe_lengths = lengths
    except AttributeError:  # __slots__ dataset: recompute per rung rather than fail
        pass
    return lengths


def _worst_case_indices(lengths, bs, sigmas=PROBE_SIGMAS):
    """Indices of `bs` items forming an unusually -- but realistically -- heavy batch.

    Start from `bs` items spread evenly over the quantiles of the length distribution
    (that alone reproduces a typical batch's totals by quadrature), then swap the `k`
    smallest picks for the `k` longest items in the dataset. `k` is the smallest value
    putting the batch `sigmas` sd above typical -- i.e. the total a run of order
    ``exp(sigmas^2 / 2)`` batches expects to see once -- for EVERY memory-relevant
    statistic at once:

        sum n     node features, kNN edge lists       (concentrates least slowly)
        sum n^2   dense edge tensors, block attention (the CGENN family's dominant term)
        P_max     padded/dense terms in B*P_max(^2)   (ParT-style attention)

    ``sum n^p`` is non-decreasing in `k` for both p, so each gets its own smallest
    sufficient `k` and the batch takes the larger -- targeting only `sum n^2` leaves
    `sum n` about 4% short at B=128, which is the whole margin for a kNN model. P_max
    needs no target: `k` is clamped to >= 1, so the dataset's longest jet is always in,
    which is the exact maximum no batch can exceed. The other clamp, `k <= bs//4`, keeps
    the grid and the longest-item pool disjoint so no item is ever repeated.

    Returns None when the batch cannot be built from this dataset.
    """
    n = np.asarray(lengths, dtype=np.int64)
    size, bs = n.size, int(bs)
    kmax = max(1, bs // 4)
    if bs < 1 or size < bs + kmax:
        return None

    order = np.argsort(n, kind="stable")
    asc = n[order].astype(np.float64)  # ascending
    grid = ((np.arange(bs) + 0.5) * (size - kmax) / bs).astype(np.int64)  # excludes the top kmax
    top = np.arange(size - kmax, size)

    k, shortfall = 1, None
    for power in (1, 2):
        moment = asc**power
        target = bs * moment.mean() + sigmas * np.sqrt(bs) * moment.std()
        gain = moment[top[::-1]] - moment[grid[:kmax]]  # gain[j]: swap in the (j+1)-th longest
        if (gain < 0).any():  # impossible for a sorted array; guards searchsorted's contract
            return None
        totals = moment[grid].sum() + np.concatenate(([0.0], np.cumsum(gain)))
        need = int(np.searchsorted(totals, target))
        if need > kmax:  # a length distribution too spread for bs//4 swaps to cover
            ratio = totals[kmax] / target
            shortfall = ratio if shortfall is None else min(shortfall, ratio)
        k = max(k, min(need, kmax))

    if shortfall is not None:
        LOGGER.warning(
            f"  probe batch for bs={bs} reaches {shortfall:.0%} of the +{sigmas:g} sd "
            f"target ({kmax} longest jets is the cap); keep some bs_safety."
        )
    return np.concatenate([order[grid[k:]], order[top[kmax - k :]]])


def _worst_case_batch(exp, bs, lengths, sigmas=PROBE_SIGMAS):
    """Collate the worst-case batch for `bs`, or None to fall back to a random one."""
    idx = _worst_case_indices(lengths, bs, sigmas)
    if idx is None:
        return None
    collate = getattr(exp.train_loader, "collate_fn", None)
    if collate is None:
        return None
    try:
        return collate([exp.data_train[int(i)] for i in idx])
    except Exception as err:  # any dataset/collate mismatch -> the random-batch path
        LOGGER.warning(f"  worst-case probe batch could not be collated ({err}); using a random one.")
        return None


def find_max_batch_size(exp, start, max_cap, safety, sigmas=PROBE_SIGMAS, refine=False):
    """Doubling search for the largest batchsize that survives a full training step.

    At each candidate it runs a real fwd + bwd + optimizer step (scaler + gradient
    clipping as in real training -- unlike ``range_test``, which deliberately omits
    clipping so it does not mask divergence), so the measured memory reflects what
    training actually uses -- not just a fwd+bwd lower bound. The search doubles
    until a CUDA OOM, so the largest size that fits is a power of two; with the
    default ``safety=1.0`` that power of two is returned unchanged.

    The probe batch is CONSTRUCTED, not drawn: `sigmas` sd above a typical batch's
    total size, with P_max at the dataset maximum, so it stands in for the worst batch
    of a long run rather than a median one (see PROBE_SIGMAS above for the numbers).
    That is what makes ``safety=1.0`` the right default -- a fractional ``safety`` is
    now only for datasets whose lengths this cannot see (JetClass, TopTagXL), where the
    probe falls back to one random batch and the log line says so.

    ``refine`` (off by default) spends 3 more probes bisecting the octave the doubling
    search discards, and in BATCH terms that octave is the bigger number here. Measured over
    60 simulated card sizes spanning two octaves: refining gains a mean 1.35x batch (median
    1.25x, up to 1.88x), while the constructed probe changes the chosen rung at all in only
    28% of cases -- the rest of the time its 1.05-1.3x is swallowed by the power-of-two
    granularity. End to end the new default is 0.82x the old default's batch (the price of
    not dying at step 10) and 1.08x with ``refine`` on; against the ``bs_safety=0.5`` the
    docs used to recommend for CGENN, refined is 2.16x. Refining is off by default because a
    power of two is the campaign's convention -- comparable across recipes -- not because it
    is unsound: the bisection only ever returns a size it has run a full training step at.

    BATCH IS NOT THROUGHPUT, and none of those ratios are speedups. jets/s = batchsize /
    step time saturates once the card is compute-bound, and past that point a bigger batch
    buys nothing. Which regime a given model is in is a measurement nobody here had made --
    so this function now times the second step at every rung and prints the jets/s curve
    with the result. Read it before deciding that the largest batch is the one you want.

    CUDA only (returns the configured batchsize on CPU). The optimizer step mutates
    the model + optimizer state, so the caller MUST re-initialise before the lr sweep.

    NOTE: it probes one batch per size, so it still measures a single step, not a
    trajectory -- fragmentation grows with run length. Pair it with
    ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` and verify the chosen
    batchsize with a short real run before launching a multi-day job.
    """
    if not torch.cuda.is_available():
        LOGGER.info("No CUDA device -> skipping batch-size search (keeping configured batchsize).")
        return int(exp.cfg.training.batchsize)

    exp.model.train()
    lengths = _probe_lengths(getattr(exp, "data_train", None))
    state = {"warned": False, "rates": {}}

    def fits(bs):
        """TWO full training steps at `bs`: True if both fit, False on a CUDA OOM.

        Two, not one, for two reasons. The second is TIMED -- the first pays for allocator
        growth, cudnn/inductor autotune and lazily-created optimizer state, so it says
        nothing about steady-state speed. And running a second step is itself a better
        memory probe: a rung that survives one step and dies on an identical repeat is a
        rung that does not fit, and the old single-step probe would have called it OK.
        """
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            with open_dict(exp.cfg):
                exp.cfg.training.batchsize = bs
            exp._init_dataloader()
            data = None if lengths is None else _worst_case_batch(exp, bs, lengths, sigmas)
            if data is None:
                if lengths is not None and not state["warned"]:
                    LOGGER.warning(
                        f"  batchsize {bs}: no worst-case batch constructible from "
                        f"{lengths.size} jets -> falling back to a random batch."
                    )
                    state["warned"] = True
                data = next(iter(exp.train_loader))

            def step():
                loss, _ = exp._batch_loss(data)
                exp.optimizer.zero_grad(set_to_none=True)
                exp.scaler.scale(loss).backward()
                exp.scaler.unscale_(exp.optimizer)
                if exp.cfg.training.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        exp.model.parameters(),
                        exp.cfg.training.clip_grad_norm,
                        error_if_nonfinite=False,
                    )
                exp.scaler.step(exp.optimizer)
                exp.scaler.update()

            step()  # warm-up: allocator growth, autotune, first-step optimizer state
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            step()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0

            peak = torch.cuda.max_memory_allocated() / 1e9
            rate = bs / dt if dt > 0 else float("nan")
            state["rates"][bs] = rate
            LOGGER.info(f"  batchsize {bs:6d}: OK  (peak {peak:.1f} GB, {rate:8.0f} jets/s)")
            return True
        except RuntimeError as err:
            if not _is_oom(err):
                raise
            LOGGER.info(f"  batchsize {bs:6d}: OOM")
            torch.cuda.empty_cache()
            return False

    LOGGER.info("Searching for the largest batchsize that fits a full training step:")
    if lengths is None:
        LOGGER.info(
            "  probe batch: ONE RANDOM BATCH (this dataset does not expose per-item "
            f"lengths) -- a median batch, so keep headroom via bs_safety (now {safety})."
        )
    else:
        LOGGER.info(
            f"  probe batch: constructed at +{sigmas:g} sd of the batch total, P_max="
            f"{int(lengths.max())} (dataset max) -- stands in for the worst batch of a long run."
        )

    last_ok, oom_at, bs = None, None, int(start)
    while bs <= max_cap:
        if not fits(bs):
            oom_at = bs
            break
        last_ok = bs
        bs *= 2

    if last_ok is None:
        LOGGER.warning(f"Even batchsize {start} does not fit; keeping {start}.")
        return int(start)

    refined = False
    if refine and oom_at is not None:
        # The doubling search brackets the ceiling in [last_ok, oom_at) and then discards
        # the whole interval -- so it returns, on average, 1/1.5 of what the card holds.
        # Bisect it back in steps of last_ok//8: 3 extra probes for up to 2x the batch.
        lo, hi, step = last_ok, oom_at, max(8, last_ok // 8)
        while hi - lo > step:
            mid = (lo + hi) // 2 // step * step
            if not lo < mid < hi:
                break
            if fits(mid):
                lo = mid
            else:
                hi = mid
        refined, last_ok = lo > last_ok, lo

    chosen = max(int(start), int(last_ok * safety))
    if safety >= 1.0:
        note = "the largest fit, refined off the power of two" if refined else "the largest fitting power of two"
    else:
        note = f"{safety:.0%} of {last_ok}"
        if lengths is not None:
            note += " -- the probe was already the worst-case batch, so this stacks on top"
    LOGGER.info(f"Largest fitting batchsize {last_ok} -> using {chosen} ({note}).")

    # LARGEST is not the same question as FASTEST, and this repo has never checked which.
    # jets/s = batchsize / step time saturates once the card is compute-bound; past that
    # point a bigger batch buys nothing but risk, and the whole doubling search is chasing
    # a number that stopped mattering several rungs ago. The steps have already been run,
    # so the curve is free -- report it and let the reader decide. Single timed steps, so
    # treat differences under ~10% as noise.
    rates = {bs: r for bs, r in sorted(state["rates"].items()) if r == r}
    if len(rates) >= 2:
        best = max(rates, key=rates.get)
        LOGGER.info(f"  jets/s by batchsize: " + "  ".join(f"{bs}:{r:.0f}" for bs, r in rates.items()))
        # `chosen` can be absent from `rates` only if its timing came back non-finite; say
        # so rather than half-guard it, since this runs after an expensive search.
        here = rates.get(last_ok)
        if here is None:
            LOGGER.info(f"  (no usable timing at the chosen batchsize {last_ok}; curve above only)")
        elif best != last_ok and rates[best] > 1.1 * here:
            LOGGER.info(
                f"  NOTE: throughput peaks at batchsize {best} ({rates[best]:.0f} jets/s), "
                f"{rates[best] / here:.2f}x the largest fit's {here:.0f}. "
                f"The largest batch that FITS is not the fastest one here."
            )
        else:
            LOGGER.info(
                f"  throughput is still rising (or flat) at the largest fit -- "
                f"{here:.0f} jets/s, the best measured."
            )
    return chosen


def _cycle(loader):
    while True:
        yield from loader


def range_test(exp, start_lr, end_lr, num_iter, beta, diverge):
    """Exponentially ramp the lr and record the EMA-smoothed training loss."""
    optimizer, scaler = exp.optimizer, exp.scaler

    # preserve the relative lr ratios between param groups (net vs framesnet vs ...)
    base_lr0 = optimizer.param_groups[0]["lr"]
    base_ratios = [pg["lr"] / base_lr0 for pg in optimizer.param_groups]
    gamma = (end_lr / start_lr) ** (1.0 / max(1, num_iter - 1))

    # The range test measures the RAW loss-vs-lr response, so neutralize the two training
    # regularizers that would only distort it (neither sets the lr SCALE -- that is the
    # optimizer type + batchsize): weight_decay is inert over a ~300-step sweep (the
    # weight-norm equilibrium needs thousands of steps), and grad clipping would cap the
    # step and can mask the high-lr divergence the test must see (a no-op for AdamW/Lion,
    # which renormalize, but not for plain SGD -> off keeps the test optimizer-agnostic).
    for pg in optimizer.param_groups:
        pg["weight_decay"] = 0.0

    lrs, losses = [], []
    avg_loss, best_loss = 0.0, float("inf")

    exp.model.train()
    iterator = iter(_cycle(exp.train_loader))
    log_every = max(1, num_iter // 20)

    for step in range(num_iter):
        lr = start_lr * gamma**step
        for pg, ratio in zip(optimizer.param_groups, base_ratios):
            pg["lr"] = lr * ratio

        data = next(iterator)
        loss, _ = exp._batch_loss(data)

        # update step -- NO grad clipping on purpose: it would cap the step and hide the
        # high-lr divergence the test needs to graph (see the note above the loop).
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_value = loss.detach().item()
        if not np.isfinite(loss_value):
            LOGGER.warning(f"Non-finite loss at step {step} (lr={lr:.2e}); stopping.")
            break

        # bias-corrected EMA of the loss
        avg_loss = beta * avg_loss + (1.0 - beta) * loss_value
        smoothed = avg_loss / (1.0 - beta ** (step + 1))
        lrs.append(lr)
        losses.append(smoothed)

        if step == 0 or smoothed < best_loss:
            best_loss = smoothed
        if smoothed > diverge * best_loss:
            LOGGER.info(f"Loss diverged at step {step} (lr={lr:.2e}); stopping early.")
            break

        if step % log_every == 0:
            LOGGER.info(f"step {step:4d}  lr={lr:.2e}  smoothed_loss={smoothed:.4f}")

    return np.array(lrs), np.array(losses)


def suggest_lr(lrs, losses, skip_start, skip_end, beta=0.98):
    """Two heuristics: `loss-min/10` (robust, the recommended one) and the
    steepest-descent point (lr at the minimum gradient of loss vs log(lr)).

    The EMA used to smooth the loss leaves a high-variance transient over its
    ~1/(1-beta) warmup window; if the gradient search sees it, the "steepest"
    point collapses onto that early dip (e.g. ~1e-7). We therefore skip the
    warmup window before searching for the steepest point. `loss-min` is taken
    over the whole trimmed curve since it is unaffected by the warmup.
    """
    n = len(lrs)
    if n <= skip_start + skip_end + 2:
        skip_start, skip_end = 0, 0
    end = n - skip_end
    lr_trim = lrs[skip_start:end]
    loss_trim = losses[skip_start:end]

    # steepest descent, ignoring the EMA warmup transient AND everything at or past the loss
    # minimum. Past the minimum the curve is rising into divergence, where the per-step loss is
    # chaotic: a single downward blip between adjacent steps at large lr yields a gradient far
    # more negative than anything in the honest falling region, and argmin(gradients) takes it.
    # Measured (ParticleNet, top tagging): an unrestricted search returned 2.42e+00 on a curve
    # whose real steepest descent is 1.32e-3 -- the same value the other reruns found. skip_end
    # alone does not remove this; divergence detection stops only a few steps into the blow-up.
    warmup = min(int(round(1.0 / (1.0 - beta))), max(0, (end - skip_start) // 3))
    grad_start = skip_start + warmup
    argmin_full = grad_start + int(np.argmin(losses[grad_start:end])) if end > grad_start else end
    grad_end = max(grad_start + 2, argmin_full + 1)  # inclusive of the minimum, never past it
    lr_grad = lrs[grad_start:grad_end]
    loss_grad = losses[grad_start:grad_end]
    if len(lr_grad) >= 2:
        gradients = np.gradient(loss_grad, np.log(lr_grad))
        steepest = float(lr_grad[int(np.argmin(gradients))])
    else:
        steepest = float(lr_trim[int(np.argmin(loss_trim))])

    argmin = int(np.argmin(loss_trim))
    min_loss_lr = float(lr_trim[argmin])

    # Is the minimum INTERIOR, or is the loss still falling when the sweep ends?
    # `loss-min/10` assumes a well-defined trough before divergence. On models whose loss
    # decreases monotonically into the blow-up (measured: ParticleNet on top tagging) the
    # argmin instead lands in the last few points, where run-to-run noise of ~1e-4 in the
    # smoothed loss flips it between grid points a factor ~2.5 apart -- three unseeded runs
    # gave 1.39e-1 / 3.82e-2 / 2.64e-2, a 5x spread, against a published 1e-2. The
    # steepest-descent point sits in the well-conditioned part of the same curves and gave
    # 1.91e-3 / 1.32e-3 / 1.91e-3, a 1.4x spread. So: when the minimum is not interior, the
    # loss-min heuristic is inapplicable, not merely noisy.
    tail = max(3, int(round(0.15 * len(loss_trim))))
    interior_min = argmin < len(loss_trim) - tail
    return steepest, min_loss_lr, lr_trim, loss_trim, interior_min


def make_plot(lrs, losses, steepest, min_loss_lr, output, title="LR range test"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(lrs, losses, color="#1f77b4")
    ax.axvline(steepest, color="#d62728", ls="--", label=f"steepest: {steepest:.2e}")
    ax.axvline(
        min_loss_lr / 10.0,
        color="#2ca02c",
        ls=":",
        label=f"loss-min / 10: {min_loss_lr / 10.0:.2e}",
    )
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("smoothed training loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    LOGGER.info(f"Saved LR-finder plot to {os.path.abspath(output)}")


# Default to the real config/ tree so a bare run matches training (clipping on, full data);
# pass `-cp config_quick ... data.dataset=mini` for a fast smoke test of the finder itself.
@hydra.main(config_path="../config", config_name="toptagging", version_base=None)
def main(cfg):
    # LR finding is single-process and never trains / evaluates / saves a model.
    # `save` is forced here rather than left to the CLI: a forgotten `save=false` would mint a
    # run directory that looks like a training run in `runs/` -- discoverable by a later
    # `-cn config` warm start and by anyone reading the tree -- for a process that never
    # trained. The flag stays accepted on the CLI (harmless, and the docs still show it).
    cfg.train = False
    cfg.evaluate = False
    cfg.plot = False
    cfg.save = False

    params = dict(DEFAULTS)
    lr_find = OmegaConf.select(cfg, "lr_find", default=None)
    if lr_find is not None:
        overrides = OmegaConf.to_container(lr_find, resolve=True)
        params.update({k: v for k, v in overrides.items() if v is not None})

    # Pin the kNN graph metric for the sweep when the model exposes one. GT/hybrid models carry
    # `model.net.knn_metric` ('deltaR' = eta-phi L2, 'minkowski' = Lorentz interval); models
    # without it (ParT, transformer, ...) are untouched. The lr SCALE is set by the optimizer +
    # batchsize + amp, not the graph metric, so pinning one metric only makes the suggested lr
    # comparable across models -- the model still trains under its own configured metric. The
    # wrappers always build the (eta, phi) points, so pinning 'deltaR' is input-safe. Pass
    # +lr_find.force_knn_metric=keep to sweep each model's own metric (or =minkowski to pin that).
    forced_metric = params["force_knn_metric"]
    if isinstance(forced_metric, str) and forced_metric.lower() in ("keep", "none", "off", ""):
        forced_metric = None
    current_metric = OmegaConf.select(cfg, "model.net.knn_metric", default=None)
    if forced_metric is not None and current_metric is not None and current_metric != forced_metric:
        with open_dict(cfg):
            cfg.model.net.knn_metric = forced_metric
        LOGGER.info(
            f"LR sweep: forcing model.net.knn_metric '{current_metric}' -> '{forced_metric}' "
            f"(metric-independent lr scale; +lr_find.force_knn_metric=keep to retain '{current_metric}')."
        )

    exp = build_experiment(cfg)

    # optional: size the batch to the GPU before the lr sweep, then sweep at that
    # batchsize so the suggested lr matches the regime you will actually train in
    #
    # COMPILE IS LEFT TO THE YAML HERE, DELIBERATELY, and that is the OPPOSITE of what
    # utils/bperf.py does -- it forces `compile=false` before its identical call to
    # find_max_batch_size. Both are right for their own job and the difference is not drift:
    #
    #   bperf needs ONE batch that BOTH states can run, because its whole output is a paired
    #   eager-vs-compiled ratio; sizing per state would compare two different batch sizes.
    #   It also does every row's search in ONE process, so sizing compiled would spend an
    #   inductor build per row and leave dynamo cache entries holding each model alive for
    #   the next row's search.
    #
    #   find_lr has no second state and no shared process. Its job is the batch you will
    #   TRAIN at, and EVERY model config carrying a `compile` key ships it TRUE (20 of the 36
    #   in config/model; no config anywhere says false), so the shipped posture IS the thing
    #   to measure. Sizing eager here would hand the campaign a batch chosen under a memory
    #   profile it never runs at.
    #
    # Consequence worth carrying: bperf's it/s numbers are therefore taken at a batch nobody
    # trains at. Fine for the speedup RATIO it reports; not a basis for ranking impls by
    # jets/s, which is what this function's number decides.
    if params["find_batch_size"]:
        bs = find_max_batch_size(
            exp,
            params["bs_start"],
            params["bs_max"],
            params["bs_safety"],
            sigmas=float(params["bs_sigmas"]),
            refine=bool(params["bs_refine"]),
        )
        with open_dict(cfg):
            cfg.training.batchsize = bs
        # the search ran optimizer steps -> rebuild a clean model/optimizer/scaler so
        # the lr sweep starts from a fresh init (as real training would)
        exp.init_model()
        exp._init_optimizer()
        exp._init_scaler()
        exp.model.to(exp.device)
        exp._init_dataloader()

    model_name = (OmegaConf.select(cfg, "model.net._target_", default="") or "").rsplit(".", 1)[
        -1
    ] or "model"

    LOGGER.info(
        f"Running LR range test: {params['start_lr']:.1e} -> {params['end_lr']:.1e} "
        f"over <= {params['num_iter']} batches (batchsize={cfg.training.batchsize})"
    )

    lrs, losses = range_test(
        exp,
        start_lr=params["start_lr"],
        end_lr=params["end_lr"],
        num_iter=params["num_iter"],
        beta=params["beta"],
        diverge=params["diverge"],
    )
    if len(lrs) < 3:
        LOGGER.error(
            "Not enough points collected before divergence; "
            "try a smaller start_lr, a larger diverge, or more num_iter."
        )
        return

    steepest, min_loss_lr, _, _, interior_min = suggest_lr(
        lrs,
        losses,
        skip_start=params["skip_start"],
        skip_end=params["skip_end"],
        beta=params["beta"],
    )
    # loss-min/10 is the robust recommendation (peak lr for an annealed schedule);
    # the steepest-descent point is reported as a usually-similar lower bound.
    suggested = min_loss_lr / 10.0
    bs = cfg.training.batchsize

    # Auto-name the artifacts after MODEL + BATCHSIZE + SUGGESTED LR, in their own directory.
    # A chained family sweep writes 8 plots in one session; a fixed name would leave one, and a
    # name without the numbers would leave you matching images to log lines by timestamp.
    if params["output"] == DEFAULTS["output"]:
        os.makedirs(params["outdir"], exist_ok=True)
        stem = f"{model_name}_bs{bs}_lr{suggested:.2e}"
        params["output"] = os.path.join(params["outdir"], f"lr_finder_{stem}.png")
    make_plot(
        lrs,
        losses,
        steepest,
        min_loss_lr,
        params["output"],
        title=f"LR range test - {model_name} (bs={bs}, suggested lr {suggested:.2e})",
    )
    np.savez(os.path.splitext(params["output"])[0] + ".npz", lr=lrs, loss=losses)

    # The recipe these numbers belong in, so a chained sweep's log says where each pair goes.
    prefix = {"toptagging": "top", "jctagging": "jc", "toptagxl": "xl"}.get(cfg.exp_type)
    recipe = f"config/training/{prefix}_{model_name}.yaml" if prefix else None
    if recipe and not os.path.isfile(recipe):
        recipe = None

    LOGGER.info("=" * 64)
    if params["find_batch_size"]:
        LOGGER.info(f"Batchsize (fit to GPU):          {bs}")
    # RECOMMEND STEEPEST-DESCENT. This reverses the original preference, on evidence: nine
    # ParticleNet/top-tagging reruns (batch 512, the published recipe) gave steepest
    # 1.32e-3 -- 1.91e-3, a 1.4x spread bracketing ParT's published 1e-3, while loss-min/10
    # gave 2.64e-2 -- 1.39e-1, a 5x spread whose every value sits ABOVE ParticleNet's
    # published 1e-2. Crucially the split held in BOTH branches of the interior test, so the
    # curve shape does not rescue loss-min/10 here: its trough is shallow and sits just
    # before divergence, where "the minimum" is only where the fall stops, not an optimum.
    # The original argument for loss-min/10 (steepest drifts low as num_iter grows,
    # davidtvs/pytorch-lr-finder#68) predicts drift with SWEEP LENGTH; at the fixed
    # num_iter=300 used here steepest is the stable one. Revisit if a hybrid disagrees --
    # this is one model on one dataset, and flipping back is this block.
    LOGGER.info(f"Suggested lr (steepest descent): {steepest:.2e}   [recommended]")
    if interior_min:
        LOGGER.info(f"Suggested lr (loss-min / 10):    {suggested:.2e}   [upper bracket]")
    else:
        LOGGER.warning(
            "No INTERIOR loss minimum: the loss was still falling when the sweep ended, so the "
            "argmin sits in the near-divergence tail. loss-min/10 is meaningless on this curve "
            "(it moves by several x between reruns); the plot will show no trough before the "
            "blow-up."
        )
        LOGGER.info(f"Suggested lr (loss-min / 10):    {suggested:.2e}   [NOT reliable here]")
    suggested = steepest
    reuse = f"training.lr={suggested:.2e}"
    if params["find_batch_size"]:
        reuse = f"training.batchsize={bs} " + reuse
    LOGGER.info(f"  ->  reuse with:  {reuse}")
    LOGGER.info(f"  ->  plot:        {params['output']}")
    # One greppable line per model: `grep FIND_LR <log>` turns a chained family sweep into the
    # table you actually have to transcribe, in order, without scrolling.
    LOGGER.info(
        f"FIND_LR  model={model_name}  batchsize={bs}  lr={suggested:.2e}"
        + (f"  ->  {recipe}" if recipe else "")
    )
    LOGGER.info("=" * 64)


if __name__ == "__main__":
    main()
