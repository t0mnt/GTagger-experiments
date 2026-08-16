"""
    python utils/find_lr.py -cn jctagging model=tag_transformer save=false
    python utils/find_lr.py -cn toptagging model=tag_transformer save=false
    python utils/find_lr.py -cn jctagging model=tag_transformer model/framesnet=learnedpd save=false

The task is selected with `-cn` (toptagging / jctagging / amplitudes / ttbar / ...)
and the dataset within a task with the usual data overrides (e.g. `data.dataset=mini`
for top tagging); the sweep simply cycles that task's training dataloader, so a
larger `+lr_find.num_iter` samples more of the data.

THE RECOMMENDATION (TRANSCRIBE line) IS SHAPE-GATED since 2026-08-16; neither
statistic is trusted unconditionally. History, because this logic has now been wrong
in both directions and the evidence trail is the guard against a third flip:

- Originally the recipe value was `loss-min / 10` (the classic fastai bracket).
- 2f29a17 flipped the recommendation to STEEPEST-DESCENT on nine ParticleNet reruns:
  steepest was stable to 1.4x while loss-min/10 spread 5x. That mistook STABILITY for
  ACCURACY. On the GraphTrans hybrids the curve falls at a near-constant shallow
  slope for decades before the stability edge; argmin(gradient) lands wherever noise
  peaks inside that plateau and BARELY MOVES between reruns or batch sizes
  (PlainGraphTrans: 3.08e-05 at bs=512 AND 4e-05 at bs=2048) -- stable because the
  plateau is, not because it tracks the edge.
- The 2026-08-15 TRANSCRIBE rule kept steepest as default and distrusted it only
  when the bracket sat >10x above. That coherence test cannot fire when the curve
  pins BOTH statistics low: ParticleNetParTGraphTrans read steepest 3.08e-05 with
  bracket 1.17e-04 (3.8x, "coherent") and the rule endorsed steepest.
- MEASURED COST of that endorsement (2026-08-16, three controlled runs, same
  architecture and config, bs=512): lr 3e-5 (steepest) gave test acc 0.9380/0.9384
  and rej(epsS=0.3) 1239/1160; lr 1e-3 (the bracket value) gave 0.9414 and 1771,
  matching the external weaver reference (0.9417) inside seed spread. Steepest was
  33x low and cost 0.32pp accuracy / 40% background rejection.

The rule now reads the CURVE, not just the two numbers (see PINNED_DECADES at
suggest_lr): steepest counts only when the descent has a DISTINCT slope peak -- the
lr-span of the region within half the steepest slope must stay under ~1.5 decades.
ParticleNet's concentrated fall passes (its nine-rerun steepest 1.32-1.91e-3 remains
the recommendation there, where the bracket read a 20-70x-high 2.64e-2-1.39e-1);
the hybrids' plateau descents fail it, and the bracket -- WITH an interior minimum --
is the recipe (PNPT above; PlainGraphTrans 4.52e-04). Pinned steepest with NO
interior minimum means nothing on the curve is anchored: rerun, at a smaller batch
if it reproduces. Hybrid brackets vary run-to-run (PNPT: 1.17e-04 vs ~1e-3 at the
same bs), so when the banner says curve-pinned, confirm with a second sweep before
committing a long run. The original loss-min/10 argument (steepest drifts low as
num_iter grows, davidtvs/pytorch-lr-finder#68) is why `num_iter` stays 300 -- keep
it short; if a suggestion looks unstable, lower it, don't raise it.

It reuses the experiment's own `_batch_loss`, optimizer, scaler and dataloader,
so the measured loss-vs-lr curve reflects the lr-scale-determining setup: optimizer
type, betas/eps, param-group ratios (`lr_factor_framesnet`), batchsize and amp. Two
regularizers are deliberately switched off for the sweep -- `weight_decay` (inert over
~300 steps) and gradient clipping (it would cap the step and mask the high-lr
divergence the test needs) -- so the curve is the raw loss-vs-lr response. The base
`training.lr` is *ignored*; only the inter-group lr ratios are preserved.

The optimizer (type/betas) and param-groups come from the chosen *training* config, and
since 2026-08-16 the finder ALIGNS that choice with the model's own recipe: when
`config/training/<prefix>_<model>.yaml` exists (the same file the FIND_LR pointer names)
and no explicit `training=` override was given, the sweep recomposes under it, so the
number and the recipe it lands in share one optimizer. An explicit `training=...` always
wins; models without a recipe sweep under the task default (`tag_gts_and_friends_default`,
AdamW -- correct for the GT hybrids, whose recipes inherit it).

Why this is load-bearing and not a convenience (the 2026-08-16 ParticleNet incident):
the task default silently became the GT AdamW recipe at the repo root commit (2f29a17),
while `top_particlenet.yaml` inherits `top_ParT` and trains with RANGER. Ranger's RAdam
rectification + Lookahead slow weights damp the early effective step, so at equal nominal
lr it moves less than AdamW and its loss-vs-lr curve sits about an order of magnitude
RIGHT of AdamW's. Bare `model=tag_particlenet` sweeps therefore measured an AdamW curve
(steepest 1.7-2.2e-4, loss-min/10 ~4.7-8.2e-3 -- both statistics down together, twice,
2026-08-15 and 2026-08-16) while the recipe, its canonical lr 1e-2 (paper) / ~1e-3
(nine-rerun steepest envelope 1.32-1.91e-3), and the FIND_LR pointer all speak Ranger.
No selector rule can rescue a sweep run under the wrong optimizer; alignment removes the
mismatch at the source.
 
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
    +lr_find.bs_draws=8             ITERABLE datasets only (JetClass, TopTagXL): probe the
                                    heaviest of N drawn batches (default 1). Those stream, so
                                    no batch can be CONSTRUCTED -- worst-of-N is the only
                                    substitute, and it is a weak one (N=8 buys roughly the
                                    p87 batch where construction targets the run's worst).
                                    Ignored where construction works. Costs N collates per
                                    rung, not N training steps.
    +lr_find.bs_safety=1.0          fraction of the largest fit to use     (default 1.0,
                                    i.e. the largest fitting power of two; <1 adds
                                    headroom but breaks the power of two)

The probe batch is CONSTRUCTED from the dataset's own jet lengths, not drawn: it sits
`bs_sigmas` sd above a typical batch's total size, with P_max at the dataset maximum.
A random draw is a MEDIAN batch, and a long run meets the worst of ~10^5 draws, which
is what used to make `bs_safety<1` necessary -- see PROBE_SIGMAS in this file for the
measured gap. JetClass and TopTagXL stream through an iterable dataset with no indexable
items, so nothing can be constructed there: the probe falls back to a drawn batch (one, or
the heaviest of `bs_draws`) and says which in the log. `bs_safety` still applies there.

e.g.  python utils/find_lr.py -cn toptagging model=tag_LorentzNetLGATrSlimGraphGPS \\
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
    bs_draws=1,  # iterable datasets only: probe the heaviest of N drawn batches
    bs_refine=False,  # bisect the octave the doubling search discards (3 more probes, no power of two)
    bs_safety=1.0,  # fraction of the largest fitting batchsize to use (1.0 keeps a power of two)
)


# Advisory ceiling on the batchsize the GPU search returns. The search maximises what FITS,
# which is not the same question as what trains best: under the epoch budget (equal data
# exposure, tag_gts_and_friends_default) a bigger batch buys FEWER optimizer steps for the
# same 20 (or 5) passes, so past some point the fit-maximal batch costs updates rather than
# saving time. Advisory only -- nothing is clamped, because the right ceiling is per-model
# and this tool measures memory, not convergence.
BS_CEILING = 512


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
    # Whole lookup guarded, not just the scan. `getattr(..., None)` suppresses only
    # AttributeError, so a `data_list` implemented as a property that raises anything else
    # would propagate and kill the batch search -- which is the one thing this helper must
    # never do, since its entire contract is "None when lengths are not cheaply available"
    # and the caller's fallback is a random batch. Not reachable for the two datasets here
    # (TaggingDataset's is a plain attribute, weaver's iterable one has none), but the next
    # dataset is exactly where a soft-fail helper earns its keep.
    try:
        data_list = getattr(dataset, "data_list", None)  # map-style TaggingDataset only
        if data_list is None or len(data_list) == 0:
            return None
        lengths = np.fromiter(
            (int(d.x.shape[0]) for d in data_list), dtype=np.int64, count=len(data_list)
        )
    except Exception:
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


def _heaviest_drawn_batch(exp, draws):
    """Heaviest of `draws` batches from the loader -- the only probe an ITERABLE dataset allows.

    JetClass and TopTagXL stream through `SimpleIterDataset`, which has no `data_list` and no
    indexable items, so `_worst_case_indices` cannot apply: you cannot ASK for a heavy batch,
    only take what the iterator yields. Their batches still vary, though -- weaver hands over
    a dense `(B, 4, P)` tensor and `dense_to_sparse_jet` immediately collapses it to the same
    `ptr` layout top-tagging uses, so `sum n` and `sum n^2` move batch to batch exactly as
    they do there. A single draw is still a median batch.

    Worst-of-K is the available substitute. The max of K draws approximates the (1 - 1/K)
    quantile, so it is much weaker than construction (K=8 buys about p87, where construction
    targets the worst batch of the whole run) -- but it is strictly better than K=1 and it
    costs only K collates plus K extractions, not K training steps.

    Weight is `sum n^2` read off `ptr`, which is uniform across both experiment types because
    both go through `_extract_batch`. Returns None on any problem, leaving the caller's single
    `next(iter(...))` in place.
    """
    if draws <= 1:
        return None
    best, best_weight = None, -1.0
    try:
        it = iter(exp.train_loader)
        for _ in range(draws):
            batch = next(it, None)
            if batch is None:
                break
            *_, ptr, _ = exp._extract_batch(batch)
            n = torch.diff(ptr).to(torch.float64)
            weight = float((n**2).sum())
            if weight > best_weight:
                best, best_weight = batch, weight
    except Exception as err:
        LOGGER.warning(f"  worst-of-{draws} probe draw failed ({err}); using a single batch.")
        return None
    return best


def find_max_batch_size(
    exp, start, max_cap, safety, sigmas=PROBE_SIGMAS, refine=False, draws=1
):
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
    trajectory -- fragmentation grows with run length, and no single-step probe sees that.
    ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` is what addresses it, but run this
    under the SAME allocator setting as the job it is sizing, and treat the setting as a
    per-CAMPAIGN decision made before the first row: it moves walltime, and walltime is a
    reported column. Verify the chosen batchsize with a short real run either way.
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
        memory probe: a rung that survives one step and dies on the repeat is a rung that
        does not fit, and the old single-step probe would have called it OK.

        Each step gets its OWN batch -- see `make_batch`. On the constructed path the two are
        identical, so the timed step measures the same work as the warm-up; on the drawn path
        (JetClass/TopTagXL) they are two independent worst-of-`draws` samples, which is a
        slightly noisier timing and a slightly stronger memory probe.
        """
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            with open_dict(exp.cfg):
                exp.cfg.training.batchsize = bs
            exp._init_dataloader()

            def make_batch():
                """A FRESH batch object per step -- never reuse one across two steps.

                `embed_tagging_data` MUTATES the caller's `ptr` in place, adding the spurion
                offsets (embedding.py: \"Safe since each batch is embedded once; embedding
                twice double-counts spurions -- clone first\"). Two steps on one object hits
                exactly that: the second embed sees a ptr already shifted by n_spurions*B and
                everything downstream is off by that much -- `IndexError: shape of the mask
                [N] does not match the shape of the indexed tensor [N + n_spurions*B]`.

                Constructing again rather than cloning: `_worst_case_indices` is
                deterministic, so both steps get an identical batch, and this works for the
                drawn path and the weaver tuple layout too, neither of which has `.clone()`.
                """
                batch = None if lengths is None else _worst_case_batch(exp, bs, lengths, sigmas)
                if batch is None:
                    batch = _heaviest_drawn_batch(exp, draws)
                if batch is None:
                    if lengths is not None and not state["warned"]:
                        LOGGER.warning(
                            f"  batchsize {bs}: no worst-case batch constructible from "
                            f"{lengths.size} jets -> falling back to a random batch."
                        )
                        state["warned"] = True
                    batch = next(iter(exp.train_loader))
                return batch

            def step():
                loss, _ = exp._batch_loss(make_batch())
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
            f"  probe batch: {'worst of ' + str(draws) + ' DRAWN batches' if draws > 1 else 'ONE RANDOM BATCH'} "
            f"(this dataset streams, so no batch can be constructed) -- keep headroom via "
            f"bs_safety (now {safety})."
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
        # FATAL, not a fallback -- the same rule bperf's --find-batchsize handler enforces
        # for exceptions. `start` was just MEASURED to OOM, so returning it dresses "this
        # does not fit" as a sizing result: find_lr would then OOM at sweep step 1, and
        # bperf -- whose fatal handler sees only EXCEPTIONS -- would time its whole matrix
        # at a size known not to fit, the exact hours-for-nothing failure f200b22 closed.
        raise RuntimeError(
            f"Even batchsize {start} does not fit a full training step on this GPU "
            f"(the probe stands in for the worst batch of a run, so a lighter median "
            f"batch may fit and still OOM mid-run). There is no size to report; lower "
            f"bs_start below {start} only if you genuinely intend to train there."
        )

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


# Curve-shape gate for the steepest-descent statistic (2026-08-16, the PNPT incident --
# see the module docstring's history). "Steepest" is only a candidate recommendation
# when the descent is CONCENTRATED: a distinct slope peak. On the GT hybrids the loss
# falls at a near-constant shallow slope over decades, argmin(gradient) lands wherever
# noise peaks inside that plateau, and transcribing it cost a campaign row 0.32pp
# accuracy / 40% rejection. Detector: the lr-span, in decades, of the region whose
# downhill slope is within PINNED_HALF of the steepest. A concentrated fall spans well
# under a decade; the hybrids' plateaus span 2+. Between them, 1.5.
PINNED_HALF = 0.5
PINNED_DECADES = 1.5


def suggest_lr(lrs, losses, skip_start, skip_end, beta=0.98):
    """Two statistics -- `loss-min/10` (bracket) and the steepest-descent point --
    plus the two curve diagnostics that decide which (if either) is a recipe:
    `interior_min` (is the bracket anchored by a real trough?) and `pinned`
    (is "steepest" just a point on a slope plateau? see PINNED_DECADES above).
    `transcribe_lr` turns the four into one recommendation.

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
        steep_idx = int(np.argmin(gradients))
        steepest = float(lr_grad[steep_idx])
        # pinned test: how many DECADES of lr fall at >= half the steepest slope? A
        # distinct peak keeps this small; a plateau descent (where the argmin above is
        # noise-arbitrary) spreads it over the plateau's whole width.
        gmin = gradients[steep_idx]
        region = lr_grad[gradients <= PINNED_HALF * gmin] if gmin < 0 else lr_grad[:0]
        steep_decades = (
            float(np.log10(region.max() / region.min())) if region.size >= 2 else 0.0
        )
        pinned = steep_decades > PINNED_DECADES
    else:
        steepest = float(lr_trim[int(np.argmin(loss_trim))])
        # too few points to certify a distinct peak -> conservative: not a recipe
        pinned, steep_decades = True, float("nan")

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
    return steepest, min_loss_lr, lr_trim, loss_trim, interior_min, pinned, steep_decades


def transcribe_lr(steepest, bracket, interior_min, pinned, ratio_bar=10.0):
    """ONE number, ONE reason -- the banner applies the module-docstring rule so the
    reader never has to (2026-08-15). REWRITTEN 2026-08-16 after the previous table's
    'coherent pair -> steepest' branch endorsed a curve-pinned 3e-5 for
    ParticleNetParTGraphTrans and cost the row 0.32pp accuracy / 40% rejection
    (module docstring, MEASURED COST). Ratio-coherence cannot detect pinning when the
    curve drags BOTH statistics low; the `pinned` curve-shape flag (suggest_lr,
    PINNED_DECADES) is the gate now.

    - PINNED steepest, interior minimum: the BRACKET (loss-min/10) is the recipe --
      steepest sits on a slope plateau, the trough reads the stability edge.
      Live cases: PNPT bs=512 3e-5 vs 1e-3 (bracket matched weaver, steepest cost
      0.32pp); PlainGraphTrans bs=512 3.08e-05 vs 4.52e-04.
      CAVEAT printed with it: hybrid brackets vary run-to-run (PNPT read 1.17e-04
      and ~1e-3 at the same bs) -- confirm with a second sweep before a long run.
    - PINNED steepest, NO interior minimum: NOTHING on this curve is anchored ->
      no recipe; rerun, at a smaller batch if it reproduces.
      Live case: ParticleNet 2026-08-15 anomaly (2.21e-04 vs 8.19e-03, steepest
      6-8x below its own nine-rerun envelope -- a shifted curve IS pinning).
    - DISTINCT steepest, interior minimum, bracket within ratio_bar and above
      steepest: both anchored and agreeing -> the BRACKET (the recipe value the
      classic range test transcribes).
    - DISTINCT steepest, interior minimum, bracket > ratio_bar above: the bracket is
      the outlier against a genuine slope peak -> STEEPEST.
      Live case: the nine ParticleNet reruns (steepest 1.32-1.91e-3 vs brackets
      2.64e-2-1.39e-1, 20-70x high; steepest matched the reproduced optimum).
    - DISTINCT steepest, interior minimum, bracket BELOW steepest: a 'bracket' under
      the steepest point is not an upper bracket; the curve is suspect above the
      peak -> STEEPEST.
    - DISTINCT steepest, NO interior minimum: the bracket is unanchored by
      definition -> STEEPEST (this is the nine-rerun ParticleNet posture).

    Returns (lr_or_None, reason_string). Pure function; unit-tested in
    tests/internal/test_find_lr_transcribe.py, including synthetic curves for the
    `pinned` detector itself.
    """
    ratio = (bracket / steepest) if steepest > 0 else float("inf")
    if pinned:
        if interior_min:
            return bracket, (
                f"loss-min/10 [steepest is curve-pinned (slope plateau); the interior "
                f"minimum anchors the bracket. Brackets vary run-to-run on these "
                f"curves -- confirm with a second sweep]")
        return None, (
            f"NO RECIPE [steepest is curve-pinned AND no interior minimum -- nothing "
            f"on this curve is anchored; rerun, at a smaller batch if it reproduces]")
    if interior_min and steepest < bracket <= ratio_bar * steepest:
        return bracket, (
            f"loss-min/10 [distinct slope peak and an interior minimum agree "
            f"(bracket/steepest = {ratio:.1f}x <= {ratio_bar:.0f}x); the bracket is "
            f"the recipe value]")
    if interior_min and ratio > ratio_bar:
        return steepest, (
            f"steepest [distinct slope peak; the bracket sits {ratio:.1f}x above it "
            f"(> {ratio_bar:.0f}x) -- the nine-rerun ParticleNet pattern, where the "
            f"trough hugs the divergence and reads high]")
    if interior_min:  # bracket at or below steepest
        return steepest, (
            f"steepest [distinct slope peak; loss-min/10 sits AT OR BELOW it "
            f"({ratio:.1f}x), which is no upper bracket -- curve suspect past the peak]")
    return steepest, (
        f"steepest [distinct slope peak and NO interior minimum -- the bracket is "
        f"unanchored on this curve shape]")


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


# task -> recipe filename prefix; shared by the FIND_LR pointer and the training alignment
RECIPE_PREFIX = {"toptagging": "top", "jctagging": "jc", "toptagxl": "xl"}


def _recipe_training_choice(exp_type, model_stems, current_choice, overrides, isfile=None):
    """Which training config should the sweep run under? None = keep the composed one.

    Pure decision function (2026-08-16 incident fix -- see the module docstring's
    'Why this is load-bearing'): a baseline must be swept under ITS OWN recipe, because
    the suggested lr is optimizer-specific and the FIND_LR pointer names that recipe.

    - An explicit `training=` (or `training@...=`) CLI override always wins: None.
    - Otherwise the first stem with a `config/training/<prefix>_<stem>.yaml` on disk is
      the target; already-current targets return None (nothing to do).
    - No prefix for this task, or no recipe file: None.

    `model_stems` is the same candidate list the FIND_LR pointer uses (hydra model
    choice first, net class name as fallback). `isfile` is injectable for tests.
    """
    isfile = isfile or os.path.isfile
    prefix = RECIPE_PREFIX.get(exp_type)
    if prefix is None:
        return None
    if any(o.split("=", 1)[0].split("@", 1)[0] == "training" for o in overrides):
        return None
    for stem in model_stems:
        if not stem:
            continue
        name = f"{prefix}_{stem}"
        if isfile(os.path.join("config", "training", f"{name}.yaml")):
            return None if name == current_choice else name
    return None


def _model_stems(cfg):
    """Candidate recipe stems, hydra model choice first (same order as the pointer)."""
    stems = []
    try:
        from hydra.core.hydra_config import HydraConfig

        choice = HydraConfig.get().runtime.choices.get("model")
        if choice:
            stems.append(choice[4:] if choice.startswith("tag_") else choice)
    except Exception:
        pass  # not under hydra.main (imported by bperf, or composed directly)
    net_class = (OmegaConf.select(cfg, "model.net._target_", default="") or "").rsplit(".", 1)[-1]
    if net_class:
        stems.append(net_class)
    return stems


def _align_training_with_recipe(cfg):
    """Recompose cfg under the model's own training recipe when one exists.

    Returns (cfg, chosen_name_or_None). Recomposition replays the CLI overrides and
    appends `training=<recipe>`, so everything the operator typed still applies; any
    failure keeps the composed cfg (alignment must never be able to kill a sweep).
    """
    try:
        from hydra import compose
        from hydra.core.hydra_config import HydraConfig

        hc = HydraConfig.get()
        overrides = list(hc.overrides.task)
        target = _recipe_training_choice(
            cfg.exp_type, _model_stems(cfg), hc.runtime.choices.get("training"), overrides
        )
        if target is None:
            return cfg, None
        cfg2 = compose(config_name=hc.job.config_name, overrides=overrides + [f"training={target}"])
        LOGGER.info(
            f"LR sweep: training config '{hc.runtime.choices.get('training')}' -> '{target}' "
            f"(the model's own recipe; optimizer={cfg2.training.optimizer}). The suggested lr "
            f"is optimizer-specific and the FIND_LR pointer names this recipe -- pass an "
            f"explicit training=... to override."
        )
        return cfg2, target
    except Exception as err:
        LOGGER.warning(f"LR sweep: training-recipe alignment skipped ({err}).")
        return cfg, None


# Default to the real config/ tree so a bare run matches training (clipping on, full data);
# pass `-cp config_quick ... data.dataset=mini` for a fast smoke test of the finder itself.
@hydra.main(config_path="../config", config_name="toptagging", version_base=None)
def main(cfg):
    # Sweep a baseline under its own recipe (2026-08-16 incident fix; must run before
    # anything reads cfg.training). Explicit training= overrides survive inside.
    cfg, aligned_training = _align_training_with_recipe(cfg)

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
    #   in config/model; the only `compile: false` in the tree is the framesnet sub-configs,
    #   which are eager in every posture), so the shipped posture IS the thing to measure.
    #   Sizing eager here would hand the campaign a batch chosen under a memory
    #   profile it never runs at.
    #
    # Consequence worth carrying: bperf's it/s numbers are therefore taken at a batch nobody
    # trains at. Fine for the speedup RATIO it reports; not a basis for ranking impls by
    # jets/s, which is what this function's number decides.
    if params["find_batch_size"]:
        LOGGER.info(
            f"We recommend a ceiling of {BS_CEILING} as performance starts to deteriorate "
            f"after that, particularly when iterations are limited by epoch."
        )
        bs = find_max_batch_size(
            exp,
            params["bs_start"],
            params["bs_max"],
            params["bs_safety"],
            sigmas=float(params["bs_sigmas"]),
            refine=bool(params["bs_refine"]),
            draws=int(params["bs_draws"]),
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

    steepest, min_loss_lr, _, _, interior_min, pinned, steep_decades = suggest_lr(
        lrs,
        losses,
        skip_start=params["skip_start"],
        skip_end=params["skip_end"],
        beta=params["beta"],
    )
    # the two statistics; which one (if either) is the recipe is transcribe_lr's call
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
    #
    # Keyed off the MODEL CONFIG NAME, not `model_name` (which is the net's class). Those
    # agree for the 8 hybrids -- tag_PlainGraphGPS -> PlainGraphGPS -> top_PlainGraphGPS.yaml
    # -- and disagree for all 8 baselines: tag_cgenn's net class is CGENN, so the old
    # derivation looked for top_CGENN.yaml, missed the real top_cgenn.yaml, and silently
    # dropped the pointer. That hit tag_{cgenn,lgatr,slim,lorentznet,particlenet,transformer,
    # ParT,MIParT} -- including the one model in the unswept 25 whose posture is still open.
    # Class name kept as the fallback so this cannot end up worse than it was.
    prefix = {"toptagging": "top", "jctagging": "jc", "toptagxl": "xl"}.get(cfg.exp_type)
    recipe = None
    if prefix:
        stems = []
        try:
            from hydra.core.hydra_config import HydraConfig

            choice = HydraConfig.get().runtime.choices.get("model")
            if choice:
                stems.append(choice[4:] if choice.startswith("tag_") else choice)
        except Exception:
            pass  # not under hydra.main (imported by bperf, or composed directly)
        stems.append(model_name)
        recipe = next(
            (p for s in stems if os.path.isfile(p := f"config/training/{prefix}_{s}.yaml")),
            None,
        )

    LOGGER.info("=" * 64)
    if params["find_batch_size"]:
        LOGGER.info(f"Batchsize (fit to GPU):          {bs}")
    # The two statistics are DIAGNOSTICS, each printed with its health flag; the
    # recommendation is the TRANSCRIBE line below, which applies the shape-gated rule
    # (transcribe_lr / module docstring). A "[default]"-labeled steepest line here
    # once let a curve-pinned 3e-5 into a recipe and cost the row 0.32pp accuracy --
    # neither statistic gets a label that invites transcribing it directly again.
    shape = (
        f"curve-pinned: half-max slope spans {steep_decades:.1f} decades > {PINNED_DECADES:g}"
        if pinned
        else f"distinct peak: half-max slope spans {steep_decades:.1f} decades"
    )
    LOGGER.info(f"steepest descent:   {steepest:.2e}   [{shape}]")
    if interior_min:
        LOGGER.info(f"loss-min / 10:      {suggested:.2e}   [interior minimum]")
    else:
        LOGGER.warning(
            "No INTERIOR loss minimum: the loss was still falling when the sweep ended, so the "
            "argmin sits in the near-divergence tail. loss-min/10 is meaningless on this curve "
            "(it moves by several x between reruns); the plot will show no trough before the "
            "blow-up."
        )
        LOGGER.info(f"loss-min / 10:      {suggested:.2e}   [NOT reliable: no interior minimum]")
    # ONE directive line -- the banner applies the transcription rule so the reader
    # never juggles the two diagnostics above under queue pressure (see transcribe_lr).
    pick, why = transcribe_lr(steepest, suggested, interior_min, pinned)
    untrusted = pick is None
    if untrusted:
        LOGGER.warning(f"TRANSCRIBE: nothing -- {why}")
        pick = steepest  # keep the lines below greppable, but tagged
    else:
        LOGGER.info(f"TRANSCRIBE: {pick:.2e} -- {why}")
    suggested = pick
    tag = "  [UNTRUSTED: no-recipe rule fired, rerun first]" if untrusted else ""
    reuse = f"training.lr={suggested:.2e}"
    if params["find_batch_size"]:
        reuse = f"training.batchsize={bs} " + reuse
    LOGGER.info(f"  ->  reuse with:  {reuse}{tag}")
    # The lr is optimizer-specific: say which training config produced it, so a
    # transcription into a DIFFERENT recipe is visibly wrong (the 2026-08-16 incident
    # was exactly this mismatch, silent).
    trained_under = aligned_training
    if trained_under is None:
        try:
            from hydra.core.hydra_config import HydraConfig

            trained_under = HydraConfig.get().runtime.choices.get("training")
        except Exception:
            trained_under = "(composed directly)"
    LOGGER.info(
        f"  ->  swept under: training={trained_under}  optimizer={cfg.training.optimizer}"
    )
    LOGGER.info(f"  ->  plot:        {params['output']}")
    # One greppable line per model: `grep FIND_LR <log>` turns a chained family sweep into the
    # table you actually have to transcribe, in order, without scrolling. lr= carries the
    # TRANSCRIBE pick (rule-applied), not raw steepest, since 2026-08-15.
    LOGGER.info(
        f"FIND_LR  model={model_name}  batchsize={bs}  lr={suggested:.2e}"
        + (f"  ->  {recipe}" if recipe else "")
        + tag
    )
    # Short by design: the reasoning was printed before the search (see BS_CEILING); this is
    # only the reminder, at the point where the number is about to be transcribed into a recipe.
    if params["find_batch_size"] and bs is not None and bs > BS_CEILING:
        LOGGER.warning(f"  ->  reminder:    over the {BS_CEILING} ceiling")
    LOGGER.info("=" * 64)


if __name__ == "__main__":
    main()
