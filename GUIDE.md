# Using this repo (with the graph-transformer hybrids)

A practical walkthrough for someone who has just cloned the repo and wants to
train models — especially the GraphTrans / GraphGPS hybrid taggers added on top of
the LLoCa baselines. For the upstream paper-reproduction commands see
[`REPRODUCE.md`](REPRODUCE.md); for the method see the papers linked in
[`README.md`](README.md).

---

## 1. Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

Get the top-tagging dataset (~1.5 GB → `data/toptagging_full.npz`):

```bash
python data/collect_data.py toptagging
```

Smoke-test the install on the tiny datasets shipped under `data/` (no GPU needed):

```bash
pytest tests/experiments/test_tag_equivariance.py -q     # 32 invariance checks
python run.py -cp config_quick -cn toptagging save=false # one quick training
```

`config_quick/` mirrors `config/` with tiny models/data — ideal for sanity checks
and reading along with print statements. `config/` is the real training setup.

---

## 2. Repo layout

| path | what |
|---|---|
| `run.py` | entry point: builds an experiment from a hydra config and trains/evaluates it |
| `utils/find_lr.py` | LR range test + optional GPU batch-size finder (see §6) |
| `config/` | real configs; `config_quick/` the tiny mirror |
| `config/model/tag_*.yaml` | one file per tagger (model definition only) |
| `config/training/top_*.yaml` | training budgets / optimizers / schedules |
| `config/model/framesnet/` | LLoCa frame predictors (for non-equivariant models) |
| `experiments/baselines/` | the network implementations |
| `experiments/tagging/wrappers.py` | the wrapper that adapts each net to the tagging pipeline |
| `tests/experiments/` | `test_tag_equivariance.py`, `test_tag_invariance.py`, `test_tag_flops.py` |

---

## 3. The model zoo

Selected with `model=tag_<name>`. Two families of hybrids were added, each in a
2×2 grid of {graph backbone} × {GraphTrans = sequential GNN→transformer, GraphGPS
= interleaved GNN‖attention per layer}:

| backbone | GraphTrans | GraphGPS | equivariance |
|---|---|---|---|
| plain MPNN + torch-MHA | `tag_PlainGraphTrans` | `tag_PlainGraphGPS` | non-equiv → LLoCa frames |
| ParticleNet EdgeConv + ParT attn | `tag_ParticleNetParTGraphTrans` | `tag_ParticleNetParTGraphGPS` | non-equiv → LLoCa frames |
| CGENN + L-GATr | `tag_CGENNLGATrGraphTrans` | `tag_CGENNLGATrGraphGPS` | **equivariant by construction** |
| LorentzNet + L-GATr-slim | `tag_LorentzNetLGATrSlimGraphTrans` | `tag_LorentzNetLGATrSlimGraphGPS` | **equivariant by construction** |

Plus the upstream baselines: `tag_ParT`, `tag_particlenet`, `tag_transformer`,
`tag_graphnet`, `tag_lgatr`, `tag_lorentznet`, `tag_MIParT`, `tag_pelican_fair`, …

**Equivariance comes from one of two routes**, and it determines whether you set a
framesnet:

- **Internally equivariant** (CGENN, LorentzNet-slim, L-GATr, pelican): equivariant
  by construction, run on `framesnet=identity` (the default in their configs). Do
  *not* give them a learned framesnet.
- **Non-equivariant + LLoCa** (Plain, ParticleNet-ParT, ParT, transformer, graphnet):
  made Lorentz-equivariant by canonicalizing inputs into a learned local frame. Set
  `model/framesnet=learnedpd` (or `learnedso13`, …) to enable it; `identity` (default)
  gives the plain non-equivariant baseline.

---

## 4. Running a training

`run.py` defaults to the tiny `config_quick` tree (a fast debug model + toy data), so
a bare `python run.py model=…` is a smoke test, NOT real training — and passing a
`training=top_<...>` recipe there fails at composition (config_quick has no such
recipes). For a real run add `-cp config` so the full `config/` tree (real models,
data, and the `top_<...>` recipes) is used:

```bash
# quick smoke tests on the debug tree (no -cp config): a couple of hundred iterations
python run.py model=tag_PlainGraphGPS model/framesnet=learnedso13   # learned frames
python run.py model=tag_LorentzNetLGATrSlimGraphGPS                   # identity frames

# a REAL training: full config tree, the hybrid's own recipe, full data, a GPU
python run.py -cp config model=tag_ParticleNetParTGraphGPS \
    training=top_ParticleNetParTGraphGPS data.dataset=full gpus=1
```

Useful overrides: `data.dataset={full,mini}`, `training.iterations=…` (add
`training.epochs=null` alongside it — the default recipes carry an epoch budget, and
`epochs` takes precedence over an explicit iteration count),
`training.batchsize=…`, `training.lr=…`, `gpus=N`, `save={true,false}`,
`model.net.knn_metric={deltaR,minkowski}`, `model.net.num_blocks=…`.

Each run prints a paste-ready LaTeX table row at the end:
`table test: <Model> & <frames> & <iters> [N trials] & <params> & <acc> & <auc> &
<rej03> & <rej05> & <rej08> & <time>s & <flops> & <kNN>` (the `[N trials]` tag and
`mean ± std` cells appear once a run directory holds more than one trial, §8; JetClass runs
emit the same-form row with per-class rejections instead of rej03/05/08 — toptagxl shares
the top-tagging columns exactly, its evaluation is fully inherited).

---

## 5. Configs: model vs training

A `config/model/tag_*.yaml` is **model definition only** — it has no LR, optimizer
or budget. Those come from the **training** config, selected separately. The
top-tagging default is now `tag_gts_and_friends_default` (AdamW, epochs=20,
CosineAnnealingWarmup, wd=0.01) — the shared GT-hybrid recipe — so the new models run
correctly with no `training=` at all (you just fill `lr`/`batchsize` from
`utils/find_lr.py`). Pass `training=top_<baseline>` to run a baseline under its own tuned
recipe (e.g. `top_transformer` = Lion, lr=3e-5, 300k iters) — see §7.

> **A baseline with no recipe file does not fail — it inherits the hybrid recipe.** The
> default above applies to *anything* that does not pass `training=`, so a baseline whose
> `config/training/top_<X>.yaml` is missing trains at AdamW / lr 1e-3 / batchsize 512 /
> wd 0.01 rather than its published setting, silently. Missing today: `tag_MIParT-L`,
> `tag_graphnet`, `tag_top_transformer`, `tag_pelican*` (`tag_cgenn` now has `top_cgenn.yaml`,
> which pins the same shared recipe explicitly so the reference row stops tracking the task
> default by accident).
>
> **And this fork changed what that fallback is.** Upstream's `config/toptagging.yaml`
> defaults to `training: top_transformer` (Lion, lr 3e-5, **weight_decay 2**, batchsize 128,
> 300k iterations); ours defaults to `tag_gts_and_friends_default` (AdamW, lr 1e-3,
> weight_decay 0.01, batchsize 512, 20 epochs). So a recipe-less baseline does not train the
> way it trained upstream — "the repo default" is not the same object it was. Relevant to
> `tag_cgenn` above all, since it is the reference row for both CGENN hybrids. (The cluster wrappers
> derive `training=top_<X>` automatically *when the file exists*, so this is the same trap
> whether you use `run.py` or `sbatch train.sbatch`.)

The 8 GT hybrids share one recipe: each `config/training/top_<hybrid>.yaml`
`defaults: [tag_gts_and_friends_default]` (AdamW, **epochs=20**,
CosineAnnealingWarmup, shared `weight_decay=0.01`, validate once/epoch) and only fills
its own `batchsize` + `lr` from `utils/find_lr.py` — that shared budget is what makes the
hybrid-vs-hybrid table fair. The upstream baselines keep their own recipes as
reference rows — `top_ParT` (Ranger, lr=1e-3, 20 epochs), `top_lorentznet` (AdamW,
lr=1e-3, 35 epochs), `top_lgatr` (Lion, lr=3e-4, wd=0.2), `top_particlenet` (lr=1e-2)
— or point them at `tag_gts_and_friends_default` to put them on the same budget.

### 5.1 JetClass

The JetClass task mirrors the top-tagging setup one-to-one, selected with
`-cn jctagging` (dataset: `python data/collect_data.py jetclass` — the full 100M-jet
training set, streamed from ROOT files, so mind the disk). Two differences to know:

- **Wider inputs.** `features: default` adds 10 particle-ID / trajectory scalars, so
  the non-equivariant models see `in_channels = 7 + 10 = 17` (wired automatically by
  `init_physics`).
- **JetClass recipes drop weight decay** (`weight_decay: 0`, matching `jc_ParT`).

The training-config structure repeats the §5 pattern: the task default is
`jc_gts_and_friends_default` (AdamW, CosineAnnealingWarmup, **epochs=5** — the
ParT-standard exposure, 1M steps × batch 512 ≈ 5 passes of the 100M train set —
validate once per nominal epoch), and each hybrid has a `config/training/jc_<hybrid>.yaml`
whose `batchsize`/`lr` are `???` until you fill them from the LR finder **on the
jctagging task** (don't copy the top-tagging values — the wider inputs move both the
memory ceiling and the loss-vs-lr curve):

```bash
python utils/find_lr.py -cn jctagging model=tag_CGENNLGATrGraphGPS save=false \
    +lr_find.find_batch_size=true +lr_find.bs_max=512
# -> fill training.batchsize / training.lr into config/training/jc_CGENNLGATrGraphGPS.yaml
```

(The default `num_iter=300` sweep length is right on JetClass too — it samples the lr
ramp, not the dataset, so the 100M-jet size changes nothing; raising it only biases the
suggestion low, per the finder's docstring. No JetClass-specific overrides are needed.)

Note the `???` markers are for humans — hydra cannot enforce them here (an OmegaConf
`???` never overrides a value inherited from `tag_default`), so an unfilled recipe
silently trains at the 512 / 1e-3 fallback instead of erroring. The JetClass baseline
reference rows keep their published budgets: `jc_ParT` (Ranger, 1M iterations),
`jc_MIParT`, `jc_lgatr` (Lion, bs=512), `jc_transformer`. To cut cost, shrink
`data.train_files_range` — the same 5 passes then apply to the subset; don't lower
the epochs.

### 5.2 TopTagXL

TopTagXL is top-tagging at JetClass scale: the same **binary qcd-vs-top** task as §5
but with 100M training jets in the JetClass streaming format, selected with
`-cp config -cn toptagxl` (dataset: `python data/collect_data.py toptagxl` — the
collector reads the file list and md5 checksums from Zenodo record 10878355's API
at download time, verifies, and extracts to `data/toptagxl`. Layout: `train_100M/`,
`test_25M/`, `val_10M/`, with file numbering running *continuously* across the
splits — `{qcd,top}_000–499 / 500–624 / 625–674` — at 100k jets per file).
Everything from §5.1 carries over:

- **Inputs are JetClass-wide** even though the task is binary: `features: default`
  adds the 10 PID/trajectory scalars, so `in_channels = 7 + 10 = 17` (wired
  automatically by `init_physics`).
- **Recipes drop weight decay and keep epochs=5** (100M jets × 5 passes — the same
  ParT-standard exposure): the family default is `xl_gts_and_friends_default`, and
  each hybrid has a `config/training/xl_<hybrid>.yaml` whose `batchsize`/`lr` are
  `???` until filled. **Seed them from the swept `jc_` values**: XL inputs are
  identical to JetClass (17 channels, same multiplicity), so the batch-size memory
  ceiling carries over unchanged and the jc lr is the right prior — only the loss
  changed (binary sigmoid vs 10-class softmax). Upstream itself runs its XL
  baselines under the JetClass recipe (the toptagxl task default is
  `jc_transformer`). Confirm with the cheap finder sweep on the toptagxl task
  rather than trusting the transfer blind (the `???` fallback caveat above applies
  here identically):

```bash
python utils/find_lr.py -cn toptagxl model=tag_PlainGraphGPS save=false \
    +lr_find.find_batch_size=true +lr_find.bs_max=512
# -> fill training.batchsize / training.lr into config/training/xl_PlainGraphGPS.yaml
python run.py -cp config -cn toptagxl model=tag_PlainGraphGPS training=xl_PlainGraphGPS
```

Two XL-specific footnotes. First, the shipped `data.val_files_range: [625, 675]` is
the dataset's **canonical 10M-jet validation split** (Zenodo's 100M/25M/10M
partition) — keep it. Under the family recipe's once-per-epoch cadence it costs
about 3% of training compute (5 forward-only passes over 10M jets, against 500M
training samples at ~3x the per-sample cost), and staying on the published split
keeps val-derived numbers comparable with the LLoCa paper's. Shrink it only for
debugging, where you validate many times per epoch. Second, TopTagXL reuses the top-tagging binary
evaluation path unchanged, so the rejection metrics and the results-table /
`utils/aggregate_table.py` machinery of §8 work as-is. Baseline reference rows run under
the task default (`jc_transformer`: AdamW, 1M fixed iterations) or their own
recipes, as in REPRODUCE.md.

---

Two checks before you sweep eight unpublished models. **(1) Validate the finder** on a
baseline with a published lr: run it on ParticleNet at that recipe's fixed batchsize
(`utils/find_lr.py -cn toptagging model=tag_particlenet training=top_particlenet
save=false`) and confirm the suggestion sits within an order of magnitude of the
published `1e-2`. **(2) Reproduce a known number** with one `save=false` ParticleNet
training under its published recipe, comparing test accuracy/AUC with the literature. A
mismatch in either is an environment or data problem, and is far cheaper to find here
than inside a multi-day hybrid run.

## 6. Choosing hyperparameters

**Learning rate (and GPU batch size) — `utils/find_lr.py`.** Runs a Leslie-Smith LR
range test under the *training config's* optimizer + param-group ratios (but with
weight-decay and grad-clipping switched off for the sweep — they'd only mask the
divergence) and reports a robust `loss-min/10` peak LR — a safe peak for the
warmup→cosine schedule (it never builds the scheduler; it ramps the LR by hand from 1e-7). The default training
is now `tag_gts_and_friends_default` (AdamW, clip=1.0, wd=0.01), so for the GT
hybrids you sweep with nothing extra. Pass `training=<recipe>` only to match a
*different* optimizer — the LR scale is optimizer-specific, so a Lion baseline (e.g.
`training=top_transformer`) must be swept under Lion. `utils/find_lr.py` now defaults to the real `config/` tree
(full data); add `data.dataset=mini` for a quick trial.

```bash
# LR only (default training is the shared AdamW gts-and-friends recipe)
python utils/find_lr.py -cn toptagging model=tag_CGENNLGATrGraphGPS save=false

# on a GPU: fit the batch size first, then sweep the LR at that size
python utils/find_lr.py -cn toptagging model=tag_LorentzNetLGATrSlimGraphGPS \
    save=false +lr_find.find_batch_size=true +lr_find.bs_max=512
```

With `+lr_find.find_batch_size=true` it doubles the batch size until CUDA OOM
(running a full train step, so the probe includes optimizer-state memory) and keeps
the largest fitting power of two (`bs_safety=1.0` default; set `<1` to trade the
power of two for headroom), then prints the batch size and LR, e.g.
`-> reuse with: training.batchsize=2048 training.lr=3.1e-04`, followed by a single greppable line naming the model and the recipe the pair belongs in (`FIND_LR model=X batchsize=N lr=L -> config/training/<prefix>_X.yaml`), so a chained sweep is transcribed with `grep FIND_LR`. Knobs:
`+lr_find.{bs_start,bs_max,bs_sigmas,bs_refine,bs_safety,num_iter,end_lr}` — keep `num_iter` short (~300;
a longer sweep biases the suggestion lower, it doesn't sharpen it).

**Why `+lr_find.bs_max=512` is on every sweep command above.** The search maximises what
FITS, which is not the question the campaign asks — it is the same advisory the tool prints
before it starts:

> *We recommend a ceiling of 512 as performance starts to deteriorate after that,
> particularly when iterations are limited by epoch.*

Under the shared epoch budget (`tag_gts_and_friends_default`, equal data exposure) a larger
batch buys **fewer optimizer steps** for the same 20 passes, so past a point the fit-maximal
batch costs updates rather than saving time. Bounding the search at 512 also makes it
cheaper for the roomy models — Plain fits far more than 512, and climbing to 8192 only to
discard the answer is wasted GPU time. The bound resolves both cases without your knowing
in advance which a model is in: **512 fits → you get 512; 512 OOMs (CGENN-GraphGPS is the
likely one) → you get the largest power of two that did.** Either way the LR is then swept
at the size you will train at, which is the point — an LR swept at 2048 is wrong for 512.
Drop the flag when you genuinely want the unbounded answer; the tool is advisory and clamps
nothing on its own, and it reminds you at the end if the chosen batch is over the ceiling. For models that expose a
`knn_metric`, the sweep pins **`deltaR`** by default so the suggested LRs are comparable across
the family (the LR scale is metric-independent — the model still *trains* under its own
configured metric); pass `+lr_find.force_knn_metric=keep` to sweep each model's own metric
instead (or `=minkowski` to pin that).

**The probe batch is built, not drawn.** Jets vary in size (top tagging: mean 49.2
constituents, max 135), so a random batch is a *median* batch while a long run meets the
worst of ~10⁵ draws — which is how a probe once picked a batch that OOM'd at step 10, hours
into a job. The search therefore constructs each probe batch from the dataset's own lengths:
`bs_sigmas` (default 5) standard deviations above a typical batch's total in **both** `sum n`
and `sum n²`, with the longest jets in it so `P_max` is at its cap. That clears the heaviest
batch of a directly simulated 50-epoch run at every batch size, so `bs_safety` is no longer
the lever for this — leave it at 1.0 on top tagging. JetClass and TopTagXL stream from files
and expose no per-item lengths, so there the probe falls back to one random batch (it logs
which of the two it used) and `bs_safety<1` still applies. The remaining blind spot is
fragmentation, which grows with run length and no single-step probe can see: run the probe
and the job under the SAME allocator, and confirm a chosen batch size with a short real run
before committing a multi-day job.

On the allocator specifically: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is what
addresses fragmentation (it lets a segment grow instead of being a fixed block, which is what
"reserved by PyTorch but unallocated" in an OOM message means). Two rules, and the second is
the one that bites. **The probe and the job must match** — sizing under one allocator and
training under another measures the wrong fragmentation. And **it is a walltime knob, so it
is uniform across a campaign or absent from it**: same rule the `-n` / `num_workers` note in
`docs/oscar-train.sbatch` already states, since walltime is a reported column. Decide it
before the first row, not during. Mid-campaign, leave it alone — a fragmentation OOM is a
recoverable single-row failure you can re-run with the variable set inline (and note as an
exception), whereas a split walltime column is only fixable by re-running everything.

**`+lr_find.bs_refine=true` recovers the octave the search throws away, and it is off by
default.** The doubling search brackets the ceiling between two powers of two and then
discards the whole octave, so the number it returns is on average 1/1.5 of what the card
holds. Refining bisects that bracket in 3 more probes: measured over 60 simulated card sizes,
a mean **1.35×** batch (median 1.25×, up to 1.88×) — against the constructed probe's
1.05–1.3×, which the power-of-two granularity swallows entirely in 72% of cases. End to end
the new default gives **0.82×** the old default's batch (the price of not dying at step 10)
and **1.08×** with refine on; against the `bs_safety=0.5` this guide used to recommend for
CGENN, refined is **2.16×**. The default is off only because a round batch size is comparable
across the recipe table; the bisection never returns a size it has not run a full training
step at.

**But batch is not throughput — read the jets/s curve the search now prints.** `jets/s =
batchsize / step time` saturates once the card is compute-bound, and past that point a bigger
batch buys risk and nothing else. Which regime each model is in had never been measured here,
so the search now times its second step at every rung and reports — SHAPE OF THE OUTPUT ONLY,
the numbers below are invented, no model here has been measured yet:

```
  batchsize   1024: OK  (peak 41.2 GB,    3180 jets/s)     <- illustrative, not measured
  batchsize   2048: OK  (peak 78.9 GB,    3260 jets/s)     <- illustrative, not measured
  jets/s by batchsize: 256:2840  512:3050  1024:3180  2048:3260
```

Flat across the top rungs means the card is saturated and `bs_refine` is not worth turning
on; still climbing means it is. Single timed steps, so treat differences under ~10% as noise.
The search runs two steps per rung rather than one for this — which also makes it a stricter
memory probe, since a rung that survives one step and dies on an identical repeat does not
actually fit.

**When `loss-min/10` does not apply.** The heuristic assumes a trough *before* the blow-up. On a
model whose loss falls monotonically into divergence the argmin lands in the near-divergence tail,
where noise of ~1e-4 in the smoothed loss flips it between grid points a factor ~2.5 apart.
Measured on ParticleNet / top tagging, three unseeded reruns of the same command gave
**1.39e-1 / 3.82e-2 / 2.64e-2** — a 5x spread against a published `lr: 1e-2` — while
steepest-descent on the same three curves gave **1.91e-3 / 1.32e-3 / 1.91e-3**, a 1.4x spread that
brackets ParT's `1e-3`. `find_lr` now detects a non-interior minimum, warns, and recommends
steepest-descent in that case — and searches for steepest-descent **only before the loss
minimum**, because past it the curve is diverging and one chaotic downward blip yields a
gradient more negative than the honest falling region (measured: an unrestricted search returned
`2.42e+00` on a curve whose real answer was `1.32e-3`).

Across six ParticleNet reruns, steepest-descent returned `1.91e-3 / 1.32e-3 / 1.91e-3 / 1.32e-3 /
1.32e-3` (excluding the one unrestricted-search artefact) — stable to 1.4x and sitting between
ParT's `1e-3` and ParticleNet's `1e-2` — while `loss-min/10` returned `1.39e-1 / 3.82e-2 /
2.64e-2 / 3.38e-2 / 9.05e-2 / 7.07e-2`, a 5x spread every value of which is above the published
`1e-2`. **`find_lr` therefore recommends steepest-descent**, with `loss-min/10` kept as the upper
bracket. That reverses the original preference, on evidence rather than theory — and note the
split held in **both** branches of the interior test, so the curve shape does not rescue
`loss-min/10` here: its trough is shallow and sits just before divergence, where "the minimum"
is only where the fall stops, not an optimum. Revisit if a hybrid disagrees; it is one model on
one dataset, and flipping back is one block in `suggest_lr`'s caller. Where a trough does exist, `loss-min/10` remains the recommendation
for the reason below.

**Number or graph?** Take the printed `loss-min/10` — it is the value the recipes are meant to
carry, and it is stable in `num_iter` in a way the steepest-descent point is not. Use the plot
(`lr_finder.png`) as a veto, not as a second opinion: you want one clean descent into a single
minimum before the divergence. If the curve is flat, double-dips, or is still falling when the
sweep ends, the printed number is summarising a curve that has no well-defined minimum and
should not be trusted — fix the sweep (shorter `num_iter`, higher `end_lr`, larger batch) rather
than eyeballing an lr off the graph. When the shape is clean the two agree anyway, which is the
only case where reading the graph would have told you something new.

**Weight decay.** No automated finder — it can't be range-tested like the LR (its
effect emerges over a full run), so sweep `weight_decay=0,0.01,0.05` (Hydra multirun)
on one model and apply the winner to all. The GT hybrids ship a shared
**`weight_decay: 0.01`** (AdamW) in `tag_gts_and_friends_default`; one value for
the whole family keeps the comparison about architecture. With decoupled decay on
normalized weights it acts mostly as an effective-LR / weight-norm knob
(scale-invariant), so a single value is fair across GNN and transformer parts alike;
norms, biases and class tokens are already excluded (the `ndim<=1` param group) and
framesnets keep `weight_decay_framesnet=0`. The Lion baselines are the exception —
Lion's decay also scales with LR, so the L-GATr (`wd=0.2`, lr=3e-4) and slim
(`wd=2`, lr=3e-5) recipes are the same `lr × wd ≈ 6e-5`; for a Lion run set
`wd ≈ 6e-5 / lr`, not a copied raw number.

**Budget / epochs.** Early stopping is OFF in the shipped recipes (`es_patience: null`),
so the epoch budget is the exact iteration count, not an upper bound. The GT hybrids
encode the fair choice in `tag_gts_and_friends_default`:
**epochs=20** (equal data exposure — derived per model as `epochs × batches_per_epoch`,
not one model's ad-hoc 20-epochs / 200k-iters) and **validate once per epoch** so
best-val checkpointing has equal granularity across the family. Check the val curve
converged; the repo always reports the best-validation checkpoint, so over-budgeting
only costs compute, not accuracy.

**Best-checkpoint metric — why `loss` is the default** (`best_model_metric` in
`tag_default`; `accuracy` is the alternative). Both are measured at the same validation
checkpoints — the difference is in what each number can *resolve*. Accuracy is a
thresholded count, `accuracy_score(labels, round(sigmoid(logits)))`: it only changes when
a jet's score crosses 0.5, so it takes values k/N only, and near convergence — where
decision flips are rare — successive checkpoints tie or differ by counting noise
(σ ≈ √(p(1−p)/N)), leaving the argmax to chance. Val loss aggregates every jet's
continuous margin, so it separates checkpoints accuracy sees as identical (better-calibrated
scores with zero flips), and it tracks the threshold-free report metrics (AUC, rejections
at ε_S = 0.3/0.5/0.8) that the table actually shows. Selection by accuracy optimizes the
one working point (0.5) the paper doesn't report, with a coarser ruler.

**After the sweeps — shakedown order for the config axes.** Once every model's
`batchsize`/`lr` are filled in, exercise the variant knobs in this order:

1. **PlainGraphGPS PE/SE variants first** (cheapest model, most toggles — confirm each
   trains): `model.net.use_edge_attr=true|false` (ParT-pair edge features, ParticleNeXt-style),
   `model.net.use_rwse=true|false` (+ `model.net.rwse_k=K`), `model.net.norm=batch|layer`.
2. **Then every HYBRID under both graph metrics**: `model.net.knn_metric=deltaR|minkowski`
   (the eight hybrids **and the `tag_particlenet` baseline** — its wrapper passes the local
   four-momenta as `v=`, and lloca's `ParticleNet` ranks layer-0 on the Minkowski interval.
   Production tree only: `config_quick`'s `tag_particlenet.yaml` has no `knn_metric` key, so
   a bare `run.py` override fails composition there; via `train.sbatch` it is fine — that
   defaults to `-cp config`. See docs/OSCAR.md §shakedown for the command forms)
   (minkowski is the Lorentz-invariant graph; deltaR the eta–phi one).
3. **Then the LLoCa-canonicalized models under PD frames**: `model/framesnet=learnedpd`
   on the Plain and ParticleNet-ParT hybrids and `tag_particlenet` (the
   internally-equivariant CGENN / LorentzNet hybrids stay on `identity`).

(The full knob catalogue lives in `todo.md` §3 and `docs/ablations.md`.)

---

## 7. Frames, xformers, and avoiding it

The built-in Transformer / L-GATr taggers and the `lgatr` frame predictor use
xformers' memory-efficient attention (saves ~2× RAM on variable-length jets); on
a recent NVIDIA GPU you normally just `pip install xformers` and it's the upstream-default
backend. Note "default" ≠ "fastest": for ragged jets **flash** (flash-attn v2
kernels, fp16/bf16) is typically the fastest backend — and the only one besides
native that honors attention-weight dropout — so `model.attention_backend=flash`
is a legitimate first choice, not just a fallback. The gap is small, though:
xformers' `memory_efficient_attention` is itself a *dispatcher* that usually
selects the same flash kernels internally when dtype/head-dim allow, so when it
does the two are near-equal — the practical differences are the dropout arg name
(`p` vs `dropout_p`), kernel-version lag (new flash releases land in flash-attn
first), and a fp32 fallback path only xformers has. (lgatr 2.0
changes this calculus: its compiled-xformers custom ops become the flagship fast
path under torch.compile — see `docs/lgatr2-migration.md`.)
The new **GraphGPS non-equivariant** models use plain `torch.nn.MultiheadAttention`,
so they need no xformers at all. If you do want a learned framesnet without
xformers, use the **MLP frame predictor**:

```bash
python run.py model=tag_PlainGraphGPS model/framesnet=learnedpd \
    model/framesnet/equivectors=equimlp     # MLP frames, no xformers (vs =lgatr)
```

(`equivectors` ∈ {`equimlp`, `pelican`, `lgatr`}; `equimlp` is the lightest and
xformers-free.) The internally-equivariant hybrids use identity frames and never
touch xformers in the framesnet — their L-GATr stages attend with dense masks
(native torch attention), so all 8 hybrids run on a GPU without xformers installed.
The only models that don't are the four baselines whose configs pin
`attention_backend: xformers` (`tag_transformer`, `tag_top_transformer`,
`tag_lgatr`, `tag_slim`); on an xformers-free install, run those with
`model.attention_backend=flash` (needs a WORKING flash-attn — always test
`python -c "import flash_attn"` first *in a clean environment*: an earlier
"image copy is ABI-broken" verdict turned out to be a leaked user-site torch
shadowing the container's, see the ~/.local note in docs/OSCAR.md) or `=flex`
(pure-torch FlexAttention, needs torch≥2.7 — slower, and its torch.compile
path is version-sensitive, so smoke-test it on your GPU first).

---

## 8. Multiple trials and the results table

- **One `run.py` invocation = one trial** (`run_idx=0`) and emits one table row.
- **Several trials of the *same* model — canonical for campaign/table rows: the
  shared-dir fresh-trial warm start.** Re-run the *same* experiment as a **fresh-trial
  warm start** — point `-cp`/`-cn` at the saved run config and pass
  `warm_start_idx=<prev run_idx> warm_start_load=false`. It increments `run_idx`, shares
  the run directory, appends this trial's raw scalars to
  `runs/<exp>/<run>/table_metrics_*.json` (lineage-keyed: a later continue-training
  *extends its parent's row* instead of faking an extra trial), and starts from a **new
  random initialization** with a fresh optimizer/scheduler. The final row reads
  `… & <iters> [N trials] & $acc ± σ$ & …`. This is canonical because the grouping is
  *explicit* — a directory IS its ensemble: continuations are bookkept, a pinned seed is
  caught at launch (degenerate-trials warning), and the failure mode of a forgotten flag
  is a visible `[N−1 trials]`, never a silently wrong row. Raw per-trial values persist in
  the JSON, so switching statistic or dropping a bad trial later needs no retraining.
  Trials of one variant run sequentially, but at campaign scale that does not stretch the
  makespan (many variants × few GPUs: the queue, not one model's 3-trial chain, is the
  critical path).
- **Independent run dirs also group at aggregation** — submit the identical command N
  times (plain `sbatch train.sbatch tag_X`); `utils/aggregate_table.py` groups run
  directories by `(task, model, frames, kNN)` and forms `mean ± std` across them at parse
  time, and each run's own log keeps reporting *that* run. Use this where wall-clock for
  ONE variant matters (three parallel jobs beat three chained ones — e.g. adding a single
  row late), or as recovery when trials were accidentally scattered across dirs. Its limit
  is that the grouping is *inferred* from the table key, so the aggregator refuses to pool
  (keeps the newest, with a note) whenever inference could lie: a group already containing
  an in-run `mean ± std` row (double-counting), disagreeing `iters`/`params`/`FLOPs` cells
  (different experiments sharing a key — budget/width ablations), or identical metrics
  across dirs (a pinned seed produced clones; nothing warns at launch on this path, unlike
  warm starts). One case it cannot detect: a deliberate continue-training run in its *own*
  dir looks like a fresh trial — keep continuations in their parent's dir (lineage handles
  them) or outside the aggregated tree.
  **Do NOT use a plain warm start (the `warm_start_load=true` default) for trials**: that
  reloads the previous model *and* the finished scheduler, so the "trial" is a correlated
  continuation of the same training — and the reloaded cosine steps past `T_max`, ramping
  the lr back *up* toward its maximum over the run. Plain warm starts are for eval-reload
  and deliberate continue-training (with `training.scheduler_scale`) only.
  Seed bookkeeping: with the default `seed=null` every trial draws a fresh init
  automatically; if you pin `seed`, vary it per trial (`seed=1`, `seed=2`, …) or all
  trials are identical. (Batch *order* is sampler-seeded and identical across trials
  either way.)
- **Different models do *not* merge** into one table — each lands in its own run
  directory with its own row. To build a comparison table, run
  `python utils/aggregate_table.py --runs runs --split test --out comparison.tex` (it collects each
  run's row and emits ONE LaTeX table PER TASK -- toptagging / toptagxl / jctagging report
  different metric columns, read from each run's saved `config.yaml` `exp_type`; JetClass
  runs emit an aggregator-compatible `table test:` row too), or do it by hand
  from the printed `table test:` lines (`grep "table test:" runs/*/*/out_0.log`).

For 3 seeds of a model: launch the run, then fresh-trial warm-start it twice more (same
`exp_name`/`run_name`, `warm_start_load=false`). For the heavy `CGENNLGATrGraphGPS`
(~4.5e11 FLOPs/jet -- roughly a day per trial on an H100-class GPU, longer on anything
slower; the run prints its own estimate early) budget accordingly; the slim model is
~300× lighter.

---

## 9. Tests

```bash
pytest tests/experiments/test_tag_equivariance.py -q   # invariance (32 cases)
pytest tests/experiments/test_tag_flops.py -q -s       # FLOPs + param counts
```

`test_tag_equivariance.py` asserts three properties on the `config_quick` models:
azimuthal invariance for every hybrid (Minkowski kNN), full SO(3)/Lorentz
invariance for the internally-equivariant ones (spurions off, fully connected,
float64), and LLoCa-frame invariance for the canonicalized ones under both learned
frames (`learnedpd` and `learnedso13`; `learnedpd` carries a looser float64 bound for
its polar-decomposition boost-precision floor). The unit-test workflow
(`.github/workflows/tests.yaml`) now runs the equivariance + invariance suites on every
push to main, on manual dispatch, and when the `ready for review` label is applied to a PR
(a push to an already-labeled PR still skips — re-apply the label or dispatch manually;
running locally before pushing remains the fastest gate).
(`test_tag_flops.py` stays out of CI: its learned-frame cases need a CUDA-matched
xformers build.)

---

## 10. Gotchas

- **The default training recipe** (`tag_gts_and_friends_default`, AdamW) now
  fits the new models — just set an LR/batchsize from `utils/find_lr.py` (§5/§6). Baselines
  still need their own `training=top_<baseline>` (e.g. Lion for the transformer).
- **`use_float64`** is `false` in production (float32); the equivariance tests flip
  it on for the exact-invariance checks. The kNN distance computations follow the
  run dtype.
- **kNN graphs are slightly discontinuous** (a transform can flip a near-tied
  neighbour), so as-configured models are azimuthally invariant only to ~1e-3; this
  is inherent to every kNN GNN and vanishes with learned frames or a fully connected
  graph. It does not affect training.
- **`norm: batch` vs `layer`** on the non-equivariant GPS models: `batch` is the
  GraphGPS default; `layer` is the padding-safe alternative for variable jet sizes.
  The equivariant GPS models use the geometry-native norm (EquiLayerNorm / RMSNorm)
  and cannot use BatchNorm on their vector/multivector streams.
