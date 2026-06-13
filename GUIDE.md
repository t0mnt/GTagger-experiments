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
pytest tests/experiments/test_tag_equivariance.py -q     # 24 invariance checks
python run.py -cp config_quick -cn toptagging save=false # one quick training
```

`config_quick/` mirrors `config/` with tiny models/data — ideal for sanity checks
and reading along with print statements. `config/` is the real training setup.

---

## 2. Repo layout

| path | what |
|---|---|
| `run.py` | entry point: builds an experiment from a hydra config and trains/evaluates it |
| `find_lr.py` | LR range test + optional GPU batch-size finder (see §6) |
| `config/` | real configs; `config_quick/` the tiny mirror |
| `config/model/tag_*.yaml` | one file per tagger (model definition only) |
| `config/training/top_*.yaml` | training budgets / optimizers / schedules |
| `config/model/framesnet/` | LLoCa frame predictors (for non-equivariant models) |
| `experiments/baselines/` | the network implementations |
| `experiments/tagging/wrappers.py` | the wrapper that adapts each net to the tagging pipeline |
| `tests/experiments/` | `test_tag_equivariance.py`, `test_tag_flops.py` |

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

```bash
# a non-equivariant hybrid, made equivariant with learned frames
python run.py model=tag_PlainGraphGPS model/framesnet=learnedso13

# an internally-equivariant hybrid (identity frames; nothing to set)
python run.py model=tag_LorentzNetLGATrSlimGraphGPS

# pick a training budget + optimizer (see §5/§7), the mini dataset, a GPU
python run.py model=tag_ParticleNetParTGraphGPS training=top_ParT \
    data.dataset=full gpus=1
```

Useful overrides: `data.dataset={full,mini}`, `training.iterations=…`,
`training.batchsize=…`, `training.lr=…`, `gpus=N`, `save={true,false}`,
`model.net.knn_metric={deltaR,minkowski}`, `model.net.num_blocks=…`.

Each run prints a paste-ready LaTeX table row at the end:
`table test: <Model> & <frames> (<iters>) & <params> & <acc> & <auc> & … & <kNN>`.

---

## 5. Configs: model vs training

A `config/model/tag_*.yaml` is **model definition only** — it has no LR, optimizer
or budget. Those come from the **training** config, selected separately. If you
don't pass `training=…`, the top-tagging default is `top_transformer`
(**Lion, lr=3e-5, weight_decay=2, 300k iters**), which was tuned for the plain
transformer and is *not* appropriate for the GNN-hybrids. Always pick a training
config (or override the keys) for the new models — see §7.

Existing training recipes worth cloning: `top_ParT` (Ranger, lr=1e-3, 20 epochs),
`top_lorentznet` (AdamW, lr=1e-3, 35 epochs), `top_lgatr` (Lion, lr=3e-4, wd=0.2),
`top_particlenet` (lr=1e-2). Make a `config/training/top_<yourmodel>.yaml` that
`defaults: [tag_default]` and sets `optimizer`, `lr`, `weight_decay`, `iterations`.

---

## 6. Choosing hyperparameters

**Learning rate (and GPU batch size) — `find_lr.py`.** Runs a Leslie-Smith LR
range test using the model's exact optimizer/param-groups/clipping, and reports a
robust `loss-min/10` peak LR.

```bash
# LR only
python find_lr.py -cp config -cn toptagging model=tag_CGENNLGATrGraphGPS save=false

# on a GPU: fit the batch size first, then sweep the LR at that size
python find_lr.py -cp config -cn toptagging model=tag_LorentzNetLGATrSlimGraphGPS \
    save=false +lr_find.find_batch_size=true
```

With `+lr_find.find_batch_size=true` it doubles the batch size until CUDA OOM,
backs off to `bs_safety` (default 0.85) for optimizer-state + fragmentation
headroom, then prints both the batch size and the LR, e.g.
`-> reuse with: training.batchsize=2048 training.lr=3.1e-04`. Verify the printed
batch size with a short real run before a multi-day job (it measures one
fwd+bwd). Knobs: `+lr_find.{bs_start,bs_max,bs_safety,num_iter,end_lr}`.

**Weight decay.** There is no automated finder — sweep it on validation. But note
the L-GATr (`wd=0.2`, lr=3e-4) vs slim (`wd=2`, lr=3e-5) values are *the same
regularization*: both use **Lion**, whose decoupled decay scales with the LR, so
the effective decay is `lr × wd ≈ 6e-5` in both. So don't copy the raw number —
once `find_lr.py` gives you an LR, set `wd = 6e-5 / lr` (for Lion) as a starting
point, then grid {0.5×, 1×, 2×} on validation AUC. The decay applies to all weight
matrices (GNN + transformer); norms, biases and class tokens are already excluded
by the optimizer, so there's no separate "transformer-only" decay to set.

**Budget / epochs.** Early stopping is on (`es_patience`), so the iteration count
is an upper bound — but its patience is large, so in practice the budget *is* the
cap. For a fair comparison, give every model you're comparing the **same generous
epoch budget** (equal data exposure) rather than copying one model's ad-hoc number
(ParT's "20 epochs" or L-GATr's "200k iters" are each tuned for that model). Check
the val curve converged; the repo always reports the best-validation checkpoint, so
over-budgeting only costs compute, not accuracy.

---

## 7. Frames, xformers, and avoiding it

The built-in Transformer / L-GATr taggers and the `lgatr` frame predictor use
xformers' memory-efficient attention (saves ~2× RAM on variable-length jets); on
an H100 you normally just `pip install xformers` and it's the recommended backend.
The new **GraphGPS non-equivariant** models use plain `torch.nn.MultiheadAttention`,
so they need no xformers at all. If you do want a learned framesnet without
xformers, use the **MLP frame predictor**:

```bash
python run.py model=tag_PlainGraphGPS model/framesnet=learnedpd \
    model/framesnet/equivectors=equimlp     # MLP frames, no xformers (vs =lgatr)
```

(`equivectors` ∈ {`equimlp`, `pelican`, `lgatr`}; `equimlp` is the lightest and
xformers-free.) The internally-equivariant hybrids use identity frames and never
touch xformers in the framesnet.

---

## 8. Multiple trials and the results table

- **One `run.py` invocation = one trial** (`run_idx=0`) and emits one table row.
- **Several trials of the *same* model** accumulate into `mean ± std` automatically:
  re-run the *same* experiment as a **warm start** (it increments `run_idx`, shares
  the run directory, and appends to `runs/<exp>/<run>/table_metrics_*.json`). The
  final row then reads `… (iters) [N trials] & $acc ± σ$ & …`.
- **Different models do *not* merge** into one table — each lands in its own run
  directory with its own row. To build a comparison table, collect the printed
  `table test:` lines from each run's log (`grep "table test:" runs/*/*/out_0.log`)
  and paste the LaTeX rows together.

For 3 seeds of a model: launch the run, then warm-start it twice more (same
`exp_name`/`run_name`). For the heavy `CGENNLGATrGraphGPS` (~4.5e11 FLOPs/jet,
~a day per trial on an H100) budget accordingly; the slim model is ~300× lighter.

---

## 9. Tests

```bash
pytest tests/experiments/test_tag_equivariance.py -q   # invariance (24 cases)
pytest tests/experiments/test_tag_flops.py -q -s       # FLOPs + param counts
```

`test_tag_equivariance.py` asserts three properties on the `config_quick` models:
azimuthal invariance for every hybrid (Minkowski kNN), full SO(3)/Lorentz
invariance for the internally-equivariant ones (spurions off, fully connected,
float64), and LLoCa-frame invariance for the canonicalized ones under a learned
`learnedso13` frame. Run these locally as your gate — CI does not pick up
`tests/experiments/`.

---

## 10. Gotchas

- **Default training config is mistuned** for the new models — always set
  `training=…` and an LR from `find_lr.py` (§5/§6).
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
