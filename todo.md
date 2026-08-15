# TODO — outstanding work

A running checklist for finishing the graph-transformer hybrid study and connecting it to
a paper. Grouped by "before training", "open design decisions", and "paper release".

---

## 1. Before training — fill in the training configs

The 8 hybrid recipes are skeletons with required `???` keys:
`config/training/top_{Plain,ParticleNetParT,CGENNLGATr,LorentzNetLGATrSlim}{GraphTrans,GraphGPS}.yaml`.

The 8 GT recipes now inherit `tag_gts_and_friends_default` (shared `epochs=20` + `scheduler=CosineAnnealingWarmup`);
only **`batchsize`, `lr`** remain `???` per model (optionally `weight_decay`):

- [ ] **Validate the finder on a published baseline FIRST**: `utils/find_lr.py -cn toptagging
      model=tag_particlenet training=top_particlenet save=false` at ParticleNet's fixed
      batchsize (512) should land within an order of magnitude of its published `lr: 1e-2`.
      The finder reports loss-min/10, not the authors' tuned value, so agreement in scale is
      the pass criterion -- a wildly different answer means the tool (or the environment) is
      wrong before eight unpublished models depend on it.
- [ ] **Reproduce a known result before the campaign**: one `save=false` ParticleNet run under
      its published recipe, checking test accuracy/AUC against the published numbers. It
      exercises data, loader, model and evaluation end to end against a known answer.
- [ ] `batchsize` ← `utils/find_lr.py +lr_find.find_batch_size=true +lr_find.bs_max=512` (largest
      power-of-two that fits YOUR GPU, bounded at 512 -- the finder measures it, so the answer is
      per-machine, not a number to copy. The bound is the epoch-budget argument: a larger batch
      buys FEWER optimizer steps for the same data exposure. GUIDE.md, under find_lr.)
- [ ] `lr` ← `utils/find_lr.py` (reported loss-min / 10).
      (`weight_decay` tuning moved to `docs/ablations.md` "Training-side minor tunes" —
      the shared 0.01 ships as the decided default.)
- [x] `epochs` (shared data-exposure budget) and `scheduler` are **decided** in `tag_gts_and_friends_default`
      (see §2); `iterations` is auto-derived at runtime (`_resolve_epoch_budget`).

## 2. Training-recipe decisions (fairness)

**Scheduler — DECIDED: `CosineAnnealingWarmup`** (set in `tag_gts_and_friends_default`), shared across the GT
hybrids, tuning only lr/batchsize/weight_decay per model — warmup matters for the transformer/
equivariant layers, and one shared schedule isolates architecture for the hybrid-vs-hybrid table.
`OneCycleLR` is the repo-proven alternative (it is warmup→cosine too) but its warmup is cosine-shaped
and it cycles AdamW's β₁ by default — minor confounds. The published **baselines** (ParT/ParticleNet/
L-GATr) keep their own recipes as reference rows (you can't out-tune the originals); optionally re-run
them under the shared schedule for one apples-to-apples row. Annealing to ~0 is desirable (the
end-of-training low-lr phase gives the best final val metric); per-module heterogeneity is handled by
warmup (peak) + AdamW + `lr_factor_framesnet`, not by raising the floor. Set a small
`cosanneal_eta_min` (e.g. 1e-6) only as a hedge against a slightly over-long schedule.

**Epochs vs iterations.** Now automated: set `training.epochs` and `iterations` is derived per model as
`epochs * len(train_loader)` (the exact batch count — reflects batchsize, subsampling, drop_last). This
equalizes **data exposure** (the standard fairness axis); note equal epochs ≠ equal gradient *updates*
(a larger-batch model gets fewer steps), and each model still anneals fully over its own iteration count.
- [x] Epoch budget **decided: `epochs=20`** (ParT-standard) in `tag_gts_and_friends_default`, shared by all 8 GT
      hybrids; bump to ~30 if they underfit (CLI: `training.epochs=30`).
- [ ] Keep the baselines' published-recipe numbers as a separate reference row in the table.
- [ ] **Paper — methods sentence for the budget.** Something like: *"All hybrid taggers are trained
      for an equal data exposure of 20 epochs on the top-tagging dataset (5 epochs on JetClass,
      the ParT-standard exposure). Because batch sizes are tuned per model, the iteration count is
      derived at run time as epochs × batches/epoch, and each model's warmup–cosine schedule
      anneals over its own iteration count. Equal epochs implies unequal optimizer-step counts
      across batch sizes; we follow the community convention of fixing data exposure (ParT: 20
      epochs; LorentzNet, PELICAN: 35). Baseline reference rows are trained under their published
      recipes."* Optionally cite the recipe drift this replaces (published iteration counts
      correspond to 20.5 / 21.3 / 32 epochs for ParT / L-GATr / the Lion transformer).

**Best-checkpoint metric.** `best_model_metric` (in `tag_default`): `loss` (default, lowest val loss) or
`accuracy` (highest val accuracy). Selection-by-loss and -by-accuracy usually track but can diverge late;
the toggle only changes which checkpoint `es_load_best_model` keeps/reports.

## 3. Ablations — CLI recipes (for the paper's ablation tables)

All via Hydra overrides on `run.py` (use `-cp config` for the full configs). Every override is
recorded per-run in `config.yaml` + the flattened MLflow params, so any sweep is reconstructable
from the run dir. **Surfaced in the results table** (`utils/aggregate_table.py` `COLUMNS`): only `frames`
(framesnet) and `kNN` (`knn_metric`); everything else (knn_k, num_layers/num_blocks, bias,
pair_input_dim, use_rwse, use_edge_attr, …) lives only in config.yaml / MLflow. Rows are grouped into ONE
table per task (toptagging / toptagxl / jctagging — different metric columns; `exp_type` read
from each run's config.yaml; JetClass emits an aggregator-compatible row too). To put a knob in
the head-to-head table, add it to `utils/aggregate_table.py`'s per-task `COLUMNS` legend **and** the
per-run `table …:` log line that the regex reads.

- **kNN graph (all networks).** count `model.net.knn_k=K` (CGENN uses `model.net.k=K`); metric
  `model.net.knn_metric=deltaR|minkowski`; fully-connected = k ≥ P−1 (`9999`, or `model.net.k=null`
  for CGENN). minkowski is the Lorentz-invariant graph (needed for full-group invariance); deltaR is
  the eta–phi graph.
- **LLoCa on/off (the non-equivariant backbones: Plain, ParticleNet-ParT).** on =
  `model/framesnet=learnedso13` (learned SO(1,3) frames → tensorial transport engaged); off / "do
  nothing" = `model/framesnet=identity` (no-op, bit-identical plain backbone). Symmetry-budget
  variants: `learnedso3` (rotations), `learnedso2`, `learnedz`, `learnedrest`, `learnedpd`;
  `randomlorentz` is the data-augmentation baseline. (CGENN / LorentzNet are already internally
  equivariant → leave on `identity`.)
  - **DECIDED: `learnedpd` — keep upstream's default. This REVERSES an earlier call here for
    `learnedso13`; the reversal is recorded because the reasoning is the useful part.**
    The so13 case rested on three claims. Checked against `heidelberg-hepml/lloca-experiments`,
    two of them do not survive:
    1. *"pd's unbounded gamma is a fragility, and our config disables the guard."* **Withdrawn.**
       Their `config/model/framesnet/learnedpd.yaml` ships `gamma_max: null` and
       `gamma_hardness: 10` — identical to ours. So ours is not a misconfiguration, it is their
       configuration, and the authors' own published runs use PD with the boost regulator off.
       The failure mode is structurally real (gamma = E/m, and
       `lloca/framesnet/equi_frames.py:550-551` returns the boost untouched when `gamma_max` is
       None) but evidently does not bite on jet data, or they would have set it. Monitor rather
       than switch: `gamma_mean` and `reg_gammamax` are already tracked in `_init_metrics`.
    2. *"Equal capacity, so 'less flexible' costs nothing."* **Backwards.** Both reach
       SO+(1,3) — pd via the boost x rotation polar decomposition, so13 via 4d orthonormalization
       — but equal REACHABILITY is not equal optimization. pd lets the network steer boost and
       rotation independently; so13 couples them through the orthonormalization. The only
       evidence anyone has on that axis is the authors', who trained both and made pd the
       default. "Unquantified" is not "absent", and there was no counter-evidence on the axis
       that decides it, which is accuracy.
    3. *The invariance floor (~5e-3 pd vs ~1e-6 so13).* **Stands as a fact, but it is not a
       performance argument.** It is measured under an adversarial float64 test that applies
       random boosts; nothing in the training or evaluation pipeline boosts events, they arrive
       in the lab frame. So it bounds how tightly the invariance CLAIM can be stated, not how
       well the model tags. It belongs in the methods sentence (§4e), not in this decision.
    Net: pd is the validated-at-scale choice and the apples-to-apples one against published
    LLoCa numbers, and the deviation would have needed evidence on accuracy that nobody has.
    **Consequence: the "on =" line above should read `learnedpd`**, with `learnedso13` demoted
    to the ablation row — the inverse of what was written, and cheaper, since so13 needs no
    extra knob.
  - [ ] **One `learnedso13` ablation row**, on one backbone (ParticleNet-ParT GraphTrans is
        the natural pick), to show the frame family is not load-bearing for the conclusions —
        and because it is the variant with the tighter invariance floor, which makes it the
        honest companion to the methods sentence about pd's.
- **ParT pairwise bias (ParticleNet-ParT GraphTrans + GraphGPS).** `model.net.bias=true|false`;
  `model.net.pair_input_dim=1|4|5|8` selects how many QCD interaction features (1=lnΔ; 4=+ln kT,
  ln z, ln m²; 5=+lnΔs²; 8=+cosθ,Δy,Δφ — see `pairwise_lv_fts`; the weaver feature ladder jumps
  5→8 when adding cosθ/Δy/Δφ, so 6/7 are not valid — `assert len(outputs)==num_outputs` enforces
  this). The learned weights compensate, so the bias stays compatible with the frame transport.
- **GraphGPS PE/SE (Plain GraphGPS).** ParT-pair edge features `model.net.use_edge_attr=true|false`
  (lnΔ, ln kT, ln z, ln m² routed through the MPNN edge channel, ParticleNeXt-style); structural encoding `model.net.use_rwse=true|false`
  (+`model.net.rwse_k=K`); norm `model.net.norm=batch|layer`. CGENN GraphGPS relative edge features:
  `model.net.use_explicit_edge_features=true|false`.
  - [ ] Add general RWSE support to all GPS models (currently PlainGraphGPS-only) if it shows
        success on Plain. Note on `rwse_k`: GraphGPS uses k=20 on sparse molecular graphs
        (ZINC, ~23 nodes, diameter ~10); jet kNN is dense and small-diameter so the walk mixes
        fast and ~4–8 steps likely capture the useful return-probability structure (higher k adds
        near-saturated, redundant dims) — sweep `{4,8,16}` on Plain before generalizing.
  - [ ] **PE/SE is a PRE-campaign gate, not a post-hoc ablation** (decided): whether the
        headline GPS rows ship with RWSE changes what those models ARE, so decide it before
        the primary runs — Plain-GPS ± RWSE (k=8) at tuned lr/batch on top tagging; if RWSE
        wins meaningfully, port it to the other three GPS models (item above) BEFORE the
        campaign; if null/negative, keep off and report as the ablation. LapPE stays post-hoc
        only (expected negative, O(P^3) — never gates the campaign). A Plain-GPS result
        transfers to the other **static-graph** GPS models (CGENN, LorentzNet); ParticleNet-ParT
        GPS rebuilds its kNN per layer, so a static structural encoding means something else
        there — test it separately or leave it off.
- **Depth (transformer / GPS blocks).** `model.net.num_layers=N` (Plain, ParticleNet-ParT) /
  `model.net.num_blocks=N` (CGENN, LorentzNet). The depth curve is the "can the transformer
  compensate for a weaker GNN" story → a performance/efficiency section (room to discuss BigBird /
  sparse attention and the flex / xformers / flash backends the L-GATr stack already supports).

Other knobs worth a sweep: width/capacity (`hidden_*_channels`, `dim`, `gnn_dims`, `embed_dim`);
input-skip (`model.net.use_input_concat`); residual-symmetry spurions on the equivariant models
(`model.net.beam_spurion`, `model.net.add_time_spurion`); dropout — incl. the uniform
`model.net.attn_dropout` knob on all four GPS models (attention-weights dropout via sdpa;
GraphGPS ships 0.5, jet lineage ships none — a one-override family row on top-tagging). Depth and width move the param
count (a table column) — pair them with FLOPs/time for a fair efficiency plot.

**Off by design: global spectral PE/SE (LapPE / SignNet / eigenvalue SE).** A LapPE node-encoder
now EXISTS behind `use_lappe` (+`lappe_k`) on PlainGraphGPS — implemented exactly as the
demonstrate-the-negative-result toggle described below — but it ships OFF, and the default-off
choice is deliberate — worth a sentence in the paper. (1) A jet has *no canonical graph*:
the kNN adjacency/Laplacian is something we construct (eta–phi or minkowski kNN, and rebuilt in
feature space for EdgeConv), so anything read off its spectrum encodes our graph-building choice, not
the physics. (2) A jet is *not position-blind*: particles carry (Δη, Δφ, pT, E) — a physically
meaningful absolute PE that LapPE would only try to reconstruct (this is exactly why ParT ships no
positional encoding). (3) Under LLoCa a PE must be a *Lorentz invariant* to preserve invariance, but
LapPE eigenvectors have sign/basis ambiguity **and** graph dependence, so they are not clean
invariants (SignNet exists only to patch the sign ambiguity). The encodings that transfer are the
relative QCD pairwise features (lnΔ, ln kT, ln z, ln m²) and, on a *static* graph, RWSE — both already
exposed above. If a reviewer wants the negative result demonstrated, run PlainGraphGPS with
`model.net.use_lappe=true` (the toggle is implemented; sign-flip augmentation handles the
eigenvector ambiguity) — expected to show it doesn't help, per the argument above.

## 3a-bis. JetClass cost: CGENN-GraphGPS dominates the campaign

Forward FLOPs/jet at P=50, measured by `tests/experiments/test_tag_flops.py` (same convention
as Table 2 of arXiv:2512.17011 — five of its rows reproduce exactly, e.g. ParT 211M,
ParticleNet 413M, LorentzNet 676M, L-GATr 2060M):

| model | GFLOPs/jet | | model | GFLOPs/jet |
|---|---|---|---|---|
| LorentzNet-slim GraphTrans | 0.36 | | PlainGraphGPS | 0.97 |
| PlainGraphTrans | 0.42 | | LorentzNet-slim GraphGPS | 1.00 |
| ParticleNet-ParT GraphTrans | 0.65 | | ParticleNet-ParT GraphGPS | 1.22 |
| CGENN GraphTrans | 6.97 | | **CGENN GraphGPS** | **62.9** |

**CGENN-GraphGPS alone is ~84% of the eight-model total.** Not a misconfiguration, but *not* the
"same model, run more often" it looks like either. The two rows differ on **layer count and width
at once**, and the widths are not equal:

| | CGENN layers | mv width | `phi_x` GP shape | all `phi_x`, GFLOPs/jet |
|---|---|---|---|---|
| GraphTrans | `cgenn_layers: 3`, once before the stack | `cgenn_hidden_x: 8` | 31 → 8 | 4.9 of 6.97 |
| GraphGPS | 1 per block × `num_blocks: 10` | `hidden_mv_channels: 16` | 55 → 16 | 57.7 of 62.9 |

**11.8×** on the CGENN stage = **3.33×** (10 layers vs 3) × **3.55×** (per layer). The per-layer
factor is the width: the dense GP costs `out × 16 × in × 256`, `out` doubles 8→16, and `in`
follows it because `message_x` concatenates `[x_i, x_j, x_i−x_j]` (3×8+7=31 → 3×16+7=55).
GPS inherits its CGENN width from the block width and declares no `cgenn_hidden_x` at all.

11.8× on the stage lands at 9.0× on the whole model (62.9/6.97) because the transformer half is
common to both and does not grow. Two further facts worth keeping: **96%** of the CGENN stage is
the *message* GP `phi_x`, which runs **per edge** (P×k = 800/jet) not per node (50) — a 16×
site multiplier the attention branch does not pay; and the same GraphTrans→GPS change costs the
other three hybrids only 1.9–2.8× (Plain 0.42→0.97, LorentzNet-slim 0.36→1.00, ParticleNet-ParT
0.65→1.22), so the 9× is CGENN's per-site cost, not something about GraphGPS.

**Is there headroom left inside `phi_x`?** Measured at the real GPS shape (6400 edges, 55→16,
4 CPU threads), the layer is **95–97.5% the geometric product** — `linear_right` (55→55) and
`linear_left` (55→16) are 1.3–2.8% each. So the standard MPNN trick of hoisting the linear
maps out of the edge loop onto nodes (legitimate here: `linear(cat[x_i, x_j, x_i−x_j, e])`
= `(L1+L3)x_i + (L2−L3)x_j + L4·e`, so both could run on 50 sites instead of 800) is worth
~2% and is not worth doing. The GP itself is already at its efficient factorization: the
`input ⊗ linear_right(input)` form is diagonal in the channel index, and expanding it to the
full `(n,p)` bilinear form would cost 55× more. What is actually left, in order:
- **bf16/AMP.** The GP is bandwidth-bound (the `(E, 55, 16, 16)` outer product is ~360 MB per
  layer at B=8), so halving element size is close to a straight 2×. Blocked on CGENN autocast
  guards — port lgatr 2.0's `@minimum_autocast_precision(torch.float32)` pattern, which keeps
  the sensitive reductions in fp32 while the bulk runs bf16. **Biggest remaining exact lever.**
- **Re-benchmark `gp_impl` on the H100 at this shape, not at model level.** Sparse cuts MACs
  16× (3.60M → 239k per edge) but trades a GEMM for a gather-reduce; on 4 CPU threads it comes
  out **2× slower than einsum** here. That is a locality result, not a GPU result — but it says
  the 16×-fewer-FLOPs argument is not self-evidently winning, and `matmul` hits TF32 tensor
  cores while `sparse` does not. All three ship behind the knob, so this is a cheap check.
- Everything else is architecture, and draws the same objection as `k` and `cgenn_hidden_x`
  below: width 16→8 (3.5×), dropping the `x_i−x_j` channel block (1.4×), `fc`→`gpmlp`
  (channel-wise GP, ~55×). Ablations, not faster implementations of this row.

Calibrating against that table's own h/GFLOP (61–210, median ~83; L-GATr improves 81 → 28 under
lgatr 2.0), CGENN-GraphGPS lands at **~2000–5000 GPU-h** for a full-JetClass-equivalent run —
weeks to months on one H100, and still weeks on four. **This needs a decision before the JetClass
campaign, not during it** — but the fix is implementation, not architecture. Measured levers, all
of which leave the model identical (details in `docs/cgenn-compile.md`, dev branch): replace the
`einsum` geometric product with lgatr 2.0's outer-product + matmul form (**5.2× on the GP, verified
bit-identical**; the GP's share of runtime varies with shape, impl and device, so read it off the
profile you are optimising, not off a number quoted here); the data-movement rewrites (`copy_` is **38%** of
runtime, 2071 calls per forward); sparse-GP (the Cayley table is 6.2% dense); `torch.compile`.
Shrinking the model — striding the local branch, cutting `k` or `cgenn_hidden_x` — is **not** on
this list: it makes the row a smaller model racing full-depth rivals, which is an ablation, not a
speedup. Top tagging is unaffected either way.

## 3b. Paper points (observations from the build/audit)

- GPS attention *rescues* a symmetry-breaking local graph: spurions off (float64, Lorentz),
  deltaR-vs-minkowski kNN breaks invariance ~14× more in GraphTrans (3.9e-3) than GraphGPS
  (2.8e-4; minkowski ≈ machine-exact both) — the invariant global L-GATr branch runs parallel
  in GPS and absorbs a non-invariant metric.
  **Scope, state it when the claim is made:** spurions-off is the *correct* isolation, not a
  convenience — with the shipped spurions on, the model breaks the symmetry deliberately and
  by far more than the metric does, so the metric's contribution could not be attributed. The
  consequence is that 14× is a statement about the MECHANISM (a parallel invariant branch
  absorbs a non-invariant graph), not about the shipped models' invariance, which is broken on
  purpose. Do not quote the ratio as a property of a campaign row.
- ParticleNeXt edge features vs ParT pairwise bias = same encoding at two sites; in GPS one is
  likely redundant, and which one localizes the pairwise signal (per Spinner). See ablations.md.
- Directed kNN for message passing, symmetrized graph for RWSE/LapPE — the GraphGPS/MNIST
  convention (parity point for the "faithful port" claim).
- RWSE k for jets is short (~8) vs GraphGPS's 20 — dense small-diameter jet kNN mixes fast.

## 4. Open design decisions / discrepancies

- [x] **CGENN-LGATr GraphGPS local branch under-fed vs its GraphTrans cousin** — fixed in two
      passes: it now injects, under `use_explicit_edge_features` (default on), all three static
      signals the GraphTrans CGENN stage does — the relative-momentum edge multivectors
      `[pᵢ−pⱼ, rawᵢ, rawⱼ]`, **and** the raw mv / raw scalar inputs re-injected as per-node
      attributes (`node_attr_x` in `theta_x`, `node_attr_h` in `theta_h`) every layer. (The first
      pass added only the edge features; the node attributes were the missing two-thirds.)
      Equivariance 3/3 (xy-rotation + full-group rotation + Lorentz boost).
- [x] CLS readout frame: **jet frame** (covariant, boost into the jet rest frame). Decided.
- [x] LLoCa transport made **strictly additive** (identity frames bit-identical to the plain backbone).
- [x] Scheduler: shared **CosineAnnealingWarmup** available; **early termination off** (`es_patience=null`),
      best-validation checkpoint still reported.

### Resolved audit trail -> docs/audit-ledger.md

The detailed audit findings formerly here (boost_jet ordering + rotation-frame decision with
its measurements; the property-based sweep incl. the BatchNorm-over-padding faithfulness
verdict; the GraphTrans-vs-GPS dropout ledger; the JetClass/plots/trials infrastructure sweep;
the jet_frames pre-publication sweep) moved to `docs/audit-ledger.md` -- they are resolved or
decided, and several back the paper's fidelity claims. Still OPEN from those sweeps:

- [ ] `torch.cuda.amp.autocast(...)` deprecation in 4 baseline files (mechanical migration).
- [ ] ParT-GPS float attn_mask + bool key_padding_mask deprecation (merge masks before torch
      makes it fatal).
- [ ] xformers pin note for the SLURM target (docs/SLURM.md install step).
- [ ] ~~Upstream lloca issue: public accessor for `_load_inner_product_factors`~~ —
      DROPPED, see §4e: the symbol is lgatr's and lloca imports it; this repo imports it
      nowhere, so there is no accessor for us to request.
- [ ] learnedpd boost-precision-floor methods sentence for the paper.
- [ ] lgatr 2.0 methods sentence (§2.4 obligation, docs/lgatr2-migration.md): the
      `tag_lgatr`/`tag_slim` reference rows are (re)trained under lgatr 2.0.0 at v2-native
      defaults (sigmoid slim gate, affine norms, sparse geometric product, tanh-GeLU, no qkv
      biases) — published-paper L-GATr numbers are indicative, not exact comparators.
- [ ] LorentzNet GPS: zero padded slots between layers for exact BN-running-stat parity
      (cosmetic; logits unaffected).
- [ ] LorentzNetKNNBlock phi_e BN normalises invalid edges (pre-existing, both variants).
- [ ] JetClass: fill the 8 `jc_<Hybrid>.yaml` `???` batchsize/lr from find_lr on jctagging.
- [ ] Rejection-metric convention differs top vs JetClass (one methods sentence, or unify).
- [ ] **Port the torchrun DDP entry path from `heidelberg-hepml/tagging-guide`.** This repo
      carries the DDP *plumbing* (`world_size` threaded through `BaseExperiment`, the
      `all_reduce` on loss, the sampler branches) but nothing ever initializes a process
      group, so `world_size` is always 1 — multi-GPU is unreachable, and the DDP regression
      was reverted rather than fixed. Directly load-bearing for §3a-bis: CGENN-GraphGPS is
      costed at ~2000–5000 GPU-h, i.e. "still weeks on four", which assumes a working
      multi-GPU path. Three concrete pieces from tagging-guide (@f159df7):
      - `run.py:43-61` — read `WORLD_SIZE`/`RANK`/`LOCAL_RANK` from the env, call
        `dist.init_process_group`, and construct the experiment as
        `EXPERIMENTS[cfg.exp_type](cfg, rank, world_size, local_rank)`. Ours takes
        `(cfg, rank=0, world_size=1)` with no `local_rank`, so device pinning per rank has
        nowhere to come from.
      - `base_experiment._step` — all-reduce the grad norm with `ReduceOp.MAX` **before**
        the skip decision. Ours decides per-rank on a local `grad_norm`; under DDP that
        desynchronizes ranks (some step, some return) and deadlocks the next collective.
        Currently latent only because `max_grad_norm: null` everywhere.
      - `config/default.yaml` — replace `gpus: -1` with their `gpu: true` + "world size is
        set by torchrun" convention, so the launcher owns topology instead of the config.
      Pairs with `docs/SLURM.md` (the `--job-name` / srun block would need the matching
      `torchrun --nproc_per_node` invocation).

## 4b. Post-campaign housekeeping

- [x] **Mechanical commit: move `find_lr.py` + `aggregate_table.py` → `utils/`** — done
      PRE-campaign after all (the postpone rationale was a hand-executed move; done
      mechanically with grep/compose verification before any recipe values existed, so
      the docs stay stable through the campaign instead of churning after it).
      Executed: `git mv` both; `config_path="../config"`; the repo-wide reference sweep
      (guides, recipe headers, the `experiments/tagging/experiment.py` guardrail string,
      both scripts' own docstrings); plus one unplanned real fix — a `sys.path` shim in
      `find_lr.py`, since from `utils/` the repo root is no longer the script dir and
      `import experiments` breaks without it. Verified: zero unqualified references left,
      hydra composes from both config trees, and a full CPU LR sweep ran end-to-end from
      the new path.

## 4b-septies. Post-merge confirmation: the training smoke — BUILT, just run it

`tests/experiments/test_training_smoke.py` exists and is the go/no-go before the
campaign. It runs 8 real optimizer steps per model on the PRODUCTION config tree and
asserts finite losses plus a non-degenerate nonzero-gradient fraction.

    CGENN_COMPILE_GATES=1 python -m pytest tests/experiments/test_training_smoke.py -q -s

Two things it encodes, both learned the hard way:
- it runs on `config/`, not `config_quick`: `tag_PlainGraphGPS` gets gradient on 230/230
  parameters under production but 1/54 under quick (`dim: 16` narrows the SAN head to
  4 units and a dead ReLU at init severs the backward — a test-config artifact, but it
  makes a quick-tree smoke prove nothing for that model);
- it asserts on GRADIENTS, not parameter movement: AdamW's decoupled weight decay moves
  a parameter whose gradient is exactly zero, which is how that case first looked fine.

EMA and checkpoint round-trip are deliberately NOT asserted here — those fail on
pre-existing `main` defects (docs/audit-ledger.md), which are out of this branch's scope.

**Run it on GPU too, and that is not a formality.** Every compile gate in this repo runs on
CPU, so inductor's CUDA backend (Triton kernels, not the C++/OpenMP lowering) has never
been exercised, and device-placement bugs are invisible to CPU runs by construction. The
test auto-selects CUDA (`base_experiment._init_backend`, and `_extract_batch` moves each
batch), so on a GPU node the same command is the GPU test. `CGENN_SMOKE_COMPILE=1` keeps
each model's shipped compile knob instead of forcing eager — use it with `-k` to take a
few models at a time, since building inductor kernels for all twelve in one process can
exhaust the box. Full sequence in `cleanup.md`'s runbook, step 5.

## 4b-bis. torch.compile scope — CLOSED (Stage 4 executed, 2026-08-08)

The revisit trigger fired and the work is done: the non-equivariant family is now
compile-gated like the equivariant one (`tests/experiments/test_nonequi_compile.py`,
log in `docs/cgenn-compile.md` Stage 4). Outcome vs the recorded worry: the
"compile-hostile structure" mostly wasn't — five of seven models reached BREAKS 0 /
RECOMP [1,1,1] (dense top-k kNN traces fine), with twins for the two real offenders
(nn.MHA's warn-break under a float ParT bias; tril_indices/nonzero PairEmbed paths).
The GraphGPS pair keeps a documented, pinned masked-BatchNorm break class and ships
`compile: false` until β-PERF shows the split graphs still win. MIParT was descoped
by operator decision (BIT-pinned, no knob).

## 4c. Versioning

`pyproject.toml` now reads **1.0.0** (operator bump, 2026-08-08) with the forward
roadmap in its comment: 1.1 once lloca 2.0 lands, 1.2 at paper (todo done, deforked,
reproduce/readme refreshed). (History: 0.9.0 was this fork's pre-release marker; the
1.1.0 before that was inherited from upstream lloca-experiments and described that
project, not this fork.)
- [ ] **Tag the pre-merge state before starting the campaign** (`git tag pre-lgatr2 &&
      git push --tags`). The campaign straddles the lgatr 1.x -> 2.0 boundary, so the
      top-tagging rows and the post-merge rows come from different code; a tag names the
      earlier state. Runs already archive their own source zip + config.yaml, so this is
      convenience, not the only provenance.

## 4d. Development records to delete at release

- [ ] **Delete `docs/audit-ledger.md`.** It records which audit findings were fixed and why
      — useful while the work is in flight (and while a re-audit might re-raise a settled
      question), useless to a reader of the published repo, whose decisions are documented
      where they apply. Its load-bearing content is already inline: e.g. the boost_jet
      entry's reasoning lives at `experiments/tagging/experiment.py:125-145`. Delete at
      release, NOT before: an auditor re-reporting a settled decision is exactly what it
      prevents, and the campaign is still running.
- [ ] Same timing for `todo.md` itself and, if you want the release lean, `docs/diffs.md`.
- [ ] **More deletions and scaffolding removal.** `cleanup.md` schedules the one-shot
      instruments (`utils/bperf.py`, `test_cgenn_compile.py`, the compile fixtures); sweep
      once more at release for anything else that exists only because the build happened —
      `utils/gp_memory_probe.py`, `bperf_results.md`, recorded fixtures whose gates were
      deleted with them, dead hydra keys, and per-model scratch configs. Rule for the sweep:
      keep what a READER of the published repo needs to rerun the paper; delete what only a
      BUILDER of it needed. Do it after the campaign, so an instrument is still there if a
      number needs re-deriving.
- [ ] **`utils/bperf.py` — KEEP, but remove `--apply`.** `cleanup.md` schedules the whole
      file for deletion as a one-shot instrument. Recommendation is to keep it: it is the
      only thing that substantiates the `compile: true` posture shipped in 20 configs, and a
      methods section claiming "compilation is worth X" should ship the tool that measured X
      — deleting it makes the claim unverifiable by re-running, which is the opposite of what
      a reproduction repo is for. Its numbers already persist in `bperf_results.md` and
      `docs/cgenn-compile.md`, but numbers are not a method.
      What SHOULD go is `--apply`: a published tool that rewrites production yamls in place
      is a footgun, the flips it existed to make are already committed, and its regex
      targeting was subtle enough to need a fix this month. Delete the flag and the
      `apply_knob` path, keep the measurement. Also drop the "one-shot instrument, schedule
      deletion" line from its own docstring when this is decided, so the file stops
      contradicting the plan.

## 4d-bis. Overnight campaign-readiness audit (2026-08-12)

Ran the tools rather than reading them. What it found, all fixed and pushed:

- **`???` is not enforced.** All 25 unswept recipes resolved to the inherited `batchsize:
  512` / `lr: 1e-3` and trained without complaint — the worst failure shape available, a
  wrong number that looks right. `run.py::_check_recipe_is_swept` now refuses to launch one;
  gated in `tests/internal/test_recipe_sweep_guard.py`. The 17 recipes calling the keys
  "REQUIRED" were corrected; all 25 now describe the marker accurately.
- **`utils/find_lr.py` invoked with `-cp config` does not run.** Hydra resolves `-cp` relative to
  the SCRIPT, so it looked for `utils/config`. Six shipped invocations were broken, including
  the `tag_cgenn.yaml` comment carrying the per-impl loop that decides the gp_impl posture.
  `find_lr.py` already defaults to `../config`; `-cp` dropped. Gated by a new check in
  `test_docs_commands.py` covering config comments and module docstrings, not just the docs.
- **My own `use_amp: false` additions were invalid and are reverted.** Those six wrappers do
  not ACCEPT the parameter (`TypeError: … got an unexpected keyword argument 'use_amp'`), so
  the key's absence was a requirement. AMP was and remains off everywhere via the
  `default=False` in `base_experiment`. Caught by `test_production_manifest` — a compose-only
  audit misses it, because compose succeeds and INSTANTIATE is what fails.

- **`find_lr`'s recipe pointer dropped for all 8 baselines.** The
  `FIND_LR … -> <recipe>` line — what a 25-model sweep is transcribed from — derived the
  recipe from the NET CLASS name, which matches for the 8 hybrids and misses for every
  baseline (`tag_cgenn` -> `top_CGENN.yaml`, absent; the real file is `top_cgenn.yaml`), so
  it silently printed nothing. Now keyed off the hydra `model` choice with the class name as
  fallback. Gated in `test_recipe_sweep_guard.py`.
- **No instantiate-level gate covered most production configs.** `test_production_manifest`
  covers the six lgatr-family models only, which is why `tag_cgenn` and `tag_lorentznet`
  carried my invalid `use_amp` key with nothing to report it. Added
  `tests/internal/test_config_target_signatures.py`: a static check that every config key is
  a parameter of the class its `_target_` names, over all 36 + quick configs in <4 s, no
  model construction. Currently clean; mutation-tested against the real bug.

- **The probe on JetClass / TopTagXL: it cannot be constructed there, and now says so.**
  Both stream through `SimpleIterDataset` — no `data_list`, no indexable items — so there is
  nothing to select from. Their batches still vary (weaver's dense `(B, 4, P)` goes straight
  through `dense_to_sparse_jet` into the same `ptr` layout top-tagging uses), so the blind
  spot is real there; only the fix is unavailable. Two changes: the fallback is hardened
  (a `data_list` property raising anything but AttributeError used to propagate and kill the
  search), and `+lr_find.bs_draws=N` probes the heaviest of N drawn batches as the one
  available substitute — worth about the p87 batch at N=8, against construction's
  worst-of-run, and off by default. **For the 8 `jc_*` recipes, use `bs_draws` AND keep
  `bs_safety<1`; that family is the one place the old headroom argument still holds.**

Confirmed clean:

- All **72** campaign (task, model, recipe) combinations compose AND resolve.
- The four LLoCa-canonicalised models **train end to end under `learnedpd`** — the frame
  family settled on today, and previously uncovered since every shipped config is `identity`
  (`CGENN_COMPILE_GATES=1 CGENN_SMOKE_OVERRIDES="model/framesnet=learnedpd" pytest
  tests/experiments/test_training_smoke.py -k "Plain or ParticleNetParT"`).
- β-PERF's `check_window` accepts both recommended windows (`--iters 1010 --window 100 1000`
  and the 110/10-100 screen). NOTE `--models CGENN` matches the two hybrids ONLY; for the
  three `gp_impl` rows as well use `--models tag_cgenn CGENN`.
- `_resolve_epoch_budget` runs before `_init_scheduler` and derives iterations from the real
  loader length, so a batch size from `find_lr` propagates correctly into the anneal.
- Deprecation ranking corrected: `torch.cuda.amp.autocast` is a **FutureWarning** (940 of them
  in a 29 s run), the ParT-GPS mask only a **UserWarning** — the reverse of an earlier note.

## 4d-ter. What the two-step probe cost, and the gate that now covers it (2026-08-13)

The audit above shipped a SECOND training step per rung in `find_max_batch_size` (timing, plus
a stricter memory probe). It hoisted **one** batch object out of the step and used it twice.
That is exactly the thing `embed_tagging_data` forbids: it adds the spurion offsets to the
caller's `ptr` **in place**, so the second embed double-counts them and dies pre-model at
`embedding.py:120` (`batch[~is_spurion]`) with

    IndexError: The shape of the mask [1235] ... does not match ... the indexed tensor [1283]

1283 − 1235 = 48 = 3 spurions × 16 jets. Deterministic and pre-model, so it took out **all five
rows** of a real β-PERF run, each with the identical message, and each then fell back to the
config `batchsize: 512` and OOM'd. Reproduced on CPU against the real dataset/collate/embedding
and fixed: `make_batch()` constructs a fresh batch per step (construct, not clone — the drawn
path and the weaver tuple layout have no `.clone()`, and `_worst_case_indices` is deterministic
so both steps still get an identical batch).

Why the ~55 gates missed it: `_FakeExp._batch_loss` had **no embedding step**, so reuse was
free in the fake and fatal in reality. Both halves of that gap are now closed —
`_FakeExp` refuses a batch object it has already consumed (17 of the 56 gates fail on the
pre-fix code), and `test_a_probe_batch_can_only_be_embedded_once` pins the premise against the
real `embed_tagging_data` so the fake cannot drift from what it stands in for.

Audited every other repeat-call site for the same class — clean: `gp_memory_probe` goes through
`_rebuild` (clones every field), the FLOPs counter clones, `init_standardization` embeds once,
and the train/val/lr-sweep loops all draw fresh batches.

**The bigger bug was β-PERF's, and it is fixed too.** `--find-batchsize` caught every exception,
printed a warning and ran the matrix at the config value anyway. That is what converted a
30-second failure into hours of H100 time — and it had already done it once, to a
`ModuleNotFoundError` (the `sys.path` note at the top of `bperf.py` records it). A sizing search
that fails is now **fatal**: `--find-batchsize` is a request to size the rows, and wanting the
yaml value is spelled by omitting the flag. New `tests/internal/test_bperf_driver.py` (11 gates,
none needing a GPU) covers the fail-fast, the window check, the `--models` filter documented in
this file, and the `--apply` regex against all 17 rows' production yamls.

- [ ] **Post-campaign: make the class impossible.** `ptr = ptr.clone()` at the top of
      `embed_tagging_data` removes the side effect for every caller, present and future, and is
      bit-identical arithmetic (only the caller's tensor is spared). NOT done now: it is a
      per-step path and the campaign has started. The comment at `embedding.py:102-104` is the
      interim warning.

      **The side effect is incidental, not designed** — measured, and this is the argument for
      the fix as well as the evidence that it changes nothing anyone relies on. Which tensor
      gets mutated moves with unrelated data keys:

      | `config.data`                         | n_spurions | `ptr` mutated | `batch.x` mutated |
      |---------------------------------------|-----------:|---------------|-------------------|
      | SHIPPED (`max_particles: null`)        | 3          | **yes**       | no                |
      | `max_particles: 128`                   | 3          | no            | no                |
      | `beam_reference: null` (spurion ablation) | 0       | no            | **yes**           |
      | `beam_reference: null`, `mass_reg: null` | 0        | no            | no                |

      With `max_particles` set, line 60 rebinds `ptr` and the caller is spared; with spurions
      off, `mass_reg` writes into `fourmomenta` instead — which IS `batch.x`, because
      `_extract_batch`'s `.to(float64)` is a no-op on an already-float64 tensor. No designed
      contract moves its side effect to a different argument when you toggle a beam setting.
      Bounded: PyG's collate copies (`torch.cat` allocates), so nothing reaches `data_list` and
      no dataset is corrupted across epochs — verified.

- [ ] **Report upstream (`heidelberg-hepml/lloca-experiments`).** The mutating line is theirs
      verbatim — fetched `experiments/tagging/embedding.py` from their `main`, the statement is
      their line 103 and the `max_particles` rebind their line 60; only our warning comment at
      102-104 is local. Worth a short issue: the two-line fix is `ptr = ptr.clone()`, and the
      `beam_reference: null` row above is the one that bites them silently (no exception, the
      caller's four-momenta are rewritten by the mass regulator). Pairs with the LLoCa issue
      already drafted.

## 4d-quater. CGENN performance program (post-campaign; plan of record 2026-08-14)

The full plan — per-item posture, correctness class, and judging gate — lives in
**docs/cgenn-compile.md, "The improvement program"** (grounded in the H100 profiles and
the kernel census; the sync fixes were measured OUT there, ~1% — do not resurrect them).
Sequencing is fixed: CGENN core → CGENN-GPS → LNetSlim-GPS *(only if its profile shows
the marshalling/micro-GEMM signature)*. Checklist:

- [x] **Pre-req**: record the hybrid BIT pins in the suite's own env
      (`CGENN_COMPILE=record pytest tests/experiments/test_hybrid_bit_pin.py -q`) —
      BIT-class rewrites have no machine check until this runs. *(Recorded 4030dd9.)*
- [x] **Phase 1.1 + 1.2** — DONE 2026-08-14 as two TOL-class rewrites (operator relaxed
      BIT→TOL that day): `b()`/`q()` diagonal collapse + `MVLinear` block-diagonal flat
      GEMM. Marshalling nodes at the sites 547→92, bmm 116→54; fp64 model-level diffs
      inside the repo bars; all gates green; fixtures + Trans pin re-recorded with the
      class change stated (GPS/LNet pins byte-identical). Full record in
      docs/cgenn-compile.md "The improvement program". GPU walltime verdict + gp_impl
      re-race + shield-retirement soak → gate day.
- [ ] **Phase 1.3** `activation_memory_budget` adoption (compiled posture; vram matrix +
      β-PERF judge it; CPU priors recorded).
- [ ] **Phase 2** CGENN-GPS — updated 2026-08-14 after the post-Phase-1 audit:
      2.1 block-glue layout RE-SCOPED (census: glue "marshalling" is mostly views +
      Linear weight transposes; mv_bridge already goes through the rewritten MVLinear
      — gate-day check only, don't rewrite blind). 2.2a receiver-degree hoist DONE,
      BIT-class, fixtures/pins passed UNCHANGED. 2.2b sorted-segment main scatter:
      prototype VERIFIED READY (receivers provably sorted from both builders on
      real batches; segment_reduce bit-equal to index_add fwd+bwd on CPU incl.
      forced-empty segments; compiles 1 graph/0 breaks on 2.13) — flip only if the
      gate-day profile still ranks the scatter after 2a AND the lab passes on the
      NGC 2.8 build. Shield contingency CORRECTED by lab: respelling sparse_gp's
      einsum does NOT clear its saved permuted views (partitioner's choice, not
      the surface form) — if the shield-off soak fails, keep the scoped shield /
      escalate via partitioner knob, not a rewrite. Then attention re-check at
      P_max; bucketing last.
- [ ] **Phase 3** LNetSlim-GPS: one `profile_sync` run = both its GPU soak and the
      is-there-anything answer.
- [ ] **TF32: LAST resort, table-wide-or-not-at-all, operator decision** (accuracy-
      reducing knob; per-model use would be an unfair row — same rule as
      expandable_segments). Protocol in the doc.
- [ ] Per merged change: the GPU GATE DAY (soak smoke via `CGENN_SMOKE_STEPS`, vram row,
      `profile_sync`) before the compiled posture ships. STATUS 2026-08-15 after two
      rounds (A6000 + H100): 2.2b ADOPTED on the H100 profile's 25-27% scatter share
      (segment_reduce swap, CPU-bit-equal, CUDA-deterministic; sortedness
      machine-checked in test_edge_builders.py); 2.1 CLOSED (no glue copy kernels on
      H100); Phase 3 kernel-side EMPTY (LNetSlim ~5% GPU-bound — host tax is 2.4's);
      2.4 bucketing = biggest remaining lever (GPS family 72-95% host). OPEN: shield
      retirement (round soaks were single-batch = vacuous, my flaw, gate fixed —
      rerun the shield-off soak after pulling); β-PERF GPS row (eager-sized batch
      OOMs compiled → set activation_memory_budget=0.5 or bs<=128 and rerun);
      OPERATOR: flip tag_cgenn gp_impl sparse→matmul (matmul 304 > einsum 296 >
      sparse 284 jets/s own-batch; find_lr rungs agree) + adopt bs=128/lr=5.57e-04.
- [ ] File the inductor stride-guard issue upstream (draft in the 2026-08-14 session log;
      signature + monkeypatch evidence in docs/cgenn-compile.md). Shield retires at the
      next NGC container upgrade — re-test with the shield off, then delete it.

## 4e. Release gate — the pre-publication list, triaged

The campaign has STARTED, which sets the rule for everything below: **anything that changes
model arithmetic is off the table until it finishes.** A mid-campaign model change makes rows
incomparable and there is no cheap way to detect that later from the result table alone. So
the split is not by importance, it is by whether the change can touch a number.

**Must do — blocks publication, changes no arithmetic:**

- [ ] **lgatr 2.0 methods sentence** (§2.4 obligation, `docs/lgatr2-migration.md`). The
      `tag_lgatr`/`tag_slim` reference rows are retrained under lgatr 2.0.0 at v2-native
      defaults (sigmoid slim gate, affine norms, sparse GP, tanh-GeLU, no qkv biases), so the
      published-paper L-GATr numbers are indicative, not exact comparators. Without this
      sentence the paper implicitly claims a comparability it does not have. Highest-value
      item on this list.
- [ ] **learnedpd boost-precision-floor methods sentence.** What it means: LLoCa's default
      frame (`learnedpd`) is built by POLAR DECOMPOSITION, whose rest-frame boost divides by
      energy, so a boost amplifies rounding and the per-edge tensorial transport accumulates
      an error floor — worse with more edges, **~5e-3 fully connected in float64**.
      `learnedso13` builds the frame by direct 4d orthonormalization and transports to
      **~1e-6**. Both are exact in real arithmetic; this is a numerical floor, not a broken
      symmetry (true breaks are O(0.1–1), far above it), which is why
      `test_tag_equivariance.FRAME_TOL` carries two bounds: `learnedso13 (1e-4, 1e-4, 1e-3)`
      vs `learnedpd (1e-2, 1e-2, 2e-2)`. The obligation is that a reader who sees "LLoCa
      frames make the backbone Lorentz invariant" must not infer machine precision under
      `learnedpd`. Draft: *"Under learned local frames the non-equivariant backbones are
      Lorentz invariant up to a numerical floor: ~1e-6 with the SO(1,3) tetrad frame, and
      ~5e-3 with the default polar-decomposition frame, whose rest-frame boost divides by
      energy and so amplifies float64 rounding across the per-edge transport. Both are exact
      in exact arithmetic."*
- [ ] **Rejection-metric convention differs top vs JetClass — DOCUMENT, do not unify.** The
      WORKING POINTS differing is the community convention for these two datasets, so
      unifying them would deviate from both literatures. Formulas as implemented:

      top tagging (`experiments/tagging/experiment.py:361`), binary QCD vs top:
          rej(eS) = 1 / fpr[ argmin_i |tpr_i - eS| ]        eS in {0.3, 0.5, 0.8}
      i.e. the NEAREST ROC grid point, no interpolation; single binary AUC.

      JetClass (`experiments/tagging/jetclassexperiment.py:198`), 10-class, ovo-macro AUC;
      per class i binarised against QCD with a renormalised two-class score:
          s_i    = p_i / (p_0 + p_i)
          rej_i(eS) = 1 / interp1d(tpr, fpr)(eS)
          eS_i   = 0.5 for all i except class 5 -> 0.99 and class 7 -> 0.995
      i.e. LINEAR INTERPOLATION of the ROC.

      Three differences, only one of which is not convention: the working points are (keep),
      the renormalised two-class score is the standard multiclass-to-binary discriminant
      (keep), but **nearest-grid-point vs interpolation is an estimator difference in a
      quantity reported under the same name**. It matters most exactly where JetClass uses
      it — eS = 0.99 / 0.995, the sparse tail of the ROC — which is the arm that already
      interpolates, so the current split is the safe way round. Not locked by the campaign:
      both are computed at EVAL from the ROC, and `evaluation.save_roc` writes
      `roc.txt` (fpr, tpr), so either estimator can be recomputed post hoc without
      retraining. One methods sentence naming both is enough.
- [ ] **Fill the 8 `jc_<Hybrid>.yaml` `???` batchsize/lr** from `find_lr` on jctagging.
      Blocking for any JetClass row: hydra will not compose a `???`. NOTE these are the ONE
      family where the constructed probe batch does not apply — JetClass streams from files
      and exposes no per-item lengths, so `find_max_batch_size` falls back to a single random
      batch and logs that it did. `bs_safety` is still the only headroom lever there; use
      `+lr_find.bs_safety=0.9` or verify with a short real run.
- [ ] **§3a-bis cost claim vs the DDP reality.** The ~2000–5000 GPU-h estimate is phrased as
      "still weeks on four", which asserts a working multi-GPU path. Nothing initializes a
      process group, so `world_size` is always 1 and multi-GPU is unreachable. Fix the
      SENTENCE, not the code — state the cost as single-GPU hours and note multi-GPU is not
      wired. Porting DDP is post-release (below).

**Should do — safe now, no arithmetic touched:**

- [ ] **`torch.cuda.amp.autocast(...)` migration**, 4 live sites — and it is the MORE urgent
      of the two deprecations, which inverts an earlier note here. Torch classes them
      differently, checked: autocast raises **FutureWarning** ("this will change"), the
      ParT-GPS mask raises only **UserWarning**. The earlier text had it backwards, saying
      the mask "will become fatal" (no removal version is announced) while calling autocast
      cosmetic. Autocast is also the noisy one: it fires per forward call — 940 warnings in a
      29-second smoke run — which on a multi-day campaign is a lot of log. Sites:
      (`particlenetpartgraphgps.py:241`, `particlenettransformer.py:916`,
      `plaingraphgps.py:403`, `plaingraphtrans.py:350`; mipart's two are commented out).
      `torch.amp.autocast("cuda", enabled=X)` is exactly equivalent, and every model ships
      `use_amp: false` so the context is inert either way. Safe mid-campaign precisely
      because it cannot move a number.
- [ ] **xformers pin note in `docs/SLURM.md`.** The NGC container's xformers is built against
      a different torch and logs a scary load failure on every run; nothing here depends on
      it. A reproducer will otherwise chase it. Documentation only.
- [x] **Log the environment toggles per run.** `docs/OSCAR.md` §2 appends two exports to
      `venv/bin/activate` (`TRITON_LIBCUDA_PATH`, `TORCHINDUCTOR_CACHE_DIR`) — correct place
      for those, but it makes them invisible per-run state: nothing in a row's config or log
      said which environment it ran under. `BaseExperiment` now logs
      `PYTORCH_CUDA_ALLOC_CONF`, `TORCHINDUCTOR_CACHE_DIR`, `TRITON_LIBCUDA_PATH`,
      `CUDA_VISIBLE_DEVICES`, `OMP_NUM_THREADS` (naming the unset ones too), so a row's
      provenance is readable from its own log. Safe mid-campaign: logging only.
- [x] **`use_amp: false` made explicit on the six equivariant models** (the 4 CGENN-/
      LorentzNet-LGATr hybrids, `tag_cgenn`, `tag_lorentznet`). They previously relied on the
      implicit default (`base_experiment.py:677` selects with `default=False`, and every
      wrapper signature is `use_amp=False`), so AMP was already off everywhere — but the
      models where AMP is a CORRECTNESS matter rather than a speed one were the only ones not
      saying so, while ParT/ParticleNet/Plain stated it. Same value, zero behaviour change,
      and the comment records the reason (AMP destroys equivariance; lgatr measured this).

- [ ] **`PYTORCH_CUDA_ALLOC_CONF` — decide BEFORE the next campaign, not during this one.**
      Order-of-magnitude estimate, since no GPU was available to measure it: a cached-pool
      allocation is ~µs and a step is ~700 ms (β-PERF's 1.41 it/s), so allocator work is
      well under 1% of step time and expandable segments changes only part of that. The
      benefit is not per-step either — it is avoiding `release_cached_blocks`
      (cudaFree + re-cudaMalloc + a device sync, ~10-100 ms and lumpy) and the OOM at the
      end of that road. So |effect| is plausibly **under ~2% either way**, i.e. BELOW the 3%
      margin β-PERF itself uses to justify a flip. That reframes it: this is not a throughput
      knob, it is OOM insurance, and the question is whether you need the insurance rather
      than what it costs.
      The measurement worth doing (cheap, and ParticleNet is the right subject because it is
      fast): one paired A/B, same model, same seed, same batch, variable on vs off, compare
      it/s. Do it to PUT A NUMBER on the knob for the next campaign and for a methods
      sentence.
      Two corrections to earlier reasoning here, both checked:
      (a) *"walltime isn't shown for top tagging"* — half right, and the half that matters
      is yours. `time` IS emitted: column 10 of the `toptagging` legend in
      `utils/aggregate_table.py`, fed by `train_time`, real wall clock from
      `base_experiment.py:777`. But the aggregator's column set is a superset of the paper's,
      and this column is not one that a top-tagging table conventionally carries. If it does
      not go in the paper, the knob touches no published number, and on a shared cluster its
      ~2% is inside that column's own node/queue/filesystem spread anyway. What survives is
      only the methods statement — whether the rows can be described as run identically.
      (b) *"retraining a fast subset makes it worse"* — only if you retrain SOME. What matters
      is uniformity of the FINAL table. If the rows already finished are the fast ones,
      re-running exactly those with the variable on gives a uniform table cheaply, and is the
      right move. If any slow row is already done, leave it off instead. Either way the
      per-run `env:` line now makes which rows had it auditable rather than remembered.

**Post-campaign — real, but each one changes a number:**

- [ ] **ParT-GPS float `attn_mask` + bool `key_padding_mask`.** CORRECTION to an earlier
      note here that said torch "will make it fatal": no removal version has been announced.
      Reproduced on this repo's torch (2.13.0+cu130) — it is a plain `UserWarning`:
      *"Support for mismatched key_padding_mask and attn_mask is deprecated. Use same type
      for both instead."* It has been a warning since ~1.11 and still is. What removal WOULD
      mean mechanically: `F._canonical_mask` currently converts the bool mask to a float one
      (`0.0` / `-inf`) and adds it to the float `attn_mask`; dropping "support for
      mismatched" means raising instead of converting. The fix is to do that merge at the
      call site — broadcast the bool `key_padding_mask` to the attention shape, cast to the
      float mask's dtype with `-inf` on masked slots, add, pass `attn_mask` only. Mechanical,
      but it is mask arithmetic, which is where a silent numerical change hides, so it needs
      a bit-identity gate against recorded fixtures. No deadline pressure — do it after the
      campaign and re-record.
- [ ] **LorentzNetKNNBlock `phi_e` BN normalises invalid edges** (`lorentznet.py:27`) and
      **LorentzNet-GPS padded slots between layers.** Both are real: BatchNorm running stats
      feed EVAL, so polluted statistics move eval logits, "cosmetic" understates it. But both
      are PRE-EXISTING and present in BOTH variants, so they shift absolute numbers equally
      and leave the baseline-vs-hybrid comparison — the paper's actual claim — intact.
      Fixing either mid-campaign would break that symmetry, which is worse than the flaw.
      Document in the methods; fix after.
- [ ] **Port the torchrun DDP entry path** from `heidelberg-hepml/tagging-guide` @f159df7:
      `run.py:43-61` (read `WORLD_SIZE`/`RANK`/`LOCAL_RANK`, `dist.init_process_group`, pass
      `local_rank` so per-rank device pinning has a source); `base_experiment._step` (all-
      reduce the grad norm with `ReduceOp.MAX` before the skip decision — ours decides per
      rank, which under DDP desynchronizes and deadlocks the next collective, latent today
      only because `max_grad_norm: null` everywhere); `config/default.yaml` (`gpus: -1` ->
      their `gpu: true` + "world size is set by torchrun", so the launcher owns topology).
      Pairs with the `docs/SLURM.md` srun block needing a matching `torchrun --nproc_per_node`.
      Explicitly post-release. Until then the grad-norm deadlock is a trap for anyone who
      tries — it is recorded in `docs/audit-ledger.md` for that reason.

**Optional:**

- [ ] ~~**Upstream lloca issue: public accessor for `_load_inner_product_factors`.**~~
      **CORRECTED — this repo has no direct exposure and there is probably nothing to send.**
      The entry said "this repo imports the private name". It does not: `grep` finds the
      symbol in no file here. The real shape is an UPSTREAM-TO-UPSTREAM coupling —
      `_load_inner_product_factors` is defined in **lgatr**
      (`lgatr/primitives/invariants.py:20`) and imported by **lloca**
      (`lloca/equivectors/lgatr.py:9`). We are a bystander.
      So our exposure is transitive only: if lgatr renames or drops that private symbol,
      lloca breaks and we break through lloca. The mitigation is the version-pin re-check
      already recorded at `docs/lgatr2-migration.md` H6, and the line-73 note there is
      exactly that check (confirming the private name survives at lgatr 2.0.0 so lloca 1.3.6
      keeps working) — not a usage.
      Still true on lloca's `dev` branch, checked 2026-08-12: `lloca/equivectors/lgatr.py`
      line 15 imports the same private name, so the coupling is not something they have
      already fixed and it is fair to mention.
      If it is ever raised it goes to **lgatr** (make it public) or to lloca (stop depending
      on a private upstream name), and both are the same group, so it is one conversation
      and one sentence inside a mail sent for other reasons. Not worth a mail of its own.

- [ ] **THERE IS a mail worth sending, and it is not about the accessor.** Three bugs this
      repo found by adversarial review and fixed locally are still present in lloca 1.3.6,
      all verified against the installed source. Line numbers are lloca's, not ours.

      **(1) Per-particle frames are silently scrambled during ParT training. Serious.**
      `LLoCaParticleTransformer.forward` calls `self.attention.prepare_frames(frames)` at
      `backbone/particletransformer.py:1088`, BEFORE `_forward_encoder`, which at `:1020`
      runs `self.trimmer(x, v, mask, uu)`. `SequenceTrimmer` (`:231-242`) draws
      `rand = torch.rand_like(mask)`, takes `perm = rand.argsort(...)`, and gathers x, v,
      mask and uu by it — a RANDOM PERMUTATION of the token order. Frames are not in that
      gather, so every token is then transported with another token's frame.
      Reachable at defaults: `trim=True` (`:863`), trimmer enabled when
      `trim and not for_inference` (`:875`), and the permutation starts only after
      `warmup_steps` — so the first few steps are correct and the run silently goes wrong
      from `warmup_steps+1`, which is exactly why a smoke test does not catch it. Global
      frames are immune (one frame for all tokens); per-particle learned frames are not.
      Our fix: pass the frames THROUGH the trimmer with x/v/mask
      (`experiments/baselines/particletransformer.py:1210-1227`).

      **STATUS ON `dev`, checked 2026-08-12 — send only (1) and (3).**
      (2) is ALREADY FIXED on dev: the loop reads `for k in list(state_dict.keys()):`.
      Report it against 1.3.6 only if you care about the released version; otherwise drop it.
      (1) is UNFIXED on dev: `prepare_frames` is still called before `_forward_encoder`, and
      the frames still do not ride the trimmer.
      (3) is UNFIXED on dev: `prepare_frames` still exists (now taking `p_ref`/`ptr` for a new
      `preserve_variance` path) and still does
      `reshape(*frames.shape[:-3], 1, frames.shape[-3], 4, 4)` then `expand` — the same
      head-dim-at--3 logic, so a flat `(N,4,4)` input still expands head-major.

      **(2) `_load_from_state_dict` mutates the dict it is iterating.** *(fixed on dev)*
      `backbone/particletransformer.py:551`: `for k in state_dict.keys():` while the body
      does `state_dict[...] = state_dict.pop(k)` for the `in_proj_weight`/`in_proj_bias`
      rename. That is a `RuntimeError: dictionary changed size during iteration`, or
      silently skipped keys depending on the path taken. One-word fix: iterate
      `list(state_dict.keys())`.

      **(3) `prepare_frames` silently mis-expands flat frames.**
      `backbone/attention.py:46-52` inserts the head dimension at `-3`. Given batch-shaped
      `(B, P, 4, 4)` that gives `(B, H, P, 4, 4)` — correct. Given FLAT `(N, 4, 4)`, which is
      what the framesnets emit when driven with `ptr`, it gives `(H, N, 4, 4)`, i.e. head-
      major, while q/k/v are `(B, H, P, ...)` — a systematic token/frame misalignment for
      B>1, with no error. Since lloca's own framesnet produces the flat layout on the sparse
      path, this is an internal contract mismatch rather than just a caller mistake; an
      assert on `frames.dim()` would have caught all of it.

      Worth including as context, not as bugs: the `learnedpd` invariance floor we measured
      (~5e-3 vs `learnedso13`'s ~1e-6 in float64 — see §3), that `gamma_hardness` is silently
      inert whenever `gamma_max is None` (`framesnet/equi_frames.py:550-551` returns early),
      and one sentence on the private lgatr import above.

## 5. Paper release — branding / identity (only the maintainer has these)

Critical (still point at the upstream LLoCa project):
- [ ] `README.md` — title ("Lorentz Local Canonicalization"), arXiv badges (2505.20280 / 2508.14898),
      author list + `heidelberg-hepml/*` links, the BibTeX block.
- [ ] `reproduce.md` — clone URL `heidelberg-hepml/lloca-experiments` + `cd lloca-experiments`,
      upstream arXiv references; **replace the manual JetClass-download line with
      `python data/collect_data.py jetclass`** (now automated).
- [ ] `REPRODUCE.md` — stale xformers claim: says running LLoCa/L-GATr taggers without xformers
      "requires modifying the data embedding and attention mask construction". No longer true —
      `model.attention_backend=flash|flex` does it as a config override (GUIDE §7, docs/OSCAR.md §2
      note). Rewrite the paragraph; upstream PR comment about it planned separately.
- [ ] `LICENSE` — copyright currently lists the upstream LLoCa authors; add your authors / mark derivative.
- [ ] **Humanize the prose** in the assistant-drafted texts/files before publication — `GUIDE.md`,
      `docs/{OSCAR,SLURM,ablations,diffs}.md`, this todo, the longer code comments: pass for
      personal voice, trim the em-dash-heavy style, keep the technical content.

Minor (stale strings / metadata):
- [ ] `pyproject.toml` — add an `authors` field (name is already `gtagger-experiments`).
- [ ] add a `CITATION.cff` for the new paper.
- [ ] `experiments/base_experiment.py:299` — `path_code = os.path.join(self.cfg.base_dir, "lloca")`
      hardcodes "lloca" for the saved-source dir → project name.
- [x] `docs/OSCAR.md` §2 — `.sif` discrepancy: CCV has corrected the
      `ngc-pytorch-container/25.08-py3-ayk4` module (it no longer `setenv`s the 24.03 sif).
      The prose now says so; the resolve-and-hard-stop guard is **kept on purpose** — it is
      free, self-checking (overrides only when the value is wrong), and the failure it
      catches is silent (a 24.03 image trains fine-looking garbage on torch 2.3). Remove it
      only if the modulefile is ever guaranteed stable across all login nodes.
- [ ] `docs/SLURM.md:79` — `#SBATCH --job-name=lloca`.
- [ ] `config/{toptagging,jctagging,ttbar}.yaml` + `config_quick/*` — debug `exp_name`s
      (`topt_local_debug`, `jc_debug`, `ttbar_debug`).
- [ ] `config/model/tag_CGENNLGATrGraphTrans.yaml` — incomplete `#should be` comment (cosmetic).
- [ ] `tests/helpers/equivariance.py:4` — upstream attribution comment; fine to keep as a credit.
- [x] **stale `tagging_features_framesnet` overrides** — the data key was renamed to
      `tagging_features`, but the old name lingered in run-command examples that would error under
      Hydra (struct mode rejects unknown-key overrides): `REPRODUCE.md` lines 227–229 & 234–235
      (×5) and `.github/workflows/experiments_tagging.yaml:39` (×1). Renamed each to
      `data.tagging_features=…` (the workflow one would otherwise have failed the tagging CI job).
- [ ] **Defork** the GitHub repo when publishing (a fork is hidden from search / awkward to Zenodo-archive);
      keep the upstream attribution in README + LICENSE.
- [ ] **LICENSE authors** — the MIT notice still names only the upstream authors
      (Spinner/Favaro/Lippmann/Pitz/Gerhartz, 2025); on publication add this repo's author+year
      copyright line while KEEPING theirs (MIT requires retaining their notice; adding a second
      line for the derived work is the standard pattern). Pairs with the defork step above.

## 6. Done (for reference)

- 2×2×2 hybrid family ({Plain, ParticleNet-ParT, CGENN-LGATr, LorentzNet-LGATr-slim} × {GraphTrans, GraphGPS}).
- Faithful LLoCa tensorial message-passing for the ParticleNet-ParT **and Plain** hybrids (MPNN/EdgeConv
  `change_local_frame` + `LLoCaAttention`), **additive** (identity frames bit-identical: 0 added params),
  jet-frame class token (GraphTrans) / invariant mean-pool (GraphGPS), rapidity clamp.
- Equivariance suite (24/24, incl. full Lorentz boost under learned `so(1,3)` frames).
- `utils/find_lr.py` batch-size finder; `utils/aggregate_table.py`; `data/collect_data.py jetclass`; `GUIDE.md`; `docs/SLURM.md`.
