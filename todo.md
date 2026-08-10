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
- [ ] `batchsize` ← `utils/find_lr.py +lr_find.find_batch_size=true` (largest power-of-two that fits YOUR GPU -- the finder measures it, so the answer is per-machine, not a number to copy).
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

**CGENN-GraphGPS alone is ~84% of the eight-model total.** Not a misconfiguration — both CGENN
configs are identical (k=16, num_blocks=10, same widths); GPS runs the Clifford-algebra MPNN
inside *every* block instead of once, which is what GraphGPS is.

Calibrating against that table's own h/GFLOP (61–210, median ~83; L-GATr improves 81 → 28 under
lgatr 2.0), CGENN-GraphGPS lands at **~2000–5000 GPU-h** for a full-JetClass-equivalent run —
weeks to months on one H100, and still weeks on four. **This needs a decision before the JetClass
campaign, not during it** — but the fix is implementation, not architecture. Measured levers, all
of which leave the model identical (details in `docs/cgenn-compile.md`, dev branch): replace the
`einsum` geometric product with lgatr 2.0's outer-product + matmul form (**5.2× on the GP, verified
bit-identical**, and the GP is ~46% of runtime); the data-movement rewrites (`copy_` is **38%** of
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
- [ ] Upstream lloca issue: request a public accessor for `_load_inner_product_factors`
      (this repo imports the private name; fine at lloca 1.3.6 + lgatr 2.0.0, re-check on
      any bump of either — docs/lgatr2-migration.md H6, Phase 5).
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
