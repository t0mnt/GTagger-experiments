# How this fork differs from upstream (lloca-experiments)

A quick aside, one line per difference. Deliberate faithful *quirks of official model
implementations* (CGENN's padded-batch-max readout, LorentzNet's dead LGEB dropout kwarg,
ParT's cls-block-only dropout zeros) are kept, commented in code, and not listed here.

## Added
- Models: the 8 GraphTrans/GraphGPS hybrids ({Plain, ParticleNet-ParT, CGENN–L-GATr,
  LorentzNet–L-GATr-slim} × {GraphTrans, GraphGPS}) with configs, quick-configs and tests.
- Tools: `utils/find_lr.py` (LR range test + GPU batch-size finder), `utils/aggregate_table.py`,
  `data/collect_data.py {jetclass, toptagxl}` (download + md5-verify + extract).
- Recipes: shared family defaults (`tag_/jc_gts_and_friends_default`) + per-model
  `top_/jc_<hybrid>.yaml`; both task configs default to the family recipe (was `jc_ParT`-style).
- Trials workflow: fresh-trial warm starts (`warm_start_load=false`), per-run
  `table_metrics_*.json` accumulation, automatic `[N trials] mean ± std` table rows.
- Epoch budget: `training.epochs` → iterations derived at runtime (`_resolve_epoch_budget`);
  `CosineAnnealingWarmup` (linear warmup → cosine) scheduler option.
- Table row extended with model name, trials tag, train time, per-jet FLOPs, kNN metric;
  `utils/aggregate_table.py` emits one table per task (top/XL/JetClass columns differ) and
  JetClass emits an aggregator-compatible row.
- Guardrail warnings: hybrid training at the unswept 512/1e-3 fallback; `seed` set together
  with a fresh trial; end-of-training loss-vs-accuracy checkpoint-selection cross-check.
- Docs: `GUIDE.md`, `docs/{SLURM,OSCAR,ablations,diffs}.md`, `todo.md` ledger.

## Changed
- `es_patience` 100 → `null`: no early termination (train the full budget; the best-validation
  checkpoint is still saved/restored/reported) — a significant plateau-allowance shift.
- Best-checkpoint restore re-pairs the EMA shadow with the restored weights (also submitted
  upstream); `best_model_metric` toggle (loss/accuracy) added for the selection metric.
- ParT's weight-decay grouping extended from the hardcoded `{"cls_token"}` to
  `net.no_weight_decay()`, covering the CLS-token hybrids.
- Packaging: project renamed `gtagger-experiments` in `pyproject.toml`.
- Deps: `lloca[xformers-attention]>=1.3.6` (`jet_frames` uses `fn.mass_regularize`, first in 1.3.6).
- CI runs the tagging equivariance+invariance suites (upstream removed its broken test line
  without a replacement, leaving tagging uncovered). The expected residual symmetry group
  (SO(2)-about-beam vs full Lorentz) is now derived from each model's spurion keys
  (`residual_symmetry_group`), keyed off the unified `beam_spurion`/`add_time_spurion` names,
  so the asserted group tracks the config instead of a hand-maintained per-model list.
- `save` defaults true in `config/` (best-val weights kept as `model_run{idx}.pt`);
  `validate_every_n_epochs_min` sentinel for once-per-epoch validation.
- DDP additionally wraps the framesnet (upstream plans a different DDP rework; multi-GPU
  remains not-recommended by upstream).
- Debug-friendly extras: `warm_start_load` non-sticky in saved configs, selection-history
  logging, extended run-summary logging.

## Fixed here first, since adopted upstream (#92 etc.)
- CGENN wrapper dense-frame edge construction; MIParT rapidity clamp; score.pdf storing
  sigmoid probabilities; `tag_lorentznet` `n_scalar` wiring; dataset mask `.all→.any`;
  `torch-geometric>=2.6` pin.
  (The `jc_lgatr` `tag_gatr→top_lgatr` recipe-base rename was fixed on `main` directly,
  NOT here — it arrives via the merge, so it is not a fork-first fix.)

## Fixed here, offered upstream (post-audit round)
- `for_inference` single-logit heads use sigmoid (softmax over a 1-wide dim is constant 1.0);
  guard/softmax on dim 1 (segmentation-safe). All four sites (`mipart.py`, `particlenet.py`,
  `particletransformer.py`, `particlenetpartgraphgps.py`). Latent: nothing sets
  `for_inference=true` and the class default is False.
- `boost_jet` forced off for pure-rotation frames (LearnedSO3/SO2; measured set).
- `pairwise_lv_fts` clamps delta_r2 **at `eps**2`** before the sqrt. `sqrt'(0)=inf`, and
  delta_r2 is exactly 0 for any bit-identical pair -- every padded-constituent pair on every
  forward, plus the diagonal wherever `remove_self_pair` leaves it -- so the backward is
  `0 * inf = NaN`; the clamp already inside the `log` cannot undo it, it is what produces the
  0. Live in the campaign path: `particlenettransformer.py` serves ParticleNetParT-GraphTrans
  and -GraphGPS, and `plaingraphgps.py` imports the same function. `eps**2` (not `eps`) floors
  delta at exactly `eps`, so `lndelta` is bit-identical to the unclamped version and `lnkt` is
  too wherever `ptmin <= 1` -- every padded pair, whose pt is itself clamped to `sqrt(eps)`.
  NOT the same as the MIParT **rapidity** clamp (`(energy - pz).clamp(min=1e-20)`), which is a
  separate fix and is already upstream via #92. An earlier attempt on `original-repo-fixes`
  (7b40292) clamped at `eps`, which shifts `lndelta` from `log(eps)` to `log(sqrt(eps))` for
  every degenerate pair -- a real feature change, and the likely reason 198eba7 reverted it.
  Still absent upstream; offer with that reasoning attached or it gets reverted again.
- Embedding order: tagging features (deta/dphi/dr, log pt) are computed BEFORE the optional
  jet-rest-frame boost. Post-boost the jet has pt~0, so pt_jet clamps and eta_jet/phi_jet read
  off a numerically-zero vector -> dphi/deta rotation-unstable, breaking even SO(2) invariance
  (xyrotation max MSE O(1e2..1e4) -> O(1e-8)). The mass regulator also excludes spurions
  (`& ~is_spurion`), so a spacelike beam spurion (m^2=-1) is not forced lightlike (which would
  void the spacelike-beam ablation).

- **`tag_particlenet` runs an in-repo port of LLoCa-ParticleNet**, not the library class.
  `experiments/baselines/particlenet.py` was regenerated from `lloca.backbone.particlenet`
  1.3.6 (it had been a stale stock-weaver copy that nothing imported), so the tensorial
  message passing -- `change_local_frame`, both `get_graph_feature` variants, `knn`,
  `hidden_reps_list` -- is byte-faithful, with exactly two additions:
  (1) `for_inference` single-logit heads use sigmoid; (2) `knn_metric` gains `minkowski`,
  which seeds the layer-0 graph by `|(p_i - p_j)^2|` instead of a squared L2 on (phi, eta).
  lloca's backbone never receives the four-momenta, so this was not expressible before;
  `ParticleNetWrapper` now passes them as dense `(B, 4, P)` `v`, consumed only by layer 0 and
  only when the metric asks for it. **`deltaR` still calls the ported (= lloca's) `knn`, so
  the default path is bit-identical** -- it deliberately does not reuse the hybrid's
  metric-aware helper, which wraps `dphi` into `[0, pi]` before the L2 (azimuth is periodic)
  where lloca does not; adopting that silently would have moved the published-reproduction
  row. That baseline-vs-hybrid difference in the deltaR graph is pre-existing and left alone.
  Four tests pin the arrangement: AST equality of the four transported helpers against the
  installed lloca, whole-model bit-parity on the deltaR path, hybrid-vs-port, and
  hybrid-vs-lloca (the third edge, so a change moving both in-repo files together still
  fails).

- `config/training/top_cgenn.yaml`: the CGENN baseline now names its recipe explicitly
  (`??? ` lr/batchsize over `tag_gts_and_friends_default`) instead of inheriting the task
  default by accident. The effective recipe is unchanged -- the task default IS that same
  base -- but the reference row for both CGENN hybrids no longer silently tracks any
  future change to it, and `sbatch train.sbatch tag_cgenn` derives it by name. The
  unswept-fallback guardrail was extended to fire for CGENN too, since the file ships the
  same `???` placeholders as the hybrid recipes and omegaconf resolves an unfilled `???`
  to the parent's value rather than raising. Differences from the official CGENN recipe
  (Adam / wd 0 / batch 32 / ~8.8 epochs) are enumerated in the file's header.

## Fixed here (input pipeline & training robustness)
- CGENN `MVLayerNorm` gain reshaped (1, C) -> (C,) so it falls under the optimizer's ndim<=1
  weight-decay exemption like every other norm gain; the official 2-d shape was silently
  weight-decayed -- an undocumented regularization asymmetry hitting only the CGENN hybrids
  (broadcasting unchanged: unsqueezed to (1, C) at use).
- **CGENN's remaining gains/biases exempted from weight decay** -- `MVSiLU.a` (1, C, dim+1,
  init ones), `MVSiLU.b` (init zeros) and `NormalizationLayer.a` (C, n_subspaces). The
  `MVLayerNorm` reshape above fixed one of four parameters of the same kind; the other three
  are missed by both structural rules (ndim > 1, and named `.a`/`.b` rather than `.bias`).
  MVSiLU computes `sigmoid(a*norms + b) * input`, so decaying `a` pulls the gate toward the
  constant `sigmoid(b)`: a prior toward deleting the nonlinearity, not toward a simpler
  function -- the standard argument for exempting norm gains, already followed everywhere else
  here via ndim<=1. The **official CGENN repo settles the reference**: `top_tagging.py` builds
  `torch.optim.Adam(model.parameters(), ...)` as one flat group and the README's top-tagging
  command passes only `--optimizer.lr=0.001`, so weight decay is Adam's default 0 (their nbody
  command *does* pass `--optimizer.weight_decay=0.0001`, so the omission is a choice). Official
  top tagging decays none of these; exempting them moves toward the reference. `weight_decay:
  0.01` still applies to every real weight. Affects `tag_cgenn` and both CGENN hybrids.
  Declared via `no_weight_decay()`, computed by walking the module tree, never hardcoded.
- The **base** optimizer grouping now honours `net.no_weight_decay()` too -- previously only
  the ParT grouping did, so a net could declare exempt parameters that the GraphGPS path
  ignored. No-op for any net that does not define the method.
- `finetuneexperiment.py`'s grouping calls `net.no_weight_decay()` instead of a hardcoded
  `{"cls_token"}`, matching `experiment.py`. No behavior change today (the branch is gated on
  ParTWrapper, and ParT returns exactly that set); the two copies had drifted.
- `save=false` keeps the best-validation checkpoint in RAM instead of on disk. `_save_model` is
  a no-op under `save=false`, so the end-of-training restore used to fail and the evaluation
  silently reported the **final iterate** -- a dry run's numbers could be compared against a
  table they were not produced under. A log carrying `Cannot load best model ...` predates this.
- Streaming loaders: `infinity_mode` (file re-cycling) is TRAIN-only -- on val/test it looped
  forever; `steps_per_epoch` bounds the train epoch (JetClass + TopTagXL).
- Finetuning: `ema_decay` read from the config top level (`cfg.ema_decay`), not
  `cfg.training.ema_decay` (the old path crashed in struct mode); a fresh-trial warm start
  (`warm_start_load=false`) re-loads the pretrained backbone for an independent finetune trial.
- miniweaver hardening: a corrupt/truncated ROOT read logs file+traceback instead of a silent
  `None` (md5 covers only the tar download, not a scratch copy, so silent skips skewed class
  balance across epochs); a glob matching no files warns instead of silently shrinking the
  requested file range; the "Nothing to load for worker N" error is parenthesized so it
  actually reports the worker id.
- Quick configs: the JetClass smoke-test file ranges pointed at file numbers that live only in
  `val_5M` (so train/test resolved to zero files -> a confusing worker crash); repointed to one
  real file per split folder (train 0, test 100, val 120).

## Conventions this fork sets (upstream has no stance)
- Hybrid-family fairness: shared AdamW/schedule/budget, per-model batchsize+lr from the LR
  finder; dropout kept per-reference (ParT-side blocks 0.1, GPS and L-GATr sides 0/none);
  `use_pre_activation_pair: false` on the PNParT hybrids (published-ParT parity); a uniform
  `attn_dropout: 0.0` knob on all four GPS models (sdpa dropout_p; equivariance-safe);
  Plain-GPS `use_edge_attr` = the ParT pair features MPNN-routed (ParticleNeXt-style
  routing ablation), OFF by default so Plain stays the bare backbone.
- JetClass recipes: `weight_decay: 0`, `epochs: 5` (ParT-standard exposure), per-model
  re-sweep of batchsize/lr on the jctagging task.

## Disclosures for the methods section (per-reference choices, not bugs)
- **Head depth is per-reference, not unified.** The four GraphTrans hybrids classify with a
  single Linear from the CLS token (the official GraphTrans head); the four GraphGPS hybrids
  use a 2-layer SAN-style MLP after mean-pool (the official GraphGPS `SANGraphHead`). Both are
  faithful to their lineage, so head capacity co-varies with the GT-vs-GPS axis by design.
- **The four non-equivariant hybrids hardcode `tagging_features="all"`** in `TaggerWrapper`,
  so the `data.tagging_features` ablation moves only the equivariant rows (headline table
  unaffected).
- **`deta` uses an unconditional sign flip** (`-(eta_i - eta_jet)`), not weaver's
  hemisphere-dependent flip -- internally consistent, but the input pipeline is not
  weaver-verbatim on this one feature.

## lgatr 2.0 migration + the compile program (2026-08-07/08, dev)
- **lgatr 1.4.4 → 2.0 (Posture B / v2-native)**: pins relaxed to `lgatr[xformers-attention]>=2.0.0` (uncapped, upstream practice);
  configs adopt v2 defaults implicitly (sigmoid vector gate, affine norms, sparse geometric
  product, tanh-GeLU); every behavioral choice verified against the lgatr authors' own
  tagging-guide environment (docs/lgatr2-migration.md, H16 addendum). Gates A–F green
  (composition, manifests, full suite, identity-frames bit-exactness, blade-table);
  G (training parity) and H (throughput) remain cluster-side.
- **Compile program (docs/cgenn-compile.md)**: all eight lgatr-family + baseline models are
  torch.compile-gated with committed fixtures and per-model gate files
  (`test_{cgenn,lorentznet}_compile.py`, `test_{cgenn,lorentznet}_hybrid_compile.py`):
  BIT (`torch.equal` vs pre-change recordings, fp32+fp64) · TOL ≤ 1e-10 · DET ·
  BREAKS 0 on cold builds · RECOMP (no per-shape re-specialization, `dynamic=True`).
  Production configs ship compiled-dynamic for all eight; `config_quick` stays eager.
  Fix families: cached-property RLock warm-up, in-trace `.item()`/tensor-iteration ints,
  bool-mask scatter → integer indices, tensor-valued `repeat_interleave` → precomputed
  gathers, 3-operand einsums → opt_einsum-path-equivalent 2-op chains (bit-identical),
  dense-mask net interfaces, PyG aggregation `dim_size` from `ptr`, and kNN edge building
  hoisted out of the compiled CGENN-hybrid nets (`build_edges`, wrapper-side).
  Wrapper knobs use in-place `nn.Module.compile()` so `state_dict` keys are unchanged
  between compiled and eager runs.
- **Compile program, Stage 4 (non-equivariant family; `test_nonequi_compile.py`)**: same
  gate battery for tag_ParT / tag_particlenet / tag_transformer / tag_PlainGraphTrans /
  tag_PlainGraphGPS / the PN-ParT hybrid pair; **MIParT descoped by operator decision**
  (BIT/hash regression pins only; wrapper asserts `compile=False`, configs carry no knob).
  New fix families, all eager-default-preserving (BIT re-verified after every edit):
  ParT SequenceTrimmer warm-up as a wrapper-ticked bool (dynamo guards python ints BY
  VALUE — an in-graph counter read recompiles per step; the final audit caught and fixed
  a warm-up off-by-one in the first ticked version — first trim on forward 5 instead of
  upstream's 6 — and the tick schedule is now proven equal to upstream's in-forward
  counter for warmup 0/1/5); all-pairs PairEmbed twins
  (`compiled_dense`) replacing data-dependent `nonzero` / int-only `torch.tril_indices`
  paths in both ParT and the PNT-local PairEmbed (eval-exact, TOL-gated; training-mode
  BN pair-multiset caveat disclosed in the log); `sdpa_plain_attention` twin
  (`compiled_attention`) for identity-frames `nn.MultiheadAttention` whose bool-mask +
  float-ParT-bias preamble fires a dynamo-skipped `warnings.warn` per block; wrapper-side
  `mark_dynamic` on every ParT net input including all three `Frames` tensors. The
  GraphGPS pair's masked BatchNorm over real nodes is a **documented data-dependent break
  class** (GraphGPS-official `norm: batch`): break-event counts pinned (11 / 7), every
  reason asserted to be that class, RECOMP still strict ([10,10,10] / [8,8,8]).
  For the posture that actually ships, see the round-5 entry below — it supersedes every
  earlier posture line in this file.
- **Train-mode blind spot closed (final audit round 2)** — the single most important
  finding of the review program. Every compile gate ran under `.eval()` with a
  `no_grad`-wrapped forward helper, so nothing constrained TRAINING numerics. Measured:
  the PairEmbed twin makes the pair BatchNorm see the padded PxP grid instead of real
  pairs only, so tag_ParT diverges by **6.5e-01** in train-mode output and **15.0** in BN
  running buffers (PNParTGraphTrans 1.5e-02 / 1.1; PNParTGraphGPS 4.4e-04 / 1.1), and one
  training step moves all 217 ParT gradients by >1%. Eval stays exact, which is why the
  TOL gates were green. BN buffers are persistent state, so a compiled-trained checkpoint
  would carry padded-grid statistics into eager eval/export/finetune. ParT and
  PNParTGraphTrans therefore flipped to `compile: false` (nothing measured was lost —
  β-PERF had not run). New `test_train_mode_differential` reproduces the table every run
  and requires the clean models to stay bit-zero. Equivariant models re-checked with the
  real compile knob in train mode: 2.9e-15 / 3.3e-16 / 2.8e-17 — clean.
  **Round 5 closed the finding rather than living with it** — see below.
- **Round 4: no gate ever ran a backward** — the gap behind the gap. The round-3
  train-mode gate is itself `no_grad`, and no tagging model anywhere in `tests/` calls
  `.backward()`. Under `no_grad` inductor lowers the INFERENCE graph fine; with autograd
  on it must lower the joint graph, and **two models that shipped `compile: true` raise
  InductorError at the first `loss.backward()`** (`tag_PlainGraphTrans`,
  `tag_LorentzNetLGATrSlimGraphGPS` — symbolic shapes under `dynamic=True`). They would
  have died at the first training step. Both flipped to `false`; a backward matrix over
  all 13 knob-bearing models confirms the other 11 are fine. New gates:
  `test_compile_true_is_backward_verified` (default suite: no config may ship
  `compile: true` unless the model is in `BACKWARD_VERIFIED`) and `test_compiled_backward`
  (env-gated: runs the compiled training step, requires finite grads).
- **Round 6 (cross-revision audit, 2026-08-10): every non-lgatr model is bit-identical to
  `main`.** The gates in this branch are self-referential by construction — fixtures were
  recorded at pre-gate HEAD, which for `tag_ParT` was already after the in-repo port
  landed, so BIT pins the port against itself. This round answered the other question
  directly: build each model from the merge-base tree AND from HEAD, assign both identical
  parameters from a seeded generator, run the same batch, compare. Result, `max|dy|`:
  `tag_ParT` 0.0 (lloca's ParticleTransformer vs the in-repo port — the fidelity check the
  fixtures could not make), and 0.0 for `tag_MIParT`, `tag_particlenet`, `tag_transformer`,
  `tag_cgenn` (so the GP rewrite and the `gp_impl: sparse` default changed nothing),
  `tag_lorentznet`, `tag_PlainGraphTrans`, `tag_PlainGraphGPS`, `tag_ParticleNetParTGraphTrans`,
  `tag_ParticleNetParTGraphGPS`. Parameter-name sets identical throughout. The six
  lgatr-bearing models are excluded on purpose — 2.0 removed the qkv `in_linear` biases and
  added affine norm gains, an upstream architecture change; their isolation evidence is
  `test_lgatr_migration_parity::test_transplant_parity` (23 passed, 0 skipped).
  One real defect surfaced, and it is on `main`: `config_quick/model/tag_particlenet.yaml`
  pointed at `lloca.backbone.particlenet.ParticleNet`, whose `forward(points, features,
  frames, mask)` takes no `v`, while the wrapper passes `v=fourmomenta_local` —
  `TypeError` on every run. Production `config/` already used the in-repo net on both
  sides, so only the quick tree was affected; this branch repoints it and the failure is
  gone. Also verified: compiled and eager `state_dict` keys are identical for `tag_ParT`
  and `tag_ParticleNetParTGraphTrans` (245/245 and 303/303, strict load OK both ways), so
  the `compile: true` flip is checkpoint-safe.
- **Round 5: the pair BatchNorm is now mask-aware, and the compiled twins are faithful in
  TRAINING, not only in eval.** Round 3 recorded the divergence and shipped `false`;
  operator ruling was that compile must be accurate to the reference, not dropped.
  `particletransformer._weighted_batchnorm1d` / `_embed_weighted` compute BN statistics
  over a WEIGHT tensor equal to the eager reference multiset — tril of REAL pairs for ParT
  (whose eager default is the sparse mask-aware path), tril over ALL positions for the two
  PN-ParT hybrids (which have no sparse path). Weighted mean/var over the grid then equal
  unweighted mean/var over the eager multiset, and it traces because `w.sum()` is a scalar
  tensor rather than a shape. Train-mode delta / BN buffers: **1.554e-15 / 1.110e-14**
  (ParT), **3.109e-15 / 5.578e-13** (PNParTGraphTrans), **2.776e-17 / 5.578e-13**
  (PNParTGraphGPS) — was 6.5e-01 / 15.0, 1.5e-02 / 1.1, 4.4e-04 / 1.1. All three then
  passed `test_compiled_backward` (217/217, 85/85, 64/64 params with nonzero finite grads)
  and joined `BACKWARD_VERIFIED`. **Posture now shipping:** `compile: true` for ParT and
  PNParTGraphTrans (clean graphs, forward and backward); `false` stays for PlainGraphTrans
  and LNetSlimGraphGPS (backward InductorError — correctness) and for the two GPS models
  (masked-BN graph splits — a *performance* posture β-PERF resolves in one line, no longer
  a correctness one). Gate: `TWIN_TRAIN_DIVERGENT` → `TWIN_TOL_MODELS` + `TRAIN_TOL=1e-10`,
  asserting agreement instead of documenting disagreement.
- **Round 4: the trimmer silently cut the learned-framesnet gradient at step 6.**
  `_forward_encoder` wraps the trimmer in `no_grad` and re-enabled grad only for the frame
  matrices; under learned frames `x`/`v` also carry framesnet gradient, so once `_trim`
  engages past warm-up the feature path was detached — bit-identical forward, **42%**
  relative framesnet-gradient error. Fixed by extending `enable_grad` over the whole
  trimmer call; verified by controlled A/B at identical warmed state (forward `max|dy|=0`).
- **EMA best-checkpoint snapshot aliased (final audit round 2)** — `torch_ema.state_dict()`
  returns `shadow_params`/`collected_params` as python *lists*, so the `save=false`
  in-RAM snapshot's `torch.is_tensor(v) ... else v` comprehension stored them by
  reference and `ema.update()` mutated the "best-val" snapshot in place (measured: a
  value captured at 1.0 read 11.4 later, same object). Under `save=false` + `ema=true`
  the reported row painted final-iterate shadows over correctly-restored best-val
  weights. Fixed with a recursive by-value copy helper; verified drift 0.0, no aliasing,
  snapshot still loads.
- **Final audit (3 independent legs; log entry in docs/cgenn-compile.md)**: fixed a
  CRITICAL latent NaN — both all-pairs PairEmbed twins computed pair features
  grad-enabled (eager wraps them in no_grad); under a learned framesnet, backward
  through sqrt(0) NaN'd the framesnet on step one. `detach()` at twin entry restores
  exact eager gradient semantics (verified in all 8 module configs + full-model +
  re-gated). Also: trimmer post-warm-up trim hoisted into `@torch.compiler.disable`
  `_trim` (per-step re-specialization regime closed); ParTWrapper soft
  `maybe_mark_dynamic` (hard marks crashed under learned frames); lgatr parity-test
  monkeypatch leak fixed (try/finally); anti-vacuous gate asserts; doc/status
  corrections and the stale MIParT explain artifact removed. Weaver-core cross-check:
  convergent compile design (trimmer disable, sparse-pair-off-under-compile), ours
  break-free and gate-pinned where theirs eats breaks / rounds lengths to 32.
- **Final operator round (2026-08-09)**: the "15 known pelican-FLOPs environment
  failures" were a MISDIAGNOSIS — the FLOPs harness forced only
  `model.compile`/`model.net.compile` eager and missed nested knobs (the pelican
  equivectors config ships `net.compile: true`), so FX hit dynamo-compiled modules.
  Recursive compile-forcing in both FLOPs tests → all 15 pass; expected suite state is
  now ZERO failures (correction markers in the migration log, OSCAR, cgenn-compile,
  cleanup). Hardening ports, both BIT/gate-sealed: the CGENN hybrid's `CliffordAlgebra.b()`
  gets the vendored copy's on-device index tensors (live per forward via MVLayerNorm;
  was a per-forward host→device transfer on GPU eager); vendored `MVSiLU` switched to
  `grades_list[1:]` per cliffordalgebra's own compiled-region convention (identical
  values). `scripts/bperf.py` added — the post-merge β-PERF matrix runner
  (measures the 10→100-iteration steady-state window, emits the table + one-line knob
  recommendations, `--apply` edits the yamls; scheduled for deletion in cleanup.md once
  its numbers land in the log). BREAK_BARS annotated as torch-2.13-measured pins.
  Operator ruling recorded in cleanup.md: compile gate files + fixture packs are
  port instruments and get DELETED post-merge (after the §4b-ter dedup port and β-PERF
  consume them), trading away provable-safe refactors knowingly.
- **CGENN `gp_impl: einsum|matmul|sparse`** on baseline + both CGENN hybrids (default
  `sparse` = lgatr 2.0's `sparse_gp` posture; `einsum` is the BIT-pinned reference; matmul
  is the dense-GEMM form). TOL-IMPL gates per impl; only `sparse` changes the FLOPs column.
- **Trials/error bars**: shared-run-dir warm-start trials are the canonical mechanism
  (lineage-keyed raw scalars in `table_metrics_*.json`); `utils/aggregate_table.py` also
  groups independent run dirs at parse time with refuse-to-pool guards (disagreeing
  iters/params/FLOPs, identical-metric seed clones, mixed in-run rows). GUIDE §8 + OSCAR §6.
