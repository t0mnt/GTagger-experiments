# Audit ledger — resolved findings and fidelity notes

Moved out of `todo.md` (which tracks ACTIONABLE work) so the resolved audit trail stays
citable — several entries back the paper's fidelity claims (BatchNorm-over-padding
faithfulness, dropout provenance, the boost_jet decision measurements).

---

## `boost_jet` — feature/boost ordering + rotation-frame interaction (RESOLVED)

### `boost_jet` — feature/boost ordering + rotation-frame interaction (DECISION NEEDED)

**Background.** `data.boost_jet` (default **`true`** on top-tagging via `config/toptagging.yaml`;
`false` on the base `tagging.yaml`, inherited by JetClass/TopTagXL) boosts every jet to its own
rest frame inside `embed_tagging_data` before the backbone sees it — "to avoid large boosts,"
i.e. a numerical-stability aid for the framesnet's frame prediction. `init_physics` forces it
**off** for equivariant models and for identity-frames non-equivariant models, so `boost_jet=true`
is reached by exactly the **non-equivariant + learned-frames rows** (the LLoCa-on canonicalized
models: ParT/transformer/graphnet baselines and the four non-equivariant hybrids). Two loci
consume the boost; the audit's R-A1 flagged both.

**Locus 1 — FIXED (embedding-level features → framesnet). Correctness bug, done.**
`embed_tagging_data` used to compute the 7 tagging features (`log_pt, log_energy, log_pt_rel,
log_energy_rel, dphi, deta, dr`) *after* the boost. In the rest frame the jet has ~0 three-momentum,
so `pt_jet` hits its clamp and `φ_jet/η_jet` come from `atan2/eta` of a numerically-zero vector →
the 4 jet-relative features (`log_pt_rel, dphi, deta, dr`) were measured against an arbitrary axis
built from float residuals of the boost. That axis is **not** covariant, so these features — which
feed the **framesnet** as scalar inputs (`scalars_withspurions = cat([scalars, tagging_features])`,
wrappers.py) — broke the model's Lorentz invariance. Measured: `xyrotation` end-to-end output
max-MSE **O(10²–10⁴)** before, **O(10⁻⁸)** after; feature distributions also mismatched the
hardcoded lab-frame standardization constants (`log_pt_rel` mean ~12 vs the code's −4.7). **Fix
(committed):** compute the features in the lab frame first, *then* boost the momenta for the
backbone (`boost_jet=false` paths bit-identical). Invariance suite 24/24 green after.
  - **Comparability sub-decision:** this changes trained-model results for every learned-frames
    row. Upstream's published LLoCa numbers were trained with the old (degenerate) features, so if
    bit-comparability with their table matters, gate the reorder behind a flag / versioned note.
    Otherwise accept it and re-baseline (recommended — the old behavior is simply wrong).

**Locus 2 — NOT fixed (backbone-level recompute). The open decision.**
The `TaggerWrapper` backbone separately recomputes *local* tagging features from the **boosted**
momenta (`get_tagging_features(fourmomenta_local, jet_local, "all")`). The embedding-level reorder
does not reach this, and it can't be patched mechanically: computing local features from *pre*-boost
momenta while the backbone sees *post*-boost momenta would be a cross-frame subtraction (features
describing a different frame than the tokens) — likely worse. The only self-consistent option is a
config choice: **turn `boost_jet` off for the frames where it misbehaves.**

*Which frames misbehave (measured — determine empirically, the set is subtle):* a frame that
**fixes the time axis** cannot restore the boosted jet's momentum, so it stays at `(M,0,0,0)` and
the local jet-relative features degenerate. Transverse `pt` of the wrapper-local jet, by frame:

| frame | wrapper-local jet pt | strands? |
|---|---|---|
| `learnedpd`, `learnedso13` (full Lorentz) | ~10²–10³ | no |
| `learnedso3` (SO(3) rotation) | ~1e-11 | **yes** |
| `learnedso2` (SO(2) about beam) | ~1e-11 | **yes** |
| `learnedz` (z-boost) | real \|p_z\| | no (degenerate only in the transverse plane) |
| `learnedrest` (contains a boost) | ~6e5 | no |

So the earlier guess "(so3, so2, rest)" was wrong to include `rest` (it boosts). The pure-rotation
frames **`so3`, `so2`** strand the jet; `z` only transversely.

*What actually degenerates (measured — NOT "dead constants"):* every channel still varies across
constituents. `log_pt_rel` becomes a shifted **exact duplicate** of `log_pt` (corr 1.0 → one wasted
channel); `dphi/deta/dr` become the constituent's **absolute** local angles at the wrong
standardization scale (`dr` mean ~2.1 vs expected ~0.2). It is **NOT an invariance break** (features
computed in the canonical frame stay invariant even when degenerate — which is why 24/24 still
passes), and it touches only these specific ablation configs.

**Physics (this is the decisive argument, and why the time vector matters).** A time direction is
provided to essentially every model: `add_time_reference: true` adds a time spurion `[1,0,0,0]`
alongside the two beam spurions `[1,0,0,±1]` — a direct input token for the equivariant models and a
framesnet input for the LLoCa models. Rotation-only frames (`so3`/`so2`) exist **precisely** for the
regime where boosts are physically meaningful — the regime the beam/time reference *defines*.
`boost_jet` then boosts each jet to rest, i.e. it **discards exactly the boost information those
frames were chosen to preserve.** So `boost_jet` + pure-rotation frames is not just numerically
degenerate — it is **self-defeating**: you picked rotation-only canonicalization to keep boost info,
then boosted it away. `boost_jet=false` for `so3`/`so2` is therefore the *physically consistent*
choice, independent of the numerical degeneration.

**The remaining tradeoff.** `boost_jet`'s original job is framesnet numerical stability (small
boosts are easier to predict frames for); turning it off feeds the framesnet lab-frame momenta.
For 500–1000 GeV top-tagging jets this is a modest-boost regime, so the stability cost is likely
minor — but it is a real feature-quality/physics-consistency **vs** framesnet-stability tradeoff,
and it only affects the `so3`/`so2` ablation rows. (Aside, not part of this decision: `boost_jet`
also boosts the beam/time **spurions** per-jet — turning fixed lab references into jet-rest-frame
ones — for *all* frames; covariant so probably fine, but note it if you revisit the spurion design.)

**Decision — TAKEN: (a), with the set determined empirically (probe: median pt/E of the
frame-local jet, quick tree, float64, random-init framesnet):**
  - [x] **(a) `init_physics` forces `boost_jet=false` for `LearnedSO3Frames`/`LearnedSO2Frames`**
        — measured stranded at pt/E ≈ 4.5e-15 (vs 0.75 for pd/so13). **`learnedz` is NOT in the
        set**: empirically un-stranded (pt/E ≈ 0.37, eta_jet ≡ 0 — its frames carry TRANSVERSE
        boosts that restore the jet's momentum in the transverse plane), so it keeps `boost_jet`.
        The earlier "z strands transversally" guess was wrong — exactly why the set had to be
        measured, not named. (`learnedrest` + `boost_jet` fails loudly inside lloca itself —
        "Trying to boost spacelike vectors into their restframe" — a separate, loud issue.)
        Also applied to `original-repo-fixes` for the upstream PR (same latent interaction there).
  - [ ] ~~(b) leave as-is~~ / ~~(c) drop so3/so2~~ — superseded by (a).
  **Not upstream-exclusive — the identical latent interaction is present on upstream** (same
  `TaggerWrapper` recompute from boosted momenta; `learnedso3.yaml`/`learnedso2.yaml` ship there;
  `init_physics` keeps `boost_jet=true` for non-equivariant **learned-frames** rows — the
  `boost_jet=False` fallback fires only for *identity* frames). So `model=tag_ParT
  model/framesnet=learnedso3` on top-tagging strands the jet on upstream too. It is kept OFF the
  upstream `original-repo-fixes` PR because it is a **design tradeoff (feature-quality vs framesnet
  stability), not a correctness bug** — no invariance break, features stay invariant-but-degenerate —
  NOT because upstream is immune. Upstream's headline results simply use full-Lorentz `learnedpd`
  frames, which don't strand; their so3/so2 subgroup ablations (if run on top-tagging with the
  default `boost_jet=true`) hit the same mild degeneration. If the decision below is taken, the
  `init_physics` fix is equally applicable upstream and could be a follow-up PR there.


---

### Audit findings (property-based sweep — permutation / mask / determinism / degenerate jets)
- [x] **BatchNorm-over-padding is FAITHFUL to official ParticleNet/ParT — verified, do NOT "fix".**
      The property test showed a real effect: in *training*, padding the same jet to more columns
      shifts the logits by up to 0.18 (ParticleNet-ParT GraphTrans), 0.08 (Plain GraphTrans), ~1e-3
      (the two GPS), because the input `bn_fts` (`nn.BatchNorm1d`, all 4 channels-first models) and the
      EdgeConv/MPNN `BatchNorm2d/1d` (Plain + ParticleNet GraphTrans) compute statistics over the
      zero-padded slots. **Checked against the references**: weaver ParticleNet does
      `self.bn_fts(features).masked_fill(padding_mask, 0)` (BN over the full padded tensor, mask
      *after*) with unmasked EdgeConv `BatchNorm2d`; weaver ParT's `Embed.input_bn` is a
      `nn.BatchNorm1d` over `(batch, channels, seq_len)` with no pre-mask (zeroed only after embed).
      Our ports reproduce both exactly, so this is intended fidelity, not a bug — masking it would
      DIVERGE from the architectures being compared. (The GraphGPS per-layer `MaskedNorm` is likewise
      faithful to the *GraphGPS* recipe's masked BatchNorm; each backbone matches its own lineage.)
      Eval is bit-exact (running stats). Nothing to change; documented for the paper's fidelity claim.
- [x] **Verified clean (all 8, float64):** determinism (bit-exact), permutation-invariance over
      particles (~1e-16), padded-VALUE leakage in eval (bit-exact), padding-COUNT invariance in eval
      (bit-exact), finite logits on degenerate 1-particle jets, **gradient coverage** (every trainable
      param reached by the loss), **batch-composition independence** (a jet's logits identical alone vs
      batched, ~1e-16 — no cross-jet leakage), and **identical-particle / collinear jets finite**.
      Set-symmetry, eval-time masking, batch isolation and numerics are sound across the family. (The
      `embed_tagging_data` in-place-`ptr` footgun found en route is documented at `embedding.py:103`.)
- [x] **Benign dead vector-path in `LorentzNetLGATrSlimGraphTrans`** (grad-coverage check). The
      `lgatr` LGATrSlim's `linear_out` multivector weights and the *last* block's MLP multivector
      weights get no gradient — they feed only the deliberately-discarded vector output
      (`out_v_channels=1`, "vector sink"; the model reads `_, s_out`). Intended, not a bug; a few wasted
      multivector params. (Earlier blocks' vector weights are live — vectors reach later scalars via
      attention. The GPS sibling routes differently and has none.)

### Audit findings (full GraphTrans-vs-GraphGPS sweep) — remaining, low priority
- [ ] **Dropout is inconsistent across the hybrid family — two layers to the decision (checked):**
      (a) *GPS local branch (latent)*: Plain + CGENN GPS apply an external `Dropout` to the local-MPNN
      output; LorentzNet + ParticleNet GPS apply none. No-op at shipped defaults — ALL FOUR GPS configs
      ship dropout 0/None, so at defaults the GPS family is behaviorally equal; only matters if the
      dropout ablation is ever run.
      (b) *GraphTrans transformer stage (LIVE)*: `tag_PlainGraphTrans` ships `dropout: 0.1` and
      `ParticleNetParTGraphTrans` gets 0.1/0.1/0.1 from its ParT-block class defaults (config sets no
      dropout keys), while CGENN/LorentzNet GraphTrans run dropout-free (`dropout_prob=None`). So the
      two non-equivariant GT hybrids train WITH dropout and the six other hybrids without — a live
      regularization asymmetry across both comparison axes (GT-vs-GPS and equivariant-vs-not).
      Each stage is *faithful to its source*: ParT blocks publish 0.1 — and the repo's `tag_ParT`
      reference row DOES train its 8 main blocks at 0.1 (its config zeroes only `cls_block_params`,
      weaver's own convention for the class-attention blocks) — L-GATr publishes none, and pure
      `tag_cgenn`/`tag_lorentznet` use 0.2 on their classification heads only (LorentzNet's LGEB
      dropout kwarg is dead code). So the consistent chains are: ParT-baseline 0.1 ↔ PNParT-GT 0.1
      (faithful), and L-GATr-none ↔ equivariant hybrids none (faithful). The one axis where dropout
      is confounded with the comparison is **GT (0.1) vs GPS (0) within the two non-equivariant
      backbones**. Decision: treat dropout as part of the reference block definition (per-reference,
      like FFN ratio and GELU/ReLU — keep as-is + one methods sentence + the existing family-wide
      dropout ablation row), OR harmonize the family to 0 (zeroing PNParT-GT breaks its faithfulness
      to the ParT baseline row it is directly compared against).

### Audit findings (infrastructure sweep: JetClass path / plots / trials)
- [x] **`jc_gts_and_friends_default` added** and `config/jctagging.yaml` now defaults to it (was
      `jc_ParT` — the same recipe-inheritance trap the top tree fixed). ParT-standard `epochs: 5`
      (1M steps x 512 = ~5 passes of 100M), CosineAnnealingWarmup, wd 0 (JetClass convention),
      validate once per nominal epoch.
- [x] **per-model `jc_<Hybrid>.yaml` recipes added (all 8)**, mirroring `top_<Hybrid>` on
      `jc_gts_and_friends_default`. See GUIDE §5.1. (`jc_lgatr`'s broken `tag_gatr` base was
      fixed on `main` directly — merges in cleanly.)
- [ ] **JetClass: fill the 8 `jc_<Hybrid>.yaml` `???` batchsize/lr** from
      `utils/find_lr.py -cn jctagging model=tag_<hybrid> save=false +lr_find.find_batch_size=true`
      before the JetClass campaign (don't copy top values — inputs are 7+10 channels; and note
      an unfilled `???` silently runs at the 512/1e-3 fallback instead of erroring).
- [ ] **Rejection-metric convention differs between experiments** (pre-existing): top-tagging uses the
      nearest-ROC-point (`argmin |tpr - epsS|`), JetClass uses `scipy.interp1d` interpolation. One
      methods sentence, or unify.
- [x] **best-checkpoint restore now re-pairs the EMA**: the end of `train()` loads the checkpoint's
      `"ema"` alongside `"model"` (when `ema: true`), so the `_ema` eval uses the EMA shadow that
      belongs to the restored best-validation checkpoint instead of the end-of-training one.

### Pre-publication audit (session: jet_frames + GT-family sanity sweep)

Training-readiness verified across all 8 GT hybrids (real `config/`): forward + backward + AdamW step
crash-free; param counts 1.16–2.53M (LorentzNet 1.83/2.46M, CGENN GNN 248k — the earlier fixes held;
small later deltas: the audit's node_attr re-injection adds +1.5k/+6.7k to the LorentzNet hybrids and
the official-CGENN knob flip removes the NormalizationLayer params, so counts are now 1.15/1.90/1.83/2.46M
for CGENN-Trans/CGENN-GPS/LN-Trans/LN-GPS);
**zero dead input channels** in either the four-momentum path or the 7 `tagging_features` (the general
form of the CGENN `node_attr` check — CGENN comes back balanced). PDFrames runs end-to-end on the 4
non-equivariant hybrids (Plain × {Trans, GPS}, ParticleNet-ParT × {Trans, GPS}); the 4 internally-equivariant
hybrids (CGENN, LorentzNet) **assert `IdentityFrames`** by design (`self.framesnet = framesnet  # not actually
used`; `IdentityFrames` is 0-param and never referenced — a perfect no-op). The `/20` momentum rescale is
inherited from the standalone references (equivariant backbones rescale manually; non-equivariant ones
canonicalize via `TaggerWrapper` + BatchNorm). Equivariance 32/32 (both frames), `test_amplitudes` fixed,
training smoke (`PlainGraphTrans + learnedpd`) ran end-to-end through evaluation.

- [ ] **`torch.cuda.amp.autocast(...)` deprecation in 4 active baseline files** (plaingraphtrans.py:300,
      plaingraphgps.py:403, particlenetpartgraphgps.py:229, particlenettransformer.py:812; mipart.py
      has 2 more in commented-out code). `FutureWarning` today, error in some future torch. Mechanical
      migration to `torch.amp.autocast('cuda', ...)`; will not change current numerics.
- [ ] **ParT-GPS mixed-type attention mask deprecation** at particlenetpartgraphgps.py:110 — float
      `attn_mask=attn_bias` paired with bool `key_padding_mask` triggers torch's "mismatched
      key_padding_mask and attn_mask is deprecated" warning. Functionally correct today (padding still
      goes to −∞); future-fatal. Fix: merge the bool padding mask into the float bias (`bias.masked_fill(pad, -inf)`)
      and pass a single float `attn_mask`.
- [ ] **`xformers` env note for the SLURM target** — the installed wheel must match the cluster's
      torch+python or the L-GATr `lgatr` equivectors silently fall back / fail to load (this is the same
      class as the 9 environment-only FLOPs failures in this dev container). Pin a known-good
      (torch, xformers) pair in `docs/SLURM.md` under the install step; matters only for runs that
      actually use `lgatr` equivectors.
- [ ] **Precision-floor note for the paper** — `learnedpd` carries a higher boost-precision floor than
      `learnedso13` (float64, polar decomposition divides by energy). Measured at ~1e-4 absolute (kNN, 10
      boosts) on the GT hybrids — far below any true symmetry break and consistent with the standalone
      baselines (ParT ~1e-4, ParticleNet ~1e-7 same conditions). The test file already encodes per-frame
      tolerances; one sentence in the methods section would head off reviewer questions.

- [x] **jet_frames lloca-compat fix** — `TaggerWrapper.jet_frames` always uses the 4d orthogonalizer
      but reused the framesnet's `ortho_kwargs` (which the PD family keys as `eps_reg`, the 3d name).
      Translate the key for the 4d call so any framesnet works.
- [x] **jet_frames missing `num_graphs`** — set-level equivectors (`pelican`) need it; mirror the main
      framesnet path. `equimlp` absorbs it via `**kwargs`.
- [x] **`test_tag_equivariance.py::test_lloca_frame_invariance`** now parametrizes over both `learnedpd`
      and `learnedso13` (LLoCa's recommended default is PD; `so13` was the only frame tested before, which
      hid the jet_frames bug). Per-frame tolerances (`so13` ≤ 1e-3, `pd` ≤ 2e-2) — 16/16.
- [x] **`test_tag_invariance.py::test_amplitudes`** had a stale config key (`data.tagging_features_framesnet=null`,
      removed upstream by `f08f7df`/`a45da1b`) — every case failed at config composition before any model ran,
      so the baseline-under-frames check was silently dead. Aligned to `data.tagging_features=null`; 16/16.

- [ ] **LorentzNet GraphGPS never zeroes padded slots between its layers** (only at the final pool), so the
      shared `LorentzNetKNNBlock`'s BatchNorms accumulate over nonzero padded state across the 10 layers
      (GraphTrans zeroes after its GNN stack). Logits are unaffected (readout is masked) but BN running
      stats drift — cosmetic; zero padded slots per layer if you want exact parity.
- [ ] **(latent, both LorentzNet variants)** `phi_e` BatchNorm in `LorentzNetKNNBlock` normalises over
      *invalid* edges too — the edge mask is applied only *after* `phi_e`. Pre-existing, shared by both
      variants (not a Trans-vs-GPS divergence); mask before `phi_e` for cleanliness.
- [x] **"LorentzNet mean"** (scalar message aggregation was mean, should be sum) — already fixed in the
      shared block (`h_msg = m.sum(-1)`, commit `8a7b5fc`) and inherited by GraphGPS; both now match
      official LorentzNet (sum scalars / mean vectors).

---

## Pre-existing defects on `main` — found by the dev-branch audit, NOT introduced by it

Recorded here rather than in `todo.md`: `todo.md` is the dev/campaign work queue, and
these are `main`-side defects outside this branch's scope. Each was verified present on
`origin/main` as well as dev. Each changes an existing recipe, so each wants its own
commit, its own test, and its own decision — not a merge rider.

### PRE-EXISTING (main-side) bug: finetune LGATr branch ignores weight_decay

Found in the final audit; **not introduced by the dev PR** (`git log origin/main` shows
the file's grouping predates it), so deliberately NOT fixed inside the merge — recorded
for a separate, testable change because it silently alters an existing recipe.

`experiments/tagging/finetuneexperiment.py::_init_optimizer`, the
`LGATrWrapper`/`LGATrSlimWrapper` branch builds its param groups with `lr` only — no
`weight_decay` key. No optimizer in `base_experiment._init_optimizer` passes a top-level
`weight_decay=`, so those groups fall back to **torch's own default**: 0.01 for AdamW,
0 for Adam/RAdam — never `cfg.training.weight_decay`. So finetuning an L-GATr backbone
silently ignores the configured decay (and, with AdamW, decays the norm gains that the
H15 exemption exists to protect). The ParT and Transformer branches in the same function
set the key correctly, so this is a per-branch omission, not a design choice.

Fix when taken: add `"weight_decay": self.cfg.training.weight_decay` (and the framesnet
variant where applicable) to that branch, plus a test asserting every group returned by
each `_init_optimizer` branch carries an explicit `weight_decay`. Note it CHANGES
finetune results, so it wants its own commit and a line in the methods notes.

### PRE-EXISTING (main-side) EMA / dtype-ordering defects — recorded, not fixed here

All four verified on `origin/main` as well as dev, so they are NOT introduced by this PR
and are deliberately left out of the merge (each changes an existing recipe and wants its
own commit + test). Found by the final audit's aliasing/state sweep.

1. **Finetune EMA is silently disabled.** `finetuneexperiment.py` does
   `self.ema = ExponentialMovingAverage(...).to(self.device)`, but `torch_ema`'s `.to()`
   returns `None` — so `self.ema` is `None` immediately after logging "Re-initializing
   EMA". Every `if self.ema is not None` then takes the non-EMA branch: no updates, no
   EMA validation, `"ema": None` in every finetune checkpoint. `base_experiment` does it
   correctly (construct, then `.to()` on a separate line). One-line fix.
2. **EMA shadow params are float32 under `use_float64: true`.** `init_model` builds the
   EMA *before* `self.model.to(dtype=self.dtype)`, and torch_ema's `.to()` is called with
   device only, so the shadows stay fp32 and `sub_`/`copy_to` silently truncate. Fix =
   build the EMA after the dtype conversion (or pass dtype).
3. **fp64 checkpoints truncate to fp32 on restore.** `load_state_dict` runs on the
   freshly-instantiated fp32 model and only afterwards is the model widened; `copy_` casts
   down and the widening cannot recover it (measured max rel error 5.4e-08 = fp32 eps).
   Affects warm-start continuation, eval-reload and finetune backbone loading. Fix =
   `.to(dtype)` before `load_state_dict`.
4. **Amplitude standardization overwritten after warm start.** `AmplitudeWrapper` /
   `GraphNetWrapper.init_standardization` have no `inited` guard (every tagging
   equivalent does), and `init_data` runs after `init_model` has loaded the checkpoint.

Also recorded (inherited from lloca/pelican, not our code): `self.__class__ =
torch.compile(self.__class__)` in the ParT port and in `pelican/nets.py` mutates the
CLASS in place, so one instance with `compile=True` routes every instance of that class
in the process through dynamo — including instances whose own config says false. This is
why the FLOPs harness's force-eager walk can be defeated within a single pytest process
if a pelican model was built compiled earlier in the run. Not currently observed as a
failure (the FLOPs suite passes 64/0/36), but it makes process-order a hidden variable.


---

## 2026-08-11 — `tag_cgenn` aborts in the compiled multi-step training smoke (pre-existing)

`CGENN_COMPILE_GATES=1 CGENN_SMOKE_COMPILE=1 pytest tests/experiments/test_training_smoke.py
-k tag_cgenn` dies with **`Fatal Python error: Aborted`** (SIGABRT, exit 134), inside
`torch/_functorch/_aot_autograd/subclass_codegen.py:codegen(compiled_function_backward)`
with ~240 inductor kernels loaded — i.e. while executing AOT's compiled backward.

**Not caused by the sparse-GP work.** Reproduced identically at `3fe5197`, the commit
before any of it, in a clean worktree.

**Cause: CPU-inductor OpenMP threading, not the model.** `OMP_NUM_THREADS=1` makes the same
command PASS — 3 passed in 21m56s (tag_cgenn plus both CGENN hybrids) where the default
thread count aborts in seconds. So the arithmetic, the graph and the gradients are fine; the
generated C++ kernels are what fall over, and only when they are threaded.

That is CPU-specific by construction: the campaign's GPU inductor emits Triton kernels and
does not use OpenMP for them, so this particular abort cannot occur there. It is still worth
knowing, because it means the compiled CGENN smoke is only runnable on this box with
`OMP_NUM_THREADS=1`, and anyone who runs it without that will read a hard crash as a model
defect.

Scope, measured, so nobody over- or under-reads it:

- `tag_PlainGraphGPS` with compile ON passes the same smoke in 24 s, so this is not "the
  compiled smoke is broken" — it is CGENN specifically.
- `test_cgenn_compile.test_compiled_backward[sparse|matmul|einsum]` PASSES: a compiled
  CGENN training step producing 37 finite gradients. So a single compiled step is fine and
  something about the multi-step smoke path is not.
- The box has 15 GB RAM with 12 GB free at the time, so it is not obviously an allocator
  death, though CPU inductor building 240 kernels for a joint graph is heavy.
- CPU inductor emits C++; the campaign's GPU inductor emits Triton. A CPU-only abort is
  weak evidence about the campaign, in either direction.

**Why no gate caught it:** `test_training_smoke` forces `compile: false` unless
`CGENN_SMOKE_COMPILE=1` is set, and nothing in CI or the standard gated run sets it. So
CGENN's compiled MULTI-STEP training has never been exercised anywhere — the compile gates
are single-step or `no_grad`, and this is the fourth time in this program that a `no_grad`
or single-shot gate has been mistaken for a statement about training.

**Before committing days to a compiled CGENN run**, still do the §3-style short real run on
the GPU that the runbook already recommends for `tag_slim`, and watch it past the first
validation — not because of this abort, which is CPU-only, but because compiled CGENN
multi-step training has no gate anywhere and the GPU path is where it would first be
exercised for real. If it dies, `model.compile=false` on the launch line is the
one-character fallback and costs only the compile speedup, which has never been measured on
GPU anyway.

To run the compiled CGENN smoke on a CPU box:

    OMP_NUM_THREADS=1 CGENN_COMPILE_GATES=1 CGENN_SMOKE_COMPILE=1 \
        pytest tests/experiments/test_training_smoke.py -k tag_cgenn    # ~22 min
