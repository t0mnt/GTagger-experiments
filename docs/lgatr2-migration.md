# lgatr 1.4.4 → 2.0.0 migration runbook

**Status: PLANNED — no migration work has been done. This document is the plan.**

- **lgatr 2.0.0 was released on PyPI on 2026-07-29.** This runbook (rev 2, same day) is pinned to that release: its CHANGELOG `[2.0.0]`, its `docs/source/v1_to_v2.rst`, and source diffs against installed 1.4.4.
- `requirements.txt` on this branch pins `lgatr[xformers-attention]==2.0.0` **exact** (H14(3); relaxed to a range only in Phase 5). **The environment has NOT been migrated** — installed lgatr is still 1.4.4, and Phase 0 (fixture capture) *requires* 1.4.4. Do not `pip install -r requirements.txt` into the fixture-capture environment.
- lloca stays frozen at `1.3.6` for the whole migration (one variable at a time).

**Rev 2 changes vs rev 1 of this runbook** (after reading the official v1→v2 doc + CHANGELOG):
1. The official migration doc is **accurate but incomplete — it documents renames only**. Every behavior-changing default lives only in the CHANGELOG (§2.3). Porting by the doc alone yields code that runs and silently instantiates a slightly different model.
2. Rev 1's Gate C (same-seed fresh-init forward comparison) is **invalid and retired**: v2 changes initialization (`linear_s` bias zeroing, Slim init/scaling refinements) and adds/removes parameters, so identical seeds cannot produce comparable models across versions. Parity is now proven by **weight transplant** (§3).
3. **Exact parity of the shipped model is impossible on v2** — two changes are baked in with no flag: the Slim GLU vector-gate `0.5 = 1/sqrt(4)` scale and the removed qkv scalar bias. What remains fully achievable is exact parity of the *verification*: transplanted weights + two documented compensations reproduce 1.4.4 outputs to fp64 precision, proving the port introduced nothing *unintended*. §4 Phase 3 then makes the shipped-model posture an explicit decision.
4. New break items: channel-last Slim **blocks** (M8), `compile_kwargs` dynamic default (M9), qkv bias (S5), GLU gate scale (S6), init refinements (S7), AMP strategy (S8).

**Rev 3 changes vs rev 2** (same day; full-source re-sieve of the v2.0.0 tag against installed 1.4.4 — every layer, primitive, net, backend, and interface file):
1. One new hard break found: **M10** — `primitives: PrimitivesConfig` is now a *required* constructor argument on directly-constructed layers (`SelfAttention`, `GeoMLP`, and `EquiLinear`, where it is the **third positional**).
2. **S2 broadened**: `norm_elementwise_affine=True` is the new default on **all** v2 nets (full LGATr included, with per-grade `(mv_channels, 5)` gains), not just slim. Conversely, the *layer-level* `EquiLayerNorm` default is `False`, so the GPS hybrid's bare constructions are safe.
3. **S5 sharpened** to the exact tensors: v2 sets `bias=False` at all four qkv construction sites; v1's qkv scalar biases live in `linear_in.linear_s` (slim) and EquiLinear's internal `s2mvs`/`mvs2s` Linears (full) with **nonzero** uniform init.
4. Several feared breaks verified as **non-breaks**, most importantly the CPU attention path (the suite survives v2's CUDA gate because `experiments/misc.py` already falls back to dense masks on CPU).
5. Scope statement: **weight transplant is a parity-verification instrument only.** Porting trained checkpoints between lgatr versions is a non-goal of this migration — no checkpoint crosses the boundary (H7).

**Rev 4 (2026-08-05) — the inventory was executed against the real wheel, not only read.**
`lgatr-2.0.0-py3-none-any.whl` was downloaded and imported side-by-side with the installed
1.4.4 (path-injection, no environment change — Phase 0 stays capturable). Every M and S row
below was checked by construction and by parameter inventory. Results:

- **Confirmed exactly as written**: M3 (`attn_ratio`), M4 (`nonlinearity` / `mlp_ratio` /
  `num_layers_mlp`), M5 (`SlimRMSNorm(v_channels, s_channels, ...)`), M6 (`in_s_channels: int = 0`),
  M8 (layers are `(..., 4, channels)`, the net stays `(..., items, channels, 4)`),
  M9 (`compile_mode` / `compile_dynamic` both raise `TypeError`), M10 (`primitives` is the
  **third positional** on `EquiLinear` and required on `SelfAttention` / `GeoMLP`),
  S1 (`nonlinearity_v="sigmoid"`), S5 (`bias=False` at exactly four qkv sites),
  S6 (`0.5` at `slim_layers.py:368-369`, comment and all), and §2.1's claim that the
  *layer-level* `EquiLayerNorm` default is `False` while nets default it `True`.
- **S2 / S5 sharpened to exact tensors** — see the diffed parameter inventory in §2.5.
- **One row was wrong**: M7 said `>=2.0.0`; `requirements.txt` on this branch actually pins
  `==2.0.0`, as the header states. Corrected below.
- **One new non-break**: `LGATrSlim.__init__` positional order changed (inert — all constructions here are keyword); recorded in §2.1.
- **External corroboration**: an independent sibling fork ported to 2.0 and its commit
  reproduces this runbook's central prediction — see §2.6.

---

## 0. TL;DR

- Effort: ~2.5 focused days for this fork (+1 for surprises), ~1 day for upstream `lloca-experiments`.
- Approach: **record → port → prove.** Golden fixtures captured on 1.4.4 *before the environment changes*; migration complete only when every Phase 4 gate passes. "Port it carefully" is not a gate.
- Verification is **transplant-based**: v1 state_dicts loaded into v2 models through an explicit key map, two compensations, and a waiver list. Fresh-init comparisons across versions prove nothing.
- Timing rule: migrate before a training campaign, never mid-campaign. All campaign results on exactly one lgatr version.

## 1. Scope and execution environment

| Surface | Files |
|---|---|
| Slim building blocks (direct construction — **channel-last hit, M8**) | `experiments/baselines/lorentznetlgatrslimgraphgps.py`; `experiments/tagging/finetuneexperiment.py:115` (replaces `net.linear_out` with a raw slim `Linear`) |
| Slim net (hydra `_target_`; net-level tensor interface **unchanged**, verified) | `config/model/tag_slim.yaml`, `amp_slim.yaml`, `eg_slim.yaml`; `experiments/baselines/lorentznetlgatrslimgraphtrans.py`; `experiments/tagging/wrappers.py` (`LGATrSlimWrapper`) |
| Full LGATr net (hydra `_target_`) | `config/model/tag_lgatr.yaml`, `amp_lgatr.yaml`, `eg_lgatr.yaml`; `experiments/baselines/CGENNLGATrGraphTransHybrid.py` |
| LGATr layers (direct construction) | `experiments/baselines/cgennlgatrgraphgps.py` |
| Frames equivectors (via lloca) | `config/model/framesnet/equivectors/lgatr.yaml` |
| Interface + attention plumbing | `experiments/tagging/wrappers.py`, `experiments/misc.py`, `experiments/{eventgen,amplitudes}/wrappers.py` |

Execution split (Claude Code web containers are **CPU-only**):
- **CPU/web:** fixture record/check, manifests, transplant checks, the 64-test suite, all porting.
- **Cluster (OSCAR):** training-parity run (Gate G), throughput (Gate H), `.sif` rebuild — now standard PyPI installs; v2 also *dropped* the `einops`/`opt_einsum`/`numpy` requirements (torch-only), slightly simplifying the image.

## 2. Verified inventory (v2.0.0 release + CHANGELOG + source diffs)

### 2.1 Confirmed NON-breaks — do not "fix" these

- **Blade layout unchanged**: `embed_vector` still writes slots 1:5 (now via `F.pad`); hybrid's local `CliffordAlgebra` blade-order match survives (re-proven in Gate F, not assumed).
- **Symmetry class unchanged**: `PrimitivesConfig.subgroup=True` ≡ 1.4.4 `use_fully_connected_subgroup=True` (10-element SO⁺(1,3) basis both).
- **Net-level `LGATrSlim.forward` interface unchanged**: `(..., items, v_channels, 4)` in/out (verified against v2 source). The channel-last change is *internal/hidden-layer* — it surfaces only where this repo constructs blocks directly (M8).
- **lloca 1.3.6 survives**: `embed_vector`, `layers.EquiLayerNorm`, private `primitives.invariants._load_inner_product_factors(device=, dtype=)` all present and compatible.
- Top-level exports, `lgatr.layers.{EquiLinear, EquiLayerNorm, GradeDropout, GeoMLP}`, the `wrappers.py:1444` flex monkeypatch target (`flex.attention`), and `experiments/misc.py` backend kwargs (`attn_bias`/`cu_seqlens_*`/`block_mask`) all survive; new `varlen` backend added.
- Conditional-network renames (`condition_*` → `*_cond`): repo uses no conditional nets (grep-verified) — n/a.
- Repo passes no `compile_mode`/`compile_dynamic`/`num_hidden_layers`/`increase_hidden_channels` outside the sites listed in M3/M4 (grep-verified).
- **Bare `EquiLayerNorm()` constructions stay valid and parameter-free** (`cgennlgatrgraphgps.py` per-layer norms): the *layer* default is `elementwise_affine=False`; the changed default is the **net-level** `norm_elementwise_affine` kwarg (S2). v2's epsilon/gain became non-persistent fp32 *buffers* (the compile+DDP fix) — state_dict untouched, math identical at `gain=1.0`.
- **Block structures identical in both families** (pre-norm → attention → residual → pre-norm → MLP → residual; verified line-level, slim `SlimBlock` vs v1 `LGATrSlimBlock` and full `LGATrBlock` both). v2 splits the v1 *shared stateless* norm into `norm1`/`norm2` (necessary once gains exist) — zero state_dict/transplant impact while affine is off, since v1 norms were parameter-free. Slim attention still concatenates the vector+scalar streams into **one** joint sdpa call (the repo's `attn_dropout` semantics survive).
- **GeoMLP recipe identical**: GeometricBilinear first, then ScalarGatedNonlinearity + EquiLinear per extra layer, GradeDropout after each; the channel-list arithmetic independently re-confirms the M4 `+1` depth mapping. ~~`get_nonlinearity` unification: no effect at the defaults (gelu both)~~ **corrected by Gate C (S9)**: the unified helper's own docstring says `"gelu" uses approximate="tanh"` — v1's full-LGATr scalar path used exact erf-GeLU. Same *name*, different *function*.
- **qkv head-width formulas identical under the rename**: `max(mv_channels·ratio // num_heads, 1)` and `max(s_channels·ratio // num_heads, 4)` verbatim on both sides — M3 is a pure rename, no width drift.
- **CPU attention verified safe** — the sieve's biggest scare, resolved: v2 hard-gates `xformers`/`flash` off-CUDA and *raises* if their kwargs (`attn_bias`, `cu_seqlens_*`) are passed on CPU. But `experiments/misc.py` **already** materializes a dense `attn_mask` on CPU ("fallback to default attention") instead of backend-specific kwargs, so both 1.4.4 and v2 dispatch to native sdpa there — the CPU test suite passes on both for the same reason. The GPS hybrids always pass a dense `attn_mask` (+ optional `dropout_p`), i.e. native on every device, both versions. (New in v2, unused by the repo: explicit `backend="..."` selection kwarg.)
- **`LGATrSlim.__init__` positional order changed** (v1 `(in_v, out_v, hidden_v, in_s, out_s, hidden_s, num_blocks, …)` → v2 `(num_blocks, in_v, out_v, hidden_v, in_s, out_s, hidden_s, …)`) but every construction here is by keyword (hydra + the wrappers), so it is inert. Listed as a non-break rather than a checklist row because there is nothing to do — just keep slim constructions keyword-only, since a positional call would take this **silently**, with plausible-looking channel counts.
- **Spurion definitions byte-identical** (lightlike `[√2,0,0,1]`, spacelike, xyplane bivector, time reference; same kwarg names/defaults). `GradeDropout` semantics unchanged. `embed_scalar`/`extract_scalar` slots unchanged (component 0). `reinsert_mv/s_channels` machinery intact on the full net.

### 2.2 Mechanical breaks — port checklist

| # | Break | Exact sites | Fix |
|---|---|---|---|
| M1 | Slim blocks renamed with `Slim` prefix, moved to `lgatr.layers.slim_layers` (re-exported from `lgatr.layers` — **use the shallow path**; the deep path moved once already during dev) | `lorentznetlgatrslimgraphgps.py:50-54` (aliases become plain imports), `finetuneexperiment.py:6` | ~7 import lines |
| M2 | `LGATrSlim` module moved; top-level `from lgatr import LGATrSlim` unchanged | `_target_` in `tag/amp/eg_slim.yaml` → `lgatr.LGATrSlim` | 3 yaml lines |
| M3 | `SelfAttentionConfig.increase_hidden_channels` → `attn_ratio` | `cgennlgatrgraphgps.py:98-99` (+ second block ~:187), `CGENNLGATrGraphTransHybrid.py:1068`, `attention:` in `tag/amp/eg_lgatr.yaml` | rename |
| M4 | `MLPConfig.activation`→`nonlinearity`, `increase_hidden_channels`→`mlp_ratio`, `num_hidden_layers`→`num_layers_mlp` (**semantics confirmed**: v2 counts all layers, `mlp.py` builds `[in] + hidden×(n−1) + [out]`, so a v1 `num_hidden_layers` value ports as `+1`; repo passes neither → defaults `1`↔`2` are equivalent) | `cgennlgatrgraphgps.py:108-109`, `CGENNLGATrGraphTransHybrid.py:1073`, `mlp:` in `tag/amp/eg_lgatr.yaml` + `framesnet/equivectors/lgatr.yaml` | rename |
| M5 | `SlimRMSNorm` requires `(v_channels, s_channels)`; bare call breaks | `lorentznetlgatrslimgraphgps.py:91` | `SlimRMSNorm(v_channels, s_channels, elementwise_affine=False)` — affine must stay off here regardless of posture: the instance is shared across call sites, only valid stateless |
| M6 | s-channel types `int \| None` → `int = 0`. **Phase −1 corrected the direction**: run-verified, v1.4.4 *RAISES* on a literal `None` (`nn.Linear(None, …)` TypeError at construction) while v2 *tolerates* it (the `scalars=None` support). No live path ever passed `None` — `init_physics` fills real ints, and on v1 it had no choice — so this is a non-break either way | Gate A confirms; nothing to port | none (row kept to pin the observed asymmetry) |
| M7 | requirements | done on this branch: `lgatr[xformers-attention]==2.0.0` — **exact** for the whole migration (H14(3)); relaxed to a range only in Phase 5. (`main` holds the mirror-image guard, `>=1.4.2, <2.0.0`, so a stray `pip install -r` there cannot pull v2 mid-campaign.) | — |
| **M8** | **Slim blocks are channel-last**: `SlimLinear`/`SlimSelfAttention`/`SlimMLP`/`SlimRMSNorm` take `(..., 4, channels)`; this repo's GPS hybrid feeds `(B, P, V, 4)` | `lorentznetlgatrslimgraphgps.py` — every block call (`linear_in` :175, attention :81, mlp :86, norm/dropout :91-92) plus the spurion `cat` at dim 2; `finetuneexperiment.py:115` must match v2's *internal* layout at the `linear_out` splice point | First pass: `transpose(-1, -2)` at block boundaries (mechanical, provable). Optional later: flip the file's internal convention to channel-last as a perf follow-up (§8), separately gated |
| M9 | v1.4.4 slim `compile_dynamic=True` default replaced by `compile_kwargs` (dynamic **no longer defaulted**) | `tag_slim.yaml` (`compile: true`) | add `compile_kwargs: {dynamic: true}` to preserve behavior — variable-length flattened batches otherwise recompile per shape |
| **M10** | **`primitives: PrimitivesConfig` is a REQUIRED constructor arg** on directly-constructed layers: `SelfAttention(cfg, primitives)`, `GeoMLP(cfg, primitives)`, and `EquiLinear(in_mv, out_mv, primitives, ...)` — third **positional**, so it also shifts any positional args after it. (`geometric_product(x, y, *, config=...)` likewise; repo has no direct primitive callers.) | `cgennlgatrgraphgps.py` (SelfAttention/GeoMLP constructions ~:98/:108 and the second block ~:187, EquiLinear), `finetuneexperiment.py:108` (`EquiLinear` linear_out splice) | build one `PrimitivesConfig()` per model and thread it, mirroring v2's own nets; a missed site is a loud `TypeError`. **Also search for code that RECONSTRUCTS a layer from an existing one** (warm-start splices, quantization/pruning passes that walk `named_modules()` and swap children): those construct an `EquiLinear` without ever naming it in an import line, and the right argument there is `child.primitives` (v2 stores it, `layers/linear.py:79`) -- taking it from the layer being replaced is correct by construction |

All current v-channel widths are ≠ 4, so a missed M8 transpose **crashes loudly** (channel-dim mismatch). Keep it that way: never write fixtures or tests with `v_channels == 4`, the one width where a layout error becomes a silent transpose-alias (H13).

### 2.3 Silent numerics deltas (CHANGELOG-only; absent from the migration doc)

| # | Delta | Effect | Handling |
|---|---|---|---|
| S1 | Slim GLU gate nonlinearity: v2 `nonlinearity_v="sigmoid"` default (v1: one `nonlinearity`, gelu, for both gate paths) | Different activations in every slim model | Flag exists: `nonlinearity_v=None` restores v1 routing. Verification runs with the pin; shipped posture decided in Phase 3 |
| S2 | `norm_elementwise_affine=True` default **on every v2 net** (full LGATr, LGATrSlim, conditionals — "all networks" per CHANGELOG): slim RMSNorm gains per-channel `weight_v`/`weight_s`; full-LGATr EquiLayerNorm gains **per-grade** `(mv_channels, 5)` + per-channel scalar (grade-wise scaling stays Pin-equivariant). 1.4.4 norms are parameter-free | Param counts shift on every *net-using* surface: `tag/amp/eg_lgatr`, `tag/amp/eg_slim`, both GraphTrans hybrids' global stacks, the equivectors LGATr. Direct-layer constructions (GPS hybrid) are **unaffected** — the layer default is `False` (§2.1), and the M5 shared-norm site must stay `False` structurally | Flag exists on all nets: `norm_elementwise_affine: false` for parity. Gains init to 1.0 ⇒ identity at transplant time either way |
| S3 | `sparse_gp=True` default: geometric product via gather-reduce — reordered, **not bit-identical** (upstream's own docstring) | Full-LGATr models reproduce dense results only to tolerance | Keep `True` for runtime (the speed carrot); `primitives={'sparse_gp': False}` **only inside tier-1 verification** |
| **S5** | **qkv scalar bias removed** from q/k/v linears in *all* models ("because redundant") — v2 `qkv.py` passes `bias=False` at all four construction sites. v1 defaulted `bias=True` with **nonzero** uniform init: slim's `linear_in.linear_s.bias` (plain `nn.Linear` default; v2's zero-init of it is listed as *new*), full model's biases inside EquiLinear's internal `s2mvs`/`mvs2s` Linears (+ the standalone scalar-slot bias when a layer has no s-inputs) | v1 state_dicts contain qkv bias values with no v2 slot, and they contribute nonzero terms to v1 outputs | **Normalize-at-record**: walk `named_modules()`, locate each attention layer's qkv projection module (slim `linear_in`; full-model qkv EquiLinear), and zero **every** `*.bias` tensor beneath it, *before* recording (a zero-bias v1 model is still a valid v1 model — the reference moves into the intersection of both architectures). Gate B: waiver for the missing params |
| **S6** | **Slim GLU vector-gate scale**: v2 multiplies the gate inner product by `0.5 = 1/sqrt(4)` (`slim_layers.py:368-369`). **No flag.** | Every slim model's forward differs even with identical weights and S1 pinned | **Exact compensation in transplant**: scale the two vector-gate chunks (`v_gates_1`, `v_gates_2` rows of each GLU's fused linear `weight_v`) by `sqrt(2)` each ⇒ inner product ×2 ⇒ cancels the 0.5 exactly. Scalar-gate path is unscaled in v2 — do not touch it. Vector path has no bias — nothing else to compensate |
| S7 | Slim init/scaling refinements (`linear_s` bias→0 at init, plus "micro speed/memory optimizations") | Same-seed fresh-init models are not comparable across versions — at all | This is *why* verification is transplant-based. Init-distribution changes are an accepted training-dynamics delta of v2 (no eval-parity impact once weights are transplanted) |
| **S9** | **GeLU flavor changed: exact (erf) → `approximate="tanh"`** via the unified `get_nonlinearity`. Found by Gate C tier-1, localized by the per-block fixtures to `ScalarGatedNonlinearity`'s scalar stream (mv exact, s off by 1.3e-4 with zero-parameter module and bit-equal inputs); v1's `gated_gelu` used erf. Affects every gelu-using path (full-LGATr GeoMLPs, slim MLP gates) at ~1e-4 relative | Tier-1 parity would cap at ~1e-4 without compensation | Shipped: accept tanh-gelu (Posture B — it is upstream's speed choice). Verification: the transplant check monkeypatches the v2 build's gelu back to v1's exact split (instrument-only, like the S6 rescale) in BOTH tiers — keeping the 1e-10 bar meaningful and letting tier 2 isolate the S3 reorder alone |
| S8 | AMP strategy changed (vector/multivector path fp32, scalar path in autocast; `naive_amp` bypass added) | Inert today: every wrapper runs `use_amp: false` | Note in decision log; re-read this row before ever enabling amp |
| S10 | **Not a library delta — a verification-scope ruling** (operator, 2026-08-07). A multiplicity-1 jet degenerates the learned-frames path; its conditioning amplifies the fp64 cross-version reassociation baseline (measured, `equivectors_lgatr` edge fixture: the n=1 jet alone at 2.8e-9 tier-1 / 6.3e-6 tier-2; every n≥2 jet ≤ 4.6e-14; bit-deterministic across runs and seeds) | Only the `equivectors_lgatr` edge batch is affected — no campaign jet has multiplicity 1 (dataset floor ≥ 4), and the equivectors-lgatr composition is not used by this repo's campaign | Rule-derived handling in `_transplant_check`, firing only for a **learned-frames** composition on a batch actually **containing an n=1 jet**: the output is compared per-jet — n≥2 jets keep the **strict** bar; the n=1 jet is held to a tripwire (1e-6 tier 1 / 1e-4 tier 2, 2–4 orders below mistake scale O(1e-2..1) so frames-path breakage stays loud). Per-block activations of that batch are diagnostic-only (the degenerate jet's particle rows thread every block tensor — measured peak 6.9e-4 inside the frames net's block 0 at tier 2 — and masking jets inside arbitrary block shapes is H13 territory); full per-block strictness is retained on the main batch. Identity-frames fixtures untouched; the bars themselves unchanged |

(S4 from rev 1 — MLP depth semantics — is resolved and folded into M4.)

### 2.4 The two postures — DECIDED: Posture B (v2-native)

Because S5 + S6 are baked in, **"identical to the v1 campaign model" is not on the menu.** The coherent choices:

- **Posture A — closest-to-v1**: pin S1 (`nonlinearity_v: null`) and S2 (`norm_elementwise_affine: false`). Minimizes gratuitous deltas; params tables keep their v1 meaning. Still differs from v1 by S5/S6.
- **Posture B — v2-native** (recommended for a fresh campaign): accept upstream defaults (sigmoid gate "more stable", affine norms) as the new baseline. Since full v1 equivalence is impossible anyway, taking v2 as-shipped is the more reproducible citation ("lgatr 2.0.0 defaults"); re-baseline the param manifests once, in the same commit.

Either way: **verification always runs in parity mode** (pins + compensations) first — it certifies the port; the posture flip afterwards is a one-commit, documented model change, not a migration step.

**Decision (2026-07-30, maintainer + review): Posture B — adopt v2 defaults wholesale for every
row.** Rationale: the campaign has not been trained yet, so there are no v1 numbers to protect;
full v1 equivalence is impossible anyway (S5/S6 baked in); one lgatr version + one settings-set
per table (H5); "lgatr 2.0.0 defaults" is the cleanest citation. Obligations this creates:
a methods sentence that the `tag_lgatr`/`tag_slim` reference rows are re-trained under 2.0
(published-paper comparisons indicative, not exact), and a one-time re-baseline of the param
manifests in the posture-flip commit. Parity-mode verification (pins + compensations) still
runs first, unchanged — the posture applies from the first campaign run onward.

### 2.5 The parameter inventory, diffed (evidence for S2 and S5)

Same slim config built on both versions (`hidden_v=8, hidden_s=16, num_blocks=1, num_heads=4`).
Only three lines differ, and they are S2 and S5 exactly:

| | tensor | shape |
|---|---|---|
| **added by v2** (S2) | `blocks.N.norm1.weight_v`, `norm1.weight_s`, `norm2.weight_v`, `norm2.weight_s` | `(v_channels,)`, `(s_channels,)` — four per block |
| **removed by v2** (S5) | `blocks.N.attention.linear_in.linear_s.bias` | `(3 · attn_ratio · s_channels,)` — one per block |

Everything else matches name-for-name and shape-for-shape.

**Read the S5 row carefully before Phase 0 record.** The tensor v2 drops is the one under
**`blocks.*.attention.`** — the qkv projection. The net *also* has a top-level `linear_in`
whose `linear_s.bias` **survives in v2** and must NOT be zeroed. The §2.3 procedure is already
scoped correctly ("locate each attention layer's qkv projection module"), but the earlier
shorthand "slim's `linear_in.linear_s`" reads as if it named the net-level module. Zeroing that
one would silently move the reference model *outside* the intersection of the two architectures
and quietly weaken Gate C. Use the fully-qualified prefix `blocks.*.attention.linear_in.` in the
record script, and assert the count of zeroed tensors equals `num_blocks`.

The `norm` → `norm1`/`norm2` split (§2.1) is visible here too, and is free while affine is off:
v1's shared norm held no parameters, so KEY_MAP has nothing to map for it.

### 2.6 One data point from a sibling fork

`heidelberg-hepml/tagger-quantization@bead965` ports a same-lineage repo to 2.0 (its
`tag_slim.yaml` is byte-identical to ours but for the `_target_`). It changes M1, M2, M7 and
nothing else — a renames-only port, so S1, S2, S5, S6 and M9 all land on it unflagged. That is
§3's thesis observed rather than argued, and it is the whole reason to mention it.

The actionable part, and it is two sites not one. `inputquant.py` walks the model swapping each
`EquiLinear` for a `QuantEquiLinear(QuantLayer, EquiLinear)` and forwards `**kwargs` to
`EquiLinear.__init__` with no `primitives` — so their *main* quantization path for any full-LGATr
model raises on 2.0.0, not just a warm-start branch. Their `parq.py` grouping, by contrast,
survives: it selects by module attribute (`net.linear_in`, `block.attention`, `block.mlp`), and
every one of those attribute names is unchanged in v2. And their `finetuneexperiment.py`
`LGATrWrapper` branch still calls
`EquiLinear(in_mv_channels=…, out_mv_channels=…, in_s_channels=…, out_s_channels=…)`, which on
2.0.0 raises `TypeError: ... missing 1 required positional argument: 'primitives'` (verified by
running it). Their `SlimLinear` splice right below is fine, because that signature is unchanged.
**We have both sites**: `finetuneexperiment.py:108` is an M10 site and needs `primitives`;
`:115` is M1 only, a pure rename. The trap is inferring the first from the second.

## 3. Why neither the test suite nor the official doc is enough

- The 64-test suite is the regression floor, not a parity proof: a gelu→sigmoid gate swap, a 0.5 gate rescale, or dropped qkv biases all produce different-but-perfectly-equivariant models. Wiring tests check channels, not values.
- The official `v1_to_v2.rst` covers **renames and config moves only** (verified against its source). S1–S8 live exclusively in the CHANGELOG. Port by both documents; trust neither alone.
- A silently-dropped stale key would *narrow* a model with no error if config casting ever ignored unknown fields; Phase 1a proves `SelfAttentionConfig(increase_hidden_channels=2)` raises `TypeError`, Gate B backstops regardless.
- Both lgatr versions can never coexist in one environment (same package name) — recorded fixtures are the only bridge across the swap, which is why Phase 0 is unskippable and first.

## 4. The workflow

### Phase −1 — side-load the wheel and re-verify §2 (BEFORE Phase 0, no environment change)

**Read `v1_to_v2.rst` and the CHANGELOG first — then confirm against the wheel.** The docs are
accurate and they point you at the right rows; they are simply not complete (renames only,
§2.3) and not empirical. The wheel can be read while 1.4.4 is still the installed package, so
every claim in §2 is checkable *before* anything becomes irreversible:

```bash
pip download lgatr==2.0.0 --no-deps -d /tmp/lgatr2 && unzip -q /tmp/lgatr2/lgatr-2.0.0-*.whl -d /tmp/lgatr2x
python -c "import sys; sys.path.insert(0, '/tmp/lgatr2x'); import lgatr; print(lgatr.__version__)"
```

`sys.path.insert` makes v2 win over the installed 1.4.4 **inside that one process only** — nothing
on disk changes, the venv is untouched, and Phase 0 can still record on 1.4.4 afterwards. Run the
whole Phase 1a checklist here, and additionally:

- **Diff the parameter inventory**, not just the signatures: build the same reduced config on both
  versions and compare `named_parameters()` name-and-shape. This is what produced §2.5 and it is
  the only step that finds an S-row you did not know to look for — a signature diff cannot see a
  norm gaining gains or a qkv losing a bias.
- **Instantiate every construction this repo actually performs**, verbatim from the call sites
  (not simplified), and record which raise. Rev 4 found M7 wrong and M11 missing this way.

Phase 1a then repeats the checklist against the *installed* release, which is cheap once written
as a script and guards against "the wheel I read is not the wheel pip resolved."

### Phase 0 — capture on 1.4.4 (BEFORE any install or edit)

`tests/experiments/test_lgatr_migration_parity.py` (sketch: Appendix B), two modes (`LGATR_PARITY=record` / default check, which **skips cleanly when fixtures are absent**). Two fixture families, split to keep git small:

1. **Production manifests** (KB-scale json, all six lgatr-touching tagging configs — `tag_lgatr`, `tag_slim`, `tag_CGENNLGATrGraphTrans`, `tag_CGENNLGATrGraphGPS`, `tag_LorentzNetLGATrSlimGraphTrans`, `tag_LorentzNetLGATrSlimGraphGPS` — plus one learned-frames composition with `equivectors=lgatr`): total param count + sorted multiset of parameter shapes. Keys are *not* compared (renames make them legitimately differ); shapes and counts must not.
2. **Reduced-config transplant fixtures** (MB-scale, committed): the same model families at reduced size (`num_blocks=2`, hidden widths halved via overrides — every layer type, rename, and compensation is still exercised; parity logic is width-independent, and full-width state_dicts would be tens of MB of git). For each: fixed seed → instantiate (hydra compose + `init_physics`, same machinery as `test_jc_wiring.py`) → `.double().eval()` → **zero every bias under each qkv projection module (S5 normalization, per the S-table procedure)** → save, per model: full state_dict; final forward outputs; **per-block intermediate activations** (forward hooks at block boundaries); **a gradient pack** (input-gradient of `loss = out.square().sum()` plus the per-parameter grad-norm vector); and the **resolved-config snapshot** (`OmegaConf.to_yaml` of the composed config) — all on a fixed seeded batch (B=4, multiplicities `[1, n<knn_k, mid, large]`). Run the record **orchestrator twice** and assert `content_hashes.json` is byte-identical (determinism precondition, §4 — content hashes, not file bytes: `torch.save`'s zip layout is process-dependent).

The recorded state_dicts double as the v1 side of the **KEY_MAP**: do not hand-write the v1→v2 key mapping. Phase 1a dumps the v2 key lists for the same reduced configs; build the map by ordered shape-matching plus the known rename rules (module moves, `norm`→`norm1`/`norm2` adds nothing while norms are parameter-free), review it once, commit it next to the fixtures.

Scope note: the transplant exists to *verify the port*, nothing else — migrating trained checkpoints across lgatr versions is a non-goal (H7), so KEY_MAP never needs to handle production-width models.

Commit script + fixtures to this branch.

### Phase 1 — environment swap + Phase 1a re-verification

(Phase −1 has already answered every §2 question; Phase 1a re-runs the same script against the
installed release, so a resolution surprise cannot slip through.)

1. Fresh session/venv: `pip install "lloca[xformers-attention]==1.3.6" "lgatr[xformers-attention]==2.0.0"`.
2. **Phase 1a (~15 min, non-negotiable):** re-verify §2 against the installed release — read `v1_to_v2.rst` **and** CHANGELOG `[2.0.0]`; run the import one-liners (`from lgatr.layers import SlimMLP, ...`; top-level symbols); confirm `SelfAttentionConfig(increase_hidden_channels=2)` raises `TypeError`; confirm M10 (`SelfAttention(cfg)` / `GeoMLP(cfg)` / `EquiLinear(i, o)` without `primitives` raise `TypeError`); confirm `import lloca.equivectors.lgatr` works; confirm `embed_vector` slots 1:4; diff `SlimMLP`/`SlimSelfAttention`/`SlimLinear` signatures against `lorentznetlgatrslimgraphgps.py` call sites; confirm block channel-last docstrings and the net-level channel-first interface; run `torch.autograd.gradcheck` on `geometric_product` with `sparse_gp=True` in fp64 (custom-backward assurance, §4); dump v2 state_dict key lists for the reduced fixture configs and build+commit KEY_MAP (Phase 0 note).

### Phase 2 — mechanical port

M1–M10 as a literal checklist, one commit per row (or code/yaml pairs) for instant bisection. M8 first pass = boundary transposes only.

### Phase 3 — parity pins (verification-only), then the posture-flip commit

The posture is already decided (§2.4: **B**, v2-native), so the S1/S2 parity pins go **only into the parity script's model-building overrides** — shipped yamls take v2 defaults untouched, and the config-snapshot comparator therefore expects *only* the M-row renames in shipped configs.

After Gates A–F pass, one **posture-flip commit** (first thing in Task C, before any training run) does three things: (1) re-records the production manifests at v2 defaults (affine gains included) as the new baseline; (2) applies the **H15 optimizer exemption** — EquiLayerNorm/SlimRMSNorm gain parameters into the `weight_decay=0` groups in BOTH grouping paths, plus the Gate-B assertion that every `*.weight_mv`/`*.weight_s` sits in a no-decay group; (3) adds the §2.4 methods-sentence TODO to `todo.md`.

### Phase 4 — gates (A–F on CPU, G–H on cluster)

| Gate | What | Pass criterion |
|---|---|---|
| A | Composition: `test_jc_wiring.py` + parity script instantiates all fixture configs on v2 | no exceptions; channel asserts hold |
| B | Production manifests vs fixtures — each manifest records `(shape, requires_grad)` per parameter, not shapes alone | identical with waivers **derived by rule, never typed by hand**: the expected-missing set is computed from the v1 keys themselves (every bias under a qkv projection module → S5), and the expected `requires_grad` flips from v2's dead-tail/zero-size freezing. Set equality asserted in **both** directions — an unexpectedly missing, present, or re-frozen parameter fails. With the Phase 3 pins there are no other legal diffs during verification; the Posture-B re-baseline is its own later, equally rule-checked step |
| C | **Transplant parity** on reduced configs: map v1 state_dict keys → v2 (KEY_MAP), apply the S6 `sqrt(2)` gate-chunk rescale, load (`strict=False` only for rule-waived keys), then compare against the recorded batch: final outputs, **per-block intermediate activations**, and **gradients** (input-gradient of a fixed scalar loss + the per-parameter grad-norm vector) | **Tier 1** (S1 pin; `sparse_gp=False` for full-LGATr models — slim has no geometric product): relative deviation < 1e-10 in fp64 (bit-exactness is deliberately not claimed — v2 dropped `opt_einsum` and reordered contractions). **Tier 2** (`sparse_gp=True`): < 1e-8 forward/activations, < 1e-6 for gradients through sparse_gp's custom backward. First diverging *block* is reported, not just pass/fail |
| D | Full existing suite | 64/64 |
| E | Identity-frames bit-exactness spot check | hybrid with identity frames ≡ plain backbone, bit-identical on v2 (internal-consistency proof) |
| F | Blade-table equivalence | audit script vs v2 `lgatr.primitives.bilinear._load_geometric_product_tensor`: agreement ≤ 2e-6 fp32, as on 1.4.4 |
| G | Training sanity (cluster) | fixed-seed 1k-iter quick runs (`tag_slim`, `tag_lgatr`) on both versions: final train loss within seed-noise band (2–3 seeds). Point-wise curve equality is **out of scope for this gate, by design** (S3/S5/S6/S7 make it unattainable) -- the gate catches gross regressions only, and says nothing about whether curve-level study belongs in `dev` later |
| H | Throughput report (cluster) | not pass/fail: it/s for `tag_lgatr` v2 `compile=True/False` vs 1.4.4, and `tag_slim` (already compiled on 1.4.4, so expect little). Quantifies the carrot; publish the number in the decision log either way |

### Why these gates discriminate, and their blind spots (closed)

Both param counts and forward outputs *legitimately* differ across versions, so the gates are built to separate "expected delta" from "mistake" **structurally**, not by eyeballing tolerances:

- **Separation argument (Gate C).** Every neutralization is *exact*: S5 is removed from the reference at record time, S6 cancels algebraically (√2·√2·0.5 = 1), S1/S2 are pinned, tier 1 forces the dense product. A perfect port therefore lands at fp64 reassociation level (≲1e-12 relative), while every plausible mistake — one missed M8 transpose, one un-rescaled GLU of twelve, a crossed KEY_MAP pair, an un-zeroed bias at record, a wrong ratio — perturbs pre-activations at O(1e-2)–O(1). That is **6+ orders of magnitude of margin**; the exact epsilon is not load-bearing, which is what makes the gate trustworthy rather than tuned-to-pass.
- **Waivers are rules, not lists (Gate B).** The "params may differ" problem is closed by *deriving* the allowed diff from the v1 state_dict plus the S-items, and asserting set equality both ways. There is no free-text allowance a tired porter can widen.
- **Blind spot 1 — the backward pass.** Forward parity cannot see a broken custom backward (v2's sparse_gp ships one that saves only its inputs) or wrongly frozen parameters (`_freeze_dead_tail`, zero-size gain freezing): a model can match forward to 1e-12 and still train wrong. Closed three ways: gradients are part of the fixtures (Gate C), `requires_grad` is part of the manifests (Gate B), and Phase 1a runs `torch.autograd.gradcheck` on `geometric_product` with `sparse_gp=True` in fp64.
- **Blind spot 2 — train-only config values.** Eval-mode fixtures are blind to `dropout`/`attn_dropout`/schedule slips introduced while renaming yaml keys (a stray `attn_dropout: 0.5` changes no parameter and no eval output). Closed by **resolved-config snapshots**: Phase 0 records `OmegaConf.to_yaml` of every composed fixture config; post-port the snapshots are diffed and the only allowed changes are exactly the renamed keys (rule-checked like Gate B).
- **Blind spot 3 — localization.** A final-output mismatch alone says "somewhere in 24 sublayers". Per-block activation fixtures turn any Gate C failure into "first divergence at block k, sublayer s".
- **Determinism precondition.** The record orchestrator runs twice; the canonical **content hashes** must be identical before anything is compared across versions — otherwise the comparison would chase RNG, not the port. Two measured facts from Phase 0 shape this (details in the parity script's docstring): (a) executing FlexAttention perturbs global torch state so every *subsequent* forward in the same process shifts by ~1 ULP — hence one pristine subprocess per model and a 1e-12 (not bit-level) in-pytest self-consistency bar; (b) `torch.save` file bytes are process-dependent at identical content (zip storage-key ordering) — hence content hashes, with `test_fixture_content_hashes` re-deriving them on every check run.

### Phase 5 — cleanup, decision log, rollback

- Sweep stale comments referencing 1.4.4 internals (`tag_LorentzNetLGATrSlimGraphGPS.yaml` "lgatr lib default is 2" mlp_ratio note; the attn_dropout comments describing lgatr's sdpa path — re-verify wording against v2).
- Append the **decision log** here: installed version, gate numbers, posture chosen, one entry per S-item.
- Upstream issue: public accessor for `_load_inner_product_factors` (lloca coupling).
- **Rollback:** revert port commits, restore `>=1.4.2, <2.0.0`, reinstall 1.4.4. Fixtures stay valid. Trigger: any unexplained Gate B/C failure standing for more than a day.

## 5. Hitches and catches

- **H1 — verify against the release, not this table.** During dev, `slim_layers.py` changed directories *within one day*; the release freezes that churn, but Phase 1a re-verification stays mandatory. Prefer shallow imports (`lgatr.layers`, top-level `lgatr`).
- **H2 — the dangerous deltas are silent** (S1–S8): all 64 tests pass through every one. Only Gates B/C see them — which is why fixtures precede everything.
- **H3 — shared `SlimRMSNorm`** (`lorentznetlgatrslimgraphgps.py:91`): with v2's affine default it would crash (missing args) or silently tie gains across call sites. `elementwise_affine=False` is structural.
- **H4 — stale-key silence**: proven loud in Phase 1a; Gate B backstops.
- **H5 — sparse_gp reorders sums**: tier-2 tolerance; never mix lgatr versions inside one results table; identity-frames ≡ plain claims stay valid (both sides share one lgatr).
- **H6 — lloca's private-API import**: fine at `1.3.6` + `2.0.0`; any bump of either re-triggers the check.
- **H7 — checkpoints**: state_dict keys and (S2/S5) shapes change; migrate before the campaign so no checkpoint survives the boundary. Cross-version checkpoint porting is explicitly a **non-goal** — the transplant machinery is a verification instrument, never a model-delivery path.
- **H8 — literal `None` s-channels**: direction corrected by Phase −1 (run-verified): v1 *raised* on a literal `None`, v2 tolerates it. Yaml `null`s are placeholders filled by `init_physics`; Gate A confirms no live `None` path — non-break.
- **H9 — CPU-only web containers**: every parity gate is fp64-CPU by design; the container's xformers is ABI-mismatched (`--no-deps`), so gates must not touch CUDA kernels. The attention path is verified safe (§2.1): on CPU, `misc.py` materializes a dense `attn_mask` instead of passing xformers/flash kwargs, so both versions dispatch native sdpa and v2's off-CUDA backend gate is never hit.
- **H10 — installs are now standard PyPI** (2.0.0 released); v2 even drops `einops`/`opt_einsum`/`numpy` deps. The `.sif` rebuild is routine — but rebuild it *once, before* the campaign, not between runs.
- **H11 — compile expectations**: slim already compiled on 1.4.4 (keep `dynamic: true` via M9); the genuinely new capability is compile for **full-LGATr** + `warmup_caches` + compiled-xformers custom ops (attention no longer graph-breaks). Enabling it is a *post-migration* enhancement gated by Gate H numbers. For compile+DDP, note v2's own fixes here (unused-param `requires_grad_(False)`, tensor-ized norm eps/gains) — mirror that pattern in any local module you compile under DDP.
- **H12 — the official migration doc is renames-only** (verified). The CHANGELOG is the behavioral source of truth. Port by both.
- **H13 — `v_channels == 4` is the silent-alias width** for the M8 layout flip (transpose becomes shape-legal). All current widths differ from 4; keep fixtures and tests that way so layout mistakes stay loud.
- **H14 — the gates' residual blind spots, stated so nobody re-derives them.** Three holes the
  Phase-4 scheme deliberately or accidentally leaves open, with their closures:
  (1) **The amp path is ungated.** Every gate runs fp64/fp32 *eager*, and S8 changed exactly the
  autocast machinery (`minimum_autocast_precision` now returns tuples + downcasts;
  vector/multivector path pinned fp32, scalars amp'd; new `naive_amp` bypass) — a regression
  living only under autocast is invisible to A–H, since even Gate G's quick runs inherit
  `use_amp: false`. Acceptable *because* every campaign config runs amp-off; the closure is a
  tripwire, not a fixture: if any config ever flips `use_amp: true`, add one loose-tolerance
  amp smoke fixture FIRST.
  (2) **Per-block activation fixtures cross the M8 layout boundary.** v1 records channel-first
  block activations; v2 slim blocks carry vectors channel-last internally — the fixture
  comparator must transpose at block boundaries, and H13's `v_channels==4` silent-alias warning
  applies *inside the comparator* too (a fixture width of 4 would make a missed transpose
  compare clean).
  (3) **`>=2.0.0` is not a verification target — pin `==2.0.0`.** With a floor pin, a 2.0.x
  patch release landing mid-migration silently moves the code under the recorded fixtures and
  Phase-1a source re-verification (requirements.txt now pins exact); relax to `>=` only in the
  Phase-5 cleanup commit, after the gates have passed against a named version.
- **H15 — Posture B's affine norm gains escape BOTH weight-decay exemptions (campaign-affecting).**
  v2's `norm_elementwise_affine=True` (the Posture-B default for every net) registers
  `weight_mv = Parameter(ones(mv_channels, 5))` — **2-d**, so it fails the base grouping's
  `param.ndim <= 1` rule (`base_experiment._init_optimizer`) AND the ParT-path rule
  (`len(shape)==1 or endswith(".bias") or in no_weight_decay()`). Every lgatr-touching row
  would silently weight-decay its norm gains toward zero — worst on `top_lgatr` (Lion,
  wd=0.2). Same disease as the CGENN `MVLayerNorm` `(1, C)` gain, one library up. Port fix:
  exempt EquiLayerNorm parameters explicitly in *both* grouping paths (name rule
  `endswith(".weight_mv")`, or isinstance-collection over `EquiLayerNorm` modules — the
  robust form), and extend Gate B to assert every `*.weight_mv` / `*.weight_s` parameter
  sits in a `weight_decay=0` group. (`weight_s` is 1-d and already exempt; asserting both
  keeps the rule self-documenting.)
- **H16 — do NOT block on the tagging-training-environment release.** The migration's *correctness* is certified by the gates, which depend only on the lgatr 2.0.0 source — not on seeing upstream's worked examples. What IS provisional until that repo lands: API-*style* choices (how they configure `compile_kwargs`, backend selection, spurion handling in production configs). Mark those as provisional in the port and diff against the new repo's usage in a cheap Phase-5 addendum when it releases; expect renames, not rework. *(Restored: this bullet was added as a second "H14" and then silently clobbered by the H14 rewrite in c6d3a9b.)*

## 6. Upstream (`heidelberg-hepml/lloca-experiments`) variant

No hybrids, no direct block construction ⇒ no M8 surface except their `finetuneexperiment` equivalent. Surface = M2/M3/M4/M9 renames + M10 at their `finetuneexperiment` `EquiLinear`/slim-`Linear` splices + flex monkeypatch + their suite as Gate D, same record→port→prove shape (S2/S5/S6 still apply to their nets — transplant needs the same compensations). ≈1 day. Offer the fixture-script pattern with the port PR.

## 7. Task split and operator protocol (Claude Code web)

> **Execution is driven from `docs/execution-playbook.md`** — it holds the canonical copies of these prompts *plus the exact check commands*, sequenced with the CGENN/LorentzNet/non-equivariant compile steps. Paste prompts from there. If this section and the playbook diverge, that is a bug: sync them in their own commit before running anything.

Three sessions. The gates catch math mistakes; **the operator's job between sessions is to catch what no gate can self-police: gate tampering and scope creep.** Each task below has a copy-pasteable prompt and the operator checklist that gates the *next* task.

### Task A — fixtures on 1.4.4

```text
On branch dev, execute Phase 0 of docs/lgatr2-migration.md exactly.
Precondition: `python -c "import lgatr; print(lgatr.__version__)"` must print 1.4.4. If the
session auto-installed dev's requirements (lgatr==2.0.0), first run
`pip install "lgatr[xformers-attention]==1.4.4" "lloca[xformers-attention]==1.3.6"` and re-verify.
Deliverables, committed to dev:
1. tests/experiments/test_lgatr_migration_parity.py per Appendix B + Phase 0: record/check
   modes; record hard-asserts lgatr==1.4.4; check skips cleanly when fixtures are absent.
2. tests/fixtures/lgatr144/: production manifests (shape + requires_grad) for the six
   lgatr-touching tagging configs + the equivectors composition; reduced-config transplant
   packs (qkv-bias-normalized state_dict, outputs, per-block activations, gradient pack,
   resolved-config snapshot).
3. Record orchestrator run TWICE; show the identical content_hashes.json comparison in your report.
4. Existing test suite still 64/64 with the new file present.
Constraints: do NOT install lgatr 2.0. Do NOT modify anything under experiments/ or config/ —
Phase 0 touches tests/ only. If a fixture cannot be recorded, stop and report; never shrink
the model list to make recording pass.
```

**Operator gate A→B** (~5 min): (1) `git show --stat` on the Task-A commits — only `tests/` paths; any `experiments/` or `config/` change = reject. (2) `du -sh tests/fixtures/lgatr144` is MBs, not tens of MBs. (3) Open the parity test once and eyeball exactly four things: the 1.4.4 hard-assert in record mode; the bars (1e-10 / 1e-8 / 1e-6); the waiver set is *derived by rule*, not a hard-coded key list; the double-record byte-identity assertion exists. (4) CI green, new test skipping cleanly.

### Task B — port + Gates A–F on 2.0.0

```text
On branch dev, execute Phases 1-3 and Gates A-F of docs/lgatr2-migration.md.
Environment: pip install -r requirements.txt (lgatr==2.0.0 exact) and lloca==1.3.6; paste
`pip freeze | grep -Ei "lgatr|lloca"` into your report.
Order:
1. Phase 1a re-verification, in full; paste its results (including the sparse_gp gradcheck and
   the v2 state_dict key dumps) BEFORE editing any file. If any item contradicts §2, STOP and
   report — do not improvise a fix.
2. Phase 2: one commit per M-row, M1-M10 in order.
3. Phase 3: S1/S2 pins go into the parity script's build overrides ONLY; shipped configs stay
   at v2 defaults (Posture B is decided, §2.4).
4. Build KEY_MAP from the recorded v1 keys + your Phase-1a v2 dumps; commit it with a short
   note of the rename rules used.
5. Run Gates A-F; paste every gate's NUMBERS (not just pass/fail) into a "Gate results"
   subsection of this runbook's decision log.
Hard constraints: never loosen a tolerance, widen a waiver rule, delete an assertion, or drop
a model to make a gate pass — bars and waivers change only via a new documented S-item, which
requires stopping and reporting first. Stop at the FIRST gate failure and report the
first-divergence block/tensor. Do not run Gates G/H, do not touch the .sif, do not open a PR.
```

**Operator gate B→C** (~15 min, the critical one):
1. `git diff <taskA-tip>..HEAD -- tests/experiments/test_lgatr_migration_parity.py` — acceptable changes: KEY_MAP content, rule-derived waiver additions tied to S-items, v2-only imports. **Any edit to a tolerance, waiver derivation, or assertion = reject the task, not the gate.**
2. `git diff <taskA-tip>..HEAD -- config/` — exactly the M-row renames (M2/M3/M4/M9), zero value-level changes (the snapshot gate enforces this; you are checking the enforcement wasn't edited).
3. Gate-results numbers: tier-1 deviations should sit well under the bar (~1e-12-ish). A tier-1 pass at 3e-11 is unexplained drift — treat as a failure to investigate, not a pass.
4. Phase-1a report exists and either matches §2 or documents what changed upstream.
5. `pip freeze` line shows `lgatr==2.0.0`, `lloca==1.3.6`.

### Task C — posture flip, cluster gates, close-out

```text
On branch dev (Gates A-F green, operator-reviewed):
1. FIRST, before any training: the posture-flip commit (Phase 3): re-record production
   manifests at v2 defaults; apply the H15 optimizer exemption in BOTH grouping paths with the
   Gate-B no-decay assertion for *.weight_mv/*.weight_s; add the §2.4 methods-sentence TODO.
2. Gate G: fixed-seed ~1k-iter quick runs (tag_slim, tag_lgatr) on 1.4.4 and 2.0.0, 2-3 seeds
   each; report final-loss bands side by side.
3. Gate H: it/s table — tag_lgatr compile on/off and tag_slim, both versions; publish the
   numbers in the decision log whatever they say.
4. Phase 5: stale-comment sweep, relax the pin to >=2.0.0,<3, complete the decision log
   (one entry per S-item).
5. Open the PR dev -> main only after 1-4 are in the log.
```

**Operator gate C→merge**: Gate-G loss bands overlap by *your* physics judgment; Gate-H table present even if unflattering; decision log complete; the PR diff is the sum of diffs you already reviewed at A→B and B→C — nothing new may appear at PR time.

### Failure protocol

Any session that hits a gate failure **stops**. Operator triage: (a) explained by a documented S-item → add the waiver/S-row in its own reviewed commit, re-run the gate; (b) unexplained → spawn a *fresh* investigation session seeded only with the failure artifact (first-divergence block, tensor names, the two fixture values) — never let the porting session debug its own gates by editing them.

## 8. Out of scope for THIS TASK (the lgatr 2.0 migration), captured for the follow-up: performance transfers to CGENN

Throughout this runbook, "out of scope" always means **out of scope for the task the section
names** -- the lgatr 2.0 migration here, the CGENN compile work in `docs/cgenn-compile.md`.
It never means "excluded from `dev`" or "decided against"; deferred items keep their revisit
triggers below.

Not migration work — a separate task after gates pass; recorded here so the thinking isn't lost:

- **Profile first**: the FLOPs tests already emit per-jet FLOPs per model — compare CGENN-hybrid rows against `tag_slim` before optimizing anything. Post-migration, the compiled attention half of every CGENN-hybrid block gets faster, making the un-optimized CGENN branch the *relative* bottleneck almost by construction (~N·k dense-Cayley contractions per jet per layer).
- **Sparse-indexed GP transfers almost verbatim** (same Cl(1,3) 16-blade algebra; 256/4096 nonzero Cayley entries, one output blade per pair): rewrite the dense Cayley einsums as precomputed (indices, signs) gathers with an input-saving backward — upstream made it default "because always faster", eager included. **Corrected sharing map (import-verified):** the two GT/GPS hybrids share ONE stack — `cgennlgatrgraphgps.py` imports `CliffordAlgebra`/`CGLayer` from `CGENNLGATrGraphTransHybrid.py` — so one rewrite there serves *both hybrids*; the FC **baseline** uses the separate `experiments/baselines/cgenn/` package (`fcgp.py`/`gp.py`), which needs its own pass. Mathematically identical, reorder-only — a documented performance change, no modeling change.
- **FC baseline only**: the all-pairs padded graph admits a dense `(B, N, N, ·)` masked-mean reformulation — no scatter, fixed shapes, compiles cleanly, better locality even eager. **The kNN hybrids keep the scatter** — sparsity is the design there; `index_add_` compiles fine with `dynamic=True`.
- Compile knobs: `dynamic=True` over batch/N; `activation_memory_budget` (torch≥2.4) if N² intermediates pinch; AMP split (multivector fp32 / scalar bf16) after the parity dust settles.
- Fullgraph pattern: `fullgraph=False` at the model root always; flip `fullgraph=True` per-module only on proven break-free leaf blocks (`LorentzNetKNNBlock` now; the CGENN block once the sparse-GP rewrite lands) as a regression guard against reintroduced graph breaks.

### LorentzNet: torch.compile readiness (same follow-up task)

- **Standalone `tag_lorentznet` (LGEB stack) is essentially compile-ready as-is**: the forward is
  pure tensor ops — `x[i]-x[j]` gathers, `normsq4`/`dotsq4`, `psi = sign·log(1+|·|)`, small MLPs
  (BatchNorm1d included), and `index_add_`-based `unsorted_segment_{sum,mean}`. No numpy, no
  `.item()`, no data-dependent Python branching in the hot path. Changes needed are placement,
  not rewrites:
  1. `dynamic=True` — the per-jet fully-connected edge count `E = Σ nᵢ(nᵢ−1)` and node count vary
     per batch; without it every new shape recompiles.
  2. Compile the **net**, not the wrapper: keep `get_edge_index_from_ptr` (cheap, shape-producing,
     a graph-break magnet) outside the compiled region and compile the LGEB stack it feeds.
  3. The in-place `index_add_` on `new_zeros` functionalizes fine under Inductor; if an older
     torch complains, the out-of-place `index_add` is a one-line swap.
  4. Compile+DDP: LorentzNet has none of v2's problem shapes (no zero-size gains, no unused
     params) — nothing to mirror.
- **No sparse_gp analogue exists**: LorentzNet is vector-only (Minkowski dots/norms — no Cayley
  contraction), so there is no 16× arithmetic cut to harvest. Its compile win is pure kernel
  fusion over many tiny ops (hidden width 72, per-edge MLPs over the fully-connected
  E ≈ N(N−1) edges — the dominant cost). Tiny-op models are overhead-bound, so the *relative*
  fusion win may exceed CGENN's — but it is bounded by launch overhead, not FLOPs; measure
  (Gate-H style) before claiming anything.
- **The hybrid `LorentzNetKNNBlock` (GPS/Trans local branch) is the easiest local branch to
  compile**: dense padded `(B, P, K)` gather + Conv2d MLPs + masked sum/mean — static-shaped,
  no scatter at all. `dynamic=True` over B/P and it should compile without a single break.

### Non-equivariant family: out of scope FOR THIS TASK (not out of scope for `dev`)

To be unambiguous about what "out of scope" means here: it scopes the **lgatr 2.0 migration
and the CGENN compile work**, the task this runbook covers. It is not a judgement that the
non-equivariant models are excluded from `dev` generally, nor that they should never be
compiled — only that compiling them is not part of *this* change and does not gate it.

Plain / ParticleNet-ParT (both variants) and the plain-transformer baselines get no compile
work here, for three reasons: (1) no forcing event — the lgatr migration is what makes the
*equivariant* GNN branches the relative bottleneck (their attention halves speed up for free;
nothing analogous happens to the non-equivariant models); (2) smaller headroom — they are
built from already-fused standard kernels (`nn.MultiheadAttention`, cuDNN Conv1d/2d, BatchNorm),
not the many-tiny-op profile where Inductor fusion pays big; the PN-ParT dynamic per-layer kNN
is graph-*rebuilding* cost, which compile does not remove; (3) accuracy is unaffected either
way — compile only moves the table's train-time column. Which is the one obligation this
scoping creates: **if any model family trains compiled and another doesn't, say so wherever
walltime/efficiency numbers are compared** (uniform-or-disclosed). Two facts that defuse the
"unfair walltime" worry: the table has mixed compiled and eager rows since BEFORE the
migration — `tag_slim` ships `compile: true` on 1.4.4 while everything else runs eager, an
upstream precedent, not something the migration introduces — and the **FLOPs column is the
compile-independent efficiency measure** (compile changes kernel launch/fusion overhead, not
arithmetic), so efficiency *claims* lean on FLOPs while walltime stays informational with a
per-row compile footnote. **Update (2026-08):** weaver-core has since added torch.compile support for ParticleNet and
ParT upstream. That removes the *feasibility* doubt — someone has implemented and validated
it for those architectures — but not reason (2), the ROI argument, and their versions are
standalone backbones rather than ones wrapped in this repo's GraphTrans/GPS machinery
(per-batch kNN rebuild, PyG scatter, LLoCa frame transport), which is where the graph breaks
would come from. So the scoping stands for this task, with a sharper revisit trigger:
post-campaign, measure compile on ParticleNetParTGraphTrans/GPS, and **if the whole table
turns out to compile, prefer uniform compilation** to the current split -- it retires the
per-row disclosure above rather than managing it. Also revisit if a profiler shows the
non-equivariant rows dominated by launch overhead, which their kernel profile makes unlikely.

## Decision log

**Task B close-out (2026-08-07).** Environment: `lgatr==2.0.0`, `lloca==1.3.6` (exact; freeze
pasted in the session report). Posture shipped: **B, v2-native** (§2.4) — shipped configs carry
v2 defaults untouched; every pin named below exists only in the parity script's build
overrides (`PARITY_PINS` / `TIER1_SPARSE_GP_OFF`). At Task B close-out the following were
deliberately not started — see the Task C entries below for what has since landed: Gates
G/H (cluster), the posture-flip commit (manifest re-baseline at v2 defaults + the H15
optimizer exemption), the `>=2.0.0,<3` pin relaxation, the PR.

### Gate results (Task B, CPU, fp64 unless stated — numbers, not pass/fail)

Relative deviations are `max|Δ| / (1 + max|ref|)` against the 1.4.4 fixtures. Bars: tier 1
< 1e-10; tier 2 < 1e-8 forward/activations, < 1e-6 input-gradients. The §4 separation
argument expects clean ports at fp64-reassociation scale (~1e-12 or below) — every green
number below sits at **2e-13 or better**, so no near-bar pass needed a drift investigation.

**Gate A — composition.** `tests/internal/test_jc_wiring.py`: 8/8 on v2. All seven fixture
compositions instantiate on v2 exception-free with channel asserts holding (each was built
repeatedly: two Gate C tier builds plus the Gate B pinned build).

**Gate B — production manifests, two-sided rule-derived check, 7/7.** The 2.x branch of
`test_production_manifest` builds each full-size config with the Phase 3 pins and asserts:
removed-set == the S5 set derived from the *stored v1 names* via `QKV_BIAS_RE`; added-set
== ∅; `requires_grad` flips == the structural freezing rule (`_expected_frozen`: v2's
zero-size freeze + `_freeze_dead_tail` semantics recomputed from the live module tree —
never read back from the flags v2 set); `(shape)|requires_grad` multiset equality after
waiver subtraction; totals differing by exactly the waived numel.

| model | v1 params | v2 params (pinned) | waived qkv biases | rg flips (rule-matched) |
|---|---|---|---|---|
| tag_lgatr | 1 079 394 | 1 074 786 | 24 (4 608) | 0 |
| tag_slim | 1 794 631 | 1 791 175 | 12 (3 456) | 3 |
| tag_CGENNLGATrGraphTrans | 1 154 664 | 1 150 824 | 20 (3 840) | 0 |
| tag_CGENNLGATrGraphGPS | 1 898 197 | 1 894 357 | 20 (3 840) | 0 |
| tag_LorentzNetLGATrSlimGraphTrans | 1 636 032 | 1 633 152 | 10 (2 880) | 0 |
| tag_LorentzNetLGATrSlimGraphGPS | 2 179 493 | 2 176 613 | 10 (2 880) | 0 |
| equivectors_lgatr | 2 267 307 | 2 267 223 | 2 (84) | 0 |

`tag_slim`'s three flips are exactly v2's freezing rules firing on the only net with an empty
output stream: `net.linear_out.weight_v (0, 6)` (zero-size) plus
`net.blocks.11.mlp.layers.{0.linear,1}.weight_v` (`_freeze_dead_tail`, net
`out_v_channels=0`); all are v1 `True` → v2 `False`, no reverse flips anywhere.

**Gate C — transplant parity** (S6 √2 rescale + S5 waivers + S9 flavor patch; tier 1 adds the
S1/S2 pins everywhere and `sparse_gp=false`; per-block activations asserted at the same bars
as the forward). Forward `main / edge`; input-gradient rows only where the fixture carries one
(the four hybrid models — `tag_slim`/`tag_lgatr` record param-grad norms only, wrapper
in-place ops block input-leaf backward; `equivectors_lgatr` has no CPU flex backward):

| model | tier-1 fwd | tier-1 ∇x | tier-2 fwd | tier-2 ∇x |
|---|---|---|---|---|
| tag_lgatr | 4.889e-17 / 2.460e-17 | — | 1.955e-16 / 1.599e-16 | — |
| tag_slim | 4.879e-17 / 4.898e-17 | — | 4.879e-17 / 4.898e-17 | — |
| tag_CGENNLGATrGraphTrans | 1.239e-16 / 1.211e-16 | 3.975e-17 / 1.675e-17 | 3.098e-16 / 3.027e-16 | 3.312e-17 / 5.611e-17 |
| tag_CGENNLGATrGraphGPS | 0.0 / 3.972e-17 | 1.971e-16 / 4.481e-17 | 0.0 / 0.0 | 6.042e-16 / 6.016e-16 |
| tag_LorentzNetLGATrSlimGraphTrans | 1.381e-16 / 1.375e-16 | 1.946e-18 / 1.731e-18 | 1.381e-16 / 1.375e-16 | 1.946e-18 / 1.731e-18 |
| tag_LorentzNetLGATrSlimGraphGPS | 0.0 / 0.0 | 2.168e-19 / 2.710e-19 | 0.0 / 0.0 | 2.168e-19 / 2.710e-19 |
| equivectors_lgatr | 1.919e-13 / 1.529e-9 † | (no pack) | 9.863e-14 / 6.281e-6 † | (no pack) |

† the n=1-jet degenerate-frames case: originally **FAIL** against the strict bars, then ruled
out of scope by the operator (S10) after the characterization below. Post-ruling the edge
batch is compared per-jet (subset-normalized): n≥2 jets strict (2.455e-14 tier 1 / 4.463e-13
tier 2, PASS), the n=1 jet against its documented tripwire (1.736e-9 vs 1e-6; 7.132e-6 vs
1e-4, PASS); its per-block activations are diagnostic-only (peak 6.937e-4 inside the frames
net, see ruling). The whole-batch values shown above are the pre-ruling measurements,
unchanged.

Slim-only models (`tag_slim`, both LorentzNet hybrids) have no geometric product, so their
tier-2 build is the tier-1 build; the bit-identical numbers across tiers are the expected
consistency check, not a copy-paste. For the full-LGATr models the tier-1→tier-2 movement
(≤6e-16) is the S3 reorder itself — at these widths it is invisible, exactly as argued in §4.

**The `equivectors_lgatr` edge batch — characterized, then ruled out of scope (S10).**
Characterized first, decided by the operator after; the tier-2 edge failure is the same item,
not a second finding.
The edge batch truncates jets to multiplicities [1, 3, 30, 49]; per-jet decomposition of the
tier-1 deviation: the n=1 jet carries **2.848e-9** while the n=3 / n=30 / n=49 jets sit at
4.574e-14 / 1.188e-14 / 6.661e-16 — and every jet of the main batch (natural multiplicities)
is ≤ 2.623e-13. So: (a) the batch-level 1.529e-9 is driven entirely by the single-particle
jet, not spread over elements; (b) it does not merely shrink at n≥2, it collapses by 4+
orders to baseline; (c) it is bit-deterministic — three repeated runs and varied-seed builds
reproduce both batches to max|Δ| = 0.0, so this is a fixed arithmetic difference, not noise;
(d) flex-specificity is untestable on CPU: flex is the only importable sparse lloca backend
here (varlen is CUDA-only; xformers/flash are CUDA-gated by lloca's registry). Consistent
interpretation (not a ruling): a 1-particle jet degenerates the learned-frames path, whose
conditioning amplifies the ~1e-13 cross-version reassociation baseline into the 1e-9 (dense)
and 1e-6 (sparse_gp) range.

**Ruling (operator, 2026-08-07): out of scope.** Grounds: the equivectors-lgatr composition
is not used by this repo's campaign, and no campaign jet has multiplicity 1 (dataset floor
≥ 4) — the degenerate input exists only in the synthetic edge-batch truncation. Implemented
as S10 (§2.3), rule-derived (learned-frames composition AND an n=1 jet actually present),
not a bar change: the degenerate batch's output is compared per-jet — its n≥2 jets stay at
the **strict** bar (gate-measured, subset-normalized: 2.455e-14 tier 1 / 4.463e-13 tier 2)
while the n=1 jet is held to a 1e-6 / 1e-4 tripwire (measured 1.736e-9 / 7.132e-6; 2–4
orders of margin below mistake scale remain). That batch's per-block activations become diagnostic-only: implementing the
ruling exposed that the degenerate rows thread the intermediate tensors at up to **6.937e-4**
(tier 2, `framesnet.equivectors.net.blocks.0.mlp.layers.0` — the amplification peaks inside
the frames net and attenuates to 6.3e-6 by the output, consistent with the conditioning
mechanism), and masking jets inside arbitrary block shapes is H13 territory; the main batch
keeps full per-block strictness. With S10, Gate C is 7/7 green in both tiers.

**Gate D — full suite on v2: 611 passed, 17 failed, 39 skipped** (667 collected; the "64/64"
criterion line predates the suite's growth; counts are as-run during Task B, pre-S10 — after
the S10 ruling the transplant member is green by direct re-run, and the H15 member goes green
with the Task C flip commit, leaving exactly the 15 environment-class failures). The 17
decompose exactly into three known classes, none an unexplained migration break:

- **15 × pelican-FLOPs environment class**: every `*-learnedpd-pelican` parametrization of
  `test_tag_flops`/`test_amp_flops` plus `test_tagging[tag_pelican_fair-identity]`. The
  installed pelican package internally calls `torch.compile(fullgraph=True)`
  (`pelican/primitives.py:104`), which refuses under `FlopCounterMode`'s torch dispatch mode
  on this container's torch 2.13 ("non-infra torch dispatch mode present"). No lgatr frame in
  any traceback; no dependency path from the lgatr version to pelican internals; the
  `learnedpd-lgatr` equivectors FLOPs rows all PASS on v2. Same environment-only family as
  the container FLOPs failures already in `docs/audit-ledger.md`.
- **1 × `test_transplant_parity[equivectors_lgatr]`** — the flagged item above appearing
  under its default tier-1 invocation; it stayed red until the operator's S10 ruling and is
  green under it (per-jet: strict for n≥2, tripwire for the n=1 jet).
- **1 × `test_weight_decay_grouping[tag_CGENNLGATrGraphTrans-tag_CGENNLGATrGraphGPS]`** —
  **H15 firing, on schedule**: under Posture B the GraphTrans hybrid's LGATr nets now carry
  affine gains (net-level v2 default passed through) and those `weight_mv` gains land in the
  *decayed* group, while the GPS hybrid's bare per-layer `EquiLayerNorm()` (structurally
  affine-off, §2.1) has none — so the decayed-kinds sets differ. This is precisely the H15
  prediction; the fix (gains into `weight_decay=0` in BOTH grouping paths + the no-decay
  assertion) is the Task C posture-flip commit's step 2 and was deliberately not pulled
  forward into this session.

**Gate E — identity-frames bit-exactness.** The suite's spot checks
(`tests/internal/test_duplicated_component_parity.py`, run standalone on v2: 31/31) assert
`torch.equal` — max|Δ| = 0.0 by definition of passing — for hybrid EdgeConv ≡ ported
ParticleNet ≡ installed lloca, per-block and whole-model, under `Frames(is_identity=True)`.
For the lgatr-bearing wrappers the hybrid-at-identity ≡ plain property is structural rather
than testable: they `assert isinstance(framesnet, IdentityFrames)` and never touch frames
(`wrappers.py:505` "not actually used"), so no second code path exists to diverge; their
lgatr content is exactly what Gate C measures. H5's version-consistency requirement (both
sides of any ≡ claim share one lgatr) holds trivially: one lgatr is installed.

**Gate F — blade-table equivalence: max|diff| = 0.000e+00** (bar ≤ 2e-6 fp32; 256 nonzeros
on both sides), now permanent as `test_blade_table_gate_f`. The index conventions differ by
design and the alignment is forced, not fitted: the hybrids contract
`out_j = a_i · cayley[i,j,k] · b_k` (output on the middle axis) while v2 contracts
`out_i = gp[i,j,k] · x_j · y_k` (output first), so identical products ⟺
`cayley.permute(1,0,2) == gp` — which holds bit-exactly. §2.1's "blade layout unchanged" is
thereby re-proven, not assumed.

**Parity-script deltas made in this session** (for the B→C diff review): (1) the S9
compensation now spans BOTH tiers (`_transplant_check`; previously tier-1-only — see the S9
entry for why tier 2 is wrong without it); (2) the Gate B 2.x branch replaced its skip
(`_manifest_check_v2` + `_expected_frozen`, rule-derived as specified in §Phase 4); (3)
`test_blade_table_gate_f` added. No tolerance, waiver derivation, or existing assertion was
edited; the flagged case stayed red until the operator's S10 ruling landed in its own commit
(the only edit that touched a comparison bound, and it is the S-item-documented one).

### Per-S-item entries (what changed, which posture shipped, why)

- **S1 — slim vector-gate nonlinearity.** v1 slim gated vectors through identity routing; v2
  adds `nonlinearity_v` defaulting to `sigmoid` ("more stable"). Shipped: v2 default — no
  yaml sets the knob (Posture B: upstream defaults are the citation). Verification pinned
  `nonlinearity_v=null` (script-only) so Gate C compared like against like; the knob is
  threaded through the slim hybrids' wrappers so the pin (and any future ablation) composes.
- **S2 — norm affine gains.** v2 flips the *net-level* `norm_elementwise_affine` default to
  True (slim: per-channel `weight_v`/`weight_s`; full LGATr: per-grade `(mv, 5)` gains); the
  *layer-level* default stays False, so bare `EquiLayerNorm()` / shared `SlimRMSNorm`
  constructions (GPS hybrids, H3) remain parameter-free structurally. Shipped: v2 default
  wherever the net-level kwarg exists — the campaign trains with gains. Verification pinned
  affine off so manifests/transplants matched v1 minus S5. Consequences owned by Task C:
  manifest re-baseline (gains included) and the H15 decay exemption — Gate D's grouping
  failure above is this obligation surfacing, on schedule.
- **S3 — `sparse_gp=True` default.** v2's sparse indexed geometric product (custom backward,
  input-saving) reorders the blade contractions — same math, different summation order, so
  bit-parity with v1's dense einsum is unattainable *by design*. Shipped: v2 default ON.
  Verification: tier 1 pins it off to prove everything else at < 1e-10; tier 2 leaves it on,
  isolating exactly this reorder — measured ≤ 6.0e-16 movement on every unflagged model
  (table above), i.e. invisible at production widths. Phase 1a `torch.autograd.gradcheck` on
  `geometric_product` with `sparse_gp=True` passed in fp64 (reported at session start).
- **S5 — qkv scalar biases removed.** v2's qkv projections are all `bias=False`; v1
  initialized those biases nonzero-uniform (slim `attention.linear_in.linear_s.bias`, full
  `attention.qkv_module.in_linear.{s2mvs,mvs2s}.bias`). No knob — baked in; shipped as v2
  ships. Verification: normalize-at-record (fixture models had exactly those biases zeroed,
  count-asserted per block, net-level `linear_in` explicitly excluded per §2.5); Gate B/C
  waive exactly the `QKV_BIAS_RE`-derived set, nothing else. Param deltas per model are the
  Gate B table's waived column (84–4 608).
- **S6 — GLU vector-gate scale.** v2 scales the slim GLU vector gate by `0.5 = 1/√4`,
  flagless. Shipped: as v2 ships. Verification compensation: multiply both gate chunks of
  every fused GLU `weight_v` (rule `GLU_WEIGHT_V_RE`, never a hand list) by √2 at transplant
  — algebraically exact (√2·√2·0.5 = 1), confirmed by the machine-precision tier-1 column.
  Never applied outside the verification instrument (H7: checkpoint porting is a non-goal).
- **S7 — init refinements.** v2 zero-inits `linear_s` biases and refines slim init/scaling;
  with S2/S5 changing the parameter set, same-seed cross-version builds are incomparable —
  this is what retired rev 1's same-seed Gate C and forced the transplant design. Shipped: v2
  inits (they affect fresh training only; fixtures carry recorded weights, so no
  compensation exists or is needed).
- **S8 — AMP strategy.** v2 keeps vector/multivector math in fp32 under autocast and adds
  the `naive_amp` bypass. Inert here: every wrapper runs `use_amp: false` and all gates are
  fp64 CPU (H14(1) documents that an autocast-only defect would be invisible to A–H). Shipped:
  no change. Standing trigger: re-read the §2.3 S8 row before ever enabling amp.
- **S9 — gelu flavor unification (found BY Gate C, the migration's one surprise).** v2's
  unified `get_nonlinearity` makes `"gelu"` the tanh approximation everywhere — its own
  docstring says so — while v1 *split* flavors: slim used exact erf-GeLU everywhere; the
  full model's `ScalarGatedNonlinearity` used tanh-GeLU on the multivector gate
  (`gated_gelu`) but erf-GeLU on the auxiliary scalar stream. First tier-1 runs failed at
  ~1e-4 with first divergence in block 0's MLP; a blanket-erf patch then *worsened* the full
  models, which is what exposed the split (the v1 gate really was tanh). Shipped: v2's
  unified tanh (Posture B; ~1e-4-scale output shift vs v1 at eval, an accepted re-baseline
  delta like S3/S5/S6). Verification: a forward-time instrument patch reproduces v1's exact
  split during the transplant comparisons only — construction-bound for slim
  (`get_nonlinearity`), forward-bound for `ScalarGatedNonlinearity`, so the patch must span
  the comparisons, and it is restored afterwards. It applies in BOTH tiers: tier 2 exists to
  isolate the S3 reorder at 1e-8, and without the S9 bridge it would instead measure the
  shipped ~1e-4 S9 delta and fail by construction. §2.1's "no effect at the defaults" line
  was corrected and the §2.3 S9 row added when this landed (5ea8afa).

### Task C entries

**Step 1 — posture-flip commit (2026-08-07; the S10 ruling closed the last open A–F item).**

- **Manifest re-baseline.** `production_manifests_v2.json` records the shipped (unpinned,
  v2-default) manifests, written only after an in-run rule check against the v1 baselines
  (`_rebaseline_rule_check`: removed == S5 exactly; added == the **structural** norm-gain
  set — the same helper the optimizer exemption uses; `requires_grad` follows the freezing
  rule on old and new parameters alike; totals exact). Numbers (v1 → v2-default totals):
  tag_lgatr 1 079 394 → 1 077 474 (−4 608 qkv, +2 688 gains / 48 tensors); tag_slim
  1 794 631 → 1 793 623 (−3 456, +2 448 / 48); CGENN-GraphTrans 1 154 664 → 1 153 064
  (−3 840, +2 240 / 40); CGENN-GPS 1 898 197 → 1 894 357 (−3 840, **+0** — bare-norm
  construction, the §2.1 structural claim measured); LN-GraphTrans 1 636 032 → 1 635 192
  (−2 880, +2 040 / 40); LN-GPS 2 179 493 → 2 176 613 (−2 880, **+0**); equivectors
  2 267 307 → 2 267 291 (−84, +68 / 4). `test_production_manifest` on 2.x now runs BOTH
  gates: the pinned rule-check vs v1 (migration evidence, unchanged) and a strict
  names/shapes/requires_grad/total regression against the v2 baseline (the campaign gate).
- **H15 exemption.** `lgatr_norm_gain_names` (structural `EquiLayerNorm`/`SlimRMSNorm`
  isinstance-collection — a name pattern would also catch SlimLinear's *real* `weight_v`
  weight, which must keep decaying) now feeds BOTH grouping paths: base (`is_bias`, net and
  framesnet alike) and the ParT/weaver path. The rule mirror in
  `test_weight_decay_grouping.py` tracks it; the previously-failing CGENN pair test is green
  again; the runbook's Gate-B extension landed as `test_norm_gains_sit_in_no_decay_groups`
  (the REAL experiment optimizer is built per model and every `*.weight_mv` / `*.weight_s`
  parameter is asserted into a `weight_decay=0` group; file total 17/17). Stated for the
  record, deliberately NOT changed: eventgen's third grouping path decays everything by
  design (on v1 too) and stays as-is; the ParT path's framesnet group has no bias split
  (pre-existing; lgatr-relevant only for the S10-ruled-unused equivectors composition).
- **Methods sentence** added to `todo.md` §4 per §2.4's obligation.

## Appendix A — evidence log

2026-07-29 (rev 1): diffed installed 1.4.4 against `dev@e8ba34d` for `__init__`, `nets/*`, `layers/*` (incl. attention + mlp configs), `interface/*`, `primitives/*` (incl. attention backends), `utils/*`; PyPI then topped at 1.4.4; lloca 1.3.6 imports checked; repo greps as cited.
2026-07-29 (rev 2): lgatr **2.0.0 on PyPI**; read CHANGELOG `[2.0.0]` in full and the `v1_to_v2.rst` summary (renames-only confirmed); verified at source: v2 net-level `LGATrSlim.forward` keeps `(..., items, v_channels, 4)` while blocks take `(..., 4, channels)`; GLU `0.5 = 1/sqrt(4)` at `layers/slim_layers.py:368-369` (vector gate only, no flag); v2 zero-inits `linear_s` bias (new ⇒ v1 didn't); `SlimLinear` scalar bias still exists (`bias=True`) — only **qkv** linears dropped it; repo has no Conditional-net usage; M4 depth semantics confirmed (`[in] + hidden×(n−1) + [out]`).
2026-07-29 (rev 3): full-source re-sieve of the **v2.0.0 tag** vs installed 1.4.4 — `layer_norm.py`, `lgatr_block.py`, `dropout.py`, `linear.py`, `attention/{self_attention,qkv}.py`, `mlp/{mlp,geometric_bilinears,nonlinearities}.py`, `primitives/{attention,attention_backends/*,bilinear,normalization,linear}.py`, `interface/spurions.py`, `nets/{lgatr,slim}.py`, `layers/slim_layers.py`. Key findings: `primitives` required on directly-constructed layers (M10; `EquiLinear`'s third positional); layer-level `EquiLayerNorm` default `elementwise_affine=False` (net-level default `True` — S2, all nets, per-grade `(mv,5)` gains on the full model); v2 `qkv.py` all `bias=False` vs v1 nonzero-uniform biases in `linear_s`/`s2mvs`/`mvs2s` (S5 exact tensors); block structures line-identical both families (`norm`→`norm1/norm2` split, stateless when affine off); GeoMLP recipe identical; qkv width formulas verbatim under rename; CPU attention safe via `misc.py` dense-mask fallback (v2 raises on xformers/flash kwargs off-CUDA, path never taken); spurion values byte-identical; `GradeDropout`/scalar-interface/`reinsert_*` unchanged; slim single-joint-sdpa call preserved (attn_dropout semantics intact); `SlimBlock` used by the repo only indirectly via `LGATrSlim` (no direct imports, grep-verified).

## Appendix B — fixture/transplant sketch

```python
# tests/experiments/test_lgatr_migration_parity.py  (sketch — implement in Task A)
import os, pathlib, pytest, torch

FIX = pathlib.Path(__file__).parent.parent / "fixtures" / "lgatr144"
RECORD = os.environ.get("LGATR_PARITY") == "record"   # only valid on lgatr 1.4.4
TIER1 = os.environ.get("LGATR_PARITY_TIER", "1") == "1"

# rename table applied to v1 state_dict keys; waivers cite S-items
KEY_MAP = {...}          # built EMPIRICALLY: v1 keys from fixtures + v2 keys from Phase 1a,
                         # ordered shape-matching + rename rules; reviewed and committed
WAIVED_MISSING = [...]   # qkv biases (S5); affine gains absent v1-side (S2)

def zero_qkv_scalar_bias(model):        # S5 normalization, BEFORE recording
    # walk named_modules, find each attention's qkv projection (slim linear_in /
    # full-model qkv EquiLinear), zero EVERY bias tensor beneath it -- the biases
    # live inside internal Linears (linear_s / s2mvs / mvs2s), not at the top level
    for mod in qkv_projection_modules(model):
        for name, p in mod.named_parameters():
            if name.endswith("bias"):
                torch.nn.init.zeros_(p)

def rescale_glu_gates(sd):              # S6 compensation, on the mapped v1 state_dict
    for key in glu_fused_linear_weight_v_keys(sd):
        w = sd[key]                     # rows = [v_pre | v_gates_1 | v_gates_2] chunks
        n = w.shape[0] // 3
        w[n:] = w[n:] * (2 ** 0.5)      # x sqrt(2) on both gate chunks -> cancels v2's 0.5
    return sd

@pytest.mark.parametrize("name", REDUCED_MODELS)
def test_transplant_parity(name):
    model = build_reduced(name)         # hydra compose + init_physics, seed 0, .double().eval()
    if RECORD:
        assert lgatr_version() == "1.4.4"          # hard precondition, never record on v2
        zero_qkv_scalar_bias(model)
        pack = run_fixed_batch(model)   # {out, block_acts, in_grad, grad_norms, cfg_yaml}
        torch.save({"sd": model.state_dict(), **pack}, FIX / f"{name}.pt")
        return                          # record mode is run TWICE; files must be byte-identical
    if not (FIX / f"{name}.pt").exists():
        pytest.skip("no 1.4.4 fixtures recorded")
    ref = torch.load(FIX / f"{name}.pt")
    sd = rescale_glu_gates(remap_keys(ref["sd"], KEY_MAP))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert set(missing) | set(unexpected) == waived_set(ref["sd"])   # rule-derived, both directions
    pack = run_fixed_batch(model)       # tier 1: sparse_gp=False for full-LGATr builds
    assert_config_diff_is_only_renames(ref["cfg_yaml"], pack["cfg_yaml"])   # blind spot 2
    tol = 1e-10 if TIER1 else 1e-8
    report_first_divergence(ref, pack, tol)        # outputs + per-block acts (blind spot 3)
    assert rel_dev(pack["out"], ref["out"]) < tol
    assert rel_dev(pack["in_grad"], ref["in_grad"]) < (tol if TIER1 else 1e-6)  # blind spot 1
```
