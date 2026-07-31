# lgatr 1.4.4 → 2.0.0 migration runbook

**Status: PLANNED — no migration work has been done. This document is the plan.**

- **lgatr 2.0.0 was released on PyPI on 2026-07-29.** This runbook (rev 2, same day) is pinned to that release: its CHANGELOG `[2.0.0]`, its `docs/source/v1_to_v2.rst`, and source diffs against installed 1.4.4.
- `requirements.txt` on this branch already declares `lgatr[xformers-attention]>=2.0.0`. **The environment has NOT been migrated** — installed lgatr is still 1.4.4, and Phase 0 (fixture capture) *requires* 1.4.4. Do not `pip install -r requirements.txt` into the fixture-capture environment.
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
- **GeoMLP recipe identical**: GeometricBilinear first, then ScalarGatedNonlinearity + EquiLinear per extra layer, GradeDropout after each; the channel-list arithmetic independently re-confirms the M4 `+1` depth mapping. `get_nonlinearity` unification: no effect at the defaults (gelu both).
- **qkv head-width formulas identical under the rename**: `max(mv_channels·ratio // num_heads, 1)` and `max(s_channels·ratio // num_heads, 4)` verbatim on both sides — M3 is a pure rename, no width drift.
- **CPU attention verified safe** — the sieve's biggest scare, resolved: v2 hard-gates `xformers`/`flash` off-CUDA and *raises* if their kwargs (`attn_bias`, `cu_seqlens_*`) are passed on CPU. But `experiments/misc.py` **already** materializes a dense `attn_mask` on CPU ("fallback to default attention") instead of backend-specific kwargs, so both 1.4.4 and v2 dispatch to native sdpa there — the CPU test suite passes on both for the same reason. The GPS hybrids always pass a dense `attn_mask` (+ optional `dropout_p`), i.e. native on every device, both versions. (New in v2, unused by the repo: explicit `backend="..."` selection kwarg.)
- **Spurion definitions byte-identical** (lightlike `[√2,0,0,1]`, spacelike, xyplane bivector, time reference; same kwarg names/defaults). `GradeDropout` semantics unchanged. `embed_scalar`/`extract_scalar` slots unchanged (component 0). `reinsert_mv/s_channels` machinery intact on the full net.

### 2.2 Mechanical breaks — port checklist

| # | Break | Exact sites | Fix |
|---|---|---|---|
| M1 | Slim blocks renamed with `Slim` prefix, moved to `lgatr.layers.slim_layers` (re-exported from `lgatr.layers` — **use the shallow path**; the deep path moved once already during dev) | `lorentznetlgatrslimgraphgps.py:50-54` (aliases become plain imports), `finetuneexperiment.py:6` | ~7 import lines |
| M2 | `LGATrSlim` module moved; top-level `from lgatr import LGATrSlim` unchanged | `_target_` in `tag/amp/eg_slim.yaml` → `lgatr.LGATrSlim` | 3 yaml lines |
| M3 | `SelfAttentionConfig.increase_hidden_channels` → `attn_ratio` | `cgennlgatrgraphgps.py:98-99` (+ second block ~:187), `CGENNLGATrGraphTransHybrid.py:1068`, `attention:` in `tag/amp/eg_lgatr.yaml` | rename |
| M4 | `MLPConfig.activation`→`nonlinearity`, `increase_hidden_channels`→`mlp_ratio`, `num_hidden_layers`→`num_layers_mlp` (**semantics confirmed**: v2 counts all layers, `mlp.py` builds `[in] + hidden×(n−1) + [out]`, so a v1 `num_hidden_layers` value ports as `+1`; repo passes neither → defaults `1`↔`2` are equivalent) | `cgennlgatrgraphgps.py:108-109`, `CGENNLGATrGraphTransHybrid.py:1073`, `mlp:` in `tag/amp/eg_lgatr.yaml` + `framesnet/equivectors/lgatr.yaml` | rename |
| M5 | `SlimRMSNorm` requires `(v_channels, s_channels)`; bare call breaks | `lorentznetlgatrslimgraphgps.py:91` | `SlimRMSNorm(v_channels, s_channels, elementwise_affine=False)` — affine must stay off here regardless of posture: the instance is shared across call sites, only valid stateless |
| M6 | s-channel types `int \| None` → `int = 0` | yaml `null`s are overwritten by `init_physics`; confirm no literal `None` reaches v2 (Gate A) | grep + Gate A |
| M7 | requirements | done on this branch: `lgatr[xformers-attention]>=2.0.0` | — |
| **M8** | **Slim blocks are channel-last**: `SlimLinear`/`SlimSelfAttention`/`SlimMLP`/`SlimRMSNorm` take `(..., 4, channels)`; this repo's GPS hybrid feeds `(B, P, V, 4)` | `lorentznetlgatrslimgraphgps.py` — every block call (`linear_in` :175, attention :81, mlp :86, norm/dropout :91-92) plus the spurion `cat` at dim 2; `finetuneexperiment.py:115` must match v2's *internal* layout at the `linear_out` splice point | First pass: `transpose(-1, -2)` at block boundaries (mechanical, provable). Optional later: flip the file's internal convention to channel-last as a perf follow-up (§8), separately gated |
| M9 | v1.4.4 slim `compile_dynamic=True` default replaced by `compile_kwargs` (dynamic **no longer defaulted**) | `tag_slim.yaml` (`compile: true`) | add `compile_kwargs: {dynamic: true}` to preserve behavior — variable-length flattened batches otherwise recompile per shape |
| **M10** | **`primitives: PrimitivesConfig` is a REQUIRED constructor arg** on directly-constructed layers: `SelfAttention(cfg, primitives)`, `GeoMLP(cfg, primitives)`, and `EquiLinear(in_mv, out_mv, primitives, ...)` — third **positional**, so it also shifts any positional args after it. (`geometric_product(x, y, *, config=...)` likewise; repo has no direct primitive callers.) | `cgennlgatrgraphgps.py` (SelfAttention/GeoMLP constructions ~:98/:108 and the second block ~:187, EquiLinear), `finetuneexperiment.py:108` (`EquiLinear` linear_out splice) | build one `PrimitivesConfig()` per model and thread it, mirroring v2's own nets; a missed site is a loud `TypeError` |

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
| S8 | AMP strategy changed (vector/multivector path fp32, scalar path in autocast; `naive_amp` bypass added) | Inert today: every wrapper runs `use_amp: false` | Note in decision log; re-read this row before ever enabling amp |

(S4 from rev 1 — MLP depth semantics — is resolved and folded into M4.)

### 2.4 The two postures — DECIDED: Posture B (v2-native)

Because S5 + S6 are baked in, **"identical to the v1 campaign model" is not on the menu.** The coherent choices:

- **Posture A — closest-to-v1**: pin S1 (`nonlinearity_v: null`) and S2 (`norm_elementwise_affine: false`). Minimizes gratuitous deltas; params tables keep their v1 meaning. Still differs from v1 by S5/S6.
- **Posture B — v2-native** (recommended for a fresh campaign): accept upstream defaults (sigmoid gate "more stable", affine norms) as the new baseline. Since full v1 equivalence is impossible anyway, taking v2 as-shipped is the more reproducible citation ("lgatr 2.0.0 defaults"); re-baseline the param manifests once, in the same commit.

Either way: **verification always runs in parity mode** (pins + compensations) first — it certifies the port; the posture flip afterwards is a one-commit, documented model change, not a migration step.

**Decision (2026-07-27, maintainer + review): Posture B — adopt v2 defaults wholesale for every
row.** Rationale: the campaign has not been trained yet, so there are no v1 numbers to protect;
full v1 equivalence is impossible anyway (S5/S6 baked in); one lgatr version + one settings-set
per table (H5); "lgatr 2.0.0 defaults" is the cleanest citation. Obligations this creates:
a methods sentence that the `tag_lgatr`/`tag_slim` reference rows are re-trained under 2.0
(published-paper comparisons indicative, not exact), and a one-time re-baseline of the param
manifests in the posture-flip commit. Parity-mode verification (pins + compensations) still
runs first, unchanged — the posture applies from the first campaign run onward.

## 3. Why neither the test suite nor the official doc is enough

- The 64-test suite is the regression floor, not a parity proof: a gelu→sigmoid gate swap, a 0.5 gate rescale, or dropped qkv biases all produce different-but-perfectly-equivariant models. Wiring tests check channels, not values.
- The official `v1_to_v2.rst` covers **renames and config moves only** (verified against its source). S1–S8 live exclusively in the CHANGELOG. Port by both documents; trust neither alone.
- A silently-dropped stale key would *narrow* a model with no error if config casting ever ignored unknown fields; Phase 1a proves `SelfAttentionConfig(increase_hidden_channels=2)` raises `TypeError`, Gate B backstops regardless.
- Both lgatr versions can never coexist in one environment (same package name) — recorded fixtures are the only bridge across the swap, which is why Phase 0 is unskippable and first.

## 4. The workflow

### Phase 0 — capture on 1.4.4 (BEFORE any install or edit)

`tests/experiments/test_lgatr_migration_parity.py` (sketch: Appendix B), two modes (`LGATR_PARITY=record` / default check, which **skips cleanly when fixtures are absent**). Two fixture families, split to keep git small:

1. **Production manifests** (KB-scale json, all six lgatr-touching tagging configs — `tag_lgatr`, `tag_slim`, `tag_CGENNLGATrGraphTrans`, `tag_CGENNLGATrGraphGPS`, `tag_LorentzNetLGATrSlimGraphTrans`, `tag_LorentzNetLGATrSlimGraphGPS` — plus one learned-frames composition with `equivectors=lgatr`): total param count + sorted multiset of parameter shapes. Keys are *not* compared (renames make them legitimately differ); shapes and counts must not.
2. **Reduced-config transplant fixtures** (MB-scale, committed): the same model families at reduced size (`num_blocks=2`, hidden widths halved via overrides — every layer type, rename, and compensation is still exercised; parity logic is width-independent, and full-width state_dicts would be tens of MB of git). For each: fixed seed → instantiate (hydra compose + `init_physics`, same machinery as `test_jc_wiring.py`) → `.double().eval()` → **zero every bias under each qkv projection module (S5 normalization, per the S-table procedure)** → save, per model: full state_dict; final forward outputs; **per-block intermediate activations** (forward hooks at block boundaries); **a gradient pack** (input-gradient of `loss = out.square().sum()` plus the per-parameter grad-norm vector); and the **resolved-config snapshot** (`OmegaConf.to_yaml` of the composed config) — all on a fixed seeded batch (B=4, multiplicities `[1, n<knn_k, mid, large]` from `tests/experiments/utils.py`). Run record mode **twice** and assert the fixture files are byte-identical (determinism precondition, §4).

The recorded state_dicts double as the v1 side of the **KEY_MAP**: do not hand-write the v1→v2 key mapping. Phase 1a dumps the v2 key lists for the same reduced configs; build the map by ordered shape-matching plus the known rename rules (module moves, `norm`→`norm1`/`norm2` adds nothing while norms are parameter-free), review it once, commit it next to the fixtures.

Scope note: the transplant exists to *verify the port*, nothing else — migrating trained checkpoints across lgatr versions is a non-goal (H7), so KEY_MAP never needs to handle production-width models.

Commit script + fixtures to this branch.

### Phase 1 — environment swap + Phase 1a re-verification

1. Fresh session/venv: `pip install "lloca[xformers-attention]==1.3.6" "lgatr[xformers-attention]==2.0.0"`.
2. **Phase 1a (~15 min, non-negotiable):** re-verify §2 against the installed release — read `v1_to_v2.rst` **and** CHANGELOG `[2.0.0]`; run the import one-liners (`from lgatr.layers import SlimMLP, ...`; top-level symbols); confirm `SelfAttentionConfig(increase_hidden_channels=2)` raises `TypeError`; confirm M10 (`SelfAttention(cfg)` / `GeoMLP(cfg)` / `EquiLinear(i, o)` without `primitives` raise `TypeError`); confirm `import lloca.equivectors.lgatr` works; confirm `embed_vector` slots 1:4; diff `SlimMLP`/`SlimSelfAttention`/`SlimLinear` signatures against `lorentznetlgatrslimgraphgps.py` call sites; confirm block channel-last docstrings and the net-level channel-first interface; run `torch.autograd.gradcheck` on `geometric_product` with `sparse_gp=True` in fp64 (custom-backward assurance, §4); dump v2 state_dict key lists for the reduced fixture configs and build+commit KEY_MAP (Phase 0 note).

### Phase 2 — mechanical port

M1–M10 as a literal checklist, one commit per row (or code/yaml pairs) for instant bisection. M8 first pass = boundary transposes only.

### Phase 3 — parity pins, then posture

Apply S1/S2 pins for verification. After Gates A–F pass, make the §2.4 posture decision in its own commit with a decision-log entry; if Posture B, re-record production manifests as the new baseline in that commit.

### Phase 4 — gates (A–F on CPU, G–H on cluster)

| Gate | What | Pass criterion |
|---|---|---|
| A | Composition: `test_jc_wiring.py` + parity script instantiates all fixture configs on v2 | no exceptions; channel asserts hold |
| B | Production manifests vs fixtures — each manifest records `(shape, requires_grad)` per parameter, not shapes alone | identical with waivers **derived by rule, never typed by hand**: the expected-missing set is computed from the v1 keys themselves (every bias under a qkv projection module → S5), and the expected `requires_grad` flips from v2's dead-tail/zero-size freezing. Set equality asserted in **both** directions — an unexpectedly missing, present, or re-frozen parameter fails. With the Phase 3 pins there are no other legal diffs during verification; the Posture-B re-baseline is its own later, equally rule-checked step |
| C | **Transplant parity** on reduced configs: map v1 state_dict keys → v2 (KEY_MAP), apply the S6 `sqrt(2)` gate-chunk rescale, load (`strict=False` only for rule-waived keys), then compare against the recorded batch: final outputs, **per-block intermediate activations**, and **gradients** (input-gradient of a fixed scalar loss + the per-parameter grad-norm vector) | **Tier 1** (S1 pin; `sparse_gp=False` for full-LGATr models — slim has no geometric product): relative deviation < 1e-10 in fp64 (bit-exactness is deliberately not claimed — v2 dropped `opt_einsum` and reordered contractions). **Tier 2** (`sparse_gp=True`): < 1e-8 forward/activations, < 1e-6 for gradients through sparse_gp's custom backward. First diverging *block* is reported, not just pass/fail |
| D | Full existing suite | 64/64 |
| E | Identity-frames bit-exactness spot check | hybrid with identity frames ≡ plain backbone, bit-identical on v2 (internal-consistency proof) |
| F | Blade-table equivalence | audit script vs v2 `lgatr.primitives.bilinear._load_geometric_product_tensor`: agreement ≤ 2e-6 fp32, as on 1.4.4 |
| G | Training sanity (cluster) | fixed-seed 1k-iter quick runs (`tag_slim`, `tag_lgatr`) on both versions: final train loss within seed-noise band (2–3 seeds). Point-wise curve equality is **out of scope by design** (S3/S5/S6/S7); this catches gross regressions only |
| H | Throughput report (cluster) | not pass/fail: it/s for `tag_lgatr` v2 `compile=True/False` vs 1.4.4, and `tag_slim` (already compiled on 1.4.4, so expect little). Quantifies the carrot; publish the number in the decision log either way |

### Why these gates discriminate, and their blind spots (closed)

Both param counts and forward outputs *legitimately* differ across versions, so the gates are built to separate "expected delta" from "mistake" **structurally**, not by eyeballing tolerances:

- **Separation argument (Gate C).** Every neutralization is *exact*: S5 is removed from the reference at record time, S6 cancels algebraically (√2·√2·0.5 = 1), S1/S2 are pinned, tier 1 forces the dense product. A perfect port therefore lands at fp64 reassociation level (≲1e-12 relative), while every plausible mistake — one missed M8 transpose, one un-rescaled GLU of twelve, a crossed KEY_MAP pair, an un-zeroed bias at record, a wrong ratio — perturbs pre-activations at O(1e-2)–O(1). That is **6+ orders of magnitude of margin**; the exact epsilon is not load-bearing, which is what makes the gate trustworthy rather than tuned-to-pass.
- **Waivers are rules, not lists (Gate B).** The "params may differ" problem is closed by *deriving* the allowed diff from the v1 state_dict plus the S-items, and asserting set equality both ways. There is no free-text allowance a tired porter can widen.
- **Blind spot 1 — the backward pass.** Forward parity cannot see a broken custom backward (v2's sparse_gp ships one that saves only its inputs) or wrongly frozen parameters (`_freeze_dead_tail`, zero-size gain freezing): a model can match forward to 1e-12 and still train wrong. Closed three ways: gradients are part of the fixtures (Gate C), `requires_grad` is part of the manifests (Gate B), and Phase 1a runs `torch.autograd.gradcheck` on `geometric_product` with `sparse_gp=True` in fp64.
- **Blind spot 2 — train-only config values.** Eval-mode fixtures are blind to `dropout`/`attn_dropout`/schedule slips introduced while renaming yaml keys (a stray `attn_dropout: 0.5` changes no parameter and no eval output). Closed by **resolved-config snapshots**: Phase 0 records `OmegaConf.to_yaml` of every composed fixture config; post-port the snapshots are diffed and the only allowed changes are exactly the renamed keys (rule-checked like Gate B).
- **Blind spot 3 — localization.** A final-output mismatch alone says "somewhere in 24 sublayers". Per-block activation fixtures turn any Gate C failure into "first divergence at block k, sublayer s".
- **Determinism precondition.** Record mode runs twice; fixture files must be byte-identical before anything is compared across versions — otherwise the comparison would chase RNG, not the port.

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
- **H8 — literal `None` s-channels**: v1 accepted `None`, v2 wants ints; yaml `null`s are placeholders filled by `init_physics` — Gate A confirms no live `None` path.
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

## 6. Upstream (`heidelberg-hepml/lloca-experiments`) variant

No hybrids, no direct block construction ⇒ no M8 surface except their `finetuneexperiment` equivalent. Surface = M2/M3/M4/M9 renames + M10 at their `finetuneexperiment` `EquiLinear`/slim-`Linear` splices + flex monkeypatch + their suite as Gate D, same record→port→prove shape (S2/S5/S6 still apply to their nets — transplant needs the same compensations). ≈1 day. Offer the fixture-script pattern with the port PR.

## 7. Task split (Claude Code web)

1. **Task A (environment must be on 1.4.4):** implement + run Phase 0; commit script + fixtures to `dev`. *Blocking precondition for everything else; do not install v2 in that session.* NB: `dev`'s own `requirements.txt` already demands `>=2.0.0`, so a session whose setup auto-installed it must first `pip install "lgatr[xformers-attention]==1.4.4"` and verify `lgatr.__version__` before recording — record-mode must hard-assert the version.
2. **Task B (fresh session):** install `lgatr==2.0.0` → Phase 1a → 2 → 3 (parity mode) → Gates A–F. Push; PR only when green.
3. **Task C (cluster/user):** Gates G–H, posture decision + possible manifest re-baseline, `.sif` rebuild, Phase 5, merge.

## 8. Out of scope, captured for the follow-up task: performance transfers to CGENN

Not migration work — a separate task after gates pass; recorded here so the thinking isn't lost:

- **Profile first**: the FLOPs tests already emit per-jet FLOPs per model — compare CGENN-hybrid rows against `tag_slim` before optimizing anything. Post-migration, the compiled attention half of every CGENN-hybrid block gets faster, making the un-optimized CGENN branch the *relative* bottleneck almost by construction (~N·k dense-Cayley contractions per jet per layer).
- **Sparse-indexed GP transfers almost verbatim** (same Cl(1,3) 16-blade algebra; 256/4096 nonzero Cayley entries, one output blade per pair): rewrite the `fcgp.py`/`gp.py` einsums as precomputed (indices, signs) gathers with an input-saving backward — upstream made it default "because always faster", eager included. One rewrite serves the FC baseline *and* the GPS hybrid (shared modules); the GraphTrans hybrid's private `CliffordAlgebra` copy needs the same treatment separately. Mathematically identical, reorder-only — a documented performance change, no modeling change.
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
