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

All current v-channel widths are ≠ 4, so a missed M8 transpose **crashes loudly** (channel-dim mismatch). Keep it that way: never write fixtures or tests with `v_channels == 4`, the one width where a layout error becomes a silent transpose-alias (H13).

### 2.3 Silent numerics deltas (CHANGELOG-only; absent from the migration doc)

| # | Delta | Effect | Handling |
|---|---|---|---|
| S1 | Slim GLU gate nonlinearity: v2 `nonlinearity_v="sigmoid"` default (v1: one `nonlinearity`, gelu, for both gate paths) | Different activations in every slim model | Flag exists: `nonlinearity_v=None` restores v1 routing. Verification runs with the pin; shipped posture decided in Phase 3 |
| S2 | `norm_elementwise_affine=True` default: slim RMSNorm gains `weight_v`/`weight_s` (v1: parameter-free) | Param counts shift (~3k for tag_slim); fairness/params tables change | Flag exists: `false` for parity. Note: gains init to 1.0 ⇒ identity at transplant time either way |
| S3 | `sparse_gp=True` default: geometric product via gather-reduce — reordered, **not bit-identical** (upstream's own docstring) | Full-LGATr models reproduce dense results only to tolerance | Keep `True` for runtime (the speed carrot); `primitives={'sparse_gp': False}` **only inside tier-1 verification** |
| **S5** | **qkv scalar bias removed** from q/k/v linears in *all* models ("because redundant"). v1 did **not** zero-init that bias (v2's bias-zeroing is listed as new) ⇒ removal changes fresh-model outputs | v1 state_dicts contain qkv s-bias values with no v2 slot | **Normalize-at-record**: zero all qkv scalar biases in the v1 model *before* recording fixtures (a zero-bias v1 model is still a valid v1 model — this moves the reference into the intersection of both architectures). Gate B: waiver for the missing params |
| **S6** | **Slim GLU vector-gate scale**: v2 multiplies the gate inner product by `0.5 = 1/sqrt(4)` (`slim_layers.py:368-369`). **No flag.** | Every slim model's forward differs even with identical weights and S1 pinned | **Exact compensation in transplant**: scale the two vector-gate chunks (`v_gates_1`, `v_gates_2` rows of each GLU's fused linear `weight_v`) by `sqrt(2)` each ⇒ inner product ×2 ⇒ cancels the 0.5 exactly. Scalar-gate path is unscaled in v2 — do not touch it. Vector path has no bias — nothing else to compensate |
| S7 | Slim init/scaling refinements (`linear_s` bias→0 at init, plus "micro speed/memory optimizations") | Same-seed fresh-init models are not comparable across versions — at all | This is *why* verification is transplant-based. Init-distribution changes are an accepted training-dynamics delta of v2 (no eval-parity impact once weights are transplanted) |
| S8 | AMP strategy changed (vector/multivector path fp32, scalar path in autocast; `naive_amp` bypass added) | Inert today: every wrapper runs `use_amp: false` | Note in decision log; re-read this row before ever enabling amp |

(S4 from rev 1 — MLP depth semantics — is resolved and folded into M4.)

### 2.4 The two postures (decide in Phase 3, after gates pass)

Because S5 + S6 are baked in, **"identical to the v1 campaign model" is not on the menu.** The coherent choices:

- **Posture A — closest-to-v1**: pin S1 (`nonlinearity_v: null`) and S2 (`norm_elementwise_affine: false`). Minimizes gratuitous deltas; params tables keep their v1 meaning. Still differs from v1 by S5/S6.
- **Posture B — v2-native** (recommended for a fresh campaign): accept upstream defaults (sigmoid gate "more stable", affine norms) as the new baseline. Since full v1 equivalence is impossible anyway, taking v2 as-shipped is the more reproducible citation ("lgatr 2.0.0 defaults"); re-baseline the param manifests once, in the same commit.

Either way: **verification always runs in parity mode** (pins + compensations) first — it certifies the port; the posture flip afterwards is a one-commit, documented model change, not a migration step.

## 3. Why neither the test suite nor the official doc is enough

- The 64-test suite is the regression floor, not a parity proof: a gelu→sigmoid gate swap, a 0.5 gate rescale, or dropped qkv biases all produce different-but-perfectly-equivariant models. Wiring tests check channels, not values.
- The official `v1_to_v2.rst` covers **renames and config moves only** (verified against its source). S1–S8 live exclusively in the CHANGELOG. Port by both documents; trust neither alone.
- A silently-dropped stale key would *narrow* a model with no error if config casting ever ignored unknown fields; Phase 1a proves `SelfAttentionConfig(increase_hidden_channels=2)` raises `TypeError`, Gate B backstops regardless.
- Both lgatr versions can never coexist in one environment (same package name) — recorded fixtures are the only bridge across the swap, which is why Phase 0 is unskippable and first.

## 4. The workflow

### Phase 0 — capture on 1.4.4 (BEFORE any install or edit)

`tests/experiments/test_lgatr_migration_parity.py` (sketch: Appendix B), two modes (`LGATR_PARITY=record` / default check, which **skips cleanly when fixtures are absent**). Two fixture families, split to keep git small:

1. **Production manifests** (KB-scale json, all six lgatr-touching tagging configs — `tag_lgatr`, `tag_slim`, `tag_CGENNLGATrGraphTrans`, `tag_CGENNLGATrGraphGPS`, `tag_LorentzNetLGATrSlimGraphTrans`, `tag_LorentzNetLGATrSlimGraphGPS` — plus one learned-frames composition with `equivectors=lgatr`): total param count + sorted multiset of parameter shapes. Keys are *not* compared (renames make them legitimately differ); shapes and counts must not.
2. **Reduced-config transplant fixtures** (MB-scale, committed): the same model families at reduced size (`num_blocks=2`, hidden widths halved via overrides — every layer type, rename, and compensation is still exercised; parity logic is width-independent, and full-width state_dicts would be tens of MB of git). For each: fixed seed → instantiate (hydra compose + `init_physics`, same machinery as `test_jc_wiring.py`) → `.double().eval()` → **zero all qkv scalar biases (S5 normalization)** → save full state_dict + forward outputs on a fixed seeded batch (B=4, multiplicities `[1, n<knn_k, mid, large]` from `tests/experiments/utils.py`).

Commit script + fixtures to this branch.

### Phase 1 — environment swap + Phase 1a re-verification

1. Fresh session/venv: `pip install "lloca[xformers-attention]==1.3.6" "lgatr[xformers-attention]==2.0.0"`.
2. **Phase 1a (~15 min, non-negotiable):** re-verify §2 against the installed release — read `v1_to_v2.rst` **and** CHANGELOG `[2.0.0]`; run the import one-liners (`from lgatr.layers import SlimMLP, ...`; top-level symbols); confirm `SelfAttentionConfig(increase_hidden_channels=2)` raises `TypeError`; confirm `import lloca.equivectors.lgatr` works; confirm `embed_vector` slots 1:4; diff `SlimMLP`/`SlimSelfAttention`/`SlimLinear` signatures against `lorentznetlgatrslimgraphgps.py` call sites; confirm block channel-last docstrings and the net-level channel-first interface.

### Phase 2 — mechanical port

M1–M9 as a literal checklist, one commit per row (or code/yaml pairs) for instant bisection. M8 first pass = boundary transposes only.

### Phase 3 — parity pins, then posture

Apply S1/S2 pins for verification. After Gates A–F pass, make the §2.4 posture decision in its own commit with a decision-log entry; if Posture B, re-record production manifests as the new baseline in that commit.

### Phase 4 — gates (A–F on CPU, G–H on cluster)

| Gate | What | Pass criterion |
|---|---|---|
| A | Composition: `test_jc_wiring.py` + parity script instantiates all fixture configs on v2 | no exceptions; channel asserts hold |
| B | Production manifests vs fixtures | identical **modulo the explicit waiver list**, each waiver citing an S-item (expected: qkv s-bias params absent per S5; gain params per S2 if Posture B). Anything unexplained = fail |
| C | **Transplant parity** on reduced configs: map v1 state_dict keys → v2 (rename table), apply the S6 `sqrt(2)` gate-chunk rescale, load (`strict=False` only for waivered keys), compare forward outputs on the recorded batch | **Tier 1** (S1 pin; `sparse_gp=False` for full-LGATr models — slim has no geometric product): fp64 max-abs-diff < 1e-12. **Tier 2** (`sparse_gp=True`): < 1e-8. Failure semantics are clean: a wrong compensation or missed transpose shows up O(1), reassociation shows up <1e-8 |
| D | Full existing suite | 64/64 |
| E | Identity-frames bit-exactness spot check | hybrid with identity frames ≡ plain backbone, bit-identical on v2 (internal-consistency proof) |
| F | Blade-table equivalence | audit script vs v2 `lgatr.primitives.bilinear._load_geometric_product_tensor`: agreement ≤ 2e-6 fp32, as on 1.4.4 |
| G | Training sanity (cluster) | fixed-seed 1k-iter quick runs (`tag_slim`, `tag_lgatr`) on both versions: final train loss within seed-noise band (2–3 seeds). Point-wise curve equality is **out of scope by design** (S3/S5/S6/S7); this catches gross regressions only |
| H | Throughput report (cluster) | not pass/fail: it/s for `tag_lgatr` v2 `compile=True/False` vs 1.4.4, and `tag_slim` (already compiled on 1.4.4, so expect little). Quantifies the carrot; publish the number in the decision log either way |

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
- **H7 — checkpoints**: state_dict keys and (S2/S5) shapes change; migrate before the campaign so no checkpoint survives the boundary.
- **H8 — literal `None` s-channels**: v1 accepted `None`, v2 wants ints; yaml `null`s are placeholders filled by `init_physics` — Gate A confirms no live `None` path.
- **H9 — CPU-only web containers**: every parity gate is fp64-CPU by design; the container's xformers is ABI-mismatched (`--no-deps`), so gates must not touch CUDA kernels (CPU `BlockDiagonalMask` only, as the suite already does).
- **H10 — installs are now standard PyPI** (2.0.0 released); v2 even drops `einops`/`opt_einsum`/`numpy` deps. The `.sif` rebuild is routine — but rebuild it *once, before* the campaign, not between runs.
- **H11 — compile expectations**: slim already compiled on 1.4.4 (keep `dynamic: true` via M9); the genuinely new capability is compile for **full-LGATr** + `warmup_caches` + compiled-xformers custom ops (attention no longer graph-breaks). Enabling it is a *post-migration* enhancement gated by Gate H numbers. For compile+DDP, note v2's own fixes here (unused-param `requires_grad_(False)`, tensor-ized norm eps/gains) — mirror that pattern in any local module you compile under DDP.
- **H12 — the official migration doc is renames-only** (verified). The CHANGELOG is the behavioral source of truth. Port by both.
- **H13 — `v_channels == 4` is the silent-alias width** for the M8 layout flip (transpose becomes shape-legal). All current widths differ from 4; keep fixtures and tests that way so layout mistakes stay loud.

## 6. Upstream (`heidelberg-hepml/lloca-experiments`) variant

No hybrids, no direct block construction ⇒ no M8 surface except their `finetuneexperiment` equivalent. Surface = M2/M3/M4/M9 renames + `finetuneexperiment` + flex monkeypatch + their suite as Gate D, same record→port→prove shape (S5/S6 still apply to their slim models — transplant needs the same compensations). ≈1 day. Offer the fixture-script pattern with the port PR.

## 7. Task split (Claude Code web)

1. **Task A (environment still on 1.4.4):** implement + run Phase 0; commit script + fixtures to `dev`. *Blocking precondition for everything else; do not install v2 in that session.*
2. **Task B (fresh session):** install `lgatr==2.0.0` → Phase 1a → 2 → 3 (parity mode) → Gates A–F. Push; PR only when green.
3. **Task C (cluster/user):** Gates G–H, posture decision + possible manifest re-baseline, `.sif` rebuild, Phase 5, merge.

## 8. Out of scope, captured for the follow-up task: performance transfers to CGENN

Not migration work — a separate task after gates pass; recorded here so the thinking isn't lost:

- **Profile first**: the FLOPs tests already emit per-jet FLOPs per model — compare CGENN-hybrid rows against `tag_slim` before optimizing anything. Post-migration, the compiled attention half of every CGENN-hybrid block gets faster, making the un-optimized CGENN branch the *relative* bottleneck almost by construction (~N·k dense-Cayley contractions per jet per layer).
- **Sparse-indexed GP transfers almost verbatim** (same Cl(1,3) 16-blade algebra; 256/4096 nonzero Cayley entries, one output blade per pair): rewrite the `fcgp.py`/`gp.py` einsums as precomputed (indices, signs) gathers with an input-saving backward — upstream made it default "because always faster", eager included. One rewrite serves the FC baseline *and* the GPS hybrid (shared modules); the GraphTrans hybrid's private `CliffordAlgebra` copy needs the same treatment separately. Mathematically identical, reorder-only — a documented performance change, no modeling change.
- **FC baseline only**: the all-pairs padded graph admits a dense `(B, N, N, ·)` masked-mean reformulation — no scatter, fixed shapes, compiles cleanly, better locality even eager. **The kNN hybrids keep the scatter** — sparsity is the design there; `index_add_` compiles fine with `dynamic=True`.
- Compile knobs: `dynamic=True` over batch/N; `activation_memory_budget` (torch≥2.4) if N² intermediates pinch; AMP split (multivector fp32 / scalar bf16) after the parity dust settles.

## Appendix A — evidence log

2026-07-29 (rev 1): diffed installed 1.4.4 against `dev@e8ba34d` for `__init__`, `nets/*`, `layers/*` (incl. attention + mlp configs), `interface/*`, `primitives/*` (incl. attention backends), `utils/*`; PyPI then topped at 1.4.4; lloca 1.3.6 imports checked; repo greps as cited.
2026-07-29 (rev 2): lgatr **2.0.0 on PyPI**; read CHANGELOG `[2.0.0]` in full and the `v1_to_v2.rst` summary (renames-only confirmed); verified at source: v2 net-level `LGATrSlim.forward` keeps `(..., items, v_channels, 4)` while blocks take `(..., 4, channels)`; GLU `0.5 = 1/sqrt(4)` at `layers/slim_layers.py:368-369` (vector gate only, no flag); v2 zero-inits `linear_s` bias (new ⇒ v1 didn't); `SlimLinear` scalar bias still exists (`bias=True`) — only **qkv** linears dropped it; repo has no Conditional-net usage; M4 depth semantics confirmed (`[in] + hidden×(n−1) + [out]`).

## Appendix B — fixture/transplant sketch

```python
# tests/experiments/test_lgatr_migration_parity.py  (sketch — implement in Task A)
import os, pathlib, pytest, torch

FIX = pathlib.Path(__file__).parent.parent / "fixtures" / "lgatr144"
RECORD = os.environ.get("LGATR_PARITY") == "record"   # only valid on lgatr 1.4.4
TIER1 = os.environ.get("LGATR_PARITY_TIER", "1") == "1"

# rename table applied to v1 state_dict keys; waivers cite S-items
KEY_MAP = {...}          # e.g. "blocks.0.mlp." prefixes etc. — fill at implementation
WAIVED_MISSING = [...]   # qkv scalar biases (S5); affine gains absent v1-side (S2)

def zero_qkv_scalar_bias(model):        # S5 normalization, BEFORE recording
    for name, p in model.named_parameters():
        if is_qkv_scalar_bias(name):    # pattern match: slim linear_in / LGATr qkv EquiLinear
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
        zero_qkv_scalar_bias(model)
        out = forward_fixed_batch(model)
        torch.save({"sd": model.state_dict(), "out": out}, FIX / f"{name}.pt")
        return
    if not (FIX / f"{name}.pt").exists():
        pytest.skip("no 1.4.4 fixtures recorded")
    ref = torch.load(FIX / f"{name}.pt")
    sd = rescale_glu_gates(remap_keys(ref["sd"], KEY_MAP))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert set(missing) | set(unexpected) <= set(WAIVED_MISSING)   # Gate B spirit, zero surprises
    out = forward_fixed_batch(model)    # tier 1: sparse_gp=False for full-LGATr builds
    assert (out - ref["out"]).abs().max() < (1e-12 if TIER1 else 1e-8)
```
