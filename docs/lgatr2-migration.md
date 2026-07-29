# lgatr 1.4.4 → 2.0 migration runbook

**Status: PLANNED — no migration work has been done. This document is the plan.**

- Prepared 2026-07-29 against installed `lgatr==1.4.4` (PyPI latest) and `heidelberg-hepml/lgatr@dev` at commit `e8ba34d` ("Fix release date", 2026-07-29).
- The 2.0 release is imminent: dev carries "Fix release date" (`e8ba34d`) and a new v1→v2 differences docs section (`35f052c`). **Read that upstream doc at execution time** — it did not exist when the inventory below was compiled from source diffs.
- `requirements.txt` currently pins `lgatr[xformers-attention]>=1.4.2, <2.0.0`. That upper bound is the fence that makes this migration deliberate; it is lifted in Phase 1, not before.
- lloca stays frozen at `1.3.6` for the whole migration (one variable at a time).

⚠ **The inventory in §2 has a shelf life.** Between two source fetches *on the same day* this plan was written, dev moved `slim_layers.py` from `lgatr/nets/` to `lgatr/layers/` (`b3615fa`). Every path below was re-verified against `e8ba34d`, but the executing session MUST re-run the Phase 1a re-verification checklist rather than trusting this table blindly.

---

## 0. TL;DR

- Effort: ~2 focused days for this fork (+1 for surprises), ~1 day for upstream `lloca-experiments`.
- Approach: **record → port → prove.** Golden parity fixtures are captured on 1.4.4 *before anything is installed or edited*; the migration is complete only when every gate in Phase 4 passes. "Port it carefully" is not a gate.
- Timing rule: migrate at the first official 2.0 PyPI release, **before** a training campaign starts, never mid-campaign. Two non-bit-identical lgatr kernels must never sit behind one results table.
- Default posture: pin v2 to 1.4.4-equivalent behavior via three parity flags (§4 Phase 3); adopt v2's behavioral changes later as separate, documented decisions.

## 1. Scope and execution environment

In scope: everything that imports lgatr in this fork —

| Surface | Files |
|---|---|
| Slim building blocks (direct construction) | `experiments/baselines/lorentznetlgatrslimgraphgps.py` |
| Slim net (hydra `_target_`) | `config/model/tag_slim.yaml`, `amp_slim.yaml`, `eg_slim.yaml`; `experiments/baselines/lorentznetlgatrslimgraphtrans.py` |
| Full LGATr net (hydra `_target_`) | `config/model/tag_lgatr.yaml`, `amp_lgatr.yaml`, `eg_lgatr.yaml`; `experiments/baselines/CGENNLGATrGraphTransHybrid.py` |
| LGATr layers (direct construction) | `experiments/baselines/cgennlgatrgraphgps.py` |
| Frames equivectors (via lloca) | `config/model/framesnet/equivectors/lgatr.yaml` |
| Interface + attention plumbing | `experiments/tagging/wrappers.py`, `experiments/misc.py`, `experiments/{eventgen,amplitudes}/wrappers.py`, `experiments/tagging/finetuneexperiment.py` |

Execution split (matters because Claude Code web containers are **CPU-only**):

- **CPU-runnable (web sessions):** fixture record/check, param manifests, the full 64-test suite, all mechanical porting. The existing suite already runs on CPU (xformers `BlockDiagonalMask` CPU path works; the container's xformers is `--no-deps`-installed and ABI-mismatched with its torch, so nothing in the gates may require CUDA kernels).
- **Cluster-only (OSCAR, user-run or GPU session):** short training-parity run (Gate G), throughput benchmark (Gate H), `.sif` rebuild.

## 2. Verified inventory (evidence, as of dev@`e8ba34d`)

### 2.1 Confirmed NON-breaks — do not "fix" these

- **Blade layout unchanged.** Dev `embed_vector` = `F.pad(vectors, (1, 11))`, still slots 1:5; `extract_vector` still reads 1:5. The hybrid's local `CliffordAlgebra` blade-order match with lgatr survives (re-prove anyway: Gate F).
- **Symmetry class unchanged.** Dev `PrimitivesConfig.subgroup=True` (10-element SO⁺(1,3) linear basis) ≡ 1.4.4 `use_fully_connected_subgroup=True` default. Same model class, relocated flag.
- **lloca 1.3.6 survives.** Its three lgatr imports — `embed_vector`, `layers.EquiLayerNorm`, and the *private* `primitives.invariants._load_inner_product_factors(device=, dtype=)` — all exist on dev with compatible signatures. `Requires-Dist: lgatr` has no version cap.
- Top-level exports stable: `LGATr`, `LGATrSlim`, `embed_vector`, `extract_scalar`, `extract_vector`, `get_num_spurions`, `get_spurions`, `SelfAttentionConfig`, `MLPConfig`.
- `lgatr.layers.{EquiLinear, EquiLayerNorm, GradeDropout, GeoMLP}` paths stable (`cgennlgatrgraphgps.py:55-62` imports survive except the config-field renames below).
- `wrappers.py:1444` flex monkeypatch survives: dev `primitives/attention_backends/flex.py:10` still has module-level `attention = flex_attention`.
- `experiments/misc.py` backend kwargs match dev's dispatch registry exactly (`attn_bias` → xformers, `cu_seqlens_*` → flash, `block_mask` → flex); dev adds a new `varlen` backend (`cu_seq_q`/`cu_seq_k`/`max_q`/`max_k`).
- The `[xformers-attention]` extra name survives (new extras: `varlen-attention`, `flex-attention`, `flash-attention`).
- Repo uses no `compile_mode`/`compile_dynamic` (removed on dev in favor of `compile_kwargs`) — verified by grep.
- Slim block constructor args (`SelfAttention`/`MLP`/`Linear`/`Dropout`) are **identical** between 1.4.4 and dev — only names/module change. Sole exception: `RMSNorm` (below).

### 2.2 Mechanical breaks — port checklist

| # | Break | Exact sites | Fix |
|---|---|---|---|
| M1 | `lgatr.nets.lgatr_slim` module gone; classes renamed `MLP/Dropout/Linear/RMSNorm/SelfAttention/GatedLinearUnit/LGATrSlimBlock` → `SlimMLP/SlimDropout/SlimLinear/SlimRMSNorm/SlimSelfAttention/SlimGLU/SlimBlock`, now in `lgatr.layers.slim_layers` (moved there from `lgatr.nets` mid-flight — **import from `lgatr.layers`**, the shallow re-export, not the deep path) | `lorentznetlgatrslimgraphgps.py:50-54` (already aliased to the exact new names — the aliases become plain imports), `finetuneexperiment.py:6` | ~7 import lines |
| M2 | `LGATrSlim` moved `lgatr.nets.lgatr_slim` → `lgatr.nets.slim`; top-level `from lgatr import LGATrSlim` unchanged | `_target_` in `tag_slim.yaml`, `amp_slim.yaml`, `eg_slim.yaml` → `lgatr.LGATrSlim`; `lorentznetlgatrslimgraphtrans.py:46` needs nothing | 3 yaml lines |
| M3 | `SelfAttentionConfig.increase_hidden_channels` → `attn_ratio` | `cgennlgatrgraphgps.py:98-99` (and the second constructor block ~:187), `CGENNLGATrGraphTransHybrid.py:1068`, `attention:` blocks in `tag_lgatr.yaml`, `amp_lgatr.yaml`, `eg_lgatr.yaml` | rename |
| M4 | `MLPConfig.activation` → `nonlinearity`; `increase_hidden_channels` → `mlp_ratio`; `num_hidden_layers` → `num_layers_mlp` | `cgennlgatrgraphgps.py:108-109`, `CGENNLGATrGraphTransHybrid.py:1073`, `mlp:` blocks in `tag/amp/eg_lgatr.yaml` **and** `config/model/framesnet/equivectors/lgatr.yaml` (`mlp.activation: gelu`) | rename |
| M5 | `SlimRMSNorm` now requires `(v_channels, s_channels)` (for new affine gains); bare `RMSNorm()` call breaks | `lorentznetlgatrslimgraphgps.py:91` | `SlimRMSNorm(v_channels, s_channels, elementwise_affine=False)` — affine **must** stay off here regardless of parity policy: this instance is shared across call sites, which is only valid stateless |
| M6 | `in_s_channels`/`out_s_channels` type `int \| None` → `int = 0` on dev configs/nets | yaml `null` placeholders are overwritten with real ints by `init_physics` — confirm no call path delivers a literal `None` (Gate A exercises this) | grep + Gate A |
| M7 | requirements: lift `<2.0.0`, pin the release (`lgatr[xformers-attention]>=2.0.0,<3`) or a git SHA pre-release | `requirements.txt:23` | 1 line + `.sif` note (§5-H10) |

### 2.3 Silent numerics deltas — the reason this runbook exists

These change model behavior with **zero errors raised and all 64 current tests passing**:

| # | Delta | Effect | Parity pin |
|---|---|---|---|
| S1 | Slim GLU vector gate: 1.4.4 applies one `nonlinearity` (gelu) to both gate paths; dev adds `nonlinearity_v` defaulting to **"sigmoid"** | Every slim model (tag_slim + both LN-slim hybrids) computes different activations | Pass `nonlinearity_v=None` (falls back to `nonlinearity`) in `LGATrSlim` yamls and at the `SlimMLP` construction in `lorentznetlgatrslimgraphgps.py:86` |
| S2 | `SlimRMSNorm` gains learnable per-channel `weight_v`/`weight_s`; net kwarg `norm_elementwise_affine=True` default (1.4.4 norm is parameter-free) | Param counts shift (~3k for tag_slim); fairness tables and params column change | `norm_elementwise_affine: false` in slim yamls; `elementwise_affine=False` at M5 |
| S3 | `PrimitivesConfig.sparse_gp=True` default routes the geometric product through a gather-reduce kernel — explicitly **not bit-identical** to the dense path (their docstring) | Full-LGATr models (tag_lgatr, CGENN hybrids' global branch, LGATrVectors framesnet) reproduce 1.4.4 only to tolerance | Keep `sparse_gp=True` for training (it is the speed carrot); use `primitives={'sparse_gp': False}` **only inside tier-1 parity checks** (Gate C) |
| S4 | `MLPConfig` default depth: 1.4.4 `num_hidden_layers=1` vs dev `num_layers_mlp=2` — repo passes neither, so defaults apply. Semantics *probably* map 1-hidden ≙ 2-layers, but this is unverified | If semantics differ, GeoMLP depth changes silently | Gate B (param manifest) is the arbiter; verify the mapping in Phase 1a |

### 2.4 Ecosystem constraints

- PyPI latest is 1.4.4; dev is unreleased. Until 2.0 ships, installing v2 means `lgatr[xformers-attention] @ git+https://github.com/heidelberg-hepml/lgatr@<sha>` — which propagates into the OSCAR `.sif` build recipe (needs git+network at build time). Prefer waiting for the PyPI release.
- lloca couples to lgatr through a private symbol (§2.1). Freeze `lloca==1.3.6` during migration; re-verify the private import if lloca is ever bumped. Worth an upstream issue asking for a public accessor.
- Old checkpoints: state_dict keys change wherever renamed classes are involved. This fork has no campaign checkpoints worth preserving — migrate before the campaign and this problem never exists. (If it ever does: key-remap script, out of scope here.)

## 3. Why the existing tests are not enough (the workflow's core principle)

The 64-test suite (32 equivariance + 24 invariance + 8 jc_wiring) is the *regression floor*, not the parity proof:

- **Equivariance tests can't see S1.** A gelu→sigmoid gate swap produces a different — but still perfectly Lorentz-equivariant — model. Tolerance-based SO⁺(1,3) checks pass before and after.
- **Wiring tests can't see S2/S4.** They assert channel counts, not parameter counts or output values.
- **A silently-dropped config key is worse than a crash.** If dev's `SelfAttentionConfig.cast` were to ignore an unknown `increase_hidden_channels: 2` from a stale yaml, hidden width silently reverts to the default (`attn_ratio=1`) — a *narrower model* with no error. (Dataclass `__init__` should raise `TypeError`; Phase 1a verifies it actually does. Gate B is the backstop either way.)

Therefore the migration is anchored on **golden fixtures recorded on 1.4.4**: fixed inputs → recorded outputs + parameter manifests, committed to the repo *before* the environment changes. Both lgatr versions can never coexist in one env (same package name), so the fixtures are the only bridge across the swap.

## 4. The workflow

### Phase 0 — capture baselines on 1.4.4 (BEFORE any install or edit)

Create `tests/experiments/test_lgatr_migration_parity.py` with two modes (sketch in Appendix B):

- `LGATR_PARITY=record pytest tests/experiments/test_lgatr_migration_parity.py` → writes `tests/fixtures/lgatr144/<model>.pt`
- default (check) mode → compares against fixtures; **skips cleanly if fixtures are absent** so CI never breaks on fresh clones.

For each of the six lgatr-touching tagging configs — `tag_lgatr`, `tag_slim`, `tag_CGENNLGATrGraphTrans`, `tag_CGENNLGATrGraphGPS`, `tag_LorentzNetLGATrSlimGraphTrans`, `tag_LorentzNetLGATrSlimGraphGPS` — plus one learned-frames composition with `equivectors=lgatr` (`LGATrVectors`; reuse the composition machinery from `test_tag_equivariance.py`), record:

1. **Param manifest**: total parameter count + the sorted multiset of parameter *shapes*. Compare shapes-and-counts, **not** state_dict keys — keys legitimately change with the class renames; shapes must not.
2. **Forward outputs**: `float64`, `.eval()` mode (BN frozen at init stats, dropout off), fixed seed for init, fixed synthetic batch from the `tests/experiments/utils.py` generator — B=4 jets with mixed multiplicities, deliberately including one jet with `n_real < knn_k` and one 1-particle jet (the historical edge cases).
3. Instantiate through hydra compose + `init_physics`, exactly like `test_jc_wiring.py`, so the *config path* is under test too — a stale yaml key that survives porting must surface here, not in a training run.

`finetuneexperiment.py` and the eventgen/amplitudes wrappers get import-smoke coverage only (their lgatr surface is top-level stable symbols).

Commit the script + fixtures to this branch. Fixtures are small (fp64 outputs for 4 jets ≈ KBs).

### Phase 1 — environment swap + Phase 1a re-verification

1. Fresh session/venv. `pip install "lloca[xformers-attention]==1.3.6"` then `pip install "lgatr[xformers-attention] @ git+https://github.com/heidelberg-hepml/lgatr@<2.0-tag-or-sha>"` (or the PyPI release if it exists — preferred). Record the exact version/sha in the decision log (§5).
2. **Phase 1a — re-verify the §2 inventory (~15 min, non-negotiable):**
   - Read lgatr's own v1→v2 migration/differences doc (added in `35f052c`).
   - `python -c "from lgatr.layers import SlimMLP, SlimDropout, SlimLinear, SlimRMSNorm, SlimSelfAttention"` and `python -c "from lgatr import LGATr, LGATrSlim, embed_vector, extract_scalar, get_num_spurions, get_spurions"`.
   - Confirm `SelfAttentionConfig(increase_hidden_channels=2)` raises `TypeError` (the stale-key trap, §3).
   - Confirm `import lloca.equivectors.lgatr` still imports (private-symbol coupling).
   - Confirm `embed_vector(torch.tensor([1.,2,3,4])).nonzero()` is slots 1–4 (layout).
   - Diff `SlimMLP`/`SlimSelfAttention`/`SlimLinear` signatures against the calls in `lorentznetlgatrslimgraphgps.py`.
   - Verify the S4 depth mapping: instantiate a `GeoMLP` both ways and compare layer counts.
3. Only now lift the `<2.0.0` pin in `requirements.txt` (M7).

### Phase 2 — mechanical port

Work through M1–M7 as a literal checklist; one commit per row (or one for code, one for yamls) so any gate failure bisects instantly.

### Phase 3 — parity pins

Apply S1/S2 pins (`nonlinearity_v`, `norm_elementwise_affine`/`elementwise_affine=False`); leave S3 at `sparse_gp=True` for runtime. Every pin gets a config comment naming this document and the 1.4.4 behavior it preserves.

### Phase 4 — gates (all must pass; A–F on CPU, G–H on cluster)

| Gate | What | Pass criterion |
|---|---|---|
| A | Composition/instantiation: `test_jc_wiring.py` (8/8) + parity script instantiates all fixture configs | no exceptions; channel asserts hold |
| B | Param manifests vs fixtures | total counts and shape multisets **identical** — zero tolerance |
| C | Forward parity vs fixtures, two tiers | **Tier 1** (parity pins + `sparse_gp=False` for full-LGATr models; slim has no geometric product, so no flag needed): fp64 max-abs-diff < 1e-12. **Tier 2** (dev defaults, `sparse_gp=True`): fp64 max-abs-diff < 1e-8 — failures beyond that are real deltas, not reassociation roundoff |
| D | Full existing suite | 64/64 (32 equivariance + 24 invariance + 8 jc_wiring) |
| E | Identity-frames bit-exactness spot check | hybrid with identity frames ≡ plain backbone, bit-identical (both sides share one lgatr, so this proves internal consistency survived) |
| F | Blade-table equivalence | audit script comparing the hybrid's local `CliffordAlgebra` against dev's `lgatr.primitives.bilinear._load_geometric_product_tensor` — agreement at 1.4.4 levels (≤2e-6 fp32). Expected pass (§2.1) but proven, not assumed |
| G | Training parity (cluster) | fixed-seed short runs (e.g. 1k iters, toptagging quick) for `tag_slim` + `tag_lgatr` on both versions; final train loss within seed-to-seed noise band (2–3 seeds). Curves diverge point-wise after enough steps under S3 — that is expected; this gate catches *gross* regressions only |
| H | Throughput report (cluster) | not pass/fail: measure it/s for `tag_lgatr` with dev `compile=True/False` and `tag_slim` (which already compiles on 1.4.4). This quantifies the carrot; if the number is small, record it and stop advertising the upgrade as a speedup |

### Phase 5 — cleanup, decision log, rollback

- Sweep stale comments that reference 1.4.4 internals (e.g. `tag_LorentzNetLGATrSlimGraphGPS.yaml` "lgatr lib default is 2" mlp_ratio note; the attn_dropout comments describing lgatr's sdpa path — re-verify wording against dev).
- Append a **decision log** section to this file: lgatr version installed, date, gates run with numbers, and an explicit entry per S-item: *pinned to 1.4.4 behavior* or *adopted v2 behavior because…*.
- Upstream PR opportunity: the private `_load_inner_product_factors` accessor issue for lloca.
- **Rollback:** revert the port commits, reinstall `lgatr==1.4.4` (restore the `<2.0.0` pin) — the fixtures remain valid either way. Rollback triggers: any Gate B/C tier-1 failure that can't be attributed to a documented S-item within a day.

## 5. Hitches and catches (consolidated)

- **H1 — dev is moving daily.** The slim module changed location *between two fetches on the day this plan was written*, and the release date is being finalized. Never port from §2.2 without running Phase 1a. Prefer shallow public imports (`lgatr.layers`, top-level `lgatr`) — they survived today's churn; deep paths didn't.
- **H2 — the dangerous deltas are silent (S1–S4).** All 64 existing tests pass through every one of them. Only Gates B/C catch them. This is why fixtures are recorded *first*.
- **H3 — shared `SlimRMSNorm` instance** (`lorentznetlgatrslimgraphgps.py:91`): with dev's affine default it would either crash (missing channel args) or, if naively given channels, silently tie learnable gains across call sites. `elementwise_affine=False` is structurally required, not a style choice.
- **H4 — stale-key silence.** If config casting ever drops unknown keys instead of raising, a missed rename silently *narrows* the model. Phase 1a proves it raises; Gate B backstops.
- **H5 — sparse_gp is not bit-identical** (upstream says so). Tier-2 tolerances, never mix lgatr versions within one results table, and all bit-exactness claims in the repo (identity-frames ≡ plain) remain valid because both sides of each claim run the same lgatr.
- **H6 — lloca's private-API import.** Survives at `1.3.6` + dev@`e8ba34d`; any lloca or lgatr bump re-triggers the check in Phase 1a.
- **H7 — checkpoint keys change.** Migrate before the campaign; then there is nothing to migrate.
- **H8 — s-channel `None` → `0`** (M6): yaml `null`s are placeholders filled by `init_physics`, but any code path handing a literal `None` to a dev net/config is a break. Grep + Gate A.
- **H9 — CPU-only web containers.** All parity gates are deliberately CPU-runnable in fp64; do not add CUDA-dependent assertions. The container's xformers is ABI-mismatched (`--no-deps` install) — only its CPU-safe pieces (`BlockDiagonalMask`) may be exercised, exactly as the existing suite already does.
- **H10 — reproducibility/install.** Git-pinned installs leak into the `.sif` build recipe (network + git at build time). Strongly prefer executing this runbook after the PyPI 2.0 release; the `<2.0.0` pin guards until then.
- **H11 — compile expectations.** `tag_slim.yaml` already sets `compile: true` on 1.4.4, so slim gains little from v2's compile work; the new capability is compile + `warmup_caches` + `sparse_gp` for **full-LGATr** models and the framesnet equivectors. If enabling those, call `warmup_caches(device, dtype)` before compiling with `mode="reduce-overhead"` (their documented graph-partition catch), and treat it as a *post-migration* enhancement, gated by Gate H numbers — not part of the parity port.

## 6. Upstream (`heidelberg-hepml/lloca-experiments`) variant

No hybrids, no direct slim-block construction. Surface = M2/M3/M4 yaml renames across the three experiment families + `finetuneexperiment` import + the flex monkeypatch check + their own test suite. Same record→port→prove shape with their tests as Gate D. ≈1 day. Offer the fixture-script pattern upstream with the port PR.

## 7. Suggested task split (Claude Code web)

1. **Task A (this environment, 1.4.4 still installed):** implement + run Phase 0; commit script + fixtures to `dev`.
2. **Task B (fresh session, after 2.0 exists on PyPI):** Phase 1 → 1a → 2 → 3, then Gates A–F. Push; open PR only when A–F are green.
3. **Task C (cluster/user):** Gates G–H, `.sif` rebuild, Phase 5 decision log, merge.

## Appendix A — evidence log (2026-07-29)

Diffed installed `lgatr==1.4.4` against dev@`e8ba34d` raw sources for: `__init__`, `nets/{__init__,slim,slim → layers/slim_layers,lgatr}`, `layers/{__init__,linear,dropout,layer_norm,attention/config,attention/self_attention,mlp/config,mlp/mlp}`, `interface/{vector,spurions}`, `primitives/{invariants,linear,bilinear,config,compile,attention_backends/__init__,attention_backends/flex}`, `utils/{misc,autocast,compile}`. Key confirmations: embed slots 1:5 unchanged; subgroup default equivalent (10-element basis both); slim constructor args identical except RMSNorm; `attention = flex_attention` present; backend kwarg registry matches `experiments/misc.py`; PyPI latest 1.4.4; lloca 1.3.6 imports compatible; repo greps for `compile_mode|compile_dynamic|nonlinearity_v|increase_hidden_channels|minimum_autocast_precision` as cited in §2.

## Appendix B — fixture script sketch

```python
# tests/experiments/test_lgatr_migration_parity.py  (sketch — implement in Task A)
import os, json, pathlib, pytest, torch

FIXTURE_DIR = pathlib.Path(__file__).parent.parent / "fixtures" / "lgatr144"
MODELS = [
    "tag_lgatr", "tag_slim",
    "tag_CGENNLGATrGraphTrans", "tag_CGENNLGATrGraphGPS",
    "tag_LorentzNetLGATrSlimGraphTrans", "tag_LorentzNetLGATrSlimGraphGPS",
    # + one learned-frames composition with equivectors=lgatr (LGATrVectors)
]
RECORD = os.environ.get("LGATR_PARITY") == "record"

def build(model_name):
    # hydra compose + init_physics, same machinery as test_jc_wiring.py;
    # torch.manual_seed(0) before instantiation; .double().eval()
    ...

def fixed_batch():
    # tests/experiments/utils.py generator, seeded; B=4, multiplicities
    # [1, n<knn_k, mid, large]; float64
    ...

@pytest.mark.parametrize("name", MODELS)
def test_parity(name):
    model = build(name)
    manifest = {
        "total": sum(p.numel() for p in model.parameters()),
        "shapes": sorted(str(tuple(p.shape)) for p in model.parameters()),
    }
    with torch.no_grad():
        out = model(fixed_batch())
    path = FIXTURE_DIR / f"{name}.pt"
    if RECORD:
        torch.save({"manifest": manifest, "out": out}, path)
        return
    if not path.exists():
        pytest.skip("no 1.4.4 fixtures recorded")          # fresh clones / CI
    ref = torch.load(path)
    assert manifest == ref["manifest"]                     # Gate B: zero tolerance
    tol = 1e-12 if os.environ.get("LGATR_PARITY_TIER") == "1" else 1e-8
    assert (out - ref["out"]).abs().max() < tol            # Gate C
```
