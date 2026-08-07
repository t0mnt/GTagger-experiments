# Execution playbook — session prompts and operator checks

**This file is the canonical operational sequence.** One step = one fresh Claude Code web session (or a cluster shell where marked). Paste the prompt verbatim; between steps run the checks yourself — a step is not done until its checks pass. Rationale lives in the runbooks (`docs/lgatr2-migration.md`, `docs/cgenn-compile.md`); if a prompt here and a runbook disagree, fix the drift in a sync commit before running anything.

**Order of play** (later steps assume earlier ones merged):

| # | Step | Where | Blocking? |
|---|---|---|---|
| 1 | **L-A** lgatr fixtures on 1.4.4 | web | FIRST — perishable (needs a 1.4.4 env) |
| 2 | **L-B** lgatr port + Gates A–F | web | after L-A checks |
| 3 | **L-C** posture flip + cluster gates + close-out | cluster | after L-B checks; ends in PR dev→main |
| 4 | **C-α** CGENN compile, Stage 1 | web | independent of L-B/L-C; run after L-A to keep review load serial |
| 5 | **C-β** CGENN cluster numbers | cluster | after C-α checks |
| 6 | **LN** LorentzNet compile, Stage 2 | web (+cluster β) | optional, cheap; after C-α |
| 7 | **NE** ParT / ParticleNet / plain-transformer compile | web (+cluster β) | policy: compile where it works; after C-α |
| 8 | **HY** hybrid compile measurement + uniform-adoption decision | cluster | POST-CAMPAIGN only |
| 9 | **SG** sparse-GP rewrite | web | GATED on profile evidence from step 8's measurements |
| 10 | **UP** upstream `lloca-experiments` port | web | optional, any time after L-C |

Before step 1, record the starting point: `PRE=$(git rev-parse origin/dev)` — every check below diffs against the previous step's saved tip.

---

## Step 1 — L-A: lgatr fixtures on 1.4.4

```text
On branch dev, execute Phase -1 and then Phase 0 of docs/lgatr2-migration.md exactly.
Precondition: `python -c "import lgatr; print(lgatr.__version__)"` must print 1.4.4. If the
session auto-installed dev's requirements (lgatr==2.0.0), first run
`pip install "lgatr[xformers-attention]==1.4.4" "lloca[xformers-attention]==1.3.6"` and re-verify.
Phase -1 FIRST, and do it by RUNNING 2.0.0, not by reading its docs: side-load the wheel
(pip download + unzip + sys.path.insert -- this changes nothing on disk, so the 1.4.4
environment Phase 0 needs stays intact) and, as a committed script under tests/:
  a. instantiate every lgatr construction this repo performs, VERBATIM from the call sites,
     and record which raise on 2.0.0;
  b. diff named_parameters() name-and-shape between 1.4.4 and 2.0.0 for the reduced configs;
  c. report every M-row and S-row of section 2 as CONFIRMED / WRONG / MISSING against what you
     observed. Read v1_to_v2.rst and the CHANGELOG first -- they are accurate and they tell you
     where to look -- but they are not COMPLETE evidence: the doc covers renames only, and the
     runbook's own corrections (M7 wrong, M11 missing, S5 mis-scoped) were all found by running
     2.0.0, not by reading about it. Documentation orients the search; the wheel settles it.
If any row comes back WRONG or MISSING, STOP and report before recording fixtures: Phase -1
exists to change what Phase 0 records, and Phase 0 is the perishable step.
Deliverables, committed to dev:
0. tests/experiments/test_lgatr_v2_inventory.py (the Phase -1 script) + its output pasted in
   your report, row by row.
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

**Your checks before step 2:**
0. Read the Phase -1 row-by-row report FIRST. If any section-2 row is reported WRONG or MISSING
   and the session recorded fixtures anyway, reject: the fixtures may encode the wrong
   normalization and Phase 0 cannot be re-run once the environment moves.
1. Check scope via `git fetch && git diff --stat $PRE..origin/dev` — every path under `tests/`; any `experiments/` or `config/` line = reject the task.
2. Check fixture weight via `du -sh tests/fixtures/lgatr144/` — MBs, not tens of MBs.
3. Ensure gate integrity by `grep -n '1\.4\.4\|1e-10\|1e-8\|1e-6' tests/experiments/test_lgatr_migration_parity.py` — the version hard-assert and all three bars present — and by reading the waiver function once: it must *compute* the allowed set from the recorded state_dict, not list keys.
4. Ensure determinism was proven by finding the two record runs + matching hashes in the session report.
5. Check the suite via dev CI (or `pytest tests/ -q`) — 64/64 plus the new file skipping cleanly.
6. Save the tip: `TASKA=$(git rev-parse origin/dev)`.

## Step 2 — L-B: lgatr port + Gates A–F on 2.0.0

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
   subsection of the runbook's decision log.
Hard constraints: never loosen a tolerance, widen a waiver rule, delete an assertion, or drop
a model to make a gate pass — bars and waivers change only via a new documented S-item, which
requires stopping and reporting first. Stop at the FIRST gate failure and report the
first-divergence block/tensor. Do not run Gates G/H, do not touch the .sif, do not open a PR.
```

**Your checks before step 3** (the critical review of the whole migration):
1. Check parity-test immutability via `git diff $TASKA..origin/dev -- tests/experiments/test_lgatr_migration_parity.py` — allowed: KEY_MAP content, rule-derived waiver additions each tied to an S-item, v2-only imports. **Any edit to a tolerance, waiver derivation, or assertion = reject the task, never the gate.**
2. Check config discipline via `git diff $TASKA..origin/dev -- config/` — exactly the M2/M3/M4/M9 renames, zero value-level changes.
3. Check gate numbers via `sed -n '/Gate results/,/^## /p' docs/lgatr2-migration.md` — tier-1 deviations ~1e-12, far under the 1e-10 bar; a pass at 3e-11 is unexplained drift → investigate, don't accept.
4. Ensure the Phase-1a report exists in the session output and either matches runbook §2 or documents what changed upstream.
5. Check the env pin via the pasted freeze: `lgatr==2.0.0`, `lloca==1.3.6`.
6. Save the tip: `TASKB=$(git rev-parse origin/dev)`.

## Step 3 — L-C: posture flip, cluster gates, close-out (cluster)

```text
On branch dev (Gates A-F green, operator-reviewed):
1. FIRST, before any training: the posture-flip commit (Phase 3): re-record production
   manifests at v2 defaults; apply the H15 optimizer exemption in BOTH grouping paths with the
   Gate-B no-decay assertion for *.weight_mv/*.weight_s; add the §2.4 methods-sentence TODO.
2. Gate G: fixed-seed ~1k-iter quick runs (tag_slim, tag_lgatr) on 1.4.4 and 2.0.0, 2-3 seeds
   each; report final-loss bands side by side.
3. Gate H: it/s table — tag_lgatr compile on/off and tag_slim, both versions; publish the
   numbers in the decision log whatever they say (ignore the first timing estimate — compile
   warmup lands in the early iterations).
4. Phase 5: stale-comment sweep, relax the pin to >=2.0.0,<3, complete the decision log
   (one entry per S-item).
5. Open the PR dev -> main only after 1-4 are in the log.
```

**Your checks before merging:**
1. Ensure ordering by `git log --oneline $TASKB..origin/dev | tail -3` — the posture-flip commit precedes any training-run commit.
2. Check the H15 fix landed in both paths via `grep -n 'weight_mv\|EquiLayerNorm' experiments/base_experiment.py experiments/tagging/experiment.py` — gain params routed to `weight_decay=0` groups, plus the Gate-B assertion.
3. Judge Gate G yourself: the two versions' final-loss bands overlap within seed noise (this is your physics call, not the session's).
4. Ensure Gate H's table is in the decision log even if unflattering.
5. Check the pin relaxed only in the Phase-5 commit (`git log -p -- requirements.txt | head -30`).
6. Ensure the PR diff equals the sum of the diffs you reviewed at steps 1→2 and 2→3 — nothing new may appear at PR time.

## Step 4 — C-α: CGENN compile, Stage 1 (web)

```text
On branch dev, execute Stage 1 of docs/cgenn-compile.md.
0. FIRST COMMIT, before any code edit: record fixtures — eager outputs of the tag_cgenn net on
   the fixed seeded batch (fp32 and fp64) under tests/fixtures/cgenn_compile/ with sha256s,
   plus the recording script/test (record/check modes, check skips when fixtures absent).
1. Mechanical rewrites, all three, in this order (§2 + the einsum note above it):
   a. the geometric-product `einsum` -> outer-product + matmul in lgatr 2.0's shape
      (`M[(i,k), j] = cayley[i,j,k]`, precomputed at init). Measured 76.1 ms -> 14.6 ms at
      realistic size and BIT-identical there, and the GP is ~46% of runtime -- do it FIRST,
      it is the largest single lever. If it fails BIT on the real fixtures it becomes a TOL
      item: stop and report, do not relax the gate.
   b. bool-mask scatter -> precomputed integer indices;
   c. tensor-valued repeat_interleave -> precomputed gather.
   (b) and (c) are pure data movement -- together they are the 38% of runtime the profile
   attributes to `aten::copy_`. No arithmetic change of any kind in any of the three.
2. `compile: false` knob in config/model/tag_cgenn.yaml + CGENNWrapper: when true,
   torch.compile(self.net, dynamic=True); wrapper stays eager; skip compiling on CPU test runs
   except the dedicated smoke test.
3. Run gates BIT / TOL / DET / BREAKS / RECOMP / SUITE (§3); paste every number and the
   dynamo-explain summary into your report; commit the explain report.
Constraints: file scope is experiments/baselines/cgenn/, experiments/tagging/wrappers.py (knob
only), config/model/tag_cgenn.yaml, tests/. Do not touch cgenn.py:382-383 (padded-mean quirk).
BIT is torch.equal — never replace it with allclose; a BIT failure is a rewrite bug, stop and
report. Do not run cluster benchmarks.
```

**Your checks before step 5:**
1. Ensure record-before-edit by `git log --reverse --stat $LASTTIP..origin/dev | head -20` — the first commit touches fixtures/tests only.
2. Check the BIT gate stayed bit-level via `grep -n 'torch.equal\|allclose' tests/experiments/test_cgenn_compile*.py` — `torch.equal` in BIT/DET, `allclose` only in TOL.
3. Check scope via `git diff --stat $LASTTIP..origin/dev` — only the four permitted areas.
4. Ensure the dynamo-explain report is committed and says 0 graph breaks; RECOMP ≤ 2 compilations across the shape sweep.
5. CI green.

## Step 5 — C-β: CGENN cluster numbers (cluster)

```text
On branch dev (C-α merged and reviewed): run the β-PERF table for docs/cgenn-compile.md —
tag_cgenn it/s eager vs compiled on the quick config and one full-size batch shape, plus the
fp32 GPU TOL spot check (rel <= 1e-5). Ignore the first timing estimate (compile warmup).
Publish the table in the doc's Log section whatever it says. Per the table-wide compile
policy: if adopted, add the per-row compile footnote to the walltime column docs; FLOPs
remains the efficiency-claim column.
```

**Your checks:** table present in the Log with both shapes; adoption decision recorded; if adopted, the footnote task is done (check `utils/aggregate_table.py` docs/legend mention).

## Step 6 — LN: LorentzNet compile, Stage 2 (web, optional)

```text
On branch dev, execute Stage 2 of docs/cgenn-compile.md: tag_lorentznet torch.compile support
by the Stage-1 recipe.
0. FIRST COMMIT: record eager fixtures for the tag_lorentznet net (fp32+fp64, sha256) under
   tests/fixtures/lorentznet_compile/ (record/check modes).
1. There are NO §2-style rewrites for LorentzNet (migration runbook §8: the LGEB stack is
   compile-clean as-is). If you find you need one, STOP and report instead of rewriting.
2. compile knob in tag_lorentznet.yaml + its wrapper, mirroring CGENN's: net compiled with
   dynamic=True; the wrapper (get_edge_index_from_ptr) stays eager.
3. Gates BIT / TOL / DET / BREAKS / RECOMP / SUITE; paste numbers; commit the explain report.
Constraints: scope is lorentznet.py, wrappers.py (knob only), tag_lorentznet.yaml, tests/.
BIT is torch.equal. No cluster runs.
```

**Your checks:** same four as C-α (record-first, `torch.equal` intact, scope, explain report), then a C-β-style cluster run for its numbers.

## Step 7 — NE: non-equivariant baselines compile (web, policy: compile where it works)

```text
On branch dev, extend the docs/cgenn-compile.md recipe to the non-equivariant baselines per
its "Table-wide compile policy": tag_particlenet, the ParT baseline, and the plain
transformer baseline. GT/GPS hybrids are OUT of scope here (post-campaign step).
0. FIRST COMMIT per model: eager fixtures (fp32+fp64, sha256, record/check modes).
1. Use weaver-core's upstream torch.compile support for ParticleNet/ParT as the reference for
   known-good graph-break handling; document every place this repo's port deviates and why.
   Known hazard: ParT's PairEmbed with dynamic sequence lengths (lloca precedent:
   @torch.compiler.disable on that module is an acceptable resolution, with a comment).
2. compile knob per model config, default false; wrappers and data-dependent stages (dynamic
   kNN rebuilds, to_dense_batch) stay eager.
3. Gates per model: BIT / TOL / DET / BREAKS / RECOMP / SUITE. A model that cannot pass
   BREAKS cleanly STAYS EAGER and gets a reasons entry in the cgenn-compile.md Log — that is
   the policy working, not a failure to fix by force.
Constraints: no math changes anywhere (BIT is torch.equal); stop and report rather than
force any model; no cluster runs.
```

**Your checks:** per-model gate numbers or a stays-eager reason in the Log — no model silently skipped; BIT untouched; scope = the three model files + their configs + tests; then a C-β-style cluster pass for the compiled ones.

## Step 8 — HY: hybrid compile + uniform-adoption decision (cluster, POST-CAMPAIGN)

```text
Post-campaign only, on a fresh branch off main: measure torch.compile on the GT/GPS hybrids,
ParticleNetParTGraphTrans and ParticleNetParTGraphGPS first (the graph breaks come from the
wrapper: per-batch kNN rebuild, PyG scatter, LLoCa transport — not the backbones). Same
gates (BIT/TOL/DET/BREAKS/RECOMP/SUITE), then throughput. Report per-model: compiles cleanly
/ stays eager, and the it/s delta. Then apply the migration runbook §8 decision rule: if the
whole table compiles, propose uniform compilation (retiring the per-row walltime footnote);
otherwise keep mixed-and-disclosed.
```

**Your checks:** the uniform-vs-mixed recommendation is explicit and follows from the per-model results; no campaign-era config changed retroactively.

## Step 9 — SG: sparse-GP rewrite (web, GATED)

Only if step 8's profiling (or the FLOPs test comparison, migration runbook §8 "profile first") shows the Cayley einsums dominate CGENN time. This is a *tolerance* workflow (arithmetic reorder — BIT can never gate it): fixtures → rewrite dense einsums to precomputed (indices, signs) gathers with input-saving backward → TOL at 1e-10 fp64 + gradcheck + SUITE → throughput. Two copies to treat, import-verified: the hybrids' shared stack in `CGENNLGATrGraphTransHybrid.py` (serves both hybrids) and the baseline `experiments/baselines/cgenn/` package. Write the prompt from the C-α template with BIT→TOL swapped and the gradcheck added.

## Step 10 — UP: upstream `lloca-experiments` port (web, optional)

```text
On a branch of heidelberg-hepml/lloca-experiments (or a fork PR branch): port lgatr 1.4.4 ->
2.0.0 per the §6 variant of GTagger-experiments' docs/lgatr2-migration.md: record fixtures on
1.4.4 first (their configs; same qkv-bias normalization and GLU-rescale compensations), then
the M2/M3/M4/M9 renames + M10 at their finetuneexperiment splices, parity pins as build
overrides, their test suite as Gate D, transplant parity as Gate C. Offer the fixture-script
pattern in the PR description. Do not change their defaults' posture — that is their call;
port at v2 defaults and say so.
```

**Your checks:** their CI green; the PR description separates mechanical renames from behavior notes (S-items) so upstream can judge the posture themselves.
