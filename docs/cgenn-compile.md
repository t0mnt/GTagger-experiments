# CGENN torch.compile support — workflow

**Status: PLANNED — no work done. Companion to `docs/lgatr2-migration.md` (same record→change→prove discipline), scoped to the regular CGENN baseline.**

- **Independent of the lgatr migration** and can run before, after, or in parallel with it: the `experiments/baselines/cgenn/` package imports no lgatr (verified); `CGENNWrapper`'s only lgatr symbol is `embed_vector` (interface-stable across 1.4.4/2.0.0) and the wrapper stays eager anyway.
- Scope: **Stage 1 = the baseline** (`experiments/baselines/cgenn/` + a `compile` knob on `CGENNWrapper`/`tag_cgenn.yaml`). Stage 2 (optional) = `tag_lorentznet` by the identical recipe. **Not here:** the hybrids' CGENN branch (whole-block compile couples to lgatr 2.0's compiled attention — post-migration task; note both hybrids share ONE stack via `CGENNLGATrGraphTransHybrid.py`, import-verified), the sparse-GP rewrite (changes numerics at tolerance level → its own workflow, only if profiling justifies), and the non-equivariant family (out of scope for THIS task per the migration runbook §8 — deferred, not rejected: no forcing event, fused-kernel profiles, uniform-or-disclosed walltime rule).

## 1. Verification regimes — where "bit-identical" is true and where it is not

The premise "forward results are bit-identical, unlike lgatr 2.0" is right in the two places that matter and wrong in one, and the gates are shaped accordingly:

- **R1 — eager after the change vs eager before (BIT, `torch.equal`).** Everything Stage 1 does to the model code is *pure data movement* (index construction replacing boolean-mask scatter and `repeat_interleave` — same values, same order, no arithmetic), and the knob defaults off. So eager outputs must be **bit-identical** to pre-change fixtures, fp32 and fp64. This is a *stronger* gate than anything the lgatr migration can have, and it is the core gate here.
- **R2 — compiled vs eager (TOL, tolerance).** NOT guaranteed bit-identical: Inductor fuses and reorders reductions (BatchNorm batch stats, `index_add_` segment sums, matmul tiling). Bar: relative ≤ 1e-10 in fp64 on CPU (≤ 1e-5 fp32 on GPU in Stage β). Bit-equality, when it happens on small CPU graphs, is recorded as a bonus — never required.
- **R3 — compiled determinism (DET, `torch.equal`).** The compiled path against *itself*, run twice on the same inputs, must be bit-identical — separates compiler nondeterminism from real deltas before anyone chases ghosts.

## 2. Inventory (evidence)

**Compile-clean already** (no `.item()`, no `nonzero`, no data-dependent Python branching in `CGENN.forward`/`CGLayer.forward`): einsums (`fcgp.py:72,75`, `gp.py:60,64`, `linear.py:48,52`, `cliffordalgebra.py:53`), `index_add_`-based `unsorted_segment_{sum,mean}`, BatchNorm1d, sigmoid gating, masked mean readout.

**Two mechanical rewrite families needed** (both pure data movement → gated by BIT):

| Site | Pattern | Replacement |
|---|---|---|
| `fcgp.py:54`, `gp.py:42` | `weight[:, :, product_paths] = self.weight` — boolean-mask assignment each forward | precomputed integer indices at init + `index_put`/flat `index_copy` (same values, same order) |
| `cgenn.py:237,256`; `fcgp.py:57-59`; `gp.py:45-47` | `repeat_interleave(x, subspaces_tensor, dim=…)` — tensor-valued repeats (dynamo must read buffer values to shape the output) | precomputed gather index registered at init + `index_select` |

**Wrapper stays eager, permanently**: `pair.nonzero` (data-dependent E), the `fourmomenta[~is_spurion] /= 20` boolean scatter, `to_dense_batch`. Compile the **net only**, `torch.compile(self.net, dynamic=True)` — N = B·P and the fully-connected E = Σnᵢ(nᵢ−1) vary per batch, so `dynamic=True` is mandatory (RECOMP gate checks it works). Root `fullgraph=False`; flip `fullgraph=True` only on proven break-free leaves later (migration §8 pattern).

**Untouchable**: the padded-mean readout quirk (`cgenn.py:382-383`) — documented official-repo parity; any "fix" there is a modeling change, not a compile change.

## 3. Gates

| Gate | What | Pass |
|---|---|---|
| BIT | eager forward (knob off and on-but-uncompiled paths are the same code) vs pre-change fixtures | `torch.equal`, fp32 AND fp64 — zero tolerance; a BIT failure means a rewrite wasn't pure data movement: fix the rewrite, never relax the gate |
| TOL | compiled net vs eager net, same batch | rel ≤ 1e-10 (fp64 CPU); Stage β adds fp32 GPU ≤ 1e-5 |
| DET | compiled net twice | `torch.equal` |
| BREAKS | `torch._dynamo.explain` over the net | 0 graph breaks; the explain report is committed next to the fixtures |
| RECOMP | forward sweep over (B, P) ∈ {(2,17),(4,64),(8,128),(3,200)} with `dynamic=True` | ≤ 2 compilations total (first + one dynamic re-trace), not per-shape |
| SUITE | full existing test suite (knob off) + one compiled CPU smoke test | 64/64 + smoke green |
| β-PERF | cluster: it/s eager vs compiled (quick config + full-size batch), fp32/bf16-off | numbers published in this doc's log whatever they say — compile is only worth shipping if this table says so |

Fixtures are trivial compared to the lgatr migration: eager outputs on a fixed seeded batch (fp32+fp64) + sha256, recorded **at current HEAD before any edit** — no state_dict, no transplant, no env swap; the whole Stage 1 fits in one session.

## 4. Task split, prompts, operator gates

### Task α — rewrites + knob + gates (CPU web session)

```text
On branch dev, execute Stage 1 of docs/cgenn-compile.md.
0. FIRST COMMIT, before any code edit: record fixtures — eager outputs of the tag_cgenn net on
   the fixed seeded batch (fp32 and fp64) under tests/fixtures/cgenn_compile/ with sha256s,
   plus the recording script/test (record/check modes, check skips when fixtures absent).
1. Mechanical rewrites per §2 (bool-mask scatter -> precomputed integer indices; tensor-repeats
   repeat_interleave -> precomputed gather). No arithmetic change of any kind.
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

**Operator gate α→β** (~5 min): (1) first commit is fixtures-only (git log order proves record-before-edit); (2) diff scope matches the constraint list exactly; (3) the BIT gate still reads `torch.equal` (not `allclose`) and TOL/DET/BREAKS/RECOMP numbers are pasted; (4) CI green.

### Task β — cluster numbers (user or GPU session)

```text
On branch dev (Task α merged and reviewed): run the β-PERF table for docs/cgenn-compile.md —
tag_cgenn it/s eager vs compiled on the quick config and one full-size batch shape, plus the
fp32 GPU TOL spot check (rel <= 1e-5). Publish the table in the doc's log section whatever it
says. If compiled training is adopted for the campaign, add the walltime-disclosure note
(migration runbook §8: uniform-or-disclosed) to the aggregate-table docs.
```

**Operator gate β→adopt**: adopt `compile: true` for the campaign only if β-PERF shows a real win at the training batch size; either way the numbers stay in the log, and the time-column disclosure rule applies if adoption is non-uniform across model families.

## 5. Extensions and non-goals

- **Stage 2 — `tag_lorentznet`**: same recipe verbatim (fixtures → BIT/TOL/DET/BREAKS/RECOMP/SUITE → β-PERF); the migration runbook §8 already holds the readiness notes (LGEB stack is compile-clean as-is; compile the net, not the wrapper; `dynamic=True`; nothing to rewrite — so Stage 2 skips §2 entirely and is mostly gate-running).
- **Hybrids**: after the lgatr migration only — v2's compiled attention removes the graph breaks that currently make whole-block compile pointless; the CGENN-branch rewrites then serve both hybrids at once through the shared `CGENNLGATrGraphTransHybrid.py` stack.
- **Sparse-GP**: separate task with a *tolerance* workflow (it reorders arithmetic — BIT can never gate it); do it only if the FLOPs/profiler comparison (migration §8 "profile first") shows the Cayley einsums dominate.
- **Non-equivariant family**: out, per migration §8 — revisit only on profiler evidence.

Effort: Task α ≈ one focused session; Task β ≈ one cluster hour. Log section (append results below):

## Log

*(empty — no work done)*
