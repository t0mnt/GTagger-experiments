# CGENN torch.compile support — workflow

**Status: EXECUTED — Stages 1–4 complete, all CPU gates green (see the dated log
entries from "Stage-1 execution" onward; β-PERF + Gates G/H remain cluster-side).
Companion to `docs/lgatr2-migration.md` (same record→change→prove discipline). The
planning sections below are kept verbatim as the decision record; where execution
superseded a plan-era sentence, the log entry is authoritative.**

- **Independent of the lgatr migration** and can run before, after, or in parallel with it: the `experiments/baselines/cgenn/` package imports no lgatr (verified); `CGENNWrapper`'s only lgatr symbol is `embed_vector` (interface-stable across 1.4.4/2.0.0) and the wrapper stays eager anyway.
- Scope: **Stage 1 = the baseline** (`experiments/baselines/cgenn/` + a `compile` knob on `CGENNWrapper`/`tag_cgenn.yaml`). Stage 2 (optional) = `tag_lorentznet` by the identical recipe. **Not here:** the hybrids' CGENN branch (whole-block compile couples to lgatr 2.0's compiled attention — post-migration task; note both hybrids share ONE stack via `CGENNLGATrGraphTransHybrid.py`, import-verified), the sparse-GP rewrite (changes numerics at tolerance level → its own workflow; **the profiling now justifies it — see the note below**), and the non-equivariant family (out of scope for THIS task per the migration runbook §8 — deferred, not rejected: no forcing event, fused-kernel profiles, uniform-or-disclosed walltime rule).

### Scope policy: if the baseline gets compile, the GT hybrids get it too

Stage 1 deliberately starts at the `tag_cgenn` baseline, but **shipping compile there and not on
the eight GT hybrids is not a stable end state.** The hybrids are the study's primary rows; a
walltime table where the reference row is compiled and the rows under test are not is a table
about compilers, not architectures. So the ordering is a sequencing decision, not a scope
decision:

1. **Stage 1** — `tag_cgenn` baseline (this document). Proves the rewrite recipe and the gates.
2. **Stage 3, post-lgatr-2.0** — the CGENN branch of the two CGENN hybrids. Deferred only because
   whole-block compile couples to v2's compiled attention; both hybrids share ONE stack
   (`CGENNLGATrGraphTransHybrid.py`, import-verified), so this is one port, not two.
3. **Stage 4** — the remaining six hybrids. Materially easier than either of the above: the
   Plain / ParticleNet-ParT / LorentzNet-slim stacks are dense, static-shaped and free of the
   data-dependent control flow that forced the §2 rewrites. Upstream weaver already ships
   compile support for ParT and ParticleNet, so the work there is mostly `dynamic=True` and
   confirming the RECOMP gate, not rewriting ops.
   *(Superseded by execution: the LorentzNet-slim pair landed with the 2026-08-07 entry, and
   the executed Stage 4 became the full non-equivariant family of eight — ParT, ParticleNet,
   transformer, Plain pair, PN-ParT pair, with MIParT descoped by operator decision. "Mostly
   dynamic=True" also proved optimistic: see the Stage-4 log entry's twin surgery.)*

The same `dynamic=True` requirement applies at every stage and for the same reason (§2): `N`
and `E` vary per batch. The uniform-or-disclosed walltime rule from the migration runbook §8
governs what may be published from a partially-compiled table in the meantime.

### Profile: the §2 rewrites are the biggest single item, ahead of the geometric product

CPU profile (4 threads, B=4, P=64, `torch.profiler`, self time). Not GPU wall-clock — but the
op MIX and the call counts are what matter here, and both transfer:

| op | CGENN-GraphTrans | CGENN-GraphGPS | calls (GPS) |
|---|---|---|---|
| `aten::copy_` | 38.5% | **38.4%** | **2071** |
| `aten::mul` | 23.0% | 24.2% | 496 |
| `aten::bmm` | 18.9% | 21.7% | 460 |
| `aten::einsum` | 4.8% | 5.1% | 336 |
| forward | 812 ms | 4082 ms | |

**Nearly 40% of the time is `copy_` — data movement, not arithmetic** — and it is precisely the two
patterns §2 already lists: the `weight[:, :, product_paths] = self.weight` boolean-mask assignment
rebuilt every forward, and the tensor-valued `repeat_interleave`. Those rewrites are pure data
movement, so they are gated by **BIT** (zero tolerance, no numerics review), which makes them both
the largest and the safest item. Do them first; they were already first in §2, and this is why.

2071 `copy_` calls in one forward at B=4 also says the GPU path is launch-bound, so compile's
fusion should pay more here than on any other model in the repo.

The actual geometric-product math (`mul` + `bmm` ≈ 46% **self-time**, that profile's shape and
impl — not a stable property; see the caveat at the matmul-mapping section) is what the sparse-GP
rewrite below targets. The two are independent and multiply.

### Upstream's own numbers (arXiv:2608.02735, *Virtues and Vices of Equivariant Transformers*)

Favaro, Plehn, Qu & Spinner benchmark exactly these optimizations on a **H100**, JetClass, 1M
iterations at batchsize 512. Their Table 2 — the `→` is torch.compile + the sparse geometric
product + micro-optimizations, all without AMP, all but ParT using sparse jet representations:

| architecture | acc | AUC | time | FLOPs | memory |
|---|---|---|---|---|---|
| Baseline transformer | 0.855 | 0.9867 | 15h → **9h** | 210M | 2.3G |
| ParT | 0.861 | 0.9878 | 33h → **19h** | 211M | 13.3G → 7.2G |
| L-GATr sparse | 0.865 | 0.9885 | 166h → **63h** | 2060M → 352M | 19.0G → 16.8G |
| L-GATr dense | 0.865 | 0.9885 | 166h → **57h** | 2060M → 1999M | 19.0G → 14.3G |
| L-GATr-slim | 0.866 | 0.9885 | 27h → **16h** | 329M | 8.1G |
| LLoCa transformer | 0.864 | 0.9882 | 28h → **12h** | 219M | 4.1G |

"The training time of all networks is reduced by values between 40% and 70%." Four things this
settles for us:

1. **Sparse vs dense is per-operation, not per-model.** Quoting §2.2: the original dense
   implementation "is faster than a sparse implementation on GPUs because of the efficiency of
   the GEMM matrix multiplication kernels. With our new L-GATr implementation, **the sparse
   variant is faster on geometric product operations. For linear operations, the dense approach
   is still faster on a GPU** but uses fewer FLOPs and runs faster on a CPU." Hence
   `L-GATr_dense` vs `L-GATr_sparse` differ *only* in the linear layer, and dense wins on GPU
   (57h vs 63h) while sparse wins on CPU. **lgatr 2.0's defaults already encode this**:
   `PrimitivesConfig(sparse_gp=True, sparse_linear=False)`. Take them as shipped for GPU
   training; the only reason to touch either is a CPU-inference study.

2. **Sparse jet representations are worth ~2× for THEM and much less for our GNN branches.**
   Concatenating a batch's particles behind a ptr instead of zero-padding saves work in
   proportion to `r = N_mean/N_max`, but only on the ops that actually run over padded slots.
   Measured on our top-tagging multiplicities at batch 512 (`E[P_max]=110.3`, `E[n]=49.2`,
   `E[n²]=2718`):

   | op class | padded | packed | saving |
   |---|---|---|---|
   | attention (quadratic) | 12156 | 2718 | **4.47×** |
   | per-node / per-token (linear) | 110.3 | 49.2 | **2.24×** |
   | kNN edges | — | — | **1.00×** (the edge *list* is built on real nodes only) |
   | kNN edge *construction* | 12156 | 2718 | **4.47×** — `generate_edges_vectorized` forms a dense `(B, P, P)` distance matrix and masks it; only its output is packed |

   Their Table 2 rows are transformers, where attention plus per-token MLPs are the whole model —
   hence ~2× overall. **Our GNN branches are already edge-sparse**, so the same change buys them
   almost nothing: a CGENN block runs ~787 edge-ops against ~110 node-ops per jet, so node ops are
   12% of the block and packing them saves **7%**, not 2×. An earlier revision of this note claimed
   ~2× for the CGENN stack; that was wrong.

   The payoff is therefore in the *global attention stage*, which needs a block-diagonal attention
   kernel (flash-attn, xformers, or recent native PyTorch) and **cannot** be used with ParT-style
   learnable attention bias. Two of our eight rows carry exactly that bias
   (`tag_ParticleNetParTGraph{Trans,GPS}`: `bias: true`, `pair_input_dim: 4`), so the family that
   would gain most is the family excluded — while CGENN-GraphGPS, the row that actually needs the
   time, gains least because its cost is the Clifford MPNN, not attention. That inversion is the
   argument for leaving sparse representations out of scope, not the refactor size.

3. **AMP is off the table for the equivariant rows.** "For the Lorentz-equivariant
   LLoCa-Transformer, L-GATr-slim, and L-GATr, training with AMP reduces performance at a rate
   that renders it impractical, so we stick to float32." Their Table 2 runs without AMP entirely.
   This is a stronger statement than the migration runbook's S8 row, which only flagged that v2
   changed the AMP strategy — the answer is not to enable it.

4. **The LLoCa row's 28h → 12h is not a lloca-2.0 number** and must not be quoted as one. The
   caption attributes the `→` to compile + sparse GP + micro-optimizations, and notes that every
   row but ParT also uses sparse jet representations. For LLoCa-Transformer the sparse GP does
   not apply (no geometric algebra), leaving three contributors the paper does not decompose:
   `torch.compile` (**already available in lloca 1.3.6** — `compile`/`compile_mode`/
   `compile_dynamic` on `backbone/{particletransformer,transformer_v2}.py` and on
   `framesnet/equi_frames.py`), sparse jet representations (a data-path change this repo does not
   have — our wrappers `to_dense_batch`), and the 2.0 micro-optimizations proper (~10× frame-to-
   frame transforms, device-sync removal, `num_items`/`num_graphs`). So the earlier 1.3–1.4×
   FLOPs-based estimate of the *lloca-2.0 share* is not refuted by this row; the row measures
   all three together.

5. **lloca 2.0 carries an accuracy-changing default of its own.** §2.1: "We improve the original
   LLoCa implementation with a new variance-preserving rescaling in the attention mechanism",
   derived in their Appendix A — the frame transformation is rescaled to `2L/‖L‖_F` so the latent
   variance is preserved (‖L‖_F = 2γ for a boost; pure rotations need no scaling), fixing
   instability on highly boosted objects. That is the `preserve_variance` option on lloca's dev
   branch, **defaulting to true**. Same class as lgatr's S1/S2: a silent default that changes
   results, now with a paper section explaining why. Treat it as an S-row when lloca is migrated.

### Top lever, measured and bit-identical: drop `einsum` for outer-product + matmul

lgatr 2.0 removed almost all of its `einsum` calls (17 → 4; `utils/einsum.py` and the whole
bilinear einsum path are gone). Its dense geometric product is now:

```python
outer = x.unsqueeze(-1) * y.unsqueeze(-2)
return outer.flatten(-2, -1) @ gp.flatten(1, 2).T
```

CGENN's is still `torch.einsum("...i,ijk,...k->...j", x, cayley, y)`. Benchmarked at realistic
size (N = 4·64·16 edges, C = 8 channels, CPU 4 threads):

| form | time | output |
|---|---|---|
| `einsum` (CGENN today) | 76.1 ms | — |
| outer + matmul (lgatr 2.0 dense) | **14.6 ms** | `torch.allclose` exact, **max abs diff 0.0** |

**5.2× — but the "bit-identical" half of this row did NOT survive the real fixtures.** The
caveat below fired: C-α applied the recipe verbatim and BIT failed at **max |diff| 8.3e-17 in
fp64** (~1 ulp, pure reassociation) on the blades-subset path, which the micro-benchmark input
above happens to dodge. The rewrite was reverted per the stop-and-report contract and
**reclassified as a TOL item**, to ride with the sparse-GP tolerance workflow (step 9) rather
than being retried under a relaxed BIT gate. The benchmark row stands as a *timing* result; read
its "output" column as "bit-identical on this synthetic input only".

Do not re-land it as a BIT rewrite. The speed claim is unaffected; only the gate class changed.

**Status (2026-08-07, operator-directed):** implemented as the `matmul` option of the
`gp_impl` knob (`config/model/tag_cgenn.yaml`), alongside `sparse` — both TOL-class, both
gated (TOL-IMPL / TOL / DET / BREAKS / RECOMP per impl), default `einsum` so the BIT
contract is untouched. The C-β note below ("compile fuses eager einsum's marshalling, so
the eager-level rewrite is a strict subset of the compile win") stands as the reason the
*default* did not change; the knob exists because the campaign decision wants measured
eager AND compiled numbers per impl, not an argument. See the Log for the profile table.

The mapping is `M[(i, k), j] = cayley[i, j, k]`, i.e. `cayley.permute(0, 2, 1).reshape(256, 16)`,
precomputed once at init. ~~The geometric product is `mul` + `bmm` ≈ 46% of CGENN's runtime, so a
5× there is ~1.6–1.8× on the whole model, before compile and before sparse-GP.~~
**Measured, and the prediction did not hold.** The model-level table in the Log has einsum
628.9 ms → matmul 603.5 ms: **0.960×, a 4% whole-model gain, not 1.6–1.8×.** Inverting Amdahl on
that (`f(1 − 1/5.2) = 1 − 0.960`) puts the fraction the rewrite actually accelerated at **~5%**.
The two are reconcilable and the reconciliation is the useful part: 46% is the `mul` + `bmm`
**self-time** share, while the correction below measures the einsum *call tree* at ~91% with
~50% `copy_` marshalling inside it. The matmul form replaces the arithmetic and leaves the
marshalling, so it can only ever collect a slice of the 46%. **Do not quote a runtime share for
the GP as a stable property** — it moves with shape, `gp_impl`, device and whether the figure is
self-time or call-tree. Read it off the profile you are optimising.

~~**This does NOT explain the 38% `copy_`** — the einsum benchmark shows a 0.1% copy/permute share,
so einsum is not marshalling operands here. The `copy_` is the §2 patterns, independently.~~
**Corrected by the C-β CPU profile (see Log):** on the real model the `einsum` call tree is
~91% of eager runtime and the ~50% `copy_` sits *inside* it — it IS operand marshalling at
model shapes (the micro-benchmark's layout happened to dodge it). Direct proof: the §2
rewrites left `copy_` at 51.5%→49.4%, while compiling fuses it down to 8.4%.

### The sparse-GP rewrite is no longer optional-looking
*(Status 2026-08-07: implemented — `gp_impl: sparse`, quasigroup gather form; see the
knob note above and the Log. The analysis below is kept as the motivating record.)*

`CliffordAlgebra.cayley` is stored **dense** at `(16, 16, 16)` and only **256 of its 4096
entries are nonzero (6.2%)** — each product of two basis blades lands on exactly one blade. So
every geometric product in the CGENN stack, baseline and hybrid alike, does **16× the necessary
arithmetic**, and the geometric product is what CGENN spends its time on.

That is the leading term behind the measured 62.9 GFLOPs/jet for CGENN-GraphGPS (todo.md
§3a-bis) — ~30× L-GATr and ~84% of the eight-model JetClass campaign. lgatr 2.0 reached the same
conclusion independently and shipped `sparse_gp=True` as its **default**, replacing the dense
contraction with a gather-reduce.

Expect well under 16× in wall-clock — a gather-reduce has worse locality than a dense einsum that
maps onto a GEMM — but lgatr making it the default is evidence the win is real; 3–8× on the
GP-dominated parts is the plausible band. Gate it under R2 (TOL, not BIT): upstream's own
docstring says the reordering is not bit-identical, so this rewrite is the one exception to §1's
BIT rule and needs its own fixtures recorded before it lands.

**Not a lever: striding the GPS local branch.** Running the CGENN MPNN every Nth block is
near-linear in cost, but it *removes parameters and capacity* — the row stops being the model the
table claims to compare and becomes a smaller one, tested against full-depth rivals. That is a
different experiment, not a faster implementation of this one. Same objection, in weaker form, to
`k` 16→8 and `cgenn_hidden_x` 8→4: they are legitimate ABLATIONS (and belong in ablations.md), not
ways to make the headline row affordable. **Everything in this document must leave the model
identical up to its stated gate.**

## 1. Verification regimes — where "bit-identical" is true and where it is not

The premise "forward results are bit-identical, unlike lgatr 2.0" is right in the two places that matter and wrong in one, and the gates are shaped accordingly:

- **R1 — eager after the change vs eager before (BIT, `torch.equal`).** Everything Stage 1 does to the model code is *pure data movement* (index construction replacing boolean-mask scatter and `repeat_interleave` — same values, same order, no arithmetic), and the knob defaults off. So eager outputs must be **bit-identical** to pre-change fixtures, fp32 and fp64. This is a *stronger* gate than anything the lgatr migration can have, and it is the core gate here.
- **R2 — compiled vs eager (TOL, tolerance).** NOT guaranteed bit-identical: Inductor fuses and reorders reductions (BatchNorm batch stats, `index_add_` segment sums, matmul tiling). Bar: relative ≤ 1e-10 in fp64 on CPU (≤ 1e-5 fp32 on GPU in Stage β). Bit-equality, when it happens on small CPU graphs, is recorded as a bonus — never required.
- **R3 — compiled determinism (DET, `torch.equal`).** The compiled path against *itself*, run twice on the same inputs, must be bit-identical — separates compiler nondeterminism from real deltas before anyone chases ghosts.

## 2. Inventory (evidence)

**Compile-clean already** (no `.item()`, no `nonzero`, no data-dependent Python branching in `CGENN.forward`/`CGLayer.forward`): ~~einsums (`fcgp.py:72,75`, `gp.py:60,64`, `cliffordalgebra.py:53`)~~ **corrected by the RECOMP gate (see Log)** — break-free, yes, but a **3-operand** einsum recomputes an opt_einsum contraction path from concrete operand sizes on every call, which re-specializes the compiled graph per batch shape; rewritten as the equivalent fixed 2-operand chains. The 2-operand einsums (`linear.py:48,52`) never consult the path machinery and stay as-is. Still clean: `index_add_`-based `unsorted_segment_{sum,mean}`, BatchNorm1d, sigmoid gating, masked mean readout.

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
| β-PERF | cluster: it/s eager vs compiled (quick config + full-size batch), fp32/bf16-off | numbers published in this doc's log whatever they say — compile is only worth shipping if this table says so *(superseded: the operator adopted compiled-dynamic defaults ahead of β-PERF, matching lgatr 2.0 / tagging-guide practice; the matrix now confirms-or-flips one-line knobs — see the posture-flip entry)* |

Fixtures are trivial compared to the lgatr migration: eager outputs on a fixed seeded batch (fp32+fp64) + sha256, recorded **at current HEAD before any edit** — no state_dict, no transplant, no env swap; the whole Stage 1 fits in one session.

## 4. Task split, prompts, operator gates

> **The execution playbook that drove steps C-α/C-β has been deleted** (`cleanup.md`, post-merge wipe) — session scaffolding, historical once the branch landed. The task split below is the record of what was run; the Log section carries every measured number it produced.

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
*(Superseded by the posture-flip entry: adoption came first, operator-directed, upstream-matching; β-PERF remains the confirm-or-flip input.)*

## 5. Extensions and non-goals

- **Stage 2 — `tag_lorentznet`**: same recipe verbatim (fixtures → BIT/TOL/DET/BREAKS/RECOMP/SUITE → β-PERF); the migration runbook §8 already holds the readiness notes (LGEB stack is compile-clean as-is; compile the net, not the wrapper; `dynamic=True`; nothing to rewrite — so Stage 2 skips §2 entirely and is mostly gate-running).
- **Hybrids**: after the lgatr migration only — v2's compiled attention removes the graph breaks that currently make whole-block compile pointless; the CGENN-branch rewrites then serve both hybrids at once through the shared `CGENNLGATrGraphTransHybrid.py` stack.
- **Sparse-GP**: separate task with a *tolerance* workflow (it reorders arithmetic — BIT can never gate it); do it only if the FLOPs/profiler comparison (migration §8 "profile first") shows the Cayley einsums dominate.
- **Non-equivariant family**: out, per migration §8 — revisit only on profiler evidence.

Effort: Task α ≈ one focused session; Task β ≈ one cluster hour. Log section (append results below):

## Log

**Stage 1 (Task C-α) — landed, all gates green (2026-08-07).**

*Session 1 (rewrites + gates, RECOMP left failing with findings):* fixtures recorded at
pre-edit HEAD (fp32+fp64 + canonical content hashes; commit order proves record-before-edit).
Rewrite (a) einsum→outer+matmul applied verbatim, failed BIT at max|diff| = 8.327e-17 fp64 on
the blades-subset path, reverted and reclassified TOL (see §"Top lever" — the timing claim
stands, the gate class changed). Rewrites (b) bool-mask scatter→integer indices and (c)
tensor-valued `repeat_interleave`→precomputed `index_select` landed BIT-green. Graph breaks
20→0 (tensor-iteration ints, in-trace `int()`, tensor-valued slice endpoints — all moved to
init). Gates then: BIT ✓, TOL 4.283e-16, DET ✓, BREAKS 0, **RECOMP FAIL: 11 unique graphs on
the first forward, +4 per (B, P) shape despite `dynamic=True`** — stop-and-report; fix needed
an interface change beyond the session's knob-only scope.

*Session 2 (operator unblocked the interface change; RECOMP root-caused and fixed):*

1. **`functools.cached_property` materializes through an RLock** (`functools.__get__`) —
   six first-call graph breaks fragmenting the net into 11 graphs, invisible to
   `dynamo.explain` because any eager warm-up forward fills the caches first (the BREAKS
   gate now explains a COLD build for exactly this reason). Fix: `CliffordAlgebra.__init__`
   touches every cached property (`_alpha/_beta/_gamma_signs`, `geometric_product_paths`) —
   same values, computed once at init.
2. **Python-int `n_nodes` in the compiled interface** specialized the graph per padded
   length. Fix: the wrapper now passes `node_mask` dense `(B, n_nodes, 1)` and the net reads
   the padded dim off that tensor (symbolic under `dynamic=True`); `n_nodes` dropped from
   `CGENN.forward` (closed interface — wrapper is the only caller). Padded-mean quirk
   untouched.
3. **3-operand `torch.einsum` recomputes an opt_einsum contraction path from concrete
   operand sizes per call** — the remaining per-shape re-specialization (and per-call python
   overhead). Fix: `cliffordalgebra.geometric_product`, `fcgp.py`, `gp.py` now run the exact
   pairwise chains opt_einsum selects at production shapes as explicit 2-operand einsums
   (which skip path computation): outer-first `bnk,bni->bnki; bnki,{m}nijk->b{m/n}j` for
   fcgp/gp and the full/scalar cayley, GEMM-first `ijk,...i->jk...; jk...,...k->...j` for
   the grade-subset (g,1,g) tables — selector reads only static blade dims. **BIT-green**
   (pairwise einsum reproduces the pathed lowering bit-exactly; the reverted outer+matmul
   rewrite was a different kernel — that revert stands).

Final Stage-1 gate numbers: **BIT `torch.equal` fp32+fp64 ✓ · TOL 4.283e-16 (bar 1e-10) ·
DET ✓ · BREAKS 0 (cold build) · RECOMP unique_graphs = 1 across the (B, P) sweep (bar ≤ 2)
· SUITE 625 passed / 15 failed / 39 skipped** — the 15 are exactly the known pelican-FLOPs
environment class (migration decision log), nothing CGENN-related *(superseded 2026-08-09:
that class was a misdiagnosed harness gap — unforced nested compile knobs; all 15 now pass
and the expected suite state is zero failures; see the migration log's correction)*. β-PERF (cluster it/s)
remains the open Stage-1 item and gates whether `compile: true` ships in `tag_cgenn.yaml` —
the knob currently stays `false`.

**C-β, CPU tranche (2026-08-07)** — fixture batch (B=4, 256 padded rows), fp32, 4 threads,
median of 30 forwards after warm-up; OLD = pre-rewrite fixtures commit (`dbb0c02`) via
worktree, NEW = post-Stage-1 HEAD:

| config | median fwd | notes |
|---|---|---|
| OLD eager | 172.7 ms | `einsum` tree ≈ 91% of runtime; `copy_` 51.5% self (einsum-internal marshalling) |
| NEW eager | 173.1 ms | statistically identical — the Stage-1 rewrites are perf-neutral eager, as designed |
| NEW compiled (`dynamic=True`) | **110.8 ms (1.56×)** | `copy_` 8.4% (inductor fused the marshalling); `bmm` 31.6% total remains; one-time compile 65.6 s |

Read-across: (1) the eager-level einsum→outer+matmul rewrite is now ~~permanently closed~~
*(superseded same day by the operator-directed `gp_impl` knob — next entry)* — inductor
already fuses eager einsum's marshalling on the compiled path, so the rewrite's win is a
strict subset of what compile delivers, and its BIT failure stands **for the default path**;
(2) the compiled model is GEMM-bound (`bmm` + fused graph ≈ 96%), i.e. the remaining fat is
the **dense 16× Cayley arithmetic** — exactly the sparse-GP lever. CPU 1.56× is the floor
of interest; the GPU number decides shipping (small GEMMs are launch-bound eager on GPU, so
the compiled win is plausibly larger there).

**gp_impl knob: `einsum | matmul | sparse` — implemented and gated (2026-08-07,
operator-directed; both forms are lgatr-2.0 imports).** `model.net.gp_impl` on
`tag_cgenn.yaml`, default `einsum` (the BIT reference path, byte-untouched):

- `matmul` — dense outer product + one GEMM (lgatr 2.0's dense form): fcgp contracts
  `(B, n·256) @ (n·256, m·16)`; gp uses a per-feature bmm.
- `sparse` — lgatr 2.0's `sparse_gp` trick adapted to CGENN's per-path weights: the blade
  Cayley table is quasigroup-like (exactly one nonzero right blade per (left, output) pair
  — 256 of 4096 entries, asserted at algebra init), so the product becomes a (16, 16)
  gather + 2-op einsum. No scatter, deterministic, 16× fewer MACs, and fcgp never
  materializes the dense `(m, n, 16³)` weight.

Gates, all green: TOL-IMPL eager-vs-reference matmul 1.646e-12 / sparse 1.511e-13 fp64
(8.4e-07 / 3.6e-06 fp32; bars 1e-10 / 1e-5); per-impl compiled TOL ≤ 4.3e-16 · DET ✓ ·
BREAKS 0 · RECOMP 1 · SUITE 629 passed / 15 failed / 43 skipped (the known pelican
environment class only; +4 passes are the TOL-IMPL gates). `state_dict` unchanged (every
new table is a non-persistent buffer), so all recorded fixtures and checkpoints stay valid.

**C-β CPU matrix** (fp32, 4 threads, fixture batch, median of 30 forwards):

| `gp_impl` | eager | compiled (`dynamic=True`) |
|---|---|---|
| einsum (default) | 173.1 ms | 110.8 ms |
| matmul | **143.3 ms** | **98.1 ms** |
| sparse | 392.6 ms | 116.1 ms |

Reading: **matmul wins CPU both ways** (1.21× eager, 1.13× compiled over einsum — 1.76×
combined over the eager default). **sparse loses on CPU despite 16× fewer MACs** — the
per-forward weight gather plus tiny j-batched GEMMs have worse locality than one
well-shaped dense GEMM (this section's "expect well under 16×" caveat, confirmed hard).
lgatr 2.0's own sparse motivation is GPU-shaped (single fused kernel, no 16×16 buffer,
memory-light), so **the β-PERF GPU matrix — it/s eager vs compiled × three impls — picks
the campaign setting**; the CPU ranking is matmul > einsum > sparse. FLOPs column:
`matmul` reorders the same arithmetic (column unchanged, no footnote); only `sparse`
changes counted FLOPs (16× less GP arithmetic) and needs the row footnoted if chosen.
Next hotspot after matmul+compile on CPU: `mm`+`bmm` 24% (the GEMMs themselves) and
`copy_`/`clone` ≈ 19% self — marshalling around *extern* MKL GEMM calls, which CPU
inductor cannot fuse into (the eager 50% `copy_` is already down to 9%). No further
CPU-side lever worth taking; the next real datum is the GPU matrix.

**Default flipped to `gp_impl: sparse` (2026-08-07, operator-directed).** This adopts
lgatr 2.0's own default posture: `PrimitivesConfig` ships `sparse_gp=True` ("under
`torch.compile` this is both faster and far lighter than the dense product") and
`sparse_linear=False` (sparse linear "has fewer FLOPs but no single fused BLAS GEMM, so on
FLOP-rich GPUs it is typically slower; it mainly helps on FLOP-bound hardware") — i.e.
sparse GP + dense linear on GPU, exactly the operator's recollection. CGENN's MVLinear has
no dense/sparse choice to make (its blade-diagonal weight structure IS the model), so the
adoption reduces to the GP default. Note the honest asymmetry: lgatr's sparse gp is a pure
gather-multiply-sum with **no matmul by design** (their comment: a matmul would materialize
the 16×16 operand; the fused gather is lighter under compile), while CGENN's weighted
sparse form must still contract channels — on CPU inductor lowers that einsum to
badly-shaped bmm, which is why sparse lost the CPU matrix above. The campaign is GPU, where
triton fuses the gather like lgatr's kernel; β-PERF can overturn the default with one yaml
line. BIT/reference gates now pin `gp_impl=einsum` explicitly, so the recorded fixtures
stay authoritative while the suite exercises the campaign posture.

**Campaign-order ruling (pre-β):** *(superseded same day: the operator confirmed compiled
CGENN still dominates the campaign budget and directed immediate implementation of both GP
forms — see the `gp_impl` entry below)* sparse-GP does NOT blind-jump the queue. Order is:
β-PERF GPU numbers for compiled tag_cgenn (cheap: knob + gates already shipped) → if
compiled CGENN still dominates the campaign budget (it is ~84% of campaign FLOPs today),
sparse-GP moves ahead of the campaign as a TOL-class task with its own fixtures — noting it
also changes the published FLOPs column (16× less GP arithmetic), so the row must be
footnoted as sparse-GP CGENN either way. If compiled CGENN fits the budget, sparse-GP stays
post-campaign as originally scoped. Stage 2 (LorentzNet, gate-running only) is cheap and can
slot before the campaign regardless.

**Stage 3 — the CGENN hybrids, ported and gated (2026-08-07, operator-directed).** The GPS
hybrid imports its CGENN stack from `CGENNLGATrGraphTransHybrid.py`, so one file's rewrites
served both models. Discipline identical to Stage 1: pre-port fixtures recorded and
committed first (`e494657`), then the full fix family: cached-property warm-up + quasigroup
gp tables in the hybrid's own `CliffordAlgebra`, int slice endpoints + `grades_list`
(tensor iteration was an in-trace `.item()` per element — 7 breaks in `qs`/`get_grade`/
`get_invariants`), bool-mask scatter → `_path_idx`, every tensor-valued
`repeat_interleave` → `blade_subspace_idx` gathers (incl. two missed gating sites in
`CGLayer.forward` and `MVSiLU`), 3-op einsums → the 2-op chains, and the full
`gp_impl: einsum|matmul|sparse` knob threaded through `CGENNBackbone`,
`CGENNLGATrGraphTrans` and `CGENNLGATrGraphGPS`, campaign default `sparse` in all four
yamls (the snapshot-diff parity gate got a value-pinned exemption for exactly this
addition).

Gate results (`test_cgenn_hybrid_compile.py`, 13/13): **BIT `torch.equal` fp32+fp64 both
hybrids ✓ · TOL-IMPL matmul/sparse ≤ 1.3e-16 · compiled TOL ≤ 1.4e-16 · DET ✓ · BREAKS
24 → 3** — the 3 survivors are all `aten.nonzero` in `generate_edges_vectorized`:
data-dependent edge building, the same class as tag_cgenn's deliberately-eager wrapper
edges, except here it lives INSIDE the net. The gate now asserts exactly that: ≤ 3 breaks,
every reason the dynamic-shape edge class, zero fix-family classes (RLock / `.item()` /
opt_einsum). **RECOMP [4, 4, 4]** across the production-regime sweep — the four fragments
are dynamic from their first compile and never re-specialize. Small-P batches (padded
length ≤ k) legitimately compile one extra regime each: `k_actual = min(k, P−1)` is a real
branch that changes topk's shape semantics (verified guard-by-guard with `guard_fail_fn`);
production batches never enter those regimes. lgatr144 parity stayed green throughout
(23/23) — the second, independent guard — and SUITE closed at 638 passed / 15 failed / 47
skipped (the known pelican environment class only).

~~Hybrid compile REMAINS unwired in configs~~ **Completed same day — edge hoist + knobs
(operator-directed).** Both hybrid nets expose `build_edges(v, mask, points)` (the static
kNN build depends only on raw inputs — these hybrids compute edges ONCE, unlike the
ParticleNet hybrids' per-block re-kNN, which this pattern must NOT be applied to), the
wrappers hoist it unconditionally outside the (possibly compiled) net — identical values in
identical order eager, BIT-verified — and `forward(..., edges=None)` keeps the in-net
fallback. Result under the strict Stage-1 bars: **BREAKS 0 (cold, both hybrids) · RECOMP
[1, 1, 1] — one whole-net dynamic graph reused across every production-regime shape · TOL
≤ 1.4e-16 · DET ✓**. Port history: 24 breaks → 3 (fix family) → 0 (hoist); fragmentation
~4 graphs → 1. `compile: false` knobs wired in all four hybrid yamls (net-only,
`dynamic=True`), ready to flip on β-PERF numbers.

**Stage 2 — tag_lorentznet, gated and knob-wired (2026-08-07).** The runbook's readiness
note was one break short of true: the LGEB stack itself is compile-clean, but the readout's
PyG `MeanAggregation` derives `dim_size = int(index.max()) + 1` — an in-trace `.item()`,
3 breaks. Fix (bit-identical on real data, and strictly more correct if a trailing event
ever had zero constituents): the wrapper passes `ptr` and the net aggregates with
`dim_size=ptr.numel() - 1` (symbolic under compile). Fixtures recorded pre-edit
(`tests/fixtures/lorentznet_compile/`), then gates: **BIT `torch.equal` fp32+fp64 ✓ ·
TOL 0.000e+00 (bit-equal, recorded as bonus) · DET ✓ · BREAKS 0 (cold) · RECOMP
unique_graphs = 1**. `compile: false` knob wired in `tag_lorentznet.yaml` (CGENNWrapper
pattern — net only, wrapper edges stay eager); flip on β-PERF numbers.

**LorentzNet-slim hybrid pair + table-wide posture flip + review pass (2026-08-07,
operator-directed).** (1) `tag_LorentzNetLGATrSlimGraphTrans/GPS` gated via
`test_lorentznet_hybrid_compile.py` — recon found them born compile-clean (dense top-k kNN
with `idx`/`nbr_mask`, no `nonzero`; 2-operand einsums only; no cached properties), and the
gates confirmed with ZERO code changes: BIT ✓ · TOL 6.6e-18 / 0.0 · DET ✓ · BREAKS 0 ·
RECOMP [1, 1, 1]. Compile knobs wired. (2) Per operator directive (lgatr 2.0 + tagging-guide
practice: dynamic compilation everywhere, zero-padded or not), the **production tree now
ships compiled-dynamic for all eight models** — tag_lgatr (net-level, superseding the
Gate-H parking), tag_slim (already), tag_cgenn, tag_lorentznet, and all four hybrids —
while **config_quick stays eager** (CPU test tree; compile-on-CPU is the env-gated smoke).
(3) The review pass caught a real bug in the wrapper knobs: `self.net =
torch.compile(self.net)` wraps in OptimizedModule and prefixes every parameter with
`_orig_mod.` — breaking checkpoint interchange between compiled and eager runs (and the
production parameter-manifest gate, which is how it surfaced). All six wrapper sites now
use in-place `nn.Module.compile(dynamic=True)`, which keeps `state_dict` keys byte-stable;
verified: compiled-knob build loads eager-recorded fixtures `strict=True` and runs. Knob
matrix (composed via hydra over both trees) asserts the full posture table.

**Sparse-jet assessment (2026-08-08, operator-directed).** Census: tag_lgatr / tag_slim
/ tag_lorentznet are ALREADY sparse (flat token/row lists, block-diagonal attention or
ptr-derived edges — no padding reaches those nets). The padded world is the MPNN/kNN side
(46 deliberate `to_dense_batch` sites): the CGENN family, the slim hybrids' dense top-k,
and the padded-attention transformers. Measured padded-slot waste under the
pad-to-batch-max policy (mini top-tagging set): **44% aggregate at batchsize 32, 50% at
128** (max 64%) — that is the entire theoretical ceiling of a sparse conversion. Verdict:
**not worth it as an optimization program.** (a) Where it is free it is already done;
(b) where it would pay most — the CGENN family — padding is load-bearing for official-repo
parity (theta_h BatchNorm runs over padded nodes and the readout divides by the padded
max, both documented parity locks), so sparse there is an ablation-class MODEL change, not
an optimization; (c) ParT already carries its own anti-waste mechanism (SequenceTrimmer
quantile-trims per batch); (d) length-bucketed batching could reclaim most of the 44-50%
without touching the per-batch function, but it changes batch composition (and therefore
the BN-over-padded training distribution) versus the official random-batch recipe —
recorded as a disclosed post-campaign option, not applied.

**CGENN speedup round 2 (2026-08-08).** Fresh angles after the round-1 CPU closure, all
measured: (i) fused right+left MVLinear (one concatenated einsum + split) wins 24% on the
pair micro-benchmark (680→516 µs) but the pair is <1% of the 173 ms forward — rejected;
(ii) sparse-impl pair-gather and weight-gather layouts re-checked — already optimal
(round 1); (iii) the parity-locked padded compute above (44–50% of slots) is the honest
remaining headroom, unlockable only as a disclosed model/recipe change; (iv) the GPU
levers stay with β-PERF (max-autotune-on-GPU column, batch-size refind, and the
`gp_impl` matrix). Independent confirmation this round: the 2-op chains are BITWISE equal
to the raw 3-operand einsum at the full table and every grade subset on random shapes
(0.00e+00), the three `gp_impl` forms agree to ≤5.3e-15 (fp64) across random channel
dims and both `include_first_order` branches, and lgatr 2.0's own `_GeometricProductSparse`
uses the identical quasigroup construction (`gp.abs().argmax(-1)` + gathered signs; label
convention transposed, unweighted, with a memory-saving custom autograd Function that our
weighted form deliberately does not need).

**Stage 4 — the non-equivariant family (2026-08-08, operator-directed; MIParT
descoped).** Fixtures-first for all eight padded-dense models
(`tests/fixtures/nonequi_compile/`, BIT `torch.equal` fp32+fp64 + content hashes; the
recording itself caught a real pre-existing bug: `config_quick/model/tag_particlenet.yaml`
targeted the library ParticleNet, which has no `v` kwarg). Gate results, per model:

- **tag_ParT (the in-repo lloca port)** needed the deepest surgery, every step
  fixture-guarded: (i) SequenceTrimmer's tensor counter branch is a `generic_jump` break —
  and the obvious int mirror STILL recompiled per step because dynamo guards python int
  attributes **by value** (guard-fail log: `_counter_int == 0`, `== 1`, …); the fix is a
  `_warmed` bool plus hoisting the `tick()` to the wrapper, outside the compiled region.
  (ii) PairEmbed's sparse pair path gathers real pairs via `nonzero` (data-dependent by
  design), and the dense path pins `seq_len` through int-only `torch.tril_indices`;
  compiled ParT therefore routes an **all-pairs broadcast twin** (`compiled_dense`) that
  is the same function at reassociation level (dense-vs-sparse probe 2.2e-15). (iii)
  `torch.compiler.is_compiling()` cannot be used for the routing — it does not
  constant-fold under `dynamo.explain` and manufactures reasonless splits; plain bool
  attributes do. (iv) `dynamic=True` alone never promoted the padded dim: the wrapper
  `mark_dynamic`s every net input INCLUDING all three `Frames` tensors (matrices/det/inv
  — one unmarked container tensor re-pins the whole graph). End state: TOL 9.0e-16 · DET
  ✓ · **BREAKS 0 · RECOMP [1,1,1]**.
- **tag_particlenet, tag_transformer, tag_PlainGraphTrans**: born compile-clean, zero
  code changes — BREAKS 0, RECOMP [1,1,1] (dense top-k kNN traces; `torch.eye(SymInt)`
  is fine; the lloca transformer's SDPA path traces whole).
- **tag_ParticleNetParTGraphTrans / tag_ParticleNetParTGraphGPS**: `nn.MultiheadAttention`
  breaks once per block — not in SDPA but in the mask preamble: a bool
  `key_padding_mask` meeting the float ParT pair bias fires the mismatched-mask
  deprecation `warnings.warn`, which is dynamo-skipped (PlainGPS, which passes no float
  bias, never warns and traces `nn.MHA` whole — that asymmetry located the break).
  Compiled routing goes through **`sdpa_plain_attention`**, a twin of the identity-frames
  MHA call (same weights: packed in_proj → SDPA → out_proj, masks canonicalized by the
  traceable `check_other=False` helper; bias_kv/add_zero_attn guarded off). The PNT-local
  PairEmbed had the same `tril_indices` pin as ParT and got the same all-pairs twin (CLS
  row/col padding preserved). Eager paths untouched (BIT re-verified). GraphTrans: TOL
  5.1e-16 · **BREAKS 0 · RECOMP [1,1,1]**.
- **tag_PlainGraphGPS / tag_ParticleNetParTGraphGPS — a documented break class, not
  zero**: the masked BatchNorm normalizes over REAL nodes only
  (`out[mask_bool] = norm(h[mask_bool])`, `norm: batch` is GraphGPS-official), and boolean
  advanced indexing lowers to `aten.nonzero` — data-dependent **by design**, same class as
  the Stage-3 kNN edge gathers. The gate pins the exact event counts (**11 / 7**) and
  asserts every break reason is that class, so any new break of any other kind still
  fails. RECOMP is strict for them too: the nonzero-split subgraphs take the real-node
  count as an unbacked dynamic dim from the first build — measured **[10,10,10] /
  [8,8,8]**, no per-shape growth. TOL 0.0 / 2.1e-17.
- **tag_MIParT — descoped (operator: "MIParT will not be done")**: BIT/hash fixtures stay
  as regression pins; excluded from the compile-gate parametrization; the wrapper asserts
  `compile=False` and its configs carry no knob (hydra struct rejects an override).

Knobs (all in-place `net.compile(dynamic=True)`, state_dict keys stable): `ParTWrapper`
(twin flags + trimmer tick + mark_dynamic) — already shipped; new: `ParticleNetWrapper`,
`TransformerWrapper` (net-level, like tag_lgatr), `PlainGraphTransWrapper`,
`PlainGraphGPSWrapper`, and the PN-ParT pair (twin-flag loop over
`compiled_attention`/`compiled_dense`). End-to-end knob probe (hydra-composed quick tree,
`model.compile=true`, fixture weights): all seven ≤ 9.0e-16 vs the eager fixtures,
deterministic, twin flags exactly where expected. **Posture — ⚠ SUPERSEDED TWICE, see
Round 5 at the end of this log for what actually ships.** As first written, the five
break-free models shipped `compile: true`; round 3 flipped ParT and PNParTGraphTrans to
`false` over training numerics; round 5 fixed those numerics and flipped both back. The
text as written at the time: *the production tree ships `compile: true` for the three
train-clean non-equivariant models (particlenet, transformer, PlainGraphTrans); ParT, both
PN-ParT hybrids and PlainGraphGPS ship `compile: false` with their reasons in the yaml;
`config_quick` stays eager everywhere.*

**⚠ SUPERSEDED — this "honest caveat" was a MIS-CLASSIFICATION; see the train-mode
finding below.** It called the twins' training difference "the same disclosure class as
compile's own fusion-order reassociation". That is wrong by fifteen orders of magnitude
(reassociation is 1e-16; the measured train delta is 6.5e-01), and the framing is what
let the issue stand through three review rounds. Kept here, struck, because the
mechanism description is still accurate and the error is instructive:

> ~~*Honest caveat carried by both all-pairs twins (ParT + PNT PairEmbed), eval-exact,
training-statistical*: in training mode the embed's BatchNorm computes batch statistics
over the full `seq_len²` grid instead of the eager multiset, and the delta differs per family.
(i) **PNT hybrids** (eager = tril over all tokens incl. padding, `remove_self_pair:
false` shipped): the grid counts every off-diagonal pair twice but the diagonal once, so
vs the stats-neutral exact doubling the diagonal's relative weight HALVES — an
O(1/seq_len) perturbation of `fts(a,a)` rows (finite: `lndelta = ln(eps)`); the diagonal
itself is in both multisets. (ii) **ParT** (eager reference = the sparse gather in BOTH
train and eval — `sparse_eval` has no training check — which feeds BN real pairs only):
the compiled twin's grid additionally contains all PADDED pairs, a contribution that
scales with the padding fraction (44–50% of slots at batch level pre-trim, §sparse-jet;
smaller once the SequenceTrimmer engages after warm-up), not with 1/seq_len. Eval is
exact in both families (running stats, elementwise; TOL-gated ≤ 9.0e-16). Compiled
training is therefore self-consistent but not statistics-identical to eager training —
same disclosure class as compile's own fusion-order reassociation, recorded here for the
methods section.~~

**Train-mode finding and posture change (2026-08-09, final audit round 2).** The
blind spot was structural, not a slip: **all five compile-gate files run under
`.eval()`, and the shared `_forward` helper is wrapped in `torch.no_grad()`** — so BIT,
TOL, DET, BREAKS and RECOMP are, every one of them, inference-mode measurements. No gate
in the program constrained training numerics or gradients for any model. Measured with a
new flags-only differential (fp64, dropout zeroed, train mode):

| model | eval max abs delta | train max abs delta | BN running buffers | posture |
|---|---|---|---|---|
| tag_ParT | 4.0e-15 | **6.5e-01** | **diverged 1.5e+01** | now `compile: false` |
| tag_ParticleNetParTGraphTrans | 1.2e-15 | **1.5e-02** | **diverged 1.1e+00** | now `compile: false` |
| tag_ParticleNetParTGraphGPS | 5.6e-17 | **4.4e-04** | **diverged 1.1e+00** | already `false` |
| tag_particlenet / tag_transformer / PlainGraphTrans / PlainGraphGPS | 0 | **0** | equal | unchanged |
| tag_cgenn / tag_lorentznet / LNetSlimGraphTrans (real compile knob) | — | 2.9e-15 / 3.3e-16 / 2.8e-17 | — | unchanged |

Exactly the three PairEmbed-twin models diverge and every other model is bit-zero, which
pins the mechanism on the **twin flags**, not on dynamo — compile itself remains
numerics-preserving in training too (the equivariant row). On tag_ParT one training step
moves every one of 217 gradients by >1% relative, 29 by >50%, worst ~107× on the pair
BatchNorm bias. BN running buffers are *persistent state*: a compiled-trained checkpoint
carries padded-grid statistics into any later eager evaluation, export or finetune.

**Ruling**: `tag_ParT` and `tag_ParticleNetParTGraphTrans` flip to `compile: false`,
joining the GPS pair. Nothing measured is given up — β-PERF has not run, so
`compile: true` on these was an unmeasured posture with a now-measured semantic cost, and
"a different training recipe" is a methods deviation rather than a walltime knob. The
production tree therefore ships compiled-dynamic for the eight equivariant models plus
the three train-clean non-equivariant ones (particlenet, transformer, PlainGraphTrans).
**Durable guard**: `test_nonequi_compile.test_train_mode_differential` reproduces the
table on every run, requires the clean models to stay bit-zero, and asserts each
divergent model's production config still reads `compile: false` — verified to FAIL on a
bare flip. The path to re-enabling is a mask-aware pair BatchNorm (traceable: masked
mean/var are tensor reductions, not data-dependent shapes) — todo §4b-quater, deliberately
not attempted in the final hours before merge.

*Measurement note for anyone re-running this*: a train-mode differential is only
meaningful with dropout fully disabled, and "fully" includes
`nn.MultiheadAttention.dropout`, which is a float ATTRIBUTE rather than an `nn.Dropout`
module. Missing it makes eager and compiled consume the RNG in different orders and
reports a spurious ~3e-02 divergence on PlainGraphTrans (it is 2.2e-16 once the attribute
is zeroed). That is stochastic regularization, not semantics — the gate's `_kill_dropout`
handles both forms.

**Round 4 (2026-08-09): the gap behind the gap — no gate ever ran a backward.**
Closing the eval-mode hole exposed a deeper one: the round-3 train-mode gate is *also*
`no_grad`, and `grep -rn "backward()" tests/` finds no tagging model anywhere in the
suite. Under `no_grad`, dynamo/AOTAutograd compiles the **inference** graph and inductor
lowers it happily; with autograd on it must emit the joint forward+backward graph, and
for two models it cannot:

| model | forward (any mode) | **first `loss.backward()`** |
|---|---|---|
| tag_PlainGraphTrans | OK | **InductorError**: `cannot determine truth value of Relational: 1 < Min(4, Max(1, s98 - 1))` |
| tag_LorentzNetLGATrSlimGraphGPS | OK | **InductorError**: `cannot determine truth value of Relational: 1 < s53` |

Both shipped `compile: true` and would have **died at the first training step**. The
symbolic width comes from `dynamic=True` meeting the kNN cap `k = min(k, max(1, P-1))`
(PlainGraphTrans) and the M8 channel-last `transpose(-1,-2)` (slim GPS); inductor's
stride ordering cannot evaluate the relation. Not dtype- or size-specific. Note the same
`k`-cap exists in the slim GraphTrans and PN-ParT files, which compile+backward cleanly —
the cap is necessary but not sufficient, so do not "fix" it by deleting the cap without
re-running the backward matrix on all four.

Both flip to `compile: false`. **New gates**: `test_compile_true_is_backward_verified`
(default suite, cheap — no production config may ship `compile: true` unless the model is
in `BACKWARD_VERIFIED`) and `test_compiled_backward` (env-gated — actually runs the
compiled training step and requires finite gradients). Membership is earned by
measurement, not assertion.

**Round 4, second finding: the trimmer silently cut the learned-framesnet gradient at
step 6.** `_forward_encoder` wraps the trimmer call in `torch.no_grad()` (upstream's
"data prep" idiom) and re-enables grad only for the frame matrices. Under a *learned*
framesnet `x` and `v` are functions of the framesnet's parameters too, so once `_trim`
engages at `warmup_steps+1` its gathers/slices **detached the feature path** — silently,
mid-run, with a bit-identical forward. Controlled A/B at identical warmed state and seed:
forward `max|dy| = 0.0`, framesnet gradients **42% relatively different**. Fixed by
extending the existing `enable_grad` to cover the whole trimmer call. Only reachable with
learned frames + `trim: true` past step 5 — the CI smoke runs 3 iterations, which is why
nothing caught it.

**Warm-up regime is also cold-build-only (same round).** `BREAKS`/`RECOMP` trace the
*un-warmed* trimmer, i.e. the first `warmup_steps` (5) forwards. Measured on tag_ParT in
train mode: `_warmed=False` → 1 graph / 0 breaks / 329 ops (what the bar pins);
`_warmed=True` → **3 graphs / 2 breaks / 329 ops** — identical op count, so this is the
`@torch.compiler.disable` on `_trim` splitting the graph exactly as intended, not a
numerical change. It is a deliberate trade (two stable splits beat per-step
re-specialization on a re-randomized quantile length), but the "BREAKS 0" headline
describes five forwards, and production trains in the 3-graph regime. Stated here rather
than implied.

**Final audit (2026-08-08, three independent legs: consistency reviewer + numerics
reviewer + max-effort whole-diff review).** Two legs converged, with executable repros,
on one CRITICAL find: both all-pairs PairEmbed twins computed the pair features
grad-enabled where every eager path uses `torch.no_grad` — under a LEARNED framesnet the
momenta carry grad and backward through `sqrt(0)` (self-pair `delta==0` in the ParT
port; unclamped `pt` at zero-padded columns in the PNT `to_ptrapphim(eps=None)`) is
`0·inf = NaN` into the framesnet on the first step. Invisible to every gate (identity
frames). Fix: `detach()` at twin entry — the exact gradient twin of eager's `no_grad`,
traceable; verified grad-into-momenta None/zero in all 8 module configs, full-model
train grads finite, and the shipped-path gates reproduce their exact numbers. Also
fixed: trimmer warm-up off-by-one (first trim on forward 5 vs upstream's 6; tick now
mirrors upstream, schedules proven equal for warmup 0/1/5); the post-warm-up trim body
hoisted into `@torch.compiler.disable` `_trim` (production training past warm-up splits
once cleanly instead of re-specializing per random-quantile maxlen — the gates trace the
un-warmed net, so that regime was ungated); ParTWrapper `mark_dynamic` →
`maybe_mark_dynamic` (hard marks raised ConstraintViolationError under learned frames,
where the lloca transport pins the seq dim; identity path keeps its single dynamic
graph); the lgatr parity test's global S9 monkeypatch now restores in try/finally;
anti-vacuous gate asserts (dynamo must trace ≥ 1 graph). Weaver-core cross-check
(operator-requested): convergent design — weaver `@torch._dynamo.disable`s the whole
trimmer forward, flips the sparse pair path off under compile (their
`compile_model → sparse_eval`), and mitigates tril recompiles by rounding trimmed
lengths to 32; we keep warm-up traced (0 breaks in the gated regime), disable only the
post-warm-up trim, and make the pair grid seq-dynamic instead of quantizing lengths.
Their `--compile` ships off with pass-through kwargs (dynamic=None automatic); our
production-true posture follows lgatr/tagging-guide instead. Their length-rounding is
worth a β-PERF side-glance on GPU (kernel-shape reuse), recorded here.

**β-PERF cluster matrix (the one remaining CPU-side-prepared decision input; everything
below is a one-line config override on an existing knob).** Measure it/s (the run's own
timing line after warm-up — ignore the first estimate, compile warm-up lands there) on the
quick config first, then one full-size-batch confirmation per winner. `dynamic=True` is
already baked into every compile path (lgatr 2.0 and tagging-guide ship dynamic-true
everywhere, zero-padded or not — our knobs match):

| run | overrides |
|---|---|
| tag_cgenn × {eager, compiled} × {einsum, matmul, sparse} | `model.compile={false,true}` `model.net.gp_impl=...` (6 runs) |
| tag_lorentznet × {eager, compiled} | `model.compile={false,true}` (2) |
| tag_slim compiled (shipping default) vs eager | `model.net.compile={true,false}` (2) |
| tag_lgatr × {eager, compiled} | `model.net.compile={false,true}` (the production yaml carries the key since the posture flip — no `+` prefix; on config_quick, where the key is absent, use `+model.net.compile=...` +`compile_kwargs.dynamic=true`) (2) |
| CGENNLGATrGraphTrans / GPS × {eager, compiled} (gp_impl stays sparse) | `model.compile={false,true}` (4) |
| tag_ParT / tag_particlenet / tag_transformer / tag_PlainGraphTrans / PNParTGraphTrans × {eager, compiled} | `model.compile={false,true}` (10) |
| tag_PlainGraphGPS / PNParTGraphGPS × {eager, compiled-with-splits} | `model.compile={false,true}` (4) — decides whether the documented masked-BN splits still net a win; ships false until they do |

Decisions the table makes: every `compile:` default; the campaign `gp_impl` (sparse is the
lgatr-2.0-posture default — GPU numbers confirm or flip to matmul in one line; only
`sparse` changes the FLOPs column and needs the row footnote); whether full-LGATr compile
ships (Gate H per H11). **Then:** Gates G/H from the migration runbook (cluster).

**Round 5 (2026-08-10): the pair BatchNorm was made mask-aware, and the round-3 posture
rows above are SUPERSEDED.** Round 3 flipped `tag_ParT` and `tag_ParticleNetParTGraphTrans`
to `compile: false` because the twin's training statistics differed from eager, and
recorded the mask-aware pair-BN as the path back. Operator ruling: take that path — a
compile knob whose semantics differ from eager is a trap regardless of the current
default. Implemented as `particletransformer._weighted_batchnorm1d` / `_embed_weighted`:
BN statistics are computed over a WEIGHT tensor equal to the eager reference multiset,
which makes the weighted mean/var over the full grid identical to the unweighted mean/var
over the eager multiset. Traceable, because `w.sum()` is a scalar tensor rather than a
shape — the exact property the eager `nonzero()` gather lacks.

The weight is per file because the reference is per file: ParT defaults to
`sparse_eval=True`, so its reference is `mask.tril(offset).nonzero()` (tril of REAL
pairs); the PN-ParT hybrids have no sparse path and no mask argument, so theirs is the
tril over ALL positions. A `(1,1,P,P)` broadcast weight is WRONG in both — `w.sum()` is
the element count the statistics divide by, so a missing batch dim undercounts by `bsz`
and made the PN-ParT BN buffers worse than doing nothing (1.14 → 464). Measured after
the fix, and the backward matrix re-run under the compiled knob:

| model | train max abs delta (was) | BN buffers (was) | compiled backward | posture |
|---|---|---|---|---|
| tag_ParT | **2.331e-15** (6.5e-01) | **1.110e-14** (15.0) | 217/217 finite | **`compile: true`** |
| tag_ParticleNetParTGraphTrans | **3.109e-15** (1.5e-02) | **5.578e-13** (1.1) | 85/85 finite | **`compile: true`** |
| tag_ParticleNetParTGraphGPS | **2.776e-17** (4.4e-04) | **5.578e-13** (1.1) | 64/64 finite | `false` (perf posture) |

So the current table-wide posture, superseding every earlier posture line in this
document: **`compile: true`** for the eight equivariant models plus `tag_particlenet`,
`tag_transformer`, `tag_ParT` and `tag_ParticleNetParTGraphTrans`; **`compile: false`**
for `tag_PlainGraphTrans` and `tag_LorentzNetLGATrSlimGraphGPS` (InductorError on the
joint backward graph — round 4, a correctness blocker) and for `tag_PlainGraphGPS` and
`tag_ParticleNetParTGraphGPS` (masked-BN graph splits, a *performance* posture that
β-PERF resolves in one line); `tag_MIParT` has no knob by operator decision; and
`config_quick` stays eager everywhere. The GPS pair's correctness question is now
closed — only the walltime question remains, which is what the β-PERF row above measures.

**Round 7 (2026-08-10): tag_PlainGraphTrans compiles — the k-cap made static.** Round 4
recorded this model as a backward-crasher and shipped `compile: false`; round 6 traced the
cause to the kNN cap `min(k, max(1, num_points - 1))`, which makes `k` a SYMBOLIC
expression under `dynamic=True` so inductor cannot order the strides of the topk output
that carries it. Fixed with a compile-only twin, the same shape as
`PairEmbed.compiled_dense`: `plaingraphtrans.knn(..., static_k=True)` keeps `k` a python
int, selected by `PlainGraphTrans.compiled_knn` which the wrapper sets when `compile=True`.

The twin is EQUAL to eager, not merely close, whenever `num_points - 1 >= k` — the cap is
inert there and both branches call `topk` with the same integer. Measured: production
padded `P` is 87–110 per batch at batchsize 128/256 against `knn_k: 16` (**0/15 batches
bind**), and the gates' batchsize-4 batches sit far above `knn_k: 4`. No regime this repo
runs can distinguish them; the twin raises rather than silently diverging otherwise.

| gate | result |
|---|---|
| BIT eager fp32 + fp64 | unchanged (twin defaults off) |
| train-mode differential | **0.000e+00** |
| TOL compiled vs eager | 1.119e-16 |
| BREAKS / RECOMP | 0 / [1, 1, 1] |
| compiled backward | 71/71 nonzero finite grads |

So `tag_PlainGraphTrans` joins `BACKWARD_VERIFIED` and ships **`compile: true`**, and it
leaves `bperf.NO_APPLY` — it is an ordinary speed row now. The deliberately unchanged part:
the identical cap line in `particlenettransformer.py` and `lorentznetlgatrslimgraphtrans.py`
is left alone, because those two already compile and backward cleanly, which is what proved
the cap necessary-but-not-sufficient in the first place. `tag_LorentzNetLGATrSlimGraphGPS`
keeps `compile: false`: that file contains no k-cap at all and its `1 < s53` comes from the
M8 channel-last transpose, so this fix does not transfer.

**Round 7d: LNetSlimGraphGPS FIXED — `recompute_views`, a supported config, no
monkeypatch.** Round 7c proved the blocker was inductor never being handed the ShapeEnv,
and concluded the fix had to be upstream. That was too narrow a reading. The failing
branch fires only for a graph output **that is a view**, and *which* tensors become graph
outputs is AOT's decision, not inductor's:

    torch._functorch.config.recompute_views = False   # default: SAVE views across fwd/bwd

AOT saves views because they are cheap to save. Here the saved view is the GPS layer's
channel-first ↔ channel-last multivector transpose. Setting `recompute_views=True` makes
the backward **recompute** it instead, so it stops being a graph output and inductor never
reaches the stride-ordering path at all. Numerics-preserving by construction — a
recomputed permute is the same permute — and measured so.

Applied as a **scoped** `torch._functorch.config.patch(...)` around the net call in
`LorentzNetLGATrSlimGraphGPSWrapper.forward`, not set globally, because the training smoke
and the FLOPs harness build many models per process.

| gate (caches force-disabled) | result |
|---|---|
| compiled-vs-eager, EVAL | **0.000e+00** |
| compiled-vs-eager, TRAIN | **0.000e+00** |
| TOL / BREAKS / RECOMP | 0.000e+00 / **0** / [1, 1, 1] |
| 3 optimizer steps | 65/65 gradients, zero non-finite |

So the model ships `compile: true`, joins `BACKWARD_VERIFIED`, and **`bperf.NO_APPLY` is
now empty** — every knob-bearing model compiles and survives a real backward.

**Methodology warning, learned the hard way.** Mid-investigation a baseline run reported
"no crash" and briefly looked like the bug had evaporated. It had not: inductor's on-disk
FX-graph cache (1.2 GB in `/tmp/torchinductor_root`) was serving an artifact compiled
earlier under a monkeypatched inductor. **Any compile experiment in this repo must set
`TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`**, or a previously-successful compile of the same
graph will mask a genuine failure. Every number in this entry was measured with caches
disabled, and the `tag_PlainGraphTrans` fix from round 7 was re-verified the same way
(71/71 grads) once the confound was found.

**Round 7c: LNetSlimGraphGPS root-caused to an UPSTREAM TORCH BUG, and proven.** The
blocker is not in this repo and is not fixable from model code.

`ir.get_fill_order` already carries a symbolic-aware branch — `argsort_sym(shape_env,
seq)` — and takes it only when a `shape_env` is supplied. But
`torch/_inductor/graph.py` calls `ir.get_stride_order(strides)` with a single argument, so
`shape_env is None` and it falls into the plain `argsort`, which does a bare Python
comparison on sympy expressions and raises `cannot determine truth value of Relational:
1 < s53`. Nothing consults the ShapeEnv, which is exactly why constraining `P` cannot
help — and why four model-side attempts all failed identically.

Proven, not inferred: monkeypatching `get_stride_order` to pass the live
`V.graph.sizevars.shape_env` makes the model compile **and train** — 3 optimizer steps,
65/65 gradients, zero non-finite, losses `0.710076 / 0.712327 / 0.709093`, bit-matching
the `dynamic=False` run. A one-line upstream fix.

**Why this model and neither parent nor its sibling.** Only GPS converts layouts.
`tag_lorentznet` and `tag_slim` each stay in their own convention, and so does
`tag_LorentzNetLGATrSlimGraphTrans` — it runs the LorentzNet stack and *then* the slim
stack, with zero `transpose(-1, -2)` in the whole file. A GPS layer fuses the local and
global branches on ONE stream per layer, so it must cross between channel-first LorentzNet
and channel-last v2 slim (M8) twice per layer. That 4-D transpose is the node that dies.
The difficulty is the GPS *topology*, not the ingredients.

Re-try trigger after a torch upgrade: check whether `_inductor/graph.py` passes a
shape_env into `get_stride_order`. If yes, flip and run the gate battery. Do not ship the
monkeypatch — patching torch internals under a physics campaign is not worth a bounded
walltime gain on a model that is already correct in eager.

**Round 7b: LNetSlimGraphGPS localized to the node, still not fixed — and one earlier
claim corrected.** Instrumenting inductor's `GraphLowering.run_node` names the failing
node exactly: `aten.permute` at `lorentznetlgatrslimgraphgps.py:109`
(`v_loc.transpose(-1, -2)`) on shape `(B, P, 4, V)` with strides `(16*P, 16, 1, 4)`.
Ordering those requires deciding `1 < P`, and nothing constrains the particle dim away
from 0/1. So the blocker is the **M8 channel-last 4-D layout meeting a symbolic particle
dim**, not any one line — proven by attempt 3 below, where fixing the transpose simply
moved the error to the reshape that rebuilds the same shape.

Correction to round 7: that entry said this module "contains no k-cap at all". Wrong — it
imports a capped `knn` from `lorentznetlgatrslimgraphtrans`. The *conclusion* survives
anyway, because substituting the static-k twin that fixed `tag_PlainGraphTrans` changes
nothing here: same error, same symbol.

| attempt | outcome |
|---|---|
| static-k kNN twin (the PlainGraphTrans fix) | same error |
| `.contiguous()` on all three M8 transposes | same error |
| `flatten(0,1) → transpose(1,2) → reshape` | permute now succeeds (strides 16,4,1); error **moves** to the reshape |
| `torch._check(P > 1)`; `mark_dynamic(min=2)` | no effect; rejected by torch 2.13 |

The lever that would work is making `P >= 2` known to the ShapeEnv so `1 < P` is
decidable. Neither available API landed it on this torch. Next things to try: a torch
upgrade, or export-style `Dim(min=2)` constraints. Until then it stays `compile: false` —
bounded walltime gain on a model that is already correct in eager, and the sole
`bperf.NO_APPLY` entry.

**Round 6c (2026-08-10): the gap the train-mode gate leaves, measured.** That gate is
deliberately FLAGS-ONLY — it sets the twin flags and never invokes dynamo, on the stated
premise that dynamo is numerics-preserving. Nothing had tested the premise in TRAIN mode,
so the shipped configuration (flags **and** dynamo) was unmeasured there. Measured now,
production tree, compiled vs eager, train mode: **tag_ParT 8.757e-16, PNParTGraphTrans
6.798e-16, tag_particlenet 1.318e-15** — the premise holds and the shipped knob is clean.

Getting that number required falling into a trap worth recording, because it cost real
time and looked exactly like a serious bug (a 4.7e-02 relative divergence at the gate's own
batch size). Dropout probability lives in THREE places in this tree: `nn.Dropout.p`,
`nn.MultiheadAttention.dropout` (a float attribute), and — the one that bites — the local
ParT port's own `Attention.dropout`, also a float attribute but on a different class.
Production ParT ships **0.1** on all eight `net.blocks.*.attn`. Miss it and eager and
compiled draw different masks; the resulting RNG desynchronization reads as a 5–9%
"divergence" that scales with nothing and reproduces perfectly. `_kill_dropout` now zeroes
all three forms plus `DropPath.drop_prob`, and says so. Zeroing the third form shifts the
gate's own tag_ParT reading from 1.554e-15 to 2.331e-15 — same machine-precision class,
different values flowing.

`TWIN_TRAIN_DIVERGENT` is accordingly replaced by `TWIN_TOL_MODELS` + `TRAIN_TOL = 1e-10`:
the gate asserts agreement now instead of documenting disagreement, and every other model
must still be bit-zero. Note the GPS backward emits `Backend compiler exception ...
aten.nonzero.default → Adding a graph break`, with dynamo briefly enabling anomaly mode to
attribute it. That is the documented masked-BN break class reappearing in the backward,
not a new failure; gradients come out finite.

---

**GPU compile census (2026-08-10, H100 NVL / NGC 25.08 / torch 2.8.0a0+nv25.08).** The
first time any model in this repo was compiled on a GPU. Every compile gate here runs on
CPU, where inductor emits C++; on CUDA it emits Triton, a different backend with different
codegen -- so the shipped `compile: true` posture had never been exercised on the hardware
the campaign runs on. beta-PERF found that out the expensive way, failing on its first row.

Two GPU-only defects, both invisible to every CPU gate by construction:

  * `CliffordAlgebra`'s `_alpha/_beta/_gamma_signs` were `functools.cached_property`, which
    writes to `instance.__dict__` and so is NOT a buffer `.to(device)` can move. `__init__`
    materialized them, so every instance carried CPU sign vectors unconditionally and
    `signs * mv` in `beta()` raised a device mismatch on the live forward path. Fixed:
    non-persistent buffers, values bit-identical, `state_dict` unchanged.

    **CORRECTION (2026-08-11) to the attribution.** An earlier revision said "pre-existing on
    `main`, in both pre-dedup copies". The `cached_property` is pre-existing -- it is upstream
    CGENN's, vendored at `50c0f38` -- but the CRASH is not, and we caused it. Upstream's
    properties are LAZY: build -> `.to(device)` -> first access computes from an already-moved
    buffer and lands on the right device, so upstream runs on GPU. What forces the bad order
    here is the cached-property warm-up loop at the end of our `__init__`, added by US in
    `ec1d4d2` for a dynamo reason (`functools.cached_property` materializes through an RLock,
    and a lock inside the traced region is a graph break). It touches every property before the
    model is ever moved, turning a latent footgun into a guaranteed failure. Verified in both
    orders rather than reasoned: build -> `.to()` -> access follows the module; build -> access
    -> `.to()` goes stale. Upstream keeps the latent hazard (worth an issue); this repo's crash
    was self-inflicted.
  * `embed()`'s `torch.zeros(..., 2**self.dim, ...)` reached the graph as a symbolic
    expression rather than the constant 16, so inductor lowered the stride as
    `libdevice.pow(2.0, ks0)` -- a float -- and Triton refused
    `pointer<fp32> + float32` while compiling `triton_poi_fused_index_put_zeros_6`.
    Fixed by using the precomputed `self.n_blades` (identical by construction).

Census after both fixes -- 8 optimizer steps per model, production configs, mini data,
each model in its own process, `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`:

**13 / 13 COMPILE** -- tag_cgenn, tag_lorentznet, both CGENN hybrids, both LorentzNet-slim
hybrids, tag_lgatr, tag_slim, tag_ParT, tag_particlenet, tag_transformer,
tag_PlainGraphTrans, tag_ParticleNetParTGraphTrans.

Scope of that claim, stated precisely: it covers each model's SHIPPED config, so
`tag_cgenn` was exercised at its default `gp_impl: sparse`. The row that originally failed
was `tag_cgenn/einsum`; `embed()` sits in `MVLinear` and is independent of `gp_impl`, so the
fix applies to all three impls, but einsum and matmul are confirmed by beta-PERF rather than
by this census. `tag_MIParT` has no compile knob; the three pelicans ship `net.compile: true`
but are not in the smoke matrix and remain GPU-unverified.

Environment prerequisite, or none of this runs: `TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib`.
Triton finds libcuda.so by parsing `ldconfig -p`, which comes up empty inside the NGC image,
so without it every compiled model dies at its first `torch.compile`. Set in
`venv/bin/activate` per docs/OSCAR.md §2; `utils/env_check.py --gpu` now checks it.

**Upstream evidence: Favaro, Plehn, Qu, Spinner, "Virtues and Vices of Equivariant
Transformers" (arXiv:2608.02735), read 2026-08-11.** The reference implementation for lgatr 2.0,
and the authors of the library we migrated to. Their §2.2 settles several things this log had
open, and contradicts one of our own measurements.

*Compile is universal and large.* "In the left panel of Figure 2, we observe significant
speedups for all architectures" -- 1.0-2.5x on GPU inference. Their Table 2 gives JetClass
TRAINING times before -> after (1M iterations, batch 512, H100): transformer 15h->9h, ParT
33h->19h, L-GATr-slim 27h->16h, LLoCa-Tr 28h->12h, L-GATr 166h->63h (sparse) / ->57h (dense).
"The training time of all networks is reduced by values between 40% and 70%." That is compile
plus the sparse GP plus micro-optimizations, not compile alone. Their configs match: `part.yaml`
and `lgatr.yaml` both ship `compile: true` with `compile_kwargs: {dynamic: true, mode: default}`.
Our 16-of-18 posture is the same posture.

*BUT compile is a LONG-RUN optimization, and that is not visible in their numbers.* The
break-even is `C < T_eager (1 - 1/s)` for one-time compile cost C and speedup s. Measured here
for tag_cgenn on an H100 at production batch: C = 86 min, s = 1.49 -> break-even at ~29k
iterations (~4.4 h of eager training). Below that, compiling is a LOSS: a one-hour eager run
becomes ~2.1 h compiled. Their runs are 1M iterations, so the cost is invisible in Table 2;
ours are 20 epochs (~378k iterations at batch 64), comfortably past it. Any short run -- a
debug job, a smoke test, config_quick -- should stay eager, which is already the shipped
posture for config_quick.

*Sparse vs dense, and where our CGENN diverges.* The paper is precise about a split we had
merged: "the sparse variant is faster on geometric product operations. For linear operations,
the dense approach is still faster on a GPU". Hence upstream's `lgatr.yaml` ships
`sparse_gp: true` with `sparse_linear: false`, and hence their two published variants differ
only in the LINEAR layer (L-GATr_dense faster on GPU, L-GATr_sparse faster on CPU). Our
`gp_impl` knob is the geometric-product half only; we have no sparse-linear variant, and by
their result we should not want one on GPU.

**Our sparse GP does not reproduce their result, and the reason is memory -- measured.** beta-PERF on an
H100 sized each row by OOM search: `gp_impl=einsum` and `matmul` both fit batch 128 (peak
80.2 GB at 128), but `sparse` peaked at 15.2 GB already at batch 16 and OOM'd at 128 -- it was
sized to 32 where the others got 64. Throughput must therefore be read as jets/s, not it/s:
einsum 2.80 it/s x 64 = 179 jets/s, matmul 2.70 x 64 = 173 jets/s, sparse at half the batch.
So our sparse GP buys 16x fewer MACs and pays for it in memory, on a device where memory is
the binding constraint -- the opposite trade to lgatr's. The likely cause is that our
implementation gathers the Cayley entries into a per-pair intermediate (`gather + 2-op
einsum`) where lgatr's sparse GP does not materialize one.

ROOT-CAUSED 2026-08-11, by instrumenting `saved_tensors_hooks` over one
`SteerableGeometricProductLayer` at production shapes (B=6400 rows, 8 x-features, 16 blades):

| gp_impl | saved for backward | (B, F, 16, 16) tensors retained |
|---|---|---|
| einsum  |  81.9 MB | **one** (52.4 MB) |
| matmul  |  81.9 MB | **one** (52.4 MB) |
| sparse  | **130.9 MB** | **two** (52.4 MB each) |

Ratio 1.60x, against the 1.50x peak-memory ratio the GPU OOM search measured independently
(15.2 vs 10.1 GB at batch 16). Same effect.

The two come from naming both halves:

    pair = input.unsqueeze(-1) * input_right[..., gp_k_idx]   # gather -> (B,F,16,16), SAVED
                                                              # (needed for d/d input)
    product = einsum("bnij,nij->bnj", pair, w)                # pair    -> (B,F,16,16), SAVED
                                                              # (needed for d/d w)

einsum/matmul name only one such tensor, so they save one. There is no cheap eager fix: any
formulation that computes `x_i * y_k(i,j)` and then contracts against a weight must keep one
of the two for the backward unless the backward RECOMPUTES the gather.

Which is exactly what lgatr does, and it is not a contraction trick -- it is a custom
`torch.autograd.Function` (`_GeometricProductSparse`) whose `setup_context` saves only the two
INPUTS and whose hand-written backward re-derives the sparse contraction. Their comment says
so: "Bilinear, so the gradients are the same sparse contraction; saving only (x, y) keeps this
lighter than dense." That is 6.6 MB where ours is 130.9 MB. Their forward is also a single
fused expression `(signs * y[..., indices] * x.unsqueeze(-2)).sum(-1)` so inductor emits one
kernel with no 16x16 buffer.

CONCLUSION: the paper's sparse-GP result does NOT transfer to this repo, because our sparse is
not their sparse. Ours is the same contraction with none of the memory engineering, and on a
memory-bound H100 the halved batch size costs more than the 16x MAC saving buys.

Two ways forward, in order of risk:
  1. **Ship einsum (or matmul) for the campaign.** On the measured numbers einsum leads at
     179 jets/s vs matmul 173, and both run at twice sparse's batch size. Costs nothing.
  2. **Port lgatr's approach**: single fused expression plus a custom autograd Function saving
     only (x, y, w). Our GP has learnable per-path weights where lgatr's has none, so the
     backward needs a third gradient -- d/dw is a straightforward contraction, but d/dy is a
     scatter-add over the (i,j) pairs mapping to each output blade, which is the fiddly part.
     Verifiable by `torch.autograd.gradcheck` in float64 plus a direct gradient comparison
     against the einsum reference. NOT attempted here: it is a hand-written backward on the
     physics path, the BIT fixtures that would have pinned it are deleted, and it would need
     fresh ones recorded first.

> **SUPERSEDED 2026-08-11 — option 2 was taken.** Everything above this line is the record
> of the diagnosis and stays as written; the conclusions it draws are no longer current, and
> a reader deciding `gp_impl` from this section alone would get three things wrong.
>
> - *"Ship einsum for the campaign"* — the recommendation was conditional on sparse's memory
>   penalty. That penalty is gone: the custom Function
>   (`experiments/baselines/cgenn/sparse_gp.py`) cut retained activations from 293.4 MB to
>   84.8 MB whole-model EAGER (3.46x), where sparse now retains **less** than einsum rather
>   than more. Compiled, the partitioner equalizes all four impls and there is no retention
>   difference at all -- see the correction further down; the Function is eager-only for
>   that reason.
>
>   **Which leaves the campaign posture UNRESOLVED, and leaning the other way.** Because
>   the Function is eager-only, the compiled sparse path today is byte-for-byte the code
>   the H100 OOM search measured -- so that finding is untouched: sparse peaked at 15.2 GB
>   by batch 16 and was sized to 32 where einsum and matmul got 64.
>   `gp_impl: sparse` therefore now rests on lgatr's default plus a CPU timing inside a 9%
>   noise floor, against a direct measurement of OUR code on the target hardware saying the
>   opposite. That is the wrong way round for this repo. Either rerun beta-PERF for the
>   CGENN rows (with `--bs-safety 1.0`, since the sizes are the finding) or ship `einsum`.
>
>   > **ANSWERED THE NEXT DAY, AND THE MEMORY HALF INVERTED.** β-PERF was rerun on
>   > 2026-08-12 (§"β-PERF, CGENN rows — the memory inversion"). The "15.2 GB at bs 16 /
>   > half the batch" numbers in the paragraph above are SUPERSEDED, not merely uncertain:
>   > they predate the Function, and the eager probe now reads sparse **6.0 GB against
>   > einsum's 10.1** at bs 16 and **47.6 vs 80.2** at bs 128, with all three impls sizing
>   > to the same 128. So sparse no longer loses on memory, eager, and the "half the jets/s"
>   > clause this paragraph used to end on has no premise left. What DID survive the rerun
>   > is the opposite of what is written here: compiled, einsum is the FASTEST of the three
>   > (1.41 vs matmul 1.36 vs sparse 1.25 it/s at the shared batch). Read the later section,
>   > not this one, and see there for why that still does not settle the posture.
> - *"NOT attempted here"* — it was attempted and shipped. The d/dy scatter that this
>   paragraph calls the fiddly part is not a scatter at all in the end: for a fixed left
>   blade the map j -> k(i, j) is a bijection, so it inverts into a gather, which is also
>   what makes the backward deterministic on CUDA where plain autograd is not.
> - *"the BIT fixtures that would have pinned it are deleted"* — they were restored from
>   `da497a9^`, and the Function has its own fixture-free gates in
>   `tests/experiments/test_sparse_gp.py` (gradcheck, gradgradcheck, forward bit-identity,
>   gradient agreement, retention, the masked-path branch, and a compiled-training-step
>   recompile sweep).
>
> The current state is the "L-GATr 2.0 trick census" section at the end of this document.
> **The one number that has NOT been re-measured is the H100 peak** that motivated all of
> this — no GPU here — so the OOM search is the first thing the next β-PERF run should redo.
> *(It was redone, 2026-08-12. See the inversion note above.)*

*Mixed precision, where we differ from upstream by omission.* The paper: "the baseline
transformer and ParT can be used with mixed precision without performance drop", while "For
the Lorentz-equivariant LLoCa-Transformer, L-GATr-slim, and L-GATr, training with AMP reduces
performance at a rate that renders it impractical, so we stick to float32." Their `part.yaml`
ships `use_amp: true`; their `default.yaml` ships `autocast_bfloat16: true`. Every model config
in THIS repo ships `use_amp: false`. For the equivariant family that matches their finding and
should stay. For `tag_ParT`, `tag_transformer` and the ParticleNet/ParT hybrids it leaves
their reported 1.5-2x AMP gain unclaimed -- a larger factor than compile, and independent of
it. Not changed here: bf16 changes training numerics and needs its own gate, so it is an
operator decision with a measurable payoff, recorded rather than taken.

*Sparse jet representations.* They report ~2x time and memory from dropping zero-padding, and
use it "for all other architectures" but NOT for ParT, because its learnable attention bias has
no variable-length kernel. This repo's sparse-jet assessment concluded the effort was not
worth it; that assessment stands for our scale, but the factor they quote is larger than
anything compile gives, and it is the honest place to look next if throughput matters.

## Table-wide compile policy

`torch.compile` changes neither accuracy (numerics-preserving up to fusion order) nor
FLOPs. It changes **walltime and nothing else** -- so the only question it raises is how
the walltime column is read, and that question is much smaller than it first looks.

**Compile where it works.** Cross-paper walltime comparability is already broken and was
never real: our own ParticleNet ran 30 h against the published 25 h, a 20% gap from
hardware and I/O alone, which is larger than a typical compile gain. Refusing a real
speedup to protect a comparison that hardware has already invalidated would trade days
or weeks of GPU time on a shared 4-GPU condo for a nicety that does not exist. Don't.

The disclosure rule does all the work that is actually needed:
- **FLOPs carries every efficiency claim** -- it is compile-independent by construction,
  which is precisely why the column exists.
- **Walltime is informational, with a per-row compile footnote.** The table has carried
  mixed rows since before any of this work (`config/model/tag_slim.yaml:22` ships
  `compile: true` upstream), so mixed-and-disclosed is the status quo, not a new debt.
- A model that will not compile cleanly simply stays eager. That is the footnote doing
  its job, not a problem to engineer around.

Two real prerequisites, neither of them a reason to hold back:
- **A correctness gate before adopting a compiled row**: fusion changes float
  accumulation order, so verify same-model behavior at tolerance (the BIT/TOL gates in
  this document) rather than assuming it.
- **Ignore the first timing estimate**: compile warmup and any recompilation land in the
  early iterations, so the run's own "training time estimate" line reads high until the
  graph settles.

Scope, in the order the work naturally falls: CGENN per this document; ParT and
ParticleNet next (weaver-core supports both upstream now, so the risk is low); the
GT hybrids post-campaign, measuring ParticleNetParTGraphTrans/GPS first, since graph
breaks would come from the wrapper (per-batch kNN rebuild, PyG scatter, LLoCa transport)
rather than the backbones.

## L-GATr 2.0 trick census → CGENN (2026-08-11)

Prompted by a fair objection: the CGENN compile work above cited lgatr 2.0's *paper* but had
never read lgatr 2.0's *compile code*. Its source was opened only when a bug forced it. This
section is the census that should have come first — every compile-facing mechanism in the
installed `lgatr` package, and what CGENN does about it.

| lgatr mechanism | where | CGENN |
|---|---|---|
| `_GeometricProductSparse` custom autograd Function | `primitives/bilinear.py` | **ADOPTED** — `experiments/baselines/cgenn/sparse_gp.py` |
| `activation_memory_budget` scoped patch | `utils/compile.py` | **ADOPTED** — wrapper knob, default off |
| `warmup_caches` / `warmup_after_apply` | `primitives/compile.py`, `utils/compile.py` | **N/A by construction** — see below |
| `minimum_autocast_precision` fp32 islands | `utils/autocast.py` | **N/A** — every model config here ships `use_amp: false` |
| `generate_vmap_rule = True` on the Function | `primitives/bilinear.py` | adopted with the Function |
| `checkpoint_blocks` (gradient checkpointing) | `nets/lgatr.py`, `nets/slim.py` | **considered, not adopted** — see below |
| `compile_kwargs` passthrough (`mode`, `fullgraph`) | `utils/compile.py` | already have `compile(dynamic=True)`; `mode="reduce-overhead"` is the cudagraphs path their warmup exists for, not our posture |
| dense-vs-sparse chosen per OPERATION | `primitives/config.py` | already encoded as the `gp_impl` knob |
| `sparse_linear` | `primitives/config.py` | no CGENN equivalent exists, and by their own result we should not want one on GPU (§"Upstream's own numbers", point 1) |

The list is the whole package, not a selection: `lgatr` has 50 modules and this is every
compile- or memory-facing mechanism in them.

`checkpoint_blocks` is the one deliberate decline. It is the same lever as
`activation_memory_budget` — recompute in the backward instead of retaining — but coarser
(whole blocks, chosen by hand) where the partitioner picks by cost, and it exists in lgatr
because their nets must also work uncompiled. Every CGENN config ships `compile: true`, so
the finer knob strictly dominates it here. Recorded as a decision rather than left as an
omission; if the compile posture ever flips off, this is the fallback.

### Item 2 — the sparse GP is now a `torch.autograd.Function`

This is the one that mattered. The measured anomaly it explains was already in this log: the
sparse contraction does **16× fewer MACs** than the dense forms and still measured a **1.50×
higher GPU peak**. Cause is retention, not arithmetic. The eager three-liner

    pair = x.unsqueeze(-1) * y[..., kidx]     # (B, N, 16, 16)
    w    = weight[..., spath] * spval
    out  = einsum("bnij,nij->bnj", pair, w)

saves **two** (B, N, 16, 16) tensors for backward — `y[..., kidx]`, which the `mul` needs, and
`pair`, which the `einsum` needs — where einsum/matmul each save exactly one. lgatr hits the
same wall and solves it the same way: a Function whose `setup_context` saves only inputs, with
a hand-written backward that recomputes the gathers.

The forward inside the Function is **the same three lines, unchanged**. Grad mode is off inside
`Function.forward`, so the two intermediates become transient rather than retained: identical
output bits, no tensors held. Measured:

| | retained for backward | vs einsum |
|---|---|---|
| einsum (BIT reference) | 293.4 MB | 1.00× |
| matmul | 293.4 MB | 1.00× |
| sparse, before | 293.4 MB + the extra pair | >1× (1.50× GPU peak) |
| **sparse, now (eager)** | **84.8 MB** | **0.29×** |

Eager only — under `torch.compile` all four land within 1 MB of each other; see the
correction below.

(whole `tag_cgenn` model, fp64, the fixture batch; `GATE-SAVED` in `test_cgenn_compile.py`.
At one GP layer alone, at (B=32768, 16 features, fp32), it is 1056 MB → 64 MB, 16.5×.)

> **CORRECTION, same day, from the closing audit.** The paragraph that stood here claimed a
> 5.6x compiled memory win. It was wrong, and the error was in the measurement: the compiled
> figures came from `saved_tensors_hooks` on the FIRST call, i.e. during compilation, before
> any warm-up. Re-measured with five warm-up steps then ten timed, at model level, fp64:
>
> | gp_impl | | time | retained | vs einsum |
> |---|---|---|---|---|
> | einsum | — | 628.9 ms | 253.6 MB | 1.000x |
> | matmul | — | 603.5 ms | 253.6 MB | 0.960x |
> | sparse | eager expression | 572.4 ms | 252.8 MB | **0.910x** |
> | sparse | the Function | 1054.5 ms | 252.9 MB | **1.677x** |
>
> **Under `torch.compile` the retention difference does not exist.** AOTAutograd's partitioner
> decides what crosses the fwd/bwd boundary and lands all four impls within 1 MB of each
> other, which is the same job the Function was written to do by hand. So compiled, the
> Function keeps only its cost -- 1.84x against the expression it replaced -- and CGENN ships
> `compile: true`, which made that the row the campaign would have run.
>
> MECHANISM, established afterwards rather than assumed: with the Function forced onto the
> compiled path, its python `forward` AND `backward` are both called on EVERY step (counts
> grow 2,3,4,5,6,7 and 1,2,3,4,5,6 over six steps). Dynamo never inlines it -- the Function
> executes as a black box and its hand-written backward is interpreted op by op inside what
> is nominally a compiled step. That is the whole 1.84x.
>
> AND IT IS NOT ABOUT OUR WEIGHTS. The obvious story -- "lgatr's GP is weightless, ours
> carries a learnable per-path weight, so their Function transfers and ours does not" -- was
> tested and is FALSE. Running lgatr's own `_GeometricProductSparse` under `torch.compile`
> and counting python entries gives the identical pattern: (2,1) (3,2) (4,3) (5,4) (6,5).
> **Dynamo does not inline ANY `torch.autograd.Function`, theirs included.** So the cost is
> a fixed python-dispatch overhead per call, for everyone, and whether it is worth paying
> depends only on how much work each call does relative to that overhead -- large GPU
> batches amortize it, our CPU model-level steps do not. The weight difference explains
> something narrower: our interpreted backward computes three gradients with einsums where
> theirs computes two with elementwise ops, so we pay more of the same overhead.
>
> Note what this says about GATE-BREAKS = 0: that gate runs through `_forward`, which is
> `no_grad`-wrapped, so it describes the INFERENCE graph. It was green the entire time the
> training path was breaking around every GP layer. Same blind spot that hid the
> recompile-per-shape bug, and the third time in this document that a `no_grad` gate has
> been mistaken for a statement about training.
>
> Fixed by making the Function eager-only: `sparse_geometric_product` branches on
> `torch.compiler.is_compiling()`, so the compiled graph contains the plain expression
> (0.910x, the fastest impl measured) and eager keeps the Function (0.88x time and 0.17x
> retained memory against the same expression -- that win is real and survives). Gated by
> `test_sparse_gp.test_compiled_path_bypasses_the_function`, which exists because no other
> gate here can see it: they all either run eager, or check a property both paths satisfy.
>
> Confirmed after the fix, same harness: sparse is back to **562.7 ms, 0.901x einsum**,
> against the 1054.5 ms it read while the Function was on the compiled path.
>
> NOISE FLOOR, so nobody over-reads the table: with the fix in, both "sparse" rows execute
> the same compiled code and still differ by 9% (615.4 vs 562.7 ms) run to run on this
> box. So the 1.84x regression was far outside noise and is real, but the einsum / matmul /
> sparse ordering here is NOT resolvable — that is β-PERF's job, on a GPU, paired.
>
> What the eager numbers below still say correctly: the retention accounting, the gradient
> agreement, and the determinism argument. What they do NOT say is anything about the
> compiled posture, which is the one that matters.

**What is measured and what is not.** Every number in this subsection is `saved_tensors_hooks`
accounting on CPU, EAGER: bytes autograd *retains* across the forward/backward boundary with
no partitioner involved. See the correction above for what compilation does to it. The 1.50x
figure it explains came from a real β-PERF OOM search on an H100 (sparse 15.2 GB vs einsum
10.1 GB at batch 16, sparse OOM'ing at 128 where the others fit) — measured with
`compile: true`, so the correction above applies to it too and the OOM search must simply be
re-run. Note also that the backward's transient peak is unchanged (two (B, N, 16, 16)
temporaries live at once), so the eager win is entirely in retention.

**Determinism, where we deliberately diverge from lgatr.** Plain autograd differentiates both
gathers with `index_add_`, which is nondeterministic on CUDA — so the sparse path never was
reproducible on GPU. The hand-written backward avoids it twice:

- **dL/dy** — for a fixed left blade `i`, `j → k(i, j)` is a *bijection* (left multiplication by
  a basis blade permutes the basis), so the scatter inverts into a gather through the new
  `algebra.gp_j_idx`. Asserted at construction, not assumed: a degenerate metric would break it,
  and a silently-wrong inverse gives wrong gradients rather than a crash.
- **dL/dweight** — the `(i, j) →` compact-path map is a fixed 0/±1 matrix (35 × 256 here), so the
  segment-sum is one small GEMM.

lgatr chose the opposite and says so: *"index_add_ is CUDA-nondeterministic, but beats the
deterministic gather+matmul alternative by ~10% on CPU and ~5% on CUDA."* Their trade differs
from ours in two ways — their GP carries no weight, so dL/dy **is** their backward, and the
gather can be placed before or after the feature contraction. Placing it after is what makes it
cheap. Per dL/dy call, campaign shape, one CPU thread:

| | index_add_ (nondet.) | gather-late (det.) | gather-early (det.) |
|---|---|---|---|
| plain layer | 40.8 ms | 47.1 ms | 121.2 ms |
| fc layer | 70.1 ms | **55.0 ms** | 194.9 ms |

gather-late is bit-identical to `index_add_`, 15% slower on the plain layer (~2% of that
layer's backward) and 21% *faster* on the fc layer. Which form lgatr benchmarked is not stated,
so their number is context rather than a contradiction.

**Every einsum in the backward is TWO-OPERAND, and that is load-bearing.** The first version
used two 3-operand einsums, which is the natural way to write the contractions. `torch.einsum`
hands three or more operands to opt_einsum, whose path search reads CONCRETE sizes — so under
`torch.compile(dynamic=True)` the graph pins to them. Measured on a compiled TRAINING step over
four distinct batch shapes: **1, 2, 3, 4 unique graphs for sparse against 1, 1, 1, 1 for
einsum** — a recompile per batch shape, and jet multiplicity varies every batch, so that is a
recompile per step for the whole campaign. Rewritten as binary chains (which also lets dL/dx
and dL/dy share one intermediate): back to 1, 1, 1, 1.

This repo had already paid for that lesson once — it is why the einsum `gp_impl` is a
two-operand chain rather than the 3-operand form, and the comment saying so is in `gp.py`. It
came back because **every RECOMP gate in this tree runs under `no_grad`**, so none of them has
ever seen the joint graph. `test_sparse_gp.test_compiled_training_step_does_not_respecialize`
now does, at layer scale, ungated, in 15 s.

**Verification.** `gradcheck` and `gradgradcheck` in fp64 (the latter proves the backward is
itself differentiable — no in-place fold onto a saved tensor); forward bit-equality and
gradient agreement ≤ 4e-16 against the exact expression replaced; the `sp_val == 0` branch,
which the Lorentz metric never reaches, gated against a deliberately reduced path set;
whole-model gradients within 2.6e-10 of the einsum reference, matching `matmul`'s 2.9e-10 on
the same parameter — and matmul is untouched by this work, so the ~6 digits are the model's own
backward conditioning. Compiled: 0 graph breaks, 1 unique graph forward, 1 unique graph across
a four-shape compiled training sweep, and a compiled training step producing 37 finite
gradients.

### Item 1 — `activation_memory_budget`, shipped off

`activation_memory_budget: null` on all three CGENN wrappers (`config/` and `config_quick/`),
applied as a scoped `torch._functorch.config.patch` around the compiled call — the same shape as
the existing `recompute_views` patch on `LorentzNetLGATrSlimGraphGPSWrapper`, and for the same
reason (the smoke test and the FLOPs harness build many models per process; a global leaks).
It is an OOM escape hatch: it buys peak activation memory with backward recompute. Reach for
`gp_impl: sparse` first — after item 2 it gives 3.5× for free, which is the pressure this knob
would otherwise relieve, and it costs no recomputation.

Verified live rather than assumed — a knob that changes nothing measurable should not ship.
Retained across the fwd/bwd boundary of the **compiled** `tag_cgenn`, fp64, fixture batch.

**Caveat, from the closing audit:** these were taken on the FIRST call, before warm-up — the
same mistake the correction below documents, so read the ABSOLUTE numbers as unreliable. The
CONTROL survives it and is what this table is for: `null` and `1.0` agree exactly (1.0 is
torch's default) while `0.5` and `0.2` differ, which is the proof the scoped patch reaches
AOT's partitioner through hydra → wrapper → call. The knob is live; its magnitude is not
established here.

| budget | `gp_impl: einsum` | `gp_impl: sparse` |
|---|---|---|
| `null` (torch default) | 253.6 MB | 45.0 MB |
| `1.0` | 253.6 MB | 45.0 MB |
| `0.5` | 41.2 MB | 24.9 MB |
| `0.2` | 41.2 MB | 12.8 MB |

`null` and `1.0` agreeing is the control (1.0 *is* torch's default); `0.5` and `0.2` differing
is the proof the scoped patch reaches AOT's partitioner through hydra → wrapper → call. The
per-impl comparison that used to be drawn from this table has been withdrawn: warmed up, the
impls do not differ in retention under compile.

### Item 3 — `warmup_caches` has no CGENN equivalent to write, and here is the proof

lgatr's primitive tables are `@lru_cache` functions keyed by `(device, dtype)`. The first call
for a new pair does a host→device copy *inside* the compiled region, which under
`mode="reduce-overhead"` partitions the captured graph — hence `warmup_caches`, called from
`_apply` so every `.to()`/`.cuda()`/`.float()` re-warms.

CGENN has no such cache. Every table is a `register_buffer(..., persistent=False)` on
`CliffordAlgebra`, an `nn.Module`, so `.to()` moves it by construction and there is no first-call
path at all. That is the same fix at a different layer, and it is stronger: their warm-up
*works around* a lazy cache, ours *has no* lazy cache. Verified rather than asserted — the CGENN
stack contains exactly one `functools.cached_property` (`geometric_product_paths`), zero
`lru_cache`, and its three call sites are `.nonzero()`, `.sum()` and `.size()`, all at `__init__`.

The stronger closure is a new gate. **`test_meta_device_forward`** runs the real CGENN nets on
the `meta` device: `.to("meta")` moves exactly what `.to("cuda")` moves and leaves exactly what
`.to("cuda")` leaves, and mixing the two raises the same error the GPU raises — a GPU
placement gate on a CPU-only runner, at no cost (meta tensors carry no storage). This is what
the header of `test_device_hygiene.py` records `FakeTensorMode` failing to do; it works here
because it is applied to the **net**, with the wrapper's data-dependent preprocessing already
run for real. Two things it taught immediately:

- `_alpha_signs` is not on the live forward path, so the first version of the self-check — which
  pinned one buffer by name — passed vacuously. It now sweeps every buffer of every algebra and
  asserts at least one unregistration fails the forward. Live set: `_beta_signs`, `cayley`.
- `tag_CGENNLGATrGraphTrans` holds **two** `CliffordAlgebra` instances (`net.algebra` for
  `mv_bridge`, `net.cgenn.algebra` for the CGENN block). Only the second is device-checked;
  `mv_bridge` reaches its algebra through embed/get_grade, which are index and slice ops.

Stated blind spot: `matmul` does *not* raise on a meta/cpu mix (measured — it promotes
silently), and `meta[cpu_idx]` does not raise, exactly as `cuda[cpu_idx]` does not. So this
catches the **crash** class — all three bugs found this session — and not the **silent-cost**
class, which is what the other two nets in that file are for.

### Upstream issue text (DavidRuhe/clifford-group-equivariant-neural-networks)

Reproduced against the upstream code shape before writing (touch-then-move poisons it; the
tensor never appears in `named_buffers()`, so nothing reports it):

> `CliffordAlgebra._alpha_signs` / `._beta_signs` / `._gamma_signs` are
> `functools.cached_property`, which stores the tensor in `instance.__dict__` — bypassing
> `nn.Module.__setattr__`, so it is not a buffer and `.to(device)` cannot move it. Any access
> before the model is moved (a CPU sanity forward, a summary tool, a warm-up) pins them to CPU
> permanently, and every later `signs * mv` in `alpha`/`beta`/`gamma` raises *"Expected all
> tensors to be on the same device"*. `register_buffer(..., persistent=False)` fixes it with
> identical values and an unchanged `state_dict`.

Worth filing: latent upstream (nothing in their `__init__` touches the properties, so the
common order is the lucky one) but unconditional here, because our dynamo warm-up loop
materializes every `cached_property` at construction — before `.to(device)`. We caused the
crash; they carry the hazard.

### Adversarial audit of this change

Ten things went wrong on the way here, eight of them found by auditing rather than by a
failing test. Recorded because the pattern is the useful part.

0. **The one that would have cost real GPU time: a recompile per batch shape.** Two
   3-operand einsums in the backward pinned the compiled training graph to concrete sizes
   (1, 2, 3, 4 unique graphs over four shapes, against 1, 1, 1, 1 for einsum). Every gate in
   the tree was green, because every RECOMP measurement runs under `no_grad` and none of them
   has ever seen the joint graph. Worse, it is a lesson this repo had **already** learned and
   written down in `gp.py`'s einsum branch — the comment was two functions away from the code
   that broke it. Found by asking "RECOMP is forward-only; what does the backward do?" rather
   than by any test. Fixed (binary chains throughout), and now gated at layer scale, ungated
   and fast, by `test_sparse_gp.test_compiled_training_step_does_not_respecialize`.
1. **The first `dL/dy` I wrote was 2.6×/3.5× slower than necessary.** Gathering `w` and `g`
   before the feature contraction is the obvious reading of the math and inflates both
   operands to (B, M, 16, 16). Gathering after gives the same bits at a third of the cost.
   Found by benchmarking a decision I had already justified in a comment.
2. **The meta gate's non-vacuity proof was itself vacuous.** It unregistered `_alpha_signs`
   and asserted the forward fails — but `alpha()` is not on the forward path, so the
   assertion could never have fired. A gate that proves it can fail must prove it against
   something live; it now sweeps every buffer and reports which ones are.
3. **…and it swept the wrong object.** `tag_CGENNLGATrGraphTrans` has two
   `CliffordAlgebra` instances; `next(...)` found the one serving `mv_bridge`, which
   reaches its algebra only through index ops. Fixed by sweeping all instances.
4. **`torch.empty_like` behind a strippable assert.** `gp_j_idx` is built by scattering
   into an empty tensor, which covers every entry only because the bijection assert holds —
   and `python -O` strips asserts. The uncovered case would have been uninitialized memory
   used as a gather index: silently wrong gradients. Now `full_like(-1)`, which is an
   out-of-range gather instead.
5. **The budget context manager held one patch slot**, so any nesting or concurrency would
   have leaked a patched config. Now a stack.
6. **The gradcheck would have been deleted by the wipe.** `test_cgenn_compile.py` is a port
   instrument scheduled for deletion in `cleanup.md` step 7; putting the only gates on a
   hand-written backward inside it would have removed them along with the scaffolding — the
   exact mistake `test_compile_posture.py` was carved out to fix, repeated. Carved out again
   into `tests/experiments/test_sparse_gp.py` (KEEP), and `cleanup.md` updated.
7. **`sp_val == 0` had no coverage.** Under the Lorentz metric all 256 blade pairs land on
   an allowed grade triple, so the masked branch and its clamp-to-path-0 are dead code in
   production — and its failure mode (clamped entries dumping gradient onto path 0) is
   silent. Now gated against a deliberately reduced path set, with the test asserting that
   masking actually occurred and that a live entry shares index 0, so a clobber would show.
8. **A TF32 coupling on the new GEMM.** `dL/dweight` ends in a matmul against the 0/±1
   selection matrix; under `float32_matmul_precision: high` that would round the gradient
   to a 10-bit mantissa for no gain, since the matrix is exactly representable. The repo
   ships `highest`, so it is inert — documented at the call site rather than guarded.
9. **One reported failure was not real.** The first `test_compiled_backward[sparse]` run
   failed with an unexpected-keyword TypeError: the process had imported `wrappers.py`
   before the budget parameter was added and read the yaml after. A stale-process artifact,
   not a defect; re-run clean.

Checked and clean: no inline sparse-GP expression survives anywhere; no wrapper is
constructed outside hydra; every `CliffordAlgebra` in the tree uses the same metric, so the
new bijection assert has exactly one case and it passes; `state_dict` is unchanged
(`persistent=False` throughout, and the BIT gate's `load_state_dict(strict=True)` passes);
FLOPs are unchanged (`FlopCounterMode` counts the einsum, which moved but did not change,
and it counts the forward only); `use_amp: false` in every model config, so the dtype
promotion in the backward is inert.

Not verified, stated rather than glossed:

- **GPU peak memory.** No GPU here. The 1.50× that motivated all of this is a real H100
  number; the effect of the fix on it is a prediction. First thing for the β-PERF matrix.
- **Wall-clock.** Also β-PERF's job. The CPU dL/dy micro-benchmarks above are the only
  timing evidence, and they cover one kernel, not a step.
- **The `meta` gate cannot see the silent-cost class**, only the crash class. Stated in the
  test file and repeated here because a green run is easy to over-read.

### β-PERF, GPS/PN rows — the split-graph question, answered (2026-08-11)

`tag_PlainGraphGPS` and `tag_ParticleNetParTGraphGPS` shipped `compile: false` with the
same closing sentence in both yamls: *"Unmeasured bounded upside vs one-line reversibility
→ false until β-PERF says otherwise."* The masked BatchNorm over real nodes is
data-dependent by design and splits the graph (11 and 7 breaks, all one documented class),
and no reference ships break-laden compile as a default. Correctness was never the blocker
— both are in `BACKWARD_VERIFIED`, both twins are train-faithful.

β-PERF has now said. Operator run, GPU, paired states:

| row | eager it/s | compiled it/s | speedup | posture |
|---|---|---|---|---|
| tag_particlenet | 5.86 | 9.47 | **1.617×** | already `true` (Stage-4 gates) |
| tag_PlainGraphGPS | 4.82 | 5.86 | **1.216×** | flipped `false → true` |
| tag_ParticleNetParTGraphGPS | 5.37 | 6.22 | **1.159×** | flipped `false → true` |

The bounded upside is 15.9–21.6%. For scale: our own ParticleNet ran 30 h against the
published 25 h, a 20% gap from hardware and I/O alone — so these gains are the size of the
discrepancy that already makes cross-paper walltime incomparable, which is the argument
the table-wide policy above is built on. "No reference ships this as a default" is an
argument about defaults for other people's users, not about a measured row on our card.

PNParTGraphGPS at 1.159× is the **lowest margin adopted anywhere in the table**, well
above the driver's 3% margin but worth naming: β-PERF's pairing does not cancel the
order effect (states always run false-then-true, so the second starts on a warmer card).
If a swapped-order rerun lands under 3%, flip it back — one character, and FLOPs carries
every efficiency claim regardless.

Still open, and the reason `utils/bperf.py` is not deletable yet: the CGENN rows. Their
last sweep is the one the sparse-GP autograd Function was built to invalidate — sparse was
sized to 32 where einsum and matmul got 64.

That one wants a batch job rather than an `interact` session: `--find-batchsize` walks an
OOM ladder per state, every rung paying its own compile, three `gp_impl` rows deep — hours,
and a dropped session takes the sweep with it. Build `bperf.sbatch` from
`docs/oscar-train.sbatch` (root-level `*.sbatch` is gitignored, which is the point: the
partition and account in it are yours, not this public repo's), changing three things —
`--mem=64G` (the OOM ladder's last surviving batch has a host-side collate to match), `-t
8:00:00`, and the command:

    sbatch -J bperf-cgenn bperf.sbatch --models tag_cgenn --find-batchsize \
        --iters 1010 --window 100 1000

`--find-batchsize` is not optional for these rows: without it every row uses the unswept
512 yaml fallback, and the sizes the OOM search chooses ARE the finding here. Read them
before the speedups. Do not pass `--apply` from a batch job — it edits production yamls,
and the point of reading the table first is to decide whether the flip is one you want.

## β-PERF, CGENN rows — the memory inversion, measured (2026-08-12)

H100, `--find-batchsize --bs-safety 1.0`, top-tagging, 322,689-parameter CGENN. The OOM
search runs **eager** and sizes one batch that both states then share.

| eager probe peak | einsum | matmul | sparse |
|---|---|---|---|
| bs 16 | 10.1 GB | 10.1 GB | **6.0 GB** |
| bs 32 | 17.5 | 17.5 | **10.4** |
| bs 64 | 37.3 | 37.4 | **22.2** |
| bs 128 | 80.2 | 80.2 | **47.6** |
| bs 256 | OOM | OOM | OOM |
| chosen | 128 | 128 | 128 |

**The ratio inverted, and this is the result the sparse-GP autograd Function was built
for.** It previously read sparse 15.2 GB at bs 16 against einsum's 10.1 — 1.50× WORSE, and
sparse got half the batch of the others. It now reads 6.0 against 10.1: **0.59×**, i.e.
einsum needs 1.68× sparse's memory, and all three size to the same 128. The earlier
CPU-measured eager win (0.17× retained at layer scale, 3.46× whole-model) transfers to the
H100. Note WHY it counts despite the Function being eager-only: `find_max_batch_size` runs
an eager training step, so the eager path is what sets the batch size both states run at.

Throughput at the shared bs 128:

| gp_impl | eager it/s | compiled it/s | verdict |
|---|---|---|---|
| einsum | **OOM at step ~10** | 1.41 | INCOMPLETE (eager row missing) |
| matmul | 1.04 | 1.36 | compile: true |
| sparse | 0.46 | 1.25 | compile: true |

Two things follow, and they pull in opposite directions.

**1. Compiled, einsum is fastest — 1.41 vs sparse's 1.25, +12.8%**, far outside β-PERF's 3%
margin. On throughput alone at a shared batch, einsum wins and `gp_impl: einsum` would be
the posture.

**2. But einsum OOM'd in a real eager run at the batch its own probe chose.** The probe said
128 fits at 80.2 GB of 93.09; ten steps into real training it needed another 11.30 GiB and
died. `find_max_batch_size`'s docstring warns about exactly this ("it probes one batch per
size; verify the chosen batchsize with a short real run"), and this is that warning firing.
sparse at the same 128 peaked 47.6 GB — 45 GB of headroom — and its eager run completed.

**Do not flip `gp_impl` on this table.** The batch was sized EAGER and shared across impls,
which is right for a paired speed comparison and wrong for the campaign question, because
`find_lr` will size each model on the SHIPPED config — compiled, per impl. Compiled peaks
are lower than eager ones, and sparse starts 1.68× lower, so sparse may well take a larger
batch than einsum can. jets/s at each impl's OWN batch is what decides, and no one has
measured that. Sequence: run `find_lr` per impl, compare jets/s (it/s × batch), then flip
if einsum still leads.

**Why the probe missed it. FIRST EXPLANATION HERE WAS WRONG, corrected by measurement.**
It said the padded length varies 87-110, that CGENN memory goes as `B * P²`, and so a probe
batch could be (110/87)² = 1.60x optimistic. Two errors. The dominant term is not `P²`: the
dense tensors are linear in `P_max`, the `pair` mask is `B * P_max²` but one byte per
element, and what actually dominates is the EDGE tensors, which scale with `sum_j n_j²` over
REAL nodes and are independent of padding. And `sum n²` is a sum of B i.i.d. terms, so it
CONCENTRATES rather than varying 1.6x. Measured over 400 random batches of the top-tagging
length distribution (mean n = 49.2, max 135):

| B | `P_max` p99/p50 | `sum n²` p99/p50 |
|---|---|---|
| 4 | 1.55x | 2.10x |
| 32 | 1.33x | 1.25x |
| 128 | 1.28x | **1.16x** |
| 512 | 1.24x | 1.07x |

So at the batch that OOM'd the dominant term varies only **1.16x** p50→p99, not 1.60x. What
actually happened is thinner and less exotic: einsum's probe peak was 80.2 GB of 93.09, i.e.
86% utilisation with 12.9 GB of headroom, and a p99 batch alone eats ~13 GB of it. The OOM
message then names the rest -- *"8.93 GiB is reserved by PyTorch but unallocated"* --
fragmentation, and suggests `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` itself. A
one-batch probe at the median plus 14% headroom is simply not enough margin, for any model
sized that close to the card.

**Operational consequence for `find_lr`, independent of which impl wins:** its batch probe
was the same one-step probe, and it was observed choosing a batch that dies in real training.
The einsum row above is what a campaign looks like without a fix: dead at step 10, hours in.

**FIXED, and not with `bs_safety`.** The two terms the probe was blind to are the two above,
and each now has its own answer rather than a shared fudge factor:

* *A median batch stands in for the worst of ~10^5 draws.* `find_max_batch_size` no longer
  DRAWS its probe batch, it CONSTRUCTS one: `bs` real jets whose total `sum n^2` sits
  `+lr_find.bs_sigmas` (default 5) standard deviations above a typical batch, with the
  dataset's longest jets included so `P_max` lands at its cap too. Against the simulated
  worst batch of a 50-epoch run that is accurate to ~1% at every batch size (1.313 vs 1.307
  at B=128; 1.053 vs 1.047 at B=4096). One probe per rung, same cost, deterministic — so a
  25-recipe sweep is reproducible. Gates in `tests/experiments/test_probe_batch.py`.
* *Fragmentation grows with run length and a one-step probe cannot see it.* That is what the
  OOM message's own 8.93 GiB is, and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is
  what addresses it. Two rules on that variable, the second added after review pointed out
  the first was not enough: the probe and the job must use the SAME allocator, AND the
  setting is a per-CAMPAIGN decision taken before the first row, because it moves walltime
  and walltime is a reported column (the `-n` / `num_workers` note in
  `docs/oscar-train.sbatch` already states this rule for a different knob). Mid-campaign,
  leave it: a fragmentation OOM is a recoverable single-row failure, a split walltime column
  is not.

`bs_safety` stays, but it is now for JetClass/TopTagXL only — iterable datasets expose no
per-item lengths, so the probe there falls back to one random batch and logs that it did.
For top-tagging, `bs_safety=0.5` on top of the constructed probe would double-count: it was
~2x headroom for what measurement says is a 1.05-1.31x problem.

**And a third thing, larger than either, that the investigation turned up by accident.** The
constructed probe changes the chosen batch in only **28%** of card sizes (measured over 60,
spanning two octaves) — the rest of the time its 1.05-1.31x vanishes into the doubling
search's power-of-two granularity. That granularity is itself the bigger loss: the search
brackets the true ceiling between `2^k` and `2^(k+1)` and then throws the whole octave away,
returning on average 1/1.5 of what the card holds. `+lr_find.bs_refine=true` bisects the
bracket in 3 more probes for a mean **1.35x** batch (median 1.25x, up to 1.88x). Off by
default, because a round batch size is what makes the recipe table comparable. Gated
end-to-end against a simulated card in `tests/experiments/test_probe_batch.py`, which is also
the first test this search has ever had.

End to end, in batch size, over the same 60 card sizes:

| posture | vs old default |
|---|---|
| `bs_safety=0.5` (what this doc used to recommend) | 0.50x |
| old default | 1.00x |
| new default (constructed probe) | **0.82x** |
| new default + `bs_refine=true` | **1.08x** |

So the safety the constructed probe buys costs 18% of the batch, and refining more than pays
it back. Refined-vs-recommended is 2.16x.

**None of which are speedups.** `jets/s = batchsize / step time` saturates once the card is
compute-bound, and nobody here had measured where that happens for these models — the
"sparse got half the batch, i.e. half the jets/s" line above is exactly that assumption,
and it is only true in the un-saturated limit. `find_max_batch_size` now times its second
step at every rung and prints the jets/s curve, so the next `find_lr` run answers it as a
side effect. If the curve is flat across the top rungs the card is saturated, `bs_refine`
is not worth turning on, and the sparse-vs-einsum batch-size argument above loses its force
in the same stroke.

**Three alternatives considered and not built, with why:**

* *Size to the median and make the training loop OOM-resilient.* Recovers the 18% and
  survives the rare heavy batch. But the naive form — catch the OOM, skip the batch — is a
  PHYSICS bug, not an engineering one: the batches that OOM are systematically the ones with
  the most constituents, which correlates with top jets, so it silently drops signal. The
  sound form (re-run the failed batch as two halves with gradient accumulation) is unbiased
  but is a change to the core training loop, and recovering cleanly from an OOM raised inside
  backward under AMP is fiddly. Not the week before a campaign.
* *Token-budget batching* — build batches to a fixed `sum n` rather than a fixed jet count,
  as seq2seq training does. This is the principled answer: the variance disappears entirely
  and you can run at the true ceiling with no headroom at all. It also makes the number of
  jets per step vary, which changes gradient noise step to step and is a confounder in a
  25-recipe comparison. Right answer for a single production model, wrong one here.
* *Per-moment sigmas.* At B=128 the constructed batch is ~10% heavier than it needs to be
  because `sum n` (which concentrates more slowly) sets `k`. Tightening this only matters at
  small batches, which is not where the campaign runs.

### The sweep spent ~95% of its wall clock on evaluation it never read

`run_once` passed `save=false` but not `evaluate=false`, and `config/default.yaml` ships
`evaluate: true` -- so every timed run followed its 110 training iterations with a full
test+val pass, 1578 batches, for numbers the driver reads none of. The row wall-clocks give
it away: matmul 125 min and sparse 119 min both COMPLETED, while einsum took 41 min because
it crashed before ever reaching evaluation. Fixed in `utils/bperf.py`; timing comes from the
"Finished iteration N" lines during TRAINING, so no measured number moves. Expect rows in
minutes rather than hours on the next sweep.

### lgatr's FUSED form: measured, and REJECTED for CGENN. Do not retry it.

lgatr's sparse GP is not just the sparse contraction -- it is that contraction written as a
SINGLE fused expression, `(signs * y[..., indices] * x.unsqueeze(-2)).sum(-1)`, which their
comment says "fuses to a single kernel under torch.compile (no 16x16 buffer), which is both
faster and far lighter on GPU". We took the contraction and the autograd Function but kept
the einsum spelling, which materializes `pair`. So: take the fused spelling too?

No. Three measurements, in the order they were made, because the first two were misleading
and the third is the one that decides.

**(1) Plain layer, B=2048, compiled, fwd+bwd -- looks like a clear win.**

| | time | retained |
|---|---|---|
| einsum | 250.2 ms | 72.0 MB |
| fused | **213.2 ms** | **8.0 MB** |

0.85x time, 0.11x retained.

**(2) Which layer actually ships? Not that one.** `CGENN.__init__` defaults to
`layer_type="fc"` and no yaml overrides it, so every CGENN model -- `tag_cgenn` and both
hybrids -- instantiates `FullyConnectedSteerableGeometricProductLayer` and nothing else
(counted: 2, 2, 4). `SteerableGeometricProductLayer` appears ZERO times, behind the
never-taken `layer_type="gpmlp"` branch. The layer the win was measured on is dead code.

**(3) On the fc layer, and on the plain layer at a matched shape, it loses.** Same B=512,
compiled, fwd+bwd:

| layer | einsum | fused | ratio |
|---|---|---|---|
| plain | 37.1 ms | 40.9 ms | **1.10x slower** |
| fc | 18.3 ms | 258.1 ms | **14.11x slower** |

So the plain-layer "win" in (1) is shape-dependent -- it reverses between B=2048 and B=512
-- and on the shipped fc layer the fused spelling is an order of magnitude worse. Rejected.

**Why, stated only as far as it was actually checked.** The first explanation written here
was that the fc contraction has GEMM structure the fused spelling throws away. That is
WRONG and the dispatch trace says so: BOTH spellings' einsum path dispatches
`aten.bmm.default`, plain and fc alike. What is left is the arithmetic-shape argument: the
two forms have the same MAC count, but einsum routes them through a tuned bmm while the
fused spelling routes them through a generated loop nest, and the gap between those two
depends on the reduction. One reduction axis (plain) is close, 1.10x. The fc form's extra
output-feature axis `m` multiplies the elementwise work by 16 before reducing over two
axes, and there the generated nest is 14x off the tuned kernel.

That also settles a question raised earlier in a way worth keeping separate from the
autograd-Function finding:

  * *Why is the autograd Function not inlined under compile?* NOT the weights. Dynamo
    inlines no `torch.autograd.Function` at all -- lgatr's own included, measured.
  * *Why does the fused spelling not transfer?* The output-feature axis that our learnable
    weight introduces, but by inflating the elementwise work, NOT by creating GEMM
    structure the fused form loses -- both spellings reach bmm. lgatr's GP is
    `(..., 16) x (..., 16) -> (..., 16)` with no `m` at all.

**(4) One more form, in the same spirit: keep the tuned bmm but loop over OUTPUT BLADES**,
so the live intermediate is `(B, N, 16)` instead of `(B, N, 16, 16)`. Bit-identical to
einsum (rel = 0.000e+00 -- the per-`j` decomposition preserves each output element's
reduction order). fc layer, B=512, compiled: 24.1 ms vs einsum's 15.7 (**1.53x slower**),
and **retained memory IDENTICAL at 18.51 MB**. The 16x smaller instantaneous intermediate
buys nothing, because autograd saves all sixteen per-`j` slices and they sum to exactly the
same bytes. Rejected. (Eager it is 1.92x FASTER, which is real but lands on the path the
autograd Function already owns.)

**What the four measurements add up to.** Retention is set by what the BACKWARD needs, and
every spelling of the same contraction needs the same information; the only lever is
RECOMPUTING it instead of saving it. The fused form gets its 9x by being a pointwise-reduce
that inductor is willing to recompute -- and pays 14x in time for giving up the bmm. So the
memory win is real but is not reachable by re-spelling the kernel at acceptable cost.

It IS reachable the other way, and that is already shipped: recompute chosen by the
PARTITIONER rather than by the kernel spelling -- `activation_memory_budget` on the CGENN
wrappers (default null), and the custom autograd Function on the eager path. Those are the
two mechanisms that survive, and between them they cover both postures. There is no third.

Nothing to implement. Recorded at this length because measurement (1) is genuinely
attractive and someone who stops there will reach the wrong conclusion -- as this document
did, for one commit, before (2), (3) and (4) were run.

---

## Compiled PEAK per gp_impl — the number the posture decision was waiting on (2026-08-12)

The plan for choosing `gp_impl` was: *"compiled peaks are lower than eager ones, and sparse
starts 1.68x lower, so sparse may well take a larger batch than einsum can. jets/s at each
impl's OWN batch is what decides, and no one has measured that."* That plan has a premise —
that sparse's eager memory advantage survives compilation — and the CORRECTION block above
already says the opposite for RETENTION. Retention is not peak, though, and it is peak that
sets the batch size, so the premise was open. Measured now.

Model level, tag_cgenn fixtures, fp64, CPU, three warm-up steps before any read. Retention
via `saved_tensors_hooks`; peak via the profiler's allocation records (a proxy — see the
caveat below).

| gp_impl | retained eager | retained compiled | peak eager | peak compiled |
|---|---|---|---|---|
| einsum | 293.44 MB | 253.63 MB | 2941 MB | 852.15 MB |
| matmul | 293.44 MB | 253.63 MB | 1901 MB | 852.15 MB |
| sparse | 84.84 MB | 252.76 MB | 3626 MB | 849.61 MB |
| **spread max/min** | **3.459x** | **1.0034x** | **1.907x** | **1.0030x** |

**Compiled, the impls are indistinguishable in BOTH retention and peak — 0.3% apart.** The
eager spread is 3.5x in retention and 1.9x in peak; compilation erases both.

*Why this is trustworthy where the eager peak column is not:* this harness reproduces two
independently recorded numbers to the decimal — the eager retention pair (293.4 -> 84.8,
3.46x, from the sparse-GP Function work) and the compiled retention row (253.6 / 253.6 /
252.8, from the CORRECTION). Both were measured months apart by different code. So the
instrument is calibrated against known answers before being read for a new one.

*The caveat, stated plainly:* the eager PEAK column disagrees in DIRECTION with the H100
(which measured eager sparse at 47.6 GB against einsum's 80.2, i.e. 0.59x, where this reads
1.23x). Do not trust the eager peak column. The disagreement is probably scale, not
instrument: these fixtures are batch 4 in fp64 against the H100's batch 128 in fp32, and the
Function trades retention for transients — its backward RECOMPUTES the two (B, N, 16, 16)
tensors it declined to save. At batch 4 one layer's transient dominates; at batch 128 the
retained activations of every layer dominate and the trade pays. Both measurements can be
right at their own scale. The compiled columns are the ones this section is for, and there
the two metrics agree with each other and with the prior retention measurement.

**What it means for `gp_impl`.** If compiled peaks are equal, every impl sizes to the SAME
batch, and "jets/s at each impl's own batch" collapses to "it/s at a shared batch" — which
beta-PERF has already measured on the H100: **einsum 1.41, matmul 1.36, sparse 1.25**, einsum
ahead by 12.8%, far outside the 3% margin. On that reading the campaign posture is `einsum`,
not `sparse`, and the memory argument that has protected `sparse` through three revisions of
this document does not survive `compile: true`.

**NOT FLIPPED, and the reason is not caution for its own sake.** This is CPU inductor; the
campaign is Triton on an H100. The partitioner's min-cut is shared between backends and is
what drives retention (hence the exact reproduction), but peak also depends on inductor's
buffer reuse and scheduling, which are backend-specific. One GPU command closes it, and it
is the same command either way:

    for i in einsum matmul sparse; do \
      python utils/find_lr.py -cn toptagging model=tag_cgenn \
      model.net.gp_impl=$i save=false +lr_find.find_batch_size=true; done

Deliberately NOT bounded by `+lr_find.bs_max=512`, unlike the recipe-filling sweeps in
GUIDE/OSCAR/SLURM: this one measures each impl's memory ceiling, and the ceiling IS the
finding (the three impls sized 64/64/32 last time, which is why their throughputs were not
comparable). A cap would hide exactly the number being asked for. Harmless in practice too —
these OOM far below 512 — but do not "fix" it to match the others.

`find_lr` sizes on the SHIPPED (compiled) config, unlike beta-PERF which forces the knob off,
and it now prints peak AND jets/s per rung. Three outcomes and what each means:

- *Same batch for all three* — confirms this table on GPU. Ship `einsum`; it is 12.8% faster
  at that batch. Update `gp_impl` in the three CGENN configs and the pin in
  `tests/experiments/test_lgatr_migration_parity.py`, which asserts `sparse` deliberately.
- *sparse sizes larger* — the CPU result does not transfer; compare jets/s at each own batch.
- *jets/s flat across the top rungs for all three* — the card is saturated, batch size is not
  the lever, and the decision is 12.8% of throughput on the einsum row alone.

**Do NOT answer this with another β-PERF run — it structurally cannot.** `find_batchsize`
composes `f"{knob}=false"` (`utils/bperf.py`), so the OOM search ALWAYS runs eager and both
states then share that one batch. That is deliberate and correct for what β-PERF reports (a
paired eager-vs-compiled ratio needs one batch), but it means every β-PERF run, however many
times it is repeated, hands the compiled rows a batch chosen by an eager probe. Compiled
peaks are the lower ones, so the compiled batch is bigger than the eager-sized 128 the
throughput table above was measured at — and a ranking at 128 is not guaranteed to hold
there. `find_lr` is the tool that differs on exactly this point: it leaves the compile
posture to the yaml, and every model config carrying a `compile` key ships it TRUE (20 of the
36 in `config/model`; the only `compile: false` in the tree is the framesnet sub-configs,
eager in every posture), so it sizes the SHIPPED model.

β-PERF is still worth one rerun for its OWN job, just not for this one: einsum's eager row is
missing (it OOM'd at step ~10, so that row has no eager/compiled ratio), and the
`evaluate=false` fix has cut its wall clock by roughly 20x since that table was produced.

Worth noting what is NOT in doubt: `gp_impl=sparse` is not "rejected for compiled CGENN". It
is what ships, it runs compiled, and the eager-only branch inside
`experiments/baselines/cgenn/sparse_gp.py` is about the autograd FUNCTION wrapper, not the
sparse contraction. What this table questions is narrower and older — whether sparse is the
right *choice* among three working impls, once compilation has erased the memory difference
that was its whole case.

---

## Upstream's compile workflow, applied as far as a CPU allows (2026-08-12)

From the lgatr/lloca author, asked about compiling CGENN:

> *"I didn't work much with the CGENN code so it would take me quite a while to get into it
> and do this properly. Also, its just a speedup thing, so assuming that the point of your
> project is performance gains I think you can regard this as a small bonus that you can also
> do later on. My torch.compile-improvement-workflow is to first use torch.profiler to find
> all CPU-GPU synchronizations (you need a GPU for that) and fix them (this is the main
> timing win usually), and afterwards look for rewrites to enable fast kernels everywhere
> instead of 4x4 or so kernels (this is what gave the lloca speedup in lloca v2.0)."*

Three things follow, in decreasing order of how much they change what we do.

### 1. His step 2 is a result we already have, and it points at `einsum`

*"Rewrites to enable fast kernels everywhere instead of 4x4 or so kernels"* is exactly what
the gp_impl measurement found, from the other direction. The sparse geometric product is
**provably minimal-arithmetic** — the Cl(1,3) Cayley table has 256 nonzeros of +-1 out of
16^3, one `k` per `(i, j)`, all 256 products on distinct component pairs, nothing factorable
— and compiled on the H100 it is the SLOWEST of the three: einsum 1.41 > matmul 1.36 >
sparse 1.25 it/s. Doing 16x fewer multiplies in a gather-shaped kernel loses to doing 16x
more in a GEMM.

So his heuristic and our measurement agree, and both point the same way: prefer the dense
form. That is independent evidence for the `gp_impl: einsum` flip that the per-impl `find_lr`
loop is meant to confirm.

### 2. His step 1, the sync census, is done as far as it can be without a GPU

Static sweep over model code (nets, wrappers, per-step embedding — not the experiment
drivers, where reading results back to the host is the point). Gated in
`tests/internal/test_no_device_sync.py` so new ones cannot appear silently.

| site | verdict |
|---|---|
| `if bad.any():` — `wrappers.py`, `jet_frames` | **one per-forward sync.** Hits the two GraphTrans wrappers that set `compute_jet_frames` (ParticleNetParT, Plain), regardless of framesnet |
| `dev.max().item()` in the same warning | benign: behind that `if` AND a once-per-process latch |
| `mask.tril(offset).nonzero(...)` — ParT PairEmbed | eager pair path only; `compiled_dense=True` routes the compiled posture to the dense twin |
| `tril_indices` | builds a static index from `torch.ones(...)`, at init |

**The one real site has a bit-identical fix.** Its fallback is already branchless —
`torch.where(bad[:, None, None], eye, trafo)` returns `trafo` unchanged when `bad` is
all-False — so hoisting it out of the `if` removes the sync without touching arithmetic; the
`if` then guards only the once-only warning, which can be bounded to the first N steps.

**Not applied.** The campaign has started, and the gain is unmeasured — a sync costs more
than its own stall (it stops CPU run-ahead, so the queue drains and the GPU idles between
kernels, which bites hardest in exactly this repo's many-small-kernels shape), but "more than
nothing" is not a number. Measure first, with the recipe below, then apply post-campaign.

### 3. The recipe for the GPU half

**This recipe ships as a tool: `utils/profile_sync.py` — run that inside the GPU
allocation rather than pasting the snippet** (same build path as `find_lr`, production
tree, shipped compile posture, smoke-tested end-to-end on CPU). The snippet below stays as
the specification of what the tool does and why.

**Corrected 2026-08-14 — the first version had three faults that would have produced the
wrong list if followed as written.** (a) The loop sat INSIDE the `profile(...)` block, so
the table it prints is dominated by compilation and autotune, not steady state — the
comment said "warm first" but the code did not. (b) It sorted by `self_cuda_time_total`,
which ranks hot KERNELS — the step-2 question — and can never surface a stall: syncs live
in CPU-side rows (`cudaStreamSynchronize`, `Memcpy DtoH`, `aten::item`) and as gaps on the
GPU timeline. (c) It profiled `_batch_loss` + `backward` only, which by construction cannot
see the per-step DRIVER reads — and for a sync hunt the driver is in scope: `loss.item()`
and the non-finite guard's `isfinite(grad_norm)` fire every step (base_experiment `_step`),
the tracker metrics add per-key `.cpu().item()` reads on learned-frames rows, and the
grad-norm histories append per-step DEVICE tensors to python lists (~0.5 GB at allocator
granularity over a 330k-step run, plus a host read whenever they are consumed). The static
census scoped the driver out deliberately for its own reasons; the PROFILER must not.

    from torch.profiler import profile, schedule, ProfilerActivity
    # scaffolding as in train(): metric lists, scheduler, model.train(); it = iter(loader)
    for step in range(8):                # warm OUTSIDE the profiler: compilation,
        exp._step(next(it), step)        # autotune, allocator growth, trimmer warm-up
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 schedule=schedule(wait=1, warmup=2, active=5),
                 with_stack=True) as prof:   # with_stack names the python line per stall
        for step in range(8, 16):
            exp._step(next(it), step)        # the REAL step -- driver reads included
            prof.step()
    # step-1 view (sync hunt): stalls are CPU-side rows and timeline gaps
    print(prof.key_averages(group_by_stack_n=5)
              .table(sort_by="self_cpu_time_total", row_limit=30))
    prof.export_chrome_trace("trace.json")   # gaps between kernel spans = drained queue
    # step-2 view (hot kernels), the ranking the old sort produced -- still wanted, second
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))

Read it for two things, in his order: `cudaStreamSynchronize` / `Memcpy DtoH` events between
kernels (step 1), then the kernel-size histogram — many sub-10us kernels is the 4x4 problem
(step 2). Do this on ONE model first; `tag_PlainGraphTrans` is the natural pick, since it is
one of the two carrying the known sync. The driver reads listed above will appear in every
model's table — they are known, not rediscoveries; the fix shape is ONE host read per step
(stack loss + norms, a single `.cpu()`) and CPU-float histories, gated bit-identical.

**Timing, and his own framing of it:** *"just a speedup thing … a small bonus you can also do
later on."* That settles the order. Profiling is read-only and can be done any time; the
rewrites change arithmetic or timing and therefore wait for the campaign to finish.

---

## Non-GPU half #2 executed, and CPU priors for the GPU verdicts (2026-08-14)

The section above names two halves doable without a GPU. The first (static sync census) was
done; the second — *"reading inductor's generated code to find the small-kernel sites"* —
was claimed doable but never done. Done now, plus retention priors for two of the queued
GPU verdicts. Environment caveat on all of it: **torch 2.13 CPU inductor** — the pinned NGC
2.8 build dies on this path with the native `double free` that
`utils/vram_compile_matrix.py` records (2.13 compiles it cleanly), and C++ codegen means
the fusion GROUPING and per-kernel op mix transfer to Triton approximately, exact splits
may not.

### The kernel census (compiled tag_CGENNLGATrGraphGPS, quick tree, gp_impl=sparse, dynamic=True)

**One forward graph (75 kernels) + one backward graph (54) — no breaks, no recompile
across batches.** At quick's `num_blocks: 2` that is ~52 kernels per block, so the
production `num_blocks: 10` model runs **~500-550 kernel launches per training step** —
the launch-bound shape, counted rather than inferred. The shortlist for the step-2
histogram, by fused-op signature:

1. **Pure marshalling, no arithmetic**: `clone_permute_squeeze_view` (5 defs) and
   `_unsafe_view_clone_permute_unsqueeze_view` (5 defs) — layout shuttling that survived
   fusion as standalone kernels. Candidates for layout unification (keep one multivector
   layout through the block so these fuse away entirely).
2. **The blade contractions**: the `bmm_cat_index_mul_permute_[slice_unsqueeze_]view`
   family (~9 defs fwd+bwd) — `bmm` over the 16-wide blade dim plus the path gather.
   This is literally the *"4×4 or so kernels"* class from the quote; the batching
   candidate is folding blades into channel GEMMs.
3. **Their backwards**: `index_add_new_zeros_*` / `index_index_put_*` — scatter-shaped,
   small, atomics-bound on CUDA.

Everything else in the two graphs is unremarkable (norms, activations, plain GEMMs).

*Post-rewrite recount (2026-08-14, after the Phase-1 executed items below):* 64 fwd + 62
bwd kernels. The forward's marshalling class shrank as predicted (75 → 64); the backward
picked up small weight-side kernels (`diag_embed_threshold_backward`,
`diagonal_index_add_new_zeros_*` — the block-diagonal weight's construction/gradient,
(m·16, n·16) tensors, trivial traffic). Node-level: marshalling ops attributed to the two
rewritten files 547 → 92, bmm 116 → 54, total aten nodes −24%. Launch-count parity on the
card is a gate-day question; the activation-path work per launch is what dropped.

**CORRECTION (2026-08-15, CPU-weakness audit):** the counts above — and the original
"~500-550 launches per production step" extrapolation — counted only `cpp_fused_*`
kernel definitions and MISSED `extern_kernels.*` calls (mm/addmm/bmm dispatched to
ATen, one launch each). Full accounting: pre-Phase-1 **435 launch sites** (129 fused +
306 extern) for the 2-block quick graphs → production extrapolation **~2,100-2,200
per compiled fwd+bwd**, not 500-550 — which also reconciles far better with the
H100-measured ~7,200 launches/step (the remainder being eager wrapper ops, optimizer,
and memcpys outside the compiled graphs). Post-Phase-1+2.2a: **393 sites** (125 fused +
268 extern; the −38 extern calls are the MVLinear per-blade bmm batches that became
single flat GEMMs).

### Retention priors: `activation_memory_budget` brings the sparse-Function trick to the compiled path

The 2026-08-13 VRAM finding (compiled GPS wants MORE than eager) has a mechanism already on
record — the partitioner flattens every impl to einsum-class retention — and a shipped but
never-measured lever. Measured (quick tree, fp64, `saved_tensors_hooks`; retention is the
partitioner's cut, so the RATIOS are backend-independent even though these are CPU numbers):

| posture | retained for backward | step time (CPU, rough) |
|---|---|---|
| eager, sparse (the pre-compile default) | ~44 MB | — |
| compiled, budget unset (= 1.0, ships) | 122.98 MB | 186 ms |
| compiled, budget 0.5 | **29.38 MB** | 202 ms (+8.6%) |
| compiled, budget 0.2 | 20.54 MB | 206 ms (+10.8%) |

Budget 0.5 does not merely recover the eager-sparse profile — it undercuts it, at
single-digit recompute cost on CPU. Combined with the throughput table (compiled einsum
1.41 > sparse 1.25 it/s), the candidate posture is **`gp_impl: einsum` +
`activation_memory_budget≈0.5`**: fastest kernels AND smaller-than-eager-sparse retention,
i.e. the batch-size ceiling and the speed row stop trading against each other. The GPU
check composes directly onto `utils/vram_compile_matrix.py`: the CGENN wrappers accept the
knob in both trees, so the axis is one override — `model.activation_memory_budget=0.5` —
per row; sweep {einsum, sparse} x budget {1.0, 0.7, 0.5}, then `find_lr` at the winner.

Upstream posture, checked against the installed package: lgatr 2.0.0's `compile_model`
also ships `activation_memory_budget=None` (torch's save-everything default) — an escape
hatch there too, not a tuned value. Its two other memory levers are not wired here at all:
`checkpoint_blocks` (block-level gradient checkpointing) and `naive_amp`.

---

## First GPU sync-hunt results (2026-08-14, H100 NVL, bs=64, full set, shipped postures)

`utils/profile_sync.py`, three models. Caveats that bound every number below: `with_stack`
inflates host-side times (per-op observer cost x thousands of ops), and bs=64 is below the
campaign regime, so host-side fixed costs are OVER-represented relative to a real run.

**tag_PlainGraphTrans — both queued sync fixes are NOT WORTH APPLYING, measured.** All
explicit stalls together — 2110 `aten::item` + 250 `cudaMemcpyAsync` + 220
`cudaStreamSynchronize` (jet_frames' among them) — cost ~1.3 ms of a 136 ms step, ~1%.
The `jet_frames` hoist (§"his step 1" above) and the driver-read consolidation are hereby
measured into irrelevance at this batch; the verdict flips only if a real-batch profile
says otherwise. The actual profile shape: CUDA busy ~12.5 ms of the 136 ms step — at THIS
batch the model is host-bound (dynamo wrapper + ~1,200 launches + cudnn heuristics), which
amortizes at campaign batch sizes and says: profile at the real batch before drawing
walltime conclusions for this row.

**tag_cgenn (sparse, compiled) — GPU-bound at ~92% (393 of 428 ms/step), and the CUDA
split confirms the kernel census's shortlist as the cost, on the card:** blade `bmm` 49.5%
of CUDA (177 calls/step), `index_put/new_zeros` scatter backwards ~17%, elementwise/copy
~14%, `clone` marshalling 6.5% — plus ~800/step `gemmk1`/`gemvx` calls (K=1-shaped
micro-GEMMs, the literal "4x4 kernels" class) at ~20% combined. Step-2 rewriting (fold
blade contractions into channel-batched GEMMs) has real headroom here: ~70% of GPU time
sits in small-contraction shapes. Sync fixes are irrelevant on this row too — the 218
ms/step of `cudaStreamSynchronize` is the CPU *waiting for a busy GPU*, not a stall to
remove.

**tag_CGENNLGATrGraphGPS — CRASHED in the compiled backward: a campaign blocker until
rerun-verified.** Inductor's runtime stride guard, on a later batch of the warm-up:

    assert_size_stride(permute_134, (s27*s77, 16, 1, 1), (1, 7168, 7168, 1))
    AssertionError: expected size 16==16, stride 6656==7168 at dim=1

A saved permuted multivector VIEW whose stride was specialized to one batch's width
(7168 = 448*16) while its sizes stayed symbolic — the LNetSlimGraphGPS view-saving class,
now observed as a batch-shape-dependent RUNTIME crash rather than a compile-time
InductorError, on the exact path every CPU gate is documented blind to. The shipped
`compile: true` row can die mid-training on an unlucky shape transition. FIX shipped
(wrappers.py: the scoped `recompute_views` patch mirrored from LNetSlim, eager default
bit-identical, verified against the audit digests) and **VERIFIED the same day: the rerun
survived all 16 compiled steps over varying shapes and produced the clean profile below.**
Longer soak before the first long run: `CGENN_SMOKE_STEPS=100 CGENN_SMOKE_COMPILE=1` on
the training-smoke gate, on the card.

**The GPS profile (fix active, bs=64, mini).** Read no absolute walls — `with_stack` on
~100k events/step inflates host time badly; unprofiled it/s belongs to β-PERF. The CUDA
split (533 ms/step busy) is the tag_cgenn picture, amplified: blade `bmm` **43.9%** (532
calls/step), scatter backwards (`index_put_new_zeros`) **17.8%**, copy/clone marshalling
**~20%**, and **1,834 `gemmk1`/`gemvx` calls per step at 21.4%** — K=1-shaped micro-GEMMs.
The L-GATr attention half does not crack the top-25 at mini shapes (re-check at production
P_max). ~7,200 kernel launches per step. torch itself printed "TensorFloat32 ... available
but not enabled" into the run log.

---

## The improvement program (operator sequencing: CGENN core → CGENN-GPS → LNetSlim-GPS)

Grounded in the two GPU profiles above and the kernel census. Per item: which POSTURE it
touches, its correctness CLASS, and the gate that judges it. The eager path is not exempt
work — everything source-level below reaches eager and compiled alike (the hybrids import
the shared package, so core fixes flow into both hybrids automatically), and the eager
BIT/TOL discipline is what keeps the campaign's reference rows meaningful.

### The stride crash is genuinely upstream, and what to expect from PyTorch

Three facts. (1) The LNetSlim instance was PROVEN upstream by monkeypatch (`334be77`):
`torch/_inductor/graph.py` calls `ir.get_stride_order(strides)` without the shape_env, so
sympy expressions hit a bare Python comparison; passing the live shape_env fixed compile
AND training. The runtime-guard variant seen here (concrete stride baked into
`assert_size_stride` while sizes stay symbolic) is the same specialize-what-should-be-
symbolic family, of which pytorch/pytorch has a long public record (e.g. issues #104653,
#125641, #136837, #143121, #143579 — per-op fixes land release by release). (2) Do not
expect a backport: PyTorch backports only critical fixes to the current release branch,
and the campaign's torch is an NVIDIA-frozen NGC build — the fix channel is a container
upgrade, not a patch to 2.8. (3) Reporting it upstream is still worth one issue with the
assert signature and the `s27*s77`-symbolic/7168-concrete pair; it improves the version
this repo migrates to next. (4) Measured on 2.13: the same model, compiled `dynamic=True`
with the `recompute_views` shield forced OFF, survived 25 varying-shape fwd+bwd steps —
where the 2.8 build crashed within ~10. CPU inductor there vs Triton on the card, so this
is strong-but-not-conclusive evidence the instance is fixed in newer torch; it is exactly
the release-by-release pattern the public issue record shows, and it means the shield is a
2.8-era measure with a natural retirement point at the next container upgrade.

### Phase 1 — CGENN core (both hybrids inherit via the package import)

Ordering rule (operator, 2026-08-14): items that REMOVE WASTE at unchanged numerics come
first; the one item that TRADES precision for speed comes last, and is not a CGENN item at
all but a table-wide decision — see item 4.

1+2. **EXECUTED 2026-08-14 as two TOL-class rewrites** (operator relaxed the original
   BIT constraint the same day: "doesn't have to be perfectly bit accurate as long as
   it's not a bug or accuracy reduction"). Items 1 (layout unification) and 2
   (blade-contraction batching) collapsed into one move each at their two real sources,
   found by fx-node attribution of the compiled GPS graph:

   - **`CliffordAlgebra.b()`/`q()` diagonal collapse** (`cliffordalgebra.py`). The
     scalar-output geometric product is a signed dot product — `cayley[:, 0, :]` is
     exactly diagonal (asserted at `__init__`; the diagonal per grade is the metric
     signature), so every `norms()`/`qs()` invariant is
     `(x * _qb_diag * y).sum(-1)` with the beta signs fused into one ±1 buffer. This
     retires the einsum chain, the per-call cayley subset gather, and two per-call
     index-tensor allocations from the single hottest marshalling site (every
     MVSiLU / MVLayerNorm / NormalizationLayer funnels through here).
   - **`MVLinear` block-diagonal flat GEMM** (`linear.py`, rank-3 fast path, both
     subspace variants). The einsum `"bmi,nmi->bni"` lowered to a per-blade bmm — 16
     `(B,m)@(m,n)` micro-GEMMs *per call* plus permute-copies of the activation both
     ways; that is the `gemvx`/`gemmk1` micro-GEMM swarm in the H100 profiles. The
     weight is now `diag_embed`-ded into one `(m·16, n·16)` block-diagonal matrix and
     the call is ONE flat GEMM with zero activation copies (16× zero-padding FLOP waste
     is nothing at mv widths 1–16; weight layout / state_dict unchanged; `diag_embed`
     backward is a diagonal gather — deterministic).

   *Measured (CPU sandbox, torch 2.13):* fx census of compiled GPS fwd+bwd —
   marshalling nodes at the two sites **547 → 92 (−83%**, remainder is weight-side
   `(m,n,16,16)` tensors, not activations); bmm nodes **116 → 54**; total aten nodes
   4730 → 3572 (−24%). Eager per-op wall at production-ish widths: grade-subset `b()`
   **10.1×**, full `b()` 2.0×, MVLinear 1.7×. Model-level EAGER step (CPU, quick tree,
   bs=64, old worktree vs new): tag_cgenn 47.5 → 43.3 s/step (**1.10×**), Trans hybrid
   5.26 → 4.33 s/step (**1.22×**) — the modest eager dividend expected of a rewrite
   whose target is marshalling and launch count, not arithmetic volume (eager CPU cost
   at these batch sizes is dominated by the edge-tensor gp/segment work the rewrite
   leaves alone). The compiled-GPU launch path is where the −83% node cut should pay;
   that verdict belongs to the gate day. *Numerics:* lab fp64 rel ~3e-16
   (grades 0/1/3/4 bit-equal; g2 + full call differ at ulp); model-level fp64
   eval-forward rel vs pre-rewrite — tag_cgenn 2.5e-11, Trans hybrid 2.3e-15, GPS
   hybrid **0.0 (bit-equal)**; worst grad (BACKWARD-TOL's `absdiff/(1+ref)` metric)
   1.1e-9 / 5.4e-16 / 1.6e-16 — all inside the repo bars (fwd 1e-10, grad 1e-8).
   *Gates:* all 17 non-BIT cgenn_compile gates green including env-gated
   BREAKS/RECOMP (no graph breaks, no per-shape recompiles) and compiled backward;
   sparse_gp + device-hygiene + no-sync suites green; 3-model eager training smoke
   green (100% nonzero-grad params). *Post-landing audit (same day):* eager-path
   instrumentation of all three models, forward+backward — every `b()`/`q()` call
   takes a diagonal path and every MVLinear call is rank-3; ZERO einsum fallbacks,
   so the census numbers describe the whole production surface. The re-recorded
   dynamo explain shows tag_cgenn's traced op count fell 223 → 155 at 1 graph /
   0 breaks. *(2026-08-15, CPU-weakness audit round: instrumentation REPEATED on the
   PRODUCTION config tree — the quick-tree-only run was itself a coverage gap — same
   verdict, zero fallbacks, and the call counts match design exactly: GPS runs 20
   hoisted mean-reduces = 10 blocks × 2, 104 diagonal b() calls, 40 rank-3 MVLinear
   calls per fwd+bwd. Model-level fp32 drift closed retroactively by diffing the
   recorded fixtures themselves — recorded state_dict and batch verified
   bit-identical first, so the y drift is pure computation: fp32 rel 4.4e-05, fp64
   2.6e-11, the fp64 number independently cross-validating tol_dump's 2.46e-11 and
   fp32 sitting ~50x under the ~2.5e-3 fp32 impl-vs-impl noise floor the
   BACKWARD-TOL docstring records. And the 2.13-only gap: on a PUBLIC torch
   2.8.0+cu128 CPU venv — the closest public cousin of the NGC campaign build — the
   diagonal b() measures fp64 rel 1.5e-16 vs the old chain, and a CGL(fc, sparse,
   edge_counts) toy compiles at 1 graph / 0 breaks through 4 varying-shape compiled
   fwd+bwd steps. Residual version gap: the NVIDIA fork itself + Triton-vs-C++
   codegen — gate day's soak still owns those.)* *Fixtures:* cgenn_compile BIT fixtures + hybrid
   pins re-recorded under the stated class change — only tag_cgenn and the Trans pin
   actually changed bits; GPS and both LNet pins came out byte-identical.
   *Still open for the GPU gate day:* walltime verdict (β-PERF); whether the shields
   can retire — now MEASURED rather than presumed (audit refinement, same day): the
   joint graph's stride-permuted saved tensors went **71 → 45**. Of the survivors,
   29 are lgatr-package-internal (unchanged by Phase 1, and present in lgatr rows
   that have run compiled on the card without this crash class), 12 are
   weight-shaped/static (the wbd and `w` permutes — no symbolic dims for the guard
   to specialize against), and **4 are activation-shaped with symbolic batch dims —
   all inside `sparse_gp_expression`'s einsum**, the shipped GP. So the crash-class
   candidate population fell ~8x and is concentrated in ONE function, but it is NOT
   zero: the shield stays until the `CGENN_SMOKE_STEPS=100` shield-off soak passes
   on the card. **Contingency corrected by its own lab (2026-08-15,
   `sparse_fix_probe.py`): respelling that einsum does NOT clear the hazard.** A
   compiled toy probe counted permute-defined saves per surface form — einsum 2
   (1 symbolic-shaped), explicit contiguous-bmm 3 (2 symbolic — WORSE: AOT saves
   the pre-contiguous view and recomputes), blockdiag flat-GEMM 2 (1 symbolic —
   neutral). The partitioner chooses the saves regardless of how the contraction
   is spelled, so if the shield-off soak fails at this site the correct mechanism
   remains the scoped `recompute_views` shield (already in place), with the
   partitioner knob as the escalation path — NOT a source rewrite. (All three
   respellings verified TOL ~1e-15 fp64 anyway, so the perf-motivated version can
   still be raced on the card if the gate-day profile ranks this site.) Also open: a
   gp_impl re-race — the old "batching the GP alone gets eaten by surrounding
   marshalling (matmul 0.960x)" result predates this rewrite, so
   einsum/matmul/sparse may reorder now that the surroundings are clean. The largest
   remaining CGENN-side marshalling site is `sparse_gp_expression` (48 permute + 12
   bmm nodes, unchanged by this pass — it is the shipped GP itself, per-path weights
   make its batching delicate): batch it only if the gate-day profile still ranks it.
3. **`activation_memory_budget` adoption** — *Posture:* compiled-only by nature. *Class:*
   posture change, numerics untouched. *Gates:* the vram matrix (budget is one override
   per row) + β-PERF ratio; CPU priors above (0.5 → 29 MB vs 123 default at +9% CPU).
4. **TF32 — LAST RESORT, and table-wide or not at all.** Demoted from first on the
   operator's rule, for two reasons that outrank its ceiling. It is the only item here
   that DECREASES precision (matmul/bmm inputs rounded fp32→tf32, 23→10 mantissa bits;
   accumulation stays fp32, range unchanged; ~1e-3 per-op relative error vs fp32's ~1e-7 —
   not the bf16-AMP upstream measured harmful, but a strict reduction nonetheless). And
   flipping it for one model family is an UNFAIR row in a cross-architecture table — the
   same class as `expandable_segments` ("a walltime knob: campaign-uniform or absent, not
   mid-flight"): every model's GEMMs would speed up under TF32, so granting it to CGENN
   alone converts an architecture comparison into a knob comparison. If items 1-3 leave
   CGENN still unacceptably slow, the decision is an OPERATOR one for the whole table,
   judged by the protocol: (a) equivariance/invariance floors under TF32 for every
   equivariant row; (b) 2-3 seeds x ~2k-step seeded A/B within seed noise, per family;
   (c) β-PERF ratios. Priors to carry into that decision: the `highest` pin's stated
   reason (the sparse Function's exact-0/±1 `sel` GEMM) is eager-only and inactive under
   the shipped compiled posture, and the K=1 micro-GEMM swarm gains nothing from TF32 —
   the ceiling is on the bmm/mm ~50% share only.

### Phase 2 — CGENN-GPS specific (after Phase 1)

1. **Block-glue layout — RE-SCOPED by the post-Phase-1 audit (2026-08-14; refined
   same day with an honest copy/view split).** Post-Phase-1, the glue's (hybrid-file
   + GPS-file) traced nodes split as: **244 view-class nodes (zero bytes moved — the
   (B,P,C,16)↔(B·P,C,16) branch crossings are stride arithmetic), 24 gathers
   (`x[i]`, `x[j]` — message-passing semantics), and 33 true copies, every one of
   which is semantic**: message cats (`[h_i, h_j, h_i−h_j]` must materialize for
   the MLP GEMM), aggregation scatters, invariant concat. **Zero layout-conversion
   copies** — the mechanism item 1 hypothesized is absent from the traced program,
   and mv_bridge already routes through the rewritten MVLinear.
   **Scope of this evidence, stated precisely:** the census is a torch-2.13 fx
   trace + CPU-inductor kernel count. Node-CLASS facts transfer to GPU (a view
   allocates nothing on any device; the aten graph is backend-independent) —
   kernel formation, Triton fusion splits, launch counts, walltime shares, NGC-2.8
   decomposition drift, and library-internal transposes on non-contiguous operands
   (invisible to fx) do NOT transfer. So this evidence kills the *mechanism*, not
   the possibility of *any* GPU-side glue cost — which is exactly why the item
   becomes a gate-day check (act if the post-Phase-1 GPU profile attributes copy
   kernels to the glue) instead of either a blind rewrite or a deletion.
2. **Static-edge plan reuse + deterministic aggregation** — edges are batch-static and
   hoisted once, but every block re-runs gather/scatter against them; the scatter
   backward is 17.8% of CUDA.
   - **2a EXECUTED 2026-08-14 (BIT-class): receiver-degree hoist.** `mean` aggregation
     (the live mode in all three models) rebuilt its per-node degree by scattering a
     FULL (E, C) ones tensor inside every reduce call — 2 per CGL layer — for a value
     that is channel-constant and depends only on the static edges. Now ONE (E, 1)
     ones-scatter per CGENN(-block) entry (per GPS net for all its blocks), threaded
     as `edge_counts` through `CGL.forward`/`reduce`/`unsorted_segment_mean` in the
     package and both hybrid twins. Bit-identical — degrees are exact small integers
     whatever the summation order, and the (N, 1) broadcast divides by the same
     values — VERIFIED the strong way: the freshly recorded BIT fixtures and all four
     hybrid pins pass UNCHANGED (no re-record), twin-parity suite green, and the
     8-step training smoke reproduced the pre-hoist loss trajectories
     digit-for-digit. This also removes the degree count from the
     CUDA-nondeterminism surface for free. *Compiled-graph accounting (census, quick
     GPS):* AOT had already CSE'd the cross-block duplicate count scatters (4
     logical → 2 unique, one per stream width), so the compiled delta is those two
     full-width (E, 128)/(E, 72) scatters collapsing to one (E, 1) — ~200x less
     count-scatter traffic, kernels 126 → 125, dynamo op count 155 → 154. EAGER had
     no such CSE: all 2·n_layers per-call ones-scatters ran every forward there
     (8/step at production depth), so the eager dividend is larger.
   - **2b (open, gate-day — prototype VERIFIED 2026-08-14): sorted-segment main
     scatter.** The enabling facts are now measured, not inferred. (1) Receivers are
     sorted on REAL batches from both builders — asserted over bs=64 mini batches:
     fully-connected tag_cgenn (E=203,784) and kNN GPS (E=13,016), every batch
     nondecreasing (structurally: kNN is arange-expanded then order-preserving
     filtered; fully-connected is `pair.nonzero` row-major). (2)
     `torch.segment_reduce(data, "sum", lengths=…)` vs the current `index_add_` on
     sorted ids: **bit-equal forward AND backward** on CPU at fp32+fp64 (same
     accumulation order when ids are sorted) — re-verified 2026-08-15 with 322
     FORCED-empty segments (padded nodes receive zero edges in every real batch;
     the first lab's random ids made empties astronomically unlikely, so the claim
     was untested exactly where real data lives): still bit-equal, empty rows
     exact 0.0, grads bit-equal; on CUDA the swap REPLACES
     nondeterministic atomics with a fixed-order reduction — that is the whole
     point, and vs a CPU reference it is TOL at worst. (3) Compiles clean on torch
     2.13: `dynamo.explain` = 1 graph / 0 breaks, compiled fwd+bwd runs with finite
     grads, with `lengths` passed as a tensor input (in adoption, computed once in
     the EAGER edge hoist as `bincount(recv)` — never in-graph). Lab:
     **2.8 verification (2026-08-15): the "one unknown"
     is substantially closed** — on a public torch 2.8.0+cu128 CPU venv,
     segment_reduce is again BIT-equal to index_add_ forward+backward including 322
     forced-empty segments, and compiles at 1 graph / 0 breaks with a working
     compiled backward. Residual gap is only the NVIDIA fork delta + Triton codegen:
     the lab is ported into the repo as **`utils/seg_reduce_probe.py`** (torch-only,
     zero repo imports — runs inside the container as-is, and on CUDA additionally
     reports segment_reduce's own run-to-run determinism); one run there clears it. Adoption
     stays conditional on the gate-day profile still ranking the data scatter
     after 2a. The same 2.8 probe run also checked the partitioner's saved-view
     behavior: the CGL toy saves 8 permute-defined tensors (2 symbolic-shaped) on
     2.8 — same class as 2.13 — consistent with the shield staying until the
     shield-off soak.
     Note for Phase 3: `lorentznet.py:157` carries the identical
     rebuild-counts-per-call `unsorted_segment_mean` (one mean per LGEB x-stream),
     so BOTH 2a and 2b transfer to the LorentzNet family verbatim if its profile
     warrants Phase 3 work.
3. **Attention-half re-check at production P_max** — invisible at mini shapes; before
   optimizing it, profile once at the full set's P_max=160 and the sized batch.
4. **Shape bucketing + `reduce-overhead`** — only if still launch-bound after 1-2
   (~7,200 launches/step today); biggest lift, RECOMP discipline applies, last.

### Phase 3 — LNetSlim-GPS (conditional: "if there is anything to optimize")

Status first: it is IN THE CLEAR for the known crash class — its `recompute_views` shield
has been active since d1ef83b and covers every view in the net's scope, so the runtime
variant seen on CGENN-GPS is suppressed there by the same mechanism. What it has NOT had
is the same GPU soak (no multi-batch compiled-training run on the card is on record);
give it the identical cheap verify — one `profile_sync` run — which doubles as the
"is there anything to optimize" answer: if its profile shows the marshalling/micro-GEMM
signature (it converts layouts twice per layer — the very trait that made it the first
stride-crash victim), the Phase-1/2 items apply; if it profiles clean and GPU-bound in
big GEMMs, it is done and Phase 3 is empty.

### GATE DAY — the runbook (2026-08-15; everything still open needs the card)

Every CPU-closable question is closed; each command below answers a specific recorded
one. Setup as in the profiling sessions: `interact -q gpu -g 1 -n 4 -m 32g -t 2:00:00`,
apptainer with `$IMG`, the venv, repo `main`.

0. **2.2b safety on the NGC build** (~2 min): `python utils/seg_reduce_probe.py` —
   torch-only, runs as-is in the container. Expect the 4-PASS verdict it prints
   (already green on 2.13 and public 2.8.0 CPU); on CUDA it also reports
   segment_reduce's run-to-run determinism, which is 2.2b's whole point.
1. **Compiled training soak, shields ON** (default; ~15 min for the three rows):
   `CGENN_COMPILE_GATES=1 CGENN_SMOKE_COMPILE=1 CGENN_SMOKE_STEPS=100 python -m pytest
   tests/experiments/test_training_smoke.py -q -s -k "tag_cgenn or CGENNLGATr"`.
   Gate: finite losses, 100% nonzero-grad params, no stride crash — this is the
   Phase-1 + 2.2a merge check on real Triton.
2. **Shield-OFF soak — the retirement test** (GPS rows):
   `CGENN_RECOMPUTE_VIEWS_SHIELD=0 CGENN_COMPILE_GATES=1 CGENN_SMOKE_COMPILE=1
   CGENN_SMOKE_STEPS=100 python -m pytest tests/experiments/test_training_smoke.py -q -s
   -k "GraphGPS"` (covers the LNetSlim twin too — same knob). NOTE: only meaningful
   from 2026-08-15 onward — earlier the soak replayed one batch and could not
   exercise the shape-guard class (round-2 correction below). Survives 100
   varying-shape steps → the shields retire (delete both wrapper blocks + the knob);
   crashes → keep them (4 recorded candidates, all in sparse_gp's einsum; respelling
   does NOT clear them — escalate via the partitioner knob, not a rewrite).
3. **profile_sync per row** (the decision-maker):
   `python utils/profile_sync.py -cn toptagging model=<row> save=false
   training.batchsize=<sized>` for tag_cgenn, tag_CGENNLGATrGraphTrans,
   tag_CGENNLGATrGraphGPS, and tag_LorentzNetLGATrSlimGraphGPS. Reads: does any COPY
   kernel attribute to the glue (→ revisit 2.1, else it stays closed); is the data
   scatter still ranked after the degree hoist (→ flip 2.2b, protocol in its entry);
   does LNetSlim-GPS show the marshalling/micro-GEMM signature (→ Phase 3 opens, and
   2a+2b transfer verbatim to lorentznet.py, else Phase 3 is empty); is it still
   launch-bound (→ bucketing, last).
4. **The vram matrix row** (`docs/oscar-vram.sbatch`) → the `activation_memory_budget`
   adoption decision (Phase 1.3), with the CPU retention priors above as the
   expected shape.
5. **β-PERF walltime verdict + the gp_impl re-race** (the surroundings are clean now;
   the 0.960x result is stale): the find_lr loop documented in `tag_cgenn.yaml`.
6. **TF32 stays parked** — last resort, table-wide or not at all, operator protocol in
   Phase 1 item 4.

### Gate day, round 1 — executed 2026-08-15 on an RTX A6000 (partition caveat)

Steps 0-3 ran on `gpu1501` (Ampere A6000, sm86), not the H100 the campaign uses — the
runbook's `interact` line didn't pin a partition. What transfers and what needs the
H100 redo:

- **Step 0 (seg_reduce probe) — VALID, transfers.** On the real NGC build
  (`2.8.0a0+34c6371d24.nv25.08`): BIT-BWD pass, empties pass, **CUDA run-to-run
  determinism of segment_reduce PASS**, compile 1 graph / 0 breaks with finite
  compiled grads → "2.2b SAFE on this build". (BIT-FWD read "not bitwise, rel
  4.5e-07" exactly as the probe predicts on CUDA — index_add_ is the
  nondeterministic side there; the software properties are arch-independent.)
- **Steps 1-2 (soaks) — POSITIVE, but do not retire the shields on this alone.**
  All CGENN rows passed the 100-step shields-ON soak, and the shield-OFF soak
  passed for all four GraphGPS rows (`-k "GraphGPS"` matched Plain/PNPT too — bonus
  coverage; both shielded models ran unshielded, clean). The stride crash is a
  host-side compile artifact so this is real evidence — but the original crash
  fired ON the H100 within ~10 varying-shape steps, and autotune/kernel selection
  differ per arch. Rerun steps 1-2 on the H100 partition (~30 min) before deleting
  the shield blocks.
- **Step 3 (profiles) — numbers are device-shaped; redo on H100 for decisions.**
  What the A6000 profiles already establish qualitatively (bs=64, mini):
  * **tag_cgenn (626 ms/step, GPU-bound):** the mean-aggregation DATA scatter
    (`index_put_mul_new_zeros`, 17.1% CUDA) and the sparse-GP pair
    clone+reduce (`fused_clone` 17.6% + `index_mul_sum` 9.4%) and bmm 33% own the
    step. → 2.2b is still ranked after 2.2a, and `sparse_gp_expression` is now the
    top CGENN-side kernel target (its TOL respellings are lab-verified; race them
    if the H100 confirms the ranking).
  * **CGENN-GPS (2535 ms/step, ~40% GPU):** same trio — scatter 17.5%, clone
    14.6%, sparse reduce 8.3%, bmm 27%.
  * **Trans hybrid and LNetSlim-GPS: wall >> Self-CUDA** (1876 vs ~247 ms/step;
    1084 vs ~84) — the host/launch-bound signature (thousands of
    cu/cudaLaunchKernel calls + dynamic-shape wrapper overhead), CAVEAT:
    `with_stack` profiling inflates host time, so confirm via the chrome trace's
    GPU-row gaps on the H100 before acting. If it holds, Phase 2.4
    (bucketing/reduce-overhead) is the lever, and **Phase 3's answer is "LNetSlim
    has no CGENN-style kernel problem"** — its scatters are ~3-6% and its CUDA
    side is conv/GEMM; its cost is the same host tax as every GPS row.
  * Pre-existing, noted: PlainGraphGPS hits an inductor backend failure
    (`aten.nonzero` from the boolean-mask BatchNorm at plaingraphgps.py:191-192)
    that dynamo converts to a tolerated graph break; model trains. Not from this
    program's changes.

### Gate day, round 2 — H100 NVL, 2026-08-15 (the decision round)

Steps 0-3 redone on the campaign card, plus step 5 (β-PERF + gp_impl race + find_lr).
Decisions taken and corrections found:

- **Step 0 on H100: identical PASS** — 2.2b cleared on the exact campaign
  card+build (segment_reduce deterministic run-to-run, compile clean).
- **CORRECTION — the soaks were vacuous for the crash class, my flaw.** The smoke
  gate replays ONE batch (`data = next(iter(loader))` before the loop — a deliberate
  overfit/capacity signal), and the soak knob I added reused that loop: a "100-step
  soak" ran 100 SAME-SHAPE steps, so the batch-shape-dependent stride-guard class it
  exists to catch could never fire. Rounds 1-2's shield-off passes are therefore
  weak evidence only. FIXED: with `CGENN_SMOKE_STEPS` set, the gate now draws a
  fresh loader batch per step (default 8-step behavior unchanged). **The shields do
  NOT retire yet.** The real retirement test, after pulling this commit:
  `CGENN_RECOMPUTE_VIEWS_SHIELD=0 CGENN_COMPILE_GATES=1 CGENN_SMOKE_COMPILE=1
  CGENN_SMOKE_STEPS=100 python -m pytest tests/experiments/test_training_smoke.py
  -q -s -k "GraphGPS"` (~15 min; now genuinely varying widths).
- **Step 3, H100 shares (bs=64, mini; the numbers that decide):**
  * tag_cgenn — 229 ms/step, ~93% GPU-bound. Data scatter
    `index_put_mul_new_zeros` **27.3%** of CUDA, bmm 24.9%, mm 18.9%, sparse-GP
    pair clone 11.7% + reduce 6.5%. → **2.2b flip condition met → ADOPTED**
    (see below). sparse_gp_expression (~18%) is the next kernel target; its TOL
    respellings are lab-verified and can be raced.
  * CGENN-GPS — 1306 ms/step, only ~28% GPU. Scatter **25.5%**, mm 26%, bmm 18%,
    clone 12%, reduce 5.3%. The other ~72% of the step is HOST time in the
    compiled wrappers (~3,100 launches/step + dynamic-shape size computation).
  * Trans hybrid — 928 ms/step, ~10% GPU. LNetSlim-GPS — 539 ms/step, ~5% GPU
    (optimizer is 21% of its tiny CUDA!). → **Phase 3 confirmed kernel-side EMPTY**;
    the GPS/Trans family's dominant cost on the H100 is the HOST tax, i.e.
    **Phase 2.4 (bucketing / reduce-overhead) is now the biggest remaining lever**,
    with the usual caveat that `with_stack` profiling inflates host time — the
    chrome-trace GPU-row gaps are the confirmatory read.
  * 2.1 block-glue: **CLOSED** — no glue-attributed copy kernel appears in any
    H100 top table (the clones are sparse-GP pair materializations, cgenn-side).
- **Step 5, β-PERF + races (full dataset, own-batch sizing):** compiled wins
  everywhere measured — einsum 4.62 it/s @ bs64, **matmul 4.75 @ 64**, sparse 2.22
  @ 128 (own-batch jets/s: **matmul 304 > einsum 296 > sparse 284**); find_lr's
  compiled rungs at bs128 agree (matmul 199 > einsum 193 > sparse 152 jets/s).
  → **RECOMMENDATION: flip tag_cgenn `gp_impl: sparse` → `matmul`** (operator
  flip; BIT fixtures re-record mechanically). The find_lr loop also converged on
  **bs=128, lr=5.57e-04** for tag_cgenn across all three impls.
  **CGENN-GPS β-PERF row INCOMPLETE:** the eager-sized bs=256 OOM'd under compile
  (91.9 GB) — the documented compiled-retention inversion (user's pre-campaign vram
  table: 2.01-2.05× for GPS). → **Phase 1.3 activates for this row:** set
  `model.activation_memory_budget=0.5` for compiled GPS (CPU prior: 29.4 MB vs
  123 default at +8.6% step) or rerun β-PERF at bs≤128; the budget knob exists on
  the wrapper already.
- **Phase 2.2b ADOPTED (this commit).** `unsorted_segment_mean`'s hoisted-counts
  path now aggregates via `torch.segment_reduce` over the sorted receivers in both
  twins — bit-equal on CPU (the pre-swap BIT fixtures + all hybrid pins pass
  UNCHANGED, again the strong verification), deterministic on CUDA. The sortedness
  invariant it depends on is machine-checked by the new
  `tests/experiments/test_edge_builders.py` (kNN + fully-connected builders,
  adversarial masks, plus degrees==bincount tying 2.2a to 2.2b). Revert condition:
  if the next profile shows the segment kernel slower than the scatter it
  replaced, revert this one commit — determinism was the primary motive, walltime
  the expected bonus.

### Gate day, round 3 — H100, 2026-08-15: the fixed soak catches the crash; shield verdict FINAL

The corrected vary-batches soak did exactly what it was built for, in one round:

- **Shields ON, 100 varying-shape compiled steps: all three CGENN rows GREEN.** This
  is the first genuine varying-shape compiled-training soak on the campaign card, and
  it covers the full merged posture — Phase 1 rewrites + 2.2a degree hoist + 2.2b
  segment_reduce (in-graph). The shielded posture is soak-stable. Loss traces drift
  across batches (0.6899→0.5744 etc.), confirming the gate now varies data.
- **Shields OFF: CGENN-GPS CRASHED**, with the classic runtime stride-guard
  signature, at `assert_size_stride(slice_31, (4*s27, 16, 4), (1, 976, 244))` vs
  actual strides (1184, 296) — 976 = 61·16 vs 1184 = 74·16: a saved activation VIEW
  whose strides bake one batch's padded width. Two vacuous rounds "passed" this;
  the fixed soak caught it within 100 varying steps. **Verdict: the CGENN-GPS shield
  is PROVEN load-bearing on the NGC 2.8 build. Retirement REJECTED until the
  container upgrade.** The knob stays (it is the retirement test's switch and just
  demonstrated its value).
- **Hazard-class CORRECTION (audit of my own census):** the live trigger is a
  SLICE-defined save (grade-slice of an edge multivector in the invariants chain),
  not one of the 4 permute-defined saves my census counted — the census's
  "permute/transpose-defined" filter UNDERCOUNTED the class. The true class is
  "saved views with padded-width-dependent strides, whatever op defines them"
  (slices included). No code action — the scoped shield covers the whole class by
  construction; the census methodology note stands corrected here.
- **LNetSlim-GPS passed shields-off** over 100 varying steps — positive evidence,
  but its shield STAYS: same defect family, near-zero cost, one 100-step sample
  against a campaign of thousands. Both shields now sit untouched until the next
  NGC container, where the 2.13 evidence says the family is fixed.
- Still open for round 4 (GPU): profile_sync on tag_cgenn + GPS to PRICE 2.2b (the
  27.3%/25.5% scatter kernel should be replaced by segment_reduce kernels — confirm
  present and compare share); β-PERF GPS row rerun with
  `model.activation_memory_budget=0.5`; the operator flips (gp_impl matmul, CGENN
  bs=128/lr=5.57e-04, GT rows back under the ceiling with finder reruns).

### What remains is measurement-gated or campaign-frozen — the in-campaign endgame

Why no further model code follows round 3 (recorded because the operator rightly
asked):

1. **Phase 2.4 (bucketing / reduce-overhead) is CAMPAIGN-FROZEN, newly realized.**
   Bucketing pads batches to a small set of widths so compiled graphs specialize —
   but tag_cgenn and the CGENN hybrids run their `theta_h` BatchNorm over PADDED
   nodes (dense B·P layout "as upstream"), and tag_cgenn's readout mean divides by
   the padded width (the documented official-repo quirk). Changing the padded width
   therefore CHANGES THOSE ROWS' ARITHMETIC, and the campaign rule ("anything that
   changes model arithmetic is off the table until it finishes") applies. cudagraphs
   / `reduce-overhead` need static shapes = the same freeze. The GPS-family host tax
   (72-95% of the step) is real and measured — and it is a POST-CAMPAIGN project.
2. **The sparse-GP respelling is measurement-gated BY DESIGN.** Unlike 2.2b it
   carries no determinism win — speed is the only motive — so the shipped GP does
   not change on a CPU hunch. The race is now a 2-minute standalone probe:
   **`utils/sparse_gp_race.py`** (torch-only, runs in the container; einsum vs
   ctg-bmm vs blockdiag at GPS-kNN and tag_cgenn-FC shapes, fwd+bwd CUDA-event
   timing, prints an adopt/close verdict at a >10%-everywhere bar). CPU smoke:
   blockdiag at 0.50-0.67x of einsum — promising, NOT the decision input.
3. **Everything else open is an operator flip** (gp_impl matmul; CGENN bs=128 /
   lr=5.57e-04; GT rows re-swept under the ceiling) or already flipped here
   (`tag_CGENNLGATrGraphGPS.yaml` budget 0.5 — the escape-hatch condition fired in
   round 2 and the knob's own comment defines exactly this trigger; numerics
   untouched).

**Round 4 — the last in-campaign GPU round (three commands):**

    # 1. price 2.2b (the scatter kernel should be gone from both tables)
    for M in tag_cgenn tag_CGENNLGATrGraphGPS; do
      python utils/profile_sync.py -cn toptagging model=$M save=false data.dataset=mini training.batchsize=64
    done
    # 2. the un-blocked GPS beta-PERF row (budget 0.5 now in config)
    python utils/bperf.py --models CGENNLGATrGraphGPS --find-batchsize
    # 3. the sparse-GP race verdict
    python utils/sparse_gp_race.py

After round 4 the program's in-campaign scope closes: adopt-or-close on the race,
revert-or-keep on 2.2b pricing, and the operator flips. Post-campaign (or at the
container upgrade): bucketing/host-tax, shield re-test, TF32 table-wide protocol.

**ParticleNet LR-sweep anomaly (2026-08-15, unresolved — rerun before transcribing).**
Today's H100 sweep read steepest 2.21e-04 with loss-min/10 8.19e-03 flagged
NOT-reliable (no interior minimum). Both statistics sit far below the finder's own
recorded nine-rerun evidence on the SAME model and batch (steepest 1.32-1.91e-3, a
1.4x rerun spread; loss-min/10 2.64e-2-1.39e-1): steepest is ~6-8x below its
documented envelope, and both moved DOWN together — a curve-level (conditions)
shift, not a selector question. The audit's cross-revision trajectory checks
verified ParticleNet's arithmetic unchanged across the range, so the candidates are
unseeded rerun variance beyond its documented spread, hardware (the nine reruns'
device is not recorded), or sweep posture. The sweep is unseeded by design
(the rerun spread IS the stability diagnostic), so the protocol is cheap:
    for i in 1 2 3; do python utils/find_lr.py -cn toptagging model=tag_particlenet \
        save=false +lr_find.find_batch_size=true; done
    python utils/find_lr.py -cn toptagging model=tag_particlenet model.compile=false \
        save=false +lr_find.find_batch_size=true   # posture control
If reruns scatter back to ~1.3-1.9e-3, today's reading was a tail draw — transcribe
from the cluster. If 2.2e-4 reproduces, the eager-control run splits
posture-vs-environment, and the finding graduates to its own investigation. Do NOT
transcribe today's pair either way: it fails the tool's own >10x rule (37x) with no
interior minimum, which per the docstring remedy means neither number is a recipe.

### Workflow: are the gates a fair check for this program? Assessment and the additions

What exists and suffices: the BIT/TOL/DET class taxonomy with gates per class; β-PERF for
paired throughput verdicts; the vram matrix for peak; the measure-first discipline (which
just correctly killed two queued "optimizations"); device hygiene; the posture gate.

Three gaps, each now closed or defined:

1. **No GPU tier existed** — every numerics/compile gate is CPU, and the stride crash
   proved CPU-green ≠ GPU-safe. Defined: the GPU GATE DAY, run on the card before adopting
   any compiled-posture change: `CGENN_COMPILE_GATES=1 CGENN_SMOKE_COMPILE=1
   CGENN_SMOKE_STEPS=100 pytest tests/experiments/test_training_smoke.py -k <touched rows>`
   (the smoke gained the soak knob for exactly this) + the vram matrix row + a
   `profile_sync` run per touched row. ~15 min per row.
2. **The hybrids' BIT pins were wiped** (da497a9) — BIT-class rewrites had no machine
   check. Closed: `tests/experiments/test_hybrid_bit_pin.py` (skips until pins are
   recorded; record with `CGENN_COMPILE=record` in the suite's own environment before the
   first rewrite).
3. **No accuracy protocol for TOL-class changes** — defined above (floors + seeded A/B +
   ratio), operator sign-off per change. It is deliberately not a pytest gate: it costs
   GPU-hours and judgment, and pretending otherwise would make it a gate nobody runs.
