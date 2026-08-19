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

### Gate day, round 4 — H100, 2026-08-16: 2.2b priced (KEEP), GPS β-PERF lands, race sent back once

All three commands ran on the campaign card (`lgouskos-h100-gcondo`, H100 NVL, NGC 25.08).

**1. profile_sync: the scatter kernel did NOT disappear — and the reason rewrites the
round-3 expectation, not the 2.2b verdict.** tag_cgenn 251 ms/step with
`triton_poi_fused_index_put_mul_new_zeros` at 27.88%; GPS 1448 ms/step with the same
kernel at 22.65% — statistically unchanged vs round 2, and NO segment_reduce kernel in
either hot table. The round-3 prediction ("the scatter kernel should be gone from both
tables") was WRONG, for a mechanism now machine-verified locally:

- **Inductor never lowers segment_reduce.** Both torch 2.13 and torch 2.8 register
  `make_fallback(aten.segment_reduce.default)` (and its backward) in
  `_inductor/lowering.py` — the op always runs as an extern ATen kernel call, so it
  can never appear as a Triton kernel in the table. TORCH_COMPILE_DEBUG output-code
  checks on both local versions confirm the call survives (no decomposition back to
  scatter: 17 and 14 `segment_reduce` mentions, 0 `index_put`/`scatter`, forward graph).
- **The hot Triton kernel is the BACKWARD of the edge gathers, which no forward-side
  rewrite touches.** A gather+segment-mean toy compiled fwd+bwd shows the forward
  output code has ZERO index_put/scatter while the backward has the index_put
  scatter-add — the autograd of `x[recv]` (index_select backward) scattering edge
  gradients into node slots, fused with the chain-rule `mul` and the `new_zeros`
  grad buffer: exactly the kernel name on the H100 table. That backward scatter
  existed before 2.2b, was never its target, and dominates the scatter share.

**2.2b verdict: KEEP.** The revert condition was "slower"; compiled-mode cost is
~neutral (the forward aggregation it replaced was never the hot scatter), the eager
path measurably won on it (round-2 numbers stand), and the FORWARD aggregation is now
deterministic on CUDA. The docs' determinism claim stays scoped to the forward — the
gather-backward atomics were non-deterministic before 2.2b and remain so.

*Post-campaign candidate recorded (NOT for now — it changes gradient arithmetic, so
campaign-frozen):* receivers are provably sorted (tests/experiments/test_edge_builders.py),
so the recv-side gathers could use a custom `SortedGather` autograd Function whose
backward is a `segment_reduce` sum instead of atomic scatter-add — removing the
27.9%/22.7% kernel AND making those gradients deterministic. Send-side gathers
(`x[i_send]`, unsorted) keep atomics either way.

Step-wall drift note: tag_cgenn ~251 ms/step vs round 2's reading (~10% apart) with
an unchanged kernel mix — treated as run-to-run/thermal noise, not a regression signal.

**2. GPS β-PERF row COMPLETE.** With `activation_memory_budget: 0.5` in the model
yaml the round-2 OOM is gone: bs256, eager 0.17 it/s → compiled 0.56 it/s,
**3.203×**, `compile: true` verdict. The budget-0.5 adoption stands (the escape-hatch
comment's trigger fired and this run closes it).

**3. sparse_gp_race: first run VOIDED on numerics, probe fixed, one 2-minute rerun
owed.** The NGC run read blockdiag at 0.22-0.25× of einsum (ADOPT-candidate bar met;
ctg-bmm stays un-adopted) — but the challengers' rel-vs-einsum error was 3.0-3.5e-04,
which is the TF32 signature, not fp32 (CPU control: ~5e-07). The probe did not pin
`float32_matmul_precision`, and NGC images default it TF32-enabled — both arms ran
under that mode (the einsum's bmm slices are TF32-eligible too), but TF32 accelerates
one flat Tensor-Core-shaped GEMM far more than 16 thin bmm slices, inflating the
ratio with a speedup the shipped model (repo pins "highest") never sees on either
arm. Under the campaign's own rule — precision is not a knob applied unfairly, and
here it also distorts the comparison itself — the 4× is not a clean verdict.
`utils/sparse_gp_race.py` now pins `torch.set_float32_matmul_precision("highest")`
and prints the setting; rerun in the container after a pull:

    python utils/sparse_gp_race.py

If the pinned run still clears >10% at every shape, blockdiag is adopted into
`sparse_gp_expression` (TOL class, fixtures re-recorded); if not, the einsum stays
and the question closes. This is the last open measurement of the in-campaign scope.

### Round 4 close-out — the pinned race adopts blockdiag; the fc contraction is respelled

**Pinned rerun (H100, `float32_matmul_precision=highest` printed and verified,
2026-08-16): blockdiag 0.76x / 0.72x / 0.81x of einsum — clears the >10%-everywhere
bar with clean fp32 numerics (rel 5.8e-07 to 1.0e-06, no TF32 tell). ADOPTED.**
ctg-bmm read 0.93-0.98x and stays un-adopted; the un-pinned run's 4x was indeed
TF32 vapor (0.22-0.25x collapsed to 0.72-0.81x once precision was pinned: turning
TF32 off cost blockdiag 0.9/11.8/22.3 ms/call across the three shapes vs the
einsum's 0.3/3.9/1.7 -- the flat GEMM was the arm TF32 had been flattering).
The question is CLOSED.

What shipped (commit of this entry):

- `sparse_gp_expression` fc branch (`weight.dim() == 3`): the einsum
  `"bnij,mnij->bmj"` became one flat GEMM against a block-diagonal weight —
  `diag_embed` over the j axis, `(E, n·256) @ (n·256, m·16)`. 16x the paper MACs,
  zero activation copies, and the H100 prefers it at every campaign shape. The dim-2
  layer keeps its einsum (not raced, not touched); the hand-written Function backward
  keeps its einsums (not raced — its determinism rationale is unchanged).
- **Class statement: TOL vs the einsum spelling it replaced** (reassociation only).
  Measured: Function-level fwd rel 6.8e-16 fp64, grads 0 to 3.5e-16
  (test_sparse_gp GATE-TOL-FWD/GATE-GRAD); model-level tag_cgenn sparse-vs-einsum
  fwd 4.7e-15 fp64 / 5.9e-06 fp32, BACKWARD-TOL worst 1.0e-10 (bars 1e-10/1e-5/1e-8);
  hybrid pins tag_CGENNLGATrGraphTrans fwd 8.9e-17 fp64 / 2.0e-07 fp32 vs the old
  recording. The eager-Function vs compiled-path bit-identity is UNCHANGED (both call
  sites share `sparse_gp_expression`) and stays gated.
- Gates updated with the class split stated: `test_sparse_gp` fc forward moved
  torch.equal → rel < 1e-13 fp64 (dim-2 stays torch.equal); the compiled-bypass gate
  now bit-compares compiled-vs-eager (the property call sites rely on) instead of
  vs the einsum reference. Trans hybrid pins re-recorded (both precisions).
- **Curio, verified not a bug: the GPS pins came back BIT-IDENTICAL** (rel exactly 0
  at both precisions) even though a spy confirms the GPS forward makes 4 dim-3
  expression calls. The blockdiag GEMM's padding terms are exact zeros (fp no-ops)
  and at the GPS pin shapes the CPU GEMM accumulates the nonzeros in the einsum-bmm's
  n-major/i-major order, so the sums land on identical bits. Shape/kernel accident,
  not a guarantee — the class stays TOL.

**Operator caveat on the pending `gp_impl sparse→matmul` flip for tag_cgenn:** that
flip was decided on the pre-blockdiag race (matmul 304 > einsum 296 > sparse 284
jets/s own-batch). Blockdiag cuts the sparse fc contraction ~25% on a profile where
the sparse-GP block was ~17-18% of CUDA — roughly +4% step-level, putting sparse
~297 jets/s, inside the noise of matmul's 304. Re-run the bperf gp_impl race before
executing the flip; the memory-side argument for sparse (eager retention) is
unchanged, so a re-race could keep sparse and skip the fixture churn.

### Post-campaign scouting: flash-clifford / flash-kingdon (first look, 2026-08-16)

Operator-directed first read of `maxxxzdn/flash-clifford` and its fork
`tBuLi/flash-kingdon` (fused Triton kernels for Clifford-algebra networks),
AFTER the in-campaign scope closed. Assessment, not adoption — everything here
changes arithmetic and is campaign-frozen by definition.

- **What they are.** Hand-written (flash-clifford) or kingdon-codegen'd
  (flash-kingdon) Triton kernels that fuse MV-GELU → weighted/fully-connected
  geometric product → MV-RMSNorm into one launch, hardcoding the nonzero cayley
  rules instead of einsum-ing a 95%-sparse table — the same waste our
  einsum/sparse/blockdiag ladder attacks, taken to the kernel level. Their README's
  baseline critique is literally our `bnij,mnijk,bnk` starting point.
- **Weight semantics MATCH.** Their weighted GP carries one weight per grade-triple
  path (2D: 10, e.g. `X1*Y1` splits into `|` and `^` with separate weights) —
  the same parametrization as our sparse tables' compact paths (Cl(1,3): 35), so a
  generated Lorentz kernel would be checkpoint-compatible with CGENN's GP layers.
- **The metric gap is the real work.** Shipped algebras are Cl(2,0), Cl(3,0)
  (+ Cl(3,0,1) PGA in flash-clifford). NO Cl(1,3). flash-kingdon's pitch is exactly
  the answer: kingdon's `Algebra(1, 3)` + its symbolic compile/CSE generates the
  Triton-compatible kernel source, so the Lorentz port is spec-writing, not
  kernel-writing. That makes flash-kingdon the route to evaluate first even though
  flash-clifford is the upstream with the benchmark numbers.
- **Integration frictions, in cost order:** (1) memory layout is
  `(MV_DIM, batch, features)` blade-MAJOR — the whole CGENN package (and the
  Phase-1 rewrites) is blade-minor `(..., 16)`; adopting means marshalling at the
  boundary (reintroducing the copies Phase 1 killed) or a package-wide relayout.
  (2) Their fused primitive is GELU→GP→RMSNorm; CGENN's stack is
  MVSiLU→GP→MVLayerNorm — different activation AND norm, so a faithful port
  generates OUR ops with kingdon rather than reusing their fused modules.
  (3) `torch.autograd.Function` wrappers around raw Triton — graph breaks under
  torch.compile unless re-wrapped as torch.library custom ops (our posture is
  0-break compiled). (4) `tl.atomic_add` in the fc-norm accumulations —
  nondeterministic float accumulation, weaker than what 2.2b just bought us.
  (5) CUDA-only: the CPU BIT/TOL gate tier cannot see these kernels; gating moves
  to the GPU GATE DAY infrastructure, with the existing eager paths kept as the
  reference implementation (a `gp_impl=flash` fourth arm fits the existing knob).
  (6) **Neither repo ships a LICENSE file** — before any vendoring, ask the
  authors (an issue or email); until then it is read-and-learn only.
- **Verdict for later:** the promising shape is NOT importing their modules but
  using kingdon codegen to emit OUR Cl(1,3) fused primitives (MVSiLU/GP/
  MVLayerNorm, blade-minor at the boundary, custom-op-wrapped, atomics-free
  backward like our hand-written sparse backward). Price it against the
  post-campaign baseline that already includes blockdiag + a possible
  SortedGather; the marshalling tax of layout conversion is the number to
  measure first.

### find_lr incident (2026-08-16): the steepest recommendation cost a row 0.32pp — rule now shape-gated

`top_ParticleNetParTGraphTrans` was filled from the finder's steepest line (3e-5,
bs=512). Three controlled runs, same architecture and config: lr 3e-5 scored 0.9380 /
0.9384 test acc with rej(epsS=0.3) 1239 / 1160; lr 1e-3 (the bracket-class value)
scored **0.9414 / 1771**, matching the external weaver reference 0.9417 inside the
0.0004 seed spread. Steepest was 33x low; rejection — the judged metric — moved five
times harder than accuracy. The scheduler was never at fault.

**Causal chain, dated by git:** (1) `2f29a17` flipped the recommendation from
loss-min/10 to steepest on nine ParticleNet reruns whose success criteria were
STABILITY (1.4x spread) and sitting between two OTHER recipes' published lrs — on the
hybrids' curves the loss falls at a near-constant shallow slope over decades, so
argmin(gradient) is noise-arbitrary inside that plateau and *stable because the
plateau is* (PlainGraphTrans: 3.08e-05 at bs=512, 4e-05 at bs=2048). Stability was
mistaken for accuracy. (2) `17142cb`'s TRANSCRIBE rule kept steepest as default and
gated distrust on bracket/steepest > 10x — a test that cannot fire when the curve
drags BOTH statistics low. PNPT read 3.08e-05 vs 1.17e-04 ("coherent", 3.8x), the
rule endorsed steepest, and the unit tests PINNED that endorsement as correct.

**Fix (same commit as this entry):** `suggest_lr` now measures the curve — the
lr-span, in decades, of the region within half the steepest slope. A distinct peak
spans well under `PINNED_DECADES = 1.5` (ParticleNet's concentrated fall); a plateau
spans 2+ (the hybrids) and flags steepest as `pinned`. `transcribe_lr` is
re-tabled: pinned + interior minimum → BRACKET (with a printed confirm-rerun caveat —
hybrid brackets varied 1.17e-04 vs ~1e-3 run-to-run at the same bs); pinned + no
interior minimum → NO RECIPE (rerun, smaller batch if it reproduces); distinct peak →
steepest when the bracket is unanchored or a >10x outlier (the nine-rerun pattern,
which the OLD rule would itself have voided at ratio 20-70x), bracket when both
anchored and agreeing. Banner labels went from `[default]`/`[upper bracket]` to
health-flag diagnostics + the one TRANSCRIBE directive. Tests rewritten — the
incident readings, both PNPT brackets, PlainGraphTrans, the nine-rerun pattern, the
anomaly refusal, plus synthetic hybrid/ParticleNet curves exercising the detector.
GUIDE.md's contradicting paragraphs rewritten to match.

**OPERATOR ACTION — the six queued hybrids:** any queued row whose lr came from a
steepest reading at bs=512 is suspect in the same direction (undertrained). Rerun the
finder after pulling this fix (the TRANSCRIBE line now resolves the plateau curves to
their brackets), or where a finder log is retained, re-read it: pinned-shaped model +
interior minimum → transcribe the printed loss-min/10, and confirm with a second
sweep. The CGENN recipe (bs=128, lr=5.57e-04) came from a finder read too — re-derive
it under the new rule before the long run, same one-command cost.

### find_lr incident, part 2 (2026-08-16): the ParticleNet curve itself was wrong — sweeps ran under the wrong OPTIMIZER

Same day, the operator's fresh H100 ParticleNet sweep read steepest 1.73e-04 with
loss-min/10 4.70e-03 — BOTH statistics far from the canonical ~1e-3 and both down
together, reproducing the 2026-08-15 anomaly (2.21e-04 / 8.19e-03). Per that
anomaly's own protocol, reproduction graduates it from tail-draw to conditions
investigation. The cause is found, verified, and fixed:

- **Cause.** The repo's ROOT commit (`2f29a17`, 2026-08-07 — the import from the
  predecessor repo) created `tag_gts_and_friends_default` (AdamW) and made it
  `-cn toptagging`'s default training config. `top_particlenet.yaml` inherits
  `top_ParT.yaml`, which trains with **Ranger** (RAdam + Lookahead, betas (0.95,
  0.999), the weaver/ParT-paper setup). Every bare `model=tag_particlenet` sweep
  since the import has therefore measured an **AdamW** loss-vs-lr curve, while the
  recipe, the canonical lr, the nine-rerun envelope (1.32–1.91e-3), and the FIND_LR
  pointer all speak Ranger. The era the operator remembers ("after 3 tweaks it
  produced ~1e-3 consistently") predates the import: those three tweaks (EMA-warmup
  skip, pre-minimum restriction, num_iter=300) fixed real selector artifacts and
  were never the regression — the import's silent default flip was.
- **Mechanism, verified.** Ranger's RAdam rectification + Lookahead slow weights
  (alpha 0.5, k 6) damp the early effective step, so at equal nominal lr it moves
  less than AdamW and its whole curve sits right of AdamW's. Probe (identical model,
  data, and ramp; only the optimizer swapped, repo's own Ranger class with
  base_experiment's kwargs): descent onset 4.79e-03 (AdamW) vs 5.49e-02 (Ranger) =
  **11.5×**; steepest 1.21e-02 vs 9.81e-02 = **8.1×** — the same factor separating
  the field readings (1.7-2.2e-4 AdamW-swept vs 1.32-1.91e-3 Ranger-era).
- **Fix.** `find_lr` now ALIGNS the sweep with the model's own recipe: when
  `config/training/<prefix>_<model>.yaml` exists (the exact file the FIND_LR pointer
  names) and no explicit `training=` was passed, the cfg is recomposed under it —
  so the number and the recipe it lands in share one optimizer by construction. An
  explicit `training=...` always wins; models without a recipe keep the task default
  (correct for the GT hybrids, whose recipes inherit the AdamW default — their
  sweeps are unchanged). The banner now prints `swept under: training=<choice>
  optimizer=<type>` so a cross-recipe transcription is visibly wrong instead of
  silent. Decision logic is a pure function
  (`_recipe_training_choice`), unit-tested beside composition pins on the two config
  facts (recipe=Ranger/1e-2; bare compose=AdamW) in
  tests/internal/test_find_lr_alignment.py.
- **The two find_lr incidents are INDEPENDENT bugs.** PNPT (a GT hybrid) swept under
  its correct AdamW recipe and was failed by the SELECTOR (fixed by the shape gate,
  part 1). ParticleNet swept under the wrong OPTIMIZER (fixed by alignment, part 2);
  no selector can rescue a wrong-curve sweep. With both fixes, the aligned
  ParticleNet curve is Ranger-shaped again — concentrated fall, distinct peak — so
  the shape gate resolves it to steepest ≈ the canonical scale.
- **2026-08-15 anomaly: CLOSED** (cause = conditions, optimizer mismatch; the
  planned eager-control run is moot). H100 validation is the same bare command as
  before — the banner should now show the alignment line and a steepest back inside
  1.32–1.91e-3:

      python utils/find_lr.py -cn toptagging model=tag_particlenet save=false \
          +lr_find.find_batch_size=true

### find_lr incident, part 3 (2026-08-16): the shape gate's boundary miss — and the decision that retires the question

The first aligned PNPT rerun exposed a calibration error in the part-1 fix: the real
curve's half-max slope region measured EXACTLY 1.5 decades, and `pinned` was a STRICT
`> 1.5` — so a boundary-sitting plateau classified "distinct", fell into the
bracket-is-outlier branch (ratio 37×), and the banner re-emitted the incident's own
3.08e-05. Two mistakes, both mine: the synthetic calibration curves (0.7 vs 3.5
decades) were far better separated than real curves, and a threshold that a live case
sits exactly ON is fragile by construction. Fixed: `PINNED_DECADES = 1.2`,
INCLUSIVE — ParticleNet's concentrated fall (~0.7) keeps a 0.5-decade margin, PNPT's
1.5 a 0.3-decade margin, and a boundary case now fails toward the bracket/refusal,
never toward a pinned steepest. Pinned by test against the measured 1.5.

**TABLE-WIDE lr DECISION (operator, 2026-08-16): every top_<hybrid>.yaml now ships
`lr: 1e-3`** — which is also `tag_default`'s own value the finder-transcribed 3e-5
had been overriding. Grounds: all eight rows share one AdamW recipe; every reliable
hybrid reading clusters in 4.5e-4..1.2e-3 (flat within ~2× of 1e-3); 1e-3 is
trained-and-validated on PNPT (0.9414 vs weaver 0.9417, inside seed spread); and two
transcription incidents in one day showed per-row single-sweep transcription carries
more risk than the ≤2× lr suboptimality it might save. The finder remains the
CONFIRMATION tool (its aligned, shape-gated readings should agree with 1e-3 within
~2× — a reading that does not is a finding, not a recipe). Scope: the top-tagging GT
table only; the jc_ tree and the non-AdamW baselines (Ranger/Lion recipes) keep
their own values.

**Batch sizes** are the one number still owed per row: `utils/find_bs.py` runs ONLY
the doubling search (own-recipe compose, shipped compile posture, worst-case probe,
cap = the 512 ceiling by default) and prints paste-ready `batchsize:` lines for all
eight in one pass:

    python utils/find_bs.py

### Aggregate-table audit (2026-08-16): the trials-policy changes are sound

Reviewed 2f29a17 (group-instead-of-dedup) + 5cec76a (policy + guards) on request.
Verdict: SOUND. Grouping is by (task, model, frames, kNN) with recency by log mtime
(entries re-sorted inside `_consolidate`, so newest-wins is order-independent);
pooling uses sample std (n−1) and matches the per-run formatter's precision; and
every case where key-inference could lie refuses toward a visible newest-wins note
instead of a silently wrong row: mixed with an in-run-aggregated row
(double-counting), disagreeing iters/params/FLOPs (ablations sharing a key), and
identical-metrics clones (pinned seed — would fabricate ±0.000 precision). The one
gap found: those guards were "exercised synthetically" in the authoring session but
never committed as tests — closed now by
tests/internal/test_aggregate_consolidate.py (5 pins, including mtime-vs-list-order).
Residual limitations, acknowledged not fixed: mtime is fragile across rsync/copies,
and the invariant comparison is string-exact (a formatter change would refuse pooling
— conservative direction, visible note).

Addendum, same day: 0d4e70f (jets-seen column + train time in hours) reviewed on
arrival — correctly layered (render-time `augment_row` AFTER `_consolidate`, so no
invariant/averaging index shifts; jets falls back to `n/a` when trial batchsizes
disagree), and the five guard pins pass against it unchanged. One watch-item: the
jets parser reads leading digits of the iters cell (`^\s*(\d+)`), so eyeball the
first real table for a k-formatted iters cell misparsing (48k -> 48 jets-seen).

### Batch sizes MEASURED and shipped (H100 NVL, 2026-08-16) — and why pending speedups don't invalidate them

find_bs first full pass (own-recipe compose, shipped compile posture, worst-case
probe, 512 ceiling): **512 for seven rows, 256 for CGENNLGATrGraphGPS** (69.7 GB peak
at 256 with activation budget 0.5 — consistent with bperf's independent 256 finding).
All eight values are pasted into the top_ yamls; every row now passes the sweep guard
and the queue is launchable at lr 1e-3. Note LorentzNetLGATrSlimGraphGPS runs 78.7 GB
at 512 (~84% of the card) — sized against the worst-case probe, so real batches sit
below it; do not move these rows to a smaller card without re-running find_bs.
CGENN-GPS trains at 256 with the table lr 1e-3: readings were measured at 512, but the
optimum's ~2x-flat basin comfortably covers a 2x batch reduction (sqrt-scaling purism
would say ~7e-4; comparability keeps 1e-3).

**Do pending speedups outdate these sizes? No, for the campaign.** SortedGather
changes the backward KERNEL, not what autograd retains — peak unmoved. The gp_impl
question concerns tag_cgenn (a baseline outside these eight), and its compiled-path
retention is ~equal across impls anyway. eta_min/scheduler/TF32 touch no memory.
The two things that WOULD move memory are both post-campaign and both re-measured by
design: flash fusion (removes intermediates — batches could only grow, so today's
values err safe; F4's gate includes a vram row) and bucketing (pads wider — its plan
already includes re-sizing). One command re-derives everything if ever in doubt:
`python utils/find_bs.py`.

### Warning triage (2026-08-16): the Dynamo lru_cache wall is benign — and now deduplicated

The hundreds-of-lines warning during compiled sizing/sweeps is Dynamo noting it
traces THROUGH `functools.lru_cache` wrappers. Source located: lgatr's
`primitives/{bilinear,linear}.py` — cached `.to(device/dtype)` helpers over
module-level CONSTANT basis tensors, i.e. pure functions of immutable state, exactly
the case where trace-through is sound. The warning is generic ("may not be sound if
it reads outside state"); ours reads none, and the compiled-vs-eager TOL (<=1e-10),
DET (bit-equal repeat calls), and hybrid-pin gates are the machine check of that
soundness. It fires once per traced call site per compile, hence the wall.
find_lr/find_bs now install `warnings.filterwarnings("once", ...)` for that message —
one visible instance instead of thousands, no blanket suppression. Also from the same
log: the inductor TF32 hint is the EXPECTED reminder of the deliberate
`highest` pin (a recorded table-wide pending decision, not an oversight); the
`torch._prims_common.check` FutureWarning is torch-internal; and the GradScaler
FutureWarning is fixed at the source (`torch.amp.GradScaler("cuda", ...)`, same
class, identical behavior).

### FINAL LEDGER RESOLUTION (2026-08-16): no post-campaign exists for CGENN — every item is DO or DON'T

Operator constraint: anything not done before the CGENN rows launch will never be
done. That collapses "post-campaign" for the whole CGENN family; each ledger item is
re-decided under it, decisions FINAL:

**JC SCHEDULE, decided here and down: LinearWarmup+Cosine for ALL GT rows, both
tables, with `cosanneal_eta_min: 1e-5` (1% of the 1e-3 peak) set now at the campaign
boundary.** flat+decay's theoretical edge in the jc few-pass regime (~0.1pp scale) is
unmeasured in-repo under AdamW and needs a re-shaded peak — two coupled unknowns
against one-schedule-per-table comparability and a peak validated under cosine. The
A/B is CLOSED, not pending. Sequencing note: eta_min landed after the three jc
GraphTrans submissions were handed over — if those jobs already started, scancel and
resubmit (minutes old), so the jc table stays uniform.

**DO NOW — the two items that survive:**
1. **SortedGather** (this commit series). Recv-side gathers get a custom Function
   whose backward is a `segment_reduce` sum over the provably-sorted receivers,
   replacing the atomic scatter-add that is the top kernel on both profiles
   (27.9%/22.7%; the send-side share stays, so expect roughly half of it back).
   Forward is BIT-identical (index_select), so the hybrid pins stand un-re-recorded;
   gradients are the same math reassociated — and the atomics they replace were
   nondeterministic run-to-run anyway, so no comparability exists today that this
   could break. Ships behind `CGENN_SORTED_GATHER` (default on) with the full gate
   battery; the CGENN rows need one 15-min GPU gate day before launch. Launch the
   five non-CGENN top rows immediately; hold tag_cgenn + the two CGENN hybrids for
   the gate.

   *Executed (same day):* `experiments/baselines/cgenn/sorted_gather.py`, wired at
   the two receiver gathers (`x[i]`, `h[i]`) of both CGL twins (the GPS layer
   delegates to the hybrid's CGL, so all three CGENN-family models are covered);
   senders keep plain autograd. Gates green:
   tests/experiments/test_sorted_gather.py — BIT forward, backward vs autograd
   1e-13 fp64 incl. exact-zero rows for zero-degree nodes, gradcheck +
   gradgradcheck, fallback paths (counts=None and the kill switch), compiled 0
   breaks / <=2 graphs over three shapes with 1e-13 compiled-vs-eager grads; plus
   the model-level BACKWARD-TOL and full compile battery re-run with the feature
   ACTIVE (dynamo_explain re-recorded — graph gains the Function; invariants stand).
   **GPU GATE DAY for the three CGENN rows (~15 min, then launch):**

       CGENN_COMPILE_GATES=1 CGENN_SMOKE_COMPILE=1 CGENN_SMOKE_STEPS=100 \
       python -m pytest tests/experiments/test_training_smoke.py \
           -k "cgenn or CGENNLGATr" -q
       # optional pricing: python utils/profile_sync.py -cn toptagging model=tag_cgenn \
       #     save=false data.dataset=mini training.batchsize=64
       # revert switch if anything is off: CGENN_SORTED_GATHER=0 (env, no code change)

2. **eta_min** (above) — free at the boundary, uniform, ~0.03pp-class upside.

**DON'T — closed forever, with reasons:**
- ~~Flash port~~ **OVERRIDDEN by the operator (2026-08-16, same day): the flash port
  is DO — see THE FLASH PLAN v2 below.** What changed: (a) the CGENN rows have not
  launched, so "before campaign" is real for them and the operator accepts the delay;
  (b) the licensing blocker dissolves under the generate-don't-vendor route — the
  kernels are OUR code emitted via kingdon (pip-installable, citable via its own
  BibTeX, its license governing), with the flash repos used as design reference only.
  The earlier DON'T reasoning was correct under its assumption (a launching queue);
  the assumption was wrong for the CGENN rows.
- **Bucketing / GPS host-tax — stays DON'T, and here is the accuracy answer asked
  for: YES, bucketing changes accuracy, by either route.** (a) Keeping the shipped
  quirk semantics while bucketing changes the numbers anyway: `theta_h` BatchNorm
  runs over PADDED nodes and the tag_cgenn readout divides by the padded width, so
  the model literally computes different values at different padded widths — that
  dependence is WHY it was frozen. (b) Fixing the quirks (mask-aware BN, true-length
  readout) is cleaner and probably slightly better, but it changes the model relative
  to the upstream-faithful reproduction and needs its own validation runs. Both
  routes touch accuracy; combined with it being the largest engineering item and
  walltime-only in payoff, it stays closed. (The flash port does not moot it — the
  GPS host tax is Python/launch overhead, not kernel time.)
- **gp_impl sparse→matmul re-race: superseded, folded into the flash race.** The
  decision point becomes flash-vs-current-best at plan step F4 (below), raced before
  any CGENN model or hybrid trains — the operator's preferred shape. If flash fails
  the bar, sparse stays (matmul's ~2% is inside the noise band, as recorded). The
  tag_cgenn recipe bs=128 / lr=5.57e-04 stays valid — 5.57e-04 is sqrt-scaled 1e-3
  at bs128, independently corroborating both numbers.
- **Shield retirement**: needs the NGC container upgrade; shields are ~free
  insurance — they simply stay on for the whole campaign.
- **TF32 table-wide**: a precision cut with unmeasured accuracy cost cannot be
  decided under queue pressure; `highest` stays, question closed.
- **Scheduler A/Bs (both tables)** and **3-point lr training sweeps**: closed by the
  schedule decision above and the table-wide 1e-3 (revive a sweep only if a row
  lands clearly below its weaver-class expectation).
- Remaining user-optional cosmetics (upstream inductor issue, scratch-branch
  deletions) are unaffected by the constraint and stay optional.

### THE FLASH PLAN v2 (2026-08-16) — executable, step-driven, before any CGENN row trains

**NEXT STEP: 5 — LAUNCH for the non-CGENN table (FLASH-2 CLOSED ×2); the CGENN rows go LAST and their arm is decided by FLASH-3 (see plan below, currently at step 0)**  ← the state marker; each executed step advances it and records
results directly under its entry. The operator drives with "do step N, then audit";
the sandbox has no GPU, so every step ends either CPU-verified or with the exact
pastable GPU commands and a hard stop until the operator pastes results back.

Ground rules carried from the whole program: measure before adopt (the >10%
adopt-or-close bar), state every class change, gates before launch, kill switches
over reverts, OUR CliffordAlgebra cayley is the reference for all generated math.
Scope note: this ports the GP CONTRACTIONS (`gp_impl=flash` fourth arm — weighted GP
dim-2 and fc-GP dim-3). The fused act→GP→norm stacks are a stretch goal ONLY if F4
adopts and time remains; the contraction arm alone targets the measured GP share.

- **Step 0 (operator GPU, ~15 min, independent of flash):** SortedGather gate day —
  runbook above. Green → the non-CGENN rows launch posture is complete; CGENN rows
  additionally wait for F4/F5.
- **Step 1 (sandbox, CPU):** F0 foundations. `pip install kingdon` (pin the version);
  record its license + BibTeX in the docs (generate-don't-vendor legal basis);
  machine-check `kingdon.Algebra(1, 3)` against `CliffordAlgebra((1,-1,-1,-1))`:
  blade order, sign conventions, and the full 16x16x16 cayley equality, as a
  committed test. DELIVERABLE: tests/internal/test_kingdon_conventions.py green, or
  a documented conversion table if conventions differ.

  *DONE (2026-08-16).* kingdon **2.1.1** pinned (test-enforced). **License: MIT**
  (c) 2022 Martin Roelfs; **citation: arXiv:2503.10451** + the repo CITATION.cff
  (github.com/tBuLi/kingdon) — generated kernels are OUR code produced with a
  citable MIT tool; the flash repos remain design reference only. Convention gates
  ALL GREEN, and better than hoped: `Algebra(1, 3)` signature [+1,-1,-1,-1], blade
  order short-lex matching ours under the IDENTITY mapping (kingdon e_{k+1} <-> our
  vector k), and the full multiplication table coefficient-exact against our cayley
  in its [left, out, right] orientation — all 4096 entries, 256 nonzeros, one output
  blade per (left, right) pair (the quasigroup property re-proven from kingdon's
  side). No conversion table needed anywhere in the generator.
- **Step 2 (sandbox, CPU):** F1 codegen. Generate the 35-path weighted-GP (dim-2)
  and fc-GP (dim-3) forward + all three gradients as pure-Python/CSE'd expression
  lists via kingdon's symbolic compile; gate vs `sparse_gp_expression` at fp64
  (<=1e-13) + gradcheck, including the masked-path case. DELIVERABLE: the generator
  script + generated reference module + gates green. This is the mathematical
  content of the kernels, fully verified before any Triton exists.

  *DONE (2026-08-16).* `utils/flash_gen.py` → committed
  `experiments/baselines/cgenn/flash_ref_p1m3.py`: forward 256 terms / 73 CSE
  temps, one grad function with all 67 outputs (16+16+35) / 108 CSE temps, flat
  arithmetic bodies in flash-clifford's kernel style, repo compact-path weight
  order (checkpoint-compatible). Terms SOURCED from kingdon's `Algebra(1,3)`
  products and asserted term-by-term against `sparse_gp_tables`/`gp_k_idx` at
  generation time. Gates (tests/internal/test_flash_ref_p1m3.py) ALL GREEN at
  ~3e-16: dim-2 fwd vs the einsum expression, dim-3/fc fwd vs the shipped
  blockdiag path, generated grads vs autograd, cross-grads vs the expression's
  autograd, gradcheck, and a regeneration-consistency pin (committed module must
  byte-match what the pinned generator emits; parity gates are torch-only and
  never skip). Deviation from the plan text, stated: the masked-path variant is
  NOT generated — the shipped Lorentz algebra masks nothing (all 35 paths live,
  asserted in the generator); reduced-path algebras are out of the port's scope.
- **Step 3 (sandbox authors, operator verifies on GPU):** F2 kernels. Transcribe the
  step-2 expressions into triton.jit kernels wrapped as torch.library custom ops
  (fake-tensor shapes registered, backward registered, NO atomics in dL/dx / dL/dy;
  dL/dweight via a fixed-order two-stage reduction — determinism is a ship
  requirement, matching SortedGather/2.2b). CPU deliverable: import-clean module,
  meta-shape tests, and a CUDA-gated parity test file. OPERATOR: one pastable
  command running GPU parity (vs einsum ref, fp32+fp64), determinism pair-run, and
  a microbenchmark vs the blockdiag contraction at campaign shapes.

  *CPU SIDE DONE (2026-08-16); GPU ROUND-TRIP PENDING — the step's hard stop.*
  `flash_kernels_p1m3.py`: `cgenn_flash::fcgp` custom op — the fc contraction, the
  shipped hot path (gpmlp-only dim-2 out of scope, stated). No transcription
  happened at all: the kernels call `triton.jit(flash_ref_p1m3._wgp_fwd/_wgp_grad)`
  — flash-kingdon's trick — so the 3e-16-gated reference IS the kernel body.
  Forward: one program per (row-block, m), n-loop register accumulation
  (flash-clifford's fc shape). Backward: one program per (row-block, n), m-loop for
  gx/gy, dL/dw per-(block, m, n) partial slots + a torch `.sum(0)` stage-2 — the
  stated departure from flash-clifford's `tl.atomic_add`. CPU composite (the
  reference wrappers) registered for the op so wiring is fully gated without a GPU:
  `torch.library.opcheck`, fwd 2.8e-16 and all grads ~2e-16 vs the shipped
  expression, dynamo 0 breaks through the op — 5/5 green. Residual GPU-day risk is
  ONLY kernel-internal (trace-time Triton semantics, register pressure at
  BLOCK 64/32) — wiring cannot be the failure. Audit fixes landed with this step:
  regeneration pin now stamps + checks BOTH kingdon and sympy versions (would have
  misfired on the container's different sympy), and the codegen-mechanism deviation
  is stated in the generated header.

  **OPERATOR — the step-3 GPU round-trip (paste results back):**

      git pull origin main
      pip install kingdon==2.1.1   # venv, one-time (conventions test; kernels don't need it)
      python -m pytest tests/experiments/test_flash_kernels_cuda.py -q -s

  Expected: parity fp64<=1e-13 / fp32<=1e-5, DET bit-equal pair-run, three
  BENCH-FLASH lines vs blockdiag (the step-4 race inputs). Paste the full output.
- **Step 4 (mixed):** F3+F4 wiring and the race. `gp_impl=flash` behind the existing
  knob (CUDA-only guard, eager+compiled through the custom op); full gate battery
  (BREAKS/RECOMP/TOL/DET + soak) on the operator's GPU; then the adopt-or-close
  race: bperf tag_cgenn flash-vs-sparse(+blockdiag+SortedGather) and profile_sync,
  pinned precision, >10%-everywhere bar. ADOPT → set `gp_impl: flash` in the three
  CGENN model yamls (config + config_quick), TOL-verify + re-record hybrid pins and
  explain (class change stated). CLOSE → sparse stays, record the price, no yaml
  churn.

  *Round-trip #3 (2026-08-16): the kernels RUN, and the op-level race is a
  landslide.* 5/6 gates green on first compile of the explicit kernels: fp32 parity
  ~2e-07 fwd and all grads, DETERMINISM bit-equal (forward + all three gradients —
  the no-atomics design paying off), and **BENCH-FLASH 0.25x / 0.21x / 0.16x of
  blockdiag** (4-6x faster, fwd+bwd, adopt bar <0.90x cleared everywhere at op
  level). The one failure was the fp64 VERIFICATION path: Triton requires
  loop-carried accumulators to keep one type, and the fp32-zeros init got
  reassigned fp64 — all 48 accumulators in both kernels now init as
  `xp.dtype.element_ty` (fp32 path semantics unchanged).

  *Step-4 wiring COMPLETE (same day, CPU-verified):* `gp_impl: flash` is a live
  fourth arm — fcgp.py routes the fc contraction to the custom op (same compact
  weight, checkpoint-compatible); gp.py raises loudly if flash reaches a gpmlp
  site; all three nets accept the value; GP_IMPLS in test_cgenn_compile gains
  "flash", and the CPU battery already passes it: TOL-IMPL 3.95e-15 fp64 /
  4.1e-06 fp32, BACKWARD-TOL worst 2.3e-10 — the same bars sparse holds. bperf's
  MATRIX gains the `tag_cgenn/flash` row (the `--models tag_cgenn CGENN` pin
  updated to four gp_impl rows). Remaining before adopt-or-close, ONE allocation:

      set -e
      # 1. kernel gates (~2 min): parity, determinism, clean uncontended bench
      python -m pytest tests/experiments/test_flash_kernels_cuda.py -q -s
      # 2. model-level compile battery, flash arms incl. compiled backward (~3 min)
      CGENN_COMPILE_GATES=1 python -m pytest tests/experiments/test_cgenn_compile.py -q -k flash
      # 3. the works-completely soak (~10 min): 100 varying-shape COMPILED training
      #    steps, flash on all three CGENN rows, shields + SortedGather active --
      #    this run IS the CGENN rows' pending SortedGather gate day too
      CGENN_SMOKE_OVERRIDES="model.net.gp_impl=flash" \
      CGENN_COMPILE_GATES=1 CGENN_SMOKE_COMPILE=1 CGENN_SMOKE_STEPS=100 \
      python -m pytest tests/experiments/test_training_smoke.py -q -s -k "cgenn or CGENNLGATr"
      # 4. ONLY reached if 1-3 green (set -e): the trimmed race -- incumbent vs
      #    challenger; einsum/matmul keep their round-4 numbers
      python utils/bperf.py --models tag_cgenn/sparse tag_cgenn/flash --find-batchsize

  Paste all three outputs; ADOPT flips the yamls + re-records pins, CLOSE keeps
  sparse — either way step 5 (launch the CGENN rows) follows immediately.

  *Round-trip #4 (2026-08-17): kernels FULLY green; the race reads adopt-shaped;
  two fixes + one command mea culpa.* Kernel gates **6/6** — fp64 parity at
  3.6e-16 (machine epsilon: the kernel IS the generated math), fp32 ~2e-07, DET
  bit-equal, bench 0.18x/0.05x/0.20x. bperf, converted to the jets/s the
  differently-sized rows require: einsum-compiled 241, matmul-compiled 294,
  sparse-compiled 310 — and **flash EAGER 297 at bs512**, the only impl to fit
  the full 512 ceiling (73.3 GB; ~0.14 GB/jet vs sparse 0.17 capped at 128 and
  einsum ~1.5 capped at 16). Flash-compiled is the open number; it failed on ONE
  mechanism: AOT's joint-graph trace runs the registered backward on fake
  tensors, and a raw Triton launch there dies on data_ptr(). FIXED as the error
  prescribes — the backward is now its own opaque custom op (`fcgp_bwd`, fake
  registered, CPU composite kept), and two new CPU gates pin the wiring (joint
  fwd+bwd compile parity at 1e-13; the bwd op under FakeTensorMode). The
  backward-TOL failure was test infra, not flash: `_grads`' probe weights were
  created without a device — first CUDA run of that gate ever; fixed. And the
  COMMAND was mine to own: `srun -n 4` means 4 TASK REPLICAS, not 4 cores —
  every round-trip so far ran 4x concurrently on one GPU (the 4x exit codes,
  the host-OOM that killed einsum's eager arm, and the bench variance between
  rounds: 6.21 vs 18.69 ms for the same shape). Direction is contention-robust
  (<0.25x both rounds) but clean numbers need `-n 1 -c 4`. bperf's table now
  carries bs + jets/s columns so cross-impl rows can never be misread again.
  **Round-trip #5 (same three commands, fixed srun) decides adopt-or-close.**
- **Step 5:** launch the three CGENN rows (top + jc CGENN recipes) under whichever
  arm won, with find_bs re-run for them ONLY if F4 adopted (kernel fusion can only
  grow the fitting batch, but measure rather than assume).

  *STEP 4 VERDICT (2026-08-17, round-trip #5, uncontended): **CLOSE — sparse stays
  for the campaign.*** Everything verified first: kernel gates 6/6 (clean bench
  0.18x/0.20x/0.20x of blockdiag — ~5x at op level), compile battery flash arms 6/6
  including the compiled backward, and the 100-step compiled varying-shape soak
  green on ALL THREE CGENN rows under flash (losses 0.69→0.58 / 0.72→0.51 /
  0.69→0.58, 100% nonzero-grad params) — **which also clears the CGENN rows'
  pending SortedGather gate day** (the soak ran with it active). Then the race:
  sparse-compiled 314 jets/s at its bs-128 ceiling vs flash-compiled 331 at bs-512
  = **+5.4% own-best, below the pre-registered >10% adopt bar.** Mechanism, visible
  in the speedup column: sparse gains 4.01x from compile (inductor fuses the einsum
  chain into the whole graph); flash gains only 1.112x — the OPAQUE custom op
  blocks fusion across its boundary, and that fusion tax eats most of the 5x
  kernel win. Flash's real victory is MEMORY (~0.14 GB/jet, 4x the batch, 3x
  headroom at equal batch), which is what nets the +5.4. Per the discipline: not
  adopted; the arm STAYS in-tree (gated, kill-switchable, checkpoint-compatible,
  selectable via `model.net.gp_impl=flash`) as the reference implementation and
  the memory escape hatch. The one continuation that attacks the mechanism itself
  — Tier-B fusion, moving MVSiLU/MVLayerNorm INSIDE the op boundary so there is
  nothing left to fuse across — is recorded as the sole flash follow-up that could
  flip the verdict; it costs 1-2 more kernel round-trips and the campaign does not
  wait on it. Recipes filled for launch: top_cgenn 128 / 5.57e-4 (sparse ceiling;
  finder value, sqrt-consistent with the table's 1e-3@512), jc CGENN-Trans
  512 / 1e-3, jc CGENN-GPS 256 / 1e-3 (top-tagging measurement; jc inputs are
  wider — find_bs pre-flight advised). **The CGENN rows are cleared to launch.**

  *FLASH-2 (2026-08-17, same day, CPU-verified — re-opens the race ONCE, on the
  operator's call: "the model takes up 89% of my runtime... anything you can do to
  make flash worth it?").* Two changes, both forward-preserving, both aimed at the
  measured mechanisms rather than new speculation:

  1. **`triton_op` conversion (attacks the step-4 CLOSE mechanism head-on).** Both
     flash ops (`cgenn_flash::fcgp`, `::fcgp_bwd`) are now registered via
     `torch.library.triton_op` + `wrap_triton` instead of the opaque
     `torch.library.custom_op` — PyTorch's documented remedy for exactly our
     verdict: inductor SEES the Triton kernel inside the op and keeps fusing the
     surrounding graph across its boundary, instead of treating it as a black box
     (the fusion tax that held flash's compile speedup to 1.112x while sparse got
     4.01x). Same kernels, same schema, same CPU composite and autograd
     registration; the hand-written `register_fake`s are gone (triton_op derives
     metadata from the now-visible body). Verified on torch 2.13 (sandbox) and
     available on the NGC container's 2.8. This is also the honest answer to
     "can sparse and triton be done together": the einsum→inductor route and the
     handwritten-kernel route were merged by making the kernel visible to the
     compiler — and it may subsume part of Tier-B (epilogue fusion) for free.
  2. **Sender-side deterministic gathers (`SortedGatherPermuted`).** SortedGather
     covered only the RECEIVER half of the 27.9%/22.7% scatter-backward kernel;
     the sender gathers `x[j]`/`h[j]` still backwarded through atomic
     scatter-add. A scatter over an UNSORTED index is still a segment sum after a
     precomputed stable-sort permutation: backward = `grad.index_select(0, perm)`
     + `torch.segment_reduce` over the sender degrees. The net forwards compute
     `send_perm = argsort(j, stable=True)` + `send_counts` ONCE per forward next
     to `edge_counts` (all three nets: package CGENN, hybrid backbone, GPS — the
     GPS reuses the hybrid's CGLayer, so two CGL forwards thread the extras).
     CLASS: forward is BIT (it IS `x[idx]`) — hybrid pins stand un-re-recorded,
     verified; gradients are the same sums reassociated, replacing atomics that
     were never run-repeatable — a strict determinism upgrade. Same
     CGENN_SORTED_GATHER kill switch; extras `None` → plain autograd.

  CPU verification, all green: flash CPU gates 8/8 (opcheck, parity ~2e-16, joint
  compile 1e-13, FakeTensorMode, AST tripwire); sorted_gather 16/16 (8 new
  permuted-twin gates: BIT fwd, grads-vs-autograd incl. empties fp64/fp32,
  gradcheck+gradgrad, fallbacks, compile 0-breaks/≤2-graphs, executable
  perm/counts contract); hybrid BIT pins 11/11 UNCHANGED; full compile battery
  25/25 under CGENN_COMPILE_GATES=1 (breaks/recomp/backward-TOL, flash arms
  included); twin-parity/autocast/sparse_gp 72/72.

  Expectation set honestly: GP is ~14% of the compiled sparse step and the
  send-half of the scatter kernel ~11-14%; with fusion recovery the plausible
  combined ceiling is ~1.3-1.5x on tag_cgenn — worth one round-trip, not a
  promise. The >10% adopt bar vs sparse-compiled **314 jets/s** stands. The CGENN
  rows HOLD for this ONE re-race (operator's ledger override: flash-before-campaign
  is fine and citable); everything else about the launch posture is unchanged.
  ADOPT flips the three yamls to `gp_impl: flash` + re-records pins (forward
  changes: fc contraction evaluation order — pins are TOL-class re-records, stated),
  CLOSE launches on sparse with flash kept as the memory escape hatch.

  *Round-trip #6 (2026-08-17): kernel gates 6/6; compile battery flash 3/6 — ONE
  new frontend rule, found and fixed, and the failure is the fusion path ENGAGING.*
  Kernel gates unchanged-green (fp64 3.6e-16, DET bit-equal, bench
  0.18x/0.20x/0.20x — matching #5, as they must: raw launches resolve the
  subfunction from module `__globals__`). The three inductor-visible gates
  (compiled backward, TOL-DET compiled, breaks-and-recomp) all died in inductor's
  compile worker at `ast_to_ttir` with `NameError('_fwd_body is not defined')` —
  note dynamo itself printed `GATE-BREAKS[flash] graph_break_count = 0` first.
  MECHANISM (read from torch source, both 2.8 and 2.13): with `triton_op`,
  inductor EMBEDS the kernel into its generated module via
  `user_defined_triton_kernel_transitive_closure_source_code`, which re-emits each
  called JITFunction's `src` — whose def line carries the wrapped function's
  ORIGINAL name (`def _wgp_fwd`) — and, unlike its ConstexprFunction branch, the
  JITFunction branch writes NO alias line for a mismatched binding. So
  `_fwd_body = triton.jit(_ref._wgp_fwd)` works at raw launch and NameErrors under
  embedding. That this path ran AT ALL is the positive signal: the opaque-op era
  never embedded anything — the fusion tax remedy is live, it just tripped on the
  alias. FIX (name-only, cannot move the gated numbers): bind the jit-wrapped
  generated bodies under their own def names (`_wgp_fwd`/`_wgp_grad`) and call
  those in the kernels. Two new CPU gates pin the rule forever:
  binding-name==`fn.__name__` invariant (vacuous-proof: asserts each kernel HAS a
  jit dependency) and an end-to-end run of torch's actual emitter asserting the
  emitted closure parses with a def for every called name (10/10 CPU flash gates).
  Audited the whole class: repo grep shows these were the only two JITFunction
  bindings anywhere; both generated bodies have EMPTY `co_names` (self-contained
  arithmetic — the closure recursion terminates at depth 1); the kernels' only
  other global is the `tl` module, which the #6 traceback itself shows resolving
  fine. Soak + race were never reached (`set -e`). **Round-trip #7, one
  allocation, same four commands** decides adopt-or-close.

  *FLASH-2 VERDICT (2026-08-17, round-trip #7): **CLOSE — twice-confirmed, final.
  Sparse stays; the CGENN rows are CLEARED TO LAUNCH with nothing pending on
  flash.*** Gates first, all green: kernel 6/6 (parity/DET/bench unchanged, as a
  name-only fix demands), compile battery 6/6 (the binding-name fix verified on
  the NGC build end-to-end), soak 3/3 (losses 0.69→0.58 / 0.72→0.51 / 0.69→0.58,
  100% nonzero-grad params). The race: sparse-compiled **313 jets/s** @128
  (3.99x compile speedup) vs flash-compiled **329** @512 (1.11x) = **+5.1%**,
  statistically the same as #5's +5.4%. SCIENCE OF THE NULL: #6 proved the
  kernel is now EMBEDDED in inductor's generated module (the NameError came from
  inside the embedding path) and #7 shows embedding recovered nothing — the
  step-4 "opaque op blocks fusion" theory is REFUTED as the dominant mechanism.
  The per-512-jet costs expose what actually binds: sparse-eager 6557 ms,
  flash-eager 1724, sparse-compiled 1639, flash-compiled 1562. Three of four sit
  on the SAME ~1.6 s floor — hand-fusing the GP (flash) and letting inductor
  fuse the graph (sparse) are two routes to one destination, and stacking them
  buys ~5-9% because (a) visibility ≠ fusibility: inductor cannot fuse
  elementwise chains INTO or ACROSS a user kernel call, so the ~32 GP calls per
  step still partition fusion regions the sparse arm keeps whole, and (b) the
  compiled step is bounded by (E, C, 16) edge-tensor traffic plus the
  activation stash, which neither arm reduces. COROLLARY, now measured twice:
  the GP contraction is no longer a lever on this model — any further win must
  REDUCE TRAFFIC, not speed math. Sender-gather (FLASH-2b): race-neutral as
  predicted (both arms carry it; 313/329 vs #5's 314/331 = noise), kept for its
  actual claim — deterministic gradients. Flash's standing, real win is memory:
  0.146 vs 0.537 GB/jet (3.7x), the in-tree escape hatch if a bigger variant or
  dataset ever OOMs sparse.

  *Breakthrough assessment (operator asked "is a dramatic speedup possible?" —
  answered and recorded as **DON'T before campaign** under the
  after-campaign-is-never constraint).* The floor arithmetic endorses exactly
  one structure-level project: the FUSED MESSAGE KERNEL — flash-attention's
  never-materialize trick applied to edges. One kernel per edge-block that
  gathers x_i/x_j by index in-kernel, computes GP + invariants + gate in
  registers, and segment-sums straight into node slots (receivers are SORTED,
  so the in-kernel reduction is atomic-free and deterministic — the SortedGather
  invariant becomes a kernel feature), with a recompute-in-backward custom
  gradient so the (E, C, 16) intermediates are NEVER written to HBM in either
  pass. It attacks both floor terms at once (edge-tensor traffic AND the
  stash); plausible yield 1.5-3x on tag_cgenn plus 2-4x batch ceilings; honest
  cost 5-10x the flash-GP port — which itself took 7 GPU round-trips — with a
  custom joint-kernel gradcheck/DET battery and real risk of never clearing the
  bars. Cheap alternatives priced and rejected: per-CGL activation
  checkpointing (equivalence-preserving, but ~+40% recompute vs ~+30-50%
  bs-efficiency by the measured probe curves — a wash); kernel autotune (caps
  at the GP's ~9% residual share); hand-written triton MVSiLU/MVLayerNorm as
  standalone ops (REGRESSIVE — inductor already fuses them for free, standalone
  ops would add the very partitions that capped flash). The campaign launches
  on sparse now.

  *OVERRIDE (2026-08-17, operator, same day): the DON'T's two premises were
  wrong, so the verdict flips to GO — recorded as FLASH-3 below.* The operator
  clarified: (1) the campaign has NOT started, and its order runs every
  non-CGENN model first — the three CGENN rows go LAST, so a kernel built in
  parallel blocks nothing; (2) the jc CGENN rows cost ~100x the other JetClass
  rows (weeks vs days) — a 1.5-2x on them saves WEEKS of wall-clock, which
  dwarfs the build cost even at flash-port-times-ten. Both facts were the
  conditions the DON'T itself named for flipping. The cheap-alternative
  rejections above stand unchanged; only the schedule calculus moved.

### THE FLASH-3 PLAN (2026-08-17) — fused CGL message kernel, parallel to the campaign

**ROUND-TRIP #7 PENDING — the #6 close (+7.3%) rests on a sizing bias the driver's own docstring warns against ("do not rank gp_impls by it/s from this driver"): BOTH arms were sized EAGER, so neither raced at its compiled ceiling, and the eager/compiled memory gap is per-arm. `--size-per-state` is now implemented + gated; #7 re-measures honestly. Command in its record below. Sparse ships meanwhile; jc GPS recipes complete (bs 512, lr 1e-3 ×3) and those rows are launchable NOW** ← state marker,
same protocol as FLASH v2: operator drives step-by-step; every step ends
CPU-verified or with exact pastable GPU commands. The campaign launches its
non-CGENN rows NOW and is never blocked by this plan; the three CGENN rows go
last, and their `gp_impl`/CGL arm is decided by whichever of {sparse, fused} has
cleared all gates when their turn comes. Ship rules carried over: TOL gates
(fp64 1e-13 vs the eager composite), DET pair-run bit-equal, gradcheck at fp64,
empty-segment nodes, the binding-name rule (test-pinned), kill switch
(CGENN_FUSED_CGL=0), arm-not-rewrite (the composite path stays; main always
shippable), adopt bar >10% own-best vs sparse-compiled — with the project's own
success target ≥1.5x, else record why and close.

- **Step 0 (operator GPU, 2×~10 min, one allocation):** measure the edge share
  the 1.5-3x estimate assumes (50-70%). Existing tool, sized batches:

      python utils/profile_sync.py -cn toptagging model=tag_cgenn save=false \
          training.batchsize=128 +prof.warm=8 +prof.active=5
      python utils/profile_sync.py -cn toptagging model=tag_CGENNLGATrGraphGPS save=false \
          training.batchsize=256 +prof.warm=8 +prof.active=5

  Deliverable back: both printed tables. I sum the edge-attributable kernels
  (messages fwd+bwd, gathers, segment reduces, their elementwise) into a
  measured ceiling per model, and the hot list picks what the kernel must fuse
  first. Top-tagging profiles stand in for jc architecturally (same code, same
  graph builder); one optional jc profile can confirm later. This step also
  re-baselines AFTER SortedGather/2.2b, which the 27.9% round-4 number predates.

  *Step 0, GPS HALF READ (2026-08-17, bs256 compiled, 5 steady steps, median
  2465 ms/step wall, CUDA busy 1870 ms = 76%): the plan's premise FAILS for the
  hybrid, and a bigger lever surfaced.* `aten::mm` is **77.7% of self-CUDA
  (7.26 s of 9.35 s)** at ~1055 calls/step — and the kernels under it are
  `sm80_xmma_gemm_f32f32` / `cutlass_80_simt_sgemm`: **fp32 WITHOUT tensor
  cores**, the direct consequence of the pinned `float32_matmul_precision:
  highest` (config/default.yaml:11). The blockdiag fc respelling routes the GP
  through mm; MVLinear/EquiLinear/GeoMLP supply the rest. The edge machinery
  the fused kernel targets reads only ~13-15% here (scatter-backward 879 ms =
  9.4%, index_mul 218 ms = 2.3%, small index/sum kernels the rest). Sync-hunt
  surplus beyond the known per-step reads: ~1443 `aten::item`/step and ~165
  `cudaStreamSynchronize`/step, with a ~24% GPU-idle wall gap — attribution
  pending (trace on disk). CONSEQUENCES, in order: (1) the **TF32 closure is
  REOPENED on new evidence** — it was closed as "a precision cut with
  unmeasured accuracy cost... under queue pressure", and the cost side is now
  measured: 78% of the slowest model's step is non-TC fp32 mm.
  `float32_matmul_precision=high` (knob already plumbed, base_experiment.py:443;
  a pure override, no code) puts every one of those GEMMs on TF32 tensor cores —
  realistic 3-5x on the big mm's, blended **~2-3x expected on the whole GPS
  step, for one word**. bf16 AMP (`model.use_amp=true`, wired, shipped off)
  stacks ~2x more on mm and halves activation traffic. Both are TRAINING-
  NUMERICS class changes (not equivalence-preserving): each gets ONE
  speed-confirm profile + accuracy validation against recorded baselines, not
  blind adoption; the fp64 TOL/BIT gate batteries keep their own explicit
  'highest' pins (test files set it locally) and are unaffected by the training
  config. (2) The fused message kernel is NOT the GPS hybrid's lever; its fate
  now rests on the tag_cgenn table (owed — it printed first in the same
  allocation) AND on the post-precision denominator: if mm gets 3-5x cheaper,
  the edge share RISES, so re-profile after the precision verdict before any
  kernel work. (3) The item()/sync surplus is a standing free-lunch candidate
  (~10-20% wall) — chase after precision lands.

  *Step 0 COMPLETE — tag_cgenn half (bs128 compiled, median 410 ms/step wall,
  CUDA 358 ms = 87% busy): same verdict, higher contrast.* `aten::mm` = **74.45%
  of self-CUDA** (1.334 of 1.792 s, 156 calls/step), all on the same
  `sm80_xmma f32f32` / `simt_sgemm` non-tensor-core kernels — at bs128 the
  blockdiag GP contractions are 15-17 ms GEMMs, 4/step each. Edge-attributable
  total ≈ 21%: scatter-backward 251.7 ms = 14.05% (4/step × 12.6 ms — this is
  the compiled backward of the indexed reads INSIDE sparse_gp_expression, the
  very thing the flash arm's bwd kernel eliminates), index_mul 4.2%, small
  index/add/segment kernels the rest. Sync surplus here: ~353 aten::item/step
  but only ~80 cudaStreamSynchronize/step and 87% CUDA-busy → ceiling ~13% on
  this model (GPS's ~24% gap is the bigger prize); model forwards grep CLEAN of
  host reads (the historical fixes hold), driver knowns ≈ 4/step, builder is
  vectorized (1 implicit sync) — remaining attribution needs the chrome trace's
  stack table, deferred until after precision. **FUSED-KERNEL CEILINGS, now
  measured:** at today's denominator the kernel buys ~1.20x on tag_cgenn (kill
  ~80% of the 21%) and ~1.13x on GPS — BELOW its own ≥1.5x success target: at
  current precision the kernel is dead on both. Post-TF32 the denominator
  shrinks ~2x and the edge share on tag_cgenn rises to ~44%, putting the kernel
  back at ~1.4-1.5x (borderline) — so the kernel go/no-go is decided by the
  post-precision re-profile, not before. A cheaper targeted alternative is
  queued ahead of it: a scatter-free respelling of sparse_gp_expression's
  BACKWARD (the 14% kernel), same TOL discipline as the forward blockdiag
  respelling, a fraction of the fused kernel's cost. Also queued from the
  step-0 readings, cheapest first: (a) inductor max-autotune on the precision
  round-trip (TORCHINDUCTOR_MAX_AUTOTUNE=1 — triton TF32 GEMMs can beat cuBLAS
  on the odd blockdiag shapes, and it retunes the scatter kernel); (b)
  split-linear node-hoisting audit — any per-edge LINEAR over gathered node
  features satisfies W[x_i;x_j;e] = (W_i x)|_i + (W_j x)|_j + W_e e, so those
  mm's can run at node count (16x fewer rows) and gather after — per-site
  audit, TOL/BIT per site; (c) validation overhead: the operator's own
  PlainGPS log shows 1049 s val / 7531 s train ≈ 12% — subsampled-val
  checkpoint selection + full val at end is config-only and worth days on
  weeks-long jc rows; (d) DDP for the jc CGENN rows — the single biggest
  wall-clock lever (×GPU count), currently unreachable (world_size always 1);
  real wiring work + a SyncBN class-change decision on theta_h BatchNorm;
  operator's call, scopable on request. NOT queued: bucketing (closed —
  accuracy), padding-free BatchNorm surgery (faithfulness pin), TF32 in the
  gate batteries (pins stay 'highest' by design).

  *PRECISION RACE MEASURED + FAIRNESS RESOLVED + PLAN REVISED (2026-08-17/18).*
  Measured at `float32_matmul_precision=high`: **tag_cgenn 410→211 ms/step wall
  (1.94x)**, CUDA 358→164 ms (2.19x), mm 74.5%→42.2% with `sm90_xmma tf32`
  kernels replacing the sm80 SIMT ones (~3.9x on the mm block; that run also had
  MAX_AUTOTUNE, whose own benchmark tables show cuBLAS winning nearly every
  shape — triton_mm at 2-100% of cuBLAS — so the win is TF32's, and
  **max-autotune is NOT adopted**; its "OutOfMemoryError: out of resource" spam
  is discarded candidate configs, harmless). **GPS 2465→1676 ms wall (1.47x)**,
  CUDA 1870→714 ms (2.62x), mm 77.7%→43.2% — and the wall/CUDA gap says GPS is
  now **SYNC-BOUND: 43% CUDA-busy** (was 76% busy at highest on tag) with the
  same ~165 syncs + ~1443 items/step. The scatter kernel is the new #1 single
  kernel on tag_cgenn (31.7% at high; 14% at highest).

  **FAIRNESS (operator challenge, resolved — this sets the precision policy):**
  per-row mixed precision is REJECTED as unfair, agreeing with the operator: an
  accuracy delta on a TF32 row is unattributable against fp32 rows, and the
  table's time(h) column would mix regimes. The two internally-fair spellings
  are uniform-`highest` (status quo; matches upstream CGENN, which simply
  inherits torch's own TF32-off default) and uniform-`high` table-wide with a
  methods disclosure (noting the published baselines themselves train fp16/bf16
  AMP). **DECISION: `highest` stays** — the operator's discomfort governs, it is
  scientifically the most defensible spelling, and the full-precision path below
  recovers most of the measured win anyway. The TF32 numbers above stay recorded
  as a one-word, table-wide, disclosed option should the calculus ever change.

  *Upstream re-scout (operator asked "have we fused as much as they do?"):*
  flash-clifford's fused kernels are `gelu→SGP/FCGP→RMSNorm` chains + grade
  linears — dense point-cloud ops ONLY: no message passing, no scatter/gather,
  no precision flags anywhere. flash-kingdon generates ONE fused
  `gelu→wGP→norm` kernel via kingdon+CSE; fully-connected layers are still on
  its TODO; same absences. So: we match their GP scope (generated,
  kingdon-checked, same basis as theirs); their signature act→GP→norm fusion is
  measured-unnecessary HERE because inductor already fuses those chains around
  the sparse spelling (compiled-sparse ≈ compiled-flash ±5%, races #5 and #7);
  and our two dominant costs (edge-gather mm's, scatter backward) do not exist
  in their workloads at all. Their repos are silent on TF32 because their
  kernels have no GEMMs for it to touch. [CORRECTED 2026-08-18, credit: the
  operator's Opus audit — an earlier revision of this note also claimed their
  fusion "pays off against EAGER baselines, which we are not"; flash-kingdon's
  README states its speedups are measured against a torch-COMPILED
  implementation, so that premise was wrong and is withdrawn. The conclusion
  stands untouched: it rests on the direct in-repo races (#5, #7, and the #5
  re-race below), which never relied on that premise.]

  **THE FULL-PRECISION PATH — step 1: GATHER-COMMUTE HOISTING (from the
  autotune shape log + fcgp.py read).** `FullyConnectedSteerableGeometricProductLayer`
  applies `linear_right` (MVLinear, and `linear_left` likewise) to the
  CONCATENATED GATHERED input `cat[x_i, x_j, x_i−x_j, e]` — a linear over
  gathers, and linears commute with gathers:
  `W·cat = (W_A+W_C)x |gathered at i| + (W_B−W_C)x |gathered at j| + W_E·e`
  (the x_diff slice folds by linearity). The giant EDGE-row mm's the profile
  names — `mm(393436×432, 432×432)` and `432×128` at bs128, the top mm kernels
  — become NODE-row mm's (14720 rows, **26.7x fewer**) for the x-slices, plus
  one gather+add; the `e`-slice is static per forward. The same split applies
  to phi_h's first Linear (h_i/h_j slices; m_invariants stays edge-level).
  CLASS: TOL (weight-slice pre-addition + gather reorder = reassociation only);
  hybrid pins re-record, stated. Estimate at `highest`: the top-3 edge mm
  families are ~70% of the mm block → mm ~267→~140 ms/step, step 358→~230 =
  **~1.55x**; with step 2 (scatter-free respelling of the sparse expression's
  BACKWARD, the 14%/12.6 ms kernel) **~1.9x combined at full fp32** — matching
  TF32's measured 1.94x without touching precision. Step 3: sync fixes once the
  new STEP-1c attribution run names the lines (profile_sync now prints the
  python frame behind every sync/item site; the empty-framesnet
  clip_grad_norm_ warning wall is also fixed). The FUSED MESSAGE KERNEL is
  SUPERSEDED unless a post-hoisting re-profile still shows ≥30% edge share:
  hoisting removes the mm's it would have fused and step 2 removes its other
  half. Sequenced, gated, and precision-independent — every step stands
  whatever the table's precision policy.

  *STEP 1 IMPLEMENTED (2026-08-18, CPU-verified): gather-commute hoisting is live.*
  `FCGP.message_right_left` (fcgp.py) computes the linear_right/linear_left halves
  of the CGL message at NODE level: with the weight sliced along the concat as
  [W_A|W_B|W_C|W_E], `W·cat[x_i, x_j, x_i−x_j, e] = (W_A+W_C)x|_i + (W_B−W_C)x|_j
  + W_E·e` — the slice combos feed `mv_apply_weight` (linear.py, the blockdiag
  fast path as a function of an explicit weight), the gathers are the
  deterministic SortedGather/SortedGatherPermuted the CGLs already thread, and
  linear_left's bias is added exactly once at the recombination. Both CGL twins
  gained `_message_x_hoisted` (fc layer type only; gpmlp untouched); the
  quadratic GP keeps the plain edge concat. KILL SWITCH: CGENN_HOIST=0 (read at
  import, fcgp.py). CLASS: TOL — reassociation only; gates
  (tests/internal/test_hoist_message.py, 5/5): forward parity 1e-15, all grads
  (x, e, both linear weights, bias, GP weight) 1e-13..1e-16 vs the taxonomy's
  1e-8 grad bar, full hybrid-CGLayer forward parity 5e-16, kill-switch identity,
  compile 0 breaks. CONSEQUENCES, all handled: hybrid BIT pins re-recorded
  (3 files; GPS fp32 pin unchanged — the same benign fp32-rounding coincidence
  as the blockdiag era), cgenn_compile BIT fixtures + content-hash manifest
  re-recorded, and the fp32 impl-TOL bar re-set 1e-5→1e-4 with the measurement
  in its comment (the einsum-vs-blockdiag reassociation seed is unchanged but
  its 4-layer amplification moved with the activations: 4e-6-class → 3.3e-5;
  CGENN_HOIST=0 reverts it; the fp64 arbiter stays ~1e-13 under its unchanged
  1e-10 bar). Full battery 25/25, internal+experiments CPU suites 297 passed.
  INTERACTION, stated: the hoist adds 4 deterministic gathers per layer, whose
  backwards are segment_reduce calls — MORE of the syncs step 2 removes; on the
  87%-CUDA-busy tag_cgenn they overlap, and step 2 retires the whole class.

  *SYNC ATTRIBUTION (from the operator's STEP-1c run, 2026-08-18): the syncs are
  `torch.segment_reduce`'s.* The NGC build captures no profiler stacks (STEP 1c
  printed its fallback), but the nesting column is decisive: aten::segment_reduce
  self-CPU 8 ms yet CPU-TOTAL 5.55 s = 27.6% — the 825 cudaStreamSynchronize and
  the expensive aten::item calls sit INSIDE it. 300 calls/5 steps = 60/step
  matches the 2.2b aggregations + SortedGather(+Permuted) backwards exactly.
  Mechanism: segment_reduce has no inductor lowering (falls back inside the
  compiled region) and its CUDA path host-reads the lengths per call. STEP 2
  DESIGN (frozen): receiver-side segments are k-BOUNDED BY CONSTRUCTION (each
  receiver has ≤ knn_k neighbors), so the deterministic segment sum respells as
  a STATIC-shape padded scatter-write — slot = arange(E) − offsets[i] (offsets =
  exclusive-cumsum of edge_counts, tensor-only), out = zeros(N, k, C)
  .index_put_((i, slot), data).sum(1) — unique slots ⇒ no atomics, no host
  reads, plain ops ⇒ inductor fuses it, and it retires BOTH the fallback syncs
  and the scatter class for: the 2.2b forward aggregations, SortedGather
  backward, and the hoist gathers' backwards. Sender-side segments are NOT
  k-bounded (a popular sender is unbounded) — sender backwards keep
  segment_reduce in v1; if their residual syncs still bind after 2a, options
  recorded: CPU-side lengths (if ATen accepts them, the .max() is free) or a
  two-level chunked reduction. TOL again → one more pin re-record, folded into
  the step-2 commit.

  *Step-1 GPU round-trip #1 (2026-08-18): crashed on MY command bug — a CPU-tier
  gate on the GPU node; nothing about the hoist failed.* The command led with
  test_hybrid_bit_pin.py, whose pins are CPU recordings: on the H100 the forward
  is cuda:0, the pin loads as cpu, torch.equal refuses cross-device, and set -e
  killed the battery/soak/profiles unrun (the 16 device-generic sorted_gather
  gates that did run passed). The gate had only ever executed in CPU sessions,
  where it belongs: BIT-identity is a SAME-DEVICE statement — CUDA and CPU
  kernels are not bitwise comparable — so the fix declares the tier in the test:
  test_bit_eager_vs_pin and the battery's test_bit_eager_vs_fixtures (identical
  latent crash, caught in the audit before it fired) now SKIP on CUDA with the
  reason printed, record-path included (a GPU re-record would silently flip the
  fixture's device). Audit also hardened FCGP.message_right_left for
  include_first_order=False (no linear_left exists there; returns left=None) and
  swept every other fixture-vs-live torch.equal in the experiments tier — none
  remain (DET gates compare same-process runs). Round-trip #2 = the same command
  MINUS the pin file (already 8/8 in the sandbox, where it is authoritative).

  *STEP 1 MEASURED (round-trip #2, 2026-08-18): tag_cgenn 418→381 ms/step
  (1.10x); GPS wall NULL — and the attribution stands corrected.* All gates
  green first (sorted_gather 16/16; battery 23 passed + the 2 CPU-tier BIT
  skips, as designed; soak 3/3, losses matching the pre-hoist trajectories).
  The A/B: tag CUDA 362→319 ms/step, mm share 74.5→66.8%; GPS CUDA 1892→1751
  (−7%) but wall 2443→2453 — UNCHANGED, because GPS is sync-bound and the
  hoist's four extra gathers/layer ADDED segment_reduce backward syncs
  (165→245/step), spending the mm win on more stalls. ATTRIBUTION CORRECTION,
  the round-trip's real lesson: the three GIANT mm kernels (17.5 + 14.8 +
  7.1×2 ms per layer ≈ 58% of tag CUDA) did NOT move — they are the sparse GP
  contraction's OWN forward+backward GEMMs, edge-level and quadratic, not
  hoistable by linearity. The hoisted linears were the mid-size family (~20%
  of the mm block), which did vanish into node-level calls. Step-0's
  shape-log inference over-attributed the giants; ON/OFF kernel deltas are
  the arbiter. VERDICT: step 1 KEPT (real 1.10x on the compute-bound net;
  its GPS share unlocks when the syncs die). New in the ON-profile: fused
  index_put-bearing backward kernels (18.9 ms×1/step tag, 25 ms×10/step GPS)
  — attribute via trace if they survive step 2. Consequence for the plan:
  the flash arm's kernel replaces exactly those giant contraction GEMMs, so
  a flash re-race is queued AFTER step 2 lands — the fusion-tax landscape
  that produced the two CLOSE verdicts has changed twice since.

  *STEP 2 IMPLEMENTED (2026-08-18, CPU-green): padded scatter-write segment
  sums are live receiver-side.* `padded_segment_sum` (sorted_gather.py) +
  slot/K threading through both CGL twins: the net forwards compute the
  per-edge in-segment rank (exclusive-cumsum arithmetic, tensor-only) and
  pass the STATIC degree bound — hybrids/GPS pass knn k itself (the
  structural bound; deliberately NOT min()'d with the symbolic P and never
  int()-cast, either of which would plant shape guards / re-specialize under
  compile(dynamic=True); k=None fully-connected mode keeps segment_reduce),
  pure CGENN's wrapper passes n_nodes−1 (an eager python int, no host read;
  MEASURED TRADE flagged in-code: the FC graph's padded buffer inflates by
  (n_nodes−1)/avg-degree — if round-trip #3 shows tag_cgenn regressing, that
  one wrapper line passes None). Routed: both twins' mean-aggregation core,
  the x_i/h_i gathers, and the hoist's rA/lA gathers; sender-side (x_j, h_j,
  rB, lB) keeps segment_reduce in v1 (unbounded degree). [CORRECTED by the
  round-trip #5 audit, below: the original text here claimed the padded sum
  is "BIT-equal ON CPU ... zero fixture churn". That was TRUE for the hybrid
  BIT pins (kNN graphs have CONSTANT degree k, so the padded buffer has no
  ragged padding and the sums agree bitwise) but FALSE for the FC net: torch's
  CPU `sum(dim=1)` reduces with vectorized (non-sequential) association, so on
  ragged FC segments padded differs from segment_reduce at reassociation level
  (6e-7 fp32 / 3e-15 fp64 model-level), and this step's commit in fact
  re-recorded the battery's fp32.pt/fp64.pt fixtures — churn the record
  under-reported. The audit made the padded route COMPILED-ONLY, which
  restored eager bit-exactly to the pre-step-2 recordings (verified).] Gates: padded
  parity 1.4e-16 + looser-K identity, slot/uniqueness executable contract,
  slot-route backward 1.2e-16, gradcheck+gradgradcheck, compile 0 breaks;
  internal+experiments 281 passed; battery 25/25 (explain artifact
  re-recorded: the slot arithmetic's ops). Expected on GPU: tag syncs
  112→~30/step, GPS 245→~75 (senders remain), attacking GPS's measured
  700 ms/step wall-CUDA gap. Round-trip #3 = round-trip #2's commands
  verbatim.

  *ROUND-TRIP #3 VOID + TWO FIXES + ONE ADOPTED FINDING (2026-08-18).* #3's
  `git pull` ABORTED ("local changes to dynamo_explain.txt would be
  overwritten") and, with set -e not yet active on that line, every command ran
  the PREVIOUS tree — its numbers replicate round-trip #2 to ~1% (tag 378
  ms/step, 560 syncs vs 381/560; GPS 2445/1225 vs 2453/1225): a NULL for step 2
  but a clean run-to-run stability reading. Cause: the battery's explain-artifact
  write ran on EVERY gate run, including cluster GPU runs, dirtying the checkout.
  FIXES: (1) the artifact write is now CPU-tier (GPU nodes never write it —
  matching the BIT fixtures' tier); (2) the runbook pull is now
  `git fetch origin main && git reset --hard origin/main` — the cluster checkout
  is a consumer, never a source of changes.
  ADOPTED (credit: the operator's Opus audit): `torch.segment_reduce`'s DEFAULT
  `unsafe=False` validates its lengths with per-call HOST READS — device syncs
  re-deriving an invariant that is true by construction and pinned STRONGER than
  the min>=0/sum==E checks by the executable bincount contracts
  (tests/experiments/test_sorted_gather.py). `unsafe=True` now set at all four
  remaining segment_reduce sites: both Function backwards — including the
  SENDER side, which step 2's k-bounded padding could not cover — and both
  twins' aggregation fallback branches. Same kernel, BIT-identical (verified:
  sorted_gather + hoist + pins + full battery all green with ZERO fixture
  re-records). COMPOSITION, stated: unsafe=True removes the validation syncs
  wherever segment_reduce still runs; the padded respelling additionally removes
  the aten fallback itself (its fusion partitions and kernel launches)
  receiver-side. Round-trip #4 measures both at once; the switches
  (CGENN_HOIST, the wrapper's knn_k line) isolate contributions only if the
  combined read demands it.

  *ROUND-TRIP #4 MEASURED (2026-08-18, H100, uncontended): STEP 2 + unsafe
  KEPT — the sync war is won; both models are now COMPUTE-BOUND.* Gates:
  sorted_gather 23/23, battery 23 passed + 2 skipped (the CPU-tier BIT gates
  skipping on CUDA, by design), soak 3/3 with loss trajectories matching the
  pre-step-2 rounds (0.690→0.574 / 0.718→0.512 / 0.693→0.584). Numbers:
  **CGENN-GPS bs256: 2453→1927 ms/step (1.27x)**, syncs 245→45/step
  (prediction was ~75), and the wall-vs-CUDA gap collapsed 600→70 ms/step
  (CUDA-busy 1856 = 96% of wall) — the launch/sync-bound regime the whole
  FLASH-3 sync track targeted is OVER. **tag_cgenn bs128: 381→365 ms/step
  (1.04x; 1.15x cumulative vs the 418 pre-hoist baseline)**, syncs
  112→32/step (prediction ~30), CUDA-busy 333 = 91% of wall. Consequences,
  in order: (1) mm is 67-68% of self-CUDA on BOTH models — the giant sparse-GP
  contraction GEMMs unchanged (tag 18.8/15.1/7.5 ms; GPS 20.3/18.1/4.8 ms) —
  so the flash re-race is now the only first-order lever; (2) sender-side
  segment_reduce (60 calls/step on GPS) is sync-free under unsafe=True and
  only 0.34% of CUDA → **step 2b (sender-side padded sums) is DEPRIORITIZED
  to dead** — its target no longer exists; (3) the residual syncs (32/45 per
  step) overlap a saturated GPU and include ~10/step from on-GPU
  aten::nonzero (tag; DtoH of the count) — recorded, not actioned, same
  reason; (4) AUDIT CATCH on the padding trade: the fused padded-aggregation
  kernel (`triton_per_fused_index_index_put_mul_new_zeros_sum`, 4/step ×
  13.3 ms) is 16% of tag CUDA — the FC graph is the one place K=n_nodes−1 is
  loose, and unsafe=True has made the segment_reduce alternative sync-free
  too, so the trade deserves its counterfactual: **CGENN_FC_PADDED=0**
  (wrappers.py, read-once shield style; both positions smoke-verified on
  CPU) rides along in round-trip #5 as a one-env A/B. GPS backward shows
  launch pressure (Command Buffer Full 791 ms/5 steps, cuLaunchKernel 44 us
  avg) — absorbed while compute-bound, becomes relevant only if flash
  shrinks the GEMMs. Audit sweep also fixed: `experiments/logger.py` used
  `logging.handlers` without importing the submodule (import-order-dependent
  latent ImportError), and the three `\psi` docstrings now raw strings (the
  SyntaxWarning in every GPU log).

  *ROUND-TRIP #5 (2026-08-18, H100, uncontended): FLASH RE-RACE **CLOSED,
  FINAL** — and the A/B is a TIE that corrects #4's attribution.* bperf,
  own-best per arm: **sparse-compiled 349 jets/s (bs128) vs flash-eager 308
  jets/s (bs256)** — the challenger is now 12% BEHIND, where the two CLOSE
  verdicts had it +5% ahead. Mechanism: the FLASH-3 sync/hoist/padded work
  landed almost entirely on the sparse arm's side of the ledger (313→349
  jets/s, +11.5% since race #7) while flash-eager gained ~3% (297→308);
  flash-COMPILED collapsed outright to 168 jets/s (0.547x) — the opaque
  custom-op boundary in a now-faster surrounding graph is pure fusion tax.
  With a −12% gap against a >10% adopt bar, no further landscape change can
  rescue the arm: **sparse is the campaign arm, flash stays in-tree solely as
  the recorded memory escape hatch (0.146 vs 0.537 GB/jet), and Tier-B fused
  norms — conditional on flash adopting — are DEAD.** The
  `CGENN_FC_PADDED=0` rider read 366 ms/step vs 365 padded, syncs 32/step
  identical, CUDA totals identical (1.668 vs 1.663 s/5) → **TIE; the padded
  default stays** (and the switch stays for free). ATTRIBUTION CORRECTION to
  #4's audit note: the 13.3-13.5 ms `triton_per_fused_index_index_put_*`
  kernels persist at 3/step with padding OFF (plus segment_reduce visibly
  back at 200 calls/17 ms), so that family is dominated by the COMPILED
  sparse-GP/aggregation backward reductions — deterministic, fused,
  bandwidth-bound — not by FC padding inflation. Consequence for the one
  residual idea (respelling the sparse-GP backward's indexed reads, e.g. as
  one-hot GEMMs): CLOSED WITHOUT A ROUND-TRIP — the kernel is already an
  atomic-free fused reduction reading tensors any spelling must read, and
  un-fusing it into cuBLAS calls is the same fusion-tax mechanism just
  measured sinking flash-compiled twice. What remains on both models is mm at
  67-68% of a 91-96% CUDA-busy step at pinned-`highest` fp32 — the fairness
  floor, not an engineering gap. FLASH-3 and the CGENN performance program
  END HERE; cumulative tag_cgenn: 4.42x compile speedup, 313→349 jets/s
  (+11.5%) across FLASH-3, syncs 112→32/step; GPS 2453→1927 ms/step (1.27x),
  245→45 syncs/step.*

  *ROUND-TRIP #5 ADVERSARIAL AUDIT (2026-08-18): one real flaw found and
  fixed — the padded segment sums ran in the EAGER posture too, where
  inductor is not there to fuse them: eager `padded_segment_sum`
  MATERIALIZES the (N, K, C) buffer — multiple GB per call on the FC graph
  at K=n_nodes−1 — a time AND memory tax on every eager run. This
  CONTAMINATED THE RE-RACE: flash's own-best posture is eager (its compiled
  posture pays the fusion tax), so the challenger raced carrying a handicap
  the incumbent's compiled posture never felt — visible in the data as
  flash-eager DROPPING 329→308 jets/s from race #7 and losing its bs-512
  ceiling to 256 (the padded buffers inflated its memory). FIX: the padded
  route is now COMPILED-ONLY (`torch.compiler.is_compiling()` gate at all
  three slot-producer sites — the same compile-twin pattern as
  sparse_gp.py; constant-folded at trace time, zero breaks). Eager keeps
  segment_reduce(unsafe=True) everywhere. PROOF the restoration is exact:
  re-recorded BIT fixtures are tensor-bit-IDENTICAL to the pre-step-2
  (step-1) recordings — which also proves unsafe=True is BIT at model level
  on this net (the old recordings predate it). Battery 25/25 green.
  VERDICT IMPACT, stated honestly: the −12% margin OVERSTATES flash's
  deficit; un-handicapped flash-eager is likely near its historical ~330
  jets/s ≈ −5% vs sparse's 349. The CLOSE verdict STANDS regardless —
  adoption needs >384 jets/s (+10% over sparse), +17% above flash's
  best-ever — but the margin in the verdict record is corrected to ~−5%
  (estimated) pending an optional one-row hygiene rerun:
  `python utils/bperf.py --models tag_cgenn/flash --find-batchsize`.
  Also checked and CLEAN: twin files code-identical (AST-compare, comments
  aside — the drift is deliberate rationale-thinning); eval runs the same
  in-place-compiled net as training (no eager-eval exposure in campaign
  configs); K is a shape-derived SymInt (no per-batch recompile);
  `test_bit_eager_vs_fixtures` did its job — it caught the eager change the
  moment the compile-twin split landed, which is what exposed the
  under-reported step-2 fixture churn corrected above.*

  *AUDIT ROUND 2 (2026-08-18, operator-requested pause): three hardenings,
  one process gap closed, one new permanent gate.* (1) HAZARD CLOSED:
  `message_right_left`'s weight slicing is only correct for the
  cat[x_i,x_j,x_diff,e] layout; called on any other FCGP (e.g. theta_x's) it
  would mis-slice SILENTLY — `w[:, 2c:3c]` can be empty with no shape error.
  Now asserted (`e_ch >= 0`); channel dims are static under compile so the
  assert folds — hoist compile gate still 0 breaks. (2) STALE COMMENT fixed:
  the hybrid backbone claimed the model passes `min(k, P-1)`; the code
  deliberately passes k itself (min() on symbolic P = shape guard, the
  RECOMP ban) — comment now states the design instead of contradicting it.
  (3) PROCESS GAP: after the compile-twin split landed, only the tag_cgenn
  battery was re-run — the hybrid/GPS COMPILED posture was not; the
  compiled training smoke for all three CGENN nets was re-run to close it
  (3/3 passed, CPU inductor, 41 min).
  (4) NEW GATE `test_degree_zero_node_compiled_vs_eager` (battery,
  CGENN_COMPILE_GATES=1): a crafted SINGLE-CONSTITUENT jet — real node,
  zero FC edges, degree-0 receiver AND sender — through forward + parameter
  gradients, eager vs compiled at fp64. This is where the padded write
  (compiled), segment_reduce (eager), clamp(min=1) divisor and hoisted
  gathers all meet, and it is data-reachable (low-multiplicity jets); the
  RECOMP sweep only pushed that shape through a no_grad forward. First
  reading: fwd 2.7e-13, grads 2.4e-10, no NaN — PASS under the 1e-12/1e-8
  bars. CHECKED CLEAN: kNN degree<=k is structural (no clamp needed at
  P<=k); GPS layer threads slot/K into its cgenn correctly; twin phi_x
  structures identical; internal suite 266 passed. NOTED, left in place:
  CGL.__init__ assigns `self.aggregation` a function then overwrites it
  with the string (upstream artifact; the string drives reduce(), the
  early raise still validates — dead but harmless).*

  *REGIONAL COMPILATION WIRED (2026-08-18, operator-approved): flash
  round-trip #6.* The one cheap experiment whose upside reaches the +10%
  bar. Mechanism: flash-compiled loses because the joint AOT graph must
  partition around the opaque custom op (measured 0.547x); flash-eager
  loses because the scalar MLPs forfeit inductor's fusion. REGIONAL takes
  the third corner: `CGENN_REGIONAL=1` + `compile=true` makes CGENNWrapper
  compile each plain-nn Sequential (phi_h/theta_h/psi_x/chi_x + head; the
  structural predicate `_compile_regional` in wrappers.py excludes anything
  GA-shaped) as its OWN unit with dynamic=True, while the orchestration —
  and the flash op inside it — stays eager Python. No joint graph ever
  contains the op → no partition seams; the MLPs still fuse. Python
  dispatch is priced: full flash-eager already runs thousands of eager ops
  at 308 jets/s, and regional strictly reduces that count. The open cost is
  LOST CROSS-UNIT FUSION, which the race prices. Gate:
  `test_regional_compile_vs_eager` (battery) — 5 units compiled, forward
  AND grads bit-identical to eager on CPU (0.0 diff). ROUND-TRIP #6
  COMMAND (one allocation, ~30 min; head as always fetch+reset, set -e):

      python utils/bperf.py --models tag_cgenn/sparse tag_cgenn/flash --find-batchsize
      CGENN_REGIONAL=1 python utils/bperf.py --models tag_cgenn/flash --find-batchsize
      # campaign prep rider (independent of the race; jc batch sizes for the
      # non-CGENN GPS rows -- NOTE the flag is --task, find_bs has no -cn):
      python utils/find_bs.py --task jctagging \
          --models PlainGraphGPS ParticleNetParTGraphGPS LorentzNetLGATrSlimGraphGPS

  Read: command 1 = sparse control (compiled column, ~349) + FAIR
  flash-eager (eager column; the materialization bug is fixed, bs512
  should return). Command 2's COMPILED column = regional (ignore its eager
  column, a duplicate). NEVER `--apply` from the regional invocation: its
  recommended one-liner would set `compile: true` in the yaml, which
  WITHOUT the env var means whole-net compile — a different arm than the
  one measured. DECISION RULE, pre-registered: regional >= 384 jets/s
  (+10% over sparse's 349) reopens flash adoption; anything less closes
  the flash arm finally, with the mega fused kernel remaining a separate
  discretionary call.*

  *ADVERSARIAL AUDIT ROUND 3 (2026-08-18, on the freshest code): one real
  crash-class find — BOTH new gates (DEG0, REGIONAL) created their probe
  weights as `torch.linspace(..., dtype=y.dtype)` with NO device argument,
  the exact cross-device bug the original `_grads` helper already carries
  the fix for (its round-trip lesson: "probe weights created without a
  device — first CUDA run of that gate ever"). On the next GPU battery, y
  is CUDA, w is CPU, and both gates raise RuntimeError — a wasted
  round-trip. Fixed (`device=y.device`), and both gates' forward bars
  relaxed 1e-12 → 1e-10 to match the battery's established model-level
  fp64 compiled-vs-eager bar (GPU reassociation exceeds CPU's; 1e-12 was
  a spurious-red risk). Unit-count expectation documented in-test (5 =
  quick tree n_layers=1; production ~17 — the assert is a floor).
  CHECKED CLEAN: regional state_dict compatibility (nn.Module.compile
  wraps in place, keys unchanged — why strict=True load works), the
  gate's BN-stats comparison ordering (both models forward from identical
  loaded state), monkeypatch env ordering, the two-invocation isolation
  of the runbook (env applies per bperf process), and the LOGGER import
  at the wrapper's regional branch.*

  *ROUND-TRIP #6 MEASURED (2026-08-18, H100, uncontended): **REGIONAL
  NULL; FLASH ARM CLOSED, third and final CLOSE — and one more
  attribution corrected.** Numbers, own-best per arm: sparse-compiled
  **356** jets/s @128 (4.413x compile, replicating 349 to +2%);
  flash-eager **338** @512 — the fair number, landing on the corrected
  ~−5% estimate (−5.1% measured) and confirming the eager-materialization
  diagnosis: with the padded sums compiled-only, eager sizing fits bs512
  again (peak 75.6 GB; 1024 OOM). Flash-REGIONAL (CGENN_REGIONAL=1):
  **337** @512 = 0.999x vs eager — the regional hypothesis is REJECTED
  cleanly: compiling the scalar MLPs as units adds nothing measurable, so
  flash-eager's deficit does NOT live in un-fused MLP time (it lives in
  the GP-adjacent fusion sparse gets and flash cannot). The SURPRISE:
  whole-net flash-COMPILED at bs512 = **382** jets/s (1.132x over its
  eager) — flash's best posture flipped to COMPILED for the first time.
  ATTRIBUTION CORRECTION to the #5 audit: the re-race's flash-compiled
  0.547x collapse (168 @256) was NOT partition-tax growth — bperf sizes
  ONCE, EAGER, for both postures ("sized once eager" in its log), so the
  eager materialization bug capped BOTH flash columns at bs256; at the
  restored bs512 the compiled posture amortizes its launch/partition
  overhead and comes out ahead. The #5 record's "compiled row was on
  essentially fair footing" claim is withdrawn. VERDICT under the
  pre-registered rule: 382 vs 356 = **+7.3% < the +10% bar (392)** —
  CLOSE, the third at own-best (+5.4%, +5.1%, +7.3%), and per the
  round-trip #6 pre-registration the arm CLOSES. Sparse ships the CGENN
  rows. The one reopening path remains the fused message kernel
  (kingdon-v3 collaboration relevant), operator-discretionary. RIDER
  RESULTS: jc GPS batchsizes measured and transcribed — PlainGraphGPS /
  PNPT-GPS / Slim-GPS all bs512 (peaks 17.8/46.9/47.2 GB, 1024 OOM);
  their jc lr values remain ??? pending the jc find_lr sweeps. TWO
  FINDINGS FROM THE RIDER LOGS: (1) plaingraphgps.py:191
  `out[mask_bool] = self.norm(h[mask_bool])` (masked BatchNorm) hits an
  inductor BACKEND EXCEPTION on this container (`aten.nonzero.default`
  → "Adding a graph break") on PlainGraphGPS and PNPT-GPS jc runs —
  tolerated (falls back, training proceeds; the top rows trained fine
  with the same code), PRE-EXISTING, not from this program; recorded as
  a candidate for the same upstream-issue filing as ledger item 8, with
  an optional respelling (hoist the nonzero outside the compiled region
  and index_select) if a jc profile shows the breaks cost real time.
  Slim-GPS is clean (different norm spelling). (2) The "UNSWEPT family
  fallback" warning in the bperf logs is BENIGN in that context — bperf
  deliberately composes without training recipes; top_cgenn.yaml is
  filled (bs 128, lr 5.57e-4) and the earlier audit's "tag_cgenn lr
  re-derivation open" flag was WRONG — the yaml's own comment records
  the aligned-finder derivation at bs 128. That ledger item is DONE.*

  *ROUND-TRIP #7 WIRED (2026-08-18): `--size-per-state` — the #6 verdict's
  sizing bias, fixed.* THE FINDING that motivates it came from reading
  bperf's own `find_batchsize` docstring after #6: it sizes each row ONCE,
  in EAGER mode, applies that batch to both postures, and says outright
  "these it/s are measured at a batch nobody trains at ... do not rank
  gp_impls by it/s from this driver." Every flash race ranked exactly that
  way. Consequence: sparse-compiled has never been raced above bs128 and
  flash-compiled never above 512, because those are EAGER ceilings — and
  the eager/compiled memory gap is PER-ARM (the same ~6x retention spread
  the docstring already flags), so the understatement is uneven and the
  +7.3% close rests on it. IMPLEMENTED: `find_batchsize(..., state=)` +
  driver flag `--size-per-state` (one search per posture, each timing run
  at its own batch). Three consequences handled in code, not prose:
  (1) the speedup column becomes a JETS/S ratio — an it/s ratio across
  different batches is not a comparison — via one formula that reduces
  exactly to the it/s ratio when the sizes agree (so the default path is
  untouched, pinned by a test); (2) `--apply` is REFUSED with
  `--size-per-state` (the verdict includes a batch change the yaml does
  not carry, so the knob alone would not reproduce it) and the report
  carries the same warning; (3) the verdict string is tagged
  "(throughput, incl. batch)". Gates: 3 new tests in
  tests/internal/test_bperf_driver.py (per-state runs each state at its
  own batch + jets/s ratio + "128/512" cell; default path still sizes ONCE
  eager with the it/s ratio; both refusals fire before any run) — 14/14.
  **SCIENTIFIC CAVEAT, pre-registered before the numbers arrive:** if #7's
  winner wins AT A DIFFERENT BATCH SIZE than the incumbent's, adoption is
  NOT a drop-in — batch size and lr are coupled, so a flash arm adopted at
  a larger batch needs its lr re-derived (the finder, at that batch) and
  its accuracy revalidated, exactly as the table-wide lr decision was.
  That cost belongs in the adopt decision alongside the >10% bar; a win
  that is purely "bigger batch fits" is a REAL wall-clock win but a recipe
  change, and must be recorded as one. ROUND-TRIP #7 (one allocation,
  ~2.5h — compiled sizing spends an inductor build per row inside the
  driver):

      python utils/bperf.py --models tag_cgenn/sparse tag_cgenn/flash \
          --find-batchsize --size-per-state
      # then, at whatever batch the flash-compiled column reports:
      python utils/profile_sync.py -cn toptagging model=tag_cgenn \
          model.net.gp_impl=flash save=false training.batchsize=<that bs> \
          +prof.warm=8 +prof.active=5

  Read: the bs cell now reads eager/compiled per row; compare the
  COMPILED jets/s columns (that is the own-best race) against the >10%
  bar. From the profile, record the flash KERNEL's share of CUDA time —
  that is the ceiling for the two cheap unpriced tricks (Triton autotune
  of the generated kernels, ~10-30% of kernel time; kingdon v3's 2-3x
  interior-op reduction) and the input the fused-message-kernel decision
  needs. NEVER `--apply` from this run (the driver refuses).*
  message pipeline (message_x = concat[x_i, x_j, edge_attr] → FCGP → gate;
  invariants; message_h scalar MLP) and freeze the fusion boundary — mv stream
  in-kernel, scalar stream in/out per step-0's table; verify edge_attr_x needs
  no gradient (static, from raw momenta). Extend utils/flash_gen.py to emit the
  per-edge fused body (gather→GP→invariants→gate) + its analytic gradient as
  generated, CSE'd, kingdon-checked functions — same generate-don't-transcribe
  basis as flash, so the math is 3e-16-gated before any kernel exists. CSR
  offsets for receiver segments = cumsum(edge_counts), computed once per
  forward beside the existing degree hoist.
- **Step 2 (sandbox → 1 GPU round-trip):** FORWARD kernel: one program per
  (receiver-segment block); loads node arrays via indices, computes messages in
  registers, segment-sums into node slots (sorted receivers → atomic-free,
  deterministic). CPU composite twin + parity/AST/binding-name gates first;
  then the round-trip runs parity fp64/fp32 + DET + bench.
- **Step 3 (sandbox → 1-2 GPU round-trips):** BACKWARD, recompute-based: saves
  only node arrays + weight + indices; recomputes per-edge quantities;
  receiver-grad via the sorted segment sum, sender-grad via send_perm (the
  FLASH-2b machinery), dL/dw via partial buffers + fixed-order torch sum(0)
  (the flash determinism pattern). gradcheck/gradgradcheck fp64, joint-compile,
  FakeTensor, empty-segment gates.
- **Step 4 (1 GPU round-trip):** net integration as a gated arm at the CGL
  boundary (composite path untouched), compile battery + 100-step soak + the
  race: `bperf --models tag_cgenn/sparse tag_cgenn/fused --find-batchsize`
  (+ hybrid row if step 0 shows the hybrids' share supports it). ADOPT re-records
  hybrid pins (forward evaluation order changes: TOL-class re-record, stated).
- **Step 5:** the CGENN rows launch on the winning arm — fused if adopted by the
  time the campaign queue reaches them, sparse otherwise. Either way the rows
  never wait on this plan.

Schedule honesty, FLASH-3 edition: flash's 7 round-trips priced a far simpler
kernel; plan for ~10-15 round-trips over 1-2 weeks, front-loaded to CPU so GPU
failures localize (the strategy that caught every flash failure in one line).
The window is the non-CGENN campaign's own runtime; if the kernel misses the
window, sparse ships and the work closes with a recorded race, not a loss.

Schedule honesty: steps 1-2 are one working session; step 3 is the risk
concentration (Triton iteration without a local GPU — mitigated by step 2 making
the math pre-verified and by a torch-reference twin for every kernel so GPU parity
failures localize); expect 2-3 operator GPU round-trips across steps 3-4. The CGENN
rows launch a few days late; the operator has accepted that trade explicitly.

### LGATr-family recipe question DECIDED (2026-08-18): uniform recipe stays; authors' numbers quoted as external

Operator measured: LGATr-slim reaches 0.9420 under its authors' recipe but 0.9406
under the table's uniform recipe (a real +0.14pp = ~3.5σ recipe effect), while the
LNSlim-GraphTrans hybrid reads 0.9412 under the uniform recipe. DECISION, same logic
as the precision-fairness call: the table's internal comparison keeps ONE training
protocol for every row — per-model author recipes would make architecture
unattributable and invite a recipe-tuning contest (each family's authors' settings
would then be owed to every other family too). The WITHIN-protocol comparison is the
valid one, and there the hybrid is fine: 0.9412 vs 0.9406 (+0.06pp, ~1.5σ). The
0.9412-vs-0.9420 reading that looks like underperformance is CROSS-protocol — two
variables, no conclusion. Paper treatment: uniform-recipe numbers form the table;
authors'-recipe numbers are quoted as external references (same pattern as the
scheduler note below). RECOMMENDED cheap decisive: ONE run of the hybrid under the
lgatr recipe — if the +0.14pp recipe effect transfers (→ ~0.942+), the
hybrid-vs-parent ordering is recipe-invariant and the claim holds under both
protocols; either outcome is one honest footnote, not a protocol fork.

### Scheduler bake-off MEASURED (PNPT, 6 arms, n=1 each, 2026-08-16): STICK with CosineAnnealingWarmup

Operator ran all six schedulers at lr 1e-3 / bs 512 / 20 epochs. Readings (test acc /
rej0.3): OneCycle 0.9417/1803; **CosineAnnealingWarmup 0.9414/1771 (highest train acc
0.9458)**; WarmRestarts 0.9413/1836 (best val 0.9408); flat+decay 0.9409/1771;
"ReduceLROnPlateau" 0.9407/1530; CosineAnnealingLR 0.9405/1771. Verdict and grounds:

- **KEEP CosineAnnealingWarmup — the FINAL marker stands.** The top four arms span
  0.0004 test acc = exactly the measured two-seed spread at fixed config, and the
  rejection spread (1771–1836) sits inside the measured 79-point seed noise. At n=1
  nothing separates them; the incumbent is one of them, is the GraphGPS-family
  precedent, and is what already-launched rows run.
- **The two effects the table DOES support both endorse the incumbent.** (1) Warmup:
  the cosine-vs-cosine+warmup pair is the cleanest single-variable measurement in the
  set — +0.0009 acc (~3 sigma of single-run noise) for the 5% ramp, matching the
  Adam-second-moment mechanism. (2) Annealing: the accidental constant-lr arm
  (ReduceLROnPlateau, patience 50 vs 20 validations — operator-diagnosed as inert,
  correctly) is the flat baseline, and every annealed arm beats it by 13–16%
  REJECTION at near-equal accuracy. Any anneal buys the rejection; the shape barely
  matters.
- **OneCycle's headline is confounded and sub-noise**: `cycle_momentum` defaulted
  True, so that arm cycled beta1 alongside lr (two variables), and its +0.0003 over
  the incumbent is under seed spread. Its weaver-matching 0.9417 is coincidence, not
  corroboration (weaver used flat+decay, 4th here). If post-table polish is ever
  wanted: 3-seed OneCycle with `cycle_momentum=False` vs incumbent — recorded, not
  scheduled.
- **flat+decay's 4th place carries no external-validity alarm**: the top-tagging
  variant is a coarse 6-step staircase (the 15-step smoothing is jc-gated,
  base_experiment.py:666), and its low train acc (0.9428) says it under-optimized —
  the published smooth version is a different arm than the one measured here.
- WarmRestarts' best-rejection reading (1836, +2% over runner-up) is one seed inside
  a 79-point noise band on the metric with 5x leverage — the one arm worth a second
  seed IF any rerun budget appears; not otherwise.
- **Paper note (operator's, agreed):** the paper will state that higher single-run
  readings were observed under other schedulers (OneCycle 0.9417 acc, WarmRestarts
  1836 rej) within the measured seed spread — honest reporting of the bake-off
  without implying a resolved ranking at n=1.

**Step-3 GPU round-trip #1 (2026-08-16): three failures, all localized, none in the
math — exactly what the CPU-gating strategy promised.** (1) Triton's globals rule:
kernels cannot read module globals unless instantiated as `tl.constexpr` — NB/NP are
now constexpr kernel ARGUMENTS at both definitions and launches (AST-checked: no
stray global reads remain). (2) Test-fixture bug, mine: the fixture moved the
ALGEBRA to CUDA before `sparse_gp_tables`, which is CPU-only by design (model init
builds tables on CPU, then buffers move) — the fixture now does what init does and
moves the three tensors. (3) Operator venv: the `pip install kingdon` pulled
ipywidgets-family deps into a venv with a stale widgetsnbextension path and broke —
resolved by REMOVING the install: kingdon is not needed on the cluster at all (the
CUDA tests never import it; the conventions test importorskips). Round-trip #2 needs
no pip at all.

### Scheduler verdict for the GT table at 20 epochs (2026-08-16, theory review)

Question: best schedule for the GT hybrids (GraphTrans + GPS families), 20 epochs,
bs<=512, AdamW. Verdict, with the reasoning recorded in the session log:

- **KEEP LinearWarmup(5%) + Cosine for the campaign, both families.** At a correct
  peak lr every credible alternative (flat+decay a la weaver, WSD/trapezoid, linear-
  to-zero) sits within ~0.03pp on the one controlled comparison available (PNPT
  cosine 0.9414 vs weaver flat+decay 0.9417, inside the 0.0004 seed spread), while
  schedule DIVERGENCE inside the table costs comparability — and the lr incidents
  just measured 0.32pp for lr errors. Schedule shape is second-order; peak lr is
  first-order. Do not differentiate GPS vs GraphTrans: the one real asymmetry
  (GraphTrans's two-stage series sees a shifting transformer-input distribution
  early, arguing for marginally longer warmup) is speculative and far below the
  comparability cost.
- **One knob is genuinely wrong: `cosanneal_eta_min: 0`.** Cosine-to-zero freezes
  the last ~5% of a 47k-step run (~2k steps below 1% of peak). Weaver decays to 1%,
  not 0. Set eta_min ~= 0.01-0.02 x peak — but this changes training arithmetic, so
  it is a CAMPAIGN-BOUNDARY change (next campaign start or a documented table-wide
  re-run), not a mid-stream flip.
- **Peak-lr / schedule coupling** (why schedule flips invalidate transcribed lrs):
  cosine holds peak briefly and tolerates a hot peak; flat+decay holds peak for 70%
  of the budget and wants ~0.75-0.8x the cosine peak. Any schedule change re-opens
  every recipe's lr.
- **Post-campaign A/B if chasing the last 0.05pp**: one model per family, fixed
  finder lr, 3 seeds: cosine vs flat+decay(70/30, exp to 1%) vs linear-to-zero;
  adopt table-wide only on a both-families win. If rows UNDERFIT at 20 epochs (val
  still falling at the end), the lever is epochs 20->30 (the recipe's own comment),
  never schedule exotica.

### The pending ledger, ranked (2026-08-16)

In value order, with owner:

1. **Re-derive queued-hybrid lrs under the fixed finder** (operator, one command per
   row) — direct accuracy recovery; the incident row already measured +0.34pp / +43%
   rej between the wrong and right value.
2. **gp_impl re-race before the tag_cgenn sparse→matmul flip** (operator, one bperf
   row) — blockdiag moved sparse from ~284 to ~297 jets/s against matmul's 304; the
   flip decision predates it and may invert.
3. **SortedGather: segment-sum backward for recv-side gathers** (post-campaign,
   code) — the top remaining kernel on both profiles (22-28%), receivers provably
   sorted, kills the atomics AND the nondeterminism. Biggest single in-repo win left.
4. **GPS host-tax / bucketing project** (post-campaign, code+measure) — 72-95% of
   GPS-family step time is host-side; nothing kernel-side matters for those rows
   until this lands. Blocked on the campaign freeze (theta_h/readout padded-width
   arithmetic), not on knowledge.
5. **Flash port plan** (post-campaign, see next section) — the GP-layer endgame.
6. **Shield retirement test at the NGC container upgrade** (operator+code, one soak)
   — the 2.13 evidence says the stride-guard family is fixed; until then both shields
   stay.
7. **TF32 table-wide protocol decision** (operator policy) — precision is a
   table-wide knob or absent; now with the race's measured example of what TF32 does
   to a single arm (4x → 1.3x).
8. **Upstream inductor issue** (user files): saved views with padded-width-dependent
   strides under dynamic shapes — the round-3 crash minimal case.
9. Scratch-branch deletions (`claude/audit-regressions-cdcc966-jebk1r`,
   `claude/find-lr-transcribe-landing`) — one UI click each; the proxy refuses
   branch deletes.

### Next-upgrade decision: flash-kingdon over flash-clifford for the SO(1,3) port

**Recommendation: invest in the flash-kingdon APPROACH (kingdon codegen), reading
flash-clifford as the scaffolding reference — not a port of either repo as-is.**
Reasoning: nobody ships Cl(1,3), so the Lorentz kernels must be authored either way;
flash-clifford's route means hand-writing ~1k LOC of p1m3 Triton per op family with
hand-derived sign tables (exactly the error class our gates exist to catch), while
kingdon's `Algebra(1,3)` + symbolic compile/CSE generates the per-blade expressions
mechanically — and our primitive stack (MVSiLU→GP→MVLayerNorm, 35-path weighted GP)
differs from their shipped GELU→GP→RMSNorm modules anyway, so codegen-to-OUR-spec is
needed regardless. flash-clifford contributes the things codegen does not: the
(MV_DIM, batch, features) layout argument, launch/fusion structure, and the benchmark
harness shape. Weight-per-grade-triple semantics match our sparse tables, so
checkpoints stay compatible.

**Supersede or compose?** COMPOSE at the program level, SUPERSEDE at the GP layer:
- Superseded where a fused kernel lands: the einsum/matmul/sparse/blockdiag
  contraction ladder and the sparse Function's hand backward — at those call sites
  only. (They were still worth building: they are the eager/CPU reference and
  fallback the flash arm is gated against, and blockdiag is the honest baseline any
  flash speedup must beat.)
- Composes untouched: 2.2a/2.2b (graph aggregation — their ops never touch
  scatter/message-passing), the SortedGather candidate, the shields and the whole
  gate/β-PERF/GPU-gate-day infrastructure (the port is MEASURED BY it), theta_h /
  readout semantics, recipes and the finder.
- Honest ceiling: post-blockdiag, the GP block is ~13-14% of tag_cgenn's CUDA — full
  fusion buys <=~1.15x step there, NOT their headline multiples (their benchmark is a
  pure Clifford-MLP stack; ours is attention/message-passing-heavy). For GPS rows the
  host tax (item 4) gates everything: fusion is invisible behind 72-95% host time.

**The plan (post-campaign, in order):**
- F0 *Legal + conventions.* Ask both authors for licenses (neither repo ships one;
  read-and-learn only until resolved). Pin a kingdon version; machine-check its
  Algebra(1,3) blade order/signs against `CliffordAlgebra` (a conversion-table test —
  OUR cayley is the reference).
- F1 *Codegen spike, CPU-checkable.* Generate the 35-path Cl(1,3) weighted-GP
  forward + grads as pure Python; gate vs `sparse_gp_expression` at fp64 and
  gradcheck BEFORE any Triton exists.
- F2 *One Triton op + the layout measurement.* Fuse MVSiLU→wGP→MVLayerNorm in a
  blade-minor-boundary wrapper; measure blade-major-inside vs blade-minor-inside —
  the marshalling tax is the make-or-break number (Phase 1 killed these copies once).
- F3 *Compile + determinism posture.* torch.library custom op (fake-tensor shapes,
  registered backward) so the 0-break/RECOMP gates hold; no atomics in x/y/w grads,
  deterministic weight-grad reduction — match what 2.2b and the sparse backward
  already bought, or don't ship.
- F4 *`gp_impl=flash` fifth arm.* CUDA-only behind the existing knob with eager
  fallback; GPU-gate-day battery (TOL vs einsum ref, DET, RECOMP, soak) + a
  pinned-precision race vs the post-blockdiag baseline at campaign shapes; adopt only
  on the race discipline's >10%-everywhere bar.
- F5 *Extend or close.* If F4 adopts: fc-GP + MVLinear fusion next, pins re-recorded
  with the class change stated. If not: record the price and close — the scaffolding
  and conversion tests remain as the Cl(1,3) reference implementation.

### FINAL PLAN (2026-08-16): pending-ledger execution, with commands

The ranked ledger above is the WHAT; this is the HOW, in execution order. Items 1-2
are operator commands runnable today; 3-7 are post-campaign code in dependency order.

**1. Re-derive the queued hybrid lrs under the shape-gated finder** (today, before
the queue runs). One sweep per QUEUED row — prune this list to what is actually
queued; the family is:

    git pull origin main
    for M in tag_PlainGraphTrans tag_PlainGraphGPS tag_ParticleNetParTGraphTrans \
             tag_ParticleNetParTGraphGPS tag_CGENNLGATrGraphTrans \
             tag_CGENNLGATrGraphGPS tag_LorentzNetLGATrSlimGraphTrans \
             tag_LorentzNetLGATrSlimGraphGPS; do
      python utils/find_lr.py -cn toptagging model=$M save=false \
          +lr_find.find_batch_size=true
    done
    # transcribe ONLY the TRANSCRIBE/FIND_LR lines; where the banner says
    # "curve-pinned", run that model a SECOND time and take the agreeing bracket
    # (hybrid brackets vary run-to-run; two sweeps is the confirmation).

  The CGENN recipe's lr (5.57e-04 at bs=128) is a finder read from the old rule —
  re-derive at its own batch before the long run:

    python utils/find_lr.py -cn toptagging model=tag_cgenn training.batchsize=128 save=false

**2. gp_impl re-race before executing the tag_cgenn sparse→matmul flip** (one bperf
call — `--models` is a substring filter, so this runs the einsum/matmul/sparse rows
in one go, sized and seeded identically):

    python utils/bperf.py --models tag_cgenn --find-batchsize

  Decision rule: flip to matmul only if it still beats post-blockdiag sparse by more
  than the ~3% single-run noise band; otherwise KEEP sparse (skips fixture churn and
  keeps the eager-retention advantage).

**3. SortedGather (post-campaign, first code item).** Custom autograd Function for
RECV-SIDE gathers only (`x[i_recv]`; senders are unsorted and keep autograd):
forward = index_select (BIT), backward = `torch.segment_reduce(grad, "sum",
lengths=degree)` over the sorted receivers instead of atomic scatter-add.
Class: forward BIT, gradients DET-not-bitwise (deterministic where atomics were not
— state the class change). Gates: grad-vs-autograd TOL at fp64, CUDA determinism
pair-run, RECOMP; adopt on a profile_sync delta (the target kernel is 22-28% of
CUDA on both profiles). Reuses 2.2a's threaded `edge_counts` as lengths — no new
graph inputs.

**4. GPS host-tax / bucketing (post-campaign, the big GPS item).** Order matters:
(a) semantic unfreeze first — make `theta_h` BatchNorm and the tag_cgenn readout
mask-aware so arithmetic no longer depends on padded width (accuracy-affecting:
needs its own short-run validation vs the frozen baseline, stated in the table);
(b) then bucket padded widths to a small set and mark them static; (c) then the
cudagraphs/`reduce-overhead` posture becomes legal. Measure each stage with the
existing instrument:

    python utils/profile_sync.py -cn toptagging model=tag_CGENNLGATrGraphGPS \
        save=false data.dataset=mini training.batchsize=64

**5. Flash port** — next section, F0-F5 with the fusion scope now explicit.

**6. Shield retirement test** — at the NGC container upgrade, nothing sooner:

    CGENN_RECOMPUTE_VIEWS_SHIELD=0 CGENN_COMPILE_GATES=1 CGENN_SMOKE_COMPILE=1 \
    CGENN_SMOKE_STEPS=100 python -m pytest tests/experiments/test_training_smoke.py \
        -k "CGENNLGATrGraphGPS or LorentzNetLGATrSlimGraphGPS"

  Both shields retire together only if 100 varying-shape steps stay green per row.

**7. TF32 policy** (operator, post-campaign): one decision, table-wide — either the
whole table reruns under `float32_matmul_precision=high` or nobody does. The race
measured what per-arm TF32 does to a comparison (a 4x that was really 1.3x); the
same distortion applies to any single-row flip.

### Flash port: fusion scope (what becomes Triton, what stays torch)

The replacement unit is the **phi_x / theta_x SUB-STACK, not the bare GP** — one
generated kernel per stack, matching flash-clifford's own fusion boundary
(act → weighted/fc GP → norm in one launch). Our norms and activations go INSIDE
the kernel; that is where fusion pays, since each is a separate kernel + a full
(E or N, feat, 16) round-trip today:

| becomes ONE Triton kernel | stays torch, why |
|---|---|
| `phi_x` fc stack: FCGP → MVLayerNorm (edge-level, E-sized — Tier A, do first) | message passing: gathers + segment_reduce (2.2b) + SortedGather — scatter is not their op class |
| `theta_x` fc stack: FCGP → MVLayerNorm (node-level — Tier B) | `theta_h` / h-stream MLPs: BatchNorm reduces over the BATCH with running stats, and its padded-width semantics are the frozen quirk — wrong kernel class twice over |
| gpmlp-type stack where configured: MVLinear → MVSiLU → SGP → MVLayerNorm | LGATr attention blocks (already SDPA/fused upstream) |
| MVSiLU + MVLayerNorm as fused epilogues/prologues of the above (never standalone kernels) | readout/invariants (post-Phase-1 `q`/`b` are cheap diagonal ops; readout semantics frozen) |

MVLayerNorm qualifies because its reduction is per-node over (features x 16) — one
Triton program's tile; BatchNorm does not because its reduction axis is the batch.
Tier A alone covers the measured hot spot (the fc-GP pair clone + einsum/blockdiag +
norm on E-sized tensors); Tier B and MVLinear fusion only proceed if Tier A's race
adopts (plan F5).

**References for the port:**
- `flash-clifford` (github.com/maxxxzdn/flash-clifford): `ops/fc_p3m0.py` — the
  fc-GP fused kernel to mirror structurally (layout, launch, fused norm, the
  `tl.atomic_add` weight-grad reduction we must REPLACE with a deterministic one);
  `modules/layer.py` — module boundary; `tests/benchmarks/` — harness shape.
- `flash-kingdon` (github.com/tBuLi/flash-kingdon): README — the full codegen
  recipe (`Algebra(p,q).compile(symbolic=True)` on a weighted-GP expression, grad
  derived by symbolic differentiation, `triton.jit` on the generated function);
  `ops/kingdon_ops.py`, `ops/vga2d.py`/`vga3d.py` — generation scaffolding to adapt
  with `Algebra(1, 3)`.
- `kingdon` (github.com/tBuLi/kingdon): the algebra/codegen engine itself; pin the
  version at F0 and machine-check its Cl(1,3) blade order/signs against OUR
  `CliffordAlgebra` cayley before trusting any generated line.
- Local scouting clones this assessment was read from: `/workspace/maxxxzdn/`
  and `/workspace/tbuli/` (session-ephemeral; re-clone as needed).
- F1 spike bootstrap (CPU-checkable, no cluster needed):

      pip install kingdon sympy
      # scratch: generate p1m3 35-path weighted-GP fwd+grad as pure Python, then
      # gate vs the shipped reference at fp64 before any Triton exists:
      #   from experiments.baselines.cgenn.sparse_gp import sparse_gp_expression

LICENSE GATE stands ahead of all of it: neither flash repo ships a license — ask
both authors (issue or email) before vendoring a line; generated-from-kingdon code
follows kingdon's own license, which is the cleaner path anyway.

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
