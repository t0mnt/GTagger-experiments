# CGENN torch.compile support — workflow

**Status: PLANNED — no work done. Companion to `docs/lgatr2-migration.md` (same record→change→prove discipline), scoped to the regular CGENN baseline.**

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

The actual geometric-product math (`mul` + `bmm` ≈ 46%) is what the sparse-GP rewrite below
targets. The two are independent and multiply.

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
precomputed once at init. The geometric product is `mul` + `bmm` ≈ 46% of CGENN's runtime, so a 5×
there is ~1.6–1.8× on the whole model, before compile and before sparse-GP.

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
| β-PERF | cluster: it/s eager vs compiled (quick config + full-size batch), fp32/bf16-off | numbers published in this doc's log whatever they say — compile is only worth shipping if this table says so |

Fixtures are trivial compared to the lgatr migration: eager outputs on a fixed seeded batch (fp32+fp64) + sha256, recorded **at current HEAD before any edit** — no state_dict, no transplant, no env swap; the whole Stage 1 fits in one session.

## 4. Task split, prompts, operator gates

> **Execution is driven from `docs/execution-playbook.md`** (steps C-α/C-β there, plus the LorentzNet Stage-2 and non-equivariant extensions this document's policy section calls for). It carries the canonical prompt copies and the exact operator check commands. On divergence, sync in a dedicated commit before running.

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
environment class (migration decision log), nothing CGENN-related. β-PERF (cluster it/s)
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
| tag_lgatr × {eager, compiled} | `+model.net.compile={false,true}` (+`compile_kwargs.dynamic=true`) (2) |
| CGENNLGATrGraphTrans / GPS × {eager, compiled} (gp_impl stays sparse) | `model.compile={false,true}` (4) |

Decisions the table makes: every `compile:` default; the campaign `gp_impl` (sparse is the
lgatr-2.0-posture default — GPU numbers confirm or flip to matmul in one line; only
`sparse` changes the FLOPs column and needs the row footnote); whether full-LGATr compile
ships (Gate H per H11). **Then:** Gates G/H from the migration runbook (cluster).

---

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
