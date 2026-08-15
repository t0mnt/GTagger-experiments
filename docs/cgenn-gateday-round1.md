# Gate day, round 1 — record (2026-08-15)

Standalone file only because the session's git-push credentials dropped mid-round and
the full cgenn-compile.md is too large for the API fallback; FOLD this section into
docs/cgenn-compile.md (directly above "Workflow: are the gates a fair check") and
delete this file when normal pushes work. Content is final.

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
