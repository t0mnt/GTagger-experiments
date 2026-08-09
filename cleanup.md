# Post-merge cleanup list

What to wipe (and when) after PR #18 merges. Rule of thumb: nothing here is deleted
before the merge — reviewers need the evidence.

**Operator ruling (2026-08-09) on the compile/BIT machinery**: the gate files and
fixture packs were created to prove the ports (lgatr 2.0, compile twins, local ParT);
the ports succeeded, so they are DELETED post-merge rather than kept as permanent
regression guards. Recorded tradeoff: deleting them forfeits provable-safe refactors of
the model files (a future refactor re-records fresh fixtures first, at whatever HEAD it
starts from — the record-before-edit recipe in each gate file's docstring). Sequencing
constraint: the todo §4b-ter CGENN dedup port consumes `test_cgenn_hybrid_compile.py` +
its fixtures — land that FIRST, then delete.

## After Gates G/H close on the cluster

- `tests/experiments/test_lgatr_migration_parity.py` — the **v1-transplant half only**
  (the lgatr-1.4.4 state-dict transplant tests and their code paths). Once 2.0 is the
  only environment the transplant can never run again. KEEP the production-manifest and
  config-snapshot-diff halves — they still catch parameter and config drift on every run.
- `tests/fixtures/lgatr144/*.pt` weight packs (large; the transplant tests' inputs).
  KEEP `production_manifests*.json`, `content_hashes.json`, and the snapshot yamls that
  the surviving gate halves read.
- `docs/execution-playbook.md` — session-operational instructions, historical after merge.

## Any time after merge (regeneratable artifacts)

- `tests/fixtures/*/dynamo_explain*.txt` — large committed explain reports. Every
  committed one belongs to a model in a compile-gate parametrization and is rewritten
  by its BREAKS gate on each gated run (`CGENN_COMPILE_GATES=1`), so the committed
  copies are convenience diffs, not sources of truth. Trimming them to the header +
  break-count summary (or deleting outright) loses nothing the gates can't reproduce
  in minutes. (The one artifact this did not hold for — MIParT's, descoped from the
  gates — was deleted in the final audit; its 7-break census lives in the Stage-4 log.)

## After the merge, in this order (operator ruling — port instruments, ports succeeded)

1. Land the todo §4b-ter CGENN dedup port (it needs `test_cgenn_hybrid_compile.py` +
   `tests/fixtures/cgenn_hybrid_compile/` to be provable).
2. Run β-PERF (`scripts/bperf.py`, below) and flip whatever knobs it decides.
3. Then delete the compile gate files (`test_cgenn_compile.py`,
   `test_lorentznet_compile.py`, `test_cgenn_hybrid_compile.py`,
   `test_lorentznet_hybrid_compile.py`, `test_nonequi_compile.py`) and ALL
   `tests/fixtures/*_compile/` packs (~43 MB, incl. the MIParT pins and the two large
   ParT packs) plus their `dynamo_explain_*.txt` artifacts.
4. `scripts/bperf.py` itself: delete once its numbers are recorded in
   `docs/cgenn-compile.md`'s log and the knob flips are committed — or stow the output
   table in the log entry as the justification record (operator's choice).

## Keep permanently (explicitly NOT cleanup)

- The decision logs: `docs/lgatr2-migration.md`, `docs/cgenn-compile.md`,
  `docs/audit-ledger.md`, `docs/diffs.md` — the paper's methods section and any future
  migration lean on these (β-PERF numbers land in the cgenn-compile log before the
  script is deleted).
- `tests/internal/` (weight-decay grouping etc.) and every equivariance/invariance/flops
  test — table-integrity guards, not port instruments.
- The production-manifest and config-snapshot halves of `test_lgatr_migration_parity.py`
  — running drift guards, not port instruments (the transplant half still goes after
  Gates G/H, as below).

## Candidate follow-ups recorded elsewhere (not cleanup)

- ~~Known 15 pelican-FLOPs environment failures~~ RESOLVED (2026-08-09): they were a
  misdiagnosed harness gap (unforced nested compile knobs in the FLOPs tests), not an
  environment class; all 15 pass since the recursive eager-forcing fix. Expected suite
  state: zero failures.
- lloca 2.0 ("dev 2") scope: see docs/cgenn-compile.md (preserve_variance S-row note).
