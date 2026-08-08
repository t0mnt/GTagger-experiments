# Post-merge cleanup list

What to wipe (and when) after PR #18 merges. Rule of thumb: nothing here is deleted
before the merge — reviewers need the evidence — and the compile/BIT machinery is
**permanent**, not cleanup.

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

- `tests/fixtures/*/dynamo_explain*.txt` — large committed explain reports. Each is
  rewritten by its BREAKS gate on every gated run (`CGENN_COMPILE_GATES=1`), so the
  committed copies are convenience diffs, not sources of truth. Trimming them to the
  header + break-count summary (or deleting outright) loses nothing the gates can't
  reproduce in minutes.

## Keep permanently (explicitly NOT cleanup)

- All compile gate files (`test_cgenn_compile.py`, `test_lorentznet_compile.py`,
  `test_cgenn_hybrid_compile.py`, `test_lorentznet_hybrid_compile.py`,
  `test_nonequi_compile.py`) and their `tests/fixtures/*_compile/` BIT fixture packs —
  these are the regression guards that make any future refactor of the model code
  provable-safe (record-before-edit + `torch.equal`). This includes the MIParT fixtures
  (compile was descoped for it, but the BIT/hash pins still guard its eager path) and
  the two large ParT packs (`tag_ParT_fp64.pt` ≈ 17 MB — the price of pinning a 2M-param
  model bit-exactly; do not trim).
- The decision logs: `docs/lgatr2-migration.md`, `docs/cgenn-compile.md`,
  `docs/audit-ledger.md`, `docs/diffs.md` — the paper's methods section and any future
  migration lean on these.
- `tests/internal/` (weight-decay grouping etc.) and every equivariance/invariance/flops
  test — table-integrity guards, not scaffolding.

## Candidate follow-ups recorded elsewhere (not cleanup)

- Known 15 pelican-FLOPs environment failures: documented in the migration decision log;
  they close by environment fix, not deletion.
- lloca 2.0 ("dev 2") scope: see docs/cgenn-compile.md (preserve_variance S-row note).
