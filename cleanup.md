# Post-merge cleanup list

**Scope check, stated plainly.** This branch was meant to be two things: the lgatr 2.0
migration, and compile support across the model table. It also grew ~1,930 lines of new
decision-log prose, 8 new test files, and 57 MB of fixtures. Some of that was load-bearing
(the compile fixes were only provable because fixtures were recorded before each edit, and
four real bugs were caught by gates that did not previously exist). Some of it was
session scaffolding that has no reason to outlive review. This file separates the two and
schedules the second for deletion.

Nothing here is deleted before the merge — reviewers need the evidence.

**Operator ruling (2026-08-09) on the compile/BIT machinery**: the gate files and fixture
packs were built to prove the ports; the ports succeeded, so they go. Recorded tradeoff:
deleting them forfeits provable-safe refactors of the model files — a future refactor
must re-record fresh fixtures first, using the record-before-edit recipe in each gate
file's docstring.

---

## Order matters — do these in sequence

1. **todo §4b-ter CGENN dedup port.** Consumes `test_cgenn_hybrid_compile.py` and its
   fixtures; it is the last thing that needs them.
2. **todo §4b-septies training smoke + checkpoint round-trip**, paired with the
   §4b-sexies fixes. This is the one regime with zero coverage.
3. **β-PERF** (`utils/bperf.py`), then commit whatever knob flips it decides.
4. **Then** the deletions below.

---

## DELETE: session scaffolding (no reason to outlive review)

- `docs/execution-playbook.md` (278 lines) — session-operational instructions: prompt
  copies and per-step operator commands. Historical the moment the branch lands.
- `utils/bperf.py` — one-shot instrument. Delete once its table is pasted into
  `docs/cgenn-compile.md` and the flips are committed. (Stow the *output table*, not the
  script.)
- `bperf_results.md` — its append-log, same fate.
- `tests/fixtures/*/dynamo_explain*.txt` — large committed explain reports; every one is
  rewritten by its BREAKS gate on each gated run, so they are convenience diffs, not
  sources of truth. Go with the gate files below.

## DELETE: port instruments, after step 1–3 above

- All five compile gate files: `test_cgenn_compile.py`, `test_cgenn_hybrid_compile.py`,
  `test_lorentznet_compile.py`, `test_lorentznet_hybrid_compile.py`,
  `test_nonequi_compile.py`.
- All `tests/fixtures/*_compile/` packs (~43 MB, including the MIParT pins and the two
  large ParT packs).
- `tests/experiments/test_lgatr_v2_inventory.py` — a v1-vs-v2 API inventory. Its job ends
  when 2.0 is the only environment anyone runs.

## DELETE: after Gates G/H close on the cluster

- The **v1-transplant half** of `test_lgatr_migration_parity.py` and
  `tests/fixtures/lgatr144/*.pt` (14 MB). Once 2.0 is the only environment the transplant
  can never run again.
  KEEP the production-manifest and config-snapshot-diff halves plus
  `production_manifests*.json` / `content_hashes.json` — those are running drift guards,
  not port instruments.

## COMPRESS rather than delete

- `docs/lgatr2-migration.md` (851 lines) and `docs/cgenn-compile.md` (799 lines) are
  decision logs written for review, at review density. After the campaign, cut each to
  the parts the paper's methods section actually cites — the migration's behavioral
  choices and gate results; the compile program's posture table, the train-mode finding,
  and the β-PERF numbers. Everything narrating *how the session got there* can go.
  Target: one document, a few hundred lines, not two of ~800.
- `todo.md` (446 lines) — the §4b-* chain is a work queue, not documentation. Each entry
  is deleted as it is completed, not archived.

## KEEP permanently (earns its place)

- `tests/experiments/test_device_hygiene.py` — small, fast, and guards a class that CPU
  runs are structurally blind to. Not a port instrument; it stays useful forever.
- `tests/internal/` and every equivariance/invariance/FLOPs test — table-integrity guards
  that predate this branch's concerns.
- `docs/diffs.md` — the one summary a future reader needs, and the natural home for
  whatever survives the compression above.
- `docs/audit-ledger.md` — findings ledger, referenced by the methods notes.

## Not cleanup — tracked in todo.md

- §4b-ter dedup port · §4b-quater mask-aware pair BatchNorm (the path to compiled ParT) ·
  §4b-quinquies + §4b-sexies pre-existing main-side bugs · §4b-septies training smoke.
- ~~15 known pelican-FLOPs failures~~ RESOLVED 2026-08-09: a misdiagnosed harness gap
  (unforced nested compile knobs), not an environment class. Expected suite state is now
  zero failures.
