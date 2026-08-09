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

## RUNBOOK — merge to campaign, in order

Copy-paste sequence. Steps 1–3 are the go/no-go; step 4 is the wipe; step 5 is the run.

    # 1. MERGE (squash or merge commit, your preference)
    #    ...merge PR #18 on GitHub...
    git checkout main && git pull
    #    OPTIONAL: `git tag pre-lgatr2 <pre-merge main sha> && git push --tags`.
    #    Convenience only -- main's history already preserves the pre-2.0 state (it is the
    #    merge commit's first parent), and every run archives its own source zip +
    #    config.yaml. The tag just spares you resolving a sha later when the paper needs to
    #    say which code produced the pre-migration top-tagging rows. Skip it if you prefer.

    # 2. SMOKE TEST — the go/no-go. 8 real optimizer steps per model, production configs.
    CGENN_COMPILE_GATES=1 python -m pytest tests/experiments/test_training_smoke.py -q -s
    #    Expect: 16 passed, every model "nonzero-grad params 100%" (>=50% is the bar).
    #    A failure here means a severed backward -- do not start the campaign.

    # 3. CONFIRM the rest still holds on main
    python -m pytest tests/ -q                            # expect 0 failures
    python utils/bperf.py                                 # optional: knob decisions
    #    paste bperf's table into docs/cgenn-compile.md, commit any knob flips

    # 4. WIPE the vestigial bits (everything below), commit once
    # 5. RUN the campaign

Nothing in step 4 is required before step 5 -- deleting is hygiene, not a prerequisite.
The CGENN dedup that used to gate this is DONE (landed pre-merge, BIT bit-identical), so
there is no longer any ordering constraint on the fixture deletion.

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

- §4b-septies training smoke — BUILT (`tests/experiments/test_training_smoke.py`), it is
  step 2 of the runbook above.
- ~~§4b-ter dedup port~~ DONE pre-merge: the hybrid now imports the CGENN machinery from
  `experiments/baselines/cgenn/*` instead of duplicating it (1354 -> 728 lines). It had
  already drifted twice — the `b()` device fix and the `_as_int_grades` coercion each
  reached only one copy — so the import is the correct end state. BIT bit-identical,
  full CGENN battery 26/26.
- §4b-quater mask-aware pair BatchNorm — **REQUIRED** (operator ruling): compile must be
  numerically faithful to the reference, not merely fast. Spec + acceptance criterion in
  todo.md; `test_train_mode_differential` is the executable gate.
- Pre-existing `main`-side defects (finetune weight_decay, the EMA/dtype family incl. the
  fp64 restore truncation) are NOT in `todo.md` — out of this branch's scope. Diagnoses
  live in `docs/audit-ledger.md` for whenever someone wants them.
- ~~15 known pelican-FLOPs failures~~ RESOLVED 2026-08-09: a misdiagnosed harness gap
  (unforced nested compile knobs), not an environment class. Expected suite state is now
  zero failures.
