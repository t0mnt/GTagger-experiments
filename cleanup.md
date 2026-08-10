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

Steps 1–2 are setup, 3–6 are the go/no-go, 7 is the wipe, 8 is the campaign. **The
campaign instructions themselves are unchanged** — step 8 hands off to `docs/OSCAR.md`
exactly as it stands. Everything before it is what this branch adds to the front.

Steps 3–4 are CPU and can run anywhere, including a laptop. Steps 5–6 need a GPU, and step
5 is the one that cannot be skipped on the grounds that "the CPU gates were green" — see
why below.

    # 1. MERGE (squash or merge commit, your preference)
    #    ...merge PR #18 on GitHub...
    git checkout main && git pull
    #    OPTIONAL: `git tag pre-lgatr2 <pre-merge main sha> && git push --tags`.
    #    Convenience only -- main's history already preserves the pre-2.0 state (it is the
    #    merge commit's first parent), and every run archives its own source zip +
    #    config.yaml. The tag just spares you resolving a sha later when the paper needs to
    #    say which code produced the pre-migration top-tagging rows. Skip it if you prefer.

    # 2. ENVIRONMENT UPGRADE -- lgatr 1.4.4 -> 2.0. NOT backwards compatible in either
    #    direction: post-merge code will not run on a 1.4.4 env, and the pre-merge code
    #    will not run on 2.0. Upgrade the env in the same step as the merge, not before.
    #
    #    Local / any plain venv:
    pip install -r requirements.txt        # lgatr[xformers-attention]>=2.0.0, lloca>=1.3.6
    #
    #    OSCAR: NO .sif rebuild is needed -- lgatr/lloca are plain pip installs into the
    #    venv, and v2 DROPPED the einops/opt_einsum/numpy requirements (torch only) --
    #    but einops is now pinned DIRECTLY in requirements.txt, because lloca imports it
    #    without declaring it and was relying on lgatr 1.4.x to supply it. Do not remove
    #    it. Re-run the venv install block of docs/OSCAR.md §2 verbatim, inside the
    #    container, with the same requirements filtering (it strips torch/xformers and the
    #    [xformers-attention] extras so the container's CUDA-tuned torch is not clobbered).
    #
    #    Then VERIFY rather than trust -- both versions, and that torch is still the
    #    container's own nv-tagged build, not a leaked pip wheel:
    python -c "import importlib.metadata as m, torch; \
        print('lgatr', m.version('lgatr'), '| lloca', m.version('lloca')); \
        print('torch', torch.__version__, torch.__file__)"
    #    want: lgatr >= 2.0.0, lloca >= 1.3.6, and on OSCAR a torch path NOT under ~/.local
    #    (the leaked-user-site failure mode is documented in docs/OSCAR.md).
    #
    #    The executable check that the env really is v2. NOT test_lgatr_v2_inventory.py --
    #    that is a SIDE-LOAD probe for planning the migration from a 1.4.4 box (it needs
    #    LGATR2_WHEEL_DIR pointing at an unpacked wheel) and can only ever report
    #    "1 skipped" post-upgrade, which reads like a pass and verifies nothing.
    #    The real gate is the parity file: it loads the recorded 1.4.4 activations and
    #    requires v2 to reproduce them, so it fails loudly on a wrong or half-upgraded env.
    python -m pytest tests/experiments/test_lgatr_migration_parity.py -q   # expect 23 passed
    #    Plus the full stack certification (provenance, user-site leak, backend binding):
    apptainer exec "$NGC_PYTORCH_CONTAINER" bash -lc \
        'source venv/bin/activate && python utils/env_check.py'

    # 3. SMOKE TEST -- the go/no-go. 8 real optimizer steps per model, PRODUCTION configs.
    CGENN_COMPILE_GATES=1 python -m pytest tests/experiments/test_training_smoke.py -q -s
    #    Expect: 16 passed, every model "nonzero-grad params 100%" (>=50% is the bar).
    #    A failure here means a severed backward -- do not start the campaign.
    #    Runs EAGER by design (see the file's docstring); step 5 covers the compiled paths.

    # 4. CONFIRM the rest still holds on main (CPU)
    python -m pytest tests/ -q                            # expect 0 failures
    CGENN_COMPILE_GATES=1 TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
        python -m pytest tests/ -q                        # + the full gate battery
    #    TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 is NOT optional for a compile gate: inductor
    #    caches compiled graphs on disk (~GB under /tmp/torchinductor_*), and a graph that
    #    compiled successfully earlier will be served from cache and MASK a real lowering
    #    failure. This was observed during the round-7d investigation.

    # 5. GPU TEST -- the one genuinely new regime. EVERY compile gate in this repo runs on
    #    CPU, so inductor's CUDA backend (Triton kernels) has never been exercised by
    #    anything here, and device-placement bugs are invisible to CPU runs by
    #    construction. Two parts, on a GPU node:
    #
    #    (a) the smoke test again -- it auto-selects CUDA (base_experiment._init_backend)
    #        and _extract_batch moves each batch to the device, so no flags are needed.
    #        This is the device-hygiene proof that test_device_hygiene.py can only
    #        approximate statically.
    CGENN_COMPILE_GATES=1 python -m pytest tests/experiments/test_training_smoke.py -q -s
    #
    #    (b) the same test with each model's SHIPPED compile knob, a few models at a time
    #        (all at once builds inductor kernels for 12 models in one process and can
    #        exhaust the box -- that is why the default forces eager):
    CGENN_COMPILE_GATES=1 CGENN_SMOKE_COMPILE=1 python -m pytest \
        tests/experiments/test_training_smoke.py -q -s -k "tag_ParT or tag_cgenn"
    #    ...repeat over the 12 models that ship compile: true. Step 6 (beta-PERF) also runs
    #    every knob-bearing model compiled on GPU through run.py, so if you are doing
    #    beta-PERF anyway, (b) is belt-and-braces rather than a separate obligation.

    # 6. beta-PERF -- decides the remaining knobs. Needs a GPU to be decision-grade
    #    (the script says so in its header line if CUDA is absent).
    #    Cheap screen over the whole matrix (defaults: --iters 110 --window 10 100):
    python utils/bperf.py --find-batchsize
    #    Decision-grade on whatever the screen left inside the margin -- 900 timed steps,
    #    all of them past the thermal ramp:
    python utils/bperf.py --models PlainGraphGPS PNParTGraphGPS \
        --find-batchsize --iters 1010 --window 100 1000
    #    The window bounds must BOTH be marks the run actually logs -- {1, 10, 100, 1000,
    #    ...} -- because validation is deliberately pushed past the end of the run so it
    #    cannot pollute the timing, and those powers of ten are then the only timing lines
    #    emitted. bperf checks this up front now; a bad window used to surface only after
    #    the whole matrix had run.
    #    --find-batchsize sizes each row EAGER with find_lr's OOM doubling search and uses
    #    that size for both states. Without it the rows run at whatever the yaml carries,
    #    which is the unswept 512 fallback for every model whose recipe still says '???'.
    #    Shells out to run.py with save=false, so it writes no run directories.
    #    Decides: the two remaining GPS compile knobs (tag_PlainGraphGPS,
    #    tag_ParticleNetParTGraphGPS -- both ship false purely on unmeasured performance
    #    grounds; correctness is settled for both) and the campaign gp_impl.
    #    --apply may now flip ANY row: NO_APPLY is empty, because every knob-bearing model
    #    compiles and survives a real backward. (tag_PlainGraphTrans was fixed by the
    #    static-k kNN twin; tag_LorentzNetLGATrSlimGraphGPS by a scoped recompute_views --
    #    both this session, both gated.) So beta-PERF is now a pure speed decision.
    #    Paste the table into docs/cgenn-compile.md and commit any flips.

    # 7. WIPE the vestigial bits (everything below), commit once. AFTER beta-PERF, not
    #    before: step 6 needs utils/bperf.py, and steps 3-5 need the gate files that
    #    step 7 deletes. Deleting is hygiene, so it can also be skipped or deferred --
    #    but it must never move EARLIER in this list.
    #    NOTE this file is itself on the delete list. The two things in it you might
    #    still want afterwards live elsewhere on purpose: the GPU procedure (5a/5b) is
    #    in tests/experiments/test_training_smoke.py's own header, which is KEEP-
    #    permanently, and the campaign is docs/OSCAR.md.
    # 8. RUN the campaign -- docs/OSCAR.md, UNCHANGED. Nothing in this branch alters how
    #    the campaign is launched, only what it runs.

Nothing in step 7 is required before step 8 -- deleting is hygiene, not a prerequisite.
The CGENN dedup that used to gate this is DONE (landed pre-merge, BIT bit-identical), so
there is no longer any ordering constraint on the fixture deletion.

---

## DELETE: session scaffolding (no reason to outlive review)

- `docs/execution-playbook.md` (278 lines) — session-operational instructions: prompt
  copies and per-step operator commands. Historical the moment the branch lands.
- `utils/bperf.py` — one-shot instrument. Delete once its table is pasted into
  `docs/cgenn-compile.md` and the flips are committed. (Stow the *output table*, not the
  script.)
- `bperf_results.md` — its append-log. NOT tracked (it is gitignored, since bperf
  appends to it on every run), so there is nothing to `git rm`: paste the table into
  `docs/cgenn-compile.md` and delete the local file.
- `tests/fixtures/*/dynamo_explain*.txt` — large committed explain reports; every one is
  rewritten by its BREAKS gate on each gated run, so they are convenience diffs, not
  sources of truth. Go with the gate files below.

## DELETE: port instruments, after step 1–3 above

- All five compile gate files: `test_cgenn_compile.py`, `test_cgenn_hybrid_compile.py`,
  `test_lorentznet_compile.py`, `test_lorentznet_hybrid_compile.py`,
  `test_nonequi_compile.py`.
  **The regression guards inside `test_nonequi_compile.py` have already been carved out**
  into `tests/experiments/test_compile_posture.py` (KEEP). Port instruments and permanent
  guards had ended up in the same file, so deleting it wholesale would have stripped the
  drift protection off the compile TWINS — which are permanent model code, not port
  scaffolding. Verified: with all five gate files and every fixture pack removed,
  `test_compile_posture.py` still passes 8/8.
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

## DONE 2026-08-10: the first half of step 7

The port instruments and their fixture packs are **deleted** (43 MB): the five gate files,
`tests/fixtures/{cgenn,cgenn_hybrid,lorentznet,lorentznet_hybrid,nonequi}_compile/`, and
`docs/execution-playbook.md`. The 14 shipped citations below were repointed in the same
commit, at `docs/cgenn-compile.md`'s Log (per-stage numbers, including the 11/7 break
counts) and at `test_compile_posture.py` (the standing posture guards).

**Still present, on purpose** — each is needed by a step that has not run yet:
`utils/bperf.py` (step 6), `tests/experiments/test_lgatr_v2_inventory.py` (step 2's env
check), `tests/fixtures/lgatr144/` + the transplant half of the parity test (Gates G/H are
still open), and this file. Delete them after step 6.

Step 4's full gate battery can no longer run — the gate files are gone. That is the
accepted cost of taking the deletion early: the battery's last verdict (760 passed /
37 skipped / 0 failed) was recorded on the tree this merge produced, and `main`'s tree is
byte-identical to the `dev` tip that was measured, so there is nothing left for it to tell
you. `python -m pytest tests/ -q` still runs and still has to be green.

## Before you wipe: the citations that outlive their target

Deleting the gate files does not only remove code — it invalidates the **references to
them**. Four production yamls (`tag_ParT`, `tag_PlainGraphGPS`,
`tag_ParticleNetParTGraphTrans`, `tag_ParticleNetParTGraphGPS`) plus
`experiments/tagging/wrappers.py` and
`experiments/baselines/CGENNLGATrGraphTransHybrid.py` cite the gate files as the AUTHORITY
for numbers they ship: the pinned break bars, "7 pinned breaks", the twin-flag rationale.
14 citations over those six files. After the wipe each one points at nothing, and the
reason a knob is set the way it is becomes unrecoverable from the shipped tree.

This is now enforced rather than remembered:
`test_compile_posture.test_no_shipped_file_cites_a_missing_test_module` fails the moment a
cited module disappears and prints the exact worklist. Repoint each comment at a surviving
home — `docs/cgenn-compile.md` keeps the per-stage numbers, the posture rules live in
`test_compile_posture.py` — **in the same commit as the deletion**. The surviving test
files are deliberately exempt: their references are historical ("carved out of…", "used to
import from…"), and that stays true after the target is gone.

## Before you wipe: the dependency that used to break it

`test_device_hygiene.py` is KEEP-permanently but used to `import` REPO/_fixed_batch/_build
from `test_cgenn_compile.py` and `test_nonequi_compile.py`, both of which the wipe deletes
— so step 7 would have turned the one surviving device test into a collection ERROR. Those
three helpers are now inlined there, and the wipe was rehearsed: with all five gate files
and every fixture pack removed, `pytest tests/ --collect-only` reports **706 tests
collected**, no errors. If you add a new keep-file, check it the same way before trusting
the wipe.

## KEEP permanently (earns its place)

- `tests/experiments/test_device_hygiene.py` — small, fast, and guards a class that CPU
  runs are structurally blind to. Not a port instrument; it stays useful forever.
- `tests/experiments/test_compile_posture.py` — the two guards that must outlive the port:
  `test_compile_true_is_backward_verified` (ungated, so it runs on every suite; it is what
  stands between a one-character yaml edit and a campaign that dies at its first optimizer
  step) and `test_train_mode_differential` (the guard that caught the eval-exact /
  train-wrong pair-BN bug). Deliberately fixture-free, so the fixture wipe cannot
  neuter it.
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
- ~~§4b-quater mask-aware pair BatchNorm~~ DONE pre-merge (operator ruling: compile must
  be numerically faithful to the reference, not merely fast). The twins now weight the
  pair-BN statistics by the eager reference multiset: train-mode delta 2.3e-15 / 3.1e-15 /
  2.8e-17 and BN buffers 1.1e-14 / 5.6e-13 / 5.6e-13, from 6.5e-01 / 1.5e-02 / 4.4e-04 and
  15.0 / 1.1 / 1.1. `tag_ParT` and `tag_ParticleNetParTGraphTrans` returned to
  `compile: true`; the GPS pair stays false on the β-PERF performance rule alone.
- Pre-existing `main`-side defects (finetune weight_decay, the EMA/dtype family incl. the
  fp64 restore truncation) are NOT in `todo.md` — out of this branch's scope. Diagnoses
  live in `docs/audit-ledger.md` for whenever someone wants them.
- ~~15 known pelican-FLOPs failures~~ RESOLVED 2026-08-09: a misdiagnosed harness gap
  (unforced nested compile knobs), not an environment class. Expected suite state is now
  zero failures.
