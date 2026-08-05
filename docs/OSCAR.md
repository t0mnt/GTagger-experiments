# Training on Brown's Oscar cluster (CCV)

A follow-along recipe for this repo on [Oscar](https://docs.ccv.brown.edu/oscar): start at
SSH, end at a results table. Oscar-specific facts (directories, partitions, `interact`,
module workflow) follow the CCV documentation; the generic SLURM+Apptainer variant lives in
[`SLURM.md`](SLURM.md), and the science workflow (which model, which knobs, seeds, tables)
in [`GUIDE.md`](../GUIDE.md).

## 0. Connect

```bash
ssh your-brown-username@ssh.ccv.brown.edu       # replace your-brown-username; Brown credentials (same as Canvas)
```

You land on a **login node** (`[you@login00X ~]$`). Login nodes are for file management,
editing, installs, and *submitting* jobs only — do **not** run trainings, tests, or
`utils/find_lr.py` on them (heavy processes get killed). Compute happens through `interact`
(interactive session on a compute node) or `sbatch` (batch job).

Every code block in this doc is safe to paste whole, and its first comment says which
shell it belongs in — a block never mixes login-node and compute-node work. The one
fact behind the "LOGIN shell" headers: `interact` must never be launched from inside
another interact. A nested session's controlling shell lives inside the *outer* job,
so it dies when the outer job's walltime expires, whatever its own `-t` says
(observed: a 12 h download killed at its parent's 30-min limit). The check is always
the same — the prompt says `loginXXX` and `echo $SLURM_JOB_ID` prints nothing;
`exit` until it does.

## 1. Know the three directories (this determines where everything goes)

| dir | path | size | properties | use it for |
|---|---|---|---|---|
| home | `~` | 100 GB, per-user | many-small-files optimized, snapshots | **repo clone + venv** |
| data | `~/data/<group>` | ≥256 GB, per-group | big-file reads, backed up, **permanent** | **the dataset + finished runs** |
| scratch | `~/scratch` | 512 GB soft / 12 TB hard | fast big-file I/O, **files unread for 30 days are PURGED** | **live `runs/` output** |

Check your quotas any time with `checkquota`. The CCV-recommended pattern is exactly what
we set up below: read inputs from `~/data`, write outputs to `~/scratch`, copy keepers back
to `~/data` when a run finishes (step 9). Mind the *inode* quota too (a venv is ~50k files — fine in
home, don't put it in data).

> **Scratch purge is per-file by atime** (last read). A 3-seed campaign finishes well inside
> 30 days, but if you pause mid-campaign, `find ~/scratch -atime +25` shows what's at risk —
> copy checkpoints you care about to `~/data` (step 9).
>
> **Tar-extraction footgun (learned the hard way):** `tar`/`tarfile` restores each file's
> *archive* timestamps, so a freshly extracted dataset whose tars were packed years ago
> arrives on scratch already looking years-idle — and the purge daemon deletes it within
> days, leaving only our fresh 0-byte `.extracted` markers behind. `collect_data.py` now
> stamps every extracted file to the current time (`_refresh_times`), so this cannot recur
> **as long as you download with a pulled repo**. If you ever extract a tar onto scratch by
> hand, follow it with `find <dir> -type f -exec touch {} +`.

## 2. One-time setup (on the login node — this part is allowed there)

Python ≥ 3.10 and a CUDA-tuned torch both come from the **NGC PyTorch container module**:
`module load ngc-pytorch-container/25.08-py3-ayk4` is *supposed* to set
`$NGC_PYTORCH_CONTAINER` to the 25.08 image, and every python command in this guide runs
inside it via `apptainer exec --nv "$NGC_PYTORCH_CONTAINER" …`. Bare `python` on Oscar is
the system 3.9 — never use it for this repo (too old, and no torch).

> **Verify the image.** The `25.08-py3-ayk4` modulefile once `setenv`'d
> `$NGC_PYTORCH_CONTAINER` to the **24.03** sif; CCV has since corrected this, and the
> module is expected to resolve a 25.08 image (check yours:
> `module show ngc-pytorch-container/25.08-py3-ayk4 | grep NGC_PYTORCH_CONTAINER`).
> The block below **keeps verifying anyway** — deliberately, not out of distrust of the
> fix: it costs nothing, it is self-checking (it only overrides when the value is wrong),
> and the failure it guards against is silent. A 24.03 image breaks this repo in two
> non-obvious ways: its python has no `ensurepip` (so `python -m venv` fails and pip
> leaks into `~/.local`), and its torch 2.3 is too old for `torch-geometric >= 2.6`
> (`torch.compiler has no attribute is_compiling` at import). If the guard ever fires
> again, the module has regressed — file a CCV ticket.

```bash
# repo + venv live in home
cd ~
git clone https://github.com/t0mnt/GTagger-experiments.git
cd GTagger-experiments

module load ngc-pytorch-container/25.08-py3-ayk4
echo "$NGC_PYTORCH_CONTAINER"        # -> the image the module chose (should now be 25.08)

# Verify rather than trust: if it isn't a 25.08 image, point at the real one directly;
# then hard-stop if we STILL don't have 25.08 (everything below would fail on 24.03).
case "$NGC_PYTORCH_CONTAINER" in
  *25.08*) : ;;   # module happened to be correct -- keep it
  *) export NGC_PYTORCH_CONTAINER="$(ls /oscar/rt/sw/external/ngc-pytorch-container/25.08-py3/*.sif | head -1)" ;;
esac
case "$NGC_PYTORCH_CONTAINER" in
  *25.08*) echo "using $NGC_PYTORCH_CONTAINER" ;;
  *) echo "ERROR: no 25.08 image resolved ($NGC_PYTORCH_CONTAINER); fix the module / path first"; return 1 2>/dev/null || exit 1 ;;
esac

# containers auto-mount only $HOME and /tmp; bind the data/scratch trees so the
# symlinks wired up below keep working inside the container
export APPTAINER_BINDPATH="/oscar/home/$USER,/oscar/scratch/$USER,/oscar/data"

# make both permanent, for new shells and for the sbatch scripts below. Persist the RESOLVED
# sif AFTER the module load so it overrides the module's (possibly wrong) setenv in new shells.
# guarded: skip inside containers (module is an env-imported function there whose lmod
# target isn't bind-mounted -> "environment: line N: .../lmod: No such file" noise on
# every apptainer exec) and silence broken-lmod contexts (post-9.6 upgrade)
# (each line is added only if absent, so re-pasting this block never stacks duplicates;
# the NGC line is replaced rather than appended so a newly resolved sif always wins)
grep -qF 'ngc-pytorch-container/25.08-py3-ayk4 2>/dev/null' ~/.bashrc 2>/dev/null || \
  echo '[ -z "$APPTAINER_CONTAINER" ] && command -v module >/dev/null 2>&1 && module load ngc-pytorch-container/25.08-py3-ayk4 2>/dev/null' >> ~/.bashrc
sed -i '/^export NGC_PYTORCH_CONTAINER=/d' ~/.bashrc
echo "export NGC_PYTORCH_CONTAINER=\"$NGC_PYTORCH_CONTAINER\"" >> ~/.bashrc
grep -qF 'export APPTAINER_BINDPATH=' ~/.bashrc || \
  echo 'export APPTAINER_BINDPATH="/oscar/home/$USER,/oscar/scratch/$USER,/oscar/data"' >> ~/.bashrc
# kill the ~/.local user-site class of bugs forever: no python (container, venv, or system)
# may ever import from ~/.local again (see the leak note below for why this matters)
grep -qF 'export PYTHONNOUSERSITE=1' ~/.bashrc || echo 'export PYTHONNOUSERSITE=1' >> ~/.bashrc

# a venv that INHERITS the container's stack (torch, torch-geometric, numpy, ...),
# created from INSIDE the container; pip adds only what the image lacks.
# Filtered out of requirements.txt before installing:
#   - torch:    keep the container's CUDA-tuned build (a pip torch would clobber it)
#   - xformers: not in the image; skipped by default (opt back in later -- see the
#     "Opting back into xformers" note below, a venv-only change)
#   - the lgatr/lloca [xformers-attention] extras -> plain lgatr/lloca (same reason)
apptainer exec "$NGC_PYTORCH_CONTAINER" bash -lc '
  # fail fast: a missing ensurepip means the WRONG image (24.03) -- do not let pip
  # silently fall back to a ~/.local user-install against the container python.
  python -m venv --system-site-packages venv || { echo "venv creation FAILED (no ensurepip -> not the 25.08 image); aborting"; exit 1; }
  source venv/bin/activate
  pip install --upgrade pip
  pip install -e .
  sed -E -e "/^torch[>=<]/d" -e "/^xformers/d" -e "s/\[xformers-attention\]//" \
      requirements.txt > /tmp/reqs-oscar.txt
  pip install -r /tmp/reqs-oscar.txt
'

# sanity: torch must still be the CONTAINER's own build -- a local +cuXXX / nvXX.XX string
# (e.g. 2.8.0a0+...nv25.08 on the 25.08 image, or 2.3.0a0...nv24.3 on 24.03), NOT a plain
# pip wheel (a bare "2.11.0+cu130"-style version here means a LEAKED pip torch is answering
# -- the ~/.local note below; NGC images ship nv-tagged pre-release builds, not pip wheels);
# torch-geometric should now be >= 2.6. Print torch.__file__ too: it must NOT live under
# ~/.local (see the leak note below).
apptainer exec "$NGC_PYTORCH_CONTAINER" bash -lc '
  source venv/bin/activate
  python -c "import torch, torch_geometric, numpy; \
             print(torch.__file__); \
             print(torch.__version__, torch_geometric.__version__, numpy.__version__)"
'
```

> **Leaked `~/.local` torch — the symptom signature** (bitten once): a warning whose path
> starts `~/.local/lib/python3.12/site-packages/torch/...` saying *"The NVIDIA driver on
> your system is too old"*, plus `torch.cuda.is_available() == False` **on a GPU node with
> `--nv`**, plus trainings printing `Using device cpu`. A pip torch in the user-site (left
> by a failed install era) shadows the container build; the pip cu-wheel lacks NGC's
> driver-forward-compat layer, so CUDA silently dies and runs fall back to CPU. Fix:
> `grep PYTHONPATH ~/.bashrc ~/.bash_profile` (delete any export found — that's how
> user-site beats the venv), then the reversible
> `mv ~/.local/lib/python3.12 ~/.local/lib/python3.12.leaked`, then re-run the sanity
> block above (want a non-`.local` `torch.__file__` and `cuda.is_available() True` with
> `--nv`). If a source-built xformers was compiled while the leak was live, re-verify
> `python -m xformers.info` on a GPU and rebuild it once if it reports undefined symbols.
>
> Three follow-ups that make the fix COMPLETE rather than symptomatic:
> **(1) `export PYTHONNOUSERSITE=1`** everywhere (persisted in `.bashrc` above and set in
> the sbatch templates) — it hard-disables user-site for every python regardless of *how*
> it was being injected, so the class of bug dies even if the mechanism was never found
> and even if a stray `pip --user` happens again.
> **(2) Re-run the §2 venv install block after sidelining.** While the leak was live, pip
> inside the venv saw the `.local` packages as "already installed" and may have SKIPPED
> them — the venv can have holes (requests, awkward, even lgatr pinned-but-old) that only
> surface once `.local` is gone. Re-running `pip install -e .` + the filtered requirements
> is idempotent and fills exactly the gaps.
> **(3) Verify lgatr provenance explicitly**: `python -c "import lgatr;
> print(lgatr.__version__, lgatr.__file__)"` — the leak carried an OLDER lgatr that could
> shadow the venv's pinned one; the printed path must be the venv.

Notes:
- **torch-geometric**: the image ships 2.3.1 but the repo needs ≥ 2.6.0 (its ptr-only
  `MeanAggregation` calls raise `NotImplementedError` on ≤ 2.5 — verified by test suite),
  so pip installs a newer one *into the venv*, shadowing the container's copy. Safe:
  PyG ≥ 2.4 is pure Python and does not pull in a new torch.
- **numpy**: the repo pins < 2.0 (weaver compatibility); if the image carries numpy 2.x,
  pip shadows it in the venv the same way. The sanity line above shows what won.
- **xformers-free is fine**, with one override to remember: all 8 hybrids and most
  baselines never touch xformers (the hybrids' L-GATr stages use dense masks → native
  torch attention, the GPS models plain torch attention, and learned frames use
  `equivectors: equimlp`, already the default; GUIDE §7). Only the four attention
  baselines whose configs pin `attention_backend: xformers` — `tag_transformer`,
  `tag_top_transformer`, `tag_lgatr`, `tag_slim` — would crash on a GPU without it.
  For those, override the backend. **Test flash first, in a CLEAN environment**:
  `apptainer exec "$NGC_PYTORCH_CONTAINER" python -c "import flash_attn; print(flash_attn.__version__)"`.
  (An earlier "image flash-attn is ABI-broken" verdict here turned out to be a
  **leak artifact**: a `~/.local` pip torch 2.11 was shadowing the container's real
  torch 2.8.0a0, and the image's flash-attn — built for 2.8 — naturally failed
  against the impostor. With the leak removed it is expected to import fine.)
  If it genuinely fails clean, `model.attention_backend=flex` is the no-build
  fallback (pure torch, slower; needs torch≥2.7 — the 25.08 image qualifies);
  `varlen` does NOT register on this image (needs torch≥2.10). Either way,
  validate the override with one §3-style quick run on gpu-debug before a real job.
- **Opting back into xformers** (to run `tag_lgatr`/`tag_slim` on their default backend
  instead of the override above). This is a **venv-only change** — no re-clone, no wipe,
  nothing outside the venv is touched; worst case `rm -rf venv` and redo this step.
  (Do NOT reach for `pip install lgatr[xformers-attention]` here: the extra just declares
  a dependency on `xformers`, whose wheel pins its *own* torch — pip would resolve that
  pin by dragging a second torch into the venv over the container build. The extra is
  right on machines where pip owns torch; in the container, `--no-deps` below is the way.)

  **No PyPI wheel can work on this image**: wheels are built for stable pip-torch
  releases, while NGC images ship nv-tagged *pre-release* builds (25.08 = torch
  `2.8.0a0+…nv25.08` — NOT 2.11; if you ever saw 2.11 here, that was the `~/.local`
  leak answering). The working path is a **source build against the container's own
  torch, at the torch-matched TAG** (the container has the full CUDA toolchain;
  ~20–40 min, use an `interact -n 8` session, not the login node):

  - **Pick the tag by the container's torch**: for torch 2.8 → `v0.0.32.post2`
    (`requirements: torch >= 2.8`, verified). Do NOT use `v0.0.35` on this image —
    its *Python* code needs torch ≥2.10 APIs (`GroupName` import) and crashes at
    import regardless of how well the extensions built.
  - Build from **git, not the PyPI sdist**: the sdist does not vendor the CUTLASS /
    flash-attention submodules, so `pip install --no-binary xformers xformers`
    "succeeds" in seconds with a tiny (~3.6 MB, `py39-none`-tagged) wheel containing
    **no compiled kernels at all** (verified). A real build takes ~20–40 min and
    produces a 100+ MB arch-tagged wheel. pip fetches git submodules for VCS
    requirements, so:

  ```bash
  # from a LOGIN shell (prompt loginXXX; echo $SLURM_JOB_ID prints nothing -- §0)
  # the ~20-40 min compile runs in a CPU session
  interact -n 8 -m 32g -t 02:00:00
  ```

  Then, on the compute node:

  ```bash
  # on the COMPUTE node
  apptainer exec --nv "$NGC_PYTORCH_CONTAINER" bash -lc '
    source venv/bin/activate
    pip uninstall -y xformers 2>/dev/null   # clear any mismatched/crippled install first
    export TORCH_CUDA_ARCH_LIST="9.0+PTX"   # H100 (NVL) = sm_90; A100 would be 8.0
    export MAX_JOBS=8
    # DO set this: at this tag the bundled fa2 build is off by default anyway, but fa3
    # defaults ON and, with TORCH_CUDA_ARCH_LIST=9.0 set, would attempt an HOURS-long
    # compile (setup.py, source-verified). Runtime flash comes from the container
    # flash_attn or torch built-in kernels instead (both APIs present since torch 2.7).
    export XFORMERS_DISABLE_FLASH_ATTN=1
    # --no-deps: never let xformers pull its own torch over the container build.
    # --no-build-isolation: build against the CONTAINER torch headers (the whole point).
    # --no-cache-dir: pip cached wheels from earlier builds against the LEAKED torch.
    # git URL: the only source that includes the CUTLASS submodules (see above).
    pip install -v --no-deps --no-build-isolation --no-cache-dir \
        "xformers @ git+https://github.com/facebookresearch/xformers.git@v0.0.32.post2"
    python -m xformers.info | head -25   # want: memory_efficient_attention cutlass ops
                                         # available, torch line = container build
  '
  ```

  Sanity: if the "build" finishes in under a minute, it did not compile anything —
  check the wheel it reports (tiny + `py39-none` = crippled; big + `cp312`-arch = real).
  `exit` back to the login shell when the build is done (the decision-tree commands
  below are light enough to run from anywhere the venv resolves).

  **The flash question, post-build** (source-verified against v0.0.32.post2, which shares
  v0.0.35's gate structure): xformers checks for a flash implementation in order —
  (1) its **own bundled `_C_flashattention`** (compiled from the git submodule during the
  build above; if present, the container's flash_attn is never touched), then
  (2) `importlib.util.find_spec("flash_attn")` followed by an **unguarded**
  `import flash_attn` + a version-bounds check (2.7.1–2.8.2 at this tag; a
  present-but-unimportable or out-of-bounds flash_attn crashes ALL of xformers at
  import — no try/except; stubs don't help, tried), then (3) the PyTorch-native flash
  path (torch's built-in kernels, torch≥2.6ish). Decision tree after the build:
  - `xformers.info` prints cleanly → done, whichever branch engaged.
  - It crashes in `import flash_attn` (branch 2) → neutralize that one branch so the
    next one engages, and re-run info:

    ```bash
    apptainer exec "$NGC_PYTORCH_CONTAINER" bash -lc '
      source venv/bin/activate
      sed -i "s/^elif importlib.util.find_spec(\"flash_attn\"):/elif False:  # Oscar: skip package flash_attn; bundled or PT-flash path instead/" \
          venv/lib/python3.12/site-packages/xformers/ops/fmha/flash.py
      python -m xformers.info | head -30
    '
    ```
  - It raises the version-bounds ImportError instead → `export
    XFORMERS_IGNORE_FLASH_VERSION_CHECK=1` is the sanctioned escape hatch (or the same sed).

  Expected good output: a commit-stamped version (`0.0.32.post2+<sha>`), `fa2F/B`
  **available** (suffix `-pt` = torch-kernel path; no suffix = bundled/package flash),
  `cutlassF/B` available, `build.cuda_version` matching the container, and — **only on a
  GPU node with `--nv`** — `pytorch.cuda: available` (a CPU node prints `not available`
  no matter how healthy the install is; don't diagnose there). `ck*` (ROCm), `fa3`, and
  `*-blackwell` unavailable are benign. Re-apply any sed after a rebuild. Finish with one
  §3-style quick run of `tag_slim` on the GPU before a real job.

Now wire the directories per §1 — dataset into `data`, run output into `scratch`:

```bash
# dataset -> ~/data (permanent, backed up). Set your allocation ONCE (find it: `ls ~/data/`);
# $USER fills in automatically. NB: don't paste raw <angle-bracket> placeholders -- the shell
# reads `<` as a redirect and errors with `bash: ...: No such file or directory`.
GROUP=your-allocation            # <-- replace with your ~/data group dir (from `ls ~/data/`)
mkdir -p ~/data/$GROUP/$USER/gtagger
apptainer exec "$NGC_PYTORCH_CONTAINER" bash -lc \
  'source venv/bin/activate && python data/collect_data.py toptagging'   # ~1.5 GB download (file mgmt: login node OK)
# mv -n, NOT mv: on a re-paste, data/toptagging_full.npz is already the symlink made
# below, and an unguarded mv would OVERWRITE the real npz in ~/data with that symlink
# (destroying the download); -n refuses to clobber the existing target and skips instead
mv -n data/toptagging_full.npz ~/data/$GROUP/$USER/gtagger/
ln -sfn ~/data/$GROUP/$USER/gtagger/toptagging_full.npz data/toptagging_full.npz

# run output -> ~/scratch (fast, purged; we copy keepers back at the end)
mkdir -p ~/scratch/gtagger_runs
ln -sfn ~/scratch/gtagger_runs runs
```

**Dataset placement rule of thumb** — three tiers by size and replaceability: the
tiny `data/*_mini.npz` smoke files ship with the repo and stay in **home**; the
1.5 GB `toptagging_full.npz` lives in **`~/data`** (permanent, backed up, symlinked
above); anything JetClass-sized (§2.1 JetClass, §2.2 TopTagXL — ~190 GB ROOT trees)
goes on **`~/scratch`** behind a symlink, accepting the 30-day-idle purge trade
since a redownload is cheaper than burning the backed-up quota (§1).

### 2.1 JetClass (only if you run the jctagging campaign)

The JetClass dataset is ~190 GB of tars extracting to roughly as much again — too big
for home, and it strains a group's `~/data` quota, so **scratch** is the natural home.
The purge risk manages itself while you train: the streaming loader reads the ROOT
files continuously, refreshing their atime — an *active* campaign is purge-safe, but
files idle > 30 days are at risk, and the window opens at the collector's timestamp
stamp, not the archive dates (see the §1 tar-extraction footgun). If a purge does hit
between download and first training run, the split dirs lose files but the `.extracted`
markers survive and would make a naive re-run skip everything — even a *partial* purge,
where surviving files from other parts hide the gap. The collector defends itself two
ways (post-`git pull`): markers it writes are **manifests** (the tar's file list), so a
skip verifies every file is still on disk and a purged part re-downloads itself
automatically; and a final **file-count summary** (train 1000 / val 50 / test 200
`.root` files) prints "ready" only when the counts are exact. Old 0-byte markers can't
be content-verified — if the summary reports a shortfall, delete the `.*.extracted`
markers for the affected tars and re-run.

The download + extraction takes hours, so it runs in a CPU interact session, never on
the login node. First, the session — nothing else in this block:

```bash
# from a LOGIN shell (§0: prompt says loginXXX / echo $SLURM_JOB_ID prints nothing --
# a nested interact dies at the OUTER job's walltime; a 12 h download once died in 30 min)
interact -n 4 -m 16g -t 12:00:00
```

Wait for the prompt to change to a compute node, then paste the work:

```bash
# on the COMPUTE node the interact above landed you on
mkdir -p ~/scratch/jetclass && ln -sfn ~/scratch/jetclass ~/GTagger-experiments/data/JetClass
cd ~/GTagger-experiments
apptainer exec "$NGC_PYTORCH_CONTAINER" bash -lc \
  'source venv/bin/activate && python data/collect_data.py jetclass' \
  && rm -f ~/scratch/jetclass/*.tar   # && : reclaim the ~190 GB of tars ONLY on a clean
                                   # collector exit -- a failed run keeps them to resume
```

When it finishes, `exit` back to the login shell.

Training swaps the config name and recipe in the *same* §4/§5 commands (science:
GUIDE §5.1 — shared epochs=5, wd=0; fill each `jc_<hybrid>.yaml`'s `???` from the
jctagging sweep, not the top-tagging one):

```bash
# §4 becomes:  python utils/find_lr.py -cn jctagging model=tag_<hybrid> save=false +lr_find.find_batch_size=true
# §5 becomes:  sbatch train.sbatch tag_<hybrid> jctagging     (recipe jc_<hybrid> is derived)
```

### 2.2 TopTagXL (only if you run the toptagxl campaign)

Same scratch treatment as JetClass (it is another ~100M-jet ROOT tree with the same
streaming loader, so the same size and atime/purge reasoning applies). The collector
reads the file list + md5 checksums from Zenodo record 10878355's API at download
time, then verifies and extracts exactly like §2.1:

```bash
# from a LOGIN shell (prompt loginXXX; echo $SLURM_JOB_ID prints nothing -- §0)
interact -n 4 -m 16g -t 12:00:00
```

Wait for the compute-node prompt, then:

```bash
# on the COMPUTE node
mkdir -p ~/scratch/toptagxl && ln -sfn ~/scratch/toptagxl ~/GTagger-experiments/data/toptagxl
cd ~/GTagger-experiments
apptainer exec "$NGC_PYTORCH_CONTAINER" bash -lc \
  'source venv/bin/activate && python data/collect_data.py toptagxl' \
  && rm -f ~/scratch/toptagxl/*.tar   # && : reclaim tar space ONLY on a clean collector exit
```

When it finishes, `exit` back to the login shell.

Commands swap exactly as in §2.1: `-cn toptagxl` + `training=xl_<hybrid>`, with the
`???` knobs filled from a `utils/find_lr.py -cn toptagxl` sweep (science: GUIDE §5.2 —
binary task on JetClass-wide inputs, shared epochs=5, wd=0; the shipped
`data.val_files_range` is the canonical 10M-jet validation split -- keep it,
it costs ~3% of training compute).

## 2.9 Environment certification (run after ANY environment change)

One command certifies the whole stack — interpreter provenance, user-site leak sentinels,
container-torch identity, CUDA, science-stack versions, lgatr provenance, xformers build
quality, and (with `--gpu`) real GPU kernels including a `memory_efficient_attention`
forward **and backward**. It encodes every failure class this doc's warnings came from,
with per-check hints. Run it after: fresh setup, `git pull`, an xformers rebuild, a
cluster upgrade — and always once before a campaign:

```bash
# on the LOGIN node (CPU context: provenance + stack checks)
apptainer exec "$NGC_PYTORCH_CONTAINER" bash -lc \
  'source venv/bin/activate && python utils/env_check.py'
```

The full certification needs a GPU allocation (a §3-style `interact -q gpu-debug -g 1 …`
from a login shell); inside it:

```bash
# on a GPU COMPUTE node (allocated with -g 1)
apptainer exec --nv "$NGC_PYTORCH_CONTAINER" bash -lc \
  'source venv/bin/activate && python utils/env_check.py --gpu'
# exit 0 + "CERTIFIED" = proceed; any FAIL names the fix section
```

> **Certification is per-GPU-architecture.** A source-built xformers compiles for the arch list
> it saw at build time — build on a `gpu-debug` A40 (sm86) and the kernels may simply not exist
> for an H100 (sm90), which surfaces as `no kernel image is available for execution on the
> device` at the first attention call, minutes into a real job. The `--gpu` leg already tests
> this the only way that matters, by running a real kernel forward and backward — but only on
> the card it is running on. **Re-run `--gpu` on the GPU class you will actually train on**
> (condo H100s, not `gpu-debug`) before the campaign; a green A40 run certifies A40s. The same
> allocation is also where throughput and the `find_lr` batch-size finding must be measured —
> an A40 number will size the campaign against the wrong machine.

### What the xformers checks actually establish (and what a missing xformers means)

`import xformers` succeeding says nothing about whether a run can use it, so the tool
separates four states. **The first three run on CPU** — a login-node run already tells you
whether the build is broken; only the kernel forward/backward needs `--gpu`.

| check | catches |
|---|---|
| `xformers real build (sha-stamped)` | the kernel-free sdist build (a bare version with no `+sha`) |
| `xformers._C loads` | the compiled extension failing to link — wrong CUDA runtime or torch ABI (e.g. `libcudart.so.12: cannot open shared object file`). **This is the state where `import xformers` still succeeds** and the run dies in the forward, hours in |
| real (non-fallback) `memory_efficient_attention` kernels | a build whose dispatcher offers only `-pt` PyTorch fallbacks (an expensive alias for `attention_backend=native`), and forward-only builds — the F/B requirement is there because a missing backward passes setup and the first forward, then dies at the first `loss.backward()` |
| lgatr / lloca registered the `xformers` backend | the decisive one: both libraries populate a backend registry at import, and an unusable xformers is silently omitted from it. Any config pinning `attention_backend: xformers` then crashes **in the forward, not at init** |

`env_check.py` is the automated form of §2.2's flash decision tree and reads the output the
same way, deliberately: **`-pt` is a code path, not a quality mark** (`fa2F@2.5.7-pt` is genuine
FlashAttention-2 reached through torch's bindings), and unavailable `ck*` / `fa3*` / `*-blackwell`
entries are benign. An earlier revision of the check treated `-pt` as a fallback and flagged a
healthy A40 build; if the two ever disagree again, §2.2 is the authority and the check is the bug.
The extension-load probe reads `xformers._cpp_lib._cpp_library_load_exception` — not an
`import xformers._C`, which raises `PyInit__C` on a *healthy* build because `_C` is a torch
library rather than a Python module.

**A missing xformers is reported as INFO, not FAIL** — the image this doc builds is
xformers-free on purpose (§2.2), and only `tag_transformer`, `tag_top_transformer`,
`tag_lgatr`, `tag_slim` pin it. If it is absent, override those four with
`model.attention_backend=native|flex` and the certification still passes.

When xformers *is* present and usable, you do not choose a kernel: lgatr calls
`memory_efficient_attention(q, k, v, **kwargs)` with no `op=`, so xformers' own dispatcher
picks among cutlass / flash / triton / ck per call from the shapes, dtype and device.

## 3. Smoke-test on a compute node

Never on the login node — grab a short interactive CPU session for the tests, then a
GPU-debug session for the model smoke:

**CPU leg** — the invariance/equivariance suites (~6 min). Session first:

```bash
# from a LOGIN shell (prompt loginXXX; echo $SLURM_JOB_ID prints nothing -- §0)
interact -n 4 -m 16g -t 00:30:00
```

On the compute node:

```bash
# on the CPU COMPUTE node (no --nv on a CPU node)
cd ~/GTagger-experiments
apptainer exec "$NGC_PYTORCH_CONTAINER" bash -lc \
  'source venv/bin/activate && pytest tests/experiments/test_tag_equivariance.py tests/experiments/test_tag_invariance.py -q'
```

Then `exit` — you must be back on `loginXXX` before the GPU leg (§0 rule 1).

**GPU leg** — one tiny training end-to-end (gpu-debug = short wait, short cap).
The `-g 1` is NOT optional: without it SLURM allocates zero GPUs and its cgroup HIDES
the node's cards — `torch.cuda.is_available()` is False even on a GPU node, with no
error anywhere. "cuda False on a GPU node" = check your allocation before anything else.

```bash
# from a LOGIN shell (prompt loginXXX; echo $SLURM_JOB_ID prints nothing -- §0)
interact -q gpu-debug -g 1 -n 4 -m 20g -t 00:30:00
```

On the GPU node:

```bash
# on the GPU COMPUTE node
cd ~/GTagger-experiments
nvidia-smi                                   # confirm you see a GPU (host side)
apptainer exec --nv "$NGC_PYTORCH_CONTAINER" bash -lc '
  source venv/bin/activate
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  python run.py -cp config_quick -cn toptagging model=tag_LorentzNetLGATrSlimGraphGPS save=false gpus=1
'
```

Then `exit` back to the login shell.

If `torch.cuda.is_available()` is `False` inside the container, the usual causes are a
missing `--nv` flag or not actually being on a GPU node (`nvidia-smi` on the host settles
which) — the torch build itself comes from the NGC image and is known-good.

## 4. Find batch size + LR per model (GPU interactive)

**First, sanity-check the finder itself on a published baseline.** ParticleNet has a
known recipe (`lr: 1e-2`, `batchsize: 512`), so running the finder against it at that
fixed batch size tells you whether the tool lands in the right neighbourhood before you
trust it for eight unpublished models. Order-of-magnitude agreement is the pass
criterion -- the finder reports loss-min/10, not the authors' tuned value:

```bash
# on the GPU COMPUTE node -- fixed batch size (no find_batch_size), published recipe
cd ~/GTagger-experiments
apptainer exec --nv "$NGC_PYTORCH_CONTAINER" bash -lc '
  source venv/bin/activate
  python utils/find_lr.py -cn toptagging model=tag_particlenet training=top_particlenet save=false
'
# compare the printed lr against config/training/top_particlenet.yaml (1e-2)
```

Then, one session per model you plan to train (or chain them in one longer session).
**`+lr_find.find_batch_size=true` is what makes it search for a batch size** — without
that flag the finder uses whatever `training.batchsize` the recipe already holds and
reports only an lr (which is exactly what the ParticleNet check above wants, since that
recipe pins 512). For your own models, both numbers are `???`, so you need the flag:

```bash
# from a LOGIN shell (prompt loginXXX; echo $SLURM_JOB_ID prints nothing -- §0)
# add -f <feature> to pin a GPU type; `nodes gpu` lists them
interact -q gpu -g 1 -n 8 -m 48g -t 02:00:00
```

On the GPU node:

```bash
# on the GPU COMPUTE node
cd ~/GTagger-experiments
apptainer exec --nv "$NGC_PYTORCH_CONTAINER" bash -lc '
  source venv/bin/activate
  python utils/find_lr.py -cn toptagging model=tag_LorentzNetLGATrSlimGraphGPS \
      save=false +lr_find.find_batch_size=true
'
#  ->  reuse with:  training.batchsize=<N> training.lr=<lr>
```

(`exit` when done — or stay in the session and chain the next model's sweep.)

Fill each printed pair into that model's `config/training/top_<Model>.yaml` (they are the
only `???` keys — the shared recipe pins epochs=20, AdamW, warmup-cosine; GUIDE §5–6).

## 5. Submit the real training

**Before the first real submission, reproduce a known result.** Train ParticleNet under
its published recipe with `save=false` (no weights, no table row) and check its test
accuracy/AUC against the published numbers. This exercises the whole path -- data,
loader, model, evaluation, table row -- against an answer you already know, so a
mismatch here is an environment or data problem, not a hypothesis about your hybrids:

```bash
# on the LOGIN node -- a full run, so submit it rather than sitting in an interact.
# Runtime depends on your cluster and dataset; the job prints its own estimate early on.
sbatch -J particlenet-check train.sbatch tag_particlenet toptagging save=false
# when it finishes: grep "table test" logs/particlenet-check-<jobid>.out
```

One parametrized sbatch file covers every model and task. It also accepts a `run.py`
command copied verbatim (from `REPRODUCE.md`, say) with `sbatch train.sbatch` in front —
`python run.py`, `-cp`, `-cn` and `model=` are all understood:

```bash
# -J <Model> names the log logs/<Model>-<jobid>.out (see "matching jobs to runs" below)
sbatch -J PlainGraphGPS train.sbatch tag_PlainGraphGPS     # top-tagging, that model's recipe
sbatch train.sbatch tag_CGENNLGATrGraphGPS save=false      # throwaway (no weights, no table row)
sbatch train.sbatch tag_PlainGraphGPS warm_start_idx=0 warm_start_load=false   # fresh-trial seed (§6)
```

One-time before the first submission (SLURM opens the `-o` log file BEFORE your script
runs — a missing `logs/` dir kills the job with no output at all, so no `mkdir` inside
the script can save you):

```bash
mkdir -p ~/GTagger-experiments/logs
```

The script is the repo's `docs/oscar-train.sbatch` template — one-time, copy it to the
working file (root-level `*.sbatch` is gitignored, so the copy — and any partition or
account you add to it — physically cannot be committed to this public repo):

```bash
# on the LOGIN node
cd ~/GTagger-experiments
cp docs/oscar-train.sbatch train.sbatch
mkdir -p logs                    # SLURM opens logs/%x-%j.out BEFORE the script runs
```

**What you fill in.** The template opens with a marked `FILL THESE IN` block — three
lines, and only the first is mandatory:

| line | when you need it |
|---|---|
| `-p <partition>` | always. `condos` shows a group condo if you have one — prefer it: newer cards than the general `gpu` pool (`nodes gpu` lists what that pool actually holds), and you queue only against your own group |
| `-A <account>` | only if your partition requires one (condo/priority accounts usually do) |
| `--mail-user` + `--mail-type=FAIL,TIME_LIMIT` | optional, recommended for multi-day runs: you get an email when a job fails or is killed at its walltime, instead of discovering it days later |

Everything else resolves at run time — the container path is absolute, `apptainer` is
looked up with a self-diagnosing guard, and model/task/overrides come from the command line. The shipped defaults (`-p gpu`, no account)
submit to Oscar's general GPU partition and work unedited. 

The template's comments explain every flag choice; the load-bearing facts:

- `--mem=48G` fits top-tagging (full npz in RAM + fp64 momenta); the streaming
  JetClass/TopTagXL runs want `64G`. After a first run, `myjobinfo` shows MaxRSS — trim to fit.
- `-t 24:00:00` default; raise for the heavy CGENN-GPS (~a day/trial on a top GPU).
- `--export=NONE` + absolute image/apptainer paths: batch jobs must NOT inherit the login
  env (stale lmod `module` function) nor `module load` the mislabeled container module —
  the §2 container-guard story, solved inside the script.
- Recipe names are derived (`tag_X` + task → `top_X`/`jc_X`/`xl_X`), and `-cp config` is
  pinned there because `run.py` defaults to the tiny `config_quick` tree.
- Arg 2 is the task only if it names one (`toptagging|jctagging|toptagxl`); anything else
  is passed through as a hydra override, as in the examples above.

```bash
# on the LOGIN node -- submit + monitor
squeue -u $USER           # ALWAYS check before sbatch: a "failed" job may still be alive
                          # (three concurrent downloads once raced this way), and startup
                          # noise in the .out is not proof of death -- sacct is
sbatch train.sbatch tag_PlainGraphGPS     # (example; see the submissions above)
myq                       # your queue; `squeue -u $USER -t PENDING --start` estimates start time
myjobinfo                 # time/memory actually used after a job finishes
# with a real job id (from myq):
#   tail -f logs/gtagger-<jobid>.out    -- live log (or runs/<exp>/<run>/out_0.log once training starts)
#   scancel <jobid>                     -- kill it
```

> **Condo partition facts** (if your group has one: `condos` names it; submit with
> `-p <group>-gcondo` and read the live limits via `scontrol show partition <name>` +
> `scontrol show node <its-node>`. The facts below are from one H100 condo and are
> typical, but check yours):
> - **`-t` is MANDATORY here: `DefaultTime=00:05:00`** — an untimed submission is killed
>   after five minutes. `MaxTime=UNLIMITED`, so be generous (`-t 48:00:00`).
> - One node, `Gres=gpu:nvidia_h100_nvl:4`, `OverSubscribe=NO`: **four whole 94 GB H100
>   NVLs, exclusively allocated** — up to 4 concurrent group jobs, never sharing a GPU,
>   so max-VRAM batch sizes cannot collide with a groupmate's job.
> - 1.54 TB RAM / 128 CPUs on the node (≈385 G / 32 CPUs per-GPU fair share): 48–64G
>   `--mem` stays polite; raising `--cpus-per-task` toward 16 helps the streaming
>   JetClass/XL loaders (more workers).
> - `State=IDLE+POWERED_DOWN`: the node powers off when idle — the first job after a
>   quiet spell sits in `CF` (configuring) for a few minutes while it boots. Normal,
>   not stuck.
>
> **The `environment: line N: .../lmod: No such file` noise — mechanism + fix.** Oscar's
> lmod exports `module` as a *shell function*; apptainer passes exported functions into
> the container, where your `.bashrc`'s `module load` lines call it — but the function's
> lmod path isn't bind-mounted inside, so it errors (bash labels env-imported-function
> errors `environment:`, one message per module line). **Harmless** — the container never
> needs `module`; `NGC_PYTORCH_CONTAINER` is exported directly. **Remediation**: guard
> the `.bashrc` module line as §2 now writes it
> (`[ -z "$APPTAINER_CONTAINER" ] && command -v module >/dev/null 2>&1 && module load … 2>/dev/null`)
> — silences containers AND the post-9.6 broken-lmod host contexts in one line. In batch,
> `--export=NONE` + absolute `IMG`/`APPTAINER` paths sidestep it entirely (that header
> applies to ANY batch job here, dataset downloads included).

**Matching a job id to its run.** The template records the job id inside the run
(`+slurm_job_id`, so it lands in `config.yaml` and the MLflow params) and SLURM's log
carries the model name when you submit with `-J <Model>`. Both directions resolve:

```bash
# on the LOGIN node -- substitute your model name / job id / run path
grep run_dir logs/PlainGraphGPS-4519312.out                        # job id -> run directory
grep slurm_job_id runs/topt_local_debug/PlainGraphGPS_1234/config.yaml  # run -> job id
myjobinfo 4519312                                                  # time/memory actually used
```

Each finished run prints its `table test: … \\` row into the log (GUIDE §4).

## 6. Seeds (3 trials → mean ± std)

After trial 1 finishes, submit the same run twice more as **fresh-trial warm starts**
(never plain warm starts — those reload the trained model and its finished scheduler;
GUIDE §8). This needs a different `-cp` (the run dir, not `config`), which the
parametrized `train.sbatch` can't express as an override — so make a one-off copy
(`cp train.sbatch seed.sbatch`) and replace its `python run.py` line with:

```bash
python run.py -cp ~/GTagger-experiments/runs/EXPNAME/RUNNAME -cn config \
    warm_start_idx=PREV_RUN_IDX warm_start_load=false   # substitute EXPNAME/RUNNAME/PREV_RUN_IDX
```

(`run_idx` is 0 for the first run, 1 after the first warm start, …; the saved `config.yaml`
in the run dir carries everything else.) The run's table row consolidates to
`[N trials] $mean ± std$` automatically.

## 7. The full campaign (which models, and which need the LR finder)

The study's grid is the 8 hybrids. **All 8 need §4** (their recipes deliberately leave
`batchsize`/`lr` as `???`); everything else in their shared recipe is already decided:

Inside a §4-style GPU session (its `interact` block first, from a login shell), one
sweep per model — fill each `top_<Model>.yaml` as they print:

```bash
# on the GPU COMPUTE node (§4's interact; raise its -t to cover all 8 sweeps)
MODELS="tag_PlainGraphTrans tag_PlainGraphGPS \
        tag_ParticleNetParTGraphTrans tag_ParticleNetParTGraphGPS \
        tag_CGENNLGATrGraphTrans tag_CGENNLGATrGraphGPS \
        tag_LorentzNetLGATrSlimGraphTrans tag_LorentzNetLGATrSlimGraphGPS"
cd ~/GTagger-experiments
for M in $MODELS; do
  apptainer exec --nv "$NGC_PYTORCH_CONTAINER" bash -lc \
    "source venv/bin/activate && python utils/find_lr.py -cn toptagging model=$M save=false +lr_find.find_batch_size=true"
done
```

Then `exit`, and submit one sbatch per model (§5) plus 2 more fresh-trial seeds each (§6).

Once the recipes are filled, shake down the config axes before (or alongside) the seed
runs, in this order — PlainGraphGPS PE/SE variants first (`model.net.use_edge_attr`,
`model.net.use_rwse`, `model.net.norm=batch|layer` — confirm each trains), then every
model under both graph metrics (`model.net.knn_metric=deltaR|minkowski`), then the
LLoCa models (Plain / ParticleNet-ParT) under PD frames (`model/framesnet=learnedpd`).
See GUIDE §6's shakedown note for the reasoning.

The **baseline reference rows** (`tag_ParT`, `tag_particlenet`, `tag_lgatr`, `tag_slim`,
`tag_lorentznet`, `tag_transformer`, …) do **not** need the LR finder — they run under
their published recipes, which already pin lr/batchsize/budget:

```bash
# the same §5 train.sbatch works as-is (recipe top_<name> is derived from tag_<name>):
sbatch train.sbatch tag_ParT
# likewise tag_lgatr, tag_slim, tag_lorentznet, tag_particlenet, ...
# if you SKIPPED the §2 xformers build, the four attention baselines need a backend
# override (flash if the image's flash-attn imports clean, else flex -- see the §2
# note; validate on gpu-debug first):
sbatch train.sbatch tag_lgatr model.attention_backend=flash
# (same for tag_slim, tag_transformer, tag_top_transformer; all other rows run as-is)
```

(Heads-up on wall time: order the queue submissions cheapest-first; `CGENNLGATrGraphGPS`
is the expensive one — budget ~a day per trial on a top GPU — while the slim models are
orders of magnitude lighter.)

## 8. The comparison table

```bash
apptainer exec "$NGC_PYTORCH_CONTAINER" bash -lc \
  'source venv/bin/activate && python utils/aggregate_table.py --runs runs --split test --out comparison.tex'
```

## 9. Save what matters (scratch purges!)

```bash
# finished runs you want to keep -> data (permanent, backed up). Set GROUP as in step 2
# (`ls ~/data/`) and replace EXPNAME with your run's exp_name; $USER fills in.
GROUP=your-allocation
cp -r ~/scratch/gtagger_runs/EXPNAME ~/data/$GROUP/$USER/gtagger/runs_keep/
```

Do this at the end of the campaign (and for any long pause > ~3 weeks). `comparison.tex`,
the `table_metrics_*.json` files, `out_*.log`, and the best-model checkpoints are the
irreplaceable parts.

## Quick reference

| task | command |
|---|---|
| run anything in the container | `apptainer exec --nv "$NGC_PYTORCH_CONTAINER" bash -lc 'source venv/bin/activate && <cmd>'` |
| my jobs / all GPU jobs | `myq` / `allq gpu` |
| GPU types available | `nodes gpu` |
| quotas | `checkquota` |
| condo limits (if any) | `condos` |
| interactive CPU / GPU | `interact -n 4 -m 16g -t 01:00:00` / `interact -q gpu -g 1` — from a login shell only (§0) |
| am I on a login shell? | `echo $SLURM_JOB_ID` — must print nothing before any `interact` |
| scratch purge check | `find ~/scratch -atime +25` |

Alternatives to raw SSH that CCV supports, if you prefer them: Open OnDemand (browser
terminal + Jupyter at the CCV portal) and VS Code Remote-SSH (docs: "Remote IDE").
