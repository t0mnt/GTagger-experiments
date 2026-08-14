#!/bin/bash
# TEMPLATE -- use via:  cp docs/oscar-vram.sbatch vram.sbatch   (from the repo root)
# Root-level *.sbatch is gitignored, so the partition/account you set in YOUR copy
# (-p / -A below) stays local and cannot be committed to this public repo.
#SBATCH -J vram
# ---- FILL THESE IN -- the only lines in this file that need your values -----------
#SBATCH -p gpu                    # YOUR PARTITION (same value as your train.sbatch)
# #SBATCH -A your-account         # uncomment + set if your partition requires an account
# ---- end fill-in -----------------------------------------------------------------
#SBATCH --gres=gpu:1
#SBATCH -n 8                      # no dataloader pressure here: this measures ONE step per
                                  # case, it does not train
#SBATCH --mem=128G                # the full top-tagging npz is loaded once PER CASE
#SBATCH -t 03:00:00               # DERIVED, from the 2026-08-13 beta-PERF log on gpu2703 --
                                  # not a round safe number. Per case: ~20 s interpreter +
                                  # torch import, ~91 s dataset load (that log's six loads:
                                  # 94.6/90.3/91.5/90.9/92.7/90.7 s), ~10 s build, and 5
                                  # steps at bs=64 (~1-4 s/step from the probe rungs;
                                  # retention and peak are separate steps -- see measure()).
                                  # Compiled adds an inductor build, backed out of that log
                                  # by subtracting the measured it/s from each row's wall
                                  # clock: ~90 s for tag_cgenn, ~180 s for the GraphTrans
                                  # hybrid. So ~2.2 min eager + ~4.7 min compiled = ~7 min
                                  # per row; 17 rows ~= 2 h, the 8 hybrids alone ~= 1 h.
                                  # 3 h is that plus headroom, NOT a guess.
                                  # Over-asking is NOT free here, whatever train.sbatch says:
                                  # SLURM bills what you consume, but backfill favours short
                                  # requests, so a fat -t buys queue time on a busy partition.
                                  # 40% of the full-dataset runtime is the SAME npz loaded 34
                                  # times -- one subprocess per case, to keep dynamo cache
                                  # entries from holding each model alive into the next
                                  # measurement. --dataset mini cuts that to seconds (P_max
                                  # 135 vs full's 160) and the RATIO is the deliverable; use
                                  # full only to confirm a row the ratio flags.
#SBATCH -o logs/%x-%j.out         # logs/ must exist BEFORE sbatch: SLURM opens this file
                                  # before the script runs -- missing dir = job dies silently
#SBATCH --export=NONE             # do NOT inherit the login env (post-9.6 it carries a stale
                                  # lmod `module` function that errors on compute nodes)

# Do NOT `module load ngc-pytorch-container/...` here: that module setenv's the MISLABELED
# 24.03 sif over your resolved path (the docs/OSCAR.md container-guard story) -- a batch job
# would silently measure the wrong torch. Use the resolved sif and apptainer binary directly.
IMG=/oscar/rt/sw/external/ngc-pytorch-container/25.08-py3/pytorch-25.08-py3.sif
APPTAINER=$(command -v apptainer || true)   # or hardcode the login-node `readlink -f` result
[ -x "$APPTAINER" ] || { echo "apptainer not on the batch PATH: hardcode APPTAINER= with the absolute path from \`readlink -f \$(command -v apptainer)\` on a login node"; exit 1; }
export APPTAINER_BINDPATH="/oscar/home/$USER,/oscar/scratch/$USER,/oscar/data"
export PYTHONUNBUFFERED=1         # live logs (batch jobs have no tty -> python buffers)
export PYTHONNOUSERSITE=1         # never import from ~/.local (the docs/OSCAR.md leak class)
# TRITON_LIBCUDA_PATH lives in venv/bin/activate (docs/OSCAR.md §2) and is REQUIRED here:
# every compiled case builds CUDA kernels through Triton, which cannot find libcuda.so
# inside the NGC image. A host-side export would not reach the container anyway.

cd "$HOME/GTagger-experiments"

# Any argument list is passed through verbatim, so:
#   sbatch vram.sbatch                              # all 17 rows, bs=64, full dataset
#   sbatch vram.sbatch --models CGENN               # just the two CGENN hybrids
#   sbatch vram.sbatch --dataset mini --bs 128      # fast pass at a bigger batch
"$APPTAINER" exec --nv "$IMG" bash -c "
  source venv/bin/activate
  python utils/vram_compile_matrix.py $*
"
# Read the RATIO column. A row that reads 'MORE compiled' is a row whose bperf batchsize
# must not be carried into a recipe, and a candidate for `activation_memory_budget`.
# The absolute numbers are bs- and P_max-dependent; the ratio is the portable part.
