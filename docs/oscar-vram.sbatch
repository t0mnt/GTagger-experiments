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
#SBATCH -t 06:00:00               # 34 cases (17 rows x 2 states). Each pays ~90 s of dataset
                                  # load plus a compile build for the compiled state; the
                                  # CGENN rows dominate. --dataset mini cuts the load ~30x if
                                  # you only want the RATIO (its P_max is 135 vs full's 160).
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
