#!/bin/bash
#SBATCH -J gtagger
#SBATCH -p gpu                    # partition; `allq gpu` shows load. gpu-he needs High-End priority
#SBATCH --gres=gpu:1
#SBATCH -n 8
#SBATCH --mem=48G                 # top-tagging: full npz in RAM + fp64 momenta -> 48G safe.
                                  # JetClass/TopTagXL (streaming, per-worker fetch buffers): 64G.
                                  # After the first run, `myjobinfo` shows MaxRSS -- trim to fit.
#SBATCH -t 24:00:00               # raise for the heavy CGENN-GPS (~a day/trial on a top GPU)
#SBATCH -o logs/%x-%j.out         # logs/ must exist BEFORE sbatch: SLURM opens this file
                                  # before the script runs -- missing dir = job dies, no output
#SBATCH --export=NONE             # do NOT inherit the login env (post-9.6 it carries a stale
                                  # lmod `module` function that errors on compute nodes)
# #SBATCH -A <account>            # only if you belong to a condo/priority account (see `condos`)
# #SBATCH -C ampere               # optionally pin a GPU architecture/feature (--constraint)

# Do NOT `module load ngc-pytorch-container/...` here: that module setenv's the MISLABELED
# 24.03 sif over your resolved path (the docs/OSCAR.md container-guard story) -- a batch job
# would silently train on the wrong torch. Use the resolved sif and apptainer binary directly.
IMG=/oscar/rt/sw/external/ngc-pytorch-container/25.08-py3/pytorch-25.08-py3.sif
APPTAINER=$(command -v apptainer || true)   # or hardcode the login-node `readlink -f` result here
[ -x "$APPTAINER" ] || { echo "apptainer not on the batch PATH: hardcode APPTAINER= with the absolute path from \`readlink -f \$(command -v apptainer)\` on a login node"; exit 1; }
export APPTAINER_BINDPATH="/oscar/home/$USER,/oscar/scratch/$USER,/oscar/data"
export PYTHONUNBUFFERED=1         # live logs (batch jobs have no tty -> python buffers)
export PYTHONNOUSERSITE=1         # never import from ~/.local (the docs/OSCAR.md leak class);
                                  # --export=NONE strips the .bashrc copy, so set it here too

MODEL=${1:?usage: sbatch train.sbatch tag_<Model> [task] [extra hydra overrides...]}
shift
# arg 2 is the task ONLY if it names one -- otherwise it's already a hydra override
# (so `sbatch train.sbatch tag_X save=false` means toptagging + save=false, not task save=false)
case "${1:-}" in
  toptagging|jctagging|toptagxl) TASK=$1; shift ;;
  *) TASK=toptagging ;;
esac
case "$TASK" in toptagging) P=top ;; jctagging) P=jc ;; toptagxl) P=xl ;; esac
# remaining args = hydra overrides, passed through via $*
cd "$HOME/GTagger-experiments"
"$APPTAINER" exec --nv "$IMG" bash -c "
  source venv/bin/activate
  python run.py -cp config -cn $TASK \
      model=$MODEL training=${P}_${MODEL#tag_} gpus=1 $*
"
# (no data.dataset override needed: config/toptagging.yaml already defaults to the
#  full dataset, and the jc/xl configs have no such key -- it would crash there)
# -cp config is REQUIRED: run.py defaults to the tiny config_quick tree, which has
# no top_<Model> recipes. Recipe names are derived (tag_X + task -> top_X/jc_X/xl_X).
#   sbatch train.sbatch tag_PlainGraphGPS                          # top-tagging
#   sbatch train.sbatch tag_PlainGraphGPS jctagging                # JetClass
#   sbatch train.sbatch tag_PlainGraphGPS toptagging save=false    # throwaway run
