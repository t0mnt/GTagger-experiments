#!/bin/bash
# TEMPLATE -- use via:  cp docs/oscar-train.sbatch train.sbatch   (from the repo root)
# Root-level *.sbatch is gitignored, so the partition/account you set in YOUR copy
# (-p / -A below) stays local and cannot be committed to this public repo.
#SBATCH -J gtagger
#SBATCH -p gpu                    # partition; `allq gpu` shows load. gpu-he needs High-End priority
#SBATCH --gres=gpu:1
#SBATCH -n 8                      # cores. jc/xl ship num_workers=2 (the published setting): it
                                  # fed the fastest published model at ~9.3k jets/s, and every
                                  # model here is slower -- but off THEIR filesystem, not yours.
                                  # If nvidia-smi shows the GPU idling, raise -n and
                                  # jc_params.num_workers together, for EVERY row (walltime is
                                  # a reported column).
#SBATCH --mem=48G                 # top-tagging: full npz in RAM + fp64 momenta -> 48G safe.
                                  # JetClass/TopTagXL (streaming, per-worker fetch buffers): 64G.
                                  # After the first run, `myjobinfo` shows MaxRSS -- trim to fit.
#SBATCH -t 24:00:00               # raise for the heavy CGENN-GPS (~a day/trial on a top GPU)
#SBATCH -o logs/%x-%j.out         # logs/ must exist BEFORE sbatch: SLURM opens this file
                                  # before the script runs -- missing dir = job dies, no output.
                                  # %x = job name: submit with `sbatch -J <Model> ...` and the
                                  # log becomes logs/<Model>-<jobid>.out (see the run_dir link below)
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

[ $# -gt 0 ] || { echo "usage: sbatch train.sbatch tag_<Model> [task] [hydra overrides...]
   or: sbatch train.sbatch <any run.py argument list, verbatim>"; exit 1; }
cd "$HOME/GTagger-experiments"

# Accepts both the short form and a run.py command copied verbatim (e.g. from REPRODUCE.md):
#   sbatch train.sbatch tag_PlainGraphGPS jctagging save=false
#   sbatch train.sbatch python run.py -cp config -cn jctagging model=tag_particlenet
CP=config; TASK=""; MODEL=""; RECIPE=""; EXTRA=""
while [ $# -gt 0 ]; do
  case "$1" in
    python|python3|run.py) shift ;;   # so a `python run.py ...` line pastes in verbatim
    -cp|--config-path) CP=$2; shift 2 ;;
    -cn|--config-name) TASK=$2; shift 2 ;;
    model=*)    MODEL=${1#model=}; shift ;;
    training=*) RECIPE=$1; shift ;;
    gpus=*)     [ "${1#gpus=}" = "1" ] || { echo "refusing gpus=${1#gpus=}: this repo's DDP path is not supported -- run one GPU per job"; exit 1; }; shift ;;
    toptagging|jctagging|toptagxl) if [ -z "$TASK" ]; then TASK=$1; else EXTRA="$EXTRA $1"; fi; shift ;;
    tag_*)      if [ -z "$MODEL" ]; then MODEL=$1; else EXTRA="$EXTRA $1"; fi; shift ;;
    *)          EXTRA="$EXTRA $1"; shift ;;
  esac
done
TASK=${TASK:-toptagging}
case "$TASK" in toptagging) P=top ;; jctagging) P=jc ;; toptagxl) P=xl ;; *) echo "unknown task $TASK"; exit 1 ;; esac
[ -n "$MODEL" ] && MODELARG="model=$MODEL" || MODELARG=""
# Derive the training recipe only when one EXISTS: the 8 hybrids ship top_/jc_/xl_ recipes,
# but baselines like tag_MIParT-L do not and must fall back to the task config's default
# (auto-adding training=jc_MIParT-L would fail composition).
if [ -z "$RECIPE" ] && [ -n "$MODEL" ] && [ -f "config/training/${P}_${MODEL#tag_}.yaml" ]; then
  RECIPE="training=${P}_${MODEL#tag_}"
fi

"$APPTAINER" exec --nv "$IMG" bash -c "
  source venv/bin/activate
  python run.py -cp $CP -cn $TASK \
      $MODELARG $RECIPE gpus=1 \
      +slurm_job_id=${SLURM_JOB_ID:-none} $EXTRA
"
# -cp defaults to `config` (run.py itself defaults to the tiny config_quick tree).
# gpus is pinned to 1: multi-GPU DDP is not supported here, so it is refused rather
# than offered. Run one GPU per job -- four such jobs fill a 4-GPU node in parallel.
# +slurm_job_id records the job id INSIDE the run (config.yaml + MLflow params), and the
# job prints its run_dir into the .out -- so either direction of jobid <-> run resolves:
#   grep run_dir logs/<Model>-<jobid>.out          # jobid -> run
#   grep slurm_job_id runs/<exp>/<run>/config.yaml # run   -> jobid
#   sbatch -J PlainGraphGPS train.sbatch tag_PlainGraphGPS         # top-tagging
#   sbatch train.sbatch tag_PlainGraphGPS jctagging                # JetClass
#   sbatch train.sbatch tag_PlainGraphGPS toptagging save=false    # throwaway run
