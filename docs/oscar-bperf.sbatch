#!/bin/bash
# TEMPLATE -- use via:  cp docs/oscar-bperf.sbatch bperf.sbatch   (from the repo root)
# Root-level *.sbatch is gitignored, so the partition/account you set in YOUR copy
# (-p / -A below) stays local and cannot be committed to this public repo.
#
# beta-PERF, batched. The interactive route (`interact` + a shell) is fine for the fast
# rows, but a CGENN sweep is not fast: --find-batchsize does an OOM search per state (a
# doubling ladder of real forward+backward steps, each with its own compile) and then two
# timed runs per row, and there are three gp_impl rows. Reserve hours, not minutes -- and
# an interactive session that drops takes the sweep with it, while this does not.
#
# Reads its arguments verbatim as utils/bperf.py flags, so anything the driver accepts works:
#   sbatch -J bperf-cgenn bperf.sbatch --models tag_cgenn --find-batchsize --iters 1010 --window 100 1000
#   sbatch bperf.sbatch --models CGENNLGATr --find-batchsize --iters 110 --window 10 100
#   sbatch bperf.sbatch                                     # the whole matrix (very long)
#
#SBATCH -J bperf
# ---- FILL THESE IN -- the only lines in this file that need your values -----------
#SBATCH -p gpu                    # YOUR PARTITION (`condos` shows a group condo if you have one)
# #SBATCH -A your-account         # uncomment + set if your partition requires an account
# #SBATCH --mail-user=you@example.edu
# #SBATCH --mail-type=FAIL,TIME_LIMIT
# ---- end fill-in -----------------------------------------------------------------
#SBATCH --gres=gpu:1              # ONE. beta-PERF compares two states of the SAME row
                                  # back-to-back on the SAME card; splitting them across
                                  # devices would put hardware variance inside the ratio,
                                  # which is the one thing the pairing exists to remove.
#SBATCH -n 8                      # match your training jobs: the dataloader is part of the
                                  # it/s being measured, so a different -n measures a
                                  # different number than the campaign will see.
#SBATCH --mem=64G                 # above the training template's 48G on purpose: --find-batchsize
                                  # walks the batch ladder UP until the GPU OOMs, and the host-side
                                  # collate for the last surviving batch scales with it.
#SBATCH -t 8:00:00                # 3 gp_impl rows x (OOM search + 2 timed runs at --iters 1010),
                                  # each run paying its own compile warm-up. 4h is usually enough
                                  # and 8h costs nothing if unused -- SLURM bills what you consume.
                                  # For the --iters 110 screen, 2h is plenty.
#SBATCH -o logs/%x-%j.out         # logs/ must exist BEFORE sbatch, or the job dies with no output
#SBATCH --export=NONE             # do NOT inherit the login env (stale lmod `module` function)

IMG=/oscar/rt/sw/external/ngc-pytorch-container/25.08-py3/pytorch-25.08-py3.sif
APPTAINER=$(command -v apptainer || true)
[ -x "$APPTAINER" ] || { echo "apptainer not on the batch PATH: hardcode APPTAINER= with the absolute path from \`readlink -f \$(command -v apptainer)\` on a login node"; exit 1; }
export APPTAINER_BINDPATH="/oscar/home/$USER,/oscar/scratch/$USER,/oscar/data"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
# TRITON_LIBCUDA_PATH and TORCHINDUCTOR_CACHE_DIR live in venv/bin/activate (docs/OSCAR.md
# §2), which the `source` below runs INSIDE the container -- one copy, covering batch jobs,
# interact sessions and find_lr. A host-side export would not cross the apptainer boundary
# (measured on gpu2703), and a second copy here would be a second thing to drift.

cd "$HOME/GTagger-experiments"
[ $# -gt 0 ] || set -- --models tag_cgenn --find-batchsize --iters 1010 --window 100 1000
case " $* " in *" --apply "*)
  echo "refusing --apply from a batch job: it EDITS production yamls, and the point of"
  echo "reading the table first is to decide whether the flip is one you want. Re-run the"
  echo "driver on the login node with --apply once you have read the numbers." >&2
  exit 1 ;;
esac

OUT="bperf-${SLURM_JOB_ID:-local}.md"
"$APPTAINER" exec --nv "$IMG" bash -c "
  source venv/bin/activate
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  python utils/bperf.py $* | tee '$OUT'
"
echo "table written to $OUT (and to the top of logs/\$SLURM_JOB_NAME-\$SLURM_JOB_ID.out)"

# WHY --find-batchsize IS NOT OPTIONAL FOR THE CGENN ROWS. Without it every row uses the
# yaml batchsize, which is the unswept 512 fallback -- and the last CGENN sweep is exactly
# where that misled: einsum and matmul both fit 128 (peak 80.2 GB) while sparse peaked at
# 15.2 GB by batch 16 and OOM'd at 128, so it was sized to 32 where the others got 64. The
# it/s columns were then not comparable and had to be read as jets/s. That 1.50x memory
# ratio is what the sparse-GP autograd Function (experiments/baselines/cgenn/sparse_gp.py)
# was built to remove -- retained activations dropped 3.46x eager, 5.6x compiled, measured
# on CPU -- so THIS run is the one that says whether the ranking actually inverted on the
# card. Read the sizes the OOM search chooses before you read the speedups.
#
# WHICH WINDOWS ARE POSSIBLE: base_experiment logs a timing line only at iterations
# 1, 10, 100, 1000, 10000, 100000, so both --window bounds must be powers-of-ten marks
# <= --iters. `--window 100 300` cannot work at any --iters; check_window rejects it in a
# second rather than after the runs.
#   cheap screen:    --iters 110  --window 10 100     (90 steps)
#   decision-grade:  --iters 1010 --window 100 1000   (900 steps, all past the thermal ramp)
#
# WHAT THE PAIRING DOES NOT CANCEL: the two states run back-to-back in a fixed order, so
# the second always starts on a warmer card. That is a bias, not noise. If a row lands
# inside the 3% margin, re-run it with the states swapped before believing the sign.
