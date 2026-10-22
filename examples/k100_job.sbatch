#!/bin/bash -l
# Edit the partition/GRES/module names to match your K100 installation.
# Supply your account with: sbatch --account=YOUR_ACCOUNT ...
#SBATCH --job-name=mace-ictc
#SBATCH --partition=DCU_K100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=dcu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=ictc-%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${ICTC_ENV:?Export ICTC_ENV as the path to your vendor-PyTorch Python environment}"
if [[ $# -eq 0 ]]; then
    printf 'Usage: sbatch [resource options] k100_job.sbatch COMMAND [ARGUMENTS...]\n' >&2
    exit 2
fi
module load compiler/dtk/dtk-26.04
if [[ -n "${ICTC_SITE_ENV:-}" ]]; then source "$ICTC_SITE_ENV"; fi
export PATH="$ICTC_ENV/bin:$PATH"
export OMP_NUM_THREADS="${ICTC_OMP_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MAX_JOBS="$OMP_NUM_THREADS"
export HIPBLASLT_ALLOW_TF32=0 NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export PYTORCH_ROCM_ARCH=gfx926
cache="${ICTC_CACHE_DIR:-$SLURM_SUBMIT_DIR/.cache/ictc}"
export XDG_CACHE_HOME="$cache/xdg" TORCH_EXTENSIONS_DIR="$cache/extensions"
export TRITON_CACHE_DIR="$cache/triton" FSCETP_ICTD_CACHE_DIR="$cache/ictd"
export TMPDIR="$cache/tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
cd "$SLURM_SUBMIT_DIR"
exec "$@"
