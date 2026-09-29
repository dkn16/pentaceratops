#!/bin/bash
#SBATCH --job-name=penta-prepare
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=penta-prepare-%j.log
set -euo pipefail

# Submit from the package directory in an environment with the package installed.
# Account, partition/QOS, and machine constraints belong in your sbatch command.
# Example (NERSC):
# sbatch -A m4031 -C cpu -q debug examples/prepare.slurm tess --target TOI-4616.01
# This is one target, not a serial population sweep.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
mission="${1:?First argument must be tess or kepler}"
shift
case "$mission" in
  tess|kepler) ;;
  *) echo "First argument must be tess or kepler" >&2; exit 2 ;;
esac
cd "${SLURM_SUBMIT_DIR:?Submit from the package directory}"
python -u "examples/${mission}_candidate.py" "$@"
