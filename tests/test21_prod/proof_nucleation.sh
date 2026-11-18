#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=g1
#SBATCH --job-name=t21_nuc1
#SBATCH --time=02:00:00
#SBATCH -o nuc1.%j.out
#SBATCH -e nuc1.%j.err
#
# One nucleation event and no more.  The loop is a hexagon of known area, so
# the plastic strain it hands to ABAQUS can be checked against that area.
# Read the result with proof_nucleation.py.
#
#   (11/18/2026, junho)
export LANG=en_US.UTF-8
source /etc/profile.d/modules.sh
module purge
module add compiler/2023.1.0 mpi/2021.12.1 mkl/2023.1.0 ABAQUS/2024
export PATH=/TGM/local/bin:$PATH
export OMP_NUM_THREADS=1
unset SLURM_NTASKS SLURM_NPROCS SLURM_NTASKS_PER_NODE SLURM_TASKS_PER_NODE \
      SLURM_JOB_CPUS_PER_NODE SLURM_CPUS_ON_NODE SLURM_NODELIST SLURM_JOB_NODELIST
export MPI_LOCALNRANKS=1
echo "host=$(hostname)  abaqus=$(which abaqus || echo NONE)"

D=/home/wnswjswk1/Codes/OpenDiS_new/tests/test21_prod
cd "$D" || exit 1
mkdir -p proof

# park whatever the previous run left behind
S=proof/stale_one_nuc; rm -rf "$S"; mkdir -p "$S"
( cd ABAQUS
  for f in *; do
    case "$f" in
      *.inp|*.for|*.f|*_bak|.gitignore) continue ;;
      surface_elements.txt|OriginalCoordNodes.txt|FEM_deltaU.txt) continue ;;
    esac
    [ -e "$f" ] && mv "$f" "../$S/"
  done )
for f in FEMSTRESS SurfaceStress; do [ -e "$f" ] && mv "$f" "$S/"; done
rm -rf output_one_nuc sample_run_one_nuc

timeout 3600 python3 -u test_nucleation_one.py > proof/run_one_nuc.log 2>&1
echo "### exit=$?"
grep -c "^step = " proof/run_one_nuc.log | sed 's/^/  steps: /'
grep -c "Nucleation loop inserted" proof/run_one_nuc.log | sed 's/^/  loops nucleated: /'
grep -n "Nucleation loop inserted\|cap is" proof/run_one_nuc.log | head -5
echo "DONE $(date +%H:%M:%S)"

echo ""
python3 proof_nucleation.py proof/run_one_nuc.log output_one_nuc
