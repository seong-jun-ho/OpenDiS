#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=g1
#SBATCH --job-name=t21_one
#SBATCH --time=01:00:00
#SBATCH -o run.%j.out
#SBATCH -e run.%j.err
#
# One case, in a directory of its own, with nothing to wait for.
#
#   sbatch run_one.sh <tag> [test script]
#
# A fresh directory per run is what keeps one run out of the next one's
# way -- not waiting for ABAQUS to exit.  An earlier version waited up to
# ten minutes per run for processes that never exit, turning two minutes
# of work into twenty; the directory is the fix, the wait was not.
#
# preflight refuses to start unless the directory holds only the four
# input files; postflight says whether the run started cold, which it did
# if the first step had no stress to read yet.
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

T=/home/wnswjswk1/Codes/OpenDiS_new/tests
TAG=${1:-run}
SRC=${2:-$T/test21_prod/test_eig_fem_frank_read_src_pydis.py}
D=$T/_run_$TAG
IN=$T/test21_prod/ABAQUS

echo "host=$(hostname)  abaqus=$(which abaqus || echo NONE)  tag=$TAG"
echo "script=$SRC"

rm -rf "$D"
mkdir -p "$D/ABAQUS" || exit 1
cp "$SRC" "$D/case.py" || exit 1
for f in test_eig_fem_elastic_VUMAT_control.inp FEM_deltaU.txt \
         surface_elements.txt OriginalCoordNodes.txt; do
  cp "$IN/$f" "$D/ABAQUS/" || exit 1
done

junk=$(cd "$D" && ls -A | grep -v -x -e ABAQUS -e case.py)
junk="$junk $(cd "$D/ABAQUS" && ls -A | grep -v -x \
      -e test_eig_fem_elastic_VUMAT_control.inp -e FEM_deltaU.txt \
      -e surface_elements.txt -e OriginalCoordNodes.txt)"
if [ -n "$(echo $junk)" ]; then
  echo "PREFLIGHT FAIL - not a clean directory: $(echo $junk)"
  exit 1
fi
echo "preflight ok"

cd "$D" || exit 1
start=$(date +%s)
timeout 3600 python3 -u case.py > run.log 2>&1
rc=$?
echo "### exit=$rc  wall=$(( $(date +%s) - start )) s"

skips=$(grep -c "no SurfaceStress yet" run.log)
if [ "$skips" -lt 1 ]; then
  echo "POSTFLIGHT FAIL - this run did not start cold"
else
  echo "postflight ok - cold start"
fi

echo "  steps           : $(grep -c '^step = ' run.log)"
echo "  loops nucleated : $(grep -c 'Nucleation loop inserted' run.log)"
echo "  step 0          : $(grep -m1 '^step = 0,' run.log)"
echo "  last step       : $(grep '^step = ' run.log | tail -1)"
echo "  first RSS       : $(grep -m1 'totals: totalR' run.log)"
grep -q Traceback run.log && { echo "  CRASHED:"; grep -A3 Traceback run.log | tail -6; }
echo "  [RN]  last      : $(grep '^\[RN\]' run.log | tail -1)"
echo "  [CLEAN] totals  :"
grep '^\[CLEAN\]' run.log | tail -3 | sed 's/^/    /'
echo "DONE $(date +%H:%M:%S)   dir=$D"
