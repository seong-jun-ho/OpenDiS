#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=g1
#SBATCH --job-name=t21_lin
#SBATCH --time=02:00:00
#SBATCH -o lin2.%j.out
#SBATCH -e lin2.%j.err
#
# Is the stress coming back from ABAQUS real?  Run the same deck twice with
# the load doubled.  Elasticity is linear, so the stress must double too.
# Fixed mass scaling pins the increment independently of the load, so both
# runs walk the same time steps and the same step can be compared.
# Read the result with proof_compare.py.
#
# Keep the strain small: at load 10000/20000 nlgeom makes the ratio 1.87.
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
INP=ABAQUS/test_eig_fem_elastic_VUMAT_control.inp
T=test_eig_fem_frank_read_src_pydis.py

cp "$INP" /tmp/_inp.orig
cp "$T" /tmp/_t.orig
# nucleation off -- the load is the only variable that changes between the runs
sed -i 's/nucleation=nucleation,/nucleation=None,/' "$T"
mkdir -p proof

run_one () {
  local LOAD=$1 TAG=$2
  echo "############ $TAG  load=$LOAD"
  sed -i "s/^_PickedSet6, 2, 2, [0-9.]*\./_PickedSet6, 2, 2, ${LOAD}./" "$INP"
  grep -n "_PickedSet6, 2, 2," "$INP"
  local S=proof/stale_$TAG; mkdir -p "$S"
  ( cd ABAQUS
    for f in *; do
      case "$f" in
        *.inp|*.for|*.f|*_bak|.gitignore) continue ;;
        surface_elements.txt|OriginalCoordNodes.txt|FEM_deltaU.txt) continue ;;
      esac
      [ -e "$f" ] && mv "$f" "../$S/"
    done )
  for f in FEMSTRESS SurfaceStress; do [ -e "$f" ] && mv "$f" "$S/"; done
  [ -d output ] && mv output "$S/output"
  rm -rf sample_run_proof_$TAG

  sed -i "s/snapshot_dir=\"[^\"]*\"/snapshot_dir=\"sample_run_proof_$TAG\"/" "$T"
  timeout 3600 python3 -u "$T" > "proof/run_$TAG.log" 2>&1
  echo "### $TAG exit=$?"
  grep -c "^step = " "proof/run_$TAG.log" | sed 's/^/  steps: /'
  # The final FEMSTRESS is what proof_compare.py reads.  Do not compare the
  # per-cycle snapshots: the two runs do not stop on the same cycle.
  cp FEMSTRESS "proof/femstress_$TAG.txt" 2>/dev/null \
    && echo "  saved proof/femstress_$TAG.txt"
  cp sample_run_proof_$TAG/Output_stress0001.txt "proof/stress_$TAG.txt" 2>/dev/null \
    && echo "  saved proof/stress_$TAG.txt ($(wc -l < proof/stress_$TAG.txt) rows)"
}

run_one 1000 x1
run_one 2000 x2

cp /tmp/_inp.orig "$INP"
cp /tmp/_t.orig "$T"
echo "deck/test restored: $(grep -o '_PickedSet6, 2, 2, [0-9.]*' $INP)  $(grep -c 'nucleation=nucleation' $T)"
echo "DONE $(date +%H:%M:%S)"
python3 proof_compare.py
