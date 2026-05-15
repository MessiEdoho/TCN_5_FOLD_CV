#!/bin/bash -l
#SBATCH --job-name=create_5fold_cv
# CPU-only, no GPU. Reads the enriched source manifest, performs
# StratifiedKFold(n_splits=6) over the 71 train mice with a 2x2
# (prevalence x volume) crossed stratification, and writes the
# 5-fold CV manifest + diagnostic CSVs to
# /home/people/22206468/scratch/INPUT_CV_PROJECT/.
# Wall-time: < 1 minute.
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

#SBATCH --partition=cs

#SBATCH -t 1-00:00:00

#SBATCH --output=/home/people/22206468/slurm-create_5fold_cv-%j.out
##SBATCH --error=/home/people/22206468/slurm-create_5fold_cv-%j.err

#SBATCH --mail-type=ALL
#SBATCH --mail-user=mercy.edoho@ucdconnect.ie

echo "===== JOB START ====="
date
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: ${SLURM_JOB_PARTITION:-unset}"

echo "----- CPU allocation -----"
echo "SLURM_CPUS_PER_TASK : ${SLURM_CPUS_PER_TASK:-unset}"
echo "SLURM_CPUS_ON_NODE  : ${SLURM_CPUS_ON_NODE:-unset}"
echo "nproc (visible)     : $(nproc)"
echo "Affinity (taskset)  : $(taskset -cp $$ 2>/dev/null || echo 'taskset unavailable')"
echo "--------------------------"

module purge
module load anaconda3
conda activate torch_v100_py310

cd ~/TCN_5_FOLD_CV

python create_5fold_cv_splits.py

echo "===== JOB END ====="
date
