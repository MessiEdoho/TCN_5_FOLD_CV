#!/bin/bash -l
#SBATCH --job-name=cv_mstcn
# One node with one GPU for PyTorch end-to-end training, 5 folds
# executed sequentially in a single allocation.
#SBATCH -N 1
# Single Python process with 10 CPUs available for DataLoader workers
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10

#SBATCH --partition=csgpu

##SBATCH --exclude=sonicgpu20
# Request 1 GPU
#SBATCH --gres=gpu:1

# 5-fold subject-disjoint cross-validation training for the
# MultiScaleTCN architecture using the HPT-best hyperparameters from
# /scratch/.../MultiScaleTCNtuning_outputs/best_multiscale_params.json
# and the 6-partition mouse manifest produced by create_5fold_cv_splits.py.
#
# Per fold: train on 4 of the 5 CV partitions, monitor val_loss and
# val_macro_F1 every epoch on the fixed early_stop_mice 6th partition
# (drives early stopping, patience=15, no warm-up), then evaluate ONCE
# on the held-out CV partition. The test-fold evaluation is raw
# (no probability smoothing, no event-level metrics, fixed threshold 0.5).
# Aggregates the 7 segment-level metrics across folds as mean +/- std.
#
# Five folds at MAX_EPOCHS=100 with patience=15 should fit comfortably
# inside the 13-day wall time used for parent training runs.
#SBATCH -t 13-00:00:00

# Email notifications at start, end, and failure
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mercy.edoho@ucdconnect.ie

echo "===== JOB START ====="
date
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU allocated: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none detected')"

# CPU allocation diagnostics: confirm SLURM gave us 10 cores AND that
# the Python process can actually use all of them (cpuset / cgroup binding).
echo "----- CPU allocation -----"
echo "SLURM_CPUS_PER_TASK : ${SLURM_CPUS_PER_TASK:-unset}"
echo "SLURM_CPUS_ON_NODE  : ${SLURM_CPUS_ON_NODE:-unset}"
echo "nproc (visible)     : $(nproc)"
echo "Affinity (taskset)  : $(taskset -cp $$ 2>/dev/null || echo 'taskset unavailable')"
echo "--------------------------"

# Activate environment (same as parent project)
module purge
module load anaconda3
conda activate torch_v100_py310

# Tell the script where the parent project lives so it can import
# MultiScaleTCN, make_loader, train_one_epoch, set_seed.
export DL_WITH_SSL_GA_PATH=${DL_WITH_SSL_GA_PATH:-$HOME/TCN_SSL_GA}
echo "DL_WITH_SSL_GA_PATH : ${DL_WITH_SSL_GA_PATH}"

cd ~/TCN_5_FOLD_CV

# Default paths are baked into the script; override here only if needed.
python train_5fold_cv_multiscaleTCN.py \
    --cv-manifest /home/people/22206468/scratch/INPUT_CV_PROJECT/manifest/data_splits_5fold_cv.json \
    --best-params /home/people/22206468/scratch/OUTPUT/MODEL3_OUTPUT/MultiScaleTCNtuning_outputs/best_multiscale_params.json \
    --output-dir  /home/people/22206468/scratch/OUTPUT_CV_PROJECT/MODEL_3_MTCN \
    --folds       0,1,2,3,4

echo "===== JOB END ====="
date
