#!/bin/bash
#================================================================
# SLURM Job Script: Lung X-Ray Segmentation
# Northeastern University - Explorer Cluster
#================================================================
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --job-name=lung_seg
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

echo "========================================"
echo "Lung X-Ray Segmentation"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

# Load modules
module purge
module load anaconda3

# Activate environment
source activate venv-lungseg

# Print GPU info
echo ""
echo "GPU Info:"
nvidia-smi
echo ""

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

echo "Starting training..."
echo "========================================"

# Run training
python src/train.py \
    --data_dir ~/lung_seg_hpc/Chest-X-Ray \
    --output_dir ~/lung_seg_hpc/outputs \
    --epochs 50 \
    --batch_size 16

echo ""
echo "========================================"
echo "Training finished at $(date)"
echo "========================================"