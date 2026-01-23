#!/bin/bash
#SBATCH --job-name=fcgmb_val
#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=validation_%j.log
#SBATCH --error=validation_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shuhang@unc.edu

# Load modules
module load autodock-gpu
module load autogrid

# Activate virtual environment
source /work/users/s/h/shuhang/benchmark/.venv/bin/activate

# Set AutoDock-GPU executable if needed (or rely on module)
export ADGPU_EXE="adgpu"

# Run the validation script
cd /work/users/s/h/shuhang/benchmark
python -u validation_test.py
