#!/bin/bash
#SBATCH --job-name=mockdock_val
#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=validation_%j.log
#SBATCH --error=validation_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shuhang@unc.edu

# Load modules
module load autodock-gpu
module load autogrid

# Activate conda env without conda init
source /work/users/s/h/shuhang/benchmark/.venv/bin/activate

# Set AutoDock-GPU executable if needed (or rely on module)
export ADGPU_EXE="adgpu"

# Run the validation script for all benchmarks
# Pass --benchmarks AKT1 CHK1 to restrict to a subset
# Pass --show-initial to also output the lower-25% compounds
cd /work/users/s/h/shuhang/benchmark
python -u validation_test.py
