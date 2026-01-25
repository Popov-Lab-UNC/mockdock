#!/bin/bash
#SBATCH --job-name=dock_variance
#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=variance_%j.log
#SBATCH --error=variance_%j.err
#SBATCH --mail-user=shuhang@unc.edu
#SBATCH --mail-type=ALL

module load autodock-gpu
module load autogrid
source /work/users/s/h/shuhang/benchmark/.venv/bin/activate

cd /work/users/s/h/shuhang/benchmark

python run_variance.py
