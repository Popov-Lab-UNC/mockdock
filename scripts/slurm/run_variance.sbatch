#!/bin/bash
#SBATCH --job-name=dock_variance
#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01-00:00:00
#SBATCH --output=variance_%j.log
#SBATCH --error=variance_%j.err
#SBATCH --mail-user=shuhang@unc.edu
#SBATCH --mail-type=ALL

module load autodock-gpu
module load autogrid
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate cctbx
source /work/users/s/h/shuhang/benchmark/.venv/bin/activate

python /work/users/s/h/shuhang/benchmark/scripts/run_variance.py --config-dir /work/users/s/h/shuhang/benchmark/top_30_configs --iterations 5 --output-dir variance_runs
