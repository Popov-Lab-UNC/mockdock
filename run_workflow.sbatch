#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1

##SBATCH --partition=general
##SBATCH -n 1
##SBATCH --cpus-per-task=32

#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=benchmark_%j.log
#SBATCH --error=benchmark_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shuhang@unc.edu

module load autodock-gpu
module load autogrid
source /work/users/s/h/shuhang/benchmark/.venv/bin/activate

cd /work/users/s/h/shuhang/benchmark
#python -u run_workflow.py --config PptT.yaml
python run_workflow.py --config PptT.yaml --stage docking