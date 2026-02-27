#!/bin/bash
#SBATCH --job-name=frag_fcgmb
#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=frag_%j.log
#SBATCH --error=frag_%j.err

module load autodock-gpu
module load autogrid

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate py312
source /work/users/s/h/shuhang/f-RAG/.venv/bin/activate

SCRIPT_DIR=/work/users/s/h/shuhang/benchmark/exps/f-rag
FRAG_ROOT=/work/users/s/h/shuhang/f-RAG
OUT_DIR=${SCRIPT_DIR}/outputs
BUDGET=5000
SEED=0

cd "${SCRIPT_DIR}"
python -u run.py \
    --budget "${BUDGET}" \
    --seed "${SEED}"
