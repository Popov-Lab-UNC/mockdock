#!/bin/bash
#SBATCH --job-name=pptt_models_noclip
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/pptt_models_noclip_%j.log
#SBATCH --error=logs/pptt_models_noclip_%j.err

set -euo pipefail

module load autodock-gpu

mkdir -p logs

BENCHMARK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${BENCHMARK_DIR}" || exit 1

CONFIG_PATH="${BENCHMARK_DIR}/src/mockdock/configs/PptT.toml"

python - <<'PY'
from pathlib import Path
try:
    import tomllib
except ImportError:
    import tomli as tomllib

cfg = tomllib.loads(Path("src/mockdock/configs/PptT.toml").read_text())
if cfg.get("low_score") is None or cfg.get("high_score") is None:
    raise SystemExit(
        "PptT low_score/high_score are not set in src/mockdock/configs/PptT.toml. "
        "Run variance first and fill thresholds before model runs."
    )
print("PptT thresholds detected; continuing.")
PY

NUM_RUNS=5
BASE_SEED=1
BENCHMARK_NAME="PptT"
CLIP_FLAG="--no-clip-reward-upper-bound"
DEST_ROOT="${BENCHMARK_DIR}/exps"
TMP_ROOT="${BENCHMARK_DIR}/_pptt_staging/no_upperbound"
BASE_TS="$(date +%Y%m%d_%H%M%S)"

mkdir -p "${TMP_ROOT}" "${DEST_ROOT}"

run_acegen_model() {
    local model="$1"
    local script_dir="${BENCHMARK_DIR}/exps/${model}"
    local out_dir="${script_dir}/outputs"
    local final_model_dir="${DEST_ROOT}/${model}"
    mkdir -p "${final_model_dir}"

    source /work/users/s/h/shuhang/acegen-open/.venv/bin/activate
    for run_idx in $(seq 1 "${NUM_RUNS}"); do
        local run_tag
        run_tag=$(printf "r%02d" "${run_idx}")
        local run_seed=$((BASE_SEED + run_idx - 1))
        local staged_parent="${TMP_ROOT}/${model}/run_${BASE_TS}_${run_tag}"
        mkdir -p "${staged_parent}"
        python -u "${script_dir}/run.py" \
            --benchmark "${BENCHMARK_NAME}" \
            --budget 1000 \
            --seed "${run_seed}" \
            --out "${out_dir}" \
            --acegen-root /work/users/s/h/shuhang/acegen-open \
            --n-warmup 25 \
            "${CLIP_FLAG}" \
            --run-dir "${staged_parent}"
        mv "${staged_parent}" "${final_model_dir}/"
    done
}

run_libinvent() {
    local model="libinvent"
    local script_dir="${BENCHMARK_DIR}/exps/${model}"
    local out_dir="${script_dir}/outputs"
    local final_model_dir="${DEST_ROOT}/${model}"
    mkdir -p "${final_model_dir}"

    source /work/users/s/h/shuhang/REINVENT4/.venv/bin/activate
    for run_idx in $(seq 1 "${NUM_RUNS}"); do
        local run_tag
        run_tag=$(printf "r%02d" "${run_idx}")
        local run_seed=$((BASE_SEED + run_idx - 1))
        local staged_parent="${TMP_ROOT}/${model}/run_${BASE_TS}_${run_tag}"
        mkdir -p "${staged_parent}"
        python -u "${script_dir}/run.py" \
            --benchmark "${BENCHMARK_NAME}" \
            --budget 1000 \
            --seed "${run_seed}" \
            --out "${out_dir}" \
            --run-dir "${staged_parent}" \
            --prior-file /work/users/s/h/shuhang/REINVENT4/priors/libinvent.prior \
            --n-warmup 25 \
            --device cuda:0 \
            --docking-backend auto \
            "${CLIP_FLAG}"
        mv "${staged_parent}" "${final_model_dir}/"
    done
}

run_genmol() {
    local model="genmol"
    local script_dir="${BENCHMARK_DIR}/exps/${model}"
    local out_dir="${script_dir}/outputs"
    local final_model_dir="${DEST_ROOT}/${model}"
    mkdir -p "${final_model_dir}"

    module load apptainer
    module load autodock-gpu
    source /work/users/s/h/shuhang/benchmark/.venv/bin/activate
    for run_idx in $(seq 1 "${NUM_RUNS}"); do
        local run_tag
        run_tag=$(printf "r%02d" "${run_idx}")
        local run_seed=$((BASE_SEED + run_idx - 1))
        local staged_parent="${TMP_ROOT}/${model}/run_${BASE_TS}_${run_tag}"
        mkdir -p "${staged_parent}"
        python -u "${script_dir}/run.py" \
            --benchmark "${BENCHMARK_NAME}" \
            --budget 1000 \
            --seed "${run_seed}" \
            --out "${out_dir}" \
            --run-dir "${staged_parent}" \
            "${CLIP_FLAG}" \
            --batch-size 64 \
            --container-image /work/users/s/h/shuhang/genmol/genmol.sif \
            --genmol-dir /work/users/s/h/shuhang/genmol \
            --workdir "${BENCHMARK_DIR}"
        mv "${staged_parent}" "${final_model_dir}/"
    done
}

echo "Starting PptT no-upperbound model suite at $(date)"
echo "Benchmark: ${BENCHMARK_NAME}"
echo "Destination root: ${DEST_ROOT}"
echo "Staging root: ${TMP_ROOT}"

for model in acegen-a2c acegen-ahc acegen-ppo acegen-ppod acegen-reinforce acegen-reinvent; do
    echo "Running ${model} ..."
    run_acegen_model "${model}"
done

echo "Running libinvent ..."
run_libinvent

echo "Running genmol ..."
run_genmol

echo "All PptT no-upperbound runs completed at $(date)"
