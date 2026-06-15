#!/bin/bash
# Shared environment helpers for benchmark scripts.

mockdock_project_root() {
    if [[ -n "${BENCHMARK_DIR:-}" ]]; then
        printf '%s\n' "${BENCHMARK_DIR}"
    elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/pyproject.toml" ]]; then
        printf '%s\n' "${SLURM_SUBMIT_DIR}"
    else
        cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
    fi
}

mockdock_setup_root() {
    export BENCHMARK_DIR="$(mockdock_project_root)"
    cd "${BENCHMARK_DIR}" || return 1
    mkdir -p logs
    export PYTHONPATH="${BENCHMARK_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
    echo "Benchmark directory: ${BENCHMARK_DIR}"
    echo "PYTHONPATH: ${PYTHONPATH}"
}

mockdock_activate_venv() {
    local venv_path="${1:-}"
    if [[ -n "${venv_path}" && -d "${venv_path}" ]]; then
        echo "Activating virtual environment: ${venv_path}"
        source "${venv_path}/bin/activate"
    elif [[ -d "${BENCHMARK_DIR}/.venv" ]]; then
        echo "Activating virtual environment: ${BENCHMARK_DIR}/.venv"
        source "${BENCHMARK_DIR}/.venv/bin/activate"
    elif [[ -d "${BENCHMARK_DIR}/venv" ]]; then
        echo "Activating virtual environment: ${BENCHMARK_DIR}/venv"
        source "${BENCHMARK_DIR}/venv/bin/activate"
    else
        echo "No benchmark virtual environment found; continuing with current Python."
    fi
    echo "Using Python: $(command -v python)"
}

mockdock_activate_conda() {
    local env_name="$1"
    local conda_sh="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
    if [[ ! -f "${conda_sh}" ]]; then
        echo "Could not find conda activation script at ${conda_sh}."
        return 1
    fi
    source "${conda_sh}"
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        local venv_bin="${VIRTUAL_ENV}/bin"
        PATH=$(echo "${PATH}" | sed -E "s|${venv_bin}:?||g")
        PATH="${PATH}:${venv_bin}"
        unset VIRTUAL_ENV
    fi
    conda activate "${env_name}"
    echo "Activated conda environment: ${env_name}"
    echo "Using Python: $(command -v python)"
}
