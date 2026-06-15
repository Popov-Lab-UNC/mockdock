#!/bin/bash
# Shared SLURM logging helpers for benchmark scripts.

mockdock_slurm_header() {
    local label="$1"
    echo "================================================================="
    echo "${label} started at $(date)"
    echo "Node: ${SLURM_NODENAME:-local}"
    echo "Job ID: ${SLURM_JOB_ID:-none}"
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    fi
    echo "================================================================="
}

mockdock_slurm_footer() {
    local label="$1"
    echo "${label} finished at $(date)"
}
