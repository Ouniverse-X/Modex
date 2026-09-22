#!/usr/bin/env bash
set -euo pipefail

project_dir=/home/beihang/projects/Modex
run_dir="$project_dir/ultrafine_runs/fixed_50_005_v2"
pid_file="$run_dir/nohup.pid"
log_file="$run_dir/run.log"
python_bin=/home/beihang/.local/bin/python3.10
dependency_dir=/tmp/modex-problem2-deps

mkdir -p "$run_dir"
if [[ -f "$pid_file" ]]; then
    existing_pid=$(<"$pid_file")
    if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
        printf 'already running: PID %s\n' "$existing_pid"
        exit 0
    fi
fi

cd "$project_dir"
nohup env \
    PYTHONPATH="$dependency_dir" \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    "$python_bin" -u run_ultrafine_fixed_boundary_v2.py \
    >>"$log_file" 2>&1 &
run_pid=$!
printf '%s\n' "$run_pid" >"$pid_file"
printf 'started: PID %s\nlog: %s\n' "$run_pid" "$log_file"
