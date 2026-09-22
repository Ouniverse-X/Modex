#!/usr/bin/env bash
set -euo pipefail

project_dir=/home/beihang/projects/Modex
run_dir="$project_dir/ultrafine_runs/fixed_50_005"
pid_file="$run_dir/nohup.pid"
log_file="$run_dir/run.log"

mkdir -p "$run_dir"
if [[ -f "$pid_file" ]]; then
    existing_pid=$(<"$pid_file")
    if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
        exit 0
    fi
fi

cd "$project_dir"
nohup env \
    PYTHONPATH=/tmp/modex-problem2-deps \
    PYTHONUNBUFFERED=1 \
    python3 -u run_ultrafine_fixed_boundary.py \
    >>"$log_file" 2>&1 &
run_pid=$!
printf '%s\n' "$run_pid" >"$pid_file"
wait "$run_pid"
