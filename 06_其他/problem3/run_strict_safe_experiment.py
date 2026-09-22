#!/usr/bin/env python3
"""Run the Q3 four-decimal, strictly-safe threshold experiment.

The production PDE and discretization are unchanged.  The solver event is
lowered to 0.14994 kg/kg, leaving room for the independently estimated field
error below the 0.14995 four-decimal display boundary.  Five refinements are
run exactly once and checkpointed atomically.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from problem3 import solve_problem3 as q3
from problem3.accelerated_problem3 import solve_model_accelerated


RUN_ROOT = PROJECT_ROOT / "problem3" / "strict_safe_experiment"
TASK_ROOT = RUN_ROOT / "tasks"
FINAL_ROOT = RUN_ROOT / "final"
ATTACHMENT = PROJECT_ROOT / "A题" / "附件" / "附件1.xlsx"
TEMPLATE = PROJECT_ROOT / "A题" / "附件" / "附件3" / "result3.xlsx"

SOLVER_EVENT_THRESHOLD = 0.14994
FOUR_DECIMAL_UPPER_EXCLUSIVE = 0.14995

TASKS = [
    {"id": "q3_strict_n640_dt0p5", "radial_cells": 640, "time_step_s": 0.5},
    {"id": "q3_strict_n1280_dt0p5", "radial_cells": 1280, "time_step_s": 0.5},
    {"id": "q3_strict_n2560_dt0p5", "radial_cells": 2560, "time_step_s": 0.5},
    {"id": "q3_strict_n2560_dt1", "radial_cells": 2560, "time_step_s": 1.0},
    {"id": "q3_strict_n2560_dt2", "radial_cells": 2560, "time_step_s": 2.0},
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def config(task: dict[str, Any]) -> q3.NumericalConfig:
    return q3.NumericalConfig(
        radial_cells=task["radial_cells"],
        time_step_s=task["time_step_s"],
        output_interval_s=60.0,
        maximum_time_h=96.0,
        plateau_start_h=3.0,
        environment_extension="fixed",
        fixed_environment_temperature_c=50.0,
        fixed_environment_moisture_kg_kg=0.05,
        picard_tolerance=1.0e-10,
        picard_max_iterations=40,
    )


def fingerprint(task: dict[str, Any]) -> str:
    sources = {
        "attachment": sha256_file(ATTACHMENT),
        "solve_problem3": sha256_file(PROJECT_ROOT / "problem3" / "solve_problem3.py"),
        "accelerated_problem3": sha256_file(
            PROJECT_ROOT / "problem3" / "accelerated_problem3.py"
        ),
        "runner": sha256_file(Path(__file__)),
    }
    return stable_hash(
        {
            "task": task,
            "solver_event_threshold": SOLVER_EVENT_THRESHOLD,
            "four_decimal_upper_exclusive": FOUR_DECIMAL_UPPER_EXCLUSIVE,
            "sources": sources,
        }
    )


def load_checkpoint(task: dict[str, Any], expected_fingerprint: str) -> Any | None:
    directory = TASK_ROOT / task["id"]
    result_path = directory / "result.pkl"
    done_path = directory / "DONE.json"
    if not result_path.exists() and not done_path.exists():
        return None
    if not result_path.is_file() or not done_path.is_file():
        raise RuntimeError(f"incomplete checkpoint: {task['id']}")
    done = json.loads(done_path.read_text(encoding="utf-8"))
    if done.get("fingerprint") != expected_fingerprint:
        raise RuntimeError(f"checkpoint fingerprint mismatch: {task['id']}")
    if done.get("result_sha256") != sha256_file(result_path):
        raise RuntimeError(f"checkpoint hash mismatch: {task['id']}")
    with result_path.open("rb") as handle:
        envelope = pickle.load(handle)
    if envelope.get("fingerprint") != expected_fingerprint:
        raise RuntimeError(f"checkpoint payload mismatch: {task['id']}")
    return envelope["result"]


def save_checkpoint(
    task: dict[str, Any], expected_fingerprint: str, result: Any
) -> None:
    directory = TASK_ROOT / task["id"]
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.pkl"
    done_path = directory / "DONE.json"
    if result_path.exists() or done_path.exists():
        raise RuntimeError(f"refusing to overwrite checkpoint: {task['id']}")
    temporary = result_path.with_name(f".{result_path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        pickle.dump(
            {"fingerprint": expected_fingerprint, "task": task, "result": result},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, result_path)
    atomic_json(
        done_path,
        {
            "fingerprint": expected_fingerprint,
            "result_sha256": sha256_file(result_path),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "drying_time_s": result.drying_time_s,
            "runtime_s": result.runtime_s,
        },
    )


def execute_task(task: dict[str, Any], expected_fingerprint: str) -> str:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    environment = q3.load_environment(ATTACHMENT, 3.0, "fixed", 50.0, 0.05)
    parameters = replace(
        q3.PhysicalParameters(),
        moisture_threshold_kg_kg=SOLVER_EVENT_THRESHOLD,
    )
    started = time.perf_counter()
    print(
        f"START {task['id']} N={task['radial_cells']} dt={task['time_step_s']}s",
        flush=True,
    )
    result = solve_model_accelerated(environment, parameters, config(task), False)
    save_checkpoint(task, expected_fingerprint, result)
    print(
        f"DONE  {task['id']} crossing={result.drying_time_s:.6f}s "
        f"runtime={time.perf_counter() - started:.3f}s",
        flush=True,
    )
    return task["id"]


def common_profile_comparison(a: Any, b: Any) -> dict[str, float]:
    regular_rows = int(math.floor(min(a.drying_time_s, b.drying_time_s) / 60.0))
    aa = np.asarray(a.moistures_kg_kg[:regular_rows], dtype=float)
    bb = np.asarray(b.moistures_kg_kg[:regular_rows], dtype=float)
    difference = np.abs(aa - bb)
    flat = int(np.argmax(difference))
    row, column = np.unravel_index(flat, difference.shape)
    return {
        "max_abs_difference": float(difference[row, column]),
        "rms_difference": float(np.sqrt(np.mean(difference * difference))),
        "max_location_time_s": float((row + 1) * 60.0),
        "max_location_radius_cm": float(column * 0.1),
        "common_regular_rows": regular_rows,
        "drying_time_difference_s": abs(a.drying_time_s - b.drying_time_s),
    }


def richardson(first: dict[str, float], second: dict[str, float], field: str) -> dict[str, float]:
    d1 = first[field]
    d2 = second[field]
    if d1 > d2 > 0.0:
        order = math.log(d1 / d2, 2.0)
        estimate = d2 / (2.0**order - 1.0)
    else:
        order = float("nan")
        estimate = d2
    return {
        "observed_order": order,
        "estimated_fine_error": estimate,
        "estimated_fine_error_with_1p25_safety": 1.25 * estimate,
    }


def assemble(results: dict[str, Any]) -> dict[str, Any]:
    fine = results["q3_strict_n2560_dt0p5"]
    spatial_1 = common_profile_comparison(
        results["q3_strict_n640_dt0p5"], results["q3_strict_n1280_dt0p5"]
    )
    spatial_2 = common_profile_comparison(
        results["q3_strict_n1280_dt0p5"], fine
    )
    temporal_1 = common_profile_comparison(
        results["q3_strict_n2560_dt2"], results["q3_strict_n2560_dt1"]
    )
    temporal_2 = common_profile_comparison(
        results["q3_strict_n2560_dt1"], fine
    )
    spatial_field = richardson(spatial_1, spatial_2, "max_abs_difference")
    temporal_field = richardson(temporal_1, temporal_2, "max_abs_difference")
    spatial_time = richardson(spatial_1, spatial_2, "drying_time_difference_s")
    temporal_time = richardson(temporal_1, temporal_2, "drying_time_difference_s")
    field_bound = (
        spatial_field["estimated_fine_error_with_1p25_safety"]
        + temporal_field["estimated_fine_error_with_1p25_safety"]
    )
    time_bound_s = (
        spatial_time["estimated_fine_error_with_1p25_safety"]
        + temporal_time["estimated_fine_error_with_1p25_safety"]
    )
    certified_continuous_s = fine.drying_time_s + time_bound_s
    engineering_safe_s = math.ceil(certified_continuous_s / 60.0) * 60.0
    certified_upper = SOLVER_EVENT_THRESHOLD + field_bound
    accepted = (
        certified_upper < FOUR_DECIMAL_UPPER_EXCLUSIVE
        and fine.diagnostics.max_radial_monotonicity_violation < 1.0e-10
    )
    if not accepted:
        raise RuntimeError(
            "strict-safe acceptance failed: "
            f"event+field_bound={certified_upper:.12f}"
        )

    FINAL_ROOT.mkdir(parents=True, exist_ok=False)
    result_xlsx = FINAL_ROOT / "result3_strict_safe.xlsx"
    q3.write_result3_xlsx(TEMPLATE, result_xlsx, fine)
    excel = q3.validate_result3_xlsx(result_xlsx, fine)
    q3.write_full_precision_csv(FINAL_ROOT / "moisture_full_precision.csv", fine)
    q3.write_table5_csv(FINAL_ROOT / "table5.csv", fine)
    q3.write_table5_markdown(FINAL_ROOT / "table5.md", fine)
    environment = q3.load_environment(ATTACHMENT, 3.0, "fixed", 50.0, 0.05)
    q3.write_environment_csv(FINAL_ROOT / "environment_input_used.csv", environment)

    metadata = {
        "status": "accepted",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "boundary_after_4h": {"temperature_c": 50.0, "moisture_kg_kg": 0.05},
        "criterion": {
            "mathematical_threshold_exclusive": 0.15,
            "four_decimal_upper_exclusive": FOUR_DECIMAL_UPPER_EXCLUSIVE,
            "solver_event_threshold": SOLVER_EVENT_THRESHOLD,
            "acceptance_expression": "solver_event_threshold + combined_field_error_bound < 0.14995",
        },
        "fine_config": asdict(fine.config),
        "fine_event_time_s": fine.drying_time_s,
        "fine_event_time_h": fine.drying_time_s / 3600.0,
        "fine_event_max_moisture": max(fine.drying_moistures_kg_kg),
        "combined_field_error_bound_kg_kg": field_bound,
        "certified_moisture_upper_bound_kg_kg": certified_upper,
        "combined_crossing_time_error_bound_s": time_bound_s,
        "certified_continuous_safe_time_s": certified_continuous_s,
        "certified_continuous_safe_time_h": certified_continuous_s / 3600.0,
        "engineering_safe_time_s": engineering_safe_s,
        "engineering_safe_time_h": engineering_safe_s / 3600.0,
        "spatial": {
            "coarse_to_medium": spatial_1,
            "medium_to_fine": spatial_2,
            "richardson_field": spatial_field,
            "richardson_time": spatial_time,
        },
        "temporal": {
            "coarse_to_medium": temporal_1,
            "medium_to_fine": temporal_2,
            "richardson_field": temporal_field,
            "richardson_time": temporal_time,
        },
        "diagnostics": {
            **asdict(fine.diagnostics),
            "mean_picard_iterations": fine.diagnostics.mean_picard_iterations,
        },
        "excel_validation": excel,
    }
    atomic_json(FINAL_ROOT / "run_metadata.json", metadata)
    report = [
        "# 第三问严格安全时刻实验",
        "",
        "- 4 h 后环境固定为 50 °C、0.05 kg/kg；",
        f"- 主网格 N=2560，空间步长 {0.02 / 2560:.10e} m；",
        "- 主时间步长 0.5 s；",
        f"- 求解器事件阈值：{SOLVER_EVENT_THRESHOLD:.5f} kg/kg；",
        f"- 合计场值误差安全上界：{field_bound:.8e} kg/kg；",
        f"- 含误差的含水率上界：{certified_upper:.12f} kg/kg < 0.14995；",
        f"- 事件时刻：{fine.drying_time_s:.6f} s；",
        f"- 连续认证安全时刻：{certified_continuous_s:.6f} s；",
        f"- 整分钟工程安全时刻：{engineering_safe_s:.0f} s = {engineering_safe_s / 3600.0:.9f} h；",
        f"- 烘干时刻合计误差安全上界：{time_bound_s:.6f} s；",
        f"- Excel：{excel['dimension']}，ZIP={excel['zip_integrity']}。",
        "",
        "认证采用 `事件阈值 + 场值误差上界 < 0.14995`，并将事件时刻加上",
        "烘干时刻误差上界后再向上取整到整分钟。",
    ]
    (FINAL_ROOT / "validation_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    manifest = {
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {
            str(path.relative_to(RUN_ROOT)): sha256_file(path)
            for path in sorted(FINAL_ROOT.rglob("*"))
            if path.is_file()
        },
    }
    atomic_json(RUN_ROOT / "COMPLETE.json", manifest)
    return metadata


def main() -> int:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if (RUN_ROOT / "COMPLETE.json").exists():
        raise RuntimeError("strict-safe experiment is already complete")
    if FINAL_ROOT.exists():
        raise RuntimeError("final directory exists without COMPLETE.json")

    fingerprints = {task["id"]: fingerprint(task) for task in TASKS}
    results: dict[str, Any] = {}
    pending: list[dict[str, Any]] = []
    for task in TASKS:
        cached = load_checkpoint(task, fingerprints[task["id"]])
        if cached is None:
            pending.append(task)
        else:
            results[task["id"]] = cached
            print(f"REUSE {task['id']}", flush=True)

    print(
        f"Q3 STRICT START tasks={len(TASKS)} pending={len(pending)} "
        f"cpus={os.cpu_count() or 1} threshold={SOLVER_EVENT_THRESHOLD}",
        flush=True,
    )
    if pending:
        with ProcessPoolExecutor(max_workers=min(len(pending), 5)) as executor:
            future_to_task = {
                executor.submit(execute_task, task, fingerprints[task["id"]]): task
                for task in pending
            }
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                task_id = future.result()
                results[task_id] = load_checkpoint(task, fingerprints[task_id])

    metadata = assemble(results)
    print(
        "Q3 STRICT COMPLETE "
        f"safe_h={metadata['certified_continuous_safe_time_h']:.12f} "
        f"engineering_h={metadata['engineering_safe_time_h']:.12f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
