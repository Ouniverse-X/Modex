#!/usr/bin/env python3
"""Idempotent second-stage ultrafine run for Questions 3 and 4.

Boundary after Attachment 1: 50 degC and 0.05 kg/kg.

Every unique (space, time) configuration is solved exactly once and stored as
an atomic, fingerprinted checkpoint.  Re-running this program verifies and
reuses completed checkpoints instead of recomputing them.  Final artifacts are
also assembled atomically and protected by a completion manifest.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pickle
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from problem3 import solve_problem3 as q3
from problem3.accelerated_problem3 import solve_model_accelerated
from problem4 import solve_problem4 as q4


ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "ultrafine_runs" / "fixed_50_005_v2"
TASK_ROOT = RUN_ROOT / "tasks"
Q3_FINAL = RUN_ROOT / "problem3_final"
Q4_FINAL = RUN_ROOT / "problem4_final"
MANIFEST_PATH = RUN_ROOT / "manifest.json"
COMPLETE_PATH = RUN_ROOT / "COMPLETE.json"
LOGICAL_CPUS = os.cpu_count() or 1


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
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def artifact_hashes(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def verified_complete() -> bool:
    if not COMPLETE_PATH.is_file():
        return False
    completion = json.loads(COMPLETE_PATH.read_text(encoding="utf-8"))
    if not MANIFEST_PATH.is_file() or completion.get("manifest_sha256") != sha256_file(
        MANIFEST_PATH
    ):
        raise RuntimeError("completed run manifest failed hash verification")
    for relative, expected in completion.get("artifact_sha256", {}).items():
        path = RUN_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"completed artifact failed hash verification: {path}")
    return True


def task_fingerprint(task: dict[str, Any], source_hashes: dict[str, str]) -> str:
    return stable_hash({"task": task, "sources": source_hashes})


def task_checkpoint(task_id: str) -> tuple[Path, Path]:
    directory = TASK_ROOT / task_id
    return directory / "result.pkl", directory / "DONE.json"


def load_checkpoint(task: dict[str, Any], fingerprint: str) -> Any | None:
    result_path, done_path = task_checkpoint(task["id"])
    if not result_path.exists() and not done_path.exists():
        return None
    if not result_path.is_file():
        raise RuntimeError(f"incomplete checkpoint has no result: {task['id']}")
    with result_path.open("rb") as handle:
        envelope = pickle.load(handle)
    if envelope.get("fingerprint") != fingerprint:
        raise RuntimeError(f"checkpoint fingerprint mismatch: {task['id']}")
    result_hash = sha256_file(result_path)
    if done_path.is_file():
        done = json.loads(done_path.read_text(encoding="utf-8"))
        if done.get("fingerprint") != fingerprint or done.get("result_sha256") != result_hash:
            raise RuntimeError(f"checkpoint completion metadata mismatch: {task['id']}")
    else:
        atomic_json(
            done_path,
            {"fingerprint": fingerprint, "result_sha256": result_hash, "recovered": True},
        )
    return envelope["result"]


def save_checkpoint(task: dict[str, Any], fingerprint: str, result: Any) -> None:
    result_path, done_path = task_checkpoint(task["id"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if result_path.exists() or done_path.exists():
        raise RuntimeError(f"refusing to overwrite checkpoint: {task['id']}")
    temporary = result_path.with_name(f".{result_path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        pickle.dump(
            {"fingerprint": fingerprint, "task": task, "result": result},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, result_path)
    atomic_json(
        done_path,
        {
            "fingerprint": fingerprint,
            "result_sha256": sha256_file(result_path),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def q3_config(task: dict[str, Any]) -> q3.NumericalConfig:
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


def q4_config(task: dict[str, Any]) -> q4.NumericalConfig:
    return q4.NumericalConfig(
        radial_cells=task["radial_cells"],
        output_interval_s=60.0,
        reference_surface_layer=0.1,
        surface_refinement_factor=10,
        relative_tolerance=task["rtol"],
        temperature_absolute_tolerance=task["rtol"],
        moisture_absolute_tolerance=task["rtol"] * 0.01,
        initial_maximum_step_s=task["initial_max_step_s"],
        drying_maximum_step_s=task["drying_max_step_s"],
        first_step_s=task["initial_max_step_s"] * 5.0e-5,
        environment_data_end_s=14400.0,
        drying_segment_s=21600.0,
        maximum_duration_s=14.0 * 86400.0,
    )


def execute_task(
    task: dict[str, Any],
    fingerprint: str,
    q3_environment: q3.EnvironmentData,
    q4_environment: q4.EnvironmentData,
    radius_data: q4.RadiusData,
    cpu_ids: list[int],
) -> tuple[str, float]:
    if hasattr(os, "sched_setaffinity") and cpu_ids:
        os.sched_setaffinity(0, set(cpu_ids))
    os.environ["OMP_NUM_THREADS"] = str(max(1, len(cpu_ids)))
    os.environ["OPENBLAS_NUM_THREADS"] = str(max(1, len(cpu_ids)))
    os.environ["MKL_NUM_THREADS"] = str(max(1, len(cpu_ids)))
    started = time.perf_counter()
    print(f"START {task['id']} cpus={cpu_ids[0]}-{cpu_ids[-1]}", flush=True)
    if task["problem"] == 3:
        result = solve_model_accelerated(
            q3_environment, q3.PhysicalParameters(), q3_config(task), False
        )
    else:
        result = q4.solve_until_dry(
            q4_environment,
            radius_data,
            q4.PhysicalParameters(),
            q4_config(task),
            task["id"],
        )
    save_checkpoint(task, fingerprint, result)
    elapsed = time.perf_counter() - started
    print(f"DONE  {task['id']} elapsed_s={elapsed:.3f}", flush=True)
    return task["id"], elapsed


def common_profile_comparison(a: Any, b: Any) -> dict[str, Any]:
    regular_end = int(math.floor(min(a.drying_time_s, b.drying_time_s) / 60.0))
    aa = np.asarray(a.moistures_kg_kg[:regular_end], dtype=float)
    bb = np.asarray(b.moistures_kg_kg[:regular_end], dtype=float)
    difference = np.abs(aa - bb)
    flat = int(np.argmax(difference))
    row, column = np.unravel_index(flat, difference.shape)
    return {
        "max_abs_difference": float(difference[row, column]),
        "rms_difference": float(np.sqrt(np.mean(difference * difference))),
        "max_location_time_s": int((row + 1) * 60),
        "max_location_radius_cm": float(column * 0.1),
        "common_regular_rows": regular_end,
        "drying_time_difference_s": abs(a.drying_time_s - b.drying_time_s),
    }


def richardson(first: dict[str, Any], second: dict[str, Any], key: str) -> dict[str, float]:
    d1 = float(first[key])
    d2 = float(second[key])
    if d1 <= 0.0 or d2 <= 0.0:
        raise RuntimeError(f"non-positive convergence difference for {key}")
    order = math.log(d1 / d2, 2.0)
    if order <= 0.0:
        # At extremely tight tolerances, a diagnostic can reach its roundoff
        # floor and lose monotone ratios.  Use the full last refinement change
        # instead of inventing a positive convergence order.
        estimate = d2
    else:
        estimate = d2 / (2.0**order - 1.0)
    return {
        "observed_order": order,
        "estimated_fine_error": estimate,
        "estimated_fine_error_with_1p25_safety": 1.25 * estimate,
    }


def build_q3_final(results: dict[str, Any], environment: q3.EnvironmentData) -> dict[str, Any]:
    if Q3_FINAL.is_dir():
        metadata = Q3_FINAL / "run_metadata.json"
        if not metadata.is_file() or not (Q3_FINAL / "result3.xlsx").is_file():
            raise RuntimeError("existing Q3 final directory is incomplete")
        validation = json.loads(metadata.read_text(encoding="utf-8"))
        if validation.get("boundary_after_4h") != {
            "temperature_c": 50.0,
            "moisture_kg_kg": 0.05,
        }:
            raise RuntimeError("existing Q3 final directory has the wrong boundary")
        print("REUSE problem3_final", flush=True)
        return validation
    fine = results["q3_n5120_dt0p25"]
    spatial_1 = common_profile_comparison(results["q3_n1280_dt0p25"], results["q3_n2560_dt0p25"])
    spatial_2 = common_profile_comparison(results["q3_n2560_dt0p25"], fine)
    temporal_1 = common_profile_comparison(results["q3_n5120_dt1"], results["q3_n5120_dt0p5"])
    temporal_2 = common_profile_comparison(results["q3_n5120_dt0p5"], fine)
    spatial_field = richardson(spatial_1, spatial_2, "max_abs_difference")
    temporal_field = richardson(temporal_1, temporal_2, "max_abs_difference")
    spatial_time = richardson(spatial_1, spatial_2, "drying_time_difference_s")
    temporal_time = richardson(temporal_1, temporal_2, "drying_time_difference_s")
    combined_field_bound = (
        spatial_field["estimated_fine_error_with_1p25_safety"]
        + temporal_field["estimated_fine_error_with_1p25_safety"]
    )
    combined_time_bound_s = (
        spatial_time["estimated_fine_error_with_1p25_safety"]
        + temporal_time["estimated_fine_error_with_1p25_safety"]
    )

    temporary = RUN_ROOT / f".problem3_final.tmp.{os.getpid()}"
    if temporary.exists() or Q3_FINAL.exists():
        raise RuntimeError("Q3 final output directory already exists without completion marker")
    temporary.mkdir(parents=True)
    result_xlsx = temporary / "result3.xlsx"
    q3.write_result3_xlsx(ROOT / "A题/附件/附件3/result3.xlsx", result_xlsx, fine)
    excel = q3.validate_result3_xlsx(result_xlsx, fine)
    q3.write_full_precision_csv(temporary / "moisture_full_precision.csv", fine)
    q3.write_table5_csv(temporary / "table5.csv", fine)
    q3.write_table5_markdown(temporary / "table5.md", fine)
    q3.write_environment_csv(temporary / "environment_input_used.csv", environment)
    validation = {
        "boundary_after_4h": {"temperature_c": 50.0, "moisture_kg_kg": 0.05},
        "fine_config": asdict(fine.config),
        "drying_time_s": fine.drying_time_s,
        "drying_time_h": fine.drying_time_s / 3600.0,
        "fine_runtime_s": fine.runtime_s,
        "fine_diagnostics": {
            **asdict(fine.diagnostics),
            "mean_picard_iterations": fine.diagnostics.mean_picard_iterations,
        },
        "spatial": {
            "coarse_to_medium": spatial_1,
            "medium_to_fine": spatial_2,
            "richardson_field": spatial_field,
            "richardson_drying_time": spatial_time,
        },
        "temporal": {
            "coarse_to_medium": temporal_1,
            "medium_to_fine": temporal_2,
            "richardson_field": temporal_field,
            "richardson_drying_time": temporal_time,
        },
        "combined_field_error_bound_kg_kg": combined_field_bound,
        "combined_drying_time_error_bound_s": combined_time_bound_s,
        "excel_validation": excel,
    }
    atomic_json(temporary / "run_metadata.json", validation)
    lines = [
        "# 第三问超精细计算验证",
        "",
        "- 4 h 后环境：50 °C、0.05 kg/kg；",
        f"- 主网格：N=5120，均匀空间步长 {0.02 / 5120:.10e} m；",
        "- 主时间步：0.25 s；",
        f"- 连续临界时刻：{fine.drying_time_s / 3600.0:.12f} h；",
        f"- 全部 60 s 输出的空间误差安全上界：{spatial_field['estimated_fine_error_with_1p25_safety']:.6e} kg/kg；",
        f"- 全部 60 s 输出的时间误差安全上界：{temporal_field['estimated_fine_error_with_1p25_safety']:.6e} kg/kg；",
        f"- 合计场值误差安全上界：{combined_field_bound:.6e} kg/kg；",
        f"- 烘干时间合计误差安全上界：{combined_time_bound_s:.6f} s；",
        f"- Excel：{excel['dimension']}，ZIP={excel['zip_integrity']}。",
        "",
        "空间与时间序列分别加密，未把两类误差混在同一收敛阶中。",
    ]
    (temporary / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.rename(temporary, Q3_FINAL)
    return validation


def build_q4_final(results: dict[str, Any], environment: q4.EnvironmentData, radius: q4.RadiusData) -> dict[str, Any]:
    if Q4_FINAL.is_dir():
        metadata = Q4_FINAL / "run_metadata.json"
        if not metadata.is_file() or not (Q4_FINAL / "result4.xlsx").is_file():
            raise RuntimeError("existing Q4 final directory is incomplete")
        validation = json.loads(metadata.read_text(encoding="utf-8"))
        if validation.get("boundary_after_4h") != {
            "temperature_c": 50.0,
            "moisture_kg_kg": 0.05,
        }:
            raise RuntimeError("existing Q4 final directory has the wrong boundary")
        print("REUSE problem4_final", flush=True)
        return validation
    fine = results["q4_n12160_fine"]
    spatial = q4.spatial_convergence(
        results["q4_n3040_fine"], results["q4_n6080_fine"], fine
    )
    temporal_1 = q4.common_field_comparison(
        results["q4_n12160_time_coarse"], results["q4_n12160_time_medium"]
    )
    temporal_2 = q4.common_field_comparison(results["q4_n12160_time_medium"], fine)
    temporal_richardson = richardson(temporal_1, temporal_2, "max_abs_difference")
    temporal_time = richardson(temporal_1, temporal_2, "drying_crossing_difference_s")
    numerical_bound = (
        float(spatial["estimated_fine_grid_max_error_with_1p25_safety"])
        + temporal_richardson["estimated_fine_error_with_1p25_safety"]
    )
    safe_end_s = q4.choose_safe_end_time(
        fine, numerical_bound, q4.PhysicalParameters().drying_threshold_kg_kg
    )
    density_consistency = q4.density_shrinkage_consistency_at_time(
        fine, q4.PhysicalParameters(), safe_end_s
    )

    temporary = RUN_ROOT / f".problem4_final.tmp.{os.getpid()}"
    if temporary.exists() or Q4_FINAL.exists():
        raise RuntimeError("Q4 final output directory already exists without completion marker")
    temporary.mkdir(parents=True)
    result_xlsx = temporary / "result4.xlsx"
    q4.write_result_xlsx(ROOT / "A题/附件/附件3/result4.xlsx", result_xlsx, fine, safe_end_s)
    excel = q4.validate_result_xlsx(result_xlsx, fine, safe_end_s)
    q4.write_full_precision_csv(
        temporary / "moisture_physical_full_precision.csv", fine, safe_end_s, False
    )
    q4.write_full_precision_csv(
        temporary / "moisture_material_coordinates_full_precision.csv", fine, safe_end_s, True
    )
    q4.write_summary_markdown(
        temporary / "summary_tables.md", fine, fine.drying_crossing_s, safe_end_s, numerical_bound
    )
    q4.write_validation_markdown(
        temporary / "validation_report.md",
        fine,
        safe_end_s,
        numerical_bound,
        spatial,
        temporal_2,
        excel,
        environment,
        radius,
        density_consistency,
    )
    validation = {
        "boundary_after_4h": {"temperature_c": 50.0, "moisture_kg_kg": 0.05},
        "fine_config": asdict(fine.config),
        "drying_crossing_s": fine.drying_crossing_s,
        "drying_crossing_h": fine.drying_crossing_s / 3600.0,
        "safe_end_s": safe_end_s,
        "safe_end_h": safe_end_s / 3600.0,
        "fine_runtime_s": fine.diagnostics.runtime_s,
        "fine_diagnostics": asdict(fine.diagnostics),
        "spatial": spatial,
        "temporal": {
            "coarse_to_medium": temporal_1,
            "medium_to_fine": temporal_2,
            "richardson_field": temporal_richardson,
            "richardson_drying_time": temporal_time,
        },
        "combined_field_error_bound_kg_kg": numerical_bound,
        "density_shrinkage_consistency_if_rho_is_wet_bulk": density_consistency,
        "excel_validation": excel,
    }
    atomic_json(temporary / "run_metadata.json", validation)
    os.rename(temporary, Q4_FINAL)
    return validation


def main() -> int:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    TASK_ROOT.mkdir(parents=True, exist_ok=True)
    (RUN_ROOT / "orchestrator.pid").write_text(str(os.getpid()) + "\n", encoding="ascii")
    if verified_complete():
        print("COMPLETE manifest verified; no computation needed", flush=True)
        return 0

    source_paths = [
        ROOT / "problem3/solve_problem3.py",
        ROOT / "problem3/accelerated_problem3.py",
        ROOT / "problem4/solve_problem4.py",
        ROOT / "A题/附件/附件1.xlsx",
        ROOT / "A题/附件/附件2.xlsx",
        ROOT / "A题/附件/附件3/result3.xlsx",
        ROOT / "A题/附件/附件3/result4.xlsx",
    ]
    source_hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths}
    tasks = [
        {"id": "q3_n1280_dt0p25", "problem": 3, "radial_cells": 1280, "time_step_s": 0.25},
        {"id": "q3_n2560_dt0p25", "problem": 3, "radial_cells": 2560, "time_step_s": 0.25},
        {"id": "q3_n5120_dt0p25", "problem": 3, "radial_cells": 5120, "time_step_s": 0.25},
        {"id": "q3_n5120_dt1", "problem": 3, "radial_cells": 5120, "time_step_s": 1.0},
        {"id": "q3_n5120_dt0p5", "problem": 3, "radial_cells": 5120, "time_step_s": 0.5},
        {"id": "q4_n3040_fine", "problem": 4, "radial_cells": 3040, "rtol": 1.0e-12, "initial_max_step_s": 0.25, "drying_max_step_s": 1.25},
        {"id": "q4_n6080_fine", "problem": 4, "radial_cells": 6080, "rtol": 1.0e-12, "initial_max_step_s": 0.25, "drying_max_step_s": 1.25},
        {"id": "q4_n12160_fine", "problem": 4, "radial_cells": 12160, "rtol": 1.0e-12, "initial_max_step_s": 0.25, "drying_max_step_s": 1.25},
        {"id": "q4_n12160_time_coarse", "problem": 4, "radial_cells": 12160, "rtol": 4.0e-12, "initial_max_step_s": 1.0, "drying_max_step_s": 5.0},
        {"id": "q4_n12160_time_medium", "problem": 4, "radial_cells": 12160, "rtol": 2.0e-12, "initial_max_step_s": 0.5, "drying_max_step_s": 2.5},
    ]
    manifest = {
        "schema": 1,
        "boundary_after_4h": {"temperature_c": 50.0, "moisture_kg_kg": 0.05},
        "tasks": tasks,
        "source_sha256": source_hashes,
        "logical_cpus": LOGICAL_CPUS,
        "platform": platform.platform(),
        "python": sys.version,
    }
    if MANIFEST_PATH.exists():
        existing = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        comparable = {k: existing[k] for k in manifest if k not in {"platform", "python"}}
        expected = {k: manifest[k] for k in manifest if k not in {"platform", "python"}}
        if comparable != expected:
            raise RuntimeError("existing ultrafine manifest does not match this run")
    else:
        atomic_json(MANIFEST_PATH, manifest)

    q3_environment = q3.load_environment(
        ROOT / "A题/附件/附件1.xlsx", 3.0, "fixed", 50.0, 0.05
    )
    q4_environment = q4.load_environment(ROOT / "A题/附件/附件1.xlsx")
    q4_environment = q4.replace(
        q4_environment,
        extension_strategy="fixed",
        extension_temperature_c=50.0,
        extension_moisture_kg_kg=0.05,
    )
    q4_environment.validate()
    radius_data = q4.load_radius(ROOT / "A题/附件/附件2.xlsx")

    # Warm the JIT once before forking.  All Q3 workers inherit compiled code;
    # no worker repeats compilation.
    warm_config = q3.NumericalConfig(
        radial_cells=20,
        time_step_s=60.0,
        output_interval_s=60.0,
        maximum_time_h=4.01,
        environment_extension="fixed",
    )
    try:
        solve_model_accelerated(q3_environment, q3.PhysicalParameters(), warm_config, False)
    except RuntimeError as error:
        if "threshold not reached" not in str(error):
            raise

    results: dict[str, Any] = {}
    pending: list[tuple[dict[str, Any], str]] = []
    for task in tasks:
        fingerprint = task_fingerprint(task, source_hashes)
        checkpoint = load_checkpoint(task, fingerprint)
        if checkpoint is None:
            pending.append((task, fingerprint))
        else:
            results[task["id"]] = checkpoint
            print(f"REUSE {task['id']}", flush=True)

    workers = min(len(pending), LOGICAL_CPUS)
    if pending:
        q3_positions = [i for i, (task, _) in enumerate(pending) if task["problem"] == 3]
        q4_positions = [i for i, (task, _) in enumerate(pending) if task["problem"] == 4]
        cpu_groups: list[list[int]] = [[] for _ in pending]
        next_cpu = 0
        # Fixed-step Q3 kernels are sequential in time and use one core each.
        for position in q3_positions:
            cpu_groups[position] = [next_cpu]
            next_cpu += 1
        # Sparse Q4 solves can use threaded numeric libraries.  Split every
        # remaining logical CPU once, without overlap, across Q4 tasks.
        remaining = np.arange(next_cpu, LOGICAL_CPUS)
        for position, group in zip(q4_positions, np.array_split(remaining, len(q4_positions))):
            cpu_groups[position] = [int(x) for x in group]
        # A resumed run containing only Q3 tasks still receives disjoint CPUs.
        for position in q3_positions:
            if not cpu_groups[position]:
                cpu_groups[position] = [position % LOGICAL_CPUS]
        print(f"parallel_workers={workers} logical_cpus={LOGICAL_CPUS}", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_map = {}
            for index, (task, fingerprint) in enumerate(pending):
                future = pool.submit(
                    execute_task,
                    task,
                    fingerprint,
                    q3_environment,
                    q4_environment,
                    radius_data,
                    [int(x) for x in cpu_groups[index]],
                )
                future_map[future] = (task, fingerprint)
            for future in as_completed(future_map):
                task, fingerprint = future_map[future]
                future.result()
                checkpoint = load_checkpoint(task, fingerprint)
                if checkpoint is None:
                    raise RuntimeError(f"missing checkpoint after task: {task['id']}")
                results[task["id"]] = checkpoint

    q3_validation = build_q3_final(results, q3_environment)
    q4_validation = build_q4_final(results, q4_environment, radius_data)
    summary = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "q3": {
            "drying_time_h": q3_validation["drying_time_h"],
            "field_error_bound": q3_validation["combined_field_error_bound_kg_kg"],
            "drying_time_error_bound_s": q3_validation["combined_drying_time_error_bound_s"],
        },
        "q4": {
            "drying_crossing_h": q4_validation["drying_crossing_h"],
            "safe_end_h": q4_validation["safe_end_h"],
            "field_error_bound": q4_validation["combined_field_error_bound_kg_kg"],
        },
    }
    report = [
        "# 超精细并行计算总览",
        "",
        "4 h 后环境固定为 50 °C、0.05 kg/kg。",
        "",
        f"- 第三问连续临界时刻：{summary['q3']['drying_time_h']:.12f} h；",
        f"- 第三问场值误差安全上界：{summary['q3']['field_error_bound']:.6e} kg/kg；",
        f"- 第四问连续临界时刻：{summary['q4']['drying_crossing_h']:.12f} h；",
        f"- 第四问安全提交时长：{summary['q4']['safe_end_h']:.12f} h；",
        f"- 第四问场值误差安全上界：{summary['q4']['field_error_bound']:.6e} kg/kg。",
    ]
    (RUN_ROOT / "SUMMARY.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    final_hashes = {}
    for directory in (Q3_FINAL, Q4_FINAL):
        for relative, digest in artifact_hashes(directory).items():
            final_hashes[str(directory.relative_to(RUN_ROOT) / relative)] = digest
    final_hashes["SUMMARY.md"] = sha256_file(RUN_ROOT / "SUMMARY.md")
    atomic_json(
        COMPLETE_PATH,
        {**summary, "artifact_sha256": final_hashes, "manifest_sha256": sha256_file(MANIFEST_PATH)},
    )
    print("ALL COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
