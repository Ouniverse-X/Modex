#!/usr/bin/env python3
"""Independent N=3040 refinement for the true-density comparison model."""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from true_density_nonaffine import NumericalConfig
from run_experiment import (
    ENVIRONMENT_PATH,
    RADIUS_PATH,
    compare_cases,
    load_archived_result,
    make_figures,
    make_workbook,
    observed_order,
    run_case,
    sha256_file,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "experiments" / "true_density_nonaffine_20260911_phase2_ultrafine"
PHASE1 = HERE / "experiments" / "true_density_nonaffine_20260911_phase1"

CASES = {
    "reference_n3040_dt1": NumericalConfig(
        radial_cells=3040,
        time_step_s=1.0,
        output_interval_s=60.0,
        maximum_time_h=60.0,
    ),
    "time_n3040_dt2": NumericalConfig(
        radial_cells=3040,
        time_step_s=2.0,
        output_interval_s=60.0,
        maximum_time_h=60.0,
    ),
    "safety_n3040_dt1": NumericalConfig(
        radial_cells=3040,
        time_step_s=1.0,
        output_interval_s=60.0,
        maximum_time_h=60.0,
        drying_threshold_kg_kg=0.14990,
    ),
}


def write_report(output_dir: Path, metadata: dict) -> None:
    convergence = metadata["convergence"]
    safety = metadata["safety"]
    reference = metadata["cases"]["reference_n3040_dt1"]
    lines = [
        "# 真实湿密度—非仿射收缩超细网格验证",
        "",
        "## 超细结果",
        "",
        f"- 参考网格：$N=3040$，$\\Delta t=1\\,\\mathrm{{s}}$；",
        f"- $\\max C=0.15$ 连续临界时刻：{reference['drying_crossing_s']:.9f} s = {reference['drying_crossing_h']:.12f} h；",
        f"- 反演长度范围：{100.0 * reference['inferred_length_min_m']:.6f}–{100.0 * reference['inferred_length_max_m']:.6f} cm；",
        f"- 临界时刻反演长度：{100.0 * reference['inferred_length_at_crossing_m']:.6f} cm。",
        "",
        "## 加密收敛",
        "",
        "| 比较 | 临界时刻差/s | 末6 h含水率最大差 | 全程含水率最大差 |",
        "|---|---:|---:|---:|",
    ]
    for name, values in convergence["comparisons"].items():
        lines.append(
            f"| {name} | {values['crossing_difference_s']:.9f} | "
            f"{values['moisture_max_abs_last_6h']:.9e} | {values['moisture_max_abs_all']:.9e} |"
        )
    lines.extend([
        "",
        f"空间观测阶：{convergence['orders']['space_moisture_last_6h']:.6f}；",
        f"临界时刻空间观测阶：{convergence['orders']['space_crossing']:.6f}。",
        "",
        "## 四位小数安全检查",
        "",
        f"- 安全复算数值阈值：{safety['numerical_target']:.8f}；",
        f"- 空间与时间误差差值之和乘1.5所得保守界：{safety['field_error_bound']:.12e}；",
        f"- 数值阈值加误差界：{safety['upper_moisture_bound_at_target']:.12f}；",
        f"- 四位小数严格显示界：{safety['display_limit']:.8f}；",
        f"- 是否通过：{safety['passes_four_decimal_guard']}；",
        f"- 建议整分钟安全时刻：{safety['safe_whole_minute_s']:.0f} s = {safety['safe_whole_minute_h']:.12f} h。",
        "",
        "误差界是基于加密差值的保守工程估计，并非数学意义上的严格后验误差上界。",
        "局部干质量守恒则由离散几何恒等式逐环直接保证。",
    ])
    (output_dir / "ultrafine_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(output_dir: Path) -> dict:
    phase1_n760 = load_archived_result(PHASE1 / "space_n760_dt1")
    phase1_n1520 = load_archived_result(PHASE1 / "reference_n1520_dt1")
    reference = load_archived_result(output_dir / "reference_n3040_dt1")
    time_dt2 = load_archived_result(output_dir / "time_n3040_dt2")
    safety_case = load_archived_result(output_dir / "safety_n3040_dt1")

    comparisons = {
        "space_760_vs_1520": compare_cases(phase1_n760, phase1_n1520),
        "space_1520_vs_3040": compare_cases(phase1_n1520, reference),
        "time_2s_vs_1s_at_n3040": compare_cases(time_dt2, reference),
    }
    orders = {
        "space_moisture_last_6h": observed_order(
            comparisons["space_760_vs_1520"]["moisture_max_abs_last_6h"],
            comparisons["space_1520_vs_3040"]["moisture_max_abs_last_6h"],
        ),
        "space_crossing": observed_order(
            comparisons["space_760_vs_1520"]["crossing_difference_s"],
            comparisons["space_1520_vs_3040"]["crossing_difference_s"],
        ),
    }
    convergence = {"comparisons": comparisons, "orders": orders}
    error_bound = 1.5 * (
        comparisons["space_1520_vs_3040"]["moisture_max_abs_last_6h"]
        + comparisons["time_2s_vs_1s_at_n3040"]["moisture_max_abs_last_6h"]
    )
    numerical_target = 0.14990
    upper_bound = numerical_target + error_bound
    crossing_s = float(safety_case["metadata"]["drying_crossing_s"])
    safe_s = math.ceil(crossing_s / 60.0) * 60.0
    safety = {
        "numerical_target": numerical_target,
        "field_error_bound": error_bound,
        "upper_moisture_bound_at_target": upper_bound,
        "display_limit": 0.14995,
        "passes_four_decimal_guard": bool(upper_bound < 0.14995),
        "target_crossing_s": crossing_s,
        "safe_whole_minute_s": safe_s,
        "safe_whole_minute_h": safe_s / 3600.0,
        "qualification": "conservative convergence estimate, not a rigorous a posteriori theorem",
    }
    if not safety["passes_four_decimal_guard"]:
        raise RuntimeError(f"ultrafine safety guard failed: {safety}")
    metadata = {
        "model": "true wet density + measured outer radius + non-affine radial mapping",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase1_source": str(PHASE1.resolve()),
        "inputs": {
            "environment": {"path": str(ENVIRONMENT_PATH), "sha256": sha256_file(ENVIRONMENT_PATH)},
            "radius": {"path": str(RADIUS_PATH), "sha256": sha256_file(RADIUS_PATH)},
            "solver": {"path": str(HERE / 'true_density_nonaffine.py'), "sha256": sha256_file(HERE / 'true_density_nonaffine.py')},
            "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        },
        "cases": {
            name: load_archived_result(output_dir / name)["metadata"] for name in CASES
        },
        "convergence": convergence,
        "safety": safety,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    make_workbook(
        output_dir / "result4-2-ultrafine.xlsx",
        reference,
        safety_case,
        convergence,
        metadata,
    )
    make_figures(output_dir, reference)
    write_report(output_dir, metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output_dir = args.output.resolve()
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    pending = {
        name: config for name, config in CASES.items()
        if not (output_dir / name / "result_full_precision.npz").exists()
    }
    if pending:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(pending))) as pool:
            futures = {
                pool.submit(run_case, name, asdict(config), str(output_dir)): name
                for name, config in pending.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                result = future.result()
                print(
                    f"[{name}] crossing={result['drying_crossing_h']:.12f} h; "
                    f"runtime={result['runtime_s']:.2f} s",
                    flush=True,
                )
    metadata = summarize(output_dir)
    print(json.dumps(metadata["safety"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
