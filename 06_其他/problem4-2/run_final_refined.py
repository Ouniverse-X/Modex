#!/usr/bin/env python3
"""Final Q4-2 suite with symmetry-axis point reconstruction."""

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
DEFAULT_OUTPUT = (
    HERE / "experiments" / "true_density_nonaffine_20260911_phase3_center_reconstructed"
)

CASES = {
    "space_n760_dt1": NumericalConfig(
        radial_cells=760, time_step_s=1.0, output_interval_s=60.0,
        maximum_time_h=60.0,
    ),
    "space_n1520_dt1": NumericalConfig(
        radial_cells=1520, time_step_s=1.0, output_interval_s=60.0,
        maximum_time_h=60.0,
    ),
    "reference_n3040_dt1": NumericalConfig(
        radial_cells=3040, time_step_s=1.0, output_interval_s=60.0,
        maximum_time_h=60.0,
    ),
    "time_n3040_dt2": NumericalConfig(
        radial_cells=3040, time_step_s=2.0, output_interval_s=60.0,
        maximum_time_h=60.0,
    ),
    "safety_n3040_dt1": NumericalConfig(
        radial_cells=3040, time_step_s=1.0, output_interval_s=60.0,
        maximum_time_h=60.0, drying_threshold_kg_kg=0.14994,
    ),
}


def write_report(output_dir: Path, metadata: dict) -> None:
    conv = metadata["convergence"]
    safety = metadata["safety"]
    ref = metadata["cases"]["reference_n3040_dt1"]
    lines = [
        "# Q4-2 圆心重构后最终超细验证",
        "",
        "## 最终结果",
        "",
        f"- 参考离散：$N=3040$，$\\Delta t=1\\,\\mathrm{{s}}$；",
        f"- $\\max C=0.15$ 连续临界时刻：{ref['drying_crossing_s']:.12f} s = {ref['drying_crossing_h']:.12f} h；",
        f"- 反演长度范围：{100.0 * ref['inferred_length_min_m']:.6f}–{100.0 * ref['inferred_length_max_m']:.6f} cm；",
        f"- 临界时刻反演长度：{100.0 * ref['inferred_length_at_crossing_m']:.6f} cm。",
        "",
        "## 收敛结果",
        "",
        "| 比较 | 临界时刻差/s | 末6 h含水率最大差 | 全程含水率最大差 |",
        "|---|---:|---:|---:|",
    ]
    for name, item in conv["comparisons"].items():
        lines.append(
            f"| {name} | {item['crossing_difference_s']:.12f} | "
            f"{item['moisture_max_abs_last_6h']:.12e} | {item['moisture_max_abs_all']:.12e} |"
        )
    lines.extend([
        "",
        f"空间观测阶（末6 h含水率）：{conv['orders']['space_moisture_last_6h']:.8f}；",
        f"空间观测阶（临界时刻）：{conv['orders']['space_crossing']:.8f}。",
        "",
        "## 四位小数安全裕量",
        "",
        f"- 安全复算阈值：{safety['numerical_target']:.8f}；",
        f"- 最细空间差、时间差之和乘1.5：{safety['field_error_bound']:.12e}；",
        f"- 阈值加误差界：{safety['upper_moisture_bound_at_target']:.12f}；",
        f"- 判定界：{safety['display_limit']:.8f}；",
        f"- 四位小数裕量检查：{safety['passes_four_decimal_guard']}；",
        f"- 建议整分钟安全时刻：{safety['safe_whole_minute_s']:.0f} s = {safety['safe_whole_minute_h']:.12f} h。",
        "",
        "有限体积未知量是单元平均值。最终实现依据圆心光滑轴对称场对 $r$ 为偶函数、"
        "且 $\\mu\\sim r^2$，用前两个单元向 $\\mu=0$ 线性外推圆心点值，"
        "避免把首单元平均值误当作点值。逐环干质量守恒仍由几何恒等式严格保持。",
        "",
        "误差界是加密差值给出的保守工程估计，不是严格数学后验误差定理。",
    ])
    (output_dir / "final_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(output_dir: Path) -> dict:
    results = {name: load_archived_result(output_dir / name) for name in CASES}
    comparisons = {
        "space_760_vs_1520": compare_cases(
            results["space_n760_dt1"], results["space_n1520_dt1"]
        ),
        "space_1520_vs_3040": compare_cases(
            results["space_n1520_dt1"], results["reference_n3040_dt1"]
        ),
        "time_2s_vs_1s_at_n3040": compare_cases(
            results["time_n3040_dt2"], results["reference_n3040_dt1"]
        ),
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
    numerical_target = float(
        results["safety_n3040_dt1"]["metadata"]["config"]["drying_threshold_kg_kg"]
    )
    upper_bound = numerical_target + error_bound
    target_crossing = float(
        results["safety_n3040_dt1"]["metadata"]["drying_crossing_s"]
    )
    safe_s = math.ceil(target_crossing / 60.0) * 60.0
    safety = {
        "numerical_target": numerical_target,
        "field_error_bound": error_bound,
        "upper_moisture_bound_at_target": upper_bound,
        "display_limit": 0.14995,
        "passes_four_decimal_guard": bool(upper_bound < 0.14995),
        "target_crossing_s": target_crossing,
        "safe_whole_minute_s": safe_s,
        "safe_whole_minute_h": safe_s / 3600.0,
        "qualification": "conservative convergence estimate, not a rigorous a posteriori theorem",
    }
    if not safety["passes_four_decimal_guard"]:
        raise RuntimeError(f"four-decimal safety guard failed: {safety}")
    metadata = {
        "model": "true wet density + measured radius + non-affine mapping + center reconstruction",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "environment": {"path": str(ENVIRONMENT_PATH), "sha256": sha256_file(ENVIRONMENT_PATH)},
            "radius": {"path": str(RADIUS_PATH), "sha256": sha256_file(RADIUS_PATH)},
            "solver": {"path": str(HERE / 'true_density_nonaffine.py'), "sha256": sha256_file(HERE / 'true_density_nonaffine.py')},
            "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        },
        "cases": {name: result["metadata"] for name, result in results.items()},
        "convergence": convergence,
        "safety": safety,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    make_workbook(
        output_dir / "result4-2-final.xlsx",
        results["reference_n3040_dt1"],
        results["safety_n3040_dt1"],
        convergence,
        metadata,
    )
    make_figures(output_dir, results["reference_n3040_dt1"])
    write_report(output_dir, metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case", action="append", choices=tuple(CASES))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = args.case or []
    existing = [name for name in requested if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing cases: {existing}")
    if requested:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(requested))) as pool:
            futures = {
                pool.submit(run_case, name, asdict(CASES[name]), str(output_dir)): name
                for name in requested
            }
            for future in as_completed(futures):
                name = futures[future]
                item = future.result()
                print(
                    f"[{name}] crossing={item['drying_crossing_h']:.12f} h; "
                    f"runtime={item['runtime_s']:.2f} s",
                    flush=True,
                )
    if args.summarize:
        missing = [name for name in CASES if not (output_dir / name).exists()]
        if missing:
            raise FileNotFoundError(f"cannot summarize; missing cases: {missing}")
        metadata = summarize(output_dir)
        print(json.dumps(metadata["safety"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
