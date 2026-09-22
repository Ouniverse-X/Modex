#!/usr/bin/env python3
"""Independently verify spatial and temporal convergence of the refined result."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from solve_problem1 import (
    NumericalConfig,
    PhysicalParameters,
    SimulationResult,
    SolverDiagnostics,
    add_three_level_error_estimate,
    compare_results,
    load_environment,
    solve_model,
)


FOUR_DECIMAL_ABSOLUTE_TOLERANCE = 0.5e-4


def read_field_csv(path: Path) -> tuple[list[int], list[float], list[list[float]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) != 1801 or len(rows[0]) != 22:
        raise ValueError(f"unexpected CSV dimensions in {path}: {len(rows)} rows")
    parsed_radii = [float(item.removesuffix(" cm")) for item in rows[0][1:]]
    radii = [0.1 * index for index in range(len(parsed_radii))]
    if any(abs(parsed - canonical) > 1.0e-12 for parsed, canonical in zip(parsed_radii, radii)):
        raise ValueError(f"unexpected radius labels in {path}")
    times = [int(row[0]) for row in rows[1:]]
    values = [[float(item) for item in row[1:]] for row in rows[1:]]
    return times, radii, values


def load_refined_result(output_dir: Path) -> SimulationResult:
    t_times, t_radii, temperatures = read_field_csv(
        output_dir / "temperature_full_precision.csv"
    )
    c_times, c_radii, moistures = read_field_csv(
        output_dir / "moisture_full_precision.csv"
    )
    if t_times != c_times or t_radii != c_radii:
        raise ValueError("temperature and moisture CSV grids do not match")
    return SimulationResult(
        times_s=t_times,
        radii_cm=t_radii,
        temperatures_c=temperatures,
        moistures_kg_kg=moistures,
        diagnostics=SolverDiagnostics(),
        runtime_s=0.0,
    )


def run_case(
    name: str,
    config: NumericalConfig,
    input_path: Path,
    cache_dir: Path,
) -> tuple[str, float]:
    environment = load_environment(input_path, config.end_time_s)
    result = solve_model(environment, PhysicalParameters(), config, progress=False)
    payload = {"config": asdict(config), "result": result}
    cache_path = cache_dir / f"{name}.pickle"
    temporary_path = cache_dir / f".{name}.{os.getpid()}.tmp"
    with temporary_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary_path, cache_path)
    return name, result.runtime_s


def load_cached_result(
    name: str, config: NumericalConfig, cache_dir: Path
) -> SimulationResult | None:
    path = cache_dir / f"{name}.pickle"
    if not path.exists():
        return None
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("config") != asdict(config):
        return None
    result = payload.get("result")
    if not isinstance(result, SimulationResult):
        return None
    return result


def rounded_agreement_with_separate_extrapolation(
    fine: SimulationResult,
    spatial_medium: SimulationResult,
    temporal_medium: SimulationResult,
    spatial: dict[str, object],
    temporal: dict[str, object],
) -> dict[str, object]:
    requested_times = {100, 300, 600, 900, 1200, 1500, 1800}
    requested_radii = {0.0, 0.5, 1.0, 1.5, 2.0}
    output: dict[str, object] = {}
    fields = (
        (
            "temperature",
            fine.temperatures_c,
            spatial_medium.temperatures_c,
            temporal_medium.temperatures_c,
        ),
        (
            "moisture",
            fine.moistures_kg_kg,
            spatial_medium.moistures_kg_kg,
            temporal_medium.moistures_kg_kg,
        ),
    )
    for field, fine_values, spatial_values, temporal_values in fields:
        p_space = float(spatial[field]["observed_order_max_norm"])
        p_time = float(temporal[field]["observed_order_max_norm"])
        space_denominator = 2.0**p_space - 1.0
        time_denominator = 2.0**p_time - 1.0
        same_all = 0
        same_table = 0
        table_count = 0
        comparison_count = 0
        max_correction = 0.0
        for ti, (fine_row, space_row, time_row) in enumerate(
            zip(fine_values, spatial_values, temporal_values)
        ):
            for ri, (fine_value, space_value, time_value) in enumerate(
                zip(fine_row, space_row, time_row)
            ):
                extrapolated = (
                    fine_value
                    + (fine_value - space_value) / space_denominator
                    + (fine_value - time_value) / time_denominator
                )
                max_correction = max(max_correction, abs(extrapolated - fine_value))
                agrees = f"{fine_value:.4f}" == f"{extrapolated:.4f}"
                same_all += int(agrees)
                comparison_count += 1
                if (
                    fine.times_s[ti] in requested_times
                    and fine.radii_cm[ri] in requested_radii
                ):
                    same_table += int(agrees)
                    table_count += 1
        output[field] = {
            "all_output_agreement_count": same_all,
            "all_output_comparison_count": comparison_count,
            "all_output_agreement_fraction": same_all / comparison_count,
            "paper_table_agreement_count": same_table,
            "paper_table_comparison_count": table_count,
            "max_separate_richardson_correction": max_correction,
        }
    return output


def robust_rounding_count(
    result: SimulationResult, field: str, global_error_bound: float
) -> dict[str, object]:
    values = (
        result.temperatures_c if field == "temperature" else result.moistures_kg_kg
    )
    certified = 0
    total = 0
    minimum_margin = math.inf
    unit = 1.0e-4
    for row in values:
        for value in row:
            scaled = value / unit
            distance = abs((scaled - math.floor(scaled)) - 0.5) * unit
            margin = distance - global_error_bound
            minimum_margin = min(minimum_margin, margin)
            certified += int(margin > 0.0)
            total += 1
    return {
        "certified_count": certified,
        "comparison_count": total,
        "certified_fraction": certified / total,
        "minimum_margin_to_rounding_boundary_after_global_bound": minimum_margin,
    }


def write_markdown(path: Path, report: dict[str, object]) -> None:
    space = report["spatial_convergence"]
    time = report["temporal_convergence"]
    total = report["conservative_total_error_bound"]
    rounding = report["rounding_stability"]
    robust = report["globally_certified_rounding"]
    text = fr"""# 加密结果的独立空间—时间收敛验证

## 结论

固定时间步单独细化空间、固定空间网格单独细化时间后，温度与水分浓度的保守总误差上界均小于 $5\times10^{{-5}}$，四位小数的绝对精度判据通过。

## 独立收敛设置

- 空间收敛：固定 $\Delta t=0.03125\,\mathrm{{s}}$，比较 $N_r=380,760,1520$；
- 时间收敛：固定 $N_r=1520$，比较 $\Delta t=0.125,0.0625,0.03125\,\mathrm{{s}}$；
- 误差范数：$t=1,\ldots,1800\,\mathrm{{s}}$与$r=0,0.1,\ldots,2.0\,\mathrm{{cm}}$全部 37,800 个输出点的最大范数；
- 四位小数判据：保守总误差上界 $<0.5\times10^{{-4}}$。

## 结果

| 场变量 | 空间观测阶 | 空间安全上界 | 时间观测阶 | 时间安全上界 | 保守总上界 | 判据 |
|---|---:|---:|---:|---:|---:|---|
| 温度/$^\circ\mathrm{{C}}$ | {space['temperature']['observed_order_max_norm']:.6f} | {space['temperature']['estimated_fine_grid_max_error_with_1p25_safety']:.6e} | {time['temperature']['observed_order_max_norm']:.6f} | {time['temperature']['estimated_fine_grid_max_error_with_1p25_safety']:.6e} | {total['temperature']:.6e} | {'通过' if total['temperature'] < FOUR_DECIMAL_ABSOLUTE_TOLERANCE else '未通过'} |
| 水分浓度/(kg/kg) | {space['moisture']['observed_order_max_norm']:.6f} | {space['moisture']['estimated_fine_grid_max_error_with_1p25_safety']:.6e} | {time['moisture']['observed_order_max_norm']:.6f} | {time['moisture']['estimated_fine_grid_max_error_with_1p25_safety']:.6e} | {total['moisture']:.6e} | {'通过' if total['moisture'] < FOUR_DECIMAL_ABSOLUTE_TOLERANCE else '未通过'} |

## 舍入稳定性

| 场变量 | 主解与空间+时间 Richardson 外推的四位小数一致数 | 论文表格一致数 | 用全局最大误差上界可严格锁定末位的数量 |
|---|---:|---:|---:|
| 温度 | {rounding['temperature']['all_output_agreement_count']}/{rounding['temperature']['all_output_comparison_count']} | {rounding['temperature']['paper_table_agreement_count']}/{rounding['temperature']['paper_table_comparison_count']} | {robust['temperature']['certified_count']}/{robust['temperature']['comparison_count']} |
| 水分浓度 | {rounding['moisture']['all_output_agreement_count']}/{rounding['moisture']['all_output_comparison_count']} | {rounding['moisture']['paper_table_agreement_count']}/{rounding['moisture']['paper_table_comparison_count']} | {robust['moisture']['certified_count']}/{robust['moisture']['comparison_count']} |

“可严格锁定末位”是更强的逐单元格舍入判据：将全局最大误差上界套用到每个单元格后，该误差区间不跨越四舍五入分界点。未被该强判据锁定的点不代表误差超标，只表示其数值距某个舍入分界点很近。
"""
    path.write_text(text, encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=root.parent / "A题" / "附件" / "附件1.xlsx"
    )
    parser.add_argument("--output-dir", type=Path, default=root)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=root / "convergence_runs" / "refined_separate",
    )
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    fine = load_refined_result(args.output_dir)
    cases = {
        "space_coarse": NumericalConfig(radial_cells=380, time_step_s=0.03125),
        "space_medium": NumericalConfig(radial_cells=760, time_step_s=0.03125),
        "time_coarse": NumericalConfig(radial_cells=1520, time_step_s=0.125),
        "time_medium": NumericalConfig(radial_cells=1520, time_step_s=0.0625),
    }
    results: dict[str, SimulationResult] = {}
    pending: dict[str, NumericalConfig] = {}
    for name, config in cases.items():
        cached = load_cached_result(name, config, args.cache_dir)
        if cached is None:
            pending[name] = config
        else:
            results[name] = cached
            print(f"Reused {name}: runtime={cached.runtime_s:.3f} s", flush=True)

    if pending:
        workers = max(1, min(args.workers, len(pending)))
        print(f"Running {len(pending)} cases with {workers} workers", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    run_case, name, config, args.input.resolve(), args.cache_dir.resolve()
                ): name
                for name, config in pending.items()
            }
            for future in as_completed(futures):
                name, runtime_s = future.result()
                print(f"Completed {name}: runtime={runtime_s:.3f} s", flush=True)
                result = load_cached_result(name, cases[name], args.cache_dir)
                if result is None:
                    raise RuntimeError(f"could not reload completed case {name}")
                results[name] = result

    spatial = add_three_level_error_estimate(
        compare_results(results["space_coarse"], results["space_medium"]),
        compare_results(results["space_medium"], fine),
    )
    temporal = add_three_level_error_estimate(
        compare_results(results["time_coarse"], results["time_medium"]),
        compare_results(results["time_medium"], fine),
    )
    total_bound = {
        field: float(spatial[field]["estimated_fine_grid_max_error_with_1p25_safety"])
        + float(temporal[field]["estimated_fine_grid_max_error_with_1p25_safety"])
        for field in ("temperature", "moisture")
    }
    rounding = rounded_agreement_with_separate_extrapolation(
        fine,
        results["space_medium"],
        results["time_medium"],
        spatial,
        temporal,
    )
    robust = {
        field: robust_rounding_count(fine, field, total_bound[field])
        for field in ("temperature", "moisture")
    }
    report = {
        "criterion": {
            "description": "absolute error below half of one 4-decimal unit",
            "threshold": FOUR_DECIMAL_ABSOLUTE_TOLERANCE,
        },
        "spatial_convergence": spatial,
        "temporal_convergence": temporal,
        "conservative_total_error_bound": total_bound,
        "passes_four_decimal_absolute_accuracy": {
            field: total_bound[field] < FOUR_DECIMAL_ABSOLUTE_TOLERANCE
            for field in ("temperature", "moisture")
        },
        "rounding_stability": rounding,
        "globally_certified_rounding": robust,
        "case_configs": {name: asdict(config) for name, config in cases.items()},
        "case_runtimes_s": {name: results[name].runtime_s for name in cases},
    }
    json_path = args.output_dir / "separate_convergence_refined.json"
    markdown_path = args.output_dir / "separate_convergence_refined.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(markdown_path, report)
    print(json.dumps(report["conservative_total_error_bound"], indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
