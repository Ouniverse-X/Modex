#!/usr/bin/env python3
"""Create publication-quality SVG/PNG figures from Attachment 1 data."""

from __future__ import annotations

import argparse
import html
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

HERE = Path(__file__).resolve().parent
SOLVE_DIR = HERE.parent / "solve"
if str(SOLVE_DIR) not in sys.path:
    sys.path.insert(0, str(SOLVE_DIR))

from solve_1 import EnvironmentData, load_environment, sha256_file


WIDTH = 1800
HEIGHT = 1250
# ImageMagick registers this family explicitly with both regular and bold faces.
FONT = "Noto Sans CJK SC"
MONO_FONT = "Noto Sans CJK SC"
INK = "#172033"
MUTED = "#667085"
GRID = "#DCE2EA"
AXIS = "#8B95A5"
BLUE = "#2563EB"
BLUE_LIGHT = "#EAF2FF"
ORANGE = "#D97706"
ORANGE_LIGHT = "#FFF4E5"
WHITE = "#FFFFFF"


@dataclass(frozen=True)
class Panel:
    x: float
    y: float
    width: float
    height: float
    y_min: float
    y_max: float

    def map_x(self, value: float, x_min: float, x_max: float) -> float:
        return self.x + (value - x_min) / (x_max - x_min) * self.width

    def map_y(self, value: float) -> float:
        return self.y + self.height - (
            (value - self.y_min) / (self.y_max - self.y_min) * self.height
        )


def text_element(
    x: float,
    y: float,
    text: str,
    size: int,
    *,
    color: str = INK,
    weight: int = 400,
    anchor: str = "start",
    family: str = FONT,
    extra: str = "",
) -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="{family}" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}" '
        f'text-anchor="{anchor}" {extra}>{html.escape(text)}</text>'
    )


def path_from_series(
    times_min: Sequence[float],
    values: Sequence[float],
    panel: Panel,
    x_min: float,
    x_max: float,
) -> str:
    points = [
        (panel.map_x(t, x_min, x_max), panel.map_y(value))
        for t, value in zip(times_min, values)
    ]
    return " ".join(
        ("M" if index == 0 else "L") + f" {x:.3f} {y:.3f}"
        for index, (x, y) in enumerate(points)
    )


def draw_panel(
    pieces: list[str],
    panel: Panel,
    times_min: Sequence[float],
    values: Sequence[float],
    *,
    x_min: float,
    x_max: float,
    x_ticks: Sequence[float],
    y_ticks: Sequence[float],
    y_format: str,
    value_format: str,
    title: str,
    unit: str,
    color: str,
    marker_stride: int,
    highlight_first_30: bool,
    delta_label: str | None,
) -> None:
    pieces.append(
        f'<rect x="{panel.x:.2f}" y="{panel.y:.2f}" width="{panel.width:.2f}" '
        f'height="{panel.height:.2f}" fill="{WHITE}"/>'
    )
    if highlight_first_30:
        start = panel.map_x(0.0, x_min, x_max)
        end = panel.map_x(30.0, x_min, x_max)
        pieces.append(
            f'<rect x="{start:.2f}" y="{panel.y:.2f}" width="{end-start:.2f}" '
            f'height="{panel.height:.2f}" fill="{BLUE_LIGHT}" opacity="0.72"/>'
        )

    for tick in y_ticks:
        y = panel.map_y(tick)
        pieces.append(
            f'<line x1="{panel.x:.2f}" y1="{y:.2f}" '
            f'x2="{panel.x+panel.width:.2f}" y2="{y:.2f}" '
            f'stroke="{GRID}" stroke-width="1.4"/>'
        )
        pieces.append(
            text_element(
                panel.x - 20,
                y + 7,
                format(tick, y_format),
                22,
                color=MUTED,
                anchor="end",
                family=MONO_FONT,
            )
        )

    for tick in x_ticks:
        x = panel.map_x(tick, x_min, x_max)
        pieces.append(
            f'<line x1="{x:.2f}" y1="{panel.y+panel.height:.2f}" '
            f'x2="{x:.2f}" y2="{panel.y+panel.height+8:.2f}" '
            f'stroke="{AXIS}" stroke-width="1.8"/>'
        )
        pieces.append(
            text_element(
                x,
                panel.y + panel.height + 40,
                f"{tick:g}",
                22,
                color=MUTED,
                anchor="middle",
                family=MONO_FONT,
            )
        )

    pieces.extend(
        [
            f'<line x1="{panel.x:.2f}" y1="{panel.y:.2f}" '
            f'x2="{panel.x:.2f}" y2="{panel.y+panel.height:.2f}" '
            f'stroke="{AXIS}" stroke-width="2"/>',
            f'<line x1="{panel.x:.2f}" y1="{panel.y+panel.height:.2f}" '
            f'x2="{panel.x+panel.width:.2f}" y2="{panel.y+panel.height:.2f}" '
            f'stroke="{AXIS}" stroke-width="2"/>',
            text_element(panel.x, panel.y - 28, title, 31, weight=650),
            text_element(
                panel.x + panel.width,
                panel.y - 28,
                unit,
                21,
                color=MUTED,
                anchor="end",
            ),
        ]
    )

    line_path = path_from_series(times_min, values, panel, x_min, x_max)
    pieces.append(
        f'<path d="{line_path}" fill="none" stroke="{color}" '
        f'stroke-width="4.5" stroke-linejoin="round" stroke-linecap="round"/>'
    )
    for index in range(0, len(values), marker_stride):
        x = panel.map_x(times_min[index], x_min, x_max)
        y = panel.map_y(values[index])
        pieces.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{WHITE}" '
            f'stroke="{color}" stroke-width="2.6"/>'
        )

    max_index = max(range(len(values)), key=values.__getitem__)
    max_x = panel.map_x(times_min[max_index], x_min, x_max)
    max_y = panel.map_y(values[max_index])
    pieces.append(
        f'<circle cx="{max_x:.2f}" cy="{max_y:.2f}" r="6" '
        f'fill="{color}" stroke="{WHITE}" stroke-width="2.5"/>'
    )

    end_x = panel.map_x(times_min[-1], x_min, x_max)
    end_y = panel.map_y(values[-1])
    if delta_label is None:
        label_y = max(panel.y + 28, min(panel.y + panel.height - 15, end_y - 15))
        pieces.append(
            text_element(
                end_x - 10,
                label_y,
                format(values[-1], value_format),
                22,
                color=color,
                weight=650,
                anchor="end",
                family=MONO_FONT,
            )
        )
    if delta_label:
        pieces.append(
            f'<rect x="{panel.x+panel.width-315:.2f}" y="{panel.y+18:.2f}" '
            f'width="295" height="52" rx="10" fill="{WHITE}" '
            f'stroke="{color}" stroke-width="1.5" opacity="0.96"/>'
        )
        pieces.append(
            text_element(
                panel.x + panel.width - 36,
                panel.y + 53,
                delta_label,
                22,
                color=color,
                weight=650,
                anchor="end",
                family=MONO_FONT,
            )
        )


def make_figure(
    data: EnvironmentData,
    *,
    first_30_minutes_only: bool,
) -> str:
    if first_30_minutes_only:
        count = 31
        times = [value / 60.0 for value in data.times_s[:count]]
        temperatures = data.temperatures_c[:count]
        moistures = data.moistures_kg_kg[:count]
        x_min, x_max = 0.0, 30.0
        x_ticks = list(range(0, 31, 5))
        temperature_limits = (27.5, 42.5)
        temperature_ticks = list(range(28, 43, 2))
        moisture_limits = (0.019, 0.0345)
        moisture_ticks = [0.020 + 0.002 * i for i in range(8)]
        subtitle = "采样间隔 60 s · 第一问计算区间 0–30 min · 数据源：附件1.xlsx"
        title = "附件 1：前 30 分钟烘房环境数据"
        highlight = False
        marker_stride = 2
        temperature_delta = (
            f"30 min：{temperatures[-1]:.3f} °C  "
            f"(+{temperatures[-1]-temperatures[0]:.3f})"
        )
        moisture_delta = (
            f"30 min：{moistures[-1]:.5f} kg/kg  "
            f"(+{moistures[-1]-moistures[0]:.5f})"
        )
    else:
        times = [value / 60.0 for value in data.times_s]
        temperatures = data.temperatures_c
        moistures = data.moistures_kg_kg
        x_min, x_max = 0.0, 240.0
        x_ticks = list(range(0, 241, 30))
        temperature_limits = (27.0, 51.5)
        temperature_ticks = [28, 32, 36, 40, 44, 48, 52]
        moisture_limits = (0.018, 0.052)
        moisture_ticks = [0.020 + 0.005 * i for i in range(7)]
        subtitle = "采样间隔 60 s · 全过程 0–240 min · 淡蓝区域为第一问 0–30 min"
        title = "附件 1：烘房环境数据随时间变化"
        highlight = True
        marker_stride = 15
        temperature_delta = None
        moisture_delta = None

    pieces = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}">',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#FAFBFD"/>',
        text_element(120, 82, title, 44, weight=700),
        text_element(120, 126, subtitle, 23, color=MUTED),
    ]

    top = Panel(145, 205, 1550, 385, *temperature_limits)
    bottom = Panel(145, 745, 1550, 385, *moisture_limits)
    draw_panel(
        pieces,
        top,
        times,
        temperatures,
        x_min=x_min,
        x_max=x_max,
        x_ticks=x_ticks,
        y_ticks=temperature_ticks,
        y_format=".0f",
        value_format=".3f",
        title="烘房温度",
        unit="温度 / °C",
        color=BLUE,
        marker_stride=marker_stride,
        highlight_first_30=highlight,
        delta_label=temperature_delta,
    )
    draw_panel(
        pieces,
        bottom,
        times,
        moistures,
        x_min=x_min,
        x_max=x_max,
        x_ticks=x_ticks,
        y_ticks=moisture_ticks,
        y_format=".3f",
        value_format=".5f",
        title="环境水分浓度",
        unit="水分浓度 / (kg/kg)",
        color=ORANGE,
        marker_stride=marker_stride,
        highlight_first_30=highlight,
        delta_label=moisture_delta,
    )
    pieces.append(text_element(920, 1203, "时间 / min", 25, color=MUTED, anchor="middle"))
    pieces.append(
        text_element(
            1695,
            1210,
            "源数据：A题/附件/附件1.xlsx",
            18,
            color=MUTED,
            anchor="end",
        )
    )
    if highlight:
        highlight_x = top.map_x(15.0, x_min, x_max)
        pieces.append(
            text_element(
                highlight_x,
                top.y + 34,
                "第一问区间",
                19,
                color=BLUE,
                weight=650,
                anchor="middle",
            )
        )
    pieces.append("</svg>")
    return "\n".join(pieces)


def export_png(svg_path: Path, png_path: Path) -> bool:
    converter = shutil.which("convert")
    if converter is None:
        return False
    subprocess.run(
        [
            converter,
            "-density",
            "192",
            str(svg_path),
            "-resize",
            "3600x2500",
            "-background",
            WHITE,
            "-alpha",
            "remove",
            "-alpha",
            "off",
            str(png_path),
        ],
        check=True,
    )
    return True


def validate_data(data: EnvironmentData) -> dict[str, object]:
    intervals = [b - a for a, b in zip(data.times_s, data.times_s[1:])]
    if len(data.times_s) != 241:
        raise ValueError(f"expected 241 rows, got {len(data.times_s)}")
    if any(abs(value - 60.0) > 1.0e-12 for value in intervals):
        raise ValueError("time samples are not uniformly spaced at 60 s")
    if not all(
        math.isfinite(value)
        for values in (data.times_s, data.temperatures_c, data.moistures_kg_kg)
        for value in values
    ):
        raise ValueError("input contains non-finite values")
    temperature_max_index = max(
        range(len(data.temperatures_c)), key=data.temperatures_c.__getitem__
    )
    moisture_max_index = max(
        range(len(data.moistures_kg_kg)), key=data.moistures_kg_kg.__getitem__
    )
    return {
        "row_count": len(data.times_s),
        "time_range_s": [data.times_s[0], data.times_s[-1]],
        "sampling_interval_s": intervals[0],
        "temperature_c": {
            "start": data.temperatures_c[0],
            "at_30_min": data.temperatures_c[30],
            "end": data.temperatures_c[-1],
            "minimum": min(data.temperatures_c),
            "maximum": max(data.temperatures_c),
            "maximum_time_min": data.times_s[temperature_max_index] / 60.0,
        },
        "moisture_kg_kg": {
            "start": data.moistures_kg_kg[0],
            "at_30_min": data.moistures_kg_kg[30],
            "end": data.moistures_kg_kg[-1],
            "minimum": min(data.moistures_kg_kg),
            "maximum": max(data.moistures_kg_kg),
            "maximum_time_min": data.times_s[moisture_max_index] / 60.0,
        },
    }


def parse_arguments() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "A题" / "附件" / "附件1.xlsx",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    args.outdir.mkdir(parents=True, exist_ok=True)
    data = load_environment(args.input, 14400.0)
    summary = {
        "source": {
            "path": str(args.input.resolve()),
            "sha256": sha256_file(args.input),
        },
        **validate_data(data),
    }
    outputs = []
    for stem, zoom in (("attachment1_full_process", False),):
        svg_path = args.outdir / f"{stem}.svg"
        png_path = args.outdir / f"{stem}.png"
        svg_path.write_text(make_figure(data, first_30_minutes_only=zoom), encoding="utf-8")
        png_written = export_png(svg_path, png_path)
        outputs.append(
            {"svg": str(svg_path.resolve()), "png": str(png_path.resolve()) if png_written else None}
        )
    summary["outputs"] = outputs
    (args.outdir / "attachment1_figure_metadata.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
