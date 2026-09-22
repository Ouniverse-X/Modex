# 第二问数值验证报告

## 总体结论

程序采用变物性热湿耦合模型、守恒型径向有限体积空间离散和 SciPy 自适应隐式 BDF 时间积分。主结果使用表面局部加密网格，并通过空间网格加密和同网格时间容差加严进行复核。

## 输入与模型口径

- 附件 1：`/home/beihang/projects/Modex/A题/附件/附件1.xlsx`；SHA-256：`7ef32870abeef420b89560b2530ff60dfe4255917805151d89988d0311af9dd7`；
- 模板：`/home/beihang/projects/Modex/A题/附件/附件3/result2.xlsx`；SHA-256：`23b261b295c1b787d000eebbca6521c37075107b6fcf78724f8d395ce1798ff4`；
- 环境数据行数：241；覆盖 0–14400 s；
- 第二问计算区间：0–10800 s，未对附件 1 作超范围外推；
- 环境数据在相邻 60 s 测点间采用分段线性插值；
- 第二问从初始状态重新计算，全部时段统一采用附录 3 物性公式；
- 题面未另给第二问的表面对流系数，程序明确沿用第一问的 $h=25\,\mathrm{W/(m^2\cdot K)}$ 与 $h_m=8\times10^{-7}\,\mathrm{m/s}$。

## 主计算配置

- 径向区间数：760；节点数：761；
- 最小/最大空间步长：5e-06 / 5e-05 m；
- BDF 相对容差：2.000e-09；
- 温度/含水率绝对容差：2.000e-09 / 2.000e-11；
- 最大内部时间步：5 s；初始时间步：0.0001 s；
- 运行时间：13.537 s；函数计算 16626 次；稀疏 LU 分解 1949 次。

## 数值范围与物性范围

- 温度：28.0000000000–49.9664128832 °C；
- 水分浓度：1.0081226637–2.5500000000 kg/kg；
- 密度：779.03970095–976.40000000 kg/m³；
- 比热容：2823.53342886–3415.29577465 J/(kg·K)；
- 导热系数：0.4007685318–0.4829577465 W/(m·K)；
- 水分扩散系数：5.5473604352e-09–1.2521860312e-08 m²/s；
- 以每秒输出作梯形积分得到的最大水分守恒绝对残差：8.089287e-11；
- 对应最大相对残差：2.010042e-03。该值包含 1 s 输出采样的积分误差，不是 BDF 内部残差。

## 三层空间网格收敛

三层网格分别为 190、380、760 个径向区间；每次加密均将两区网格步长减半。比较范围包括 1–10800 s、0–2 cm 的全部输出点。

| 场变量 | 中网格—细网格最大差 | 观测阶 | 细网格 Richardson 估计误差 | 1.25 安全系数上界 | 最大差位置 |
|---|---:|---:|---:|---:|---|
| 温度/°C | 9.995240e-06 | 1.8790 | 3.732122e-06 | 4.665153e-06 | $t=660\,\mathrm{s}$, $r=2.0\,\mathrm{cm}$ |
| 水分浓度/(kg/kg) | 3.492218e-05 | 1.9999 | 1.164155e-05 | 1.455194e-05 | $t=287\,\mathrm{s}$, $r=1.8\,\mathrm{cm}$ |

## 同网格时间积分复核

在 760 区间网格上，将 BDF 相对/绝对容差进一步缩小 4 倍，并将最大内部时间步由 5 s 缩小为 2.5 s。

| 场变量 | 主计算—加严计算最大差 | RMS 差 | 最大差位置 |
|---|---:|---:|---|
| 温度/°C | 5.360042e-06 | 6.411866e-08 | $t=10380\,\mathrm{s}$, $r=2.0\,\mathrm{cm}$ |
| 水分浓度/(kg/kg) | 4.365391e-08 | 3.752949e-10 | $t=1\,\mathrm{s}$, $r=2.0\,\mathrm{cm}$ |

## Excel 文件检查

- 工作表：温度, 水分浓度；
- 温度工作表：A1:V10801；
- 水分浓度工作表：A1:V10801；
- 每个工作表均为 10801 行、22 列，时间为 1–10800 s，空间为 0–2.0 cm；
- 已逐格核验 453,600 个数值单元格，与内存中求解结果四舍五入至四位小数后的最大差为 0.000e+00；
- Excel 数值写入前四舍五入至四位小数，高精度结果另存 CSV。

## 复现实验配置

```json
{
  "fine_main": {
    "radial_cells": 760,
    "end_time_s": 10800.0,
    "output_interval_s": 1.0,
    "surface_layer_m": 0.002,
    "surface_refinement_factor": 10,
    "relative_tolerance": 2e-09,
    "temperature_absolute_tolerance": 2e-09,
    "moisture_absolute_tolerance": 2e-11,
    "maximum_step_s": 5.0,
    "first_step_s": 0.0001
  },
  "medium": {
    "radial_cells": 380,
    "end_time_s": 10800.0,
    "output_interval_s": 1.0,
    "surface_layer_m": 0.002,
    "surface_refinement_factor": 10,
    "relative_tolerance": 2e-09,
    "temperature_absolute_tolerance": 2e-09,
    "moisture_absolute_tolerance": 2e-11,
    "maximum_step_s": 5.0,
    "first_step_s": 0.0001
  },
  "coarse": {
    "radial_cells": 190,
    "end_time_s": 10800.0,
    "output_interval_s": 1.0,
    "surface_layer_m": 0.002,
    "surface_refinement_factor": 10,
    "relative_tolerance": 2e-09,
    "temperature_absolute_tolerance": 2e-09,
    "moisture_absolute_tolerance": 2e-11,
    "maximum_step_s": 5.0,
    "first_step_s": 0.0001
  },
  "fine_tight_time": {
    "radial_cells": 760,
    "end_time_s": 10800.0,
    "output_interval_s": 1.0,
    "surface_layer_m": 0.002,
    "surface_refinement_factor": 10,
    "relative_tolerance": 5e-10,
    "temperature_absolute_tolerance": 5e-10,
    "moisture_absolute_tolerance": 5e-12,
    "maximum_step_s": 2.5,
    "first_step_s": 5e-05
  }
}
```
