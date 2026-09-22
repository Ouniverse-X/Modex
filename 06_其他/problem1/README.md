# A 题第一问 Python 求解程序

本目录用于完成 A 题第一问。程序使用题目附件 1 的烘房温度和水分浓度数据，求解固定半径圆柱药材内的一维径向非稳态传热、传质方程。

## 数值方法

- 空间：守恒型径向有限体积法，并在最外侧 2 mm 区域作 10 倍局部加密；
- 时间：首步后向 Euler，后续采用二阶 BDF2；
- 非线性：水分扩散系数 $D(C)$ 使用 Picard 迭代；
- 边界：圆柱中心为对称边界，表面为对流换热/传质 Robin 边界；
- 环境数据：对附件 1 的 60 s 数据作分段线性插值；
- 依赖：仅使用 Python 3 标准库。

模型推导见 [`../A题/第一问解答思路与公式推导.md`](../A题/第一问解答思路与公式推导.md)。

## 运行

在项目根目录执行：

```bash
python3 problem1/solve_problem1.py
```

默认主计算采用 1520 个径向区间和 $0.03125\,\mathrm{s}$ 时间步，并自动使用 760/380 个径向区间及 $0.0625/0.125\,\mathrm{s}$ 时间步进行三层收敛比较。主网格在内部区域的步长为 $2.5\times10^{-5}\,\mathrm{m}$，在最外侧 2 mm 内的步长为 $2.5\times10^{-6}\,\mathrm{m}$。验证报告根据三层结果计算观测收敛阶和 Richardson 剩余误差估计，并将带 1.25 安全系数的误差上界与 $5\times10^{-5}$ 的四位小数精度判据比较。

运行单元测试：

```bash
python3 -m unittest problem1/test_problem1.py -v
```

独立空间—时间收敛验证（默认并行使用最多 4 个 CPU 核心）：

```bash
python3 problem1/verify_refined_convergence.py
```

## 输出文件

- `result1.xlsx`：提交用 Excel 文件；
- `summary_tables.md`：论文表 1、表 2；
- `temperature_full_precision.csv`：完整高精度温度结果；
- `moisture_full_precision.csv`：完整高精度水分浓度结果；
- `environment_input_used.csv`：实际使用的附件 1 数据副本；
- `validation_report.md`：输入、守恒、收敛和 Excel 结构检查；
- `run_metadata.json`：参数、校验指标、文件哈希和运行环境。
- `separate_convergence_refined.json`：空间与时间误差的独立数值校验结果；
- `separate_convergence_refined.md`：独立收敛验证报告。

Excel 中的计算结果使用 `0.0000` 数字格式，写入值按题意保留四位小数；CSV 文件额外保留更高精度，便于复核。
