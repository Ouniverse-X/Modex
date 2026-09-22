# A 题第二问 Python 求解程序

本目录实现 A 题第二问的固定半径圆柱热湿耦合模型。程序从 $t=0$ 重新计算，在预热和恒温阶段统一采用题目附录 3 的变物性公式。

## 数值模型

$$
\rho(C)c_p(C)\frac{\partial T}{\partial t}
=\frac{1}{r}\frac{\partial}{\partial r}
\left[rk(C)\frac{\partial T}{\partial r}\right],
$$

$$
\frac{\partial C}{\partial t}
=\frac{1}{r}\frac{\partial}{\partial r}
\left[rD(C,T_K)\frac{\partial C}{\partial r}\right].
$$

- 空间离散：守恒型节点中心径向有限体积法；
- 网格：最外侧 2 mm 使用 10 倍局部加密；
- 时间积分：SciPy 自适应隐式 BDF；
- 耦合：温度和含水率组成同一个方法线 ODE 系统，由 BDF 隐式联立求解；
- 边界：中心对称，表面采用对流换热和对流传质 Robin 边界；
- 环境数据：对附件 1 的 60 s 数据作分段线性插值；
- 温度：扩散系数中的温度严格使用 K；
- 第二问未另给表面传递系数，程序明确沿用第一问的 $h=25\,\mathrm{W/(m^2\cdot K)}$、$h_m=8\times10^{-7}\,\mathrm{m/s}$。

模型推导见 [`../A题/第二问建模与实现.md`](../A题/第二问建模与实现.md)。

## 安装依赖

```bash
python3 -m pip install -r problem2/requirements.txt
```

## 运行

在项目根目录执行完整高精度计算：

```bash
python3 problem2/solve_problem2.py
```

默认执行以下计算：

1. 760 个径向区间的主计算；
2. 380、190 个径向区间的空间收敛计算；
3. 760 区间、时间容差加严 4 倍的复算。

只运行轻量级单元测试：

```bash
python3 -m unittest problem2/test_problem2.py -v
```

跳过收敛复算：

```bash
python3 problem2/solve_problem2.py --skip-convergence --skip-temporal-check
```

## 输出

- `result2.xlsx`：提交用结果文件；
- `summary_tables.md`：论文表 3、表 4；
- `temperature_full_precision.csv`：每秒、每 0.1 cm 的高精度温度；
- `moisture_full_precision.csv`：每秒、每 0.1 cm 的高精度水分浓度；
- `environment_input_used.csv`：实际使用的附件数据；
- `validation_report.md`：空间收敛、时间误差、物性范围和 Excel 结构检查；
- `run_metadata.json`：参数、版本、哈希和诊断信息。

Excel 中的数值按题意写入四位小数；CSV 保留 12 位小数用于复核。
