# A 题第三问 Python 数值实验

本目录使用第二问的固定半径、变物性热湿模型，计算药材所有位置的干基含水率均低于 $0.15\,\mathrm{kg/kg}$ 的时刻，并生成第三问要求的完整结果。

完整的建模假设、控制方程推导、离散格式、终止判据和收敛验证见 [第三问建模与实现](../A题/第三问建模与实现.md)。

## 模型

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

物性采用题目附录 3，且扩散系数中的温度使用

$$
T_K=T_{{}^\circ\mathrm C}+273.15.
$$

附件 1 的实测数据通过分段线性插值使用到 4 h。默认将末 1 h（3–4 h）的温度和环境水分浓度均值延拓到后续恒温干燥阶段。

## 数值实现

- 空间：一维径向守恒有限体积法；
- 时间：首步后向 Euler，后续二阶 BDF2；
- 耦合：每个时间步内对 $T$、$C$ 进行 Picard 迭代；
- 非线性通量：使用四点 Gauss–Legendre 求积计算界面 Kirchhoff/割线平均扩散系数；
- 终止条件：使用未舍入结果检查 $\max_r C(r,t)<0.15$；
- 环境依赖：仅使用 Python 3 标准库，不需要 NumPy、SciPy 或 openpyxl；
- Excel：直接在题目 `result3.xlsx` 模板的 OOXML 结构中写入结果。

界面扩散系数没有使用简单调和平均。附录 3 的 $D(C,T)$ 在干燥表层变化很快，调和平均会产生不合理的数值表层阻力和明显网格依赖。程序采用

$$
\overline D_{i+1/2}
=\int_0^1
D\!\left(C_i+s(C_{i+1}-C_i),T_{i+1/2}\right)\,\mathrm ds
$$

构造界面通量。

## 运行

在项目根目录执行完整实验：

```bash
python3 problem3/solve_problem3.py --progress
```

默认主计算使用 160 个径向区间、5 s 时间步，即

$$
\Delta r=\frac{2\,\mathrm{cm}}{160}=0.0125\,\mathrm{cm},
\qquad
\Delta t=5\,\mathrm{s}.
$$

程序自动执行以下三层收敛实验：

| 层级 | 径向区间数 | 时间步长 |
|---|---:|---:|
| 粗 | 40 | 20 s |
| 中 | 80 | 10 s |
| 细/主结果 | 160 | 5 s |

快速运行而不做收敛实验：

```bash
python3 problem3/solve_problem3.py --skip-convergence
```

使用附件最后一个测量点作为 4 h 后的恒定环境边界：

```bash
python3 problem3/solve_problem3.py \
  --environment-extension last_point \
  --experiment-name last_point_N160_dt5
```

运行单元测试：

```bash
python3 -m unittest problem3/test_problem3.py -v
```

## 输出

默认情况下，每次运行都会创建独立目录：

```text
problem3/experiments/<时间戳_N_dt_边界策略>/
```

已有实验不会被覆盖。`problem3/experiments/index.csv` 记录每次成功实验的参数、烘干时间、目录和结果文件哈希。

自动收敛实验的每个层级也会完整保存在

```text
<实验目录>/convergence_levels/N<空间区间数>_dt<时间步>s/
```

其中包含该层级的完整时间序列、表 5 和运行诊断，不只保留最终的收敛差值。

- `result3.xlsx`：按题目模板生成的提交结果；
- `table5.md`、`table5.csv`：论文表 5；
- `moisture_full_precision.csv`：每隔 60 s、每隔 0.1 cm 的未舍入结果；
- `environment_input_used.csv`：实际读取的附件 1 数据；
- `convergence.csv`：三层网格的烘干时间与剖面比较；
- `validation_report.md`：守恒、迭代、单调性、收敛及 Excel 结构检查；
- `run_metadata.json`：参数、输入哈希、环境延拓和运行环境信息。

所有计算和终止判断均使用未舍入双精度值，只有论文表格和 `result3.xlsx` 中的水分浓度保留四位小数。
