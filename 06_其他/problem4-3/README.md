# problem4-3：固定长度自由半径对照实验

本目录实现“附录4真实湿密度 + 逐环干质量守恒 + 长度固定为25 cm”的第四问对照方案。径向变形允许非仿射，附件2半径不参与正向求解，只用于独立验证。

## 文件

- `fixed_length_free_radius.py`：自由边界有限体积求解器；
- `run_experiment.py`：四组网格/时间步实验、归档、Excel和图形生成；
- `test_fixed_length_free_radius.py`：几何闭式解、逐环守恒和短程积分回归测试；
- `模型推导与实现.md`：完整假设、公式推导、出处、离散方法和结果解释；
- `experiments/fixed_length_free_radius_20260911_final/`：正式72小时实验结果。

## 复现

在本目录执行：

```bash
python -m unittest -v test_fixed_length_free_radius.py
python run_experiment.py --workers 4
```

正式脚本默认拒绝覆盖已有实验目录。四组算例已存在时，如只需重新生成汇总文件和图：

```bash
python run_experiment.py --resume --summarize-only
```

运行环境需要 `numpy`、`numba`、`openpyxl` 和 `matplotlib`。求解器复用了仓库 `problem4-2/true_density_nonaffine.py` 中已经验证的三对角线性求解、物性和有限体积面通量原语；Q4-3特有的固定长度几何、自由半径更新和完整时间积分核均位于 `fixed_length_free_radius.py`。正式元数据记录了两份源文件及附件的SHA-256，以保证结果可追溯。

## 正式结果入口

- `experiments/fixed_length_free_radius_20260911_final/result4-3.xlsx`：摘要、附件2逐点比较、60秒预测时序、代表剖面、收敛性和原始输入；
- `experiments/fixed_length_free_radius_20260911_final/结果报告.md`：关键数值结论；
- `experiments/fixed_length_free_radius_20260911_final/figures/radius_attachment2_comparison.png`：实测—预测及残差主图；
- `experiments/fixed_length_free_radius_20260911_final/run_metadata.json`：配置、诊断、哈希和软件环境；
- 各算例的 `result_full_precision.npz`：未舍入全精度数组。

参考解为 \(N=3040\)、\(\Delta t=1\,\mathrm{s}\)，结果输出间隔60 s，积分至72 h。外侧10%累计干质量区域采用10倍加密。
