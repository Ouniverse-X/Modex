# problem4-2：真实湿密度—非仿射收缩对照模型

本目录不修改或覆盖 `problem4/` 的原第四问结果。对照模型采用：

- 附录4密度为真实局部湿体积密度；
- 附件2半径为真实外边界；
- 累计干质量坐标逐环保证干固体质量守恒；
- 不使用均匀仿射缩放；
- 圆柱长度 \(L(t)\) 由守恒相容条件反演。

完整公式见 [模型推导与数值实现.md](./模型推导与数值实现.md)。

## 文件

- `true_density_nonaffine.py`：Numba 加速有限体积求解器；
- `test_true_density_nonaffine.py`：物性、网格、初始几何、非仿射映射和局部干质量守恒测试；
- `run_experiment.py`：一期 \(N=380/760/1520\)、\(\Delta t=4/2/1\,\mathrm s\) 收敛实验；
- `run_ultrafine.py`：二期 \(N=3040\) 超细加密和四位小数安全阈值实验；
- `run_final_refined.py`：加入圆心点值重构后的三期最终实验；
- `experiments/`：按实验批次独立存放的结果，不覆盖原模型或其他批次。

## 复算

```bash
python -m unittest discover -s problem4-2 -p 'test_*.py' -v
python problem4-2/run_experiment.py --workers 5
python problem4-2/run_ultrafine.py --workers 3
python problem4-2/run_final_refined.py --case space_n760_dt1 --case space_n1520_dt1 --case reference_n3040_dt1 --case time_n3040_dt2 --case safety_n3040_dt1 --workers 3
python problem4-2/run_final_refined.py --summarize
```

如果某批实验因外部原因中断，只能对同一输出目录显式使用 `--resume`：

```bash
python problem4-2/run_ultrafine.py --resume --workers 3
```

脚本默认拒绝覆盖已经存在的完整实验目录。

## 结果解释

反演长度是两组题面数据与干质量守恒共同给出的相容性输出。若它显著增大，应视为“真实密度 + 实测半径 + 无额外结构变量”闭合关系的物理预警，而不是把它直接宣称为药材真实伸长。

一期、二期结果保留为数值实现审计记录；其中终止判断曾以首单元平均值近似圆心点值，不再作为最终答案。当前最高精度结果位于 `experiments/true_density_nonaffine_20260911_phase3_center_reconstructed/`：

- \(N=3040,\Delta t=1\,\mathrm s\) 连续临界时刻：\(47.876372607560\,\mathrm h\)；
- 通过四位小数误差裕量检查的整分钟安全时长：\(47.916666666667\,\mathrm h\)；
- 最终结果表：`result4-2-final.xlsx`；
- 验证报告：`final_validation.md`；
- 完整双精度数组：各算例子目录内的 `result_full_precision.npz`。
