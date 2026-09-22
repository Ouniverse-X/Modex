# A 题第四问 Python 求解程序

本目录实现考虑半径收缩的圆柱热湿耦合模型。详细推导见
[第四问建模与实现](../A题/第四问建模与实现.md)。

## 安装依赖

```bash
python3 -m pip install -r problem4/requirements.txt
```

## 正式高精度计算

```bash
python3 problem4/solve_problem4.py
```

默认计算包括：

1. 760 个材料坐标区间的主计算；
2. 190、380、760 三层空间网格收敛复算；
3. 760 区间、误差容差缩小 4 倍且最大时间步减半的复算；
4. 全区域干燥临界时刻定位；
5. 数值误差安全裕量与整分钟工艺时长判定；
6. `result4.xlsx` 逐格写入检查；
7. 题面密度若按实际湿体积密度解释时，与附件半径及干质量守恒的一致性诊断。

## 测试

```bash
python3 -m unittest problem4/test_problem4.py -v
```

## 输出

- `result4.xlsx`：按 60 s 间隔保存固定物理距离和移动表面的水分浓度；
- `summary_tables.md`：烘干时长和表 6；
- `validation_report.md`：空间、时间收敛及守恒检查；
- `moisture_physical_full_precision.csv`：实际厘米位置的未舍入结果；
- `moisture_material_coordinates_full_precision.csv`：材料坐标结果；
- `run_metadata.json`：输入哈希、参数、软件版本和诊断信息。

验证报告区分两类检查：材料坐标水分库存—边界通量残差用于验证离散 PDE；`rho/(1+C)` 干质量代理量用于检查“题面密度是实际湿密度”这一附加解释与收缩数据是否相容。后者属于模型结构诊断，不是数值误差。

环境数据超过 4 h 后保持附件 1 末值，半径数据超过 72 h 后保持
1.198 cm。两者均属于长期计算所需的显式延拓假设。
