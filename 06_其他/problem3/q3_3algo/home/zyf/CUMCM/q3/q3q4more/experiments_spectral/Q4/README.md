# Q4: spectral

验证状态：OBSERVED_DIFFERENCE_PASS

本文件是指定数学模型的数值解，不是实测结果；误差判据不是严格上界。

主配置：{"q": 4, "tag": "Cheb-p64", "degree": 64, "breaks": [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.96, 0.985, 0.997, 1.0], "resolution": 576, "role": "space", "method": "Radau", "rtol": 2e-09, "atol_T": 1e-10, "atol_C": 1e-11, "max_step": 30.0}

网格：{"degree_per_element": 64, "radial_subdomains_xi": [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.96, 0.985, 0.997, 1.0], "nodes_per_field": 577, "initial_min_radial_gap_um": 0.036082141920967814, "initial_max_radial_gap_um": 98.16491409164865, "note": "Spectral polynomial order and DOF, not comparable to second-order FVM node count"}

物性按题面附录；外界附件完整保留，之后60秒过渡到50°C/0.05 kg/kg。

轴向长度默认不收缩；二维 exposed 表示端面沿用侧面换热/传质系数，insulated 为绝热不透湿控制。

忽略潜热、辐射和端面以外的外界反馈；干基含水率不加体积浓度压缩项。

XLSX/中截面CSV在当时实际厘米距离处直接计算谱多项式/Q2形函数；不是对稀疏观测点做二次线性插值。

高精度对照使用 fields.npz 中共同坐标 probes；snapshots 为原始自由度场。

fields.npz 包含 6h 全场剖面、终态和每60秒的共同观测点；没有将高密度插值冒充细网格求解。

端面边界改变的是物理模型，端面对照差异不能叫做原一维算法的离散误差。

事件：{"lower_s": 183925.97592618153, "upper_s": 183925.975983402, "width_s": 5.7220458984375e-05, "time_h": 51.090548884278334, "C_max_lower": 0.14999999998658164, "C_max_upper": 0.1499999999865898, "method": "All-element Chebyshev derivative-root extrema plus nodal checks; float64, not a certified interval bound", "meaning": "Numerical-interpolant event bracket, NOT a PDE/model error bound"}

初始均匀含水率与非零表面失水通量不相容；高阶离散可能出现短时小幅超调，程序记录并不裁剪，过大则报错。range_tolerance是安全阈值，不是精度目标。

所有对照见 comparison_report.json。通过仅指共同输出时刻及事件时刻的差异判据，不表示每个瞬间的严格误差。缺少对照、未烘干均不标为通过。
