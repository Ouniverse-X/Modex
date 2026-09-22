# Q4: axisymmetric_fem

验证状态：NEEDS_REFINEMENT_OR_REVIEW

本文件是指定数学模型的数值解，不是实测结果；误差判据不是严格上界。

主配置：{"q": 4, "tag": "Q2-64x96-exposed", "nr": 64, "nz": 96, "grade_r": 7.0, "grade_z": 7.0, "end": "exposed", "resolution": 6144, "role": "space", "method": "BDF", "rtol": 2e-09, "atol_T": 1e-10, "atol_C": 1e-11, "max_step": 30.0, "late_max_step": 300.0, "jacobian_mode": "block"}

网格：{"radial_Q2_elements": 64, "axial_Q2_elements_half_length": 96, "nodes_per_field": 24897, "full_coupled_DOF": 49794, "radial_grade_beta": 7.0, "axial_grade_beta": 7.0, "radial_outer_band_fraction": 0.1, "axial_end_band_fraction": 0.25, "elements_allocated_to_each_band_fraction": 0.5, "initial_radial_node_gap_um": [0.22317574234431703, 397.74756441743295], "initial_axial_node_gap_um": [2.2390497574770185, 2122.2205041145776], "domain": "s in [0,1], eta in [0,1] symmetric half-cylinder; Lobatto-integrated Q2, not Q1/FVM"}

物性按题面附录；外界附件完整保留，之后60秒过渡到50°C/0.05 kg/kg。

时间积分 forcing-knot 策略：aligned；critical 不改变分段线性输入值，只避免在每个连续导数折点重启 BDF。

轴向长度默认不收缩；二维 exposed 表示端面沿用侧面换热/传质系数，insulated 为绝热不透湿控制。

忽略潜热、辐射和端面以外的外界反馈；干基含水率不加体积浓度压缩项。

XLSX/中截面CSV在当时实际厘米距离处直接计算谱多项式/Q2形函数；不是对稀疏观测点做二次线性插值。

高精度对照使用 fields.npz 中共同坐标 probes；snapshots 为原始自由度场。

fields.npz 包含 6h 全场剖面、终态和每60秒的共同观测点；没有将高密度插值冒充细网格求解。

端面边界改变的是物理模型，端面对照差异不能叫做原一维算法的离散误差。

事件：{"lower_s": 183925.90255773324, "upper_s": 183925.9026292588, "width_s": 7.152557373046875e-05, "time_h": 51.09052850812745, "C_max_lower": 0.14999999998095428, "C_max_upper": 0.14999999998111194, "method": "All Q2 elements: tensor Bernstein convex-hull enclosure with de Casteljau refinement; float64 roundoff allowance", "meaning": "Numerical-interpolant event bracket, NOT a PDE/model error bound"}

初始均匀含水率与非零表面失水通量不相容；高阶离散可能出现短时小幅超调，程序记录并不裁剪，过大则报错。range_tolerance是安全阈值，不是精度目标。

所有对照见 comparison_report.json。通过仅指共同输出时刻及事件时刻的差异判据，不表示每个瞬间的严格误差。缺少对照、未烘干均不标为通过。
