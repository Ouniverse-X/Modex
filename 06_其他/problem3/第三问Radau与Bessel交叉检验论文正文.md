# 第三问 Radau IIA 与 Bessel 交叉检验

## 1. 交叉检验目的

第三问采用有限体积法离散圆柱形药材内部的热湿耦合方程，并用 BE/BDF2 格式推进至全域含水率达到规定阈值。网格加密和时间步减半能够检验同族数值解是否收敛，但不能完全排除同一离散框架中的系统性误差。为此，本文进一步设置两种彼此独立且作用不同的交叉检验：

1. 使用五阶 Radau IIA 隐式 Runge–Kutta 方法复算第三问完整非线性模型，检验时间积分算法和临界事件定位；
2. 将程序退化为具有 Bessel 解析解的常系数圆柱扩散问题，检验圆柱空间算子、中心条件和表面 Robin 条件。

两项检验的逻辑关系可概括为

$$
\boxed{
\begin{aligned}
\text{Radau IIA}
&:\ \text{验证完整非线性问题的时间推进},\\
\text{Bessel 解析解}
&:\ \text{验证常系数圆柱扩散程序内核}.
\end{aligned}
}
$$

前者属于异构算法对照，后者属于解析基准对照，二者共同构成对第三问数值解的交叉证据。

## 2. 验收基准

第三问主计算采用 \(N=2560\) 个径向区间和固定时间步长

$$
\Delta t=0.5\,\mathrm s.
$$

独立的空间、时间加密试验给出的含水率场综合误差安全上界为

$$
\boxed{
\varepsilon_C^{(3)}
=
3.813662\times10^{-6}\,\mathrm{kg/kg}
},
$$

临界时刻综合误差安全上界为

$$
\boxed{
\varepsilon_t^{(3)}
=
0.0141598\,\mathrm s
}.
$$

因此，若异构算法在相同空间网格上的差异满足

$$
\Delta_C
\le
\varepsilon_C^{(3)},
\qquad
\Delta_t
\le
\varepsilon_t^{(3)},
$$

即可认为异构复算结果与既有离散误差预算相容。

## 3. Radau IIA 异构时间积分复算

### 3.1 半离散方程

空间有限体积离散后，将全部节点的温度和含水率组成状态向量

$$
\boldsymbol y
=
\left(
T_0,\ldots,T_N,
C_0,\ldots,C_N
\right)^{\mathsf T}.
$$

原偏微分方程组转化为刚性常微分方程组

$$
\boxed{
\frac{\mathrm d\boldsymbol y}{\mathrm dt}
=
\boldsymbol F(t,\boldsymbol y)
}.
$$

第三问主程序第一步采用后向 Euler 格式，后续采用 BDF2：

$$
\frac{
3\boldsymbol y^{n+1}
-4\boldsymbol y^n
+\boldsymbol y^{n-1}
}{
2\Delta t
}
=
\boldsymbol F
\left(
t_{n+1},\boldsymbol y^{n+1}
\right).
$$

交叉复算不再使用多步历史值，而采用三阶段五阶 Radau IIA 配置法。其配置节点为

$$
c_1=\frac{4-\sqrt6}{10},
\qquad
c_2=\frac{4+\sqrt6}{10},
\qquad
c_3=1.
$$

每个时间步同时求解三个隐式阶段：

$$
\boldsymbol Y_i
=
\boldsymbol y_n
+\Delta t
\sum_{j=1}^{3}
a_{ij}
\boldsymbol F
\left(
t_n+c_j\Delta t,
\boldsymbol Y_j
\right),
\qquad i=1,2,3.
$$

三阶段 Radau IIA 的系数矩阵为

$$
\boldsymbol A
=
\begin{bmatrix}
\dfrac{88-7\sqrt6}{360}
&
\dfrac{296-169\sqrt6}{1800}
&
\dfrac{-2+3\sqrt6}{225}
\\[6pt]
\dfrac{296+169\sqrt6}{1800}
&
\dfrac{88+7\sqrt6}{360}
&
\dfrac{-2-3\sqrt6}{225}
\\[6pt]
\dfrac{16-\sqrt6}{36}
&
\dfrac{16+\sqrt6}{36}
&
\dfrac19
\end{bmatrix}.
$$

该方法具有刚性准确性（stiffly accurate），更新值等于最后一个阶段：

$$
\boldsymbol y_{n+1}
=
\boldsymbol Y_3.
$$

Radau IIA 属于隐式 Runge–Kutta 方法，与主程序使用的 BDF2 在方法结构和局部误差估计方式上均不同，因此适合用于排查由单一时间积分算法造成的共同偏差。

### 3.2 复算配置

Radau 复算保持以下内容与主计算完全一致：

- 空间网格 \(N=2560\)；
- 圆柱有限体积空间离散；
- 附录 3 的状态相关物性；
- 附件 1 的环境数据及其分段线性插值；
- 第四小时后的长期环境边界；
- 全域最大含水率事件函数；
- 用于比较的时间和空间采样点。

Radau 的误差控制参数为

$$
\mathrm{rtol}=2\times10^{-10},
$$

$$
\mathrm{atol}_T=2\times10^{-10},
\qquad
\mathrm{atol}_C=2\times10^{-12}.
$$

附件数据覆盖阶段的最大内部时间步限制为 \(5\,\mathrm s\)，长期干燥阶段限制为 \(30\,\mathrm s\)。实际计算共进行 \(106643\) 次右端函数计算、\(94\) 次 Jacobian 计算和 \(24678\) 次 LU 分解。

### 3.3 场值误差

在全部公共输出点上定义含水率最大差

$$
\Delta_C
=
\max_{i,n}
\left|
C_{i,n}^{\mathrm{BDF2}}
-C_{i,n}^{\mathrm{Radau}}
\right|.
$$

实际复算得到

$$
\boxed{
\Delta_C
=
1.020126\times10^{-6}\,\mathrm{kg/kg}
}.
$$

最大差异出现在

$$
t=60\,\mathrm s,
\qquad
r=2.0\,\mathrm{cm},
$$

即初始表面边界层位置。该差异满足

$$
1.020126\times10^{-6}
<
3.813662\times10^{-6},
$$

因而处于独立网格与时间收敛试验给出的综合误差界限内。

在 \(72408\) 个公共含水率数据点中，BDF2 与 Radau 有 \(72402\) 个点在保留四位小数后完全一致，一致率为

$$
\frac{72402}{72408}
\times100\%
=
99.9917\%.
$$

其余 6 个点位于四舍五入分界值附近，未舍入数据的差异仍满足上述误差判据。

### 3.4 临界时刻误差

第三问的连续事件函数为

$$
g(t)
=
\max_{0\le r\le R}C(r,t)-0.15.
$$

两种算法均以同一个连续事件函数的零点作为临界时刻，避免把 \(60\,\mathrm s\) 的文件输出间隔混入事件定位误差。BDF2 主计算得到

$$
t_*^{\mathrm{BDF2}}
=
206900.2913398705\,\mathrm s,
$$

Radau IIA 得到

$$
t_*^{\mathrm{Radau}}
=
206900.2908076993\,\mathrm s.
$$

两者差异为

$$
\boxed{
\Delta_t
=
\left|
t_*^{\mathrm{BDF2}}
-t_*^{\mathrm{Radau}}
\right|
=
5.321713\times10^{-4}\,\mathrm s
}.
$$

该差异仅占综合事件误差安全上界的约 \(3.76\%\)，并满足

$$
5.321713\times10^{-4}
<
0.0141598\,\mathrm s.
$$

因此，第三问临界时刻不依赖于某一种特定的时间积分方法，BDF2 主计算通过了 Radau IIA 异构算法复算。

## 4. Bessel 解析解基准检验

### 4.1 常系数圆柱基准问题

Radau 复算能够检验完整非线性时间推进的一致性，但 BDF2 与 Radau 仍共享同一套圆柱空间离散。为检验空间算子本身，将温度或含水率统一记为 \(u(r,t)\)，冻结物性并令环境值保持常数，得到

$$
S\frac{\partial u}{\partial t}
=
\frac{1}{r}
\frac{\partial}{\partial r}
\left(
r\Gamma\frac{\partial u}{\partial r}
\right),
\qquad
0<r<R.
$$

其中 \(S>0\) 为储存系数，\(\Gamma>0\) 为传递系数。定义

$$
a=\frac{\Gamma}{S}.
$$

圆心和表面边界分别为

$$
\left.
\frac{\partial u}{\partial r}
\right|_{r=0}
=0,
$$

$$
-\Gamma
\left.
\frac{\partial u}{\partial r}
\right|_{r=R}
=
H\left[u(R,t)-u_\infty\right].
$$

令

$$
\theta(r,t)=u(r,t)-u_\infty
$$

并作变量分离

$$
\theta(r,t)=X(r)G(t).
$$

代入控制方程可得

$$
\frac{1}{r}
\frac{\mathrm d}{\mathrm dr}
\left(
r\frac{\mathrm dX}{\mathrm dr}
\right)
+
\mu^2X
=0,
$$

$$
\frac{\mathrm dG}{\mathrm dt}
+a\mu^2G
=0.
$$

径向方程是零阶 Bessel 方程。排除在圆心发散的第二类 Bessel 函数后，

$$
X(r)=J_0(\mu r).
$$

令

$$
\lambda=\mu R,
\qquad
\operatorname{Bi}=\frac{HR}{\Gamma},
$$

利用

$$
\frac{\mathrm dJ_0(x)}{\mathrm dx}
=
-J_1(x),
$$

表面 Robin 条件化为特征方程

$$
\boxed{
\lambda J_1(\lambda)
-\operatorname{Bi}J_0(\lambda)
=0
}.
$$

取该方程的第一正根 \(\lambda_1\)，可构造单模态精确解

$$
\boxed{
u_{\mathrm{ex}}(r,t)
=
u_\infty
+
A J_0
\left(
\lambda_1\frac rR
\right)
\exp
\left(
-a\frac{\lambda_1^2}{R^2}t
\right)
}.
$$

该解在任意时刻严格满足控制方程、圆心对称条件和表面 Robin 条件，因而可以直接计算数值解的真实误差。

### 4.2 误差与观测阶

在全部输出节点和时刻上定义最大绝对误差

$$
E_N
=
\max_{i,n}
\left|
u_{i,n}^{\mathrm{num}}
-u_{\mathrm{ex}}(r_i,t_n)
\right|.
$$

当径向区间数依次取

$$
N=190,\qquad380,\qquad760
$$

时，后两级观测阶为

$$
p
=
\frac{\ln(E_{380}/E_{760})}{\ln2}.
$$

实际计算结果见表 1。

**表 1　Bessel 解析解基准检验结果**

| 基准场 | 时间算法 | \(E_{190}\) | \(E_{380}\) | \(E_{760}\) | 后两级观测阶 |
|---|---|---:|---:|---:|---:|
| 温度 | BDF | \(7.728715\times10^{-5}\) | \(1.932187\times10^{-5}\) | \(4.830463\times10^{-6}\) | \(2.000001\) |
| 温度 | Radau | \(7.728715\times10^{-5}\) | \(1.932186\times10^{-5}\) | \(4.830469\times10^{-6}\) | \(1.999999\) |
| 含水率 | BDF | \(1.613753\times10^{-5}\) | \(4.035232\times10^{-6}\) | \(1.008864\times10^{-6}\) | \(1.999920\) |
| 含水率 | Radau | \(1.613753\times10^{-5}\) | \(4.035233\times10^{-6}\) | \(1.008867\times10^{-6}\) | \(1.999916\) |

温度基准问题的参数为

$$
\operatorname{Bi}_T
=
1.3888888889,
\qquad
\lambda_{1,T}
=
1.418472518827063.
$$

含水率基准问题的参数为

$$
\operatorname{Bi}_m
=
3.2404045432,
\qquad
\lambda_{1,C}
=
1.822083048275662.
$$

含水率特征方程的数值残差仅为

$$
\left|
\lambda_{1,C}J_1(\lambda_{1,C})
-\operatorname{Bi}_mJ_0(\lambda_{1,C})
\right|
=
2.22\times10^{-16},
$$

处于双精度浮点舍入尺度。

从表 1 可见，每次将 \(N\) 加倍后，最大误差约缩小至原来的四分之一，BDF 和 Radau 的观测阶均稳定接近 2。两种时间算法得到的误差几乎相同，表明此时误差主要由空间离散产生，而不是时间推进误差。因此，第三问程序所使用的圆柱几何权重、内部扩散通量、圆心对称条件和表面 Robin 条件均通过了常系数解析基准检验。

## 5. 交叉检验结论

Radau IIA 对第三问完整非线性模型的复算结果表明：

$$
\Delta_C
=
1.020126\times10^{-6}
<
\varepsilon_C^{(3)},
$$

$$
\Delta_t
=
5.321713\times10^{-4}\,\mathrm s
<
\varepsilon_t^{(3)}.
$$

Bessel 解析基准进一步表明，空间误差随网格加密呈稳定二阶收敛：

$$
p\approx2.
$$

因此，两项相互独立的检验分别从完整非线性时间推进和常系数圆柱空间内核两个层面支持第三问数值结果。第三问连续临界时刻为

$$
\boxed{
t_*
=
206900.2913398705\,\mathrm s
=
57.472303149964\,\mathrm h
}.
$$

论文中考虑有效数字后可报告为

$$
\boxed{
t_*\approx57.4723\,\mathrm h
}.
$$

需要强调的是，Radau 与 Bessel 检验验证的是数值程序对既定方程的求解正确性。Bessel 基准采用固定域常系数模型，不直接验证附录 3 非线性经验公式的物理准确性；上述亚秒级误差也不包括长期环境延拓、一维圆柱近似和边界传递模型等模型形式误差。

## 参考文献

[1] HAIRER E, WANNER G. Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems. 2nd ed. Berlin: Springer, 1996.

[2] CRANK J. The Mathematics of Diffusion. 2nd ed. Oxford: Clarendon Press, 1975.

[3] PATANKAR S V. Numerical Heat Transfer and Fluid Flow. Washington: Hemisphere Publishing Corporation, 1980.

[4] ROACHE P J. Verification and Validation in Computational Science and Engineering. Albuquerque: Hermosa Publishers, 1998.

## 复现文件

- [Radau 实际复算记录](../cross_validation/results/q3_radau.json)
- [Bessel 实际复算记录](../cross_validation/results/bessel.json)
- [交叉检验汇总报告](../cross_validation/results/validation_report.md)
- [交叉检验程序](../cross_validation/run_cross_validation.py)
