# Transformer-MPC 车辆轨迹跟踪模型说明

本文档说明 `/Users/dsacbery/Study/code/TRANS` 项目中各模块的数学原理、机理和数据交换方式。重点解释 Transformer 风险预测模块如何与 MPC 控制器耦合，以及离线训练和在线闭环控制阶段的数据如何流动。

## 1. 总体架构

本项目实现的是一个基于 Python 的基础 Transformer-MPC 轨迹跟踪原型。系统不让 Transformer 直接输出前轮转角，而是让 Transformer 输出可解释的风险中间量，再由风险映射层将这些中间量转化为 MPC 权重、约束和参考速度修正。

整体闭环为：

```text
参考轨迹 / 工况参数
    -> 动态自行车车辆模型
    -> 轨迹误差计算
    -> 历史特征窗口 X_k
    -> Transformer 风险预测
    -> 风险限幅 / 滤波 / 映射
    -> LTV-MPC 求解
    -> 前轮转角 delta 与纵向加速度 ax
    -> 车辆模型更新
```

核心思想是：

$$
\text{Transformer}:\ X_k \mapsto \hat r_k
$$

$$
\text{Risk Mapper}:\ \hat r_k \mapsto \left(Q_k,\ R_k,\ \mathcal C_k,\ v_{\text{ref},k}^{\text{adapt}}\right)
$$

$$
\text{MPC}:\ \left(x_k,\ Q_k,\ R_k,\ \mathcal C_k,\ v_{\text{ref},k}^{\text{adapt}}\right)\mapsto \delta_k
$$

其中 Transformer 负责识别车辆动力学时序中的风险趋势，MPC 负责在约束内求解控制量。

## 2. 模块对应关系

| 模块 | 文件 | 作用 |
|---|---|---|
| 参数配置 | `trans_mpc/config.py` | 定义车辆、仿真、MPC、Transformer、训练参数 |
| 车辆模型 | `trans_mpc/vehicle_model.py` | 动态自行车模型与低附着轮胎力饱和 |
| 参考轨迹 | `trans_mpc/reference_path.py` | 标准双移线轨迹、航向角、曲率、参考速度 |
| 误差计算 | `trans_mpc/tracking_error.py` | 横向误差、航向误差、误差变化率、输入特征 |
| 工况管理 | `trans_mpc/scenario_manager.py` | 高附着、低附着突变、噪声、初始偏差等场景 |
| MPC 求解 | `trans_mpc/mpc_solver.py` | LTV-MPC 状态空间、目标函数、约束和 QP 求解 |
| 控制器封装 | `trans_mpc/controllers.py` | PID、Fixed MPC、Rule-risk MPC、Transformer-MPC |
| 风险标签 | `trans_mpc/risk_labels.py` | 离线未来窗口风险标签构造 |
| 风险映射 | `trans_mpc/risk_mapper.py` | 风险限幅、滤波、MPC 参数自适应 |
| Transformer | `trans_mpc/transformer_model.py` | PyTorch Transformer 风险预测器 |
| 数据集 | `trans_mpc/dataset.py` | 历史窗口构造、标准化、训练集划分 |
| 仿真主循环 | `trans_mpc/simulator.py` | 统一闭环仿真与日志记录 |

## 3. 参考轨迹模块

标准双移线轨迹由两个平滑换道函数叠加生成：

$$
y_{\text{ref}}(x)
= w\left[
\frac{1+\tanh\left(\frac{x-x_1}{s}\right)}{2}
-
\frac{1+\tanh\left(\frac{x-x_2}{s}\right)}{2}
\right]
$$

其中：

- $w$ 为目标车道宽度；
- $x_1$ 为第一次换道中心位置；
- $x_2$ 为返回原车道中心位置；
- $s$ 为换道过渡平滑系数。

参考航向角由轨迹一阶导数得到：

$$
\psi_{\text{ref}}(x)=\arctan\left(\frac{dy_{\text{ref}}}{dx}\right)
$$

参考曲率为：

$$
\kappa_{\text{ref}}(x)
=
\frac{y_{\text{ref}}''(x)}
{\left(1+\left(y_{\text{ref}}'(x)\right)^2\right)^{3/2}}
$$

这些量在 `ReferencePath` 中以数组形式存储：

$$
\mathcal P=
\left\{
x_i,\ y_i,\ \psi_{\text{ref},i},\ \kappa_{\text{ref},i},\ v_{\text{ref},i}
\right\}_{i=1}^{N_p}
$$

## 4. 动态自行车车辆模型

车辆状态在项目中表示为：

$$
s=
\left[
X,\ Y,\ \psi,\ v_x,\ v_y,\ r,\ \beta,\ a_y,\ \delta,\ a_x
\right]^T
$$

其中：

- $X,Y$ 为全局坐标；
- $\psi$ 为车辆航向角；
- $v_x,v_y$ 为纵向和横向速度；
- $r$ 为横摆角速度；
- $\beta$ 为质心侧偏角；
- $a_y$ 为横向加速度；
- $\delta$ 为前轮转角；
- $a_x$ 为纵向加速度。

### 4.1 轮胎侧偏角

前后轮侧偏角采用动态自行车模型中的小型车辆侧偏定义：

$$
\alpha_f
=
\delta
-
\arctan
\left(
\frac{v_y+l_f r}{\max(|v_x|, v_{\min})}
\right)
$$

$$
\alpha_r
=
-
\arctan
\left(
\frac{v_y-l_r r}{\max(|v_x|, v_{\min})}
\right)
$$

其中 $l_f,l_r$ 分别为质心到前后轴距离。

### 4.2 线性侧偏刚度与低附着饱和

名义侧向力为：

$$
F_{yf}^{0}=C_f\alpha_f
$$

$$
F_{yr}^{0}=C_r\alpha_r
$$

为了模拟低附着路面，侧向力会被摩擦上限截断。前后轴静态法向载荷为：

$$
F_{zf}=\frac{mgl_r}{l_f+l_r}
$$

$$
F_{zr}=\frac{mgl_f}{l_f+l_r}
$$

饱和后的侧向力为：

$$
F_{yf}
=
\operatorname{clip}
\left(
F_{yf}^{0},
-\mu F_{zf},
\mu F_{zf}
\right)
$$

$$
F_{yr}
=
\operatorname{clip}
\left(
F_{yr}^{0},
-\mu F_{zr},
\mu F_{zr}
\right)
$$

这里的 $\mu$ 只在仿真环境内部用于生成低附着响应，不作为 Transformer 的在线输入。

### 4.3 横向动力学

横向速度和横摆角速度动态为：

$$
\dot v_y
=
\frac{F_{yf}+F_{yr}}{m}
-
v_x r
$$

$$
\dot r
=
\frac{l_fF_{yf}-l_rF_{yr}}{I_z}
$$

横向加速度为：

$$
a_y
=
\frac{F_{yf}+F_{yr}}{m}
$$

车辆全局运动学为：

$$
\dot X
=
v_x\cos\psi
-
v_y\sin\psi
$$

$$
\dot Y
=
v_x\sin\psi
+
v_y\cos\psi
$$

$$
\dot \psi = r
$$

离散更新采用欧拉积分：

$$
s_{k+1}
=
s_k
+
T_s f(s_k,u_k,\mu_k)
$$

其中 $u_k=[\delta_k,a_{x,k}]^T$。

质心侧偏角为：

$$
\beta_k
=
\arctan
\left(
\frac{v_{y,k}}
{\max(|v_{x,k}|, v_{\min})}
\right)
$$

## 5. 轨迹误差计算模块

仿真时首先寻找车辆当前位置到参考轨迹的最近点：

$$
i^\*
=
\arg\min_i
\left[
(X-X_{\text{ref},i})^2
+
(Y-Y_{\text{ref},i})^2
\right]
$$

记最近点参考航向为 $\psi_{\text{ref}}$。位置误差投影到 Frenet 法向得到横向误差：

$$
e_y
=
-\sin\psi_{\text{ref}}(X-X_{\text{ref}})
+
\cos\psi_{\text{ref}}(Y-Y_{\text{ref}})
$$

航向误差为：

$$
e_\psi
=
\operatorname{wrap}(\psi-\psi_{\text{ref}})
$$

误差变化率由离散差分得到：

$$
\dot e_y(k)
=
\frac{e_y(k)-e_y(k-1)}{T_s}
$$

$$
\dot e_\psi(k)
=
\frac{
\operatorname{wrap}
\left(
e_\psi(k)-e_\psi(k-1)
\right)
}{T_s}
$$

## 6. Transformer 输入特征与数据窗口

单个时刻的在线可用特征为：

$$
z_k=
\left[
v_x,\ v_y,\ r,\ a_y,\ \beta,\ \delta,\ \dot\delta,\ a_x,\ e_y,\ e_\psi,\ \dot e_y,\ \dot e_\psi,\ \kappa_{\text{ref}},\ v_{\text{ref}}
\right]_k^T
$$

输入不包含：

$$
\mu_k,\quad
\mu_{k+1:k+H},\quad
\text{外部路面附着预瞄}
$$

Transformer 使用长度为 $L$ 的历史窗口：

$$
X_k
=
\left[
z_{k-L+1},
z_{k-L+2},
\cdots,
z_k
\right]
\in
\mathbb R^{L\times 14}
$$

训练前对特征做标准化：

$$
\tilde z
=
\frac{z-\mu_z}{\sigma_z}
$$

其中 $\mu_z,\sigma_z$ 由训练数据统计得到，并在在线推理中复用。

## 7. 风险标签构造

Transformer 的监督目标不是真实附着系数 $\mu$，而是未来窗口内车辆响应是否接近风险边界。对每个时刻 $k$，定义未来窗口：

$$
\mathcal H_k
=
\{k,k+1,\cdots,k+H\}
$$

### 7.1 横向误差增长风险

代码中使用速度和曲率自适应横向误差阈值：

$$
\theta_{e_y}
=
\operatorname{clip}
\left(
0.35+0.025v+3.5|\kappa|,
0.45,
1.6
\right)
$$

未来横向误差风险为：

$$
r_{ey}(k)
=
\operatorname{clip}
\left(
\frac{
\max_{j\in\mathcal H_k}|e_y(j)|
}{\theta_{e_y}},
0,
1
\right)
$$

### 7.2 稳定性风险

横摆角速度阈值为：

$$
\theta_r
=
\operatorname{clip}
\left(
|v\kappa|+0.35,
0.35,
0.95
\right)
$$

横向加速度阈值为：

$$
\theta_{a_y}
=
\operatorname{clip}
\left(
v^2|\kappa|+2.5,
2.5,
7.0
\right)
$$

稳定性风险由未来窗口中最危险的稳定性指标决定：

$$
r_{\text{stab}}(k)
=
\operatorname{clip}
\left(
\max
\left[
\frac{\max|\beta|}{0.10},
\frac{\max|r|}{\theta_r},
\frac{\max|a_y|}{\theta_{a_y}}
\right],
0,
1
\right)
$$

### 7.3 综合低附着风险与速度修正标签

控制饱和风险为：

$$
r_u(k)
=
\operatorname{clip}
\left(
\frac{\max_{\mathcal H_k}|\delta|}{0.50},
0,
1
\right)
$$

综合风险为：

$$
r_{\text{low}}(k)
=
\operatorname{clip}
\left(
0.45r_{ey}
+
0.45r_{\text{stab}}
+
0.10r_u,
0,
1
\right)
$$

曲率速度风险为：

$$
r_\kappa
=
\operatorname{clip}
\left(
\frac{v^2|\kappa|}{7.0},
0,
1
\right)
$$

安全速度缩放标签为：

$$
k_v(k)
=
\operatorname{clip}
\left(
1
-
0.40r_{\text{low}}
-
0.15r_\kappa,
k_{\min},
1
\right)
$$

最终 Transformer 输出目标为：

$$
y_k=
\left[
r_{\text{low}},
r_{ey},
r_{\text{stab}},
k_v
\right]_k^T
$$

## 8. Transformer 风险预测模块

项目中的 `RiskTransformer` 使用 Encoder-only Transformer。输入窗口 $X_k$ 先投影到模型维度：

$$
H_0
=
X_kW_e+b_e+P
$$

其中 $P$ 为正弦位置编码：

$$
P_{t,2i}
=
\sin
\left(
\frac{t}{10000^{2i/d}}
\right)
$$

$$
P_{t,2i+1}
=
\cos
\left(
\frac{t}{10000^{2i/d}}
\right)
$$

自注意力计算为：

$$
Q=HW_Q,\quad K=HW_K,\quad V=HW_V
$$

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

多头注意力为：

$$
\operatorname{MHA}(H)
=
\operatorname{Concat}
\left(
\text{head}_1,\cdots,\text{head}_h
\right)W_O
$$

每层 Encoder 可写为：

$$
\bar H_l
=
\operatorname{LN}
\left(
H_{l-1}
+
\operatorname{MHA}(H_{l-1})
\right)
$$

$$
H_l
=
\operatorname{LN}
\left(
\bar H_l
+
\operatorname{FFN}(\bar H_l)
\right)
$$

项目使用最后一个时间步的编码向量 $h_k$ 作为窗口表征：

$$
h_k=H_L[-1]
$$

风险头输出为：

$$
\hat r_{\text{raw}}
=
W_o h_k+b_o
$$

前三个风险通过 Sigmoid 限制到 $[0,1]$：

$$
\hat r_{\text{low}},
\hat r_{ey},
\hat r_{\text{stab}}
=
\sigma
\left(
\hat r_{\text{raw},1:3}
\right)
$$

速度缩放系数限制在 $[k_{\min},1]$：

$$
\hat k_v
=
k_{\min}
+
(1-k_{\min})
\sigma
\left(
\hat r_{\text{raw},4}
\right)
$$

因此：

$$
\hat y_k
=
\left[
\hat r_{\text{low}},
\hat r_{ey},
\hat r_{\text{stab}},
\hat k_v
\right]^T
$$

## 9. 风险映射模块

Transformer 输出首先经过异常检查与限幅：

$$
\hat r_i
\leftarrow
\operatorname{clip}
\left(
\hat r_i,
0,
1
\right)
$$

$$
\hat k_v
\leftarrow
\operatorname{clip}
\left(
\hat k_v,
k_{\min},
1
\right)
$$

若输出非有限值，则退化为规则风险。

### 9.1 快升慢降滤波

风险进入 MPC 前会被滤波：

$$
\bar r_k
=
\rho \bar r_{k-1}
+
(1-\rho)\hat r_k
$$

其中：

$$
\rho=
\begin{cases}
\rho_{\text{up}}, & \hat r_k>\bar r_{k-1} \\
\rho_{\text{down}}, & \hat r_k\le \bar r_{k-1}
\end{cases}
$$

项目中 $\rho_{\text{up}}<\rho_{\text{down}}$，因此风险上升更快，风险回落更慢。

### 9.2 MPC 权重自适应

横向误差风险主要调节轨迹跟踪权重：

$$
Q_y^{\text{adapt}}
=
\operatorname{sat}
\left(
Q_y^0(1+1.2\bar r_{ey}),
Q_y^0,
Q_y^{\max}
\right)
$$

$$
Q_\psi^{\text{adapt}}
=
\operatorname{sat}
\left(
Q_\psi^0(1+\bar r_{ey}),
Q_\psi^0,
Q_\psi^{\max}
\right)
$$

稳定性风险主要调节稳定性惩罚和控制平顺性：

$$
Q_\beta^{\text{adapt}}
=
\operatorname{sat}
\left(
Q_\beta^0(1+2.4\bar r_{\text{stab}}),
Q_\beta^0,
Q_\beta^{\max}
\right)
$$

$$
Q_r^{\text{adapt}}
=
\operatorname{sat}
\left(
Q_r^0(1+2.0\bar r_{\text{stab}}),
Q_r^0,
Q_r^{\max}
\right)
$$

$$
R_{\Delta\delta}^{\text{adapt}}
=
\operatorname{sat}
\left(
R_{\Delta\delta}^0(1+2.0\bar r_{\text{stab}}),
R_{\Delta\delta}^0,
R_{\Delta\delta}^{\max}
\right)
$$

### 9.3 稳定性约束自适应收紧

侧偏角、横摆角速度和横向加速度约束随稳定性风险升高而收紧：

$$
\beta_{\max}^{\text{adapt}}
=
\operatorname{sat}
\left(
\beta_{\max}^0(1-c_\beta \bar r_{\text{stab}}),
\beta_{\min},
\beta_{\max}^0
\right)
$$

$$
r_{\max}^{\text{adapt}}
=
\operatorname{sat}
\left(
r_{\max}^0(1-c_r \bar r_{\text{stab}}),
r_{\min},
r_{\max}^0
\right)
$$

$$
a_{y,\max}^{\text{adapt}}
=
\operatorname{sat}
\left(
a_{y,\max}^0(1-c_{a_y}\bar r_{\text{stab}}),
a_{y,\min},
a_{y,\max}^0
\right)
$$

转角变化率上限也会被收紧：

$$
\dot\delta_{\max}^{\text{adapt}}
=
\operatorname{sat}
\left(
\dot\delta_{\max}^0(1-0.45\bar r_{\text{stab}}),
\dot\delta_{\min},
\dot\delta_{\max}^0
\right)
$$

### 9.4 参考速度修正

参考速度缩放由 Transformer 输出的 $\hat k_v$ 和综合风险共同决定：

$$
\gamma_v
=
\min
\left(
\bar k_v,
1-c_v\bar r_{\text{low}}
\right)
$$

$$
v_{\text{ref}}^{\text{adapt}}
=
v_{\text{ref}}\cdot
\operatorname{sat}
\left(
\gamma_v,
k_{\min},
1
\right)
$$

随后控制器将该速度修正转化为纵向加速度：

$$
a_x
=
\operatorname{clip}
\left(
0.8
\left(
v_{\text{ref}}^{\text{adapt}}-v_x
\right),
-a_x^{\max},
a_x^{\max}
\right)
$$

这一步是 Transformer-MPC 在低附着区间产生稳定性优势的重要机理：风险升高后，车辆不只是改变转角优化权重，还会降低实际行驶速度。

## 10. LTV-MPC 控制器

MPC 使用横向误差状态：

$$
x=
\left[
e_y,\ e_\psi,\ v_y,\ r
\right]^T
$$

控制输入为：

$$
u=\delta
$$

扰动输入为参考曲率：

$$
w=\kappa_{\text{ref}}
$$

### 10.1 连续时间横向模型

在当前速度 $v_x$ 下线性化：

$$
\dot x
=
A_c(v_x)x
+
B_c(v_x)u
+
E_c(v_x)\kappa_{\text{ref}}
$$

其中：

$$
A_c=
\begin{bmatrix}
0 & v_x & 1 & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & -\frac{C_f+C_r}{mv_x} &
\frac{-l_fC_f+l_rC_r}{mv_x}-v_x \\
0 & 0 & \frac{-l_fC_f+l_rC_r}{I_zv_x} &
-\frac{l_f^2C_f+l_r^2C_r}{I_zv_x}
\end{bmatrix}
$$

$$
B_c=
\begin{bmatrix}
0\\
0\\
\frac{C_f}{m}\\
\frac{l_fC_f}{I_z}
\end{bmatrix}
$$

$$
E_c=
\begin{bmatrix}
0\\
-v_x\\
0\\
0
\end{bmatrix}
$$

采用欧拉离散化：

$$
A_d=I+T_sA_c
$$

$$
B_d=T_sB_c
$$

$$
E_d=T_sE_c
$$

即：

$$
x_{k+1}
=
A_dx_k
+
B_d\delta_k
+
E_d\kappa_{\text{ref},k}
$$

### 10.2 MPC 目标函数

MPC 预测时域为 $N$。在第 $i$ 个预测步，参考横摆角速度为：

$$
r_{\text{ref},i}
=
v_{\text{ref},i}^{\text{adapt}}\kappa_{\text{ref},i}
$$

侧偏角近似为：

$$
\beta_i
\approx
\frac{v_{y,i}}{v_x}
$$

目标函数为：

$$
J
=
\sum_{i=1}^{N}
\left[
Q_y e_{y,i}^2
+
Q_\psi e_{\psi,i}^2
+
Q_\beta \beta_i^2
+
Q_r(r_i-r_{\text{ref},i})^2
+
R_\delta\delta_i^2
+
R_{\Delta\delta}(\delta_i-\delta_{i-1})^2
\right]
$$

在 Transformer-MPC 中，$Q_y,Q_\psi,Q_\beta,Q_r,R_\delta,R_{\Delta\delta}$ 不再是固定值，而是由风险映射层动态生成。

### 10.3 MPC 约束

项目中主要约束为：

$$
|\delta_i|
\le
\delta_{\max}
$$

$$
|\delta_i-\delta_{i-1}|
\le
\dot\delta_{\max}^{\text{adapt}}T_s
$$

$$
|\beta_i|
\le
\beta_{\max}^{\text{adapt}}
$$

$$
|r_i|
\le
r_{\max}^{\text{adapt}}
$$

优化问题为二次规划：

$$
\begin{aligned}
\min_{\delta_{0:N-1}}\quad & J \\
\text{s.t.}\quad
& x_{i+1}=A_dx_i+B_d\delta_i+E_d\kappa_i \\
& \delta,\ \Delta\delta,\ \beta,\ r \text{ satisfy constraints}
\end{aligned}
$$

求解器使用 CVXPY + OSQP，在线只执行第一个控制量：

$$
\delta_k^\*=u_0^\*
$$

## 11. Transformer-MPC 的结合机理

Transformer-MPC 不是端到端神经网络控制器，而是“学习风险中间量 + 可解释 MPC 参数调度”的结构。

### 11.1 离线训练阶段的数据交换

离线数据生成流程为：

```text
Scenario Manager
    -> Vehicle Model
    -> Baseline Controllers
    -> Simulator Logs
    -> Risk Label Generator
    -> Dataset Windows
    -> Transformer Training
```

数学上，第 $n$ 条仿真轨迹会产生日志：

$$
\mathcal D^{(n)}
=
\left\{
z_k^{(n)},\ e_k^{(n)},\ s_k^{(n)},\ u_k^{(n)}
\right\}_{k=1}^{T_n}
$$

风险标签模块使用未来窗口生成：

$$
y_k^{(n)}
=
g
\left(
s_{k:k+H}^{(n)},
e_{k:k+H}^{(n)},
u_{k:k+H}^{(n)}
\right)
$$

数据集样本为：

$$
\left(
X_k^{(n)},y_k^{(n)}
\right)
$$

其中：

$$
X_k^{(n)}
=
\left[
z_{k-L+1}^{(n)},\cdots,z_k^{(n)}
\right]
$$

训练目标为：

$$
\min_{\theta}
\sum_{n,k}
\mathcal L
\left(
F_\theta(X_k^{(n)}),
y_k^{(n)}
\right)
$$

项目中使用 Smooth L1 损失：

$$
\mathcal L
=
\operatorname{SmoothL1}
\left(
\hat y_k,
y_k
\right)
$$

### 11.2 在线控制阶段的数据交换

在线控制时刻 $k$ 的数据交换如下：

```text
车辆状态 s_k
    -> 误差计算得到 e_y, e_psi, kappa_ref, v_ref
    -> 拼接在线特征 z_k
    -> 历史队列形成 X_k
    -> 标准化后输入 Transformer
    -> 输出风险 r_low, r_ey, r_stab, k_v
    -> Risk Mapper 生成 MPCParameters
    -> LTV-MPC 求解 delta_k
    -> 速度修正生成 ax_k
    -> 车辆模型执行 [delta_k, ax_k]
```

即：

$$
z_k
=
\phi(s_k,e_k,u_{k-1},\mathcal P)
$$

$$
X_k
=
\left[z_{k-L+1},\cdots,z_k\right]
$$

$$
\hat y_k
=
F_\theta
\left(
\operatorname{Norm}(X_k)
\right)
$$

$$
\Theta_k
=
M(\hat y_k)
=
\left(
Q_k,R_k,\mathcal C_k,v_{\text{ref},k}^{\text{adapt}}
\right)
$$

$$
\delta_k
=
\operatorname{MPC}
\left(
x_k,\Theta_k
\right)
$$

$$
a_{x,k}
=
K_v
\left(
v_{\text{ref},k}^{\text{adapt}}-v_{x,k}
\right)
$$

最终执行：

$$
u_k
=
\left[
\delta_k,\ a_{x,k}
\right]^T
$$

### 11.3 为什么这种结合方式更可解释

若直接让神经网络输出控制量：

$$
u_k=F_\theta(X_k)
$$

则控制安全性主要依赖黑箱网络。当前项目采用：

$$
F_\theta(X_k)
\rightarrow
\hat r_k
\rightarrow
(Q_k,R_k,\mathcal C_k,v_{\text{ref}}^{\text{adapt}})
\rightarrow
\operatorname{MPC}
\rightarrow
u_k
$$

这样有三个优点：

1. **边界清晰**：Transformer 只负责风险预测，不直接控车。
2. **物理可解释**：风险升高时，权重增大、约束收紧、速度降低。
3. **可降级**：若 Transformer 输出异常，可退化为规则风险 MPC 或固定 MPC。

## 12. 四类控制器对比机理

### 12.1 PID

PID 控制器使用横向误差和航向误差反馈：

$$
\delta_k
=
-
\left(
K_y e_y
+
K_\psi e_\psi
+
K_d \dot e_y
\right)
$$

PID 不考虑未来曲率、车辆稳定性约束和低附着风险，因此在某些短时区间内横向误差可能较小，但容易带来较大的 $\beta$ 和 $a_y$。

### 12.2 Fixed MPC

Fixed MPC 使用固定权重和固定约束：

$$
Q_k=Q_0,\quad
R_k=R_0,\quad
\mathcal C_k=\mathcal C_0
$$

它能处理约束优化，但不能根据风险主动调整保守程度。

### 12.3 Rule-risk MPC

规则风险由当前状态直接计算：

$$
r_{ey}^{\text{rule}}
=
\operatorname{clip}
\left(
\frac{|e_y|}{1.2}
+
\frac{|\dot e_y|}{8.0},
0,
1
\right)
$$

$$
r_{\text{stab}}^{\text{rule}}
=
\operatorname{clip}
\left(
\max
\left[
\frac{|\beta|}{0.10},
\frac{|r|}{0.75},
\frac{|a_y|}{6.0}
\right],
0,
1
\right)
$$

$$
r_{\text{low}}^{\text{rule}}
=
0.5r_{ey}^{\text{rule}}
+
0.5r_{\text{stab}}^{\text{rule}}
$$

Rule-risk MPC 可解释，但只依赖当前时刻，缺少历史时序模式建模。

### 12.4 Transformer-MPC

Transformer-MPC 用历史窗口学习风险：

$$
\hat r_k
=
F_\theta
\left(
z_{k-L+1:k}
\right)
$$

相较规则风险，它能够利用：

$$
\left[
\text{历史控制输入},
\text{历史车辆响应},
\text{误差增长趋势},
\text{参考曲率变化}
\right]
$$

因此它不是只看“当前是否已经危险”，而是学习“未来窗口是否会变得危险”。

## 13. 仿真日志与结果分析中的综合指标

为了寻找 Transformer-MPC 的优势区间，项目新增了综合跟踪-稳定性代价：

$$
J_{\text{score}}
=
\frac{|e_y|}{1.5}
+
\frac{|\beta|}{0.10}
+
\frac{|a_y|}{8.0}
+
0.15\frac{|\delta|}{0.5}
$$

该指标不是 MPC 内部求解目标，而是结果分析指标，用于同时观察轨迹误差、侧偏稳定性、横向加速度和控制幅值。

若：

$$
J_{\text{score}}^{\text{Transformer-MPC}}
<
J_{\text{score}}^{\text{baseline}}
$$

则认为 Transformer-MPC 在该时刻相对对应基线具有综合优势。

优势区间由连续满足上述条件的时间段构成：

$$
\mathcal I
=
\left[
t_a,t_b
\right],
\quad
\forall t\in\mathcal I:
J_T(t)<J_B(t)
$$

其中 $T$ 表示 Transformer-MPC，$B$ 表示某个对比基线。

## 14. 当前原型的边界

当前项目是概念验证原型，具有以下边界：

1. 动态自行车模型和轮胎饱和是简化模型，不等同于 CarSim 高保真模型。
2. quick-run 结果只覆盖短时间窗口，不能直接作为最终论文统计结论。
3. Transformer 训练样本较少，风险预测需要通过更多工况增强。
4. 当前 MPC 使用欧拉离散和简化约束，后续可加入软约束、松弛变量和更完整的横向加速度约束。
5. 当前架构的重点是验证“风险预测中间量增强 MPC”的数据链路和控制机理。

## 15. 总结

本项目的 Transformer-MPC 结合机理可以概括为：

$$
\boxed{
\text{历史动力学时序}
\rightarrow
\text{Transformer 风险预测}
\rightarrow
\text{风险映射层}
\rightarrow
\text{自适应 LTV-MPC}
\rightarrow
\text{受约束轨迹跟踪控制}
}
$$

Transformer 的作用是从车辆历史状态、控制输入、轨迹误差和参考曲率中预测短时风险。MPC 的作用是在风险调度后的权重、约束和速度参考下求解可执行控制量。两者之间通过 `RiskOutput` 和 `MPCParameters` 交换数据，使学习模块和优化控制模块保持清晰边界。
