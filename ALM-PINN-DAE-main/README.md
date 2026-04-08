# RES-PINN-DAE
**基于MindSpore的残差物理信息神经网络求解微分代数方程**
## 简介
评估电力系统的动态安全需要求解一组以显着刚度为特征的非线性微分代数方程 (DAE)。近年来，基于深度学习的建模方法在解决此类刚性动态问题方面表现出了巨大的潜力。这项工作提出了一种基于华为MindSpore深度学习框架开发的残差物理信息神经网络（RES-PINN），旨在有效学习具有“无限刚度”特性的DAE的解轨迹。该方法采用隐式龙格库塔（IRK）时间离散方案来解决刚度问题，直接将物理方程嵌入到损失函数中，并通过最小化残差驱动网络学习物理一致的解。为了准确地施加代数约束并平衡约束满足与模型优化目标，将增强拉格朗日方法（ALM）集成到学习过程中。通过罚函数和拉格朗日乘子的协同作用，实现了近硬约束流形一致性，并显着减少了长期模拟中的数值漂移。引入的残差跳跃连接有效地缓解了梯度消失，增强了动态变量和代数变量的协同学习，使模型即使在故障扰动下也能保持较高的拟合精度。借助MindSpore的自动微分机制和高性能Ascend NPU平台，该方法在训练过程中实现了大幅加速。
### 核心特性
- **IRK时间离散**：100级隐式龙格库塔方案解决刚度问题
- **增广拉格朗日约束**：ALM方法精确施加代数约束
- **残差跳跃连接**：缓解梯度消失，增强动态/代数变量协同学习
- **NPU加速**：华为Ascend NPU平台训练加速
## 环境配置
```
镜像: mindspore_2.4.10_cann_8.0.0_py_3.10_euler_2.10.11_aarch64
硬件: Ascend NPU | 系统: EulerOS 2.10.11 | Python: 3.10
主要依赖: MindSpore 2.4.10, NumPy 1.20.3, SciPy 1.7.3, DeepXDE 1.8.3
```
## 项目结构
```
src/
├── main.py, config.py          # 入口与配置
├── data/                       # 数据模块 (dae_data, irk_weights, irk100.txt)
├── models/                     # 模型模块 (RES-PINN, attention, fnn, conv1d)
├── training/                   # 训练模块 (supervisor, checkpoint)
├── utils/                      # 工具模块 (losses, metrics, plots)
├── scripts/                    # 实验脚本
└── logs/                       # 输出日志
```
## 快速开始
```bash
cd src
bash run_single_npu.sh              # 单卡训练
bash run_distributed_4npu.sh        # 四卡分布式
python main.py --epochs 10000 --lr 0.001 --dyn-type attention
```
## 工作原理
### 1. 网络架构
```
输入层 → [动态变量网络Y] → ω₁,ω₂,δ₂,δ₃ (4个状态×101个时间节点)
  ↓
(ω₁,ω₂,δ₂,δ₃)  →  [代数变量网络Z]  →  V₃ (电压×101个时间节点)
```
- **Res-PINN**：残差物理信息神经网络，支持 `fnn`/`attention`/`Conv1D` 架构
- **动态网络Y**：4层×100宽度，输出所有IRK节点的状态预测
- **代数网络Z**：2层×40宽度，预测代数变量V₃
### 2. IRK时间离散
采用100级隐式Runge-Kutta方案将连续DAE离散为残差形式：
```
y_n = Y - h·(F·A^T)    # 动力学残差: F=[F₀,F₁,F₂,F₃], A=IRK权重矩阵
G(y,z) = 0             # 代数约束: 功率平衡方程
```
### 3. 物理方程（三母线系统）
**动力学方程**：
```
F₀ = (1/M₁)·(-D·ω₁ + f₁ + f₂)    # 发电机1角速度
F₁ = (1/M₂)·(-D·ω₂ - f₁)         # 发电机2角速度  
F₂ = ω₂ - ω₁                      # 功角差变化率
F₃ = -ω₁ - (1/D_d)·f₂            # 负载功角变化率
```
**代数约束**（功率平衡）：
```
G = 2b·V₃² - b·V₃·V₁·cos(δ₃) - b·V₃·V₂·cos(δ₃-δ₂) + Q_l = 0
```
### 4. 增广拉格朗日损失
```
L_total = w_dyn·L_dyn + w_alg·L_alg
L_dyn = Σ MSE(f_i)                           # 动力学残差
L_alg = λ·√(MSE(g)) + (μ/2)·MSE(g)          # ALM代数约束
```
- **λ**：拉格朗日乘子，自适应更新 `λ ← λ + μg`
- **μ**：罚参数，渐进增大 `μ ← min(ρμ, μ_max)`
### 5. 训练与推理流程
```
[训练阶段]
DeepXDE采样 → dae_data加载 → supervisor.train() → Adam优化 → checkpoint保存
[推理阶段]
X₀ → predict() → 单步预测 → integrate() → N步迭代 → 完整轨迹
```
## 应用场景
三母线电力系统故障仿真：正常运行(0-4s, b=10) → 线路故障(4以后, b=5)
变量: ω₁,ω₂(角速度), δ₂,δ₃(功角), V₃(电压)
## 0-16s内动态变量(ω₁,ω₂,δ₂,δ₃)和代数变量(V₃)真解与预测解对比（t=4s故障扰动）：
![故障扰动对比](src/logs/visualize_fault/fault_at_4s_comparison.png)
## 故障后100个时间步的预测误差
![故障后100个时间步的预测误差](src/logs/fault_error_plots/fault_injection_100.png)
## 轨迹预测结果
![轨迹预测结果](src/logs/mindspore_pinn_4npu/trajectories_enhanced_size.png)
## 训练Loss曲线
![Loss曲线](src/logs/mindspore_pinn_4npu/loss_curve.png)
## L2相对误差
动态变量(ω₁,ω₂,δ₂,δ₃)和代数变量(V₃)的L₂相对误差：
![L2相对误差](src/logs/mindspore_pinn_4npu/L2relative_error.png)



