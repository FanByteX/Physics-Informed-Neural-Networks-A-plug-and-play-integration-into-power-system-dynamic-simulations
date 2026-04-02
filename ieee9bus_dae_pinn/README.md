# IEEE 9-Bus DAE-PINN

基于 DAE-PINNs 架构的 IEEE 9 节点 3 机系统物理信息神经网络实现。

## 项目结构

```
ieee9bus_dae_pinn/
├── models.py          # 神经网络架构
├── physics.py         # 物理约束和损失函数
├── data_handler.py    # 数据生成
├── trainer.py         # 训练管理器
├── main.py           # 主入口
└── README.md         # 本文档
```

## 系统说明

### IEEE 9-Bus 系统
- 3 台发电机
- 每台发电机 4 个微分状态变量 (E'q, E'd, δ, ω)
- 9 个节点，每个节点 2 个代数变量 (V, θ)
- 总计：12 个微分变量 + 18 个代数变量 = 30 个状态变量

### 网络架构
遵循 DAE-PINNs 的设计：

1. **动态网络 (Y)**:
   - Stacked 模式：12 个独立的前馈神经网络（每个状态变量一个）
   - Combined 模式：1 个统一的前馈神经网络
   - 输出：IRK 阶段预测值

2. **代数网络 (Z)**:
   - 1 个前馈神经网络
   - 输出：18 × (IRK_stages + 1) 个代数变量预测值

### 物理约束

- **动态方程**: 发电机摇摆方程、励磁方程
- **代数方程**: 功率平衡方程
- **IRK 残差**: 隐式 Runge-Kutta 离散化

## 使用方法

### 基本训练

```bash
python main.py
```

### 自定义参数

```bash
python main.py \
    --num_train 20000 \
    --epochs 50000 \
    --lr 1e-4 \
    --batch_size 256 \
    --stacked \
    --activation tanh \
    --h 0.04 \
    --loss_weight_dyn 1.0 \
    --loss_weight_alg 1.0 \
    --scheduler plateau \
    --patience 2000
```

### 主要参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_train` | 10000 | 训练样本数 |
| `--num_test` | 1000 | 测试样本数 |
| `--batch_size` | None | 批大小（None=全批次）|
| `--num_IRK_stages` | 10 | IRK 阶段数 |
| `--stacked` | True | 使用独立网络架构 |
| `--activation` | tanh | 激活函数 |
| `--epochs` | 10000 | 训练轮数 |
| `--lr` | 1e-3 | 学习率 |
| `--h` | 0.04 | 时间步长 |
| `--loss_weight_dyn` | 1.0 | 动态损失权重 |
| `--loss_weight_alg` | 1.0 | 代数损失权重 |
| `--scheduler` | None | 学习率调度器 |
| `--test_every` | 100 | 测试间隔 |
| `--save_every` | 1000 | 保存间隔 |

### 从检查点恢复

```bash
python main.py --resume ./logs/ieee9bus_pinn_best.pth
```

## 配置文件

项目依赖于以下配置文件（位于 `../config_files/`）：

- `config_machines_dynamic.yaml`: 发电机动态参数
  - 惯性常数 H
  - 阻尼系数 D
  - 电抗参数 X'd, X'q
  - 时间常数 T'd0, T'q0
  
- `config_machines_static.yaml`: 发电机静态参数
  - 发电机母线编号
  - 基准功率
  
- `network_admittance.pt`: 网络导纳矩阵

## 输出

训练过程会在 `./logs/` 目录下生成：

- `ieee9bus_pinn_best.pth`: 最佳模型检查点
- `ieee9bus_pinn_epoch{N}.pth`: 周期性保存的检查点

检查点包含：
- 模型权重
- 优化器状态
- 损失历史
- 最佳损失值

## 依赖

- Python >= 3.7
- PyTorch >= 1.9
- NumPy
- PyYAML
- DeepXDE (用于状态空间采样)

## 参考

本实现参考了以下项目：
- [DAE-PINNs](https://github.com/etorobot/DAE-PINNs) - 架构设计参考
- PINNs-Plug-n-Play-Integration - IEEE 9-bus 系统配置

## 架构对比

| 特性 | DAE-PINNs (3节点) | 本项目 (IEEE 9节点) |
|------|-------------------|---------------------|
| 节点数 | 3 | 9 |
| 发电机数 | 2 (简化) | 3 |
| 微分变量 | 4 | 12 |
| 代数变量 | 2 | 18 |
| 状态总数 | 6 | 30 |
| 适用场景 | 算法验证 | 工业级测试系统 |
