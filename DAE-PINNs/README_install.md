# DAE-PINNs 安装说明

## 环境要求

- Python 3.7 - 3.9
- PyTorch 1.9+ (当前环境: PyTorch 1.11.0)

## 快速安装

### 方案1: 使用 pip 安装（推荐）

```bash
# 进入项目目录
cd /home/PINNs-Plug-n-Play-Integration/DAE-PINNs

# 安装依赖
pip install -r requirements.txt
```

### 方案2: 使用 conda 创建新环境

```bash
# 创建新环境
conda env create -f environment.yml

# 激活环境
conda activate dae-pinn
```

### 方案3: 使用完整 conda 配置文件

```bash
# 使用提供的完整包列表（包含所有依赖）
conda create --name dae-pinn --file DAE-PINNs-req.txt
```

## 已验证的环境

当前系统已安装以下依赖：

| 包名 | 版本 | 状态 |
|------|------|------|
| Python | 3.9.17 | ✓ |
| NumPy | 1.23.5 | ✓ |
| SciPy | 1.13.1 | ✓ |
| Matplotlib | 3.9.4 | ✓ |
| PyTorch | 1.11.0 | ✓ |
| DeepXDE | 0.13.6 | ✓ |
| PyYAML | 6.0.3 | ✓ |
| TQDM | 4.65.0 | ✓ |

**所有依赖已安装！环境准备就绪。**

## 重要说明

### DeepXDE 版本兼容性

- **PyTorch 1.x**: 使用 DeepXDE 0.13.6
- **PyTorch 2.0+**: 使用 DeepXDE 1.0+

当前环境使用 **PyTorch 1.11.0 + DeepXDE 0.13.6**，已经过验证。

### DeepXDE 后端配置

DeepXDE 会自动检测已安装的深度学习框架。当前配置：

```python
import deepxde as dde
# 自动使用 PyTorch 后端
# 输出: Using backend: pytorch
```

如需手动指定后端，可以设置环境变量：

```bash
export DDE_BACKEND=pytorch
```

或者在代码中设置：

```python
import os
os.environ['DDE_BACKEND'] = 'pytorch'
```

## 运行示例

安装完成后，可以运行示例脚本：

```bash
# 运行简单示例
cd /home/PINNs-Plug-n-Play-Integration/DAE-PINNs/src
python example_powerNet.py

# 运行训练示例
python run_training_examples.py --num_train 1000 --epochs 10000

# 运行宽度分析
python run_width_analysis.py
```

## 故障排除

### 1. DeepXDE 版本冲突

**问题**: `RuntimeError: DeepXDE requires PyTorch>=2.0.0`

**解决**: 安装兼容版本
```bash
pip uninstall -y deepxde
pip install deepxde==0.13.6
```

### 2. 导入错误

**问题**: `ImportError: cannot import name 'dotdict'`

**解决**: 确保在项目根目录运行，或将 `src` 目录添加到 PYTHONPATH
```bash
export PYTHONPATH="${PYTHONPATH}:/home/PINNs-Plug-n-Play-Integration/DAE-PINNs/src"
```

### 3. CUDA 相关问题

**问题**: PyTorch 无法识别 GPU

**解决**: 检查 CUDA 版本
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

如果 CUDA 不可用，代码会自动使用 CPU。

## 项目结构

```
DAE-PINNs/
├── src/
│   ├── models/
│   │   └── DAEnn.py          # 神经网络模型
│   ├── data/
│   │   └── DAE.py            # 数据生成
│   ├── utils/
│   │   ├── plots.py          # 绘图函数
│   │   └── utils.py          # 工具函数
│   ├── example_powerNet.py   # 基本示例
│   ├── run_training_examples.py
│   ├── run_width_analysis.py
│   └── supervisor.py         # 训练管理器
├── requirements.txt          # pip 依赖列表
├── environment.yml           # conda 环境配置
├── DAE-PINNs-req.txt        # 完整 conda 包列表
└── README_install.md        # 本文档
```

## 参考链接

- [DeepXDE 官方文档](https://deepxde.readthedocs.io/)
- [PyTorch 官方文档](https://pytorch.org/docs/)
- [DAE-PINNs GitHub](https://github.com/etorobot/DAE-PINNs)
