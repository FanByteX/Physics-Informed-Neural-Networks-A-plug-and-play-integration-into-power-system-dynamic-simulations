#!/bin/bash
# DAE-PINNs 环境安装脚本

echo "========================================"
echo "DAE-PINNs 环境安装"
echo "========================================"

# 方案1: 使用 conda 创建新环境（推荐）
echo ""
echo "方案1: 创建新的 conda 环境"
echo "命令: conda env create -f environment.yml"
echo ""

# 方案2: 使用现有环境，pip 安装依赖
echo "方案2: 使用现有 Python 环境，pip 安装依赖"
echo "命令: pip install -r requirements.txt"
echo ""

# 检查 Python 版本
echo "检查 Python 版本..."
python --version

echo ""
echo "建议安装步骤:"
echo "1. 创建 conda 环境 (推荐):"
echo "   conda create -n dae-pinn python=3.7"
echo "   conda activate dae-pinn"
echo "   pip install -r requirements.txt"
echo ""
echo "2. 或使用提供的完整 conda 配置:"
echo "   conda create --name dae-pinn --file DAE-PINNs-req.txt"
echo ""
echo "3. 或直接安装到当前环境:"
echo "   pip install -r requirements.txt"
