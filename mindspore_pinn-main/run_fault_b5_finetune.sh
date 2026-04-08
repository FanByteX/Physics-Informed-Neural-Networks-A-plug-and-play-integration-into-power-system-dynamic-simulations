#!/usr/bin/env bash

# ========== b=5 渐进式微调优化 - 目标<2% ==========

echo "======================================================================"
echo "🎯 b=5 渐进式微调优化 - 目标所有变量<2%"
echo "======================================================================"
echo ""
echo "📊 当前状态 (logs/fault-b5):"
echo "  ✅ δ2: 1.31% (已达标)"
echo "  ✅ δ3: 0.19% (已达标)"
echo "  ✅ V3: 0.09% (已达标)"
echo "  ❌ ω1: 7.59% → 目标: <2%"
echo "  ❌ ω2: 7.80% → 目标: <2%"
echo ""
echo "💡 渐进式微调策略 (3步温和优化 - 保持depth=4架构):"
echo "  步骤1: dyn-weight 32→48,  增加数据到10k  → logs/fault-b5-finetune-step1/"
echo "  步骤2: dyn-weight 48→64,  增加数据到12k → logs/fault-b5-finetune-step2/"
echo "  步骤3: dyn-weight 64→80,  精细打磨      → logs/fault-b5-finetune-step3/"
echo ""
echo "📁 原始数据保护: logs/fault-b5/ 目录不会被覆盖"
echo "======================================================================"
echo ""

# 准备：复制初始模型到微调目录
echo "📋 准备阶段: 从 logs/fault-b5/ 复制模型到微调目录"
if [ -f "./logs/fault-b5/model.pth" ]; then
    echo "✅ 找到fault-b5的训练模型"
    mkdir -p ./logs/fault-b5-finetune-step1
    mkdir -p ./logs/fault-b5-finetune-step2
    mkdir -p ./logs/fault-b5-finetune-step3
    
    cp ./logs/fault-b5/model.pth ./logs/fault-b5-finetune-step1/
    echo "  → 复制到 fault-b5-finetune-step1/"
else
    echo "❌ 未找到 logs/fault-b5/model.pth"
    echo "请先确保 logs/fault-b5/ 目录存在并包含训练好的模型"
    exit 1
fi
echo ""

# ========== 步骤1: 提升到48 ==========
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔹 步骤 1/3: dyn-weight 32 → 48 (提升50%)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "策略: 温和提升权重,增加训练数据"
echo "预期: ω1/ω2误差从7.6%降至5-6%"
echo "日志: ./logs/fault-b5-finetune-step1/"
echo ""

python fault_powerNet_b5.py \
    --log-dir ./logs/fault-b5-finetune-step1/ \
    --num-test 500 --use-scheduler --patience 2500 --batch-size 1048 \
    --unstacked --dyn-depth 4 --dyn-width 100 --h 0.1 --N 80 \
    --dyn-type attention --alg-type attention \
    --dyn-activation sin --alg-activation sin \
    --test-every 500 --scheduler-type plateau \
    --alg-weight 1.0 --num-train 10000 --num-val 100 \
    --use-tqdm \
    --dyn-weight 48.0 \
    --epochs 20000 \
    --lr 5e-5 \
    --start-from-best

echo "✅ 步骤1完成! 检查进度..."
python -c "
import numpy as np
data = np.load('logs/fault-b5-finetune-step1/L2Relative_error.npz')
error = data['error'][-1]
print(f'  ω1: {error[0]*100:.2f}% (目标<2%)')
print(f'  ω2: {error[1]*100:.2f}% (目标<2%)')
"
echo ""

# 为步骤2准备模型
echo "📋 准备步骤2: 复制步骤1的最佳模型"
cp ./logs/fault-b5-finetune-step1/model.pth ./logs/fault-b5-finetune-step2/
echo ""

# ========== 步骤2: 提升到64并扩展网络 ==========
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔹 步骤 2/3: dyn-weight 48 → 64 (保持depth=4)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "策略: 达到b=10的权重水平,增加训练数据量"
echo "预期: ω1/ω2误差从5-6%降至3-4%"
echo "日志: ./logs/fault-b5-finetune-step2/"
echo ""

python fault_powerNet_b5.py \
    --log-dir ./logs/fault-b5-finetune-step2/ \
    --num-test 500 --use-scheduler --patience 3000 --batch-size 1048 \
    --unstacked --dyn-depth 4 --dyn-width 100 --h 0.1 --N 200 \
    --dyn-type attention --alg-type attention \
    --dyn-activation sin --alg-activation sin \
    --test-every 500 --scheduler-type plateau \
    --alg-weight 1.0 --num-train 12000 --num-val 100 \
    --use-tqdm \
    --dyn-weight 64.0 \
    --epochs 25000 \
    --lr 3e-5 \
    --start-from-best

echo "✅ 步骤2完成! 检查进度..."
python -c "
import numpy as np
data = np.load('logs/fault-b5-finetune-step2/L2Relative_error.npz')
error = data['error'][-1]
print(f'  ω1: {error[0]*100:.2f}% (目标<2%)')
print(f'  ω2: {error[1]*100:.2f}% (目标<2%)')
"
echo ""

# 为步骤3准备模型
echo "📋 准备步骤3: 复制步骤2的最佳模型"
cp ./logs/fault-b5-finetune-step2/model.pth ./logs/fault-b5-finetune-step3/
echo ""

# ========== 步骤3: 冲刺到80 ==========
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 步骤 3/3: dyn-weight 64 → 80 (最终冲刺, depth=4)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "策略: 精细打磨,冲刺<2%目标,保持架构一致性"
echo "目标: ω1/ω2误差 <2%"
echo "日志: ./logs/fault-b5-finetune-step3/"
echo ""

python fault_powerNet_b5.py \
    --log-dir ./logs/fault-b5-finetune-step3/ \
    --num-test 500 --use-scheduler --patience 3500 --batch-size 1048 \
    --unstacked --dyn-depth 4 --dyn-width 100 --h 0.1 --N 80 \
    --dyn-type attention --alg-type attention \
    --dyn-activation sin --alg-activation sin \
    --test-every 500 --scheduler-type plateau \
    --alg-weight 1.0 --num-train 12000 --num-val 100 \
    --use-tqdm \
    --dyn-weight 80.0 \
    --epochs 30000 \
    --lr 2e-5 \
    --start-from-best

echo ""
echo "======================================================================"
echo "🎉 渐进式微调完成！"
echo "======================================================================"
echo ""
echo "📊 训练历程总结:"
echo "  步骤1: dyn-weight=48,  depth=4, data=10k, epochs=20k, lr=5e-5"
echo "  步骤2: dyn-weight=64,  depth=4, data=12k, epochs=25k, lr=3e-5"
echo "  步骤3: dyn-weight=80,  depth=4, data=12k, epochs=30k, lr=2e-5"
echo ""
echo "📈 优化路径: 32 → 48 → 64 → 80 (2.5倍提升)"
echo "🏗️  架构保持: depth=4不变(避免权重不兼容), data 8k → 12k"
echo "📈 总训练轮次: 75000 (20k+25k+30k)"
echo "⏱️  预计总耗时: 2.5-3小时"
echo ""
echo "📁 结果目录:"
echo "  原始训练: ./logs/fault-b5/ (已保护,未覆盖)"
echo "  微调结果: ./logs/fault-b5-finetune-step{1,2,3}/"
echo ""

# 最终验证
python -c "
import numpy as np
import os

print()
print('=' * 70)
print('📊 各阶段误差对比:')
print('-' * 70)

# 原始
if os.path.exists('logs/fault-b5/L2Relative_error.npz'):
    data0 = np.load('logs/fault-b5/L2Relative_error.npz')
    e0 = data0['error'][-1]
    print(f'原始 (dyn-weight=32): ω1={e0[0]*100:.2f}%, ω2={e0[1]*100:.2f}%, δ2={e0[2]*100:.2f}%')

# 步骤1
if os.path.exists('logs/fault-b5-finetune-step1/L2Relative_error.npz'):
    data1 = np.load('logs/fault-b5-finetune-step1/L2Relative_error.npz')
    e1 = data1['error'][-1]
    print(f'步骤1 (dyn-weight=48): ω1={e1[0]*100:.2f}%, ω2={e1[1]*100:.2f}%, δ2={e1[2]*100:.2f}%')

# 步骤2
if os.path.exists('logs/fault-b5-finetune-step2/L2Relative_error.npz'):
    data2 = np.load('logs/fault-b5-finetune-step2/L2Relative_error.npz')
    e2 = data2['error'][-1]
    print(f'步骤2 (dyn-weight=64): ω1={e2[0]*100:.2f}%, ω2={e2[1]*100:.2f}%, δ2={e2[2]*100:.2f}%')

# 步骤3 (最终)
if os.path.exists('logs/fault-b5-finetune-step3/L2Relative_error.npz'):
    data3 = np.load('logs/fault-b5-finetune-step3/L2Relative_error.npz')
    e3 = data3['error'][-1]
    print(f'步骤3 (dyn-weight=80): ω1={e3[0]*100:.2f}%, ω2={e3[1]*100:.2f}%, δ2={e3[2]*100:.2f}%')
    print('-' * 70)
    
    all_pass = e3[0] < 0.02 and e3[1] < 0.02 and e3[2] < 0.02
    if all_pass:
        print('🎉🎉🎉 恭喜! 所有变量均达标! (<2%)')
    else:
        if e3[0] < 0.03:
            print('✅ 接近目标! 建议继续运行一轮精细打磨')
        else:
            print('⚠️  建议进一步优化: dyn-weight提升到96-128')
    
    print()
    print('📈 总体优化效果:')
    if os.path.exists('logs/fault-b5/L2Relative_error.npz'):
        print(f'  ω1: {e0[0]*100:.2f}% → {e3[0]*100:.2f}% (降低 {(e0[0]-e3[0])/e0[0]*100:.1f}%)')
        print(f'  ω2: {e0[1]*100:.2f}% → {e3[1]*100:.2f}% (降低 {(e0[1]-e3[1])/e0[1]*100:.1f}%)')
print('=' * 70)
"

echo ""
echo "📊 对比分析: python compare_b_values.py"
echo "======================================================================"
