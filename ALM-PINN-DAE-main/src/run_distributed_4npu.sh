#!/bin/bash
# ========================================================================
# MindSpore 4卡 NPU 分布式训练启动脚本
# 使用 MindSpore 的 mpirun 进行多卡并行训练
# ========================================================================
set -e
echo "========================================================================"
echo "🚀 Starting Distributed Training on 4 Ascend NPUs"
echo "========================================================================"
# 使用与单卡最佳参数一致的配置，针对V3振荡增强代数约束
EPOCHS=20000
BATCH_SIZE=1048  # 每卡batch size，与单卡一致
LR=1e-5  # 与单卡最佳参数相同
LOG_DIR="logs/mindspore_pinn_4npu"
# 权重配置：DYN_WEIGHT保持单卡最佳，ALG_WEIGHT增大解决V3振荡
DYN_WEIGHT=64.0   # 与单卡最佳一致
ALG_WEIGHT=10.5   # 从1.0增大到10.0，专门强化V3（代数变量）约束
# 从当前4卡训练的最佳模型继续优化（专注V3）
PRETRAIN_MODEL="${LOG_DIR}/model.ckpt"
START_FROM_BEST=""

if [ -f "${PRETRAIN_MODEL}" ]; then
    echo "✅ Found 4-card trained model: ${PRETRAIN_MODEL}"
    echo "🎯 Will continue training to improve V3 fitting"
    START_FROM_BEST="--start-from-best"
    export PRETRAIN_MODEL_PATH="${PRETRAIN_MODEL}"
else
    echo "⚠️  4-card model not found, trying single-card model..."
    PRETRAIN_MODEL="logs/mindspore_pinn_single_npu0/model.ckpt"
    if [ -f "${PRETRAIN_MODEL}" ]; then
        echo "✅ Found single-card trained model: ${PRETRAIN_MODEL}"
        START_FROM_BEST="--start-from-best"
        export PRETRAIN_MODEL_PATH="${PRETRAIN_MODEL}"
    else
        echo "⚠️  No pretrained model found, starting from scratch"
    fi
fi
# 清理旧的日志
mkdir -p ${LOG_DIR}
# 使用 mpirun 启动4卡分布式训练
echo ""
echo "📊 Training Configuration:"
echo "  - NPU Cards: 4"
echo "  - Per-Card Batch Size: ${BATCH_SIZE}"
echo "  - Total Throughput: $((BATCH_SIZE * 4)) samples/iteration"
echo "  - Learning Rate: ${LR}"
echo "  - Epochs: ${EPOCHS}"
echo "  - Dynamic Weight: ${DYN_WEIGHT}"
echo "  - Algebraic Weight: ${ALG_WEIGHT}"
echo "  - Network: Attention-based (unstacked)"
echo "  - Strategy: Same as single-card best parameters"
echo "  - Log Directory: ${LOG_DIR}"
echo ""
# MindSpore 分布式训练启动命令
# 使用 mpirun 在4个NPU上启动训练
mpirun -n 4 \
    --output-filename ${LOG_DIR}/rank_log \
    --allow-run-as-root \
    python mindspore_pinn.py \
    --distributed \
    --log-dir ${LOG_DIR} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --lr ${LR} \
    --num-train 6000 \
    --num-val 100 \
    --num-test 500 \
    --use-scheduler \
    --scheduler-type plateau \
    --patience 2000 \
    --unstacked \
    --dyn-type attention \
    --alg-type attention \
    --dyn-activation sin \
    --alg-activation sin \
    --dyn-depth 4 \
    --dyn-width 100 \
    --dyn-weight ${DYN_WEIGHT} \
    --alg-weight ${ALG_WEIGHT} \
    --h 0.1 \
    --N 160 \
    --test-every 1000 \
    --use-tqdm \
    ${START_FROM_BEST}

echo ""
echo "========================================================================"
echo "✅ Distributed Training Completed!"
echo "========================================================================"
echo "📁 Results saved to: ${LOG_DIR}/"
echo "📊 Check loss curve: ${LOG_DIR}/loss.png"
echo "📈 Check trajectories: ${LOG_DIR}/trajectories.png"
echo "========================================================================"

#   训练结果：Best at step 20000:
#   Train Loss: 1.238e-05
#   Test Loss: 1.233e-05
#   Test Metrics: []
# 'train' took 6682.037024 s
