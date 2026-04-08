#!/bin/bash
# ========================================================================
# MindSpore 单卡 NPU 训练启动脚本
# 支持选择指定的NPU卡进行训练
# ========================================================================

set -e

# 默认使用 NPU-0，可通过参数修改
DEVICE_ID=${1:-0}

echo "========================================================================"
echo "🚀 Starting Single-NPU Training"
echo "========================================================================"

# 训练参数配置
EPOCHS=10000
BATCH_SIZE=128
LR=1e-3
LOG_DIR="logs/mindspore_pinn_single_npu${DEVICE_ID}"

# 检查是否从最佳模型开始
START_FROM_BEST=""
if [ -f "${LOG_DIR}/model.pth" ]; then
    echo "✅ Found existing model, will start from best checkpoint"
    START_FROM_BEST="--start-from-best"
fi

# 清理旧的日志
mkdir -p ${LOG_DIR}

echo ""
echo "📊 Training Configuration:"
echo "  - NPU Card: ${DEVICE_ID}"
echo "  - Batch Size: ${BATCH_SIZE}"
echo "  - Learning Rate: ${LR}"
echo "  - Epochs: ${EPOCHS}"
echo "  - Log Directory: ${LOG_DIR}"
echo ""

# 单卡训练启动命令
python mindspore_pinn.py \
    --device-id ${DEVICE_ID} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --lr ${LR} \
    --log-dir ${LOG_DIR} \
    --test-every 500 \
    --use-tqdm \
    ${START_FROM_BEST}

echo ""
echo "========================================================================"
echo "✅ Training Completed!"
echo "========================================================================"
echo "📁 Results saved to: ${LOG_DIR}/"
echo "📊 Check loss curve: ${LOG_DIR}/loss.png"
echo "📈 Check trajectories: ${LOG_DIR}/trajectories.png"
echo "========================================================================"
