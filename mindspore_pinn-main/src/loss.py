import numpy as np
import matplotlib.pyplot as plt

# 加载数据
data = np.load('logs/mindspore_pinn_4npu/loss-history.npz')
steps = data['steps']
train_loss = data['loss_train'].flatten()
test_loss = data['loss_test'].flatten()

# 创建图表
plt.figure(figsize=(10, 7))

# 绘制曲线，使用更深的颜色
plt.semilogy(steps, train_loss, linewidth=3, color='#0000CD', label='Train loss')
plt.semilogy(steps, test_loss, linewidth=3, linestyle='--', color='#B22222', label='Test loss')

plt.xlabel('Iteration', fontsize=16,fontweight='bold')
plt.ylabel('Mean Squared Error', fontsize=16,fontweight='bold')
# 设置坐标轴刻度数字的字号
plt.xticks(fontsize=14)  # X轴刻度数字字号
plt.yticks(fontsize=14)  # Y轴刻度数字字号

plt.legend(fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig('logs/mindspore_pinn_4npu/loss_curve.png', dpi=300, bbox_inches='tight')
plt.show()
