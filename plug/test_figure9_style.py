"""
快速预览 Figure 9 样式效果的脚本
使用模拟数据，无需运行完整仿真
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# 模拟数据
np.random.seed(42)
timesteps_to_study = [0.004, 0.006, 0.008, 0.012, 0.018, 0.020, 0.022, 0.030, 0.032, 0.036, 0.040]

# 模拟两个状态变量的误差数据
# 状态8 (delta') 和 状态10 (V)
error_dist_pure_8 = 0.01 * np.power(timesteps_to_study, np.array(timesteps_to_study)*50) * (1 + 0.1*np.random.randn(len(timesteps_to_study)))
error_dist_hybrid_8 = 0.005 * np.power(timesteps_to_study, np.array(timesteps_to_study)*30) * (1 + 0.1*np.random.randn(len(timesteps_to_study)))
error_dist_pure_10 = 0.008 * np.power(timesteps_to_study, np.array(timesteps_to_study)*40) * (1 + 0.1*np.random.randn(len(timesteps_to_study)))
error_dist_hybrid_10 = 0.003 * np.power(timesteps_to_study, np.array(timesteps_to_study)*25) * (1 + 0.1*np.random.randn(len(timesteps_to_study)))

# 上须值数据 (约为中位数的1.5-2倍)
upper_whisker_pure_8 = error_dist_pure_8 * (1.8 + 0.2*np.random.rand(len(timesteps_to_study)))
upper_whisker_hybrid_8 = error_dist_hybrid_8 * (1.8 + 0.2*np.random.rand(len(timesteps_to_study)))
upper_whisker_pure_10 = error_dist_pure_10 * (1.8 + 0.2*np.random.rand(len(timesteps_to_study)))
upper_whisker_hybrid_10 = error_dist_hybrid_10 * (1.8 + 0.2*np.random.rand(len(timesteps_to_study)))

# 绘图
plotting_states_final = [8, 10]
fig = plt.figure(figsize=(14, 5))
gs = gridspec.GridSpec(1, 2)
axes_list = []

for i, state_plot in enumerate(plotting_states_final):
    ax = plt.subplot(gs[0, i])
    axes_list.append(ax)
    ax.grid()
    
    if state_plot == 8:
        median_pure = error_dist_pure_8
        median_hybrid = error_dist_hybrid_8
        upper_pure = upper_whisker_pure_8
        upper_hybrid = upper_whisker_hybrid_8
    else:
        median_pure = error_dist_pure_10
        median_hybrid = error_dist_hybrid_10
        upper_pure = upper_whisker_pure_10
        upper_hybrid = upper_whisker_hybrid_10
    
    ax.plot(timesteps_to_study, median_pure, color='b', linestyle='-', linewidth=3, alpha=0.9, label='Pure solver')
    ax.plot(timesteps_to_study, median_hybrid, color='r', linestyle='-', linewidth=2.5, alpha=0.9, label='Hybrid solver')
    ax.plot(timesteps_to_study, upper_pure, color='b', linestyle='--', alpha=0.6)
    ax.plot(timesteps_to_study, upper_hybrid, color='r', linestyle='--', alpha=0.6)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\mathbf{\Delta t\ [s]}$', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    # 设置x轴显示所有时间步长刻度
    ax.set_xticks(timesteps_to_study)
    ax.set_xticklabels([f'{dt:.3f}' for dt in timesteps_to_study], rotation=60, ha='right', fontsize=11)
    
    # 设置y轴显示更多刻度（对数尺度）
    from matplotlib.ticker import LogLocator, FuncFormatter
    ax.yaxis.set_major_locator(LogLocator(numticks=15))
    ax.yaxis.set_minor_locator(LogLocator(subs='auto', numticks=15))

# 调整子图底部边距以容纳旋转的x轴标签
plt.subplots_adjust(bottom=0.2)

# 设置y轴标签（加粗）
axes_list[0].set_ylabel(r"$\boldsymbol{|\delta'_3 - \hat{\delta}'_3|} \ [\mathrm{rad}]$", fontsize=20)
axes_list[1].set_ylabel(r"$\boldsymbol{|V_3 - \hat{V}_3|} \ [\mathrm{p.u.}]$", fontsize=20)

# 添加统一的图例在顶部居中
legend_elements = [
    Line2D([0], [0], color='b', linestyle='-', linewidth=3, label='Pure solver'),
    Line2D([0], [0], color='b', linestyle='--', alpha=0.6, label='Upper whisker'),
    Line2D([0], [0], color='r', linestyle='-', linewidth=2.5, label='Hybrid solver'),
    Line2D([0], [0], color='r', linestyle='--', alpha=0.6, label='Upper whisker')
]
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=4, fontsize=14, frameon=True, edgecolor='black', fancybox=False)

plt.tight_layout()
import os
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/Figure_9_style_test.png', dpi=150, bbox_inches='tight')
plt.savefig('outputs/Figure_9_style_test.pdf', format='pdf', bbox_inches='tight')
print('样式测试图已保存：outputs/Figure_9_style_test.png / .pdf')
plt.close()

# ============================================================
# 箱线图可视化版本
# ============================================================
# 为每个时间步生成模拟的误差分布数据（中位数+上须值是之前的简化数据）
# 这里模拟完整的箱线图数据：最小值、Q1、中位数、Q3、最大值

# 生成每个时间步的多个样本（用于箱线图）
n_samples = 30  # 每个时间步的样本数

# 为状态8 (delta') 生成样本数据
error_samples_pure_8 = []
error_samples_hybrid_8 = []
for i, dt in enumerate(timesteps_to_study):
    base_pure = error_dist_pure_8[i]
    base_hybrid = error_dist_hybrid_8[i]
    # 生成正态分布样本
    samples_pure = base_pure * (1 + 0.3*np.random.randn(n_samples))
    samples_hybrid = base_hybrid * (1 + 0.3*np.random.randn(n_samples))
    error_samples_pure_8.append(samples_pure)
    error_samples_hybrid_8.append(samples_hybrid)

# 为状态10 (V) 生成样本数据
error_samples_pure_10 = []
error_samples_hybrid_10 = []
for i, dt in enumerate(timesteps_to_study):
    base_pure = error_dist_pure_10[i]
    base_hybrid = error_dist_hybrid_10[i]
    samples_pure = base_pure * (1 + 0.3*np.random.randn(n_samples))
    samples_hybrid = base_hybrid * (1 + 0.3*np.random.randn(n_samples))
    error_samples_pure_10.append(samples_pure)
    error_samples_hybrid_10.append(samples_hybrid)

# 绘制箱线图
fig2 = plt.figure(figsize=(16, 6))
gs2 = gridspec.GridSpec(1, 2)

for i, state_plot in enumerate(plotting_states_final):
    ax = plt.subplot(gs2[0, i])
    ax.grid(axis='y', alpha=0.3)
    
    if state_plot == 8:
        samples_pure = error_samples_pure_8
        samples_hybrid = error_samples_hybrid_8
        title = r"$\delta'_3$ Prediction Error"
    else:
        samples_pure = error_samples_pure_10
        samples_hybrid = error_samples_hybrid_10
        title = r"$V_3$ Prediction Error"
    
    # 准备箱线图数据 - 每个时间步有两个箱线（Pure和Hybrid并排）
    positions = []
    box_data = []
    colors = []
    
    for j, dt in enumerate(timesteps_to_study):
        # Pure solver box
        positions.append(j * 3)
        box_data.append(samples_pure[j])
        colors.append('steelblue')
        # Hybrid solver box
        positions.append(j * 3 + 1)
        box_data.append(samples_hybrid[j])
        colors.append('indianred')
    
    # 绘制箱线图
    bp = ax.boxplot(box_data, positions=positions, widths=0.8, patch_artist=True,
                    showfliers=False,  # 不显示离群点
                    medianprops=dict(color='black', linewidth=1.5),
                    whiskerprops=dict(linewidth=1),
                    capprops=dict(linewidth=1))
    
    # 设置箱体颜色
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_yscale('log')
    ax.set_xlabel(r'$\mathbf{\Delta t\ [s]}$', fontsize=18)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # 设置x轴刻度标签（显示在每对箱线图中间）
    tick_positions = [j * 3 + 0.5 for j in range(len(timesteps_to_study))]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f'{dt:.3f}' for dt in timesteps_to_study], rotation=60, ha='right', fontsize=10)
    ax.set_xlim(-1, len(timesteps_to_study) * 3 - 0.5)

# 设置y轴标签
axes_box = fig2.get_axes()
axes_box[0].set_ylabel(r"$\boldsymbol{|\delta'_3 - \hat{\delta}'_3|} \ [\mathrm{rad}]$", fontsize=18)
axes_box[1].set_ylabel(r"$\boldsymbol{|V_3 - \hat{V}_3|} \ [\mathrm{p.u.}]$", fontsize=18)

# 添加图例
legend_elements_box = [
    Line2D([0], [0], color='steelblue', linewidth=10, alpha=0.7, label='Pure solver'),
    Line2D([0], [0], color='indianred', linewidth=10, alpha=0.7, label='Hybrid solver')
]
fig2.legend(handles=legend_elements_box, loc='upper center', bbox_to_anchor=(0.5, 1.02), 
            ncol=2, fontsize=14, frameon=True, edgecolor='black', fancybox=False)

plt.tight_layout()
plt.subplots_adjust(bottom=0.2, top=0.88)
plt.savefig('outputs/Figure_9_boxplot_test.png', dpi=150, bbox_inches='tight')
plt.savefig('outputs/Figure_9_boxplot_test.pdf', format='pdf', bbox_inches='tight')
print('箱线图已保存：outputs/Figure_9_boxplot_test.png / .pdf')
plt.close()
