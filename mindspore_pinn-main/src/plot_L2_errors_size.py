#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
L2相对误差可视化脚本
从 L2Relative_error.npz 读取数据，绘制5个变量的误差曲线
绘图风格对齐 enhance_trajectory_plot（基准）
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def stylize_axes(ax, xlabel=None, ylabel=None, legend=True):
    """统一的坐标轴样式（对齐 enhance_trajectory_plot 基准）"""
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=10)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')

    if legend:
        ax.legend(loc='upper right', fontsize=10)


def custom_lineplot(ax, x, y, label=None, xlims=None, ylims=None,
                    color="red", linestyle="--", linewidth=2.5,
                    marker=None, alpha=0.9):
    """自定义线条绘图（对齐 enhance_trajectory_plot 基准）"""
    ax.plot(
        x, y,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        marker=marker,
        label=label,
        alpha=alpha
    )
    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)


def plot_L2_errors(log_dir='logs/mindspore_pinn_4npu'):
    """
    绘制5个变量的L2相对误差图
    完全对齐 enhance_trajectory_plot 的风格
    """
    # 加载L2误差数据
    data_file = os.path.join(log_dir, 'L2Relative_error.npz')
    if not os.path.exists(data_file):
        print(f"❌ 未找到数据文件: {data_file}")
        return

    data = np.load(data_file)
    N = data['N']              # 时间步数向量
    error = data['error']      # 误差矩阵 (N, 5)

    print(f"✅ 加载数据成功: {data_file}")
    print(f"   时间步数: {N.shape}, 误差矩阵: {error.shape}")

    # 变量名（对齐基准的 LaTeX 风格）
    var_names = [r'$\omega_1(t)$', r'$\omega_2(t)$', r'$\delta_2(t)$', r'$\delta_3(t)$', r'$V_3(t)$']

    # 创建5个子图（对齐基准 figsize）
    fig, axes = plt.subplots(5, 1, figsize=(12, 21.5))

    for i in range(5):
        ax = axes[i]
        error_i = error[:, i].reshape(-1, )

        # 误差曲线：按基准 Pred 风格（红色虚线、2.5线宽、alpha=0.9、无marker）
        custom_lineplot(
            ax, N, error_i,
            color="red",
            linestyle="--",
            linewidth=2.5,
            marker=None,
            label=None,
            alpha=0.9
        )

        # 设置 ylabel
        ylabel_text = rf"$L_2$ Relative Error ({var_names[i]})"

        # 只有最后一个子图显示 xlabel（对齐基准）
        xlabel_text = "Time Steps N" if i == 4 else None

        # 应用统一样式（本图不画 legend，和你原脚本一致）
        stylize_axes(ax, xlabel=xlabel_text, ylabel=ylabel_text, legend=False)

    # 布局（对齐基准：tight_layout）
    plt.tight_layout()

    # 保存图片（对齐基准：PDF透明、PNG不透明）
    output_pdf = os.path.join(log_dir, 'L2_errors_all.pdf')
    output_png = os.path.join(log_dir, 'L2_errors_all.png')

    fig.savefig(output_pdf, dpi=300, bbox_inches='tight', transparent=True)
    fig.savefig(output_png, dpi=300, bbox_inches='tight', transparent=False)
    plt.close(fig)

    print(f"✅ L2误差图已保存:")
    print(f"   PDF: {output_pdf}")
    print(f"   PNG: {output_png}")

    # 打印误差统计
    print(f"\n📊 L2相对误差统计:")
    for i, var in enumerate([r'\omega_1', r'\omega_2', r'\delta_2', r'\delta_3', r'V_3']):
        print(f"   {var}: 最大={error[:, i].max():.4e}, 最小={error[:, i].min():.4e}, 平均={error[:, i].mean():.4e}")


if __name__ == '__main__':
    # 只处理 mindspore_pinn_4npu 目录
    log_dir = 'logs/mindspore_pinn_4npu'

    if os.path.exists(log_dir):
        print(f"\n处理目录: {log_dir}")
        plot_L2_errors(log_dir)
        print("\n🎉 完成！")
    else:
        print(f"❌ 目录不存在: {log_dir}")
        print("请确保在正确的工作目录下运行脚本")
