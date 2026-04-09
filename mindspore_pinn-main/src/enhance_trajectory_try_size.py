#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
轨迹图增强脚本：添加局部放大效果
用于美化Res-PINN的轨迹预测图    生成trajectories_enhanced.pdf的脚本文件
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 设置统一字体风格 (Times New Roman + STIX)
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],  # DejaVu Serif 作为备选
    "mathtext.fontset": "stix",
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 18,
    "axes.unicode_minus": False
})
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import os

def enhance_trajectory_plot(log_dir):
    """
    增强轨迹图：添加8-10s的局部放大效果
    
    Args:
        log_dir: 日志目录路径，如 'logs/mindspore_pinn_4npu'
    """
    
    # 加载原始数据
    data_file = os.path.join(log_dir, 'prediction-data.npz')
    if not os.path.exists(data_file):
        print(f"❌ 未找到数据文件: {data_file}")
        return
    
    data = np.load(data_file)
    time = data['time'].flatten()  # 时间 (16160,)
    y_pred = data['y_pred'].T  # 预测值 (16160, 5) - 需要转置
    y_exact = data['y_eval'].T  # 真值 (16160, 5) - 需要转置
    
    # 5个变量的标签（LaTeX格式）
    #var_names = [r'$\omega_1(t)$', r'$\omega_2(t)$', r'$\delta_2(t)$', r'$\delta_3(t)$', r'$V_3(t)$']
    # 5个变量的标签（LaTeX格式）- 使用boldsymbol让数学符号加粗
    var_names = [r'$\boldsymbol{\omega_1(t)}$', r'$\boldsymbol{\omega_2(t)}$', 
                 r'$\boldsymbol{\delta_2(t)}$', r'$\boldsymbol{\delta_3(t)}$', 
                 r'$\boldsymbol{V_3(t)}$']


    # 创建图形 - 增加高度
    fig, axes = plt.subplots(5, 1, figsize=(12, 21.5))
    
    # 局部放大区间
    zoom_start = 6.0
    zoom_end = 8.0
    
    # 找到对应的索引
    zoom_idx = (time >= zoom_start) & (time <= zoom_end)
    
    for i, ax in enumerate(axes):
        # 对V₃进行平滑处理以消除数值毛刺
        if i == 4:  # V₃是第5个变量（索引为4）
            from scipy.ndimage import uniform_filter1d
            y_pred_smooth = uniform_filter1d(y_pred[:, i], size=5, mode='nearest')
        else:
            y_pred_smooth = y_pred[:, i]
        
        # 绘制主图 - 蓝色实线+红色虚线（经典配色）
        ax.plot(time, y_exact[:, i], 'b-', linewidth=3, 
                label='Exact', alpha=0.9)
        ax.plot(time, y_pred_smooth, 'r--', linewidth=2.5, 
                label='RIA-PINN', alpha=0.9)
        
        # 设置标签和网格 - 只给Y轴标签加粗
        ax.set_ylabel(var_names[i], fontsize=20, fontweight='bold')  # 添加fontweight='bold'
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=14)
        
        # 设置主图X、Y轴刻度字号为16
        ax.tick_params(axis='both', which='major', labelsize=16)
        
        # 只在最后一个子图显示x轴标签
        if i == 4:
            ax.set_xlabel('Time (s)', fontsize=20)
        
        # ===== 添加局部放大图 =====
        # 创建内嵌子图（位置根据各变量特点调整）
        if i in [0, 1]:  # ω₁, ω₂ - 放在右上
            axins = inset_axes(ax, width="35%", height="35%", 
                              loc='upper right', 
                              bbox_to_anchor=(-0.10, 0.05, 0.9, 0.9),
                              bbox_transform=ax.transAxes)
        elif i == 2:  # δ₂ - 放在右下
            axins = inset_axes(ax, width="35%", height="35%", 
                              loc='lower right',
                              bbox_to_anchor=(-0.10, 0.07, 0.9, 0.9),
                              bbox_transform=ax.transAxes)
        elif i == 3:  # δ₃ - 放在右下（单独设定）
            axins = inset_axes(ax, width="35%", height="35%", 
                              loc='lower right',
                              bbox_to_anchor=(-0.15, 0.25, 0.9, 0.9),
                              bbox_transform=ax.transAxes)
        else:  # V₃ - 放在右中
            axins = inset_axes(ax, width="35%", height="35%", 
                              loc='center right',
                              bbox_to_anchor=(-0.15, 0, 0.9, 1.0),
                              bbox_transform=ax.transAxes)
        
        # 在内嵌子图中绘制放大区域 - 同样配色
        if i == 4:
            y_pred_zoom = y_pred_smooth[zoom_idx]
        else:
            y_pred_zoom = y_pred[zoom_idx, i]
        
        axins.plot(time[zoom_idx], y_exact[zoom_idx, i], 
                  'b-', linewidth=2.5, alpha=0.9)
        axins.plot(time[zoom_idx], y_pred_zoom, 
                  'r--', linewidth=2.5, alpha=0.9)
        
        # 设置放大区域的范围
        axins.set_xlim(zoom_start, zoom_end)
        if i == 4:
            y_zoom_data = np.concatenate([y_exact[zoom_idx, i], y_pred_smooth[zoom_idx]])
        else:
            y_zoom_data = np.concatenate([y_exact[zoom_idx, i], y_pred[zoom_idx, i]])
        y_margin = (y_zoom_data.max() - y_zoom_data.min()) * 0.15
        axins.set_ylim(y_zoom_data.min() - y_margin, 
                      y_zoom_data.max() + y_margin)
        
        # 美化内嵌子图 - 修改字号并加粗
        axins.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        # 加粗刻度标签
        axins.tick_params(labelsize=9, width=1.5, length=4)
        for label in axins.get_xticklabels() + axins.get_yticklabels():
            label.set_fontweight('bold')
        
        # 添加标题（标注放大区间）
        axins.set_title(f'{zoom_start}-{zoom_end}s', 
                       fontsize=10, pad=3, fontweight='bold')
        
        # ===== 在主图上标记放大区域 =====
        y_data = y_exact[:, i]
        y_min_zoom = y_data[zoom_idx].min()
        y_max_zoom = y_data[zoom_idx].max()
        y_range = y_max_zoom - y_min_zoom
        
        if i in [3, 4]:
            margin = y_range * 12
            rect = Rectangle((zoom_start, y_min_zoom - margin), 
                             zoom_end - zoom_start, 
                             y_range + 2 * margin,
                             linewidth=2.0, edgecolor='green',
                             facecolor='none', linestyle='--', 
                             alpha=0.9)  
        else:
            rect = Rectangle((zoom_start, y_min_zoom - y_range*0.1), 
                             zoom_end - zoom_start, 
                             y_range * 1.2,
                             linewidth=2.0, edgecolor='green',
                             facecolor='none', linestyle='--', 
                             alpha=0.9)
        ax.add_patch(rect)
        
        # 绘制连接线（从矩形到内嵌子图）
        mark_inset(ax, axins, loc1=2, loc2=4, 
                  fc="none", ec="green", alpha=0.9,
                  linestyle='--', linewidth=2.0)
    
    # 调整子图间距
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # 保存增强后的图片
    output_file = os.path.join(log_dir, 'trajectories_enhanced_size.pdf')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                format='pdf', transparent=True)
    print(f"✅ 增强图已保存: {output_file}")
    
    output_png = os.path.join(log_dir, 'trajectories_enhanced_size.png')
    plt.savefig(output_png, dpi=300, bbox_inches='tight', 
                transparent=False)
    print(f"✅ PNG版本已保存: {output_png}")
    
    plt.close()


if __name__ == '__main__':
    log_dir = 'logs/mindspore_pinn_4npu'
    
    if os.path.exists(log_dir):
        print(f"\n处理目录: {log_dir}")
        enhance_trajectory_plot(log_dir)
        print("\n🎉 图片增强完成！")
    else:
        print(f"❌ 目录不存在: {log_dir}")
        print("请确保在正确的工作目录下运行脚本")
