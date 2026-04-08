#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gc
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch

# =========================
# 基准风格（对齐 enhance_trajectory_plot）
# =========================
LABEL_FONTSIZE = 22
TICK_FONTSIZE = 12
LEGEND_FONTSIZE = 12

GRID_ALPHA = 0.3
GRID_LINESTYLE = '--'
GRID_LINEWIDTH = 0.8  # 基准里没显式写，但这里给一个温和值（效果与基准一致）

LINE_ALPHA = 0.9
LW_MAIN = 3.0     # 基准 Exact: 3
LW_PRED = 2.5     # 基准 Pred: 2.5

# 放大图（inset）字号（保持你脚本最终覆盖后的值：9/8）
INSET_TITLE_FONTSIZE = 9
INSET_TICK_FONTSIZE = 8

# 字体设置（确保英文正常显示，无乱码）
plt.rcParams["font.family"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


def init_target_plot_dir(plot_dir="logs/fault_error_plots"):
    """初始化图表保存目录（不存在则创建）"""
    plot_dir = os.path.abspath(plot_dir)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir, exist_ok=True)
        print(f"📁 Plot directory created: {plot_dir}")
    else:
        print(f"📁 Plots will be saved to: {plot_dir}")
    return plot_dir


def calculate_l2_relative_error(y_pred, y_exact):
    """计算L2相对误差（百分比，用于delta2、delta3、V3）"""
    numerator = np.linalg.norm(y_pred - y_exact)
    denominator = np.linalg.norm(y_exact)
    return (numerator / denominator) * 100.0 if denominator > 1e-12 else 0.0


def calculate_l2_absolute_error(y_pred, y_exact):
    """计算L2绝对误差（实际数值偏差，用于w1、w2）"""
    return np.linalg.norm(y_pred - y_exact)


def calculate_errors_after_fault(npz_path, fault_strat=4.0, h=0.1, N_fault=100):
    if not os.path.exists(npz_path):
        print(f"❌ Error: npz file does not exist - {npz_path}")
        return None

    print(f"📂 Reading npz data: {npz_path}")
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print(f"❌ Failed to read npz file: {str(e)}")
        return None

    # 提取数据（time是(20200,1)，需展平为1维）
    time_arr = data['time'].reshape(-1,)  # 关键：展平时间数组
    y_pred = data['y_pred'][:5, :]
    y_exact = data['y_eval'][:5, :]
    var_names = [r'$\omega_1$', r'$\omega_2$', r'$\delta_2$', r'$\delta_3$', r'$V_3$']
    err_type = ['Absolute Error', 'Absolute Error', 'Relative Error (%)', 'Relative Error (%)', 'Relative Error (%)']

    # -------------------- 核心筛选：严格0.1s步长点（排除近似值） --------------------
    time_offset = time_arr - fault_strat
    valid_mask = (time_offset >= -1e-5) & (np.isclose(np.mod(time_offset, h), 0.0, atol=1e-5))
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) < N_fault:
        N_fault = len(valid_indices)
        print(f"⚠️  Warning: Only {N_fault} valid points (strict {h}s step) found (need 100)")
    else:
        print(f"✅ Found {len(valid_indices)} valid points (strict {h}s step) - enough for 100 points")

    selected_indices = valid_indices[:N_fault]
    final_time = time_arr[selected_indices]
    final_pred = y_pred[:, selected_indices]
    final_exact = y_exact[:, selected_indices]
    final_N = len(final_time)

    unique_time_count = len(np.unique(np.round(final_time, 4)))
    print(f"✅ Final selected: {final_N} points | Unique time points: {unique_time_count}")
    print(f"   Time range: {final_time[0]:.4f}s ~ {final_time[-1]:.4f}s (step={h}s)")

    # -------------------- 后续误差计算（不变） --------------------
    pointwise_err = np.zeros((5, final_N))
    for i in [0, 1]:
        pointwise_err[i] = np.abs(final_pred[i] - final_exact[i])

    eps = 1e-12
    for i in [2, 3, 4]:
        pointwise_err[i] = np.abs(final_pred[i] - final_exact[i]) / (np.abs(final_exact[i]) + eps) * 100.0

    # 打印逐点误差（此时无重复）
    print(f"\n📌 Post-fault {final_N}-step errors (strict {h}s step, no duplicates):")
    for i in range(5):
        print(f"\n[Variable {var_names[i]} ({err_type[i]})]")
        for step_idx in range(final_N):
            time_val = round(final_time[step_idx], 2)  # 显示2位小数，统一格式
            err_val = pointwise_err[i, step_idx]
            if i in [0, 1]:
                print(f"[{step_idx+1:3d}] {time_val:.2f}s: {err_val:.6f}", end='  ')
            else:
                print(f"[{step_idx+1:3d}] {time_val:.2f}s: {err_val:.4f}%", end='  ')
            if (step_idx + 1) % 5 == 0:
                print()
        print()

    # 统计量
    l2_err, max_err, mean_err = [], [], []
    for i in range(5):
        if i in [0, 1]:
            l2_err.append(calculate_l2_absolute_error(final_pred[i], final_exact[i]))
        else:
            l2_err.append(calculate_l2_relative_error(final_pred[i], final_exact[i]))
        max_err.append(np.max(pointwise_err[i]))
        mean_err.append(np.mean(pointwise_err[i]))

    return {
        'time': final_time,
        'var_names': var_names,
        'err_type': err_type,
        'pointwise_err': pointwise_err,
        'l2_err': l2_err,
        'max_err': max_err,
        'mean_err': mean_err,
        'fault_strat': fault_strat,
        'h': h,
        'N_steps': final_N,
        'actual_end_time': final_time[-1] if final_N > 0 else 0.0
    }

def add_axes_background(ax, color='#FE9B1C', alpha=0.08):
    """
    给整个子图添加淡色背景（不影响数据和网格）
    """
    rect = Rectangle(
        (0, 0), 1, 1,
        transform=ax.transAxes,
        facecolor=color,
        edgecolor='none',
        alpha=alpha,
        zorder=0
    )
    ax.add_patch(rect)
def plot_error_statistics(error_data, save_dir):
    """
    生成完整误差图表（适配4.0s~20.0s范围，100个唯一数据点，颜色加深）
    包含3个子图：w1/w2绝对误差折线图、delta2/delta3/V3相对误差折线图、最大误差对比柱状图
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    plot_filename = f"{timestamp}_fault_error_4.0-20.0s.png"
    save_path = os.path.join(save_dir, plot_filename)

    time_arr = error_data['time']
    pointwise_err = error_data['pointwise_err']
    var_names = error_data['var_names']
    err_type = error_data['err_type']
    fault_strat = error_data['fault_strat']
    actual_end_time = error_data['actual_end_time']
    target_start = 4.0
    target_end = 20.0

    markers = ['o', '^', 's', '*', 'D']
    colors = ['#2ca02c', '#d62728', '#ff7f0e', '#1f77b4', '#9467bd']

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1,
        figsize=(14, 18),
        dpi=300,
        tight_layout=True
    )

    # -------------------- 子图1：w1、w2 绝对误差 --------------------
    time_sorted_idx = np.argsort(time_arr)
    time_sorted = time_arr[time_sorted_idx]

    for i in [0, 1]:
        err_sorted = pointwise_err[i, time_sorted_idx]
        ax1.plot(
            time_sorted, err_sorted,
            color=colors[i],
            linewidth=2,
            label=f"{var_names[i]} ({err_type[i]})",
            alpha=0.8,
            marker=markers[i],
            markersize=6,
            markevery=4
        )

        max_err_idx = np.argmax(err_sorted)
        max_err_val = err_sorted[max_err_idx]
        max_err_time = time_sorted[max_err_idx]

        if i == 0:
            x_offset, y_offset = 15, 25
        else:
            x_offset, y_offset = 15, -40

        ax1.annotate(
            f'{var_names[i]} Max: {max_err_val:.6f}',
            xy=(max_err_time, max_err_val),
            xytext=(x_offset, y_offset),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9),
            arrowprops=dict(
                arrowstyle='->',
                connectionstyle='arc3,rad=0.1',
                color='black',
                linewidth=1.2
            ),
            fontsize=11,
            color=colors[i],
            fontweight='bold',
            zorder=10
        )

    ax1.set_xlim([target_start, target_end])
    ax1.set_xticks(np.arange(target_start, target_end + 0.1, 2.0))

    y1_max = max([np.max(pointwise_err[i, time_sorted_idx]) for i in [0, 1]])
    ax1.set_ylim(0, y1_max * 1.35)


    ax1.axvline(x=fault_strat, color='gray', linestyle='--', linewidth=2)
    

    ax1.set_xlabel('Time (s)', fontsize=LABEL_FONTSIZE, fontweight='bold')
    ax1.set_ylabel('Absolute Error', fontsize=LABEL_FONTSIZE, fontweight='bold')
    ax1.tick_params(axis='both', which='major', labelsize=20)
    add_axes_background(ax1)

    fault_patch = Patch(facecolor='#FE9B1C', alpha=0.3, edgecolor='black', label=f'Fault Injection ({fault_strat}s)')
    handles, labels = ax1.get_legend_handles_labels()
    handles.append(fault_patch)
    ax1.legend(handles=handles, loc='upper right', fontsize=14, framealpha=0.9, markerscale=1.8)

    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)

        # inset 1
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    axins1 = inset_axes(ax1, width="70%", height="70%", loc='center',
                        bbox_to_anchor=(0.15, 0.25, 0.70, 0.70), bbox_transform=ax1.transAxes)

    zoom_start = 9.0
    zoom_end = 11.0
    zoom_mask = (time_sorted >= zoom_start) & (time_sorted <= zoom_end)

    for i in [0, 1]:
        err_sorted = pointwise_err[i, time_sorted_idx]
        axins1.plot(time_sorted[zoom_mask], err_sorted[zoom_mask],
                    color=colors[i], linewidth=2, alpha=0.8,
                    marker=markers[i], markersize=5, markevery=2)

    axins1.set_xlim(zoom_start, zoom_end)
    axins1.set_title(f'{zoom_start}-{zoom_end}s', fontsize=12, pad=3,fontweight='bold')
    axins1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    axins1.tick_params(labelsize=10)

    # ✅ 仅加粗放大图刻度数字（不改字号）
    for label in axins1.get_xticklabels() + axins1.get_yticklabels():
        label.set_fontweight('bold')

    mark_inset(ax1, axins1, loc1=2, loc2=4, fc="none", ec="green",alpha=0.9,
               linestyle='--', linewidth=2)

    # -------------------- 子图2：delta2、delta3、V3 相对误差 --------------------
    for i in [2, 3, 4]:
        err_sorted = pointwise_err[i, time_sorted_idx]
        ax2.plot(
            time_sorted, err_sorted,
            color=colors[i],
            linewidth=2.5,
            label=f"{var_names[i]} ({err_type[i]})",
            alpha=1.0,
            marker=markers[i],
            markersize=7,
            markevery=4
        )

        max_err_idx = np.argmax(err_sorted)
        max_err_val = err_sorted[max_err_idx]
        max_err_time = time_sorted[max_err_idx]

        ax2.scatter(max_err_time, max_err_val, color='black', s=60, zorder=5)

        ax2.annotate(
            f'{var_names[i]} Max: {max_err_val:.2f}%',
            xy=(max_err_time, max_err_val),
            xytext=(8, 8),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9),
            fontsize=11,
            color=colors[i],
            fontweight='bold'
        )

    ax2.set_xlim([target_start, target_end])
    ax2.set_xticks(np.arange(target_start, target_end + 0.1, 2.0))

    y2_max = max([np.max(pointwise_err[i, time_sorted_idx]) for i in [2, 3, 4]])
    ax2.set_ylim(0, y2_max * 1.2)

    ax2.axvline(x=fault_strat, color='gray', linestyle='--', linewidth=2)

    ax2.set_xlabel('Time (s)', fontsize=LABEL_FONTSIZE, fontweight='bold')
    ax2.set_ylabel('Relative Error (%)', fontsize=LABEL_FONTSIZE, fontweight='bold')
    ax2.tick_params(axis='both', which='major', labelsize=20)
    add_axes_background(ax2)

    fault_patch = Patch(facecolor='#FE9B1C', alpha=0.3, edgecolor='black', label=f'Fault Injection ({fault_strat}s)')
    handles, labels = ax2.get_legend_handles_labels()
    handles.append(fault_patch)
    ax2.legend(handles=handles, loc='upper right', fontsize=14, framealpha=0.9, markerscale=1.8)
    #ax2.legend(loc='upper right', fontsize=14, framealpha=0.9, markerscale=1.8)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)

    # inset 2
    axins2 = inset_axes(ax2, width="70%", height="70%", loc='center',
                        bbox_to_anchor=(0.15, 0.25, 0.70, 0.70), bbox_transform=ax2.transAxes)

    for i in [2, 3, 4]:
        err_sorted = pointwise_err[i, time_sorted_idx]
        axins2.plot(time_sorted[zoom_mask], err_sorted[zoom_mask],
                    color=colors[i], linewidth=2.5, alpha=1.0,
                    marker=markers[i], markersize=6, markevery=2)

    axins2.set_xlim(zoom_start, zoom_end)
    axins2.set_title(f'{zoom_start}-{zoom_end}s', fontsize=12, pad=3,fontweight='bold')
    axins2.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    axins2.tick_params(labelsize=10)

    # ✅ 仅加粗放大图刻度数字（不改字号）
    for label in axins2.get_xticklabels() + axins2.get_yticklabels():
        label.set_fontweight('bold')

    mark_inset(ax2, axins2, loc1=2, loc2=4, fc="none", ec="green",alpha=0.9,
               linestyle='--', linewidth=2)
    # -------------------- 子图3：最大误差柱状图 --------------------
    x_pos = np.arange(5)
    bar_width = 0.35

    bars = ax3.bar(
        x_pos,
        error_data['max_err'],
        width=bar_width,
        color=colors,
        alpha=1.0
    )

    for i, bar in enumerate(bars):
        bar_height = bar.get_height()
        if i in [0, 1]:
            label_text = f'{bar_height:.6f}'
        else:
            label_text = f'{bar_height:.2f}%'

        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar_height + (0.02 if i in [0, 1] else 0.05),
            label_text,
            ha='center',
            va='bottom',
            fontsize=14,
            fontweight='bold'
        )

    # ⚠️ 这里按你的原逻辑，只是保持你已经改过的字号写法，不再动语义
    ax3.set_xlabel('Time (s)', fontsize=LABEL_FONTSIZE, fontweight='bold')
    ax3.set_ylabel('Maximum Error', fontsize=LABEL_FONTSIZE, fontweight='bold')
    ax3.tick_params(axis='both', which='major', labelsize=20)

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(
        [f'{var}\n({err_type[i]})' for i, var in enumerate(var_names)],
        fontsize=12,
        rotation=0,
        fontweight='bold'
    )

    max_bar_height = max(error_data['max_err'])
    ax3.set_ylim(0, max_bar_height * 1.1)
    add_axes_background(ax3)
    ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.8, axis='y')

    plt.tight_layout(pad=3.0)
    # 保存PNG
    fig.savefig(
        save_path,
        bbox_inches='tight',
        facecolor='white',
        dpi=300
    )

    # 保存PDF
    pdf_path = save_path.replace('.png', '.pdf')
    fig.savefig(
        pdf_path,
        bbox_inches='tight',
        facecolor='white'
    )

    plt.close(fig)
    gc.collect()

    print(f"\n✅ Error plot saved successfully!")
    print(f"   Time range: {target_start:.1f}s ~ {actual_end_time:.1f}s | Data points: {error_data['N_steps']}")
    print(f"   PNG path: {os.path.abspath(save_path)}")
    print(f"   PDF path: {os.path.abspath(pdf_path)}")

    return save_path



def print_error_report(error_data):
    """打印误差报告（适配到20s的时间范围）"""
    print("\n" + "="*80)
    print("📋 Post-Fault Error Analysis Report (w1/w2: Absolute Error; others: Relative Error)")
    print("="*80)
    print(f"Fault Injection:      {error_data['fault_strat']}s")
    print(f"Target end time: 20.0s")
    print(f"Actual end time: {error_data['actual_end_time']:.2f}s")
    print(f"Time step:       {error_data['h']}s")
    print(f"Unique steps:    {error_data['N_steps']} (target: 100)")
    print(f"Data range:      {error_data['time'][0]:.2f}s ~ {error_data['actual_end_time']:.2f}s")
    print("="*80)

    print(f"\n{'Variable':<12} {'Error Type':<15} {'L2 Error':<20} {'Max Error':<20} {'Mean Error':<20}")
    print("-"*87)
    for i in range(5):
        var_name = error_data['var_names'][i]
        et = error_data['err_type'][i]
        if i in [0, 1]:
            l2_str = f"{error_data['l2_err'][i]:.6f}"
            max_str = f"{error_data['max_err'][i]:.6f}"
            mean_str = f"{error_data['mean_err'][i]:.6f}"
        else:
            l2_str = f"{error_data['l2_err'][i]:.4f}%"
            max_str = f"{error_data['max_err'][i]:.4f}%"
            mean_str = f"{error_data['mean_err'][i]:.4f}%"
        print(f"{var_name:<12} {et:<15} {l2_str:<20} {max_str:<20} {mean_str:<20}")
    print("-"*87)

    print("\nNote: w1 and w2 use absolute error (due to small values); others use relative error (%)")
    print("="*80 + "\n")


def main():
    print("="*80)
    print("🔬 Post-Fault Error Calculation (4.0s+ | 0.1s step | 100 points)")
    print("="*80)

    target_save_dir = init_target_plot_dir("logs/fault_error_plots")
    npz_data_path = os.path.abspath("./logs/fault-b5-finetune-step2/prediction-data.npz")
    print(f"npz file path: {npz_data_path}")

    error_data = calculate_errors_after_fault(
        npz_path=npz_data_path,
        fault_strat=4.0,
        h=0.1,
        N_fault=100
    )

    if error_data:
        print_error_report(error_data)
        plot_error_statistics(error_data, target_save_dir)
        print("🎉 All tasks completed!")
        print(f"✅ Final time range: {error_data['time'][0]:.2f}s ~ {error_data['time'][-1]:.2f}s")
    else:
        print("\n❌ Task failed. Check npz data!")
    print("="*80)


if __name__ == "__main__":
    main()
