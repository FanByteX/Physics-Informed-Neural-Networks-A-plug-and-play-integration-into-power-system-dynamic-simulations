"""
故障时刻 t=4s 的混合PINN可视化 (正确版本)
==========================================
参照 example_powerNet.py 和 fault_powerNet_b5.py 的方法
- 0-4s: 使用 dae-pinns-best 模型 (b=10)
- 4-10s: 使用 fault-b5-finetune-step2 模型 (b=5)
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import deepxde as dde
# 新增导入：垃圾回收与文件清理
import gc
import shutil
import glob

from utils.utils import dotdict
from models.DAEnn import three_bus_PN
from data.DAE import dae_data
from supervisor import supervisor
from metrics import l2_relative_error

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 运行前清理函数（仅清理与本脚本相关的图像与缓存，不删除npz/pth）
def clean_previous_logs(logs_dir="./logs"):
    if not os.path.exists(logs_dir):
        return
    removed = []
    # 目标清理模式（与本脚本输出相关）
    patterns = [
        "fault_at_4s_comparison*.png",
        "fault_at_4s_error_100*.png",   # 其他脚本生成的相关错误图
        "*_tmp.png",
        "*.pdf",
        "*.tmp",
        "*.cache",
    ]
    for pat in patterns:
        for fpath in glob.glob(os.path.join(logs_dir, pat)):
            # 保护数据与模型文件
            if fpath.endswith(".npz") or fpath.endswith(".pth"):
                continue
            try:
                if os.path.isdir(fpath):
                    shutil.rmtree(fpath, ignore_errors=True)
                else:
                    os.remove(fpath)
                removed.append(os.path.basename(fpath))
            except Exception as e:
                print(f"警告: 删除失败 {fpath}: {e}")
    # 清理 __pycache__
    pycache_dir = os.path.join(logs_dir, "__pycache__")
    if os.path.isdir(pycache_dir):
        try:
            shutil.rmtree(pycache_dir, ignore_errors=True)
            removed.append("__pycache__")
        except Exception as e:
            print(f"警告: 删除失败 {pycache_dir}: {e}")
    if removed:
        print("🧹 已移除旧图与缓存: " + ", ".join(sorted(set(removed))))
    else:
        print("ℹ️ 未发现需清理的旧图或缓存。")
    # 关闭所有Matplotlib图形并触发垃圾回收
    plt.close('all')
    gc.collect()


def scipy_integrate(func, X0, h, IRK_times, N=0, method='BDF'):
    """
    使用scipy积分 (带错误检查和自动降级策略)
    """
    V0 = 0.7
    t_span = [0.0, h * N]
    t_sim = np.array([t_span[0]])
    
    for k in range(1, N + 1):
        temp = (k - 1) * h + IRK_times * h
        t_sim = np.vstack((t_sim, temp))
        t_next = np.array([k * h])
        t_sim = np.vstack((t_sim, t_next))
    
    print(f"  求解时间范围: {t_span[0]:.2f}s - {t_span[1]:.2f}s, 步数: N={N}, 步长: h={h}")
    
    # 首先尝试Radau方法（对刚性问题更稳定）
    try:
        print(f"  正在使用Radau方法求解...")
        sol = solve_ivp(func, t_span, [X0[0], X0[1], X0[2], X0[3], V0], 
                        method='Radau', 
                        t_eval=t_sim.reshape(-1,),
                        rtol=1e-7,      # 更严格的相对容差
                        atol=1e-10,     # 更严格的绝对容差
                        max_step=h/5)   # 限制最大步长为h/5
        
        # 检查是否有NaN
        if np.isnan(sol.y).any():
            nan_idx = np.where(np.isnan(sol.y).any(axis=0))[0][0]
            print(f"  ⚠️  Radau方法在 t≈{sol.t[nan_idx]:.2f}s 处产生NaN")
            print(f"  尝试使用更小的步长限制...")
            
            # 再次尝试，使用更严格的步长限制
            sol = solve_ivp(func, t_span, [X0[0], X0[1], X0[2], X0[3], V0], 
                            method='Radau', 
                            t_eval=t_sim.reshape(-1,),
                            rtol=1e-8, 
                            atol=1e-11,
                            max_step=h/10,  # 进一步限制步长
                            first_step=h/100)  # 设置初始步长
            
        if sol.success:
            print(f"  ✅ Radau方法求解成功")
        else:
            print(f"  ⚠️  Radau方法状态: {sol.message}")
            print(f"  求解到时间: t={sol.t[-1]:.2f}s (目标: {t_span[1]:.2f}s)")
        
    except Exception as e:
        print(f"  ❌ Radau方法失败: {e}")
        print(f"  尝试使用LSODA方法（自动刚性检测）...")
        
        # 降级到LSODA（自动在刚性/非刚性方法间切换）
        sol = solve_ivp(func, t_span, [X0[0], X0[1], X0[2], X0[3], V0], 
                        method='LSODA', 
                        t_eval=t_sim.reshape(-1,),
                        rtol=1e-7, 
                        atol=1e-10,
                        max_step=h/5)
        
        print(f"  LSODA方法状态: {'成功' if sol.success else sol.message}")
    
    # 检查最终结果
    if np.isnan(sol.y).any():
        nan_indices = np.where(np.isnan(sol.y).any(axis=0))[0]
        print(f"  ⚠️  警告: 仍有 {len(nan_indices)} 个时间点包含NaN")
        if len(nan_indices) > 0:
            first_nan = nan_indices[0]
            print(f"  ⚠️  第一个NaN: 索引={first_nan}, t≈{sol.t[first_nan]:.2f}s")
            # 输出NaN前后的值
            if first_nan > 0:
                print(f"      NaN前一个时间点: t={sol.t[first_nan-1]:.4f}s")
                print(f"      状态值: {sol.y[:, first_nan-1]}")
    else:
        print(f"  ✅ 无NaN值，数据完整")
    
    y_test = sol.y
    return t_sim[1:, :], y_test[:, 1:]


def plot_hybrid_trajectories(t_normal, y_normal_exact, y_normal_pred,
                             t_fault, y_fault_exact, y_fault_pred,
                             fault_time, save_path):
    """
    绘制混合轨迹图（深蓝色实线 vs 橙红色虚线）
    """
    # 合并时间和数据
    t_all = np.vstack((t_normal, t_fault))
    y_exact_all = np.hstack((y_normal_exact, y_fault_exact))
    y_pred_all = np.hstack((y_normal_pred, y_fault_pred))
    
    t_flat = t_all.reshape(-1,)
    
    ylims = [None, None, None, None, (0.2, 1.2)]
    xlabel = [None, None, None, None, 'Time (s)']
    ylabel = [r'$\omega_1(t)$', r'$\omega_2(t)$', r'$\delta_2(t)$', r'$\delta_3(t)$', r'$V_3(t)$']
    fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(12, 22.25))    
    for i in range(5):
        # 蓝色实线 - Exact
        ax[i].plot(t_flat, y_exact_all[i, :].reshape(-1,), 
                   color='b', linewidth=3, linestyle='-', 
                   label='Exact', zorder=2)
        
        # 红色虚线 - Predicted
        if i in [2, 3]:
            indices = np.arange(0, len(t_flat), 8)
            ax[i].plot(t_flat[indices], y_pred_all[i, indices], 
                       color='r', linewidth=2.5, linestyle='--',
                       label='Predicted (Res-PINN)', zorder=3)
        else:
            ax[i].plot(t_flat, y_pred_all[i, :].reshape(-1,), 
                       color='r', linewidth=2.5, linestyle='--',
                       label='Predicted (Res-PINN)', zorder=3)
        
        # 故障时刻标注线
        ax[i].axvline(x=fault_time, color='blue', linestyle=':', 
                      linewidth=3, alpha=0.8, label='Fault @ t=4s', zorder=1)
        
        # 故障区域阴影
        ax[i].axvspan(fault_time, t_flat.max(), alpha=0.08, color='#FE9B1C',
                      label='Fault Region (b=5)' if i == 0 else '')
        
        # 设置坐标轴标签 - 字号扩大2号：20→22
        if ylims[i] is not None:
            ax[i].set_ylim(ylims[i])
        if xlabel[i] is not None:
            ax[i].set_xlabel(xlabel[i], fontsize=22, fontweight='bold')  # 20→22
        ax[i].set_ylabel(ylabel[i], fontsize=22, fontweight='bold')  # 20→22
        
        # 设置主图X、Y轴刻度字号 - 字号扩大2号：16→18
        ax[i].tick_params(axis='both', which='major', labelsize=18)  # 16→18
        
        # 图例 - 保持14不变
        # 图例 - 保持14不变，调整位置避免遮挡
        if i == 0:  # 第1个子图：往下移
            ax[i].legend(fontsize=14, loc='lower right', framealpha=0.9,
                        bbox_to_anchor=(0.82, 0.05))
        elif i in [2, 4]:  # 第3、5子图：往右移
            ax[i].legend(fontsize=14, loc='lower right', framealpha=0.9,
                        bbox_to_anchor=(0.95, 0.10))
        else:  # 第2、4子图保持原样
            ax[i].legend(fontsize=14, loc='lower right', framealpha=0.9,
                        bbox_to_anchor=(0.82, 0.10))
        
        ax[i].grid(True, alpha=0.3)
        
        # 在4s故障时刻添加局部放大图
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
        
        if i in [2, 4]:
            axins = inset_axes(ax[i], width="54%", height="54%", 
                              loc='center',
                              bbox_to_anchor=(0.25, 0.12, 0.54, 0.54), 
                              bbox_transform=ax[i].transAxes)
        else:
            axins = inset_axes(ax[i], width="35%", height="35%", 
                              loc='upper right',
                              bbox_to_anchor=(0.05, 0.05, 0.9, 0.9), 
                              bbox_transform=ax[i].transAxes)
        
        # 放大区域：3.5s - 4.5s
        t_zoom_mask = (t_flat >= 3.5) & (t_flat <= 4.5)
        t_zoom = t_flat[t_zoom_mask]
        
        # 绘制放大区域的数据
        axins.plot(t_zoom, y_exact_all[i, t_zoom_mask], 'b-', linewidth=2)
        if i in [2, 3]:
            zoom_indices = np.arange(0, len(t_flat), 8)[np.isin(np.arange(0, len(t_flat), 8), np.where(t_zoom_mask)[0])]
            axins.plot(t_flat[zoom_indices], y_pred_all[i, zoom_indices], 'r--', linewidth=1.5)
        else:
            axins.plot(t_zoom, y_pred_all[i, t_zoom_mask], 'r--', linewidth=1.5)
        
        # 故障时刻标注
        axins.axvline(x=fault_time, color='blue', linestyle=':', linewidth=2, alpha=0.8)
        
        # 设置放大图范围
        axins.set_xlim(3.5, 4.5)
        y_zoom_data = np.concatenate([y_exact_all[i, t_zoom_mask], y_pred_all[i, t_zoom_mask]])
        y_margin = (y_zoom_data.max() - y_zoom_data.min()) * 0.1
        axins.set_ylim(y_zoom_data.min() - y_margin, y_zoom_data.max() + y_margin)
        
        # 美化放大图 - 字号扩大2号：9→11
        axins.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        axins.tick_params(labelsize=11, width=1.5, length=4)  # 9→11
        
        # 加粗放大图刻度数字
        for label in axins.get_xticklabels() + axins.get_yticklabels():
            label.set_fontweight('bold')
        
        # 放大图标题 - 字号扩大2号：10→12
        axins.set_title('3.5-4.5s', fontsize=12, pad=3, fontweight='bold')  # 10→12
        
        # 绘制连接框：绿色
        mark_inset(ax[i], axins, loc1=2, loc2=4, fc="none", ec="green", 
                  alpha=0.9, linestyle='--', linewidth=2.0)
    
    plt.tight_layout()
    
    # 保存PNG和PDF两种格式
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    pdf_path = save_path.replace('.png', '.pdf')
    fig.savefig(pdf_path, bbox_inches='tight')
    
    plt.close(fig)
    print(f"✅ Trajectory plot saved: {save_path}")
    print(f"✅ PDF version saved: {pdf_path}")

def main():
    print("=" * 80)
    print("🔬 Hybrid PINN Visualization - Fault at t=4s")
    print("=" * 80)
    print("Normal Model: logs/dae-pinns-best/model.pth")
    print("Fault Model: logs/fault-b5-finetune-step2/model.pth")
    print("Fault Time: t=4.0s")
    print("Configuration: h=0.1s, N=40+120=160 steps (Total 16s)")
    print("=" * 80)
    
    # 参数配置 (与训练时保持一致)
    fault_time = 4.0
    h = 0.1  # 步长 (与训练时一致)
    N_normal = 40   # 0-4s (4/0.1 = 40 steps)
    N_fault = 120   # 4-16s (12/0.1 = 120 steps)
    num_IRK_stages = 100
    
    # 设备配置
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(f"Device: {device}")
    
    # 构建网络架构 (unstacked模式)
    dynamic = dotdict()
    dynamic.num_IRK_stages = num_IRK_stages
    dynamic.state_dim = 4
    dynamic.activation = "sin"
    dynamic.initializer = "Glorot normal"
    dynamic.dropout_rate = 0.0
    dynamic.batch_normalization = None
    dynamic.layer_normalization = None
    dynamic.type = "attention"
    dynamic.layer_size = [4, 100, 100, 100, 100, 4 * (num_IRK_stages + 1)]
    
    algebraic = dotdict()
    algebraic.num_IRK_stages = num_IRK_stages
    algebraic.layer_size = [4, 40, 40, num_IRK_stages + 1]
    algebraic.activation = "sin"
    algebraic.initializer = "Glorot normal"
    algebraic.dropout_rate = 0.0
    algebraic.batch_normalization = None
    algebraic.layer_normalization = None
    algebraic.type = "attention"
    
    # ========== 阶段1: 正常运行 (0-4s, b=10) ==========
    print("\n" + "=" * 80)
    print("Stage 1: Normal Operation (0-4s, b=10)")
    print("=" * 80)
    
    # 创建正常模型
    print("Loading normal model...")
    nn_normal = three_bus_PN(
        dynamic, algebraic,
        dyn_in_transform=lambda x: x,
        alg_out_transform=lambda x: torch.nn.functional.softplus(x),
        stacked=False
    ).to(device)
    
    # 创建supervisor (需要args对象)
    class Args:
        def __init__(self):
            self.num_IRK_stages = num_IRK_stages
            self.h = h
    
    args = Args()
    geom = dde.geometry.Hypercube([-.5, -.5, -.5, -.5], [.5, .5, .5, .5])
    X_dummy = geom.random_points(100)
    
    def dummy_dae(model, y_n, h, IRK_weights):
        return [y_n[:, 0:1]], [y_n[:, 0:1]]
    
    data_normal = dae_data(X_dummy, X_dummy, args, device=str(device), func=dummy_dae)
    super_normal = supervisor(data_normal, nn_normal, device=str(device))
    
    # 初始条件
    X0 = np.array([0., 0., .1, .1])
    
    # 正常阶段PINN预测
    normal_model_path = "./logs/dae-pinns-best/model.pth"
    if not os.path.exists(normal_model_path):
        print(f"❌ Error: Normal model not found {normal_model_path}")
        return
    
    print(f"Predicting normal phase (N={N_normal} steps)...")
    y_pred_normal = super_normal.integrate(X0, N=N_normal, dyn_state_dim=4, 
                                          model_restore_path=normal_model_path)
    
    # 正常阶段精确解
    def power_net_normal(t, x):
        eps = 0.0001
        m_1, m_2, d, d_d, b = .052, .0531, .05, .005, 10.
        v_1, v_2, p_g, p_l, q_l = 1.02, 1.05, -2.0, 3.0, .1
        
        w1, w2, d2, d3, v3 = x
        
        f_1 = b * v_1 * v_2 * np.sin(d2) + b * v_2 * v3 * np.sin(d2 - d3) + p_g
        f_2 = b * v_1 * v3 * np.sin(d3) + b * v_2 * v3 * np.sin(d3 - d2) + p_l
        g = 2 * b * (v3 ** 2) - b * v3 * v_1 * np.cos(d3) - b * v3 * v_2 * np.cos(d3 - d2) + q_l
        
        F0 = (1 / m_1) * (-d * w1 + f_1 + f_2)
        F1 = (1 / m_2) * (-d * w2 - f_1)
        F2 = (w2 - w1)
        F3 = (-w1 - (1 / d_d) * f_2)
        F4 = (-(1 / (eps * v3)) * g)
        
        return F0, F1, F2, F3, F4
    
    t_normal, y_exact_normal = scipy_integrate(power_net_normal, X0, h, 
                                               data_normal.IRK_times, N=N_normal)
    
    print(f"✅ Normal phase completed")
    print(f"   Time points: {t_normal.shape[0]}")
    print(f"   PINN shape: {y_pred_normal.shape}")
    print(f"   Exact shape: {y_exact_normal.shape}")
    
    # ========== 阶段2: 故障运行 (4-15s, b=5) ==========
    print("\n" + "=" * 80)
    print("Stage 2: Fault Operation (4-15s, b=5)")
    print("=" * 80)
    
    # 创建故障模型
    print("Loading fault model...")
    nn_fault = three_bus_PN(
        dynamic, algebraic,
        dyn_in_transform=lambda x: x,
        alg_out_transform=lambda x: torch.nn.functional.softplus(x),
        stacked=False
    ).to(device)
    
    data_fault = dae_data(X_dummy, X_dummy, args, device=str(device), func=dummy_dae)
    super_fault = supervisor(data_fault, nn_fault, device=str(device))
    
    # 从正常阶段的最后状态开始
    X0_fault = y_pred_normal[:4, -1]  # 取最后一列的动态状态
    
    # 故障阶段PINN预测
    fault_model_path = "./logs/fault-b5-finetune-step2/model.pth"
    if not os.path.exists(fault_model_path):
        print(f"❌ Error: Fault model not found {fault_model_path}")
        return
    
    print(f"Predicting fault phase (N={N_fault} steps)...")
    y_pred_fault = super_fault.integrate(X0_fault, N=N_fault, dyn_state_dim=4,
                                        model_restore_path=fault_model_path)
    
    # 故障阶段精确解 (从精确解的最后状态开始)
    X0_fault_exact = y_exact_normal[:, -1]  # scipy的完整状态(包含V3)
    
    def power_net_fault(t, x):
        eps = 0.0001
        m_1, m_2, d, d_d, b = .052, .0531, .05, .005, 5.
        v_1, v_2, p_g, p_l, q_l = 1.02, 1.05, -2.0, 3.0, .1
        
        w1, w2, d2, d3, v3 = x
        
        f_1 = b * v_1 * v_2 * np.sin(d2) + b * v_2 * v3 * np.sin(d2 - d3) + p_g
        f_2 = b * v_1 * v3 * np.sin(d3) + b * v_2 * v3 * np.sin(d3 - d2) + p_l
        g = 2 * b * (v3 ** 2) - b * v3 * v_1 * np.cos(d3) - b * v3 * v_2 * np.cos(d3 - d2) + q_l
        
        F0 = (1 / m_1) * (-d * w1 + f_1 + f_2)
        F1 = (1 / m_2) * (-d * w2 - f_1)
        F2 = (w2 - w1)
        F3 = (-w1 - (1 / d_d) * f_2)
        F4 = (-(1 / (eps * v3)) * g)
        
        return F0, F1, F2, F3, F4
    
    t_fault, y_exact_fault = scipy_integrate(power_net_fault, X0_fault_exact, h,
                                            data_fault.IRK_times, N=N_fault)
    
    # 调整故障阶段的时间偏移
    t_fault = t_fault + fault_time
    
    print(f"✅ Fault phase completed")
    print(f"   Time points: {t_fault.shape[0]}")
    print(f"   PINN shape: {y_pred_fault.shape}")
    print(f"   Exact shape: {y_exact_fault.shape}")
    
    # ========== 阶段3: 绘制合并图 ==========
    print("\n" + "=" * 80)
    print("Stage 3: Generate Combined Visualization")
    print("=" * 80)
    
    save_path = "./logs/visualize_fault/fault_at_4s_comparison.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 新增：出图前清理旧日志与缓存
    clean_previous_logs("./logs")
    
    plot_hybrid_trajectories(t_normal, y_exact_normal, y_pred_normal,
                            t_fault, y_exact_fault, y_pred_fault,
                            fault_time, save_path)
    
    # ========== 阶段4: 误差分析 ==========
    print("\n" + "=" * 80)
    print("Stage 4: Error Analysis")
    print("=" * 80)
    
    # 合并数据
    y_pred_all = np.hstack((y_pred_normal, y_pred_fault))
    y_exact_all = np.hstack((y_exact_normal, y_exact_fault))
    
    errors = []
    var_names = ['ω1', 'ω2', 'δ2', 'δ3', 'V3']
    for i in range(y_exact_all.shape[0]):
        error = l2_relative_error(y_pred_all[i, :], y_exact_all[i, :])
        errors.append(error)
        print(f"  {var_names[i]}: L2 Relative Error = {error*100:.4f}%")
    
    max_error = max(errors)
    print(f"\nMaximum Relative Error: {max_error*100:.4f}%")
    
    # 保存数据
    t_all = np.vstack((t_normal, t_fault))
    print("即将保存npz到：", os.path.abspath("./logs/visualize_fault/fault_at_4s_data.npz"))
    np.savez(
        "./logs/visualize_fault/fault_at_4s_data.npz",
        t=t_all, y_exact=y_exact_all, y_pred=y_pred_all,
        errors=errors, fault_time=fault_time
    )
    
    print("\n" + "=" * 80)
    print("✅ All Completed!")
    print("=" * 80)
    print(f"📊 Trajectory Plot: {save_path}")
    print(f"💾 Data File: ./logs/visualize_fault/fault_at_4s_data.npz")
    print("=" * 80)


if __name__ == "__main__":
    main()
