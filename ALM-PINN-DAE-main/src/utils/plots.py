"""
可视化模块
用于生成各种训练结果图表：
1. plot_loss_history: 损失曲线图 (对数坐标)
2. plot_three_bus_all: 电力系统轨迹图 (所有5个变量)
3. plot_L2relative_error: L2相对误差图
4. plot_regression: 回归分析图 (预测 vs 真实值)
所有图表使用英文标签，以避免中文编码问题
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def stylize_axes(ax, size=25, legend=True, xlabel=None, ylabel=None, title=None, 
                 xticks=None, yticks=None, xticklabels=None, yticklabels=None, 
                 top_spine=True, right_spine=True):
    """美化坐标轴"""
    ax.spines['top'].set_visible(top_spine)
    ax.spines['right'].set_visible(right_spine)
    ax.xaxis.set_tick_params(top=False, direction='out', width=1)
    ax.yaxis.set_tick_params(right=False, direction='out', width=1)
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)
    if legend:
        leg = ax.legend(fontsize=14, framealpha=0.9)
    return ax


def custom_logplot(ax, x, y, label="Loss", xlims=None, ylims=None, 
                   color='blue', linestyle='solid', marker=None):
    """对数坐标绘图"""
    if marker is None:
        ax.semilogy(x, y, color=color, label=label, linestyle=linestyle)
    else:
        ax.semilogy(x, y, color=color, label=label, linestyle=linestyle, marker=marker)
    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)
    return ax


def custom_lineplot(ax, x, y, label=None, xlims=None, ylims=None, 
                    color="red", linestyle="solid", linewidth=2.0, marker=None):
    """普通线性绘图"""
    if label is not None:
        if marker is None:
            ax.plot(x, y, color=color, label=label, linestyle=linestyle, linewidth=linewidth)
        else:
            ax.plot(x, y, color=color, label=label, linestyle=linestyle, linewidth=linewidth, marker=marker)
    else:
        if marker is None:
            ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth)
        else:
            ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, marker=marker)
    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)
    return ax


def plot_loss_history(loss_history, fname="./logs/loss.png", size=25, figsize=(8, 6)):
    """绘制损失历史曲线"""
    loss_train = np.array(loss_history.loss_train)
    loss_test = np.array(loss_history.loss_test)
    all_losses = np.concatenate([loss_train, loss_test])
    finite_losses = all_losses[np.isfinite(all_losses)]
    
    if len(finite_losses) > 0:
        min_loss = max(np.min(finite_losses), 1e-8)
        max_loss = max(np.percentile(finite_losses, 99.9), 1e-5)
    else:
        min_loss = 1e-8
        max_loss = 1e5
    
    fig, ax = plt.subplots(figsize=figsize)
    custom_logplot(ax, loss_history.steps, loss_train, label="Train", color='blue', linestyle='solid')
    custom_logplot(ax, loss_history.steps, loss_test, label="Test", color='red', linestyle='dashed')
    ax.set_ylim(min_loss, max_loss)
    stylize_axes(ax, size=size, xlabel="Iterations", ylabel="Mean Squared Error")
    fig.tight_layout()
    fig.savefig(fname, dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)


def plot_three_bus_all(t, y_eval, y_pred, fname="logs/trajectories.png", size=25, figsize=(12, 16)):
    """
    绘制所有五个变量的轨迹图
    Exact: 深蓝色实线; Predicted: 橙红色短虚线 (linewidth=1.5)
    """
    t = t.reshape(-1,)
    labels = [r"$\omega_1$", r"$\omega_2$", r"$\delta_2$", r"$\delta_3$", r"$V_3$"]
    fig, ax = plt.subplots(nrows=5, ncols=1, figsize=figsize)
    
    for i in range(5):
        y_pred_i = y_pred[i, ...].reshape(-1,)
        # 对V3（代数变量）进行轻度平滑处理以消除数值毛刺
        if i == 4:  # V3 is the 5th variable (index 4)
            from scipy.ndimage import uniform_filter1d
            y_pred_i = uniform_filter1d(y_pred_i, size=3, mode='nearest')
        # Exact: 深蓝色实线
        custom_lineplot(ax[i], t, y_eval[i, ...].reshape(-1,), 
                       label="Exact", color="#00008B", linestyle="solid", linewidth=2.5)
        # Predicted: 橙红色短虚线，线宽1.5
        custom_lineplot(ax[i], t, y_pred_i, 
                       label="Predicted", color="#FF4500", linestyle=(0, (3, 3)), linewidth=1.5)
        stylize_axes(ax[i], size=size, xlabel='Time (s)' if i == 4 else None, ylabel=labels[i])
    
    fig.savefig(fname, dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)


def plot_L2relative_error(N, error, fname="./logs/L2relative_error.png", 
                          size=20, figsize=(8, 6), var_name=None):
    """绘制 L2 相对误差图"""
    error = error.reshape(-1,)
    fig, ax = plt.subplots(figsize=figsize)
    custom_lineplot(ax, N, error, color="green", linestyle="dashed", 
                   linewidth=3.0, marker='s', label=None)
    
    if var_name is not None:
        ylabel_text = rf"$L_2$ Relative Error (${var_name}$)"
    else:
        ylabel_text = r"$L_2$ Relative Error"
    
    stylize_axes(ax, size=size, xlabel="Time Steps N", ylabel=ylabel_text, legend=False)
    fig.tight_layout()
    fig.savefig(fname, dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)


def plot_regression(predicted, y, fname="./logs/regression-voltage.png", 
                    size=20, figsize=(8, 6), x_line=None, y_line=None):
    """绘制回归分析图"""
    predicted = predicted.reshape(-1,)
    y = y.reshape(-1,)
    
    if x_line is None:
        x_line = [y.min(), y.max()]
        y_line = [y.min(), y.max()]
    
    fig, ax = plt.subplots(figsize=figsize)
    custom_lineplot(ax, x_line, y_line, color="#FF4500", linestyle="dashed", linewidth=3.0)
    ax.scatter(predicted, y, color='blue', marker='o', s=10, alpha=0.5)
    stylize_axes(ax, size=size, xlabel="Predicted", ylabel="Exact", legend=False)
    fig.tight_layout()
    fig.savefig(fname, dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)
