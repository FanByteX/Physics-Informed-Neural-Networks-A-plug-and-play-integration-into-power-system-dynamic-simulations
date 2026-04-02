#!/usr/bin/env python
"""
DAE-PINNs 推理脚本
使用训练好的模型进行推理和轨迹预测

使用方法:
    python inference.py --model-path ./logs/dae-pinns-best/model.pth --N 80 --h 0.1
"""

import os
import argparse
import torch
import numpy as np
import deepxde as dde
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from utils.utils import dotdict
from models.DAEnn import three_bus_PN
from data.DAE import dae_data
from supervisor import supervisor


def build_network_from_config(config, device):
    """
    从配置字典构建网络
    """
    dynamic = dotdict()
    dynamic.num_IRK_stages = config['num_IRK_stages']
    dynamic.state_dim = config.get('dyn_state_dim', 4)
    
    def dyn_input_feature_layer(x):
        return torch.cat((x, torch.cos(np.pi * x), torch.sin(np.pi * x), 
                          torch.cos(2 * np.pi * x), torch.sin(2 * np.pi * x)), dim=-1)

    def alg_output_feature_layer(x):
        return torch.nn.functional.softplus(x)

    dynamic.activation = config['dyn_activation']
    dynamic.initializer = "Glorot normal"
    dynamic.dropout_rate = config['dropout_rate']
    dynamic.batch_normalization = None if config['dyn_bn'] == "no-bn" else config['dyn_bn']
    dynamic.layer_normalization = None if config['dyn_ln'] == "no-ln" else config['dyn_ln']
    dynamic.type = config['dyn_type']

    if config['unstacked']:
        dim_out = dynamic.state_dim * (dynamic.num_IRK_stages + 1)
    else:
        dim_out = dynamic.num_IRK_stages + 1
    
    if config['use_input_layer']:
        dynamic.layer_size = [dynamic.state_dim * 5] + [config['dyn_width']] * config['dyn_depth'] + [dim_out]
    else:
        dynamic.layer_size = [dynamic.state_dim] + [config['dyn_width']] * config['dyn_depth'] + [dim_out]
        dyn_input_feature_layer = None

    algebraic = dotdict()
    algebraic.num_IRK_stages = config['num_IRK_stages']
    dim_out_alg = algebraic.num_IRK_stages + 1
    algebraic.layer_size = [dynamic.state_dim] + [config['alg_width']] * config['alg_depth'] + [dim_out_alg]
    algebraic.activation = config['alg_activation']
    algebraic.initializer = "Glorot normal"
    algebraic.dropout_rate = config['dropout_rate']
    algebraic.batch_normalization = None if config['alg_bn'] == "no-bn" else config['alg_bn']
    algebraic.layer_normalization = None if config['alg_ln'] == "no-ln" else config['alg_ln']
    algebraic.type = config['alg_type']

    nn = three_bus_PN(
        dynamic, 
        algebraic, 
        dyn_in_transform=dyn_input_feature_layer, 
        alg_out_transform=alg_output_feature_layer,
        stacked=not config['unstacked'],
    ).to(device)
    
    return nn


def build_network_from_args(args, device):
    """
    从命令行参数构建网络
    """
    dynamic = dotdict()
    dynamic.num_IRK_stages = args.num_IRK_stages
    dynamic.state_dim = 4
    
    def dyn_input_feature_layer(x):
        return torch.cat((x, torch.cos(np.pi * x), torch.sin(np.pi * x), 
                          torch.cos(2 * np.pi * x), torch.sin(2 * np.pi * x)), dim=-1)

    def alg_output_feature_layer(x):
        return torch.nn.functional.softplus(x)

    # 默认值与训练参数一致
    dynamic.activation = args.dyn_activation
    dynamic.initializer = "Glorot normal"
    dynamic.dropout_rate = args.dropout_rate
    dynamic.batch_normalization = None if args.dyn_bn == "no-bn" else args.dyn_bn
    dynamic.layer_normalization = None if args.dyn_ln == "no-ln" else args.dyn_ln
    dynamic.type = args.dyn_type

    unstacked = args.unstacked
    if unstacked:
        dim_out = dynamic.state_dim * (dynamic.num_IRK_stages + 1)
    else:
        dim_out = dynamic.num_IRK_stages + 1
    
    use_input_layer = args.use_input_layer
    if use_input_layer:
        dynamic.layer_size = [dynamic.state_dim * 5] + [args.dyn_width] * args.dyn_depth + [dim_out]
    else:
        dynamic.layer_size = [dynamic.state_dim] + [args.dyn_width] * args.dyn_depth + [dim_out]
        dyn_input_feature_layer = None

    algebraic = dotdict()
    algebraic.num_IRK_stages = dynamic.num_IRK_stages
    dim_out_alg = algebraic.num_IRK_stages + 1
    algebraic.layer_size = [dynamic.state_dim] + [args.alg_width] * args.alg_depth + [dim_out_alg]
    algebraic.activation = args.alg_activation
    algebraic.initializer = "Glorot normal"
    algebraic.dropout_rate = args.dropout_rate
    algebraic.batch_normalization = None if args.alg_bn == "no-bn" else args.alg_bn
    algebraic.layer_normalization = None if args.alg_ln == "no-ln" else args.alg_ln
    algebraic.type = args.alg_type

    nn = three_bus_PN(
        dynamic, 
        algebraic, 
        dyn_in_transform=dyn_input_feature_layer, 
        alg_out_transform=alg_output_feature_layer,
        stacked=not unstacked,
    ).to(device)
    
    return nn


def load_model_auto(model_path, device, default_args=None):
    """
    自动从 checkpoint 加载模型
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
        print(f"从 checkpoint 读取模型配置:")
        print(f"  - num_IRK_stages: {config['num_IRK_stages']}")
        print(f"  - unstacked: {config['unstacked']}")
        print(f"  - dyn_width: {config['dyn_width']}, dyn_depth: {config['dyn_depth']}")
        print(f"  - alg_width: {config['alg_width']}, alg_depth: {config['alg_depth']}")
        nn = build_network_from_config(config, device)
    elif default_args is not None:
        print("checkpoint 中没有 model_config，使用提供的默认参数")
        nn = build_network_from_args(default_args, device)
        config = None
    else:
        raise ValueError(
            "checkpoint 中没有 model_config，且没有提供默认参数。\n"
            "请使用 --num-irk-stages, --dyn-width, --dyn-depth 等参数手动指定模型结构。"
        )
    
    # 加载权重
    nn.load_state_dict(checkpoint['state_dict'])
    nn.to(device)
    nn.eval()
    
    print(f"模型已加载: {model_path}")
    return nn, config


def scipy_ground_truth(X0, args, N=20):
    """
    使用 scipy 计算真实轨迹
    """
    def power_net_dae(t, y):
        """电力系统 DAE 方程"""
        M_1, M_2, D, D_d, b = .052, .0531, .05, .005, 10.
        V_1, V_2, P_g, P_l, Q_l = 1.02, 1.05, -2.0, 3.0, .1
        eps = 0.0001
        
        w1, w2, d2, d3, v3 = y
        
        f_1 = b * V_1 * V_2 * np.sin(d2) + b * V_2 * v3 * np.sin(d2 - d3) + P_g
        f_2 = b * V_1 * v3 * np.sin(d3) + b * V_2 * v3 * np.sin(d3 - d2) + P_l
        g = 2 * b * (v3 ** 2) - b * v3 * V_1 * np.cos(d3) - b * v3 * V_2 * np.cos(d3 - d2) + Q_l
        
        F0 = (1 / M_1) * (- D * w1 + f_1 + f_2)
        F1 = (1 / M_2) * (- D * w2 - f_1)
        F2 = (w2 - w1)
        F3 = (- w1 - (1 / D_d) * f_2)
        F4 = (- (1 / (eps * v3)) * g)
        
        return [F0, F1, F2, F3, F4]
    
    V0 = 0.7  # 固定电压初始条件
    y0_full = [X0[0], X0[1], X0[2], X0[3], V0]
    
    # 计算时间点（与训练一致）
    nu = args.num_IRK_stages
    tmp = np.float32(np.loadtxt('./data/IRK_weights/Butcher_IRK%d.txt' % (nu), ndmin=2))
    IRK_times = tmp[nu**2 + nu:]
    
    t_span = [0.0, args.h * N]
    t_sim = np.array([t_span[0]])
    for k in range(1, N + 1):
        temp = (k - 1) * args.h + IRK_times * args.h
        t_sim = np.vstack((t_sim, temp))
        t_next = np.array([k * args.h])
        t_sim = np.vstack((t_sim, t_next))
    
    sol = solve_ivp(power_net_dae, t_span, y0_full, method='RK45', t_eval=t_sim.reshape(-1,))
    
    return t_sim[1:], sol.y[:, 1:]


def plot_comparison(t, y_pred, y_true, save_path=None):
    """
    绘制预测与真实轨迹对比图
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    labels = ['w1 (ω1)', 'w2 (ω2)', 'd2 (δ2)', 'd3 (δ3)', 'v3 (V3)']
    
    for i, (ax, label) in enumerate(zip(axes.flat[:5], labels)):
        ax.plot(t.flatten(), y_true[i, :].T, 'r-', label='Ground Truth', linewidth=2)
        ax.plot(t.flatten(), y_pred[i, :].T, 'b--', label='PINN Prediction', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(label)
        ax.legend()
        ax.set_title(f'{label} comparison')
        ax.grid(True, alpha=0.3)
    
    # 隐藏最后一个子图
    axes[2, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='DAE-PINNs 推理')
    parser.add_argument('--model-path', type=str, required=True, help='模型权重路径 (.pth)')
    parser.add_argument('--output-dir', type=str, default='./inference_results', help='输出目录')
    
    # 推理参数
    parser.add_argument('--N', type=int, default=80, help='预测步数')
    parser.add_argument('--h', type=float, default=0.1, help='时间步长')
    
    # 模型结构参数 (如果 checkpoint 中包含 model_config 则自动读取)
    # 注意：参数名使用下划线，与 dae_data 和训练脚本一致
    parser.add_argument('--num_IRK_stages', type=int, default=100, help='IRK stages')
    parser.add_argument('--unstacked', action='store_true', default=True, help='Unstacked output')
    parser.add_argument('--use_input_layer', action='store_true', default=False, help='Use input layer')
    parser.add_argument('--dyn_width', type=int, default=100, help='Dynamic network width')
    parser.add_argument('--dyn_depth', type=int, default=4, help='Dynamic network depth')
    parser.add_argument('--dyn_activation', type=str, default='sin', help='Dynamic network activation')
    parser.add_argument('--dyn_type', type=str, default='attention', help='Dynamic network type')
    parser.add_argument('--dyn_bn', type=str, default='no-bn', help='Dynamic batch normalization')
    parser.add_argument('--dyn_ln', type=str, default='no-ln', help='Dynamic layer normalization')
    parser.add_argument('--alg_width', type=int, default=40, help='Algebraic network width')
    parser.add_argument('--alg_depth', type=int, default=2, help='Algebraic network depth')
    parser.add_argument('--alg_activation', type=str, default='sin', help='Algebraic network activation')
    parser.add_argument('--alg_type', type=str, default='attention', help='Algebraic network type')
    parser.add_argument('--alg_bn', type=str, default='no-bn', help='Algebraic batch normalization')
    parser.add_argument('--alg_ln', type=str, default='no-ln', help='Algebraic layer normalization')
    parser.add_argument('--dropout_rate', type=float, default=0.0, help='Dropout rate')
    
    # 初始状态
    parser.add_argument('--X0', type=float, nargs=4, default=[0., 0., 0.1, 0.1], 
                        help='初始状态 [w1, w2, d2, d3]')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    nn, config = load_model_auto(args.model_path, device, default_args=args)
    
    # 更新 args 中的 num_irk_stages（如果从 config 读取）
    if config is not None:
        args.num_irk_stages = config['num_IRK_stages']
    
    # 创建 data 对象（用于 IRK 权重）
    geom = dde.geometry.Hypercube([-.5, -.5, -.5, -.5], [.5, .5, .5, .5])
    X_dummy = geom.random_points(10)
    data = dae_data(X_dummy, X_dummy, args, device=device, func=lambda *args: [0])
    
    # 创建 supervisor
    super = supervisor(data, nn, device=device)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始状态
    X0 = np.array(args.X0)
    print(f"\n开始预测")
    print(f"初始状态: {X0}")
    print(f"预测步数: {args.N}, 时间步长: {args.h}")
    print(f"预测时长: {args.N * args.h} 秒")
    
    # 使用 supervisor.integrate 进行预测
    print("\n正在用模型预测...")
    y_pred = super.integrate(X0, N=args.N, dyn_state_dim=4, model_restore_path=None)
    
    # scipy 真实值
    print("正在计算真实轨迹...")
    t, y_true = scipy_ground_truth(X0, args, N=args.N)
    
    # 计算误差
    print("\n计算 L2 相对误差...")
    l2_errors = []
    for i in range(5):
        l2_err = np.linalg.norm(y_pred[i, :] - y_true[i, :]) / np.linalg.norm(y_true[i, :])
        l2_errors.append(l2_err)
        print(f"  变量 {i+1}: L2 相对误差 = {l2_err:.4e}")
    
    avg_l2_error = np.mean(l2_errors)
    print(f"\n平均 L2 相对误差: {avg_l2_error:.4e}")
    
    # 绘图
    plot_path = os.path.join(args.output_dir, 'trajectory_comparison.png')
    plot_comparison(t, y_pred, y_true, save_path=plot_path)
    
    # 保存结果
    result_path = os.path.join(args.output_dir, 'prediction_results.npz')
    np.savez(result_path, 
             prediction=y_pred, 
             ground_truth=y_true, 
             time=t,
             X0=X0, 
             l2_errors=l2_errors,
             avg_l2_error=avg_l2_error,
             N=args.N,
             h=args.h)
    print(f"\n结果已保存: {result_path}")


if __name__ == '__main__':
    main()
