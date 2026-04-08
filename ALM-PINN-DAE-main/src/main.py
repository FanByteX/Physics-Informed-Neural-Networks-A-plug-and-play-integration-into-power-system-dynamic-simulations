"""
MindSpore 版本的 RES-PINN (残差物理信息神经网络)
主入口文件

主要模块说明：
1. config: 命令行参数解析
2. utils: 工具函数、损失函数、评估指标、可视化
3. data: 数据加载和 IRK 权重
4. models: RES-PINN 模型架构、DAE 模型、物理方程
5. training: 训练管理器、检查点
"""
import os
import sys
import numpy as np
import mindspore
import mindspore.ops as ops

# 禁用 DeepXDE 的 Horovod 检测（MindSpore 使用自己的分布式训练）
if 'OMPI_COMM_WORLD_SIZE' in os.environ:
    _ompi_vars = {}
    for key in list(os.environ.keys()):
        if key.startswith('OMPI_') or key.startswith('PMI_'):
            _ompi_vars[key] = os.environ.pop(key)

import deepxde as dde

# 恢复 MPI 环境变量供 MindSpore 使用
if '_ompi_vars' in locals():
    os.environ.update(_ompi_vars)

# 导入项目模块
from config import parse_args
from utils import dotdict, l2_relative_error, plot_loss_history, plot_three_bus_all, plot_L2relative_error, plot_regression
from data import dae_data
from training import ModelCheckPoint, supervisor
from models import three_bus_PN, power_net_dae, scipy_integrate, power_net_dae_scipy, fnn, attention, Conv1D


def run_fault_b5(args):
    """
    完整的训练流程，包括：
    1. 设备配置 (NPU/CPU)
    2. 模型创建 (动态变量 + 代数变量)
    3. 数据准备 (DeepXDE 几何采样)
    4. 训练执行 (优化器 + 检查点)
    5. 结果可视化 (损失、轨迹、误差图)
    6. 误差评估 (L2 相对误差)
    """
    print("starting...\n")
    print("=" * 80)
    print("Lines fault training - b=5")
    print("=" * 80)
    print()

    if os.path.isdir(args.log_dir) == False:
        os.makedirs(args.log_dir, exist_ok=True)

    # MindSpore NPU设备配置（支持单卡/多卡）
    from mindspore import context
    from mindspore.communication import init, get_rank, get_group_size
    
    distributed = args.distributed
    rank_id = 0
    device_num = 1
    
    if distributed:
        # 分布式训练初始化
        print("=" * 60)
        print("Initializing Distributed Training on Multi-NPU")
        print("=" * 60)
        init()
        rank_id = get_rank()
        device_num = get_group_size()
        print(f"Rank {rank_id}/{device_num} initialized")
        
        context.set_context(
            mode=context.PYNATIVE_MODE,
            device_target="Ascend",
            device_id=rank_id,
            save_graphs=False
        )
        device = f"Ascend:{rank_id}"
        print(f"Rank {rank_id}: Using device NPU-{rank_id}")
        
        original_log_dir = args.log_dir
        if rank_id != 0:
            args.log_dir = f"/tmp/mindspore_rank_{rank_id}"
            os.makedirs(args.log_dir, exist_ok=True)
        print(f"Rank {rank_id}: Logs directory: {args.log_dir}")
            
    else:
        # 单卡训练
        try:
            device_id = args.device_id if hasattr(args, 'device_id') else 0
            context.set_context(
                mode=context.PYNATIVE_MODE,
                device_target="Ascend",
                device_id=device_id
            )
            device = f"Ascend:{device_id}"
            print(f"Using single NPU device: NPU-{device_id}")
        except Exception as e:
            print(f"Warning: Failed to set Ascend device: {e}")
            print("Falling back to CPU")
            context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
            device = "CPU"
            print(f"Using device: {device}")
    
    print("=" * 60)

    # 配置动态变量网络
    dynamic = dotdict()
    dynamic.num_IRK_stages = args.num_IRK_stages
    dynamic.state_dim = 4

    def dyn_input_feature_layer(x):
        return ops.cat((x, ops.cos(np.pi * x), ops.sin(np.pi * x), 
                       ops.cos(2 * np.pi * x), ops.sin(2 * np.pi * x)), axis=-1)
    
    def alg_output_feature_layer(x):
        return ops.softplus(x)

    dynamic.activation = args.dyn_activation
    dynamic.initializer = "Glorot normal"
    dynamic.dropout_rate = args.dropout_rate
    dynamic.batch_normalization = None if args.dyn_bn == "no-bn" else args.dyn_bn
    dynamic.layer_normalization = None if args.dyn_ln == "no-ln" else args.dyn_ln
    dynamic.type = args.dyn_type

    if args.unstacked:
        dim_out = dynamic.state_dim * (dynamic.num_IRK_stages + 1)
    else:
        dim_out = dynamic.num_IRK_stages + 1

    if args.use_input_layer:
        dynamic.layer_size = [dynamic.state_dim * 5] + [args.dyn_width] * args.dyn_depth + [dim_out]
    else:
        dynamic.layer_size = [dynamic.state_dim] + [args.dyn_width] * args.dyn_depth + [dim_out]
        def identity_transform(x):
            return x
        dyn_input_feature_layer = identity_transform

    # 配置代数变量网络
    algebraic = dotdict()
    algebraic.num_IRK_stages = args.num_IRK_stages
    dim_out_alg = algebraic.num_IRK_stages + 1
    algebraic.layer_size = [dynamic.state_dim] + [args.alg_width] * args.alg_depth + [dim_out_alg]
    algebraic.activation = args.alg_activation
    algebraic.initializer = "Glorot normal"
    algebraic.dropout_rate = args.dropout_rate
    algebraic.batch_normalization = None if args.alg_bn == "no-bn" else args.alg_bn
    algebraic.layer_normalization = None if args.alg_ln == "no-ln" else args.alg_ln
    algebraic.type = args.alg_type

    # 创建模型
    nn_model = three_bus_PN(
        dynamic,
        algebraic,
        dyn_in_transform=dyn_input_feature_layer,
        alg_out_transform=alg_output_feature_layer,
        stacked=not args.unstacked,
    )

    # 数据生成
    geom = dde.geometry.Hypercube([-.5, -.5, -.5, -.5], [.5, .5, .5, .5])
    
    if distributed:
        np.random.seed(1234 + rank_id * 1000)
        X_train = geom.random_points(args.num_train)
        np.random.seed(3456 + rank_id * 1000)
        X_test = geom.random_points(args.num_test)
        print(f"Rank {rank_id}: Generated {len(X_train)} training samples")
    else:
        np.random.seed(1234)
        X_train = geom.random_points(args.num_train)
        np.random.seed(3456)
        X_test = geom.random_points(args.num_test)
    
    def pinn_func(model, y_n, h, IRK_weights):
        return power_net_dae(model, y_n, h, IRK_weights)
    
    data = dae_data(X_train, X_test, args, device=str(device), func=pinn_func)

    # 创建训练管理器
    superv = supervisor(data, nn_model, device=device)
    optimizer = mindspore.nn.Adam(nn_model.trainable_params(), learning_rate=args.lr)
    
    if args.use_scheduler:
        if args.scheduler_type == "plateau":
            print("Warning: LR scheduler not fully supported in MindSpore, using fixed LR")
            scheduler = None
        else:
            scheduler = None
    else:
        scheduler = None

    superv.compile(optimizer, loss_weights=[args.dyn_weight, args.alg_weight], 
                  scheduler=scheduler, scheduler_type=args.scheduler_type)

    model_name = 'model.ckpt' if args.model_name == 'no-name' else ('model_' + args.model_name + '.ckpt')
    save_path = os.path.join(args.log_dir, model_name)
    chcker = ModelCheckPoint(save_path, save_better_only=True, every=1000)
    
    if args.start_from_best and os.path.exists(save_path):
        try:
            loss_hist_path = os.path.join(args.log_dir, 'loss-history.npz')
            if os.path.exists(loss_hist_path):
                prev_history = np.load(loss_hist_path)
                prev_loss_test = prev_history['loss_test']
                best_prev_loss = min([sum(l) if isinstance(l, (list, np.ndarray)) else l for l in prev_loss_test])
                chcker.best = best_prev_loss
                print(f"Checkpoint best loss initialized to: {best_prev_loss:.6e}")
        except Exception as e:
            print(f"Warning: Could not restore checkpoint best value: {e}")
    
    restore_path = save_path if args.start_from_best else None
    if distributed and args.start_from_best and 'original_log_dir' in locals():
        restore_path = os.path.join(original_log_dir, model_name)
        if not os.path.exists(restore_path):
            print(f"Rank {rank_id}: Model not found at {restore_path}, starting from scratch")
            restore_path = None
    if args.start_from_best:
        print("starting from best model so far...")

    # 执行训练
    loss_history, state = superv.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_every=args.test_every,
        num_val=args.num_val,
        events=[chcker],
        model_restore_path=restore_path,
        use_tqdm=args.use_tqdm,
    )

    # 保存结果
    print("plotting train and test loss...\n")
    plot_loss_history(loss_history, fname=os.path.join(args.log_dir, 'loss.png'))
    np.savez(os.path.join(args.log_dir, 'loss-history'),
             steps=np.array(loss_history.steps),
             loss_train=np.array(loss_history.loss_train),
             loss_test=np.array(loss_history.loss_test))

    # 轨迹预测和评估
    X0 = [0., 0., .1, .1]
    X0_npy = np.array(X0)
    y_pred = superv.integrate(X0_npy, N=args.N, dyn_state_dim=4, model_restore_path=save_path)

    t, y_eval = scipy_integrate(power_net_dae_scipy, X0, args, superv.data.IRK_times, N=args.N)
    print("plotting trajectory...\n")
    plot_three_bus_all(t, y_eval, y_pred, fname=os.path.join(args.log_dir, 'trajectories.png'), 
                      size=20, figsize=(12, 16))

    # L2 相对误差
    error_data = np.empty((args.N, 5))
    for k in range(1, args.N + 1):
        y_pred_k = superv.integrate(X0_npy, N=k, dyn_state_dim=4, model_restore_path=save_path)
        _, y_eval_k = scipy_integrate(power_net_dae_scipy, X0, args, superv.data.IRK_times, N=k)
        for i in range(5):
            error_data[k - 1, i] = l2_relative_error(y_pred_k[i, ...], y_eval_k[i, ...])

    N_vec = np.arange(1, args.N + 0.1)
    var_names = [r'\omega_1', r'\omega_2', r'\delta_2', r'\delta_3', 'V_3']
    for k in range(5):
        fname_k = 'L2relative_error_' + str(k) + '.png'
        fname = os.path.join(args.log_dir, fname_k)
        plot_L2relative_error(N_vec, error_data[:, k], fname=fname, size=20, figsize=(8, 6), 
                             var_name=var_names[k])
    np.savez(os.path.join(args.log_dir, "L2Relative_error"), N=N_vec, error=error_data)

    # 回归图
    x_line = [-.5, .5]
    y_line = [-.5, .5]
    plot_regression(y_pred[-2, ...], y_eval[-2, ...], 
                   fname=os.path.join(args.log_dir, 'regression-voltage.png'), 
                   size=20, figsize=(8, 6), x_line=x_line, y_line=y_line)

    # 保存预测数据
    np.savez(os.path.join(args.log_dir, "prediction-data"), y_pred=y_pred, y_eval=y_eval, time=t)


def main():
    """主函数入口"""
    args = parse_args()
    run_fault_b5(args)
if __name__ == "__main__":
    main()
