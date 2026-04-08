"""
电力网络 DAE 物理模型模块
power_net_dae: 电力网络 DAE 方程，定义物理约束
scipy_integrate: 使用 SciPy 求解器计算真实解，用于对比
"""
import numpy as np
import mindspore.ops as ops
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


def power_net_dae(model, y_n, h, IRK_weights):
    """
    电力网络 DAE 方程
    定义物理约束，用于 PINN 损失计算
    
    Args:
        model: 神经网络模型
        y_n: 输入状态
        h: 时间步长
        IRK_weights: IRK 权重矩阵
        
    Returns:
        f: 动力学方程残差列表
        g: 代数方程残差列表
    """
    T = 1.0
    M_1, M_2, D, D_d, b = .052, .0531, .05, .005, 5.
    V_1, V_2, P_g, P_l, Q_l = 1.02, 1.05, -2.0, 3.0, .1

    yn = y_n.copy()
    w1, w2, d2, d3, v3 = model(yn)
    
    xi_w1 = w1[..., :-1]
    xi_w2 = w2[..., :-1]
    xi_d2 = d2[..., :-1]
    xi_d3 = d3[..., :-1]
    zeta_v3 = v3[..., :-1]

    f_1 = b * V_1 * V_2 * ops.sin(xi_d2) + b * V_2 * zeta_v3 * ops.sin(xi_d2 - xi_d3) + P_g
    f_2 = b * V_1 * zeta_v3 * ops.sin(xi_d3) + b * V_2 * zeta_v3 * ops.sin(xi_d3 - xi_d2) + P_l

    F0 = T * (1 / M_1) * (- D * xi_w1 + f_1 + f_2)
    F1 = T * (1 / M_2) * (- D * xi_w2 - f_1)
    F2 = T * (xi_w2 - xi_w1)
    F3 = T * (- xi_w1 - (1 / D_d) * f_2)

    f0 = yn[..., 0:1] - (w1 - h * ops.matmul(F0, IRK_weights.T))
    f1 = yn[..., 1:2] - (w2 - h * ops.matmul(F1, IRK_weights.T))
    f2 = yn[..., 2:3] - (d2 - h * ops.matmul(F2, IRK_weights.T))
    f3 = yn[..., 3:4] - (d3 - h * ops.matmul(F3, IRK_weights.T))

    G = 2 * b * (v3 ** 2) - b * v3 * V_1 * ops.cos(d3) - b * v3 * V_2 * ops.cos(d3 - d2) + Q_l
    g = - (T / v3) * G

    return [f0, f1, f2, f3], [g]


def scipy_integrate(func, X0, args, IRK_times, N=0):
    """
    使用 SciPy 求解器计算真实解
    
    Args:
        func: DAE 方程函数
        X0: 初始条件
        args: 参数配置
        IRK_times: IRK 时间节点
        N: 时间步数
        
    Returns:
        t_sim: 时间向量
        y_test: 解向量
    """
    # 计算故障瞬间(t=0+)的V3真实初始值
    T = 1.0
    M_1, M_2, D, D_d, b = .052, .0531, .05, .005, 5.
    V_1, V_2, P_g, P_l, Q_l = 1.02, 1.05, -2.0, 3.0, .1
    
    # 在t=0时刻，动态变量的初值已知：X0 = [w1_0, w2_0, d2_0, d3_0]
    # 代数方程: 2*b*V3^2 - b*V3*V1*cos(d3) - b*V3*V2*cos(d3-d2) + Q_l = 0
    def algebraic_eq(v3):
        return 2 * b * (v3 ** 2) - b * v3 * V_1 * np.cos(X0[3]) - b * v3 * V_2 * np.cos(X0[3] - X0[2]) + Q_l
    
    V0 = fsolve(algebraic_eq, 0.98)[0]
    
    t_span = [0.0, args.h * N]
    t_sim = np.array([])
    
    # 只包含故障后的时间点，跳过t=0初始状态
    for k in range(1, N + 1):
        temp = (k - 1) * args.h + IRK_times * args.h
        if len(t_sim) == 0:
            t_sim = temp
        else:
            t_sim = np.vstack((t_sim, temp))
        t_next = np.array([k * args.h])
        t_sim = np.vstack((t_sim, t_next))
    
    # 求解时仍然从t=0开始，但输出时只返回t>0的点
    sol = solve_ivp(func, t_span, [X0[0], X0[1], X0[2], X0[3], V0], 
                   method=args.method, t_eval=np.concatenate([[0.0], t_sim.reshape(-1,)]))
    y_test = sol.y
    
    # 返回时跳过t=0的解，只返回故障后的状态
    return t_sim, y_test[:, 1:]


def power_net_dae_scipy(t, x):
    """
    用于 SciPy 积分的 DAE 方程
    
    Args:
        t: 时间
        x: 状态向量 [w1, w2, d2, d3, v3]
        
    Returns:
        导数向量
    """
    eps = 0.0001
    m_1, m_2, d, d_d, b = .052, .0531, .05, .005, 5.
    v_1, v_2, p_g, p_l, q_l = 1.02, 1.05, -2.0, 3.0, .1
    
    w1, w2, d2, d3, v3 = x
    f_1 = b * v_1 * v_2 * np.sin(d2) + b * v_2 * v3 * np.sin(d2 - d3) + p_g
    f_2 = b * v_1 * v3 * np.sin(d3) + b * v_2 * v3 * np.sin(d3 - d2) + p_l
    g = 2 * b * (v3 ** 2) - b * v3 * v_1 * np.cos(d3) - b * v3 * v_2 * np.cos(d3 - d2) + q_l
    
    F0 = (1 / m_1) * (- d * w1 + f_1 + f_2)
    F1 = (1 / m_2) * (- d * w2 - f_1)
    F2 = (w2 - w1)
    F3 = (- w1 - (1 / d_d) * f_2)
    F4 = (- (1 / (eps * v3)) * g)
    
    return F0, F1, F2, F3, F4
