# Plug 项目输出图片说明

本文档描述 `outputs/` 目录下各图片的含义及生成命令。

---

## Figure_1.png

**运行命令：**
```bash
cd plug
python main.py --study_selection 1
```

**图片含义：**

轨迹概览图，包含 6 个子图，对比三种求解器（PINN混合求解器、纯RK求解器、真值）：

| 子图 | 标题 | 说明 |
|------|------|------|
| 左上 | $\delta - \theta$ Evolution | 三台发电机的 δ-θ 角度差演化 |
| 右上 | $\omega$ Evolution | 三台发电机的角速度 ω 演化 |
| 左中 | $I_d$ Evolution | 三台发电机的 d 轴电流演化 |
| 右中 | $I_q$ Evolution | 三台发电机的 q 轴电流演化 |
| 左下 | $V_m$ Evolution | 三台发电机的端电压幅值演化 |
| 右下 | $\theta$ Evolution | 发电机间的相角差演化 |

**图例：**
- 实线（彩色）：PINN 混合求解器
- 虚线（彩色）：纯 RK 求解器
- 点划线（黑色）：真值（Assimulo 仿真）

---

## Figure_4.png

**运行命令：**
```bash
cd plug
python main.py --study_selection 2 --sim_time 10
```

**图片含义：**

10 秒仿真的轨迹与误差对比图，包含 2 个子图：

| 子图 | 标题 | 说明 |
|------|------|------|
| 左 | $\delta - \theta$ Gen. 3 | 第 3 台发电机的 δ-θ 演化及误差 |
| 右 | $V_m$ Gen. 3 | 第 3 台发电机的端电压幅值演化及误差 |

**图表元素：**
- 主 Y 轴：轨迹曲线
- 次 Y 轴（填充区域）：$\ell_1$ 误差
- 橙色：纯 RK 求解器
- 蓝色：PINN 混合求解器
- 黑色虚线：真值

---

## Figure_5.png

**运行命令：**
```bash
cd plug
python main.py --study_selection 2 --sim_time 2
```

**图片含义：**

2 秒仿真的轨迹与误差对比图，结构与 Figure_4 相同：

| 子图 | 标题 | 说明 |
|------|------|------|
| 左 | $\delta - \theta$ Gen. 3 | 第 3 台发电机的 δ-θ 演化及误差 |
| 右 | $V_m$ Gen. 3 | 第 3 台发电机的端电压幅值演化及误差 |

---

## Figure_6.png

**运行命令：**
```bash
cd plug
python main.py --study_selection 3
```

**图片含义：**

不同时间步长下的最大误差对比图，包含 2 个子图：

| 子图 | 标题 | 说明 |
|------|------|------|
| 左 | $\delta - \theta$ Gen. 3 | δ-θ 的最大 $\ell_1$ 误差随时间步长变化 |
| 右 | $V_m$ Gen. 3 | 电压幅值的最大 $\ell_1$ 误差随时间步长变化 |

**测试的时间步长：** `[5ms, 8ms, 10ms, 20ms, 25ms, 40ms]`

**图例：**
- Pure solver（纯 RK 求解器）
- Hybrid solver（PINN 混合求解器）

---

## Figure_7.png

**运行命令：**
```bash
cd plug
python main.py --study_selection 4
```

**图片含义：**

随机初始条件下的误差分布图，包含 2 个子图：

| 子图 | 标题 | 说明 |
|------|------|------|
| 左 | $\delta_3 - \omega_3$ | 第 3 台发电机的 δ-ω 误差分布 |
| 右 | $V_{m,3}$ | 第 3 台发电机的电压幅值误差分布 |

**测试方法：** 从 10 秒真值数据中随机选取 30 个初始点进行 2 秒仿真

**图例：**
- 橙色：纯 RK 求解器误差
- 蓝色：PINN 混合求解器误差

---

## Figure_8.png

**运行命令：**
```bash
cd plug
python main.py --study_selection 5
```

**图片含义：**

多时间步长的误差统计分布图，包含 4 个子图：

| 子图 | 标题 | 说明 |
|------|------|------|
| 1 | $\delta_3 - \omega_3$ | δ-ω 的误差中位数和上须 |
| 2-4 | $V_{m,3}$ | 电压幅值的误差中位数和上须 |

**测试的时间步长：** `[6ms, 8ms, 10ms, 14ms, 20ms, 24ms, 34ms, 40ms]`

**测试方法：** 每个时间步长进行 100 次随机初始条件的 1 步仿真

**图例：**
- 红色实线：纯 RK 求解器中位数
- 红色虚线：纯 RK 求解器上须
- 绿色实线：PINN 混合求解器中位数
- 绿色虚线：PINN 混合求解器上须

**坐标轴：** X 轴和 Y 轴均为对数尺度

---

## 快速生成所有图片

```bash
cd plug
python main.py --study_selection 1
python main.py --study_selection 2 --sim_time 2
python main.py --study_selection 2 --sim_time 10
python main.py --study_selection 3
python main.py --study_selection 4
python main.py --study_selection 5
```

---

## 命令行参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--study_selection` | 2 | 研究场景选择 (1-5) |
| `--sim_time` | 2.0 | 仿真时长（秒） |
| `--time_step_size` | 0.04 | 时间步长（秒） |
| `--machine` | 3 | 使用的 PINN 模型 (1, 2, 3) |
| `--rk_scheme` | trapezoidal | RK 积分格式 (trapezoidal, backward_euler) |
| `--event_type` | w_setpoint | 事件类型 (p_setpoint, w_setpoint) |
| `--event_location` | 3 | 事件位置 (1, 2, 3) |
| `--gpu` | 0 | GPU 设备编号 |
