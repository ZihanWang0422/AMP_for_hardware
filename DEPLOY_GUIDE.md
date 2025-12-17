# Go1 AMP Sim2Real 部署指南

## 目录结构

```
AMP_for_hardware/
├── deploy_go1.sh                          # 快速启动脚本
├── deploy/
│   ├── deploy_real/
│   │   ├── deploy_go1_lowlevel.py        # 主部署脚本
│   │   ├── test_joystick.py              # 遥控器测试脚本
│   │   └── test_lowlevel_pose.py         # 姿态测试脚本
│   └── exported_policy/
│       └── go1/
│           └── policy_45_continus.pt     # AMP 策略文件
└── unitree_legged_sdk/                    # Unitree SDK
```

## 前置准备

### 1. 机器人准备
机器人必须先进入 **Damping Mode**:
```
控制序列: [L2+A] → [L2+B] → [L1+L2+START]
```
完成后机器人应该趴在地上，关节可以自由移动。

### 2. 网络配置
确保机器人和电脑在同一网络:
- 机器人 IP: `192.168.123.10`
- 电脑 IP: `192.168.123.162` (或同一网段)
- 测试连接: `ping 192.168.123.10`

### 3. Conda 环境
```bash
conda activate amp_hw
```

## 快速启动

### 方式一：使用启动脚本（推荐）

```bash
cd /home/wzh/amp/isaacgym/AMP_for_hardware

# 测试遥控器
./deploy_go1.sh test

# 部署策略
./deploy_go1.sh
```

### 方式二：直接运行 Python 脚本

```bash
cd /home/wzh/amp/isaacgym/AMP_for_hardware

# 激活环境
conda activate amp_hw

# 测试遥控器
python deploy/deploy_real/test_joystick.py

# 部署策略
python deploy/deploy_real/deploy_go1_lowlevel.py --policy deploy/exported_policy/go1/policy_45_continus.pt
```

## 控制流程

### 遥控器按键功能

| 按键 | 功能 | 说明 |
|------|------|------|
| **Start** | 站立 | 从趴姿平滑过渡到站立（2秒） |
| **A** | 启动 RL 策略 | 切换到策略控制模式 |
| **Select** | 停止并退出 | 紧急停止，切换到阻尼模式 |
| **左摇杆 Y** | 前进/后退 | 范围: -1.5 ~ +1.5 m/s |
| **左摇杆 X** | 左右侧移 | 范围: -0.5 ~ +0.5 m/s |
| **右摇杆 X** | 转向 | 范围: -1.0 ~ +1.0 rad/s |

### 操作步骤

1. **启动脚本**
   ```bash
   ./deploy_go1.sh
   ```

2. **等待连接**
   - 脚本会自动连接机器人
   - 显示当前关节角度和 IMU 数据

3. **进入阻尼模式**
   - 自动进入阻尼模式（Damping Mode）
   - 关节可以自由移动
   - 等待 Start 键

4. **按 Start 键站立**
   - 机器人从当前姿态平滑过渡到站立
   - 过渡时间: 2 秒
   - Kp: 20 → 60, Kd: 1 → 2

5. **按 A 键启动策略**
   - 切换到 RL 策略控制
   - 使用摇杆控制运动
   - 每秒打印状态信息

6. **按 Select 键停止**
   - 紧急停止
   - 切换到阻尼模式
   - 安全退出

## 配置说明

### AMP 策略配置（与训练环境一致）

```python
# 控制频率
control_dt = 0.002      # 500Hz 底层控制
policy_dt = 0.03        # 33Hz 策略推理 (decimation=15)

# PD 增益
kp_walk = 80.0          # 行走时 Kp
kd_walk = 1.0           # 行走时 Kd

# 动作缩放
action_scale = 0.25     # 与训练一致

# 观测缩放
obs_scales_ang_vel = 0.25
obs_scales_dof_pos = 1.0
obs_scales_dof_vel = 0.05

# 命令缩放
command_scale = [2.0, 2.0, 0.25]

# 默认姿态 (FL, FR, RL, RR)
default_angles = [0.0, 0.9, -1.8] * 4
```

### 观测维度（45维）

根据 `go1_amp_config.py` 和 `legged_robot.py`:

```python
obs = [
    base_ang_vel * 0.25,           # 3维: 角速度
    projected_gravity,              # 3维: 投影重力
    commands * [2.0, 2.0, 0.25],   # 3维: 速度命令
    (dof_pos - default) * 1.0,     # 12维: 关节角度偏差
    dof_vel * 0.05,                # 12维: 关节速度
    last_actions                    # 12维: 上一步动作
]
```

**注意**: 45维观测去掉了 `base_lin_vel`（相比 48 维特权观测）

### 关节顺序映射

- **训练环境顺序**: FL, FR, RL, RR (前左, 前右, 后左, 后右)
- **SDK 顺序**: FR, FL, RR, RL (前右, 前左, 后右, 后左)
- **映射数组**: `[3,4,5, 0,1,2, 9,10,11, 6,7,8]`

## 调试工具

### 1. 测试遥控器

```bash
./deploy_go1.sh test
```

输出示例:
```
摇杆: LX=+0.00 LY=+0.85 RX=-0.32 RY=+0.00 | 按键: Start, A
>>> 检测到 Start 键被按下！
>>> 检测到 A 键被按下！
```

### 2. 测试姿态

```bash
python deploy/deploy_real/test_lowlevel_pose.py
```

验证默认姿态 `[0.0, 0.9, -1.8]` 是否稳定站立。

### 3. 查看配置

```bash
conda run -n amp_hw python -c "
from deploy.deploy_real.deploy_go1_lowlevel import Go1Config
c = Go1Config()
print(f'kp_walk: {c.kp_walk}')
print(f'kd_walk: {c.kd_walk}')
print(f'action_scale: {c.action_scale}')
print(f'num_obs: {c.num_obs}')
"
```

## 故障排查

### 问题1: 未检测到机器人

**症状**: `警告: 未检测到有效的关节数据`

**解决方案**:
1. 检查机器人是否开机
2. 检查网络: `ping 192.168.123.10`
3. 检查 IP 配置: `ifconfig` 或 `ip addr`

### 问题2: 遥控器无响应

**症状**: 按键无反应，摇杆数据全为 0

**解决方案**:
1. 运行 `./deploy_go1.sh test` 测试遥控器
2. 检查遥控器是否开机
3. 检查遥控器与机器人配对

### 问题3: 机器人未进入阻尼模式

**症状**: 脚本启动前机器人关节无法移动

**解决方案**:
必须手动进入 Damping Mode:
```
[L2+A] → [L2+B] → [L1+L2+START]
```

### 问题4: 策略控制不稳定

**症状**: 机器人抖动或摔倒

**解决方案**:
1. 降低 Kp/Kd:
   ```bash
   python deploy_go1_lowlevel.py --kp-walk 60 --kd-walk 0.8
   ```
2. 检查地面是否平整
3. 检查默认姿态是否正确站立

### 问题5: Conda 环境问题

**症状**: `ModuleNotFoundError` 或 `ImportError`

**解决方案**:
```bash
# 确认环境存在
conda env list

# 激活环境
conda activate amp_hw

# 检查 Python 路径
which python

# 安装缺失依赖
conda install pytorch numpy
```

## 安全注意事项

1. **测试环境**
   - 在空旷、平坦的地面测试
   - 周围 2 米内无障碍物
   - 准备紧急停止（Select 键）

2. **机器人状态**
   - 确保电量充足（>30%）
   - 检查关节无异响
   - 观察温度（不应过热）

3. **控制权限**
   - 始终保持遥控器在手
   - 熟悉 Select 紧急停止
   - 首次测试速度命令从小开始

## 性能参数

- **控制频率**: 500 Hz (底层), 33 Hz (策略)
- **延迟**: < 30ms (网络 + 推理 + 控制)
- **站立时间**: ~2 秒
- **最大速度**: vx=1.5 m/s, vy=0.5 m/s, yaw=1.0 rad/s

## 文件清单

| 文件 | 说明 |
|------|------|
| `deploy_go1.sh` | 快速启动脚本 |
| `deploy_go1_lowlevel.py` | 主部署脚本（793行） |
| `test_joystick.py` | 遥控器测试工具 |
| `test_lowlevel_pose.py` | 姿态验证工具 |
| `policy_45_continus.pt` | AMP 策略文件 |
| `go1_amp_config.py` | 训练环境配置 |
| `DEPLOY_GUIDE.md` | 本文档 |

## 联系与支持

如有问题，请检查:
1. 本文档的"故障排查"部分
2. 运行测试脚本收集信息
3. 查看终端输出的详细错误信息
