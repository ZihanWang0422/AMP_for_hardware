## Go1 sim2sim2real

### sim2sim

#### 文件路径说明
- **Policy 模型**：放置在 `/deploy/exported_policy/go1/` 目录下
- **MuJoCo XML 场景**：放置在 `/deploy/assets/go1/` 目录下

#### 运行命令
进入 `deploy/deploy_mujoco` 目录，运行：
```bash
python3 jit_mujoco_keyboard.py --model policy_45_continus.pt --xml scene.xml
```

#### 键盘控制
- **前进/后退**：I/K 或 小键盘 8/2（步进：0.1 m/s）
- **左右平移**：U/O 或 小键盘 7/9（步进：0.05 m/s）
- **左右转向**：J/L 或 小键盘 4/6（步进：0.1 rad/s）
- **紧急停止**：空格 或 小键盘 5
- **退出仿真**：Q 或 ESC

#### 可选参数
- `--headless`：无可视化运行
- `--action-scale`：动作缩放因子（默认：0.25）
- `--policy-hz`：策略推理频率（默认：33 Hz）
- `--sim-dt`：仿真时间步长（默认：0.002 秒）

### sim2real

### 1. **准备工作**
- ✅ 已完成:在 deploy_policy.py 中添加了您的 checkpoint 路径 `../../exported_policy/go1/policy_45_continus.pt`
- 启动 Go1 机器人
- 通过以太网连接到机器人
- 确保可以 SSH 到 NX (192.168.123.15)

### 2. **让机器人进入阻尼模式**
使用手柄操作:
- `L2+A`
- `L2+B` 
- `L1+L2+START`
机器人应该会躺在地上

### 3. **发送代码到 Go1**
```bash
cd /home/wzh/amp/isaacgym/AMP_for_hardware/deploy/go1_gym_deploy/scripts
./send_to_unitree.sh
```

### 4. **在 Go1 上安装部署代码**
SSH 到 Go1 后运行:
```bash
chmod +x installer/install_deployment_code.sh
cd ~/go1_gym/go1_gym_deploy/scripts
chmod +x ../installer/install_deployment_code.sh 
sudo ../installer/install_deployment_code.sh
```

### 5. **运行部署**
**确保机器人处于安全位置并悬挂起来**

打开两个 SSH 终端:

**终端 1:**
```bash
cd ~/go1_gym/go1_gym_deploy/autostart
./start_unitree_sdk.sh
```

**终端 2:**
```bash
ssh unitree@192.168.123.15
cd ~/go1_gym/go1_gym_deploy/docker
sudo make autostart && sudo docker exec -it foxy_controller bash
```

**在 Docker 容器内:**
```bash
cd /home/isaac/go1_gym && rm -r build && python3 setup.py install && cd go1_gym_deploy/scripts && python3 deploy_policy.py
```

### 6. **控制机器人**
- **R2**: 第一次按 - 机器人伸展腿部(校准)
- **R2**: 第二次按 - 启动策略执行
- **L1/R1**: 在 deploy_policy.py 中的策略列表之间切换(如果有多个策略)

## 您的配置
当前 deploy_policy.py 配置:
- Policy 路径: `../../exported_policy/go1/policy_45_continus.pt`
- 实验名称: `go1_policy_45_continus`
- 最大速度: `max_vel=3.5`
- 最大偏航速度: `max_yaw_vel=5.0`

现在您可以按照上述步骤进行部署了!需要注意的是,一定要确保机器人在安全的位置并被悬挂起来进行第一次测试。