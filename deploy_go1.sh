#!/bin/bash
# Go1 部署启动脚本
# 自动激活 conda 环境并运行部署脚本

set -e

echo "=================================="
echo "Go1 AMP Sim2Real 部署"
echo "=================================="
echo ""

# 检查是否在正确目录
if [ ! -d "deploy/deploy_real" ]; then
    echo "错误: 请在 AMP_for_hardware 根目录运行此脚本"
    exit 1
fi

# 检查 conda 环境
if ! conda env list | grep -q "amp_hw"; then
    echo "错误: conda 环境 'amp_hw' 不存在"
    echo "请先创建环境: conda create -n amp_hw python=3.8"
    exit 1
fi

# 解析参数
SCRIPT_MODE="deploy"
POLICY_PATH="deploy/exported_policy/go1/policy_45_continus.pt"

if [ "$1" == "test" ] || [ "$1" == "joystick" ]; then
    SCRIPT_MODE="test"
    echo "模式: 测试遥控器"
    echo ""
    echo "运行命令:"
    echo "  conda run -n amp_hw python deploy/deploy_real/test_joystick.py"
    echo ""
    conda run -n amp_hw python deploy/deploy_real/test_joystick.py
    exit 0
fi

# 部署模式
echo "模式: 部署 AMP 策略"
echo ""
echo "检查策略文件..."
if [ ! -f "$POLICY_PATH" ]; then
    echo "错误: 策略文件不存在: $POLICY_PATH"
    exit 1
fi
echo "✓ 策略文件: $POLICY_PATH"
echo ""

echo "前置条件检查:"
echo "  1. 机器人是否已开机？"
echo "  2. 网络连接是否正常？ (ping 192.168.123.10)"
echo "  3. 机器人是否在 Damping Mode？"
echo "     控制序列: [L2+A], [L2+B], [L1+L2+START]"
echo ""
read -p "确认以上条件都满足，按 Enter 继续，Ctrl+C 取消..."
echo ""

echo "运行命令:"
echo "  conda run -n amp_hw python deploy/deploy_real/deploy_go1_lowlevel.py --policy $POLICY_PATH"
echo ""

# 运行部署脚本
conda run -n amp_hw python deploy/deploy_real/deploy_go1_lowlevel.py --policy "$POLICY_PATH" "$@"
