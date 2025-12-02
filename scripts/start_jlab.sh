#!/bin/bash
###############################################################
#  TRACE_26ACL - One-click JupyterLab Starter
#  Author: chengjie.zheng001
#  Description:
#      - 自动进入正确项目目录
#      - 自动启动 tmux 会话 (jlab)
#      - 自动进入 apptainer 容器
#      - 自动启动 JupyterLab
###############################################################

# === 配置区：根据你机器实际情况设置 ===

PROJECT_DIR="/beacon/data01/chengjie.zheng001/Projects/07TRACE_LLMs/26ACL/TRACE_26ACL"
SIF_PATH="$PROJECT_DIR/apptainer/trace.sif"
SESSION_NAME="jlab"
JUPYTER_PORT=8888

# === 开始执行 ===

echo ">>> Checking existing tmux session..."
tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? -eq 0 ]; then
    echo "======================================================="
    echo " tmux session '$SESSION_NAME' 已存在。"
    echo " JupyterLab 很可能已经在运行。"
    echo " 你可以 attach："
    echo ""
    echo "     tmux attach -t $SESSION_NAME"
    echo ""
    echo " 或 kill 后重启："
    echo "     tmux kill-session -t $SESSION_NAME"
    echo "======================================================="
    exit 0
fi

echo ">>> Starting new tmux session: $SESSION_NAME"
tmux new-session -d -s $SESSION_NAME

echo ">>> Sending startup commands to tmux window..."

tmux send-keys -t $SESSION_NAME "cd $PROJECT_DIR" C-m
tmux send-keys -t $SESSION_NAME \
    "apptainer shell --nv --bind \$(pwd):/workspace $SIF_PATH" C-m

# 进入容器后启动 jupyter lab
tmux send-keys -t $SESSION_NAME \
    "cd /workspace && jupyter lab --ip=0.0.0.0 --port=$JUPYTER_PORT --no-browser" C-m

echo "======================================================="
echo " JupyterLab 已在后台启动 (via tmux session: $SESSION_NAME)"
echo ""
echo " 你可以 attach 到 tmux 查看："
echo "     tmux attach -t $SESSION_NAME"
echo ""
echo " 你可以从本地浏览器访问："
echo "     http://<你的服务器IP>:$JUPYTER_PORT/lab"
echo ""
echo " 若你用 SSH Tunnel，则访问："
echo "     http://127.0.0.1:$JUPYTER_PORT/lab"
echo ""
echo "======================================================="
