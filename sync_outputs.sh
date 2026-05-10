#!/bin/bash

set -e

REMOTE=$1

if [ -z "$REMOTE" ]; then
    echo "Usage:"
    echo "./sync_outputs.sh root@117.50.248.132:23"
    exit 1
fi

# ===== 解析 remote =====
REMOTE_USER_HOST=$(echo ${REMOTE} | cut -d':' -f1)
REMOTE_PORT=$(echo ${REMOTE} | cut -d':' -f2)

BASE_DIR="/workspace/CVPR-2026-Workshop-WM-Track-main"

REMOTE_OUTPUT_DIR="${BASE_DIR}/outputs"
LOCAL_OUTPUT_DIR="${BASE_DIR}/"

mkdir -p "${LOCAL_OUTPUT_DIR}"

echo "======================================"
echo "REMOTE HOST : ${REMOTE_USER_HOST}"
echo "REMOTE PORT : ${REMOTE_PORT}"
echo "======================================"

rsync -avzP \
    -e "ssh -p ${REMOTE_PORT}" \
    ${REMOTE_USER_HOST}:${REMOTE_OUTPUT_DIR} \
    ${LOCAL_OUTPUT_DIR}

echo "======================================"
echo "outputs 同步完成"
echo "======================================"

# power-shell not work,use it in wsl2  ssh -p 23 root@106.75.127.8
#
# REMOTE="106.75.127.8" ; rsync -avzP -e "ssh -p 23" root@${REMOTE}:/workspace/CVPR-2026-Workshop-WM-Track-main/outputs/ "/mnt/d/Workspace/Main WORK/alpha/WM/video/giga-world-1/"