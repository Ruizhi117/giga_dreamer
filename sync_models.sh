#!/bin/bash

set -e

TASK=$1
REMOTE=$2
PASSWORD=$3

if [ -z "$TASK" ] || [ -z "$REMOTE" ] || [ -z "$PASSWORD" ]; then
    echo "Usage:"
    echo "./sync_models.sh task3 root@106.75.127.8:23 your_password"
    exit 1
fi

# =========================
# 解析远程信息
# =========================
REMOTE_USER_HOST=$(echo ${REMOTE} | cut -d':' -f1)
REMOTE_PORT=$(echo ${REMOTE} | cut -d':' -f2)

BASE_DIR="/workspace/CVPR-2026-Workshop-WM-Track-main"

REMOTE_MODEL_DIR="${BASE_DIR}/experiments/baseline_wm/${TASK}/models"
LOCAL_MODEL_DIR="${BASE_DIR}/experiments/baseline_wm/${TASK}/models"

mkdir -p "${LOCAL_MODEL_DIR}"

echo "======================================"
echo "TASK         : ${TASK}"
echo "REMOTE HOST  : ${REMOTE_USER_HOST}"
echo "REMOTE PORT  : ${REMOTE_PORT}"
echo "======================================"

# =========================
# 检查 sshpass
# =========================
if ! command -v sshpass &> /dev/null; then
    echo "[ERROR] sshpass 未安装"
    echo "Ubuntu/WSL2:"
    echo "sudo apt install sshpass -y"
    exit 1
fi

# =========================
# 获取 checkpoint 列表
# =========================
CHECKPOINTS=$(sshpass -p "${PASSWORD}" \
    ssh -o StrictHostKeyChecking=no \
    -p ${REMOTE_PORT} \
    ${REMOTE_USER_HOST} \
    "ls ${REMOTE_MODEL_DIR}")

echo "发现 checkpoints:"
echo "${CHECKPOINTS}"

echo "======================================"

# =========================
# 并行 rsync
# =========================
MAX_JOBS=4

sync_one() {
    pt=$1

    echo "[SYNC] ${pt}"

    mkdir -p "${LOCAL_MODEL_DIR}/${pt}"

    sshpass -p "${PASSWORD}" \
    rsync -azP \
        --partial \
        --inplace \
        -e "ssh -o StrictHostKeyChecking=no -p ${REMOTE_PORT}" \
        ${REMOTE_USER_HOST}:${REMOTE_MODEL_DIR}/${pt}/transformer \
        ${LOCAL_MODEL_DIR}/${pt}/
}

export -f sync_one
export PASSWORD
export REMOTE_PORT
export REMOTE_USER_HOST
export REMOTE_MODEL_DIR
export LOCAL_MODEL_DIR

# GNU parallel 优先
if command -v parallel &> /dev/null; then
    echo "[INFO] 使用 GNU parallel 并行传输"

    echo "${CHECKPOINTS}" | \
        parallel -j ${MAX_JOBS} sync_one {}

else
    echo "[INFO] 使用 bash 后台任务并行"

    JOBS=0

    for pt in ${CHECKPOINTS}
    do
        sync_one ${pt} &

        JOBS=$((JOBS+1))

        if [ ${JOBS} -ge ${MAX_JOBS} ]; then
            wait
            JOBS=0
        fi
    done

    wait
fi

echo "======================================"
echo "同步完成"
echo "======================================"

#sudo apt update
#sudo apt install sshpass parallel rsync -y