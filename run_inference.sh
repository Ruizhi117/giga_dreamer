#!/bin/bash

set -e

TASK=$1

if [ -z "$TASK" ]; then
    echo "Usage:"
    echo "./run_inference.sh task3"
    exit 1
fi

BASE_DIR="/workspace/CVPR-2026-Workshop-WM-Track-main"
MODEL_DIR="${BASE_DIR}/experiments/baseline_wm/${TASK}/models"

GPUS=(0 1 2 3)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate giga_torch

CHECKPOINTS=($(ls ${MODEL_DIR} | sort -V))

idx=0

for pt in "${CHECKPOINTS[@]}"
do
    epoch=$(echo ${pt} | sed -E 's/checkpoint_epoch_([0-9]+)_step_.*/\1/')

    gpu=${GPUS[$((idx % ${#GPUS[@]}))]}

    echo "==================================="
    echo "checkpoint : ${pt}"
    echo "epoch      : ${epoch}"
    echo "gpu        : ${gpu}"
    echo "==================================="

    cd ${BASE_DIR}

    CUDA_VISIBLE_DEVICES=${gpu} \
    python scripts/inference.py \
        --device_list 0 \
        --transformer_model_path ${MODEL_DIR}/${pt}/transformer/ \
        --output_dir outputs/${TASK}_${epoch} \
        --task ${TASK} \
        --mode offline \
        --policy_norm_stats_path NA \
        --policy_ckpt_dir NA \
        > logs_${TASK}_${epoch}.txt 2>&1 &

    idx=$((idx + 1))
done

echo "==============================="
echo "所有任务已后台提交"
echo "==============================="
