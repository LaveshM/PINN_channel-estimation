#!/bin/bash

mkdir -p logs

SNR_LIST=(-10 -5 0 5)
SPLIT_LIST=("loc" "random" "bloc")

GPU_COUNT=4
GPU_IDS=(0 1 2 3)

job_index=0

for snr in "${SNR_LIST[@]}"; do
for split in "${SPLIT_LIST[@]}"; do

    gpu=${GPU_IDS[$((job_index % GPU_COUNT))]}

    echo "Launching snr=$snr split=$split on GPU $gpu"

    CUDA_VISIBLE_DEVICES=$gpu python train.py \
        --smomp_file data/snr${snr}/initial_estimate_ls_real.npy \
        --accurate_file Dataset/3D_channel_15GHz_2x2_Pt50_real.npy \
        --user_positions_file data/ue_positions_noisy_0.5.txt \
        --split_type $split \
        --user_noise 0.5 \
        --snr $snr \
        > logs/snr${snr}_${split}.log 2>&1 &

    job_index=$((job_index + 1))

    # limit to 4 parallel jobs
    if (( job_index % GPU_COUNT == 0 )); then
        wait
    fi

done
done

wait

echo "All jobs done"