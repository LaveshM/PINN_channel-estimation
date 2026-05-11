#!/bin/bash

mkdir -p logs

SNR_LIST=(-10 -5 0 5)
SPLIT_LIST=("random" "bloc")

GPU_IDS=(0)          # multiple GPUs here
GPU_COUNT=${#GPU_IDS[@]}

MAX_JOBS_PER_GPU=1

# track jobs per GPU
declare -A gpu_jobs

# init counters
for gpu in "${GPU_IDS[@]}"; do
    gpu_jobs[$gpu]=0
done

job_index=0

for snr in "${SNR_LIST[@]}"; do
for split in "${SPLIT_LIST[@]}"; do

    gpu=${GPU_IDS[$((job_index % GPU_COUNT))]}

    echo "Launching snr=$snr split=$split on GPU $gpu"

    CUDA_VISIBLE_DEVICES=$gpu python train.py \
        --smomp_file data/snr${snr}/initial_estimate_ls_real.npy \
        --accurate_file data/3D_channel_15GHz_2x2_Pt50_real.npy \
        --user_positions_file data/ue_positions_noisy_0.5.txt \
        --split_type $split \
        --user_noise 0.5 \
        --snr $snr \
        --continue_training \
        > logs/snr${snr}_${split}.log 2>&1 &

    # update GPU job count
    gpu_jobs[$gpu]=$((gpu_jobs[$gpu] + 1))

    # if GPU reached limit → wait for all jobs to finish
    if (( gpu_jobs[$gpu] % MAX_JOBS_PER_GPU == 0 )); then
        wait
        gpu_jobs[$gpu]=0
    fi

    job_index=$((job_index + 1))

done
done

wait

echo "All jobs done"