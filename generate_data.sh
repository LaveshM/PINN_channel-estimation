#!/bin/bash

mkdir -p data/snr-10 data/snr-5 data/snr0 data/snr5

python make_correct_channels.py \
    --csv Dataset/15GHz_concatenated_data.csv \
    --out data/3D_channel_15GHz_2x2_Pt50.npy \
    --pt 50 --bw 4e8 &

for noise in 0, 0.1, 0.5
do
    python add_user_noise.py \
        --noise ${noise} &
done

for snr in -10 -5 0 5
do
    python init_estimation.py \
      --true-channels data/3D_channel_15GHz_2x2_Pt50.npy \
      --output data/snr${snr}/initial_estimate_ls.npy \
      --snr ${snr} &

done

wait

echo "All runs completed."