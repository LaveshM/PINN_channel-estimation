#!/bin/bash

mkdir -p data/snr-10 data/snr-5 data/snr0 data/snr5

for snr in -10 -5 0 5
do
    python init_estimation.py \
      --true-channels Dataset/3D_channel_15GHz_2x2_Pt50.npy \
      --output data/snr${snr}/initial_estimate_ls.npy \
      --snr ${snr} &
done

wait

echo "All runs completed."