#!/bin/bash


python train.py         --smomp_file data/snr0/initial_estimate_ls_real.npy         --accurate_file data/channels_augmented_real.npy         --user_positions_file Dataset/ue_positions.txt         --split_type new_random         --user_noise 0.5         --snr 0  --continue_training &