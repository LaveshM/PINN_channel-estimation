import pandas as pd
import numpy as np
np.random.seed(42)
# load file
df = pd.read_csv("Dataset/ue_positions_noisy.txt", sep=r"\s+")

# add Gaussian noise to x and y
noise_std = 0.5   # adjust as needed

df["x"] += np.random.normal(0, noise_std, size=len(df))
df["y"] += np.random.normal(0, noise_std, size=len(df))

# save
df.to_csv("data/ue_positions_noisy_0.5.txt", sep=' ', index=False)

print(df.head())