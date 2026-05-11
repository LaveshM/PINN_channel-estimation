# import pandas as pd
# import numpy as np
# np.random.seed(42)
# # load file
# df = pd.read_csv("Dataset/ue_positions.txt", sep=r"\s+")

# # add Gaussian noise to x and y
# noise_std = 0.5   # adjust as needed

# df["x"] += np.random.normal(0, noise_std, size=len(df))
# df["y"] += np.random.normal(0, noise_std, size=len(df))

# # save
# df.to_csv("data/ue_positions_noisy_0.5.txt", sep=' ', index=False)

import pandas as pd
import numpy as np
import argparse

# argument parser
parser = argparse.ArgumentParser(description="Add Gaussian noise to UE positions")
parser.add_argument("--noise", type=float, required=True,
                    help="Gaussian noise standard deviation")

args = parser.parse_args()
noise_std = args.noise

# reproducibility
np.random.seed(42)

# load file
df = pd.read_csv("Dataset/ue_positions.txt", sep=r"\s+")

# add Gaussian noise
df["x"] += np.random.normal(0, noise_std, size=len(df))
df["y"] += np.random.normal(0, noise_std, size=len(df))

# output file
output_file = f"data/ue_positions_noisy_{noise_std}.txt"

# save
df.to_csv(output_file, sep=' ', index=False)

print(f"Saved noisy dataset to: {output_file}")