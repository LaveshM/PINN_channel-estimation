import numpy as np

ch0 = np.load("/workspace/data/run_0000/channels.npy")
ch0_pwr = np.sum(np.abs(ch0)**2, axis=(1,2,3))

print(f"{'Run':<8} {'N_blocked':<12} {'Blk%':<8} {'Atten mean(dB)':<17} {'Atten median(dB)':<18} {'Atten min(dB)':<15} {'Atten max(dB)'}")
print("-" * 90)

for r in range(1, 11):
    path = f"/workspace/data/run_{r:04d}"
    ch = np.load(f"{path}/channels.npy")
    mask = np.load(f"{path}/blocked_mask.npy")

    ch_pwr = np.sum(np.abs(ch)**2, axis=(1,2,3))

    # attenuation = how much power dropped vs run_0000 for the same user
    ratio = ch0_pwr[mask] / (ch_pwr[mask] + 1e-300)
    atten_db = 10 * np.log10(ratio)

    n = mask.sum()
    print(f"run_{r:04d}  {n:<12} {100*n/len(mask):<8.1f} "
          f"{atten_db.mean():<17.2f} {np.median(atten_db):<18.2f} "
          f"{atten_db.min():<15.2f} {atten_db.max():.2f}")
