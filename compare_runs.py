import numpy as np

runs_to_check = list(range(1, 11)) + list(range(11, 21))

print(f"{'Run':<10} {'Shape':<24} {'Blk%':<7} {'Blk_pwr(dB)':<14} {'Unblk_pwr(dB)':<16} {'Ratio(dB)'}")
print("-" * 85)

for r in runs_to_check:
    path = f"/workspace/data_old/run_{r:04d}"
    try:
        ch = np.load(f"{path}/channels.npy")
        mask = np.load(f"{path}/blocked_mask.npy")
        
        ch_flat = ch.reshape(len(mask), -1)
        blocked = ch_flat[mask]
        unblocked = ch_flat[~mask]
        
        blocked_pwr = float(np.mean(np.abs(blocked)**2)) if len(blocked) > 0 else float('nan')
        unblocked_pwr = float(np.mean(np.abs(unblocked)**2)) if len(unblocked) > 0 else float('nan')
        
        total = len(mask)
        n_blocked = int(mask.sum())
        pct = 100*n_blocked/total
        
        b_db = 10*np.log10(blocked_pwr+1e-30)
        u_db = 10*np.log10(unblocked_pwr+1e-30)
        ratio = b_db - u_db
        
        tag = " <-- new model" if r >= 11 else ""
        print(f"run_{r:04d}   {str(ch.shape):<24} {pct:<7.1f} {b_db:<14.2f} {u_db:<16.2f} {ratio:.2f}{tag}")
    except Exception as e:
        print(f"run_{r:04d}: ERROR - {e}")
