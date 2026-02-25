import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

with h5py.File("pinball_data.h5", "r") as f:
    time = f["timeseries/time"][:]
    CL2  = f["timeseries/CL2"][:]
    CL3  = f["timeseries/CL3"][:]
    CD   = f["timeseries/CD"][:]

# --- 1. Plot the full signal so you can eyeball it ---
fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
axes[0].plot(time, CL2, label="CL2", linewidth=0.8)
axes[0].plot(time, CL3, label="CL3", linewidth=0.8)
axes[0].set_ylabel("Lift")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(time, CD, color="black", linewidth=0.8)
axes[1].set_ylabel("Drag CD")
axes[1].set_xlabel("Time (s)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("full_signal.png", dpi=150)

# --- 2. Compute rolling standard deviation of CL2 ---
# High std = oscillating = shedding. Low std = settled/steady.
window = 500  # samples
roll_std = np.array([
    CL2[max(0, i-window):i].std()
    for i in range(1, len(CL2)+1)
])

fig2, ax = plt.subplots(figsize=(16, 4))
ax.plot(time, roll_std, color="purple", linewidth=0.8)
ax.set_ylabel("Rolling std of CL2")
ax.set_xlabel("Time (s)")
ax.set_title("High values = active shedding")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("rolling_std.png", dpi=150)

# --- 3. Find where shedding starts ---
threshold = 0.05 * roll_std.max()  # 5% of max std
shedding_start_idx = np.argmax(roll_std > threshold)
shedding_start_time = time[shedding_start_idx]
print(f"Shedding appears to start around t={shedding_start_time:.1f}s")

# --- 4. Find peaks to measure shedding frequency ---
peaks, _ = find_peaks(CL2[shedding_start_idx:], height=0.01)
if len(peaks) > 1:
    periods = np.diff(time[shedding_start_idx:][peaks])
    print(f"Mean shedding period: {periods.mean():.3f}s")
    print(f"Shedding frequency:   {1/periods.mean():.3f} Hz")
    print(f"Strouhal number (St): {1/periods.mean():.3f}")  # St = f*D/U, D=U=1 here

# --- 5. Extract a clean shedding window (last 20% of data) ---
start = int(0.8 * len(time))
print(f"\nClean shedding window: t={time[start]:.1f}s to t={time[-1]:.1f}s")
print(f"  Mean CD in window: {CD[start:].mean():.4f}")
print(f"  Std  CL2 in window: {CL2[start:].std():.4f}")