"""
Fluidic Pinball Re=100 – Vortex Shedding Validation

Focuses on 10-15 vortex shedding cycles after the initial transient,
computes time-averaged and oscillation statistics, and compares against
Table SI 4 reference values.

Reference (Table SI 4, Re=100 uncontrolled baseline):
  CD1_mean  : ~1.010
  CD2_mean  : ~1.010
  CD3_mean  : ~1.010
  CD_total  : ~3.030
  CL2_amp   : ~0.170   (half peak-to-peak)
  CL3_amp   : ~0.170
  Strouhal  : ~0.134   (St = f·D/U, D=U=1)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks, welch
import os, json
import h5py

# Setup
OUT = "pinball_analysis"
os.makedirs(OUT, exist_ok=True)

# Table SI 4 reference values
REF = {
    "CD1_mean":      1.010,
    "CD2_mean":      1.010,
    "CD3_mean":      1.010,
    "CD_total_mean": 3.030,
    "CL2_amp":       0.170,
    "CL3_amp":       0.170,
    "St":            0.134,
}

# Load data
# Load data
print("Loading simulation data...")
with h5py.File("pinball_timeseries.h5", "r") as d:
    time   = d["timeseries/time"][:].astype(np.float64)
    CL1    = d["timeseries/CL1"][:].astype(np.float64)
    CL2    = d["timeseries/CL2"][:].astype(np.float64)
    CL3    = d["timeseries/CL3"][:].astype(np.float64)
    CD1    = d["timeseries/CD1"][:].astype(np.float64)
    CD2    = d["timeseries/CD2"][:].astype(np.float64)
    CD3    = d["timeseries/CD3"][:].astype(np.float64)
    drag   = d["timeseries/drag_total"][:].astype(np.float64)
dt     = float(time[1] - time[0])
print(f"  t = 0 ... {time[-1]:.0f} s,  dt = {dt} s,  N = {len(time):,} steps")

# Detect shedding onset and select cycles
T_ONSET = 260.0
T_START_CYCLES = 280.0

mask_full = time >= T_ONSET
peaks_idx, _   = find_peaks(CL2[mask_full], prominence=0.005, distance=int(5/dt))
troughs_idx, _ = find_peaks(-CL2[mask_full], prominence=0.002, distance=int(5/dt))
t_peaks        = time[mask_full][peaks_idx]
t_troughs      = time[mask_full][troughs_idx]

periods = np.diff(t_peaks)
T_shed  = float(np.median(periods))
f_shed  = 1.0 / T_shed
St      = f_shed

print(f"\nShedding detected after t = {T_ONSET:.0f} s")
print(f"  {len(peaks_idx)} peaks  |  median period T = {T_shed:.3f} s  |  St = {St:.4f}")

# Pick 12 cycles starting at first clean peak after T_START_CYCLES
N_CYC = 12
valid_starts = t_peaks[t_peaks >= T_START_CYCLES]
if len(valid_starts) >= N_CYC + 1:
    t_win_start = valid_starts[0]
    t_win_end   = valid_starts[N_CYC]
    n_cycles    = N_CYC
else:
    t_win_start = t_peaks[0]
    t_win_end   = t_peaks[-1]
    n_cycles    = len(t_peaks) - 1

print(f"  Analysis window: t = {t_win_start:.1f} - {t_win_end:.1f} s  ({n_cycles} cycles)")

win  = (time >= t_win_start) & (time <= t_win_end)
t_w  = time[win]
CL1w = CL1[win];  CL2w = CL2[win];  CL3w = CL3[win]
CD1w = CD1[win];  CD2w = CD2[win];  CD3w = CD3[win]
Dw   = drag[win]

# Statistics
pk_w, _  = find_peaks( CL2w, prominence=0.003, distance=int(5/dt))
tr_w, _  = find_peaks(-CL2w, prominence=0.003, distance=int(5/dt))
pk3_w, _ = find_peaks(-CL3w, prominence=0.002, distance=int(5/dt))
tr3_w, _ = find_peaks( CL3w, prominence=0.002, distance=int(5/dt))

def half_pp(arr, peak_i, trough_i):
    n = min(len(peak_i), len(trough_i))
    if n == 0:
        return float(np.ptp(arr) / 2)
    return float(np.mean(arr[peak_i[:n]] - arr[trough_i[:n]]) / 2)

stats = {
    "n_cycles":          n_cycles,
    "t_win_start_s":     float(t_win_start),
    "t_win_end_s":       float(t_win_end),
    "T_shedding_s":      float(T_shed),
    "T_shedding_std_s":  float(np.std(periods)),
    "f_shedding_Hz":     float(f_shed),
    "St_simulation":     float(St),
    "St_reference":      REF["St"],
    "CD1_mean":          float(np.mean(CD1w)),
    "CD2_mean":          float(np.mean(CD2w)),
    "CD3_mean":          float(np.mean(CD3w)),
    "CD_total_mean":     float(np.mean(Dw)),
    "CD1_std":           float(np.std(CD1w)),
    "CD2_std":           float(np.std(CD2w)),
    "CD3_std":           float(np.std(CD3w)),
    "CL1_mean":          float(np.mean(CL1w)),
    "CL2_mean":          float(np.mean(CL2w)),
    "CL3_mean":          float(np.mean(CL3w)),
    "CL2_amp":           half_pp(CL2w, pk_w, tr_w),
    "CL3_amp":           half_pp(-CL3w, pk3_w, tr3_w),
    "CL2_rms":           float(np.sqrt(np.mean((CL2w - CL2w.mean())**2))),
    "CL3_rms":           float(np.sqrt(np.mean((CL3w - CL3w.mean())**2))),
}

freq_cl2, psd_cl2 = welch(CL2w - CL2w.mean(), fs=1/dt, nperseg=min(4096, len(CL2w)//2))
f_peak_spec = freq_cl2[np.argmax(psd_cl2)]
St_spec     = f_peak_spec
stats["St_spectral"] = float(St_spec)

# Comparison Table
cmp_rows = []
pairs = [
    ("CD1_mean",      "CD1_mean",      "C_D1 (mean)"),
    ("CD2_mean",      "CD2_mean",      "C_D2 (mean)"),
    ("CD3_mean",      "CD3_mean",      "C_D3 (mean)"),
    ("CD_total_mean", "CD_total_mean", "C_D total (mean)"),
    ("CL2_amp",       "CL2_amp",       "C_L2 amplitude"),
    ("CL3_amp",       "CL3_amp",       "C_L3 amplitude"),
    ("St_simulation", "St",            "Strouhal (St)"),
]
for sk, rk, label in pairs:
    sv = stats[sk]; rv = REF[rk]
    err = 100 * abs(sv - rv) / abs(rv)
    cmp_rows.append({"label": label, "sim": sv, "ref": rv, "err_pct": err})

print("\n" + "="*68)
print(f"  {'Quantity':<24}  {'Simulated':>10}  {'Table SI4':>10}  {'Error %':>8}")
print("  " + "-"*62)
for r in cmp_rows:
    flag = "  OK" if r["err_pct"] < 5 else ("  ~" if r["err_pct"] < 20 else "  FAIL")
    print(f"  {r['label']:<24}  {r['sim']:>10.4f}  {r['ref']:>10.4f}  {r['err_pct']:>7.1f}%{flag}")
print("="*68)

stats["comparison"] = cmp_rows
with open(f"{OUT}/stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print(f"\nSaved {OUT}/stats.json")

# FIGURE 1 -- Full timeseries with annotated regions
print("\nPlotting Fig 1 - full timeseries...")
fig, axes = plt.subplots(3, 1, figsize=(18, 9), sharex=True)
fig.suptitle("Fluidic Pinball Re=100 -- Full Timeseries (Uncontrolled)", fontsize=14, fontweight="bold")

axes[0].plot(time, drag, color="black", lw=0.5, label="Total C_D")
axes[0].axhline(REF["CD_total_mean"], color="red", ls=":", lw=1.8, label=f"SI4 ref = {REF['CD_total_mean']:.3f}")
axes[0].axvspan(0, T_ONSET, alpha=0.07, color="orange", label="Transient")
axes[0].axvspan(t_win_start, t_win_end, alpha=0.15, color="blue", label=f"Analysis window ({n_cycles} cycles)")
axes[0].set_ylabel("Total C_D"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.25)

axes[1].plot(time, CD1, lw=0.5, label="CD1 (front)",  color="tab:blue")
axes[1].plot(time, CD2, lw=0.5, label="CD2 (top)",    color="tab:orange")
axes[1].plot(time, CD3, lw=0.5, label="CD3 (bottom)", color="tab:green")
axes[1].axhline(REF["CD1_mean"], color="gray", ls=":", lw=1.8, label=f"SI4 ref = {REF['CD1_mean']:.3f}")
axes[1].axvspan(0, T_ONSET, alpha=0.07, color="orange")
axes[1].axvspan(t_win_start, t_win_end, alpha=0.15, color="blue")
axes[1].set_ylabel("C_D / cylinder"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.25)

axes[2].plot(time, CL2, lw=0.5, label="CL2 (top)",    color="tab:orange")
axes[2].plot(time, CL3, lw=0.5, label="CL3 (bottom)", color="tab:green")
axes[2].plot(time, CL1, lw=0.4, label="CL1 (front)",  color="tab:blue", alpha=0.6)
axes[2].axvspan(0, T_ONSET, alpha=0.07, color="orange")
axes[2].axvspan(t_win_start, t_win_end, alpha=0.15, color="blue")
axes[2].set_ylabel("C_L"); axes[2].set_xlabel("Time (s)")
axes[2].legend(fontsize=8); axes[2].grid(alpha=0.25)

for ax in axes:
    ax.axvline(T_ONSET,     color="darkorange", lw=1.2, ls="--")
    ax.axvline(t_win_start, color="steelblue",  lw=1.2, ls="--")
    ax.axvline(t_win_end,   color="steelblue",  lw=1.2, ls="--")

plt.tight_layout()
plt.savefig(f"{OUT}/fig1_full_timeseries.png", dpi=150)
plt.close()
print(f"  Saved fig1_full_timeseries.png")

# FIGURE 2 -- Analysis window: timeseries + spectrum + phase portrait + bar
print("Plotting Fig 2 - cycle analysis detail...")

fig = plt.figure(figsize=(18, 12))
fig.suptitle(
    f"Fluidic Pinball Re=100 -- {n_cycles} Vortex Shedding Cycles  "
    f"[t = {t_win_start:.0f} - {t_win_end:.0f} s]",
    fontsize=13, fontweight="bold"
)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.32)

ax = fig.add_subplot(gs[0, 0])
ax.plot(t_w, Dw, "k", lw=1.1, label=f"CD_total  mean = {stats['CD_total_mean']:.4f}")
ax.axhline(stats["CD_total_mean"], color="k",   ls="--", lw=1, alpha=0.5)
ax.axhline(REF["CD_total_mean"],   color="red", ls=":",  lw=2, label=f"SI4 = {REF['CD_total_mean']:.3f}")
ax.set_title("Total Drag C_D"); ax.set_ylabel("C_D"); ax.legend(fontsize=8); ax.grid(alpha=0.25)

ax = fig.add_subplot(gs[1, 0])
ax.plot(t_w, CD1w, lw=1.1, color="tab:blue",   label=f"CD1  {stats['CD1_mean']:.4f} +/- {stats['CD1_std']:.5f}")
ax.plot(t_w, CD2w, lw=1.1, color="tab:orange", label=f"CD2  {stats['CD2_mean']:.4f} +/- {stats['CD2_std']:.5f}")
ax.plot(t_w, CD3w, lw=1.1, color="tab:green",  label=f"CD3  {stats['CD3_mean']:.4f} +/- {stats['CD3_std']:.5f}")
ax.axhline(REF["CD1_mean"], color="gray", ls=":", lw=2, label=f"SI4 = {REF['CD1_mean']:.3f}")
ax.set_title("Drag per Cylinder"); ax.set_ylabel("C_D"); ax.legend(fontsize=7.5); ax.grid(alpha=0.25)

ax = fig.add_subplot(gs[2, 0])
ax.plot(t_w, CL2w, lw=1.1, color="tab:orange",
        label=f"CL2  mean={stats['CL2_mean']:.4f}  amp={stats['CL2_amp']:.5f}")
ax.plot(t_w, CL3w, lw=1.1, color="tab:green",
        label=f"CL3  mean={stats['CL3_mean']:.4f}  amp={stats['CL3_amp']:.5f}")
ax.plot(t_w, CL1w, lw=0.8, color="tab:blue",
        label=f"CL1  mean={stats['CL1_mean']:.5f}", alpha=0.7)
ax.axhline( REF["CL2_amp"], color="darkorange", ls=":", lw=1.5,
            label=f"SI4 CL2 amp = +/-{REF['CL2_amp']:.3f}")
ax.axhline(-REF["CL3_amp"], color="darkgreen",  ls=":", lw=1.5)
ax.set_title("Lift Coefficients"); ax.set_ylabel("C_L"); ax.set_xlabel("Time (s)")
ax.legend(fontsize=7.5); ax.grid(alpha=0.25)

ax = fig.add_subplot(gs[0, 1])
fmask = (freq_cl2 > 0.02) & (freq_cl2 < 0.5)
ax.semilogy(freq_cl2[fmask], psd_cl2[fmask], color="tab:orange", lw=1.2, label="PSD  C_L2")
ax.axvline(St,       color="steelblue", ls="--", lw=2,
           label=f"Sim  f={St:.4f} Hz  St={St:.4f}")
ax.axvline(REF["St"], color="red", ls=":", lw=2,
           label=f"SI4  St={REF['St']:.4f}")
ax.set_title("Power Spectrum of C_L2"); ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD")
ax.legend(fontsize=8); ax.grid(alpha=0.25)

ax = fig.add_subplot(gs[1, 1])
ax.plot(CL2w, CL3w, color="tab:purple", lw=0.7, alpha=0.8)
ax.scatter(CL2w[0],  CL3w[0],  color="green", s=50, zorder=5, label="start")
ax.scatter(CL2w[-1], CL3w[-1], color="red",   s=50, zorder=5, label="end")
ax.set_title("Phase Portrait: C_L2 vs C_L3")
ax.set_xlabel("C_L2"); ax.set_ylabel("C_L3")
ax.legend(fontsize=8); ax.grid(alpha=0.25)

ax = fig.add_subplot(gs[2, 1])
labels = [r["label"] for r in cmp_rows]
sims   = [r["sim"]   for r in cmp_rows]
refs   = [r["ref"]   for r in cmp_rows]
errs   = [r["err_pct"] for r in cmp_rows]
x = np.arange(len(labels)); w = 0.36
ax.bar(x - w/2, sims, w, label="Simulated", color="steelblue", alpha=0.85)
ax.bar(x + w/2, refs, w, label="Table SI4", color="tomato",    alpha=0.75)
ax.set_title("Simulated vs Table SI4"); ax.set_ylabel("Value")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")
for i, e in enumerate(errs):
    ymax = max(sims[i], refs[i])
    ax.text(x[i], ymax + ymax * 0.03, f"{e:.0f}%",
            ha="center", fontsize=7, color="darkred", fontweight="bold")

plt.savefig(f"{OUT}/fig2_cycle_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved fig2_cycle_analysis.png")

# FIGURE 3 -- Zoomed last 4 cycles + amplitude growth + period evolution
print("Plotting Fig 3 - zoomed cycles + amplitude growth...")

fig, axes = plt.subplots(2, 2, figsize=(18, 10))
fig.suptitle("Fluidic Pinball Re=100 -- Zoomed Cycles & Amplitude Growth",
             fontsize=13, fontweight="bold")

n_zoom = 4
if len(t_peaks) >= n_zoom + 1:
    tz0 = t_peaks[-(n_zoom + 1)]
    tz1 = t_peaks[-1]
    mz  = (time >= tz0) & (time <= tz1)
    tz  = time[mz] - tz0

    ax = axes[0, 0]
    ax.plot(tz, CD1[mz], lw=1.4, color="tab:blue",
            label=f"CD1  mean={CD1[mz].mean():.4f}")
    ax.plot(tz, CD2[mz], lw=1.4, color="tab:orange",
            label=f"CD2  mean={CD2[mz].mean():.4f}")
    ax.plot(tz, CD3[mz], lw=1.4, color="tab:green",
            label=f"CD3  mean={CD3[mz].mean():.4f}")
    ax.axhline(REF["CD1_mean"], color="red", ls=":", lw=2,
               label=f"SI4 = {REF['CD1_mean']:.3f}")
    ax.set_title(f"Drag -- Last {n_zoom} Cycles (t={tz0:.0f}-{tz1:.0f}s)")
    ax.set_xlabel("Time within window (s)"); ax.set_ylabel("C_D")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ax.plot(tz, CL2[mz], lw=1.4, color="tab:orange",
            label=f"CL2  amp={float(np.ptp(CL2[mz])/2):.5f}")
    ax.plot(tz, CL3[mz], lw=1.4, color="tab:green",
            label=f"CL3  amp={float(np.ptp(CL3[mz])/2):.5f}")
    ax.axhline( REF["CL2_amp"],  color="darkorange", ls=":", lw=2,
               label=f"SI4 CL2 amp  {REF['CL2_amp']:.3f}")
    ax.axhline(-REF["CL3_amp"],  color="darkgreen",  ls=":", lw=2,
               label=f"SI4 CL3 amp -{REF['CL3_amp']:.3f}")
    ax.axhline(CL2[mz].mean(), color="tab:orange", ls="--", lw=1, alpha=0.5)
    ax.axhline(CL3[mz].mean(), color="tab:green",  ls="--", lw=1, alpha=0.5)
    ax.set_title(f"Lift -- Last {n_zoom} Cycles")
    ax.set_xlabel("Time within window (s)"); ax.set_ylabel("C_L")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

# Amplitude growth curve
m_all  = time >= T_ONSET
pk_all, _ = find_peaks(CL2[m_all], prominence=0.005, distance=int(5/dt))
tr_all, _ = find_peaks(-CL2[m_all], prominence=0.002, distance=int(5/dt))
n_pairs   = min(len(pk_all), len(tr_all))
amp_growth = (CL2[m_all][pk_all[:n_pairs]] - CL2[m_all][tr_all[:n_pairs]]) / 2
t_amp      = time[m_all][pk_all[:n_pairs]]

ax = axes[1, 0]
ax.plot(t_amp, amp_growth, "o-", color="tab:orange", lw=1.5, ms=5,
        label="CL2 amplitude (half p-p)")
ax.axhline(REF["CL2_amp"], color="red", ls=":", lw=2,
           label=f"SI4 target = {REF['CL2_amp']:.3f}")
ax.axvline(t_win_start, color="steelblue", ls="--", lw=1.5, label="Analysis window")
ax.axvline(t_win_end,   color="steelblue", ls="--", lw=1.5)
ax.set_title("C_L2 Amplitude Evolution")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude (half p-p)")
ax.legend(fontsize=8); ax.grid(alpha=0.25)

# Period evolution
ax = axes[1, 1]
ax.plot(t_peaks[:-1], periods, "s-", color="tab:purple", lw=1.5, ms=5,
        label="Shedding period T (s)")
ax.axhline(T_shed, color="tab:purple", ls="--", lw=1.5, alpha=0.5,
           label=f"Median T = {T_shed:.3f} s")
ax.axhline(1/REF["St"], color="red", ls=":", lw=2,
           label=f"SI4 T = {1/REF['St']:.3f} s")
ax.axvline(t_win_start, color="steelblue", ls="--", lw=1.5)
ax.axvline(t_win_end,   color="steelblue", ls="--", lw=1.5)
ax.set_title("Shedding Period Evolution")
ax.set_xlabel("Time of peak (s)"); ax.set_ylabel("Period T (s)")
ax.legend(fontsize=8); ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig(f"{OUT}/fig3_amplitude_growth.png", dpi=150)
plt.close()
print(f"  Saved fig3_amplitude_growth.png")

# Clean summary printout
print("\n" + "="*72)
print("  VALIDATION SUMMARY vs Table SI4  (Re=100, Uncontrolled)")
print("="*72)
print(f"  Cycles analysed   : {n_cycles}  (t = {t_win_start:.0f} - {t_win_end:.0f} s)")
print(f"  Shedding period T : {T_shed:.3f} +/- {np.std(periods):.3f} s")
print(f"  Shedding freq f   : {f_shed:.4f} Hz")
print(f"  St (peaks)        : {St:.4f}    |  SI4: {REF['St']:.4f}")
print(f"  St (spectral)     : {St_spec:.4f}")
print()
print(f"  {'Quantity':<24}  {'Simulated':>10}  {'SI4 Ref':>10}  {'Error':>8}  Status")
print("  " + "-"*64)
for r in cmp_rows:
    status = "PASS" if r["err_pct"] < 5 else ("~OK" if r["err_pct"] < 20 else "FAIL")
    print(f"  {r['label']:<24}  {r['sim']:>10.4f}  {r['ref']:>10.4f}  {r['err_pct']:>7.1f}%  {status}")

print(f"\nAll outputs in: ./{OUT}/")
for fn in sorted(os.listdir(OUT)):
    print(f"  {fn}")