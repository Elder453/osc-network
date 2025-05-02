#!/usr/bin/env python3
# ---------------------------------------------------------------
#  Bar‑plots ▸ comparing model variants (mean ± sd over 3 seeds)
# ---------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from io import StringIO
from pathlib import Path
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({
    'axes.labelsize': 14,     # axes labels (xlabel, ylabel)
    'axes.titlesize': 16,     # titles
    'xtick.labelsize': 12,    # tick labels
    'ytick.labelsize': 12, 
    'legend.fontsize': 9,    # legend text size
    'figure.titlesize': 17,   # figure title size
})

# ---------- 1. read data from CSV file --------------------------
# Assuming the CSV is created by the bash script
df = pd.read_csv("experiment_results.csv")

# Map model variants based on exp_name prefix
def map_model_type(row):
    prefix = row['exp_name_prefix']
    if prefix.startswith('B'):
        return "baseline"
    elif prefix == 'KTFF':
        return "osc-based"
    elif prefix == 'KFFF':
        return r"osc-based (no $\mathbf{\Omega}$)"
    elif prefix == 'KTTF':
        return r"osc-based (no $\mathbf{J}^{\mathrm{OUT}}$)"
    elif prefix == 'KTFT':
        return r"osc-based (symm. $\mathbf{J}$)"
    elif prefix == 'KFFT':
        return r"osc-based (no $\mathbf{\Omega}$, symm. $\mathbf{J}$)"
    else:
        return "unknown"

df['model'] = df.apply(map_model_type, axis=1)
df['m'] = df['n_train_glyphs']
df['sd_test'] = df['SD_test_acc']
df['rmst_test'] = df['RMTS_test_acc']

# ---------- 2. aggregate ---------------------------------
agg = (
    df.groupby(["m", "model"])
      .agg(sd_mean=("sd_test", "mean"),
           sd_std =("sd_test", "std"),
           rmts_mean=("rmst_test", "mean"),
           rmts_std =("rmst_test", "std"))
      .reset_index()
)

# ---------- 3. plotting helper ----------------------------------
def _bar_plot(metric, fname):
    """
    metric : 'sd' | 'rmts'
    fname  : output png filename
    """
    # Extended color palette
    colors = {
        "baseline": "#e49797",                  # softer reddish pink
        "osc-based": "#67c2c7",                 # softer teal/cyan
        r"osc-based (no $\mathbf{\Omega}$)": "#77cfa1",  # lighter green
        r"osc-based (no $\mathbf{J}^{\mathrm{OUT}}$)": "#b39ddb", # softer purple
        r"osc-based (symm. $\mathbf{J}$)": "#f3c97e",    # lighter gold
        r"osc-based (no $\mathbf{\Omega}$, symm. $\mathbf{J}$)": "#f3c97e" # same lighter gold
    }
    
    fig, ax = plt.subplots(figsize=(10, 5))

    x_vals = sorted(agg["m"].unique(), reverse=True)  # 95, 50, 15
    x_pos = np.arange(len(x_vals))
    m_to_pos = {m: i for i, m in enumerate(x_vals)}  # Map m values to positions
    
    models = ["baseline", 
              "osc-based", 
              r"osc-based (no $\mathbf{\Omega}$)", 
              r"osc-based (no $\mathbf{J}^{\mathrm{OUT}}$)", 
              r"osc-based (symm. $\mathbf{J}$)",
              r"osc-based (no $\mathbf{\Omega}$, symm. $\mathbf{J}$)"
             ]
    models = [m for m in models if m in agg["model"].unique()]  # only include models that exist in the data
    
    n_models = len(models)
    width = 0.7 / n_models  # adjust width based on number of models
    
    for i, model in enumerate(models):
        if model not in agg["model"].unique():
            continue
            
        sub = agg[agg["model"] == model]
        
        # For each m value, plot at the correct position
        for _, row in sub.iterrows():
            m_val = row['m']
            pos = m_to_pos[m_val]
            mean = row[f"{metric}_mean"]
            std = row[f"{metric}_std"]
            
            offset = pos + (i - n_models/2 + 0.5) * width
            ax.bar(offset, mean, width=width,
                  yerr=std, capsize=4,
                  color=colors[model], label=model if m_val == x_vals[0] else "",
                  edgecolor="black", linewidth=0.6)

    # ---- cosmetics -------------------------------------------------
    # Add appropriate title based on metric
    if metric == "sd":
        ax.set_title("Same/Different")
    else:  # rmts
        ax.set_title("Relational Match-to-Sample")
        
    ax.set_xlabel("Number of Icons Shown During Training")
    ax.set_ylabel("Test Accuracy")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_vals)  # This ensures labels match positions
    ax.set_ylim(0.65, 1)
    ax.set_yticks(np.arange(0.65, 1.04, 0.05))  # Add y-ticks every 0.1
    ax.legend(loc="upper right")  # Horizontal legend
    ax.tick_params(axis='both', which='both', direction='out')
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(fname, dpi=600)
    plt.close(fig)
    print(f"saved {fname}")

# ---------- 4. make & save both plots ----------------------------
Path("figs").mkdir(exist_ok=True)

_plot_history_cfg = [
    ("sd",   "figs/sd_test_accuracy.png"),
    ("rmts", "figs/rmts_test_accuracy.png"),
]
for metric, fname in _plot_history_cfg:
    _bar_plot(metric, fname)