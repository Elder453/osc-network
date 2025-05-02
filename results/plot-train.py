#!/usr/bin/env python3
# ---------------------------------------------------------------
#  Average batch‑level training accuracy for all model variants
# ---------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import glob
import wandb                       # optional –– only used with --wandb
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({
    'axes.labelsize': 14,     # axes labels (xlabel, ylabel)
    'axes.titlesize': 16,     # titles
    'xtick.labelsize': 12,    # tick labels
    'ytick.labelsize': 12, 
    'legend.fontsize': 12,    # legend text size
    'figure.titlesize': 17,   # figure title size
})

# ----------------------- helpers --------------------------------
def _load_padded(model_type, run_dirs, task):
    """stack curves from several runs → (n_runs, max_len) with NaN‑padding"""
    # Process directories based on model type
    if model_type == "baseline":
        # Baseline uses the old naming convention
        fname = f"baseline_{task}_train_batch_acc.npy"
        # BTFF directories
        model_dirs = [d for d in run_dirs if d.name.startswith("BTFF")]
    else:
        # For oscillator-based models, use the new naming convention
        if model_type == "osc-default":  # KTFF - default osc-based
            prefix = "kuramoto_JoutFalse_OmegaTrue_JsymmFalse"
            # KTFF directories
            model_dirs = [d for d in run_dirs if d.name.startswith("KTFF")]
        elif model_type == "osc-no-omega":  # KFFF - no Ω
            prefix = "kuramoto_JoutFalse_OmegaFalse_JsymmFalse"
            # KFFF directories
            model_dirs = [d for d in run_dirs if d.name.startswith("KFFF")]
        elif model_type == "osc-no-jout":  # KTTF - no J-out
            prefix = "kuramoto_JoutTrue_OmegaTrue_JsymmFalse"
            # KTTF directories
            model_dirs = [d for d in run_dirs if d.name.startswith("KTTF")]
        elif model_type == "osc-symm-j":  # KTFT - symmetric J
            prefix = "kuramoto_JoutFalse_OmegaTrue_JsymmTrue"
            # KTFT directories
            model_dirs = [d for d in run_dirs if d.name.startswith("KTFT")]
        elif model_type == "osc-no-omega-symm-j":
            prefix = "kuramoto_JoutFalse_OmegaFalse_JsymmTrue"  # KFTF
            model_dirs = [d for d in run_dirs if d.name.startswith("KFFT")]
        
        fname = f"{prefix}_{task}_train_batch_acc.npy"
    
    print(f"Processing {model_type} ({len(model_dirs)} seed directories)")
    
    # Try to load all files
    curves = []
    for d in model_dirs:
        try:
            data = np.load(d / fname)
            curves.append(data)
            print(f"  Loaded {d.name}/{fname}, shape: {data.shape}")
        except Exception as e:
            print(f"  Error loading {fname} from {d.name}: {e}")
    
    # If no curves were found, return empty array
    if not curves:
        print(f"Warning: No data found for {model_type} on {task}")
        return np.array([[]])
        
    max_len = max(len(c) for c in curves)
    
    # Pad all curves to the same length
    padded = [np.pad(c, (0, max_len - len(c)), constant_values=np.nan) for c in curves]
    return np.vstack(padded)

def _plot_mean(ax, arr, label, color):
    if arr.size == 0:
        print(f"Warning: No data to plot for {label}")
        return
    mean, std = np.nanmean(arr, 0), np.nanstd(arr, 0)
    x = np.arange(len(mean))
    ax.plot(x, mean, lw=2.0, label=label, color=color, alpha=0.8)  # More transparent line
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.25)

# ------------------------- main ---------------------------------
def main(base_dir, glyph_number, out_prefix, log_wandb=False):
    """
    Main function to process data and create plots.
    
    Args:
        base_dir: Base directory containing run folders
        glyph_number: Glyph number to filter directories (e.g., "95")
        out_prefix: Prefix for output files
        log_wandb: Whether to log results to W&B
    """
    base_path = Path(base_dir)
    
    # Find all matching directories
    run_dirs = []
    
    # Define the prefixes we're looking for
    prefixes = ["BTFF", "KTFF", "KFFF", "KTTF", "KTFT", "KFFT"]
    
    for prefix in prefixes:
        # Look for directories matching pattern prefix_g{glyph_number}_s*
        pattern = f"{prefix}_g{glyph_number}_s*"
        matching_dirs = list(base_path.glob(pattern))
        
        if matching_dirs:
            print(f"Found {len(matching_dirs)} directories for {prefix}: {[d.name for d in matching_dirs]}")
            run_dirs.extend(matching_dirs)
        else:
            print(f"Warning: No directories found matching {base_path}/{pattern}")
    
    if not run_dirs:
        print(f"Error: No run directories found in {base_dir}")
        return

    # Define model types and their labels (using Unicode symbols)
    model_types = [
        ("baseline", "baseline"),
        ("osc-default", "osc-based"),
        ("osc-no-omega", r"osc-based (no $\mathbf{\Omega}$)"),
        ("osc-no-jout", r"osc-based (no $\mathbf{J}^{\mathrm{OUT}}$)"),
        ("osc-symm-j", r"osc-based (symm. $\mathbf{J}$)"),
        ("osc-no-omega-symm-j", r"osc-based (no $\mathbf{\Omega}$, symm. $\mathbf{J}$)")
    ]
    
    # Define tasks
    tasks = [
        ("sd", "sd"),
        ("rmts", "rmts")
    ]

    # Define colors matching the other script
    colors = {
        "baseline": "#e49797",      # baseline
        "osc-default": "#67c2c7",   # osc-based
        "osc-no-omega": "#77cfa1",  # osc-based (no Ω)
        "osc-no-jout": "#b39ddb",   # osc-based (no J-out)
        "osc-symm-j": "#f3c97e",    # osc-based (symmetric J)
        "osc-no-omega-symm-j": "#f3c97e"  # osc-based (no Ω, symmetric J)
    }
    
    outs = []

    # Create one plot for each task (SD and RMTS)
    for task_id, task_name in tasks:
        fig, ax = plt.subplots(figsize=(5, 5))
        
        # Load and plot data for each model type
        for model_id, model_label in model_types:
            try:
                data = _load_padded(model_id, run_dirs, task_id)
                if data.size > 1:  # Only plot if we have data
                    _plot_mean(ax, data, model_label, colors[model_id])
            except Exception as e:
                print(f"Error processing {model_id} for {task_id}: {e}")
        
        # Add appropriate title based on task
        # if task_id == "sd":
        #     ax.set_title("Same/Different")
        # else:  # rmts
        #     ax.set_title("Relational Match-to-Sample")

        ax.set_xlabel("Batch Number")
        ax.set_ylabel("Training Accuracy")
        ax.set_ylim(0.3, 1)
        # Set y-ticks every 0.1
        ax.set_yticks(np.arange(0.3, 1.1, 0.1))
        
        ax.legend(loc="lower right")  # Horizontal legend
        fig.tight_layout()

        out_png = f"figs/{task_name}_{out_prefix}_g{glyph_number}.png"
        fig.savefig(out_png, dpi=600)
        outs.append(out_png)
        plt.close(fig)

    # ---------- optionally log to W&B ----------------------------
    if log_wandb:
        wandb.init(project="kuramoto-relational-model", job_type="analysis")
        for p in outs:
            wandb.log({f"avg_curve/{Path(p).stem}": wandb.Image(p)})
        wandb.finish()

    print("saved:")
    for p in outs:
        print("  •", p)

# ------------------------- CLI ----------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Average batch‑level training accuracy over seeds "
                    "and plot curves for all model variants.")
    parser.add_argument("base_dir", 
                        help="Base directory containing run folders (e.g., 'runs')")
    parser.add_argument("--glyph", default="95",
                        help="Glyph number to process (default: 95)")
    parser.add_argument("--out_prefix", default="train_acc",
                        help="prefix for the output PNG files")
    parser.add_argument("--wandb", action="store_true",
                        help="log the figures to the current W&B run")
    
    args = parser.parse_args()
    
    main(args.base_dir, args.glyph, args.out_prefix, log_wandb=args.wandb)