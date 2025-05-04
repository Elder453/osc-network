"""
Visualization utilities.

Provides functions for visualizing oscillator dynamics, energy curves,
and model outputs.
"""

import os
import math
import tempfile
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import wandb


def _wandb_log_animation(anim, tag, fps=2):
    """
    Save a matplotlib FuncAnimation to GIF and log it to Weights & Biases.

    Parameters
    ----------
    anim : matplotlib.animation.FuncAnimation
        Animation object to save and log
    tag : str
        Key prefix in wandb (e.g. "epoch10_osc_same")
    fps : int
        Frames-per-second for the GIF
    """
    tmpdir = Path(tempfile.mkdtemp())
    gif_path = tmpdir / f"{tag}.gif"
    gif_path.parent.mkdir(parents=True, exist_ok=True)

    anim.save(gif_path, writer="pillow", fps=fps)
    wandb.log({
        f"{tag}_gif": wandb.Video(str(gif_path), format="gif"),
    })


def _best_grid(n: int) -> tuple[int, int]:
    """
    Return (rows, cols) giving the most square-ish grid ≥ n.
    Used to lay out per-feature sub-plots when N=2.          
    
    Parameters
    ----------
    n : int
        Number of items to arrange in a grid
        
    Returns
    -------
    tuple
        (rows, cols) dimensions for the grid
    """
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def collect_energy_values(model, sample_batch, device):
    """
    Collects energy values with separate tracking for same/different pairs.
    
    Parameters
    ----------
    model : nn.Module
        The model to evaluate
    sample_batch : tuple
        Batch of samples containing ((img1, img2), labels)
    device : torch.device
        Device to run computation on
        
    Returns
    -------
    dict
        Formatted energy values for visualization
    """
    model.eval()
    
    with torch.no_grad():
        # unpack batch
        (img1, img2), labels = sample_batch
        img1, img2 = img1.to(device), img2.to(device)
        labels = labels.to(device)
        
        # forward pass to get energy values
        _, oscillator_data = model(img1, img2)
        energy_values = oscillator_data[4]  # extract energy values
        
        formatted_energy = {
            # overall average energy across all samples
            "all": {
                step: {
                    "mean": energy.mean().item(),
                    "std": energy.std().item(),
                    "n": energy.size(0)  # sample size for CIs
                } 
                for step, energy in enumerate(energy_values)
            }
        }
        
        # separate energy for same/different pairs
        same_mask = (labels == 1)
        diff_mask = (labels == 0)
        
        if same_mask.any():
            n_same = same_mask.sum().item()
            formatted_energy["same"] = {
                step: {
                    "mean": energy[same_mask].mean().item(),
                    "std": energy[same_mask].std().item() if n_same > 1 else 0.0,
                    "n": n_same
                }
                for step, energy in enumerate(energy_values)
            }
            
        if diff_mask.any():
            n_diff = diff_mask.sum().item()
            formatted_energy["different"] = {
                step: {
                    "mean": energy[diff_mask].mean().item(),
                    "std": energy[diff_mask].std().item() if n_diff > 1 else 0.0,
                    "n": n_diff
                }
                for step, energy in enumerate(energy_values)
            }
    
    return formatted_energy


def plot_energy_one_epoch(energy_values, epoch, ax=None, confidence_level=0.95):
    """
    Plot energy evolution for one epoch with confidence intervals.
    
    Parameters
    ----------
    energy_values : dict
        Outer keys: "all", "same", "different".
        Inner dict: {step: {"mean": float, "std": float, "n": int}}.
    epoch : int
        Zero-based epoch index (will be printed 1-based)
    ax : matplotlib.axes.Axes, optional
        If given, plot there; otherwise create a new figure/axes
    confidence_level : float
        Confidence level for the intervals (e.g., 0.95 for 95% CI)
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot
    """
    own_ax = ax is None
    if own_ax:
        fig, ax = plt.subplots(figsize=(10, 5))
    
    SALMON = "#e49797"
    TEAL = "#67c2c7"
    BLACK = "#000000"
    
    colors = {
        "all": BLACK,
        "same": TEAL,
        "different": SALMON
    }
    
    # z-score for confidence level (e.g., 1.96 for 95% CI)
    import scipy.stats as stats
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    for cat, data in energy_values.items():
        steps = sorted(data.keys())
        means = [data[step]["mean"] for step in steps]
        
        # calculate standard errors
        stds = [data[step]["std"] for step in steps]
        ns = [data[step]["n"] for step in steps]
        se = [std / (n ** 0.5) for std, n in zip(stds, ns)]
        
        # calculate CI bounds
        ci_lower = [mean - z_score * se_val for mean, se_val in zip(means, se)]
        ci_upper = [mean + z_score * se_val for mean, se_val in zip(means, se)]
        
        ax.plot(
            steps,
            means,
            label=cat.lower() + " pairs",
            color=colors.get(cat, "gray"),
            lw=2.0,
        )
        
        # add shaded CI
        ax.fill_between(steps, ci_lower, ci_upper, 
                        color=colors.get(cat, "gray"), alpha=0.2)
    
    ax.set_xlabel("kuramoto step")
    ax.set_ylabel("energy")

    label = "init" if epoch < 0 else f"epoch {epoch+1}"
    ax.set_title(f"{label}")
    
    ax.legend(loc="lower left", frameon=True, edgecolor='lightgray', framealpha=0.8)
    
    ax.grid(False)
    
    if own_ax:
        plt.tight_layout()
        plt.close()
        return fig
    return None


def build_energy_animation(all_epoch_energies,
                           categories=("all", "same", "different"),
                           fname="energy_evolution.gif",
                           fps=2):
    """
    Create and save a GIF that re-draws the energy curve for every epoch.
    
    Parameters
    ----------
    all_epoch_energies : dict
        Epoch index → { category → { step → {"mean": float, "std": float, "n": int} } }
        If the dict contains key -1, that frame is treated as the initial un-trained network.
    categories : tuple
        Which energy traces to draw
    fname : str
        Output filename for the GIF
    fps : int
        Frames per second for the animation
        
    Returns
    -------
    str
        Path to the saved GIF file
    """
    epochs = sorted(all_epoch_energies.keys())
    steps = list(next(iter(all_epoch_energies.values()))["all"].keys())

    plt.style.use("seaborn-v0_8-paper")
    fig, ax = plt.subplots(figsize=(20, 5))

    COLORS = {"all": "#000000", "same": "#67c2c7", "different": "#e49797"}
    lines = {}
    for cat in categories:
        (ln,) = ax.plot([], [], lw=2.0,
                        color=COLORS.get(cat, "gray"),
                        label=f"{cat} pairs")
        lines[cat] = ln

    # fixed axes limits
    y_vals = []
    for ep in epochs:
        for cat in categories:
            if cat in all_epoch_energies[ep]:
                y_vals.extend([all_epoch_energies[ep][cat][step]["mean"] for step in all_epoch_energies[ep][cat]])
    ax.set_xlim(min(steps), max(steps))
    ax.set_ylim(min(y_vals) * 1.05, max(y_vals) * 1.05)

    ax.set_xlabel("Kuramoto Step")
    ax.set_ylabel("Energy")
    ax.grid(alpha=.3)
    ax.legend(loc="lower left", frameon=True,
              edgecolor="lightgray", framealpha=0.8)
    ax.grid(False)

    # animation callbacks
    def init():
        for ln in lines.values():
            ln.set_data([], [])
        return tuple(lines.values())

    def update(frame_idx):
        epoch = epochs[frame_idx]
        label = "initialization" if epoch < 0 else f"epoch {epoch+1}"
        ax.set_title(f"{label}")

        for cat, ln in lines.items():
            if cat in all_epoch_energies[epoch]:
                ys = [all_epoch_energies[epoch][cat][step]["mean"] for step in steps]
                ln.set_data(steps, ys)
            else:                     # that category is absent this epoch
                ln.set_data([], [])
        return tuple(lines.values())

    # make and save GIF
    ani = FuncAnimation(fig, update, frames=len(epochs),
                        init_func=init, blit=False,
                        interval=1000 / fps)

    Path(fname).parent.mkdir(parents=True, exist_ok=True)
    ani.save(fname, writer="pillow", fps=fps)
    plt.close(fig)

    return fname


def plot_energy_init_vs_trained(all_epoch_energies, 
                               categories=("all", "same", "different"), 
                               init_epoch=-1, 
                               final_epoch=49, 
                               confidence_level=0.95):
    """
    Plot energy curves at initialization and after training with confidence intervals.
    
    Parameters
    ----------
    all_epoch_energies : dict
        Nested dictionary with energy information for each epoch
    categories : tuple
        Which categories to plot (e.g., "all", "same", "different")
    init_epoch : int
        Epoch index for initialization (typically -1)
    final_epoch : int
        Epoch index for final trained state
    confidence_level : float
        Confidence level for the intervals (e.g., 0.95 for 95% CI)
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot
    """
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({
        'axes.labelsize': 14,     # axes labels (xlabel, ylabel)
        'axes.titlesize': 16,     # titles
        'xtick.labelsize': 12,    # tick labels
        'ytick.labelsize': 12, 
        'legend.fontsize': 7,     # legend text size
        'figure.titlesize': 17,   # figure title size
    })
     
    steps = list(all_epoch_energies[init_epoch]["all"].keys())
    
    COLORS = {"all": "#000000", "same": "#67c2c7", "different": "#e49797"}
    
    # z-score for given confidence level
    import scipy.stats as stats
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    fig, ax = plt.subplots(figsize=(10, 4), dpi=600)
    
    for cat in categories:
        if cat in all_epoch_energies[init_epoch]:
            # extract values for initialization
            init_means = [all_epoch_energies[init_epoch][cat][step]["mean"] for step in steps]
            
            # calculate standard errors
            init_stds = [all_epoch_energies[init_epoch][cat][step]["std"] for step in steps]
            init_ns = [all_epoch_energies[init_epoch][cat][step]["n"] for step in steps]
            init_se = [std / (n ** 0.5) for std, n in zip(init_stds, init_ns)]
            
            # calculate CI bounds
            init_ci_lower = [mean - z_score * se for mean, se in zip(init_means, init_se)]
            init_ci_upper = [mean + z_score * se for mean, se in zip(init_means, init_se)]
            
            ax.plot(steps,
                    init_means,
                    linestyle='--', 
                    linewidth=2.0,
                    alpha=0.7,
                    color=COLORS.get(cat, 'gray'),
                    label=f'"{cat}" pairs (init)' if cat != "all" else f'{cat} pairs (init)')
            
            # add shaded CI
            ax.fill_between(steps, init_ci_lower, init_ci_upper, 
                           color=COLORS.get(cat, 'gray'), alpha=0.15)
        
        if cat in all_epoch_energies[final_epoch]:
            # extract values for final epoch
            final_means = [all_epoch_energies[final_epoch][cat][step]["mean"] for step in steps]
            
            # calculate standard errors for CIs
            final_stds = [all_epoch_energies[final_epoch][cat][step]["std"] for step in steps]
            final_ns = [all_epoch_energies[final_epoch][cat][step]["n"] for step in steps]
            final_se = [std / (n ** 0.5) for std, n in zip(final_stds, final_ns)]
            
            # calculate CI bounds
            final_ci_lower = [mean - z_score * se for mean, se in zip(final_means, final_se)]
            final_ci_upper = [mean + z_score * se for mean, se in zip(final_means, final_se)]
            
            ax.plot(steps, 
                    final_means, 
                    linestyle='-', 
                    linewidth=2.0,
                    color=COLORS.get(cat, 'gray'),
                    label=f'"{cat}" pairs (epoch {final_epoch+1})' if cat != "all" else f'{cat} pairs (epoch {final_epoch+1})')
            
            # add shaded CI
            ax.fill_between(steps, final_ci_lower, final_ci_upper, 
                           color=COLORS.get(cat, 'gray'), alpha=0.2)
    
    ax.set_xlabel("Kuramoto Step")
    ax.set_ylabel("Energy")
    ax.grid(False)
    ax.legend(frameon=True, edgecolor='lightgray', framealpha=0.8)
    
    plt.tight_layout()
    plt.close()
    return fig


def visualize_oscillators_2d(
    trajectory1,
    trajectory2,
    c_terms1,
    c_terms2,
    max_features: int = 16,
    trail_length: int = 5,
    animate: bool = True,
    marker_interval: int = 1,
):
    """
    2-D visualizer for oscillator trajectories in the complex plane.
    
    Parameters
    ----------
    trajectory1 : list
        First trajectory of oscillators
    trajectory2 : list
        Second trajectory of oscillators
    c_terms1 : torch.Tensor
        Coupling terms for first trajectory
    c_terms2 : torch.Tensor
        Coupling terms for second trajectory
    max_features : int
        Maximum number of features to display
    trail_length : int
        Length of the trajectory trail to display
    animate : bool
        Whether to create an animation or just the final frame
    marker_interval : int
        Place a marker every N steps to show step size changes
        
    Returns
    -------
    tuple
        (Animation, Figure) - Animation is None if animate=False
    """
    plt.style.use("seaborn-v0_8-paper")
    
    TEAL = "#67c2c7"  
    SALMON = "#e49797" 
    BLACK = "#000000"
    
    # scale factor - reduce unit sphere size
    SCALE_FACTOR = 0.8
    
    num_steps = len(trajectory1)
    D = trajectory1[0].shape[1]
    features = min(D, max_features)
    rows, cols = _best_grid(features)
    
    # shape = [steps, D, 2]
    traj1_np = np.array([t[0].cpu().numpy()[:, :2] for t in trajectory1])
    traj2_np = np.array([t[0].cpu().numpy()[:, :2] for t in trajectory2])

    # scale trajectories by the scale factor
    traj1_np = traj1_np * SCALE_FACTOR
    traj2_np = traj2_np * SCALE_FACTOR
    
    coh_np = 0.5 * (traj1_np + traj2_np) # [steps, D, 2]
    
    # additional radius scaling to separate overlapping trajectories
    traj1_np_scaled = traj1_np * 0.97  # inner orbit
    traj2_np_scaled = traj2_np * 1.03  # outer orbit
    
    # scale the c-terms by the scale factor
    c1_np = c_terms1[0].cpu().numpy()[:, :2] * SCALE_FACTOR
    c2_np = c_terms2[0].cpu().numpy()[:, :2] * SCALE_FACTOR
    
    # create figure with proper spacing
    fig = plt.figure(figsize=(3*cols, 3*rows + 0.8), dpi=200)  # add extra height for title/legend
    
    gs = fig.add_gridspec(
        rows, cols,
        top=0.85,    # pull title up
        bottom=0.04, # push subplots down
        left=0.04,   # tighten left
        right=0.96,  # tighten right
        wspace=0.03,
        hspace=0.03
    )
    
    # create axes from gridspec
    axes = []
    for i in range(rows):
        for j in range(cols):
            axes.append(fig.add_subplot(gs[i, j]))
    
    axes = np.array(axes)
    
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color=TEAL, label=r'$osc_{1}$',
                  markerfacecolor=TEAL, markersize=20, linewidth=0),
        plt.Line2D([0], [0], marker='o', color=SALMON, label=r'$osc_{2}$',
                  markerfacecolor=SALMON, markersize=20, linewidth=0),
        
        # c-term directions (hollow triangles)
        plt.Line2D([0], [0], marker='^', color=TEAL, label=r'$c_{1}$',
                   markerfacecolor=TEAL, markersize=20, linewidth=0),
        plt.Line2D([0], [0], marker='^', color=SALMON,  label=r'$c_{2}$',
                   markerfacecolor=SALMON, markersize=20, linewidth=0),

        # step marker legend item
        plt.Line2D([0], [0], marker='.', color='#444444', markersize=12, 
                  linewidth=0, label='step marker'),

        # coherence vector
        plt.Line2D([0], [0], marker='^', color=BLACK, label=r'$\rho_{d}$',
                   markerfacecolor=BLACK, markersize=20, linewidth=0),
    ]
    fig.legend(handles=legend_elements,
               loc="upper center",
               bbox_to_anchor=(0.5, 0.98),
               ncol=3,                      # 3 columns in 2 rows for 6 items
               frameon=True,
               edgecolor="lightgray",
               framealpha=0.85,
               fontsize=18,
               borderaxespad=0.2)
    
    # title for the entire figure with time step
    time_title = fig.suptitle(f"t = 0", fontsize=28, y=0.89)
    
    for a in axes[features:]:
        a.set_visible(False)
    
    def _init():
        for i in range(features):
            a = axes[i]
            a.clear()
            a.set_aspect("equal", "box")
            axis_limit = SCALE_FACTOR * 1.2  # add 20% padding around scaled unit circle
            a.set_xlim([-axis_limit, axis_limit]) 
            a.set_ylim([-axis_limit, axis_limit])
            
            # remove all borders, axes and ticks
            a.axis('off')
            
            a.plot([], [])
        return axes
    
    def _update(frame):
        # update the global time title (lowercase)
        time_title.set_text(f"t = {frame}")
        
        # in animation mode, use a sliding window; in still mode, show full trajectory
        if animate:
            start = max(0, frame - trail_length + 1)
        else:
            # for final frame still, show the entire trajectory
            start = 0
            
        for i in range(features):
            a = axes[i]
            a.clear()
            a.set_aspect("equal", "box")
            axis_limit = SCALE_FACTOR * 1.1  # add 20% padding around scaled unit circle
            a.set_xlim([-axis_limit, axis_limit]) 
            a.set_ylim([-axis_limit, axis_limit])
            
            # remove all borders, axes and ticks
            a.axis('off')
            
            # draw unit circle
            circ = plt.Circle((0, 0), SCALE_FACTOR, 
                              edgecolor="#555555", 
                              facecolor="none", 
                              alpha=0.8, 
                              linewidth=1.5
            )
            a.add_patch(circ)
            
            # coherence vector
            coh_vec = coh_np[frame, i]  # (x,y) for that feature
            a.quiver(0, 0,
                     coh_vec[0], coh_vec[1],
                     color=BLACK,      
                     angles="xy", scale_units="xy", scale=1,
                     width=0.012, alpha=0.6, zorder=4)
            
            # c-term arrows
            a.quiver(0, 0, 
                    c1_np[i, 0], c1_np[i, 1], 
                    color=TEAL, angles="xy", scale_units="xy", scale=1,
                    width=0.015, alpha=0.9)
            a.quiver(0, 0, 
                    c2_np[i, 0], c2_np[i, 1], 
                    color=SALMON, angles="xy", scale_units="xy", scale=1,
                    width=0.015, alpha=0.9)
            
            # trajectory trails
            if frame > start:
                # draw continuous trails
                a.plot(
                    traj1_np_scaled[start:frame+1, i, 0],
                    traj1_np_scaled[start:frame+1, i, 1], 
                    "-", color=TEAL, 
                    alpha=0.9, linewidth=3.0
                )
                a.plot(
                    traj2_np_scaled[start:frame+1, i, 0],
                    traj2_np_scaled[start:frame+1, i, 1], 
                    "-", color=SALMON, 
                    alpha=0.9, linewidth=3.0
                )
                
                # add step markers along the trajectories
                marker_indices = range(start, frame+1, marker_interval)
                
                # draw step markers
                a.scatter(
                    traj1_np_scaled[marker_indices, i, 0],
                    traj1_np_scaled[marker_indices, i, 1],
                    color="#5aa3a7",
                    s=10,  # smaller than endpoint markers
                    zorder=6,
                    alpha=0.75
                )
                a.scatter(
                    traj2_np_scaled[marker_indices, i, 0],
                    traj2_np_scaled[marker_indices, i, 1],
                    color="#c97f7f",
                    s=10,  # smaller than endpoint markers
                    zorder=6,
                    alpha=0.75
                )
            
            # current positions with larger markers
            a.scatter(traj1_np_scaled[frame, i, 0], traj1_np_scaled[frame, i, 1], 
                     c=TEAL, marker="o", s=100,
                     edgecolor="white", linewidth=0.8, zorder=5, alpha=0.9)
            a.scatter(traj2_np_scaled[frame, i, 0], traj2_np_scaled[frame, i, 1], 
                     c=SALMON, marker="o", s=100,
                     edgecolor="white", linewidth=0.8, zorder=5, alpha=0.9)
        
        return axes

    ani = None
    if animate:  # use the parameter directly
        ani = FuncAnimation(fig, _update,
                          frames=num_steps,
                          init_func=_init,
                          interval=400,
                          blit=False)
    else:
        _update(num_steps-1)
        
    return ani, fig


def visualize_image_pair(img1, img2, prob, label=None):
    """
    Visualize a pair of images with consistent academic paper styling.
    
    Parameters
    ----------
    img1, img2 : torch.Tensor
        Input image tensors
    prob : float
        Model prediction probability
    label : int, optional
        Ground truth label (1=same, 0=different)
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure with the input image pair
    """
    plt.style.use("seaborn-v0_8-paper")
    
    # create figure for image pair
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # display first image
    img1_np = img1[0].cpu() if img1.dim() == 4 else img1.cpu()
    if img1_np.shape[0] == 1:  # grayscale
        img1_np = img1_np.squeeze()
        ax1.imshow(img1_np, cmap="gray")
    else:  # RGB or other multi-channel
        img1_np = img1_np.permute(1, 2, 0)
        ax1.imshow(img1_np)
    
    # display second image
    img2_np = img2[0].cpu() if img2.dim() == 4 else img2.cpu()
    if img2_np.shape[0] == 1:  # grayscale
        img2_np = img2_np.squeeze()
        ax2.imshow(img2_np, cmap="gray")
    else:  # RGB or other multi-channel
        img2_np = img2_np.permute(1, 2, 0)
        ax2.imshow(img2_np)
    
    ax1.set_title("Image 1", fontsize=12)
    ax2.set_title("Image 2", fontsize=12)
    ax1.axis("off")
    ax2.axis("off")
    
    # title with prediction and ground truth
    title = ""
    if label is not None or prob is not None:
        if label is not None:
            title += f"ground truth: {'same' if label == 1 else 'different'}"
        if label is not None and prob is not None:
            title += "  |  "
        if prob is not None:
            title += f"prediction: {'same' if prob > 0.5 else 'different'} ({prob:.2f})"
    
    fig.suptitle(title, fontsize=12)
    
    plt.tight_layout()
    return fig


def visualize_energy_curve(energy_vals, prob=None, label=None):
    """
    Visualize the energy curve with consistent academic paper styling.
    
    Parameters
    ----------
    energy_vals : list of torch.Tensor
        Energy values from model output
    prob : float, optional
        Model prediction probability
    label : int, optional
        Ground truth label (1=same, 0=different)
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure with the energy curve plot
    """
    plt.style.use("seaborn-v0_8-paper")
    
    TEAL = "#67c2c7"
    SALMON = "#e49797"
    BLACK = "#000000"
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    steps = list(range(len(energy_vals)))
    energies = [e[0].item() for e in energy_vals]
    
    # plot energy curve
    ax.plot(steps, energies, "-", color=BLACK, lw=2.0, alpha=0.8)
    ax.set_xlabel("kuramoto step", fontsize=11)
    ax.set_ylabel("energy", fontsize=11)
    ax.grid(False)
    
    plt.tight_layout()
    return fig


def visualize_oscillator_features(traj1, traj2, c1, c2, max_features=16):
    """
    Visualize oscillator trajectories for multiple features.
    
    Parameters
    ----------
    traj1, traj2 : list of torch.Tensor
        Trajectories of the oscillators
    c1, c2 : torch.Tensor
        Coupling terms
    max_features : int
        Maximum number of features to display
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure with the oscillator trajectories grid
    """
    plt.style.use("seaborn-v0_8-paper")
    
    TEAL = "#67c2c7"
    SALMON = "#e49797"
    BLACK = "#000000"
    
    # extract dimensionality information
    D = traj1[0].shape[1]  # embedding dimension
    features = min(D, max_features)
    
    # calculate grid dimensions
    cols = math.ceil(math.sqrt(features))
    rows = math.ceil(features / cols)
    
    # create figure with appropriate size
    fig = plt.figure(figsize=(3*cols, 3*rows + 0.8))
    
    legend_elements = [
        # oscillators
        plt.Line2D([0], [0], marker='o', color=TEAL, label=r'$osc_1$',
                  markerfacecolor=TEAL, markersize=8, linewidth=0),
        plt.Line2D([0], [0], marker='o', color=SALMON, label=r'$osc_2$',
                  markerfacecolor=SALMON, markersize=8, linewidth=0),

        # c-term directions
        plt.Line2D([0], [0], marker='^', color=TEAL, label=r'$c_{1}$',
                   markerfacecolor=TEAL, markersize=9, linewidth=0),
        plt.Line2D([0], [0], marker='^', color=SALMON,  label=r'$c_{2}$',
                   markerfacecolor=SALMON, markersize=9, linewidth=0),

        # coherence
        plt.Line2D([0], [0], marker='^', color=BLACK, label=r'$\rho$',
                  markerfacecolor=BLACK, markersize=8, linewidth=0),
    ]
    
    fig.legend(handles=legend_elements,
              loc="upper center",
              bbox_to_anchor=(0.5, 0.98),
              ncol=3,
              frameon=True,
              edgecolor="lightgray",
              framealpha=0.8,
              fontsize=10)
    
    # create grid of subplots
    gs = fig.add_gridspec(rows, cols, top=0.9, wspace=0.1, hspace=0.1)
    
    traj1_np = np.array([t[0].cpu().numpy()[:, :2] for t in traj1])  # [steps, D, 2]
    traj2_np = np.array([t[0].cpu().numpy()[:, :2] for t in traj2])
    
    # scale trajectories to separate them
    traj1_np_scaled = traj1_np * 0.97  # inner orbit
    traj2_np_scaled = traj2_np * 1.03  # outer orbit
    
    # get stimulus vectors
    c1_np = c1[0].cpu().numpy()[:, :2]  # [D, 2]
    c2_np = c2[0].cpu().numpy()[:, :2]  # [D, 2]
    
    # compute coherence vectors
    coh_np = 0.5 * (traj1_np + traj2_np)  # [steps, D, 2]
    
    # final frame index
    final_idx = len(traj1) - 1
    
    # plot each feature
    for i in range(features):
        row = i // cols
        col = i % cols
        ax = fig.add_subplot(gs[row, col])
        
        # draw unit circle
        circle = plt.Circle((0, 0), 1.0, 
                            edgecolor="#888888", 
                            facecolor="none", 
                           alpha=0.7, 
                           linewidth=0.8
        )
        ax.add_patch(circle)
        
        # draw coherence vector
        coh_vec = coh_np[final_idx, i]
        ax.quiver(0, 0,
                 coh_vec[0], coh_vec[1],
                 color=BLACK,
                 angles="xy", scale_units="xy", scale=1,
                 width=0.007, alpha=0.9, zorder=4)
                 
        # draw c-term arrows
        ax.quiver(0, 0, 
                 c1_np[i, 0], c1_np[i, 1], 
                 color=TEAL, angles="xy", scale_units="xy", scale=1,
                 width=0.006, alpha=0.9)
        ax.quiver(0, 0, 
                 c2_np[i, 0], c2_np[i, 1], 
                 color=SALMON, angles="xy", scale_units="xy", scale=1,
                 width=0.006, alpha=0.9)
        
        # draw trajectories
        ax.plot(traj1_np_scaled[:, i, 0], traj1_np_scaled[:, i, 1], 
               "-", color=TEAL, alpha=0.8, linewidth=1.5)
        ax.plot(traj2_np_scaled[:, i, 0], traj2_np_scaled[:, i, 1], 
               "-", color=SALMON, alpha=0.8, linewidth=1.5)
        
        # mark ending positions
        ax.scatter(traj1_np_scaled[final_idx, i, 0], traj1_np_scaled[final_idx, i, 1], 
                  c=TEAL, s=40, edgecolor="white", linewidth=0.5, zorder=5, alpha=0.8)
        ax.scatter(traj2_np_scaled[final_idx, i, 0], traj2_np_scaled[final_idx, i, 1], 
                  c=SALMON, s=40, edgecolor="white", linewidth=0.5, zorder=5, alpha=0.8)
        
        # configure axis
        ax.set_aspect("equal", "box")
        ax.set_xlim([-1.3, 1.3])
        ax.set_ylim([-1.3, 1.3])
        ax.axis('off')  # remove borders, axes and ticks
    
    plt.subplots_adjust(top=0.85, wspace=0.1, hspace=0.1)
    return fig


def visualize_single_example(model, img1, img2, label=None, device=None, max_features=16, alpha=1.0):
    """
    Generate all visualization components for a single example.
    
    Parameters
    ----------
    model : nn.Module
        The KuramotoRelationalModel to visualize
    img1, img2 : torch.Tensor
        Input image tensors
    label : int, optional
        Ground truth label (1=same, 0=different)
    device : torch.device, optional
        Device to run the model on
    max_features : int
        Maximum number of features to display
    alpha : float
        Strength of feature alignment in oscillator initialization
        
    Returns
    -------
    tuple
        (img_fig, energy_fig, osc_fig, model_outputs)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    if img1.dim() == 3: img1 = img1.unsqueeze(0)
    if img2.dim() == 3: img2 = img2.unsqueeze(0)
    img1, img2 = img1.to(device), img2.to(device)
    
    # forward pass
    with torch.no_grad():
        logits, osc_data = model(img1, img2, alpha)
        prob = torch.sigmoid(logits)[0, 0].item()
        final_osc1, final_osc2, traj1, traj2, energy_vals, c1, c2 = osc_data
    
    # generate individual visualizations
    img_fig = visualize_image_pair(img1, img2, prob, label)
    energy_fig = visualize_energy_curve(energy_vals, prob, label)
    osc_fig = visualize_oscillator_features(traj1, traj2, c1, c2, max_features)
    
    # return individual figures and model outputs
    return img_fig, energy_fig, osc_fig, (logits, osc_data)


def log_example_visuals(model, sample_batch, epoch, device, alpha=1.0, embedding_dim=64, animate=True):
    """
    Logs one 'same' pair and one 'different' pair to WandB, plus 2-D oscillator
    animations (first epoch and last epoch only).

    Parameters
    ----------
    model : nn.Module
        The KuramotoRelationalModel (already .eval()-ed outside)
    sample_batch : tuple
        Mini-batch that yields ((img1,img2), label)
    epoch : int
        Current epoch index
    device : torch.device
        Device to run computation on
    alpha : float
        Strength of feature alignment in oscillator initialization
    embedding_dim : int
        Dimension of feature embeddings
    animate : bool
        Whether to create animations
    """
    model.eval()
    with torch.no_grad():
        # take first batch
        (imgs1, imgs2), labels = sample_batch

        # find at least one example of each class
        try:
            idx_same = (labels == 1).nonzero(as_tuple=True)[0][0].item()
            idx_diff = (labels == 0).nonzero(as_tuple=True)[0][0].item()
        except IndexError:
            # batch did not contain both classes – just skip
            return

        # SAME example
        same_img1, same_img2 = imgs1[idx_same], imgs2[idx_same]
        img_fig, energy_fig, osc_fig, (logits, osc_data) = visualize_single_example(
            model,
            same_img1,
            same_img2,
            label=1,
            device=device,
            max_features=embedding_dim,
            alpha=alpha
        )
        
        # log each component separately for the "same" example
        wandb.log({
            f"examples_same/epoch_{epoch}_images": wandb.Image(img_fig),
            f"examples_same/epoch_{epoch}_energy": wandb.Image(energy_fig),
            f"examples_same/epoch_{epoch}_oscillators": wandb.Image(osc_fig)
        })
        
        plt.close(img_fig)
        plt.close(energy_fig)
        plt.close(osc_fig)

        # DIFFERENT example
        diff_img1, diff_img2 = imgs1[idx_diff], imgs2[idx_diff]
        img_fig, energy_fig, osc_fig, (logits, osc_data) = visualize_single_example(
            model,
            diff_img1,
            diff_img2,
            label=0,
            device=device,
            max_features=embedding_dim,
            alpha=alpha
        )
        
        # log each component separately for the "different" example
        wandb.log({
            f"examples_diff/epoch_{epoch}_images": wandb.Image(img_fig),
            f"examples_diff/epoch_{epoch}_energy": wandb.Image(energy_fig),
            f"examples_diff/epoch_{epoch}_oscillators": wandb.Image(osc_fig)
        })
        
        plt.close(img_fig)
        plt.close(energy_fig)
        plt.close(osc_fig)
        
        # oscillator animations (first & last epoch only)
        if epoch in (-1, model.oscillator_network.num_steps - 1):
            # reuse the "same" and "different" cases above
            for tag, (i1, i2) in {
                "osc_same": (imgs1[idx_same], imgs2[idx_same]),
                "osc_diff": (imgs1[idx_diff], imgs2[idx_diff]),
            }.items():
                i1 = i1.unsqueeze(0).to(device)
                i2 = i2.unsqueeze(0).to(device)
                _, osc_data = model(i1, i2, alpha)
                _, _, traj1, traj2, _, c1, c2 = osc_data

                anim, fig = visualize_oscillators_2d(
                    traj1, traj2, c1, c2,
                    max_features=embedding_dim,
                    trail_length=25,
                    animate=animate,
                )
                if animate:
                    _wandb_log_animation(anim, tag=f"{tag}/epoch_{epoch}")
                wandb.log({f"{tag}/epoch_{epoch}_still": wandb.Image(fig)})
                plt.close(fig)