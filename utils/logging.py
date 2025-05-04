"""
Logging utilities.

Provides functions for setting up logging and tracking model performance.
"""

import os
import logging
import torch
import wandb
from typing import Dict


def create_logger(logging_dir):
    """
    Configure and return a logger that writes to both console and file.
    
    Parameters
    ----------
    logging_dir : str
        Directory to store log files
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    os.makedirs(logging_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(logging_dir, "log.txt")),
        ],
    )
    return logging.getLogger(__name__)


def log_gradient_norms(model: torch.nn.Module, component_name: str) -> None:
    """
    Log the gradient norms for a specific component of the model to Weights & Biases.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model containing the component
    component_name : str
        Name of the component to log gradients for
    """
    if not hasattr(model, component_name):
        return
    
    component = getattr(model, component_name)
    grad_dict = {}
    
    for name, param in component.named_parameters():
        if param.grad is not None:
            # compute different norms
            grad_l1 = param.grad.abs().mean().item()
            grad_l2 = param.grad.norm(2).item()
            grad_max = param.grad.abs().max().item()
            
            # log to wandb
            grad_dict[f"{component_name}.{name}.grad_l1"] = grad_l1
            grad_dict[f"{component_name}.{name}.grad_l2"] = grad_l2
            grad_dict[f"{component_name}.{name}.grad_max"] = grad_max
    
    # log all gradients at once
    if grad_dict:
        wandb.log(grad_dict)


def plot_history(train_h: Dict[str, list], title: str, save_dir: str, tag: str):
    """
    Draw a single-axis figure that shows batch-level training accuracy.
    
    Parameters
    ----------
    train_h : dict
        Dictionary with training history
    title : str
        Plot title
    save_dir : str
        Directory to save the plot
    tag : str
        Tag for the output filename
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    if not train_h["train_acc"]:  # nothing to plot
        return

    plt.style.use("seaborn-v0_8-paper")
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(train_h["train_acc"],
            label="train accuracy",
            lw=2.0, color="tab:blue")

    ax.set_xlabel("Batch number")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(alpha=.3)
    ax.set_title(title)
    ax.legend(loc="lower right")

    out_path = Path(save_dir) / f"{tag}_train_accuracy.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    # optional: still push to weights & biases
    wandb.log({f"history/{tag}": wandb.Image(str(out_path))})