"""
Main entry point for Kuramoto Relational Learning experiments.

This script handles parsing arguments and launching training for same/different
and relational match-to-sample tasks.
"""

import argparse
import numpy as np
import wandb
from pathlib import Path
import torch

from config import get_args
from models.kuramoto import KuramotoRelationalModel
from models.baseline import BaselineSameDiffModel
from training.trainer import train_samediff
from training.rmts_trainer import train_rmts


def main():
    """
    Main entry point for the Kuramoto Relational Learning experiments.
    
    Handles argument parsing and delegates to training functions.
    """
    # Load arguments
    args = get_args()
    
    # Initialize history dictionaries
    history_sd = {"train_acc": []}     # same/different
    history_rmts = {"train_acc": []}   # RMTS

    # Train the same/different model
    trained_model = train_samediff(args, history_sd)
    
    # Train the RMTS classifier using the trained model
    train_rmts(args, history_rmts, trained_model)


if __name__ == "__main__":
    main()