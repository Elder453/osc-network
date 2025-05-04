"""
Configuration settings for experiments.

Contains default hyperparameters and command-line argument handling.
"""

import argparse
from argparse import Namespace


def get_args():
    """
    Get configuration arguments for the experiments.
    
    Returns
    -------
    argparse.Namespace
        Parsed arguments with default values
    """
    parser = argparse.ArgumentParser(description="Oscillatory Relational Learning")
    
    # Experiment settings
    parser.add_argument("--exp_name", type=str, default="kuramoto-run", 
                        help="Experiment name")
    parser.add_argument("--seed", type=int, default=123, 
                        help="Random seed for reproducibility")
    parser.add_argument("--load_model", type=str, default="final", 
                        help="Which model to load: 'final' or 'best_val'")
    parser.add_argument("--model_type", type=str, default="kuramoto", 
                        choices=["kuramoto", "baseline"], 
                        help="Model type to use")
    parser.add_argument("--animate", action="store_true", default=True,
                        help="Whether to create animations for visualization")
    
    # Training settings
    parser.add_argument("--epochs", type=int, default=50, 
                        help="Number of epochs to train")
    parser.add_argument("--rmts_epochs", type=int, default=50, 
                        help="Number of epochs to train RMTS")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, 
                        help="Learning rate")
    parser.add_argument("--finetune", type=str, default=None, 
                        help="Path to checkpoint for finetuning")
    
    # Model parameters
    parser.add_argument("--embedding_dim", type=int, default=16, 
                        help="Feature embedding dimension (D)")
    parser.add_argument("--oscillator_dim", type=int, default=2, 
                        help="Oscillator dimension (N)")
    parser.add_argument("--num_steps", type=int, default=25, 
                        help="Number of Kuramoto steps (T)")
    parser.add_argument("--step_size", type=float, default=0.1, 
                        help="Step size for Kuramoto updates (gamma)")
    parser.add_argument("--use_omega", action="store_true", default=True,
                        help="Whether to use natural frequency matrices")
    parser.add_argument("--omega_kappa", type=float, default=2.0, 
                        help="Concentration parameter for sampling frequencies")
    parser.add_argument("--disable_between", action="store_true", default=False,
                        help="Whether to disable between-object coupling")
    parser.add_argument("--symmetric_j", action="store_true", default=False,
                        help="Whether to enforce symmetry in coupling matrices")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, 
                        help="Hidden dimension for MLP classifiers")
    parser.add_argument("--alpha", type=float, default=1.0, 
                        help="Strength of feature alignment in initialization")
    
    # Dataset parameters
    parser.add_argument("--n_train_glyphs", type=int, default=95, 
                        help="Number of glyphs used in training")
    parser.add_argument("--n_train", type=int, default=5000, 
                        help="Number of training examples")
    parser.add_argument("--n_val", type=int, default=64, 
                        help="Number of validation examples")
    parser.add_argument("--n_test", type=int, default=5000, 
                        help="Number of test examples")
    parser.add_argument("--num_workers", type=int, default=2, 
                        help="Number of dataloader workers")
    
    args = parser.parse_args()
    return args


def get_default_args():
    """
    Get default arguments without parsing command line.
    
    Useful for notebooks or scripts where command-line args aren't available.
    
    Returns
    -------
    argparse.Namespace
        Default arguments
    """
    return Namespace(
        exp_name="kuramoto-run",
        seed=123,
        load_model="final",
        model_type="kuramoto",
        animate=True,
        
        # Training settings
        epochs=50,
        rmts_epochs=50,
        batch_size=64,
        lr=1e-3,
        finetune=None,
        
        # Model parameters
        embedding_dim=16,
        oscillator_dim=2,
        num_steps=25,
        step_size=0.1,
        use_omega=True,
        omega_kappa=2.0,
        disable_between=False,
        symmetric_j=False,
        mlp_hidden_dim=64,
        alpha=1,
        
        # Dataset parameters
        n_train_glyphs=95,
        n_train=5000,
        n_val=64,
        n_test=5000,
        num_workers=2
    )