"""
Training pipeline for relational match-to-sample (RMTS) task.

Contains functions for the RMTS training loop, validation, and testing
using the pre-trained same/different model as a feature extractor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from models.kuramoto import KuramotoRelationalModel
from models.baseline import BaselineSameDiffModel, BaselineRMTSClassifier
from models.classification import RMTSClassifier
from data.datasets import IconRelMatchToSampleDataset
from data.transforms import COMMON_TRANSFORM


def train_rmts(args, history_rmts, trained_model=None):
    """
    Main training function for the relational match-to-sample task.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    history_rmts : dict
        Dictionary to store training history
    trained_model : nn.Module, optional
        Pre-trained same/different model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create job directory
    jobdir = Path("runs") / args.exp_name
    jobdir.mkdir(parents=True, exist_ok=True)
    
    # load the frozen encoder if not provided
    if trained_model is None:
        if args.model_type == "kuramoto":
            elder_model = KuramotoRelationalModel(
                input_channels=1,
                embedding_dim=args.embedding_dim,
                oscillator_dim=args.oscillator_dim,
                num_steps=args.num_steps,
                step_size=args.step_size,
                use_omega=args.use_omega,
                omega_kappa=args.omega_kappa,
                symmetric_j=args.symmetric_j,
                mlp_hidden_dim=args.mlp_hidden_dim
            ).to(device)
        else:
            elder_model = BaselineSameDiffModel(
                input_channels=1,
                embedding_dim=args.embedding_dim,
                oscillator_dim=args.oscillator_dim,
            ).to(device)

        # load trained weights (the path from same/diff run)
        model_ckpt = "model_final" if args.load_model == "final" else "model_best_val"
        ckpt_path = f"runs/{args.exp_name}/{model_ckpt}.pt"
        state_dict = torch.load(ckpt_path, map_location=device)
        elder_model.load_state_dict(state_dict)
        elder_model.to(device)
    else:
        elder_model = trained_model.to(device)
        
    # freeze the model
    elder_model.eval()
    for param in elder_model.parameters():
        param.requires_grad = False

    # create RMTS datasets
    # "TRAIN" on first m icons
    rmts_train_dataset = IconRelMatchToSampleDataset(
        image_dir="./imgs/",
        num_samples=args.n_train,
        transform=COMMON_TRANSFORM,
        icon_indices=range(args.n_train_glyphs),  # only icons [0..n_train_glyphs-1]
        seed=args.seed
    )
    # "VAL" also on first m icons
    rmts_val_dataset = IconRelMatchToSampleDataset(
        image_dir="./imgs/",
        num_samples=args.n_val,
        transform=COMMON_TRANSFORM,
        icon_indices=range(args.n_train_glyphs), # also [0..n_train_glyphs-1]
        seed=args.seed
    )
    # "TEST" on last 100-m icons => out-of-distribution
    test_indices = (range(args.n_train_glyphs, 100) 
                    if args.n_train_glyphs != 100 
                    else range(100))
    rmts_test_dataset = IconRelMatchToSampleDataset(
        image_dir="./imgs/",
        num_samples=args.n_test,
        transform=COMMON_TRANSFORM,
        icon_indices=test_indices,
        seed=args.seed
    )

    # create dataloaders
    rmts_train_loader = DataLoader(
        rmts_train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    rmts_val_loader = DataLoader(
        rmts_val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False
    )
    rmts_test_loader = DataLoader(
        rmts_test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False
    )
    
    # create RMTSClassifier
    if args.model_type == "kuramoto":
        rmts_classifier = RMTSClassifier(
            coherence_dim=args.embedding_dim, 
            hidden_dim=args.mlp_hidden_dim,
        ).to(device)
    else:
        rmts_classifier = BaselineRMTSClassifier(
            coherence_dim=args.embedding_dim, 
            hidden_dim=args.mlp_hidden_dim
        ).to(device)
    get_vec = elder_model.get_coherence_vector

    # training setup
    optimizer = optim.Adam(rmts_classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    num_epochs = args.rmts_epochs

    # train RMTS classifier
    for epoch in range(num_epochs):
        # training phase
        rmts_classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for (s1, s2, t1a, t1b, t2a, t2b), label in rmts_train_loader:
            # move everything to device
            s1, s2 = s1.to(device), s2.to(device)
            t1a, t1b = t1a.to(device), t1b.to(device)
            t2a, t2b = t2a.to(device), t2b.to(device)
            label = label.long().to(device)  # shape [batch_size]

            # zero out old grads
            optimizer.zero_grad()

            # extract coherence vectors from the frozen model
            source_vec = get_vec(s1, s2)   # [batch_size, D=64]
            t1_vec = get_vec(t1a, t1b)     # [batch_size, 64]
            t2_vec = get_vec(t2a, t2b)     # [batch_size, 64]

            # forward pass in RMTS classifier => 2 logits
            logits = rmts_classifier(source_vec, t1_vec, t2_vec)    # shape [batch_size, 2]

            # compute cross-entropy
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * label.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == label).sum().item()
            train_total += label.size(0)

            # record batch accuracy
            if history_rmts is not None:
                history_rmts["train_acc"].append(
                    (preds == label).float().mean().item()
                )

        avg_train_loss = train_loss / train_total
        avg_train_acc = train_correct / train_total

        # validation phase
        rmts_classifier.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for (s1, s2, t1a, t1b, t2a, t2b), label in rmts_val_loader:
                s1, s2 = s1.to(device), s2.to(device)
                t1a, t1b = t1a.to(device), t1b.to(device)
                t2a, t2b = t2a.to(device), t2b.to(device)
                label = label.long().to(device)

                source_vec = get_vec(s1, s2)
                t1_vec = get_vec(t1a, t1b)
                t2_vec = get_vec(t2a, t2b)
                logits = rmts_classifier(source_vec, t1_vec, t2_vec)

                loss = criterion(logits, label)
                val_loss += loss.item() * label.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == label).sum().item()
                val_total += label.size(0)

        avg_val_loss = val_loss / val_total
        avg_val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss={avg_train_loss:.4f}, Acc={avg_train_acc:.4f} | "
              f"Val Loss={avg_val_loss:.4f}, Acc={avg_val_acc:.4f}")

    # test evaluation
    rmts_classifier.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for (s1, s2, t1a, t1b, t2a, t2b), label in rmts_test_loader:
            s1, s2 = s1.to(device), s2.to(device)
            t1a, t1b = t1a.to(device), t1b.to(device)
            t2a, t2b = t2a.to(device), t2b.to(device)
            label = label.long().to(device)

            source_vec = get_vec(s1, s2)
            t1_vec = get_vec(t1a, t1b)
            t2_vec = get_vec(t2a, t2b)
            logits = rmts_classifier(source_vec, t1_vec, t2_vec)

            loss = criterion(logits, label)
            test_loss += loss.item() * label.size(0)
            preds = torch.argmax(logits, dim=1)
            test_correct += (preds == label).sum().item()
            test_total += label.size(0)

    avg_test_loss = test_loss / test_total
    avg_test_acc = test_correct / test_total
    print(f"\nRMTS test loss={avg_test_loss:.4f} | acc={avg_test_acc:.4f}")

    print("R-MTS training complete.")
    if args.model_type == "kuramoto":
        np.save(jobdir / f"{args.model_type}_Jout{args.disable_between}_Omega{args.use_omega}_Jsymm{args.symmetric_j}_rmts_train_batch_acc.npy", 
                np.asarray(history_rmts["train_acc"]))
    else:
        np.save(jobdir / f"{args.model_type}_rmts_train_batch_acc.npy", 
                np.asarray(history_rmts["train_acc"]))
                
    return rmts_classifier