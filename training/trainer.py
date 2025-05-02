"""
Training pipeline for same/different task.

Contains functions for the main training loop, validation, and testing
of Kuramoto Relational Learning models.
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np
from pathlib import Path
from tqdm import tqdm
from contextlib import nullcontext
from torch.utils.data import DataLoader, random_split

from models.kuramoto import KuramotoRelationalModel
from models.baseline import BaselineSameDiffModel
from data.datasets import IconSameDiffDataset
from data.transforms import COMMON_TRANSFORM
from utils.logging import create_logger, log_gradient_norms, plot_history
from utils.visualization import (
    collect_energy_values, plot_energy_one_epoch, build_energy_animation,
    plot_energy_init_vs_trained, log_example_visuals
)


def run_epoch(
    *,
    model,
    dataloader,
    criterion,
    accelerator,
    epoch_idx: int,
    mode: str,                   # "train" | "val" | "test"
    optimizer=None,
    scheduler=None,
    log_every: int = 100,
    grad_log_every: int = 100,
    alpha: float = 1.0,
    history_sd: dict = None
):
    """
    Run one epoch of training or evaluation.
    
    Parameters
    ----------
    model : nn.Module
        Model to train or evaluate
    dataloader : torch.utils.data.DataLoader
        Dataloader for the epoch
    criterion : torch.nn.Module
        Loss function
    accelerator : accelerate.Accelerator
        Accelerator for distributed training
    epoch_idx : int
        Current epoch index
    mode : str
        "train", "val", or "test"
    optimizer : torch.optim.Optimizer, optional
        Optimizer for training
    scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        Learning rate scheduler
    log_every : int
        How often to log batch metrics
    grad_log_every : int
        How often to log gradient norms
    alpha : float
        Strength of feature alignment in oscillator initialization
    history_sd : dict, optional
        Dictionary to store training history
        
    Returns
    -------
    tuple
        (average_loss, average_accuracy)
    """
    is_train = mode == "train"
    model.train() if is_train else model.eval()

    tot_loss = tot_correct = tot_samples = 0
    loader = tqdm(dataloader, desc=f"E{epoch_idx+1} [{mode}]", leave=False, disable=True)

    grad_ctx = nullcontext() if is_train else torch.no_grad()

    for step, (imgs, labels) in enumerate(loader, 1):
        img1, img2 = imgs                              # tuple unpack
        img1, img2, labels = (
            t.to(accelerator.device, non_blocking=True) for t in (img1, img2, labels)
        )
        labels = labels.float().unsqueeze(1)

        with grad_ctx, accelerator.autocast():
            logits, *_ = model(img1, img2, alpha)
            loss = criterion(logits, labels)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()                 # one LR update per *batch*

        # calculate stats
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = probs > 0.5
            batch_acc = (preds == labels.bool()).float().mean().item()

        if mode == "train" and history_sd is not None:
            history_sd["train_acc"].append(batch_acc)

        bs = labels.size(0)
        tot_loss += loss.item() * bs
        tot_correct += (preds == labels.bool()).sum().item()
        tot_samples += bs

        # wandb batch logging
        if step % log_every == 0:
            log_dict = {
                f"{mode}/batch_loss": loss.item(),
                f"{mode}/batch_acc": batch_acc,
            }
            if is_train:
                log_dict["lr"] = optimizer.param_groups[0]["lr"]
            wandb.log(log_dict)

        # optional gradient histograms
        if is_train and step % grad_log_every == 0:
            log_gradient_norms(model, "feature_extractor")
            if hasattr(model, "oscillator_network"):
                log_gradient_norms(model, "oscillator_network")
                log_gradient_norms(model, "coherence_measurement")
            log_gradient_norms(model, "classifier")

        loader.set_postfix(loss=f"{loss.item():.3f}", acc=f"{batch_acc:.3f}")

    avg_loss = tot_loss / tot_samples
    avg_acc = tot_correct / tot_samples
    return avg_loss, avg_acc


def train_samediff(args, history_sd):
    """
    Main training function for the same/different task.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    history_sd : dict
        Dictionary to store training history
        
    Returns
    -------
    torch.nn.Module
        Trained model
    """
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from torch.optim.lr_scheduler import (
        LinearLR,
        CosineAnnealingLR,
        SequentialLR,
    )
    
    # accelerator setup
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device
    accelerator.print(f"Running on {accelerator.device}")
    set_seed(args.seed)
    
    # for reproducibility
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # create job directory
    jobdir = os.path.join("runs", args.exp_name)
    os.makedirs(jobdir, exist_ok=True)
    
    # set up logging
    logger = create_logger(jobdir)
    logger.info(f"Experiment directory created at {jobdir}")
    
    # dataset creation
    logger.info("Generating training/validation datasets...")
    logger.info(range(args.n_train_glyphs))
    full_dataset = IconSameDiffDataset(
        image_dir="./imgs/",
        num_samples=args.n_train + args.n_val,
        transform=COMMON_TRANSFORM,
        icon_indices=range(args.n_train_glyphs),
        seed=args.seed
    )
    train_dataset, val_dataset = random_split(
        full_dataset,
        [args.n_train, args.n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    logger.info("Generating test dataset (OOD)...")
    logger.info(range(args.n_train_glyphs, 100))
    test_dataset = IconSameDiffDataset(
        image_dir="./imgs/",
        num_samples=args.n_test,
        transform=COMMON_TRANSFORM,
        icon_indices=(range(args.n_train_glyphs, 100) if args.n_train_glyphs != 100 else range(100)),
        seed=args.seed
    )
    
    # dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers>0)
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers>0)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers>0)
    )
    
    # initialize logging
    wandb_config = {
        "model_type": "KuramotoRelationalModel" if args.model_type == "kuramoto" else "BaselineSameDiffModel",
        "dataset_type": "shape_same_different",
        "embedding_dim": args.embedding_dim,
        "oscillator_dim": args.oscillator_dim,
        "num_steps": args.num_steps,
        "step_size": args.step_size,
        "batch_size": args.batch_size,
        "num_epochs": args.epochs,
        "learning_rate": args.lr,
        "train_samples": args.n_train,
        "val_samples": args.n_val,
        "use_omega": args.use_omega,
        "omega_kappa": args.omega_kappa,
        "symmetric_j": args.symmetric_j,
        "alpha": args.alpha,
    }
    wandb.init(project="kuramoto-relational-model", name=args.exp_name, config=wandb_config)
    
    # model & trainer preparation
    logger.info("Initializing model and optimizer...")
    
    if args.model_type == "kuramoto":
        model = KuramotoRelationalModel(
            input_channels=1,
            embedding_dim=args.embedding_dim,
            oscillator_dim=args.oscillator_dim,
            num_steps=args.num_steps,
            step_size=args.step_size,
            use_omega=args.use_omega,
            omega_kappa=args.omega_kappa,
            disable_between=args.disable_between,
            symmetric_j=args.symmetric_j,
            mlp_hidden_dim=args.mlp_hidden_dim,
        )
    else:   # cnn-only baseline
        model = BaselineSameDiffModel(
            input_channels=1,
            embedding_dim=args.embedding_dim,
            oscillator_dim=args.oscillator_dim,
        )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    if args.finetune:
        logger.info(f"Loading checkpoint from {args.finetune}...")
        ckpt = torch.load(args.finetune, map_location="cpu")
        accelerator.wait_for_everyone()
        accelerator.unwrap_model(model).load_state_dict(ckpt)

    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader
    )

    # lr schedule
    total_batches = args.epochs * len(train_loader)      # step every batch
    warmup_steps = int(0.05 * total_batches)            # 5% of the run

    warmup_sched = LinearLR(
        optimizer,
        start_factor=1/25,        # lr rises from args.lr/25 → args.lr
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_sched = CosineAnnealingLR(
        optimizer,
        T_max=total_batches - warmup_steps,
        eta_min=args.lr * 1e-3,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_steps],
    )
    
    # model parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {total_params}")
    wandb.config.update({"total_params": total_params})
    
    # watch model gradients
    wandb.watch(model, log="all", log_freq=100)

    # track best val performance
    best_val_acc = 0.0
    best_epoch = -1
    
    # start training
    logger.info(f"Start training for {args.epochs} epochs")
    sample_batch = next(iter(train_loader))
    all_epoch_energies = {}

    # baseline energy before training
    if args.model_type == "kuramoto":
        init_energy = collect_energy_values(model, sample_batch, accelerator.device)
        all_epoch_energies[-1] = init_energy
        wandb.log({"energy/plot":
                   wandb.Image(plot_energy_one_epoch(init_energy, epoch=-1))})

    # baseline visuals before training
    if args.model_type == "kuramoto":
        log_example_visuals(
            model, 
            sample_batch, 
            epoch=-1, 
            device=accelerator.device,
            alpha=args.alpha,
            embedding_dim=args.embedding_dim,
            animate=args.animate
        )

    for epoch in range(args.epochs):
        start = time.perf_counter()
        train_loss, train_acc = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            accelerator=accelerator,
            epoch_idx=epoch,
            mode="train",
            optimizer=optimizer,
            scheduler=scheduler,
            alpha=args.alpha,
            history_sd=history_sd
        )

        val_loss, val_acc = run_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            accelerator=accelerator,
            epoch_idx=epoch,
            mode="val",
            alpha=args.alpha
        )
        elapsed = time.perf_counter() - start

        # check if this is the best validation accuracy so far
        if val_acc > best_val_acc and epoch > 24:
            best_val_acc = val_acc
            best_epoch = epoch
            
            # save the best model
            accelerator.save(accelerator.unwrap_model(model).state_dict(),
                           os.path.join(jobdir, "model_best_val.pt"))
            logger.info(f"\nNew best model saved at epoch {epoch+1} with validation accuracy {val_acc:.4f}")

        # epoch-level summary
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,  "train/acc": train_acc,
            "val/loss": val_loss,      "val/acc": val_acc,
        })

        logger.info(f"\nEpoch {epoch+1}/{args.epochs} | {elapsed:6.1f}s\n"
                    f"train loss:{train_loss:.4f} | acc:{train_acc:.4f}\n"
                    f"val   loss:{val_loss:.4f} | acc:{val_acc:.4f}")

        if args.model_type == "kuramoto":
            # energy visualization
            energy_vals = collect_energy_values(model, sample_batch, accelerator.device)
            all_epoch_energies[epoch] = energy_vals
            wandb.log({"energy/plot": wandb.Image(plot_energy_one_epoch(energy_vals, epoch))})
    
            # sample visualizations every 5 epochs
            if epoch % 5 == 0 or epoch == args.epochs - 1:
                log_example_visuals(
                    model, 
                    sample_batch, 
                    epoch, 
                    accelerator.device,
                    alpha=args.alpha,
                    embedding_dim=args.embedding_dim,
                    animate=args.animate
                )

    # end training
    accelerator.save(accelerator.unwrap_model(model).state_dict(),
                     os.path.join(jobdir, "model_final.pt"))

    if args.model_type == "kuramoto":
        # initialization vs final epoch's energy
        e_plot = plot_energy_init_vs_trained(all_epoch_energies, init_epoch=-1, final_epoch=args.epochs - 1)
        wandb.log({"energy/plot_0_end": wandb.Image(e_plot)})

        # energy GIF once at the end
        gif = build_energy_animation(all_epoch_energies,
                                     fname=os.path.join(jobdir, "energy_evolution.gif"))
        wandb.log({"energy/gif": wandb.Video(gif, format="gif")})

    plot_history(
        history_sd,
        title="Same/Different",
        save_dir=jobdir,
        tag="same_diff",
    )
    
    # test set evaluation
    test_loss, test_acc = run_epoch(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        accelerator=accelerator,
        epoch_idx=args.epochs,
        mode="test",
        alpha=args.alpha
    )
    wandb.log({"test/loss": test_loss, "test/acc": test_acc})
    logger.info(f"\nS/D test loss={test_loss:.4f} | acc={test_acc:.4f}")
    print(f"\nS/D test loss={test_loss:.4f} | acc={test_acc:.4f}")
    
    print("S/D training complete.\n")
    if args.model_type == "kuramoto":
        np.save(Path(jobdir) / f"{args.model_type}_Jout{args.disable_between}_Omega{args.use_omega}_Jsymm{args.symmetric_j}_sd_train_batch_acc.npy", 
                np.asarray(history_sd["train_acc"]))
    else:
        np.save(Path(jobdir) / f"{args.model_type}_sd_train_batch_acc.npy", 
                np.asarray(history_sd["train_acc"]))

    wandb.finish(quiet=True)
    accelerator.end_training()
    
    # extract the model for RMTS training
    unwrapped_model = accelerator.unwrap_model(model)
    
    return unwrapped_model