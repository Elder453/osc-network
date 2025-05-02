import math
import tempfile, os
import argparse
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from itertools import cycle
from PIL import Image, ImageDraw, ImageOps
from torchvision import transforms
import wandb
from tqdm import tqdm
import time
from typing import Tuple, List, Optional, Any, Sequence
from accelerate import Accelerator
from pathlib import Path

from contextlib import nullcontext
from accelerate.utils import set_seed    
from torch.optim.lr_scheduler import (
    LinearLR,
    CosineAnnealingLR,
    SequentialLR,
)

from matplotlib.animation import FuncAnimation
try:
    from IPython import get_ipython
    IN_NOTEBOOK = get_ipython() is not None
except Exception:
    IN_NOTEBOOK = False

if IN_NOTEBOOK:
    from tqdm.notebook import tqdm
else:
    from tqdm.auto import tqdm
## Utility Functions
def _wandb_log_animation(anim, tag, fps=2):
    """
    Save a matplotlib FuncAnimation to GIF and log it to Weights & Biases.

    Parameters
    ----------
    anim : matplotlib.animation.FuncAnimation
    tag  : str   – key prefix in wandb (e.g. "epoch10_osc_same")
    fps  : int   – frames‑per‑second for the GIF
    """
    tmpdir   = Path(tempfile.mkdtemp())
    gif_path = tmpdir / f"{tag}.gif"
    gif_path.parent.mkdir(parents=True, exist_ok=True)

    # write the GIF
    anim.save(gif_path, writer="pillow", fps=fps)

    # push to W&B
    wandb.log({
        f"{tag}_gif"  : wandb.Video(str(gif_path), format="gif"),
    })

def load_icons(image_dir: str, icon_indices: Sequence[int]):
    """
    Load and return a list of PIL.Image from `image_dir` for the given indices.
    Raises if none found.
    """
    icons = []
    for i in icon_indices:
        path = os.path.join(image_dir, f"{i}.png")
        if os.path.exists(path):
            icons.append(Image.open(path).convert("L"))
    if not icons:
        raise ValueError(f"No images found in {image_dir} with indices: {icon_indices}")
    return icons

def create_logger(logging_dir):
    """Configure logger to write to both console and file."""
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

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    """Save model and optimizer state to a checkpoint file."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)

def log_gradient_norms(model: nn.Module, component_name: str) -> None:
    """
    Log the gradient norms for a specific component of the model.
    
    Args:
        model: The model containing the component
        component_name: Name of the component to log gradients for
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
## Data Generation
class RandomResizeAndPad(object):
    """
    Randomly resizes an image and then pastes it onto a fixed-size canvas
    at a random position.

    Args:
        canvas_size (tuple): Output canvas size as (width, height). Set this to (32, 32) for your icons.
        scale_range (tuple): Range of scale factors (min, max). E.g., (0.7, 1.0).
        fill: Background fill color (default black; for RGB use a tuple like (0,0,0)).
    """
    def __init__(self, canvas_size=(32, 32), scale_range=(0.7, 1.0), fill=(0)):
        self.canvas_size = canvas_size
        self.scale_range = scale_range
        self.fill = fill

    def __call__(self, img):
        # Sample a random scale factor from the provided range.
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        orig_w, orig_h = img.size
        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))
        
        # Resize the icon image.
        resized_img = img.resize((new_w, new_h), resample=Image.BICUBIC)

        # Invert the colors: black becomes white and vice versa.
        # Note: ImageOps.invert requires the image to be in mode 'RGB' or 'L'.
        inverted_img = ImageOps.invert(resized_img.convert("L"))
        
        # Create a blank canvas with the given size and fill color.
        canvas = Image.new("L", self.canvas_size, self.fill)
        
        # Calculate maximum offsets such that the icon fits within the canvas.
        max_x = self.canvas_size[0] - new_w
        max_y = self.canvas_size[1] - new_h
        
        # Randomly select the top-left coordinates.
        x_offset = random.randint(0, max_x) if max_x > 0 else 0
        y_offset = random.randint(0, max_y) if max_y > 0 else 0
        
        # Paste the resized image onto the canvas at the random location.
        canvas.paste(inverted_img, (x_offset, y_offset))
        return canvas
COMMON_TRANSFORM = transforms.Compose([
    RandomResizeAndPad(canvas_size=(32, 32), scale_range=(0.7, 1.0), fill=(0)),
    transforms.ToTensor(),
])
class IconSameDiffDataset(Dataset):
    """
    Generates pairs of icons (same or different) from a folder of up to 100 icons.
    Each sample = (icon_tensor1, icon_tensor2), label.
      label = 1 if same, 0 if different.
    """
    def __init__(
        self,
        image_dir: str,
        num_samples: int = 10000,
        transform = None,
        icon_indices: Optional[List[int]] = None,
        seed: int = 0
    ):
        """
        Args:
            image_dir: folder with up to 100 icons named 0.png, 1.png, ..., 99.png
            num_samples: how many total pairs to generate
            transform: optional transforms to apply (e.g. transforms.ToTensor())
            icon_indices: which icon indices to actually load (e.g. range(80))
        """
        super().__init__()
        self.image_dir = image_dir
        self.num_samples = num_samples
        self.transform = transform or transforms.ToTensor()
        
        # load icons via helper
        if icon_indices is None:
            icon_indices = range(100)
        self.icons = load_icons(self.image_dir, icon_indices)

        rng = random.Random(seed)
        self._pairs: list[tuple[int, int, int]] = []     # (idxA, idxB, label)
        for _ in range(num_samples):
            if rng.random() < 0.5:                       # “same” case
                j = rng.randrange(len(self.icons))
                self._pairs.append((j, j, 1))
            else:                                        # “different” case
                j, k = rng.sample(range(len(self.icons)), 2)
                self._pairs.append((j, k, 0))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        j, k, label = self._pairs[idx]
        iconA, iconB = self.icons[j], self.icons[k]
        return (self.transform(iconA), self.transform(iconB)), label

def visualize_icon_samediff_sample(dataset, idx=None):
    """
    Visualizes a sample from IconSameDiffDataset.
    
    Args:
        dataset: an instance of IconSameDiffDataset.
        idx: Optional index. If None, a random sample is visualized.
    """
    if idx is None:
        idx = random.randint(0, len(dataset) - 1)
    
    # Retrieve sample: ((img1_tensor, img2_tensor), label)
    (img1, img2), label = dataset[idx]
    
    # Convert tensors to numpy arrays for visualization.
    # For an RGB image (shape: [3, H, W]), permute to [H, W, 3].
    if img1.shape[0] != 1:
        img1_np = img1.permute(1, 2, 0).cpu().numpy()
        cmap_1 = None
    else:
        img1_np = img1.squeeze().cpu().numpy()
        cmap_1 = "gray"
        
    if img2.shape[0] != 1:
        img2_np = img2.permute(1, 2, 0).cpu().numpy()
        cmap_1 = None
    else:
        img2_np = img2.squeeze().cpu().numpy()
        cmap_1 = "gray"

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img1_np, cmap=cmap_1)
    plt.title('image 1')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img2_np, cmap=cmap_1)
    plt.title('image 2')
    plt.axis('off')
    
    plt.suptitle(f"Same label (1=same, 0=different): {label}")
    plt.show()


# For the same/diff dataset:
samediff_dataset = IconSameDiffDataset(
    image_dir="./imgs/",
    num_samples=5,
    transform=COMMON_TRANSFORM
)

visualize_icon_samediff_sample(samediff_dataset)
plt.close()
class IconRelMatchToSampleDataset(Dataset):
    """
    Relational Match-to-Sample Dataset using icons.
    Each item yields (S1, S2, T1a, T1b, T2a, T2b), label in {0,1}
    indicating which target matches the source's relation.
    """
    def __init__(
        self,
        image_dir: str,
        num_samples: int = 2000,
        transform = None,
        icon_indices: Optional[List[int]] = None,
        seed: int = 0
    ):
        super().__init__()
        self.image_dir = image_dir
        self.num_samples = num_samples
        self.transform = transform or transforms.ToTensor()

        if icon_indices is None:
            icon_indices = range(100)

        # load icons via helper
        self.icons = load_icons(self.image_dir, icon_indices)

        rng = random.Random(seed)
        self._samples: list[tuple[tuple[int,int], tuple[int,int],
                                  tuple[int,int], int]] = []
        for _ in range(num_samples):
            src_same   = rng.random() < 0.5
            source     = self._make_pair(rng, src_same)
            pair_same  = self._make_pair(rng, True)
            pair_diff  = self._make_pair(rng, False)

            # decide which target is correct
            if rng.random() < 0.5:          # T1 correct
                t1, t2, label = (pair_same, pair_diff, 0) if src_same else (pair_diff, pair_same, 0)
            else:                           # T2 correct
                t1, t2, label = (pair_diff, pair_same, 1) if src_same else (pair_same, pair_diff, 1)

            self._samples.append((source, t1, t2, label))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        On-the-fly RMTS sampling.
        """
        source_pair, t1_pair, t2_pair, label = self._samples[idx]
        s1, s2 = source_pair
        t1a, t1b = t1_pair
        t2a, t2b = t2_pair

        # apply transforms
        s1 = self.transform(s1)
        s2 = self.transform(s2)
        t1a = self.transform(t1a)
        t1b = self.transform(t1b)
        t2a = self.transform(t2a)
        t2b = self.transform(t2b)

        return (s1, s2, t1a, t1b, t2a, t2b), label

    def _make_pair(self, rng, is_same: bool):
        """
        Return (iconA, iconB) for either same or different.
        """
        if is_same:
            idx = rng.randrange(len(self.icons))
            return (self.icons[idx], self.icons[idx])
        else:
            idxA = rng.randrange(len(self.icons))
            while True:
                idxB = rng.randrange(len(self.icons))
                if idxB != idxA:
                    break
            return (self.icons[idxA], self.icons[idxB])

def visualize_icon_rmts_sample(dataset, idx=None):
    """
    Visualizes a sample from IconRelMatchToSampleDataset.
    
    Args:
        dataset: an instance of IconRelMatchToSampleDataset.
        idx: Optional sample index. If None, one sample is chosen randomly.
    """
    if idx is None:
        idx = random.randint(0, len(dataset) - 1)
        
    # Retrieve sample: (s1, s2, t1a, t1b, t2a, t2b), label 
    (s1, s2, t1a, t1b, t2a, t2b), label = dataset[idx]
    
    # Convert tensors to numpy arrays. For RGB images, permute dims.
    def to_numpy(img):
        if img.shape[0] != 1:
            return img.permute(1, 2, 0).cpu().numpy(), None
        else:
            return img.squeeze().cpu().numpy(), "gray"
    
    s1_np, cmap_s1 = to_numpy(s1)
    s2_np, cmap_s1 = to_numpy(s2)
    t1a_np, cmap_s1 = to_numpy(t1a)
    t1b_np, cmap_s1 = to_numpy(t1b)
    t2a_np, cmap_s1 = to_numpy(t2a)
    t2b_np, cmap_s1 = to_numpy(t2b)
    
    plt.figure(figsize=(12, 8))
    
    # Row 1: Source pair
    plt.subplot(3, 2, 1)
    plt.imshow(s1_np, cmap=cmap_s1)
    plt.title('Source Image 1')
    plt.axis('off')
    
    plt.subplot(3, 2, 2)
    plt.imshow(s2_np, cmap=cmap_s1)
    plt.title('Source Image 2')
    plt.axis('off')
    
    # Row 2: First target pair
    plt.subplot(3, 2, 3)
    plt.imshow(t1a_np, cmap=cmap_s1)
    plt.title('Target 1, Image 1')
    plt.axis('off')
    
    plt.subplot(3, 2, 4)
    plt.imshow(t1b_np, cmap=cmap_s1)
    plt.title('Target 1, Image 2')
    plt.axis('off')
    
    # Row 3: Second target pair
    plt.subplot(3, 2, 5)
    plt.imshow(t2a_np, cmap=cmap_s1)
    plt.title('Target 2, Image 1')
    plt.axis('off')
    
    plt.subplot(3, 2, 6)
    plt.imshow(t2b_np, cmap=cmap_s1)
    plt.title('Target 2, Image 2')
    plt.axis('off')
    
    plt.suptitle(f"RMTS Sample - Label (0: Target1 correct, 1: Target2 correct): {label}")
    plt.tight_layout()
    plt.show()


# For the RMTS dataset:
rmts_dataset = IconRelMatchToSampleDataset(
    image_dir="./imgs/",
    num_samples=10,
    transform=COMMON_TRANSFORM
)

visualize_icon_rmts_sample(rmts_dataset)
plt.close()
## Feature Extractor (CNN)
class FeatureExtractor(nn.Module):
    """
    CNN feature extractor to convert images to feature embeddings.
    
    Input: Image tensor of shape [batch_size, channels, height, width]
    Output: Feature embedding of shape [batch_size, embedding_dim]
    """
    def __init__(self, input_channels: int = 1, embedding_dim: int = 64):
        super(FeatureExtractor, self).__init__()
        self.embedding_dim = embedding_dim

        # CNN architecture
        # First convolution: 1 -> 16 channels, kernel=5, stride=2, padding=2
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=2, padding=2)
        # Second convolution: 16 -> 32 channels, kernel=5, stride=2, padding=2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        # After conv2, a 64x64 input becomes approximately 16x16
        # Fully connected layer: flattens 32 channels of 16x16 spatial map → embedding_dim vector
        self.fc = nn.Linear(32 * 8 * 8, embedding_dim)

        self.flatten = nn.Flatten()

        # param initialization
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.conv1.bias)
        
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.conv2.bias)
        
        nn.init.kaiming_uniform_(self.fc.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of feature extractor.
        
        Args:
            x: Input image tensor [batch_size, channels, height, width]
        
        Returns:
            Feature embedding [batch_size, embedding_dim]
        """
        # First conv + ReLU: reduces resolution from 64x64 to roughly 32x32
        x = F.relu(self.conv1(x))
        
        # Second conv + ReLU: reduces further to roughly 16x16
        x = F.relu(self.conv2(x))
        
        # Flatten the feature map (32 channels × 16 × 16 spatial) into a vector
        x = self.flatten(x)
        
        # project to embedding dimension
        x = self.fc(x)
        
        return x
# --------------------------------------------------------------------- #
#  BASELINE (no Kuramoto): CNN encoder ➜ 2‑layer MLP                    #
# --------------------------------------------------------------------- #
class BaselineSameDiffModel(nn.Module):
    """
    Two images → shared FeatureExtractor → concat → 2‑layer MLP.
    Returns (logits, None) so the train‑loop keeps working.
    """
    def __init__(self, 
                 input_channels: int = 1,
                 embedding_dim: int = 64,
                 oscillator_dim: int = 2,
                ):
        super().__init__()
        self.feature_extractor = FeatureExtractor(
            input_channels=input_channels,
            embedding_dim=embedding_dim,
        )

        # Project to R^{D×N} space (similar to c_terms and implicitly osc states)
        self.projector = nn.Linear(embedding_dim, embedding_dim * oscillator_dim)
        self.embedding_dim = embedding_dim
        self.oscillator_dim = oscillator_dim
        
        # coherence measurement
        self.coherence_measurement = CoherenceMeasurement(embedding_dim, oscillator_dim)
        
        # classifier on coherence vector
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
        e1 = self.feature_extractor(img1)          # [B,D]
        e2 = self.feature_extractor(img2)          # [B,D]
    
        # Project to R^{D×N}
        p1 = self.projector(e1).view(-1, self.embedding_dim, self.oscillator_dim)
        p2 = self.projector(e2).view(-1, self.embedding_dim, self.oscillator_dim)
        
        # Normalize (like oscillators)
        p1 = F.normalize(p1, p=2, dim=2) 
        p2 = F.normalize(p2, p=2, dim=2)
        
        # Compute coherence directly
        coherence_vector = self.coherence_measurement(p1, p2)
        
        # Classify
        logit = self.classifier(coherence_vector)
        
        # Return a structure compatible with the Kuramoto model's output
        # - p1, p2 as final oscillator states
        # - [p1], [p2] as trajectories (just single-step here)
        # - None for energy values
        # - None for c_terms
        return logit, (p1, p2, [p1], [p2], None, None, None)

    # ----- helper needed by the RMTS pipeline --------------------------
    @torch.no_grad()
    def get_coherence_vector(self, imgA: torch.Tensor, imgB: torch.Tensor):
        """Return D-dim coherence vector for an (A,B) pair."""
        self.eval()
        eA = self.feature_extractor(imgA)
        eB = self.feature_extractor(imgB)
        
        # Project to R^{D×N}
        pA = self.projector(eA).view(-1, self.embedding_dim, self.oscillator_dim)
        pB = self.projector(eB).view(-1, self.embedding_dim, self.oscillator_dim)
        
        # Normalize
        pA = F.normalize(pA, p=2, dim=2)
        pB = F.normalize(pB, p=2, dim=2)
        
        # Compute coherence
        coherence_vector = self.coherence_measurement(pA, pB)
        
        return coherence_vector

class BaselineRMTSClassifier(nn.Module):
    """
    Three coherence vectors (D each) → 2‑layer MLP → 2 logits.
    """
    def __init__(self, coherence_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(3 * coherence_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, src_vec, t1_vec, t2_vec):
        x = torch.cat([src_vec, t1_vec, t2_vec], dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
## Kuramoto Oscillator Network
@torch.jit.script
def compute_within_coupling_optimized(oscillators: torch.Tensor, J_in: torch.Tensor) -> torch.Tensor:
    """
    Compute within-object coupling forces with optimized implementation.
    
    Mathematical formula:
    F^{IN}_{i,d}(t) = \sum_{d'=1}^D J^{IN}_{d,d'} x_{i,d'}(t)
    
    Where:
    - i is the object index
    - d, d' are feature dimensions
    - J^{IN}_{d,d'} defines coupling between features within same object
    
    Args:
        oscillators: Oscillator states [batch_size, D, N]
        J_in: Within-object coupling matrix [D, D, N, N]
    
    Returns:
        Within-object coupling forces [batch_size, D, N]
    """
    # Using einsum for efficient computation without intermediate tensors
    # This directly implements the sum over d' dimension
    # "ijkl,bjl->bik" means:
    # - i: first D dimension (output feature)
    # - j: second D dimension (sum over this - corresponds to d')
    # - k,l: oscillator dimensions N
    # - b: batch dimension
    return torch.einsum("ijkl,bjl->bik", J_in, oscillators)


@torch.jit.script
def compute_between_coupling_optimized(oscillators_other: torch.Tensor, J_out: torch.Tensor) -> torch.Tensor:
    """
    Compute between-object coupling forces with optimized implementation.
    
    Mathematical formula:
    F^{OUT}_{i,d}(t) = \sum_{d'=1}^D J^{OUT}_{d,d'} x_{3-i,d'}(t)
    
    Where:
    - i is the object index (3-i is the other object)
    - d, d' are feature dimensions
    - J^{OUT}_{d,d'} defines coupling between features of different objects
    
    Args:
        oscillators_other: Other object's oscillator states [batch_size, D, N]
        J_out: Between-object coupling matrix [D, D, N, N]
    
    Returns:
        Between-object coupling forces [batch_size, D, N]
    """
    return torch.einsum("ijkl,bjl->bik", J_out, oscillators_other)


class KuramotoOscillatorNetwork(nn.Module):
    """
    Implements Kuramoto dynamics on feature embeddings for relational learning.
    
    Takes feature embeddings from TWO (2) objects and applies Kuramoto
    dynamics to create oscillator synchronization patterns that reflect the
    relationship between the objects.
    """
    def __init__(
        self, 
        embedding_dim: int, 
        oscillator_dim: int, 
        num_steps: int = 10, 
        step_size: float = 0.1,
        use_omega=True, 
        omega_kappa=5.0,
        disable_between: bool = False,
        symmetric_j: bool = False,
    ):
        """
        Initialize the Kuramoto Oscillator Network.
        
        Args:
            embedding_dim: Dimension D of feature embeddings
            oscillator_dim: Dimension N of each oscillator
            num_steps: Number of Kuramoto update steps (T)
            step_size: Step size for discrete updates (gamma)
        """
        super(KuramotoOscillatorNetwork, self).__init__()
        self.embedding_dim = embedding_dim    # D
        self.oscillator_dim = oscillator_dim  # N
        self.num_steps = num_steps            # T
        self.step_size = step_size            # gamma
        self.symmetric_j = symmetric_j
        
        # # natural frequency matrices set to 0 (for now)
        self.use_omega = use_omega
        if self.use_omega:
            Omega = self.sample_fixed_omegas(embedding_dim,
                                    oscillator_dim,
                                    kappa=omega_kappa,
                                    max_freq=1.0,
                                    device='cpu')
            # register as *buffer* (no grad, moves with .to(device))
            self.register_buffer("Omega", Omega)
        else:
            self.register_buffer("Omega", torch.zeros(
                embedding_dim, oscillator_dim, oscillator_dim))

        # # init parameters
        self.W_d = nn.Parameter(torch.empty(embedding_dim, oscillator_dim))
        self.b_d = nn.Parameter(torch.empty(embedding_dim, oscillator_dim))
        nn.init.xavier_uniform_(self.W_d)
        nn.init.zeros_(self.b_d)

        # ---- dense parameters ------------------------------------------------
        self.J_in = nn.Parameter(torch.empty(embedding_dim,
                                             embedding_dim,
                                             oscillator_dim,
                                             oscillator_dim))
        nn.init.xavier_uniform_(self.J_in)
        if self.symmetric_j:
            with torch.no_grad():
                self.J_in.copy_(self._symmetrize(self.J_in))

        self.disable_between = disable_between
        if self.disable_between:
            self.register_buffer("J_out",
                                 torch.zeros_like(self.J_in, requires_grad=False))
        else:
            self.J_out = nn.Parameter(torch.empty_like(self.J_in))
            nn.init.xavier_uniform_(self.J_out)
            if self.symmetric_j and not self.disable_between:
                with torch.no_grad():
                    self.J_out.copy_(self._symmetrize(self.J_out))


    # ------------------------------------------------------------------ #
    #  Symmetric view (dense-share): no extra memory, fast               #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _symmetrize(J: torch.Tensor) -> torch.Tensor:
        # sym across feature dims (d,d′) then oscillator dims (n,n′)
        J = 0.5 * (J + J.permute(1, 0, 2, 3))
        return 0.5 * (J + J.permute(0, 1, 3, 2))

    @property
    def J_in_sym(self) -> torch.Tensor:
        return self._symmetrize(self.J_in) if self.symmetric_j else self.J_in

    @property
    def J_out_sym(self) -> torch.Tensor:
        if self.disable_between:
            return self.J_out          # zero buffer
        return self._symmetrize(self.J_out) if self.symmetric_j else self.J_out
                
    def compute_conditional_stimulus(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute conditional stimulus (data terms) from embeddings.
        
        Args:
            embeddings: Feature embeddings [batch_size, D]
        
        Returns:
            Conditional stimulus terms [batch_size, D, N]
        """
        batch_size, D = embeddings.shape

        # # Reshape for broadcasting to match required [batch_size, D, N] shape
        # c_terms = embeddings.unsqueeze(2).expand(-1, -1, self.oscillator_dim)

        
        # expand embeddings to [batch_size, D, 1]
        feature_values = embeddings.unsqueeze(2)
        
        # multiply by weights to get [batch_size, D, N]
        # c_{i,d} = W_d * E_i[d] + b_d
        c_terms = feature_values * self.W_d.unsqueeze(0) + self.b_d.unsqueeze(0)
        return c_terms   
    
    def initialize_oscillators(self, c_terms: torch.Tensor) -> torch.Tensor:
        """
        Initialise x_{i,d}(0) with a small, *feature‑aligned* phase vector:
            x = 𝒩(0,σ²)  +  α · c_terms
        then project back to the unit sphere.
        """
        sigma      = 0.01                         # noise std‑dev
        alpha      = 1.0                          # stimulus scaling factor (tuneable)
        noise  = torch.randn_like(c_terms) * sigma
        oscillators = noise + alpha * c_terms     # feature‑aligned start
        oscillators = F.normalize(oscillators, p=2, dim=2)
        return oscillators

    @staticmethod
    def sample_fixed_omegas(D: int, N: int,
                            kappa: float = 5.0,
                            max_freq: float = 1.0,
                            device=None) -> torch.Tensor:
        """
        Returns Ω  ∈  ℝ[D, N, N]   (skew‑symmetric, no grad)
    
        - scalar frequency ω ~ von Mises(μ=0, κ)   (circular → centred at 0)
        - rotation AXIS is random unit vector on S^{N-1}
        """
        vm = torch.distributions.VonMises(
            torch.zeros(D, device=device), torch.full((D,), kappa, device=device)
        )

        omega = vm.sample() * max_freq                               # [D]
        if N == 2:
            Omega = torch.zeros(D, 2, 2, device=device)
            Omega[:, 0, 1] =  omega
            Omega[:, 1, 0] = -omega
            return Omega
        
        u = F.normalize(torch.randn(D, N, device=device), dim=1) # [D,N]
        # Skew‑symm: Ω_d = ω_d (u_d ⊗ û_d – û_d ⊗ u_d)
        Omega = omega.view(-1, 1, 1) * (u.unsqueeze(2) - u.unsqueeze(1)) / 2
        return Omega
    
    def project_to_tangent_space(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Project vector y onto the tangent space at x.
        
        Args:
            x: Base point on the sphere [batch_size, D, N]
            y: Vector to project [batch_size, D, N]
        
        Returns:
            Projection of y onto tangent space at x [batch_size, D, N]
        """
        # calculate the dot product <x, y> along the last dimension
        dot_product = torch.sum(x * y, dim=2, keepdim=True)
        
        # projection formula: y - <x, y>x
        return y - dot_product * x
    
    def compute_within_coupling(self, oscillators: torch.Tensor) -> torch.Tensor:
        """
        Compute within-object coupling forces.
        
        Args:
            oscillators: Oscillator states [batch_size, D, N]
        
        Returns:
            Within-object coupling forces [batch_size, D, N]
        """
        J_in = self.J_in_sym
        return compute_within_coupling_optimized(oscillators, J_in)
    
    
    def compute_between_coupling(self, oscillators_other: torch.Tensor) -> torch.Tensor:
        """
        Compute between-object coupling forces.
        
        Args:
            oscillators_other: OTHER object's oscillators [batch_size, D, N]
        
        Returns:
            Between-object coupling forces for first object [batch_size, D, N]
        """
        if self.disable_between:
            # same shape as “within” result, but all zeros
            return torch.zeros_like(oscillators_other) 
        J_out = self.J_out_sym
        return compute_between_coupling_optimized(oscillators_other, J_out)
        

    def calculate_energy(self, 
                        x1: torch.Tensor, 
                        x2: torch.Tensor,
                        c1: torch.Tensor, 
                        c2: torch.Tensor) -> torch.Tensor:
        """
        Calculate Lyapunov energy of the Kuramoto system using the formula:
        E = -1/2 * sum_{i,j} x_i^T J_{ij} x_j - sum_i c_i^T x_i
        
        Args:
            x1, x2: Oscillator states [batch_size, D, N]
            c1, c2: Conditional stimulus terms [batch_size, D, N]
        
        Returns:
            Energy values for batch [batch_size]
        """
        # Get the appropriate J matrices
        J_in = self.J_in_sym

        # ---- Coupling energy --------------------------------------------------
        # Within‑object terms
        E_w1  = torch.einsum('bdn,ijkl,bjl->b',
                             x1, J_in, x1)          # x1ᵀ J_in x1
        E_w2  = torch.einsum('bdn,ijkl,bjl->b',
                             x2, J_in, x2)
        # Between‑object (double‑count avoided by *no* factor ½ here)
        if self.disable_between:
            E_bt = 0.0
        else:
            # x1ᵀ J_out x2 + x2ᵀ J_out x1
            J_out = self.J_out_sym
            E_bt_1to2 = torch.einsum('bdn,ijkl,bjl->b', x1, J_out, x2)
            E_bt_2to1 = torch.einsum('bdn,ijkl,bjl->b', x2, J_out, x1)
            E_bt = E_bt_1to2 + E_bt_2to1
    
        E_cpl = - 0.5 * (E_w1 + E_w2 + E_bt)              # final coupling energy
    
        # ---- Stimulus alignment ----------------------------------------------
        E_stim = -(x1 * c1).sum((1, 2)) - (x2 * c2).sum((1, 2))
    
        # ---- Total -----------------------------------------------------------
        return E_cpl + E_stim        # shape [batch]
    
    
    def kuramoto_step(self, 
                     oscillators1: torch.Tensor, 
                     oscillators2: torch.Tensor,
                     c_terms1: torch.Tensor, 
                     c_terms2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one step of Kuramoto dynamics update.
        
        Args:
            oscillators1: First object's oscillators [batch_size, D, N]
            oscillators2: Second object's oscillators [batch_size, D, N]
            c_terms1: First object's conditional stimulus terms [batch_size, D, N]
            c_terms2: Second object's conditional stimulus terms [batch_size, D, N]
        
        Returns:
            Updated oscillators for both objects
        """
        # natural frequency  Ω x
        if self.use_omega:
            nat1 = torch.einsum('dij,bdj->bdi', self.Omega, oscillators1)
            nat2 = torch.einsum('dij,bdj->bdi', self.Omega, oscillators2)
        else:
            nat1 = nat2 = 0.
        
        # compute within-object coupling forces
        f_in1 = self.compute_within_coupling(oscillators1)
        f_in2 = self.compute_within_coupling(oscillators2)
        
        # compute between-object coupling forces
        if self.disable_between:
            f_out1 = f_out2 = 0.0
        else:
            f_out1 = self.compute_between_coupling(oscillators2)
            f_out2 = self.compute_between_coupling(oscillators1)
        
        # compute total forces
        total_force1 = c_terms1 + f_in1 + f_out1
        total_force2 = c_terms2 + f_in2 + f_out2
        
        # project onto tangent space
        delta_x1 = nat1 + self.project_to_tangent_space(oscillators1, total_force1)
        delta_x2 = nat2 + self.project_to_tangent_space(oscillators2, total_force2)
        
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            # apply update with step size & normalize to maintain unit norm
            new_oscillators1 = F.normalize(oscillators1 + self.step_size * delta_x1, 
                                           p=2, 
                                           dim=2)
            new_oscillators2 = F.normalize(oscillators2 + self.step_size * delta_x2, 
                                           p=2, 
                                           dim=2)
        
        return new_oscillators1, new_oscillators2
    
    def run_dynamics(self, 
                    oscillators1: torch.Tensor, 
                    oscillators2: torch.Tensor,
                    c_terms1: torch.Tensor, 
                    c_terms2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Run Kuramoto dynamics for T steps.
        
        Args:
            oscillators1: Initial oscillators for first object [batch_size, D, N]
            oscillators2: Initial oscillators for second object [batch_size, D, N]
            c_terms1: Conditional stimulus terms for first object [batch_size, D, N]
            c_terms2: Conditional stimulus terms for second object [batch_size, D, N]
        
        Returns:
            Final oscillator states (time T), trajectories, and energy values
        """
        current_oscillators1 = oscillators1
        current_oscillators2 = oscillators2
        
        # store all states for viz
        osc_trajectory1 = [current_oscillators1]
        osc_trajectory2 = [current_oscillators2]

        # Initialize energy tracking
        energy_values = []
        
        # Calculate initial energy
        initial_energy = self.calculate_energy(
            current_oscillators1, current_oscillators2, 
            c_terms1, c_terms2
        )
        energy_values.append(initial_energy)
        
        # num_steps == T
        for t in range(self.num_steps):
            current_oscillators1, current_oscillators2 = self.kuramoto_step(
                current_oscillators1, current_oscillators2, 
                c_terms1, c_terms2
            )
            
            osc_trajectory1.append(current_oscillators1)
            osc_trajectory2.append(current_oscillators2)

            # ADDED: Calculate energy after this step
            step_energy = self.calculate_energy(
                current_oscillators1, current_oscillators2, 
                c_terms1, c_terms2
            )
            energy_values.append(step_energy)
        
        # return final states and all intermediate states
        return current_oscillators1, current_oscillators2, osc_trajectory1, osc_trajectory2, energy_values
    
    def forward(self, 
               embeddings1: torch.Tensor, 
               embeddings2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass through the Kuramoto Oscillator Network.
        
        Args:
            embeddings1: Feature embedding of first object [batch_size, D]
            embeddings2: Feature embedding of second object [batch_size, D]
        
        Returns:
            Final oscillator states and all intermediate states
        """
        # compute data terms
        c_terms1 = self.compute_conditional_stimulus(embeddings1)
        c_terms2 = self.compute_conditional_stimulus(embeddings2)
        
        # initialize oscillators
        oscillators1 = self.initialize_oscillators(c_terms1)
        oscillators2 = self.initialize_oscillators(c_terms2)
        
        # run kuramoto dynamics
        final_osc1, final_osc2, osc_trajectory1, osc_trajectory2, energy_values = self.run_dynamics(
            oscillators1, oscillators2, 
            c_terms1, c_terms2
        )
        
        return (final_osc1, final_osc2, 
                osc_trajectory1, osc_trajectory2, 
                energy_values, 
                c_terms1, c_terms2)
## Coherence Measurement
class CoherenceMeasurement(nn.Module):
    """
    Takes final oscillator states and computes coherence measures across multiple objects
    using the norm of the sum for each feature dimension.
    """
    def __init__(
        self, 
        embedding_dim: int, 
        oscillator_dim: int
    ):
        """        
        Args:
            embedding_dim: Dimension D of feature embeddings
            oscillator_dim: Dimension N of each oscillator
        """
        super(CoherenceMeasurement, self).__init__()
        self.embedding_dim = embedding_dim    # D
        self.oscillator_dim = oscillator_dim  # N
    
    def compute_coherence(self, *oscillators_list) -> torch.Tensor:
        """
        Compute coherence features using the formula:
        ρ = (ρ_1, ..., ρ_D), where ρ_d = ‖∑_{i=1}^n x_{i,d}(T)‖_2
            
        Args:
            *oscillators_list: Variable number of oscillator tensors, 
                              each with shape [batch_size, D, N]
        
        Returns:
            Coherence vector [batch_size, D]
        """
        # Sum all oscillators
        # Each oscillator has shape [batch_size, D, N]
        summed_oscillators = sum(oscillators_list)  # Shape: [batch_size, D, N]
        
        # Compute L2 norm along the oscillator dimension (N)
        # Shape: [batch_size, D]
        coherence = torch.norm(summed_oscillators, p=2, dim=2)
        
        return coherence
    
    def forward(self, *oscillators_list) -> torch.Tensor:
        """
        Args:
            *oscillators_list: Variable number of oscillator tensors,
                              each with shape [batch_size, D, N]
        
        Returns:
            Coherence vector [batch_size, D]
        """
        return self.compute_coherence(*oscillators_list)
## Classification (MLP)
class ClassificationMLP(nn.Module):
    """
    MLP for same/different classification.
    
    Takes coherence feature vector and outputs probability of "same" relation.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        """
        Args:
            input_dim: Dimension of the coherence feature vector
            hidden_dim:  hidden layer dimension
        """
        super(ClassificationMLP, self).__init__()
        
        # MLP layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.
        
        Args:
            x: Coherence feature vector [batch_size, input_dim]
        
        Returns:
            Probability of "same" relation [batch_size, 1]
        """
        # first hidden layer with relu activation
        x = F.relu(self.fc1(x))
        
        # second hidden layer with relu activation
        logit = self.fc2(x)
        
        return logit
class RMTSClassifier(nn.Module):
    """
    Tiny MLP that takes 3 coherence vectors:
       - source_vec in R^D
       - target1_vec in R^D
       - target2_vec in R^D
    concatenates them to a 3D vector,
    and outputs 2 logits => {T1 correct, T2 correct}.
    """
    def __init__(self, coherence_dim=64, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(3 * coherence_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)  # 2 classes => T1=correct or T2=correct

    def forward(self, source_vec, t1_vec, t2_vec):
        # shape [batch_size, 3D]
        x = torch.cat([source_vec, t1_vec, t2_vec], dim=1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)  # shape [batch_size, 2]
        return logits

## Complete Relational Model
class KuramotoRelationalModel(nn.Module):
    """
    Complete Kuramoto-based relational learning model.
    
    Combines all components: feature extraction, oscillator network,
    coherence measurement, and classification.
    """
    def __init__(self, 
                 input_channels: int = 1, 
                 embedding_dim: int = 64, 
                 oscillator_dim: int = 4, 
                 num_steps: int = 10, 
                 step_size: float = 0.1,
                 mlp_hidden_dim: int = 64,
                 use_omega: bool = True,
                 omega_kappa: float = 1.0,
                 disable_between: bool = False,
                 symmetric_j: bool = False
    ):
        """
        Args:
            input_channels: Number of input image channels
            embedding_dim: Dimension D of feature embeddings
            oscillator_dim: Dimension N of each oscillator
            num_steps: Number of Kuramoto update steps T
            step_size: Step size for Kuramoto updates (gamma)
            num_coherence_features: Number of coherence features J
            mlp_hidden_dim: hidden layer dimension of classification MLP
        """
        super(KuramotoRelationalModel, self).__init__()
        
        # feature extractor
        self.feature_extractor = FeatureExtractor(
            input_channels=input_channels,
            embedding_dim=embedding_dim
        )
        
        # kuramoto oscillator network
        self.oscillator_network = KuramotoOscillatorNetwork(
            embedding_dim=embedding_dim,
            oscillator_dim=oscillator_dim,
            num_steps=num_steps,
            step_size=step_size,
            use_omega=use_omega,
            omega_kappa=omega_kappa,
            disable_between=disable_between,
            symmetric_j=symmetric_j,
        )
        
        # coherence measurement
        self.coherence_measurement = CoherenceMeasurement(
            embedding_dim=embedding_dim,
            oscillator_dim=oscillator_dim,
        )
        
        # classification mlp
        self.classifier = ClassificationMLP(
            input_dim=embedding_dim,
            hidden_dim=mlp_hidden_dim,
        )
    
    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]
    ]:
        """
        Forward pass through the complete model.
        
        Args:
            image1: First image tensor [batch_size, channels, height, width]
            image2: Second image tensor [batch_size, channels, height, width]
        
        Returns:
            Tuple containing:
            - relation_prob: Probability of "same" relation from Kuramoto model [batch_size, 1]
            - oscillator_states: Tuple of (final_osc1, final_osc2, osc_trajectory1, osc_trajectory2)
        """
        # extract feature embeddings
        embedding1 = self.feature_extractor(image1)
        embedding2 = self.feature_extractor(image2)
        
        # run kuramoto dynamics
        final_osc1, final_osc2, osc_trajectory1, osc_trajectory2, energy_values, c1, c2 = self.oscillator_network(embedding1, embedding2)
        
        # compute coherence measures
        coherence_vector = self.coherence_measurement(final_osc1, final_osc2)
        
        # classify based on coherence
        relation_prob = self.classifier(coherence_vector)
        
        return relation_prob, (final_osc1, final_osc2, 
                               osc_trajectory1, osc_trajectory2, 
                               energy_values,
                              c1, c2)

    def get_coherence_vector(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Use the existing pipeline to return the D-dimensional coherence vector 
        for the given pair (img1, img2), instead of the final classification probability.
        """
        self.eval()  # ensure eval mode; no dropout, no BN updates
        with torch.no_grad():
            # Step 1: extract embeddings
            emb1 = self.feature_extractor(img1)   # [batch_size, D]
            emb2 = self.feature_extractor(img2)   # [batch_size, D]
            # Step 2: run oscillator network
            final_osc1, final_osc2, *_ = self.oscillator_network(emb1, emb2)
            # Step 3: measure coherence
            coherence_vec = self.coherence_measurement(final_osc1, final_osc2)  # [batch_size, D]
        return coherence_vec
## Visualization Helpers: Energy
def collect_energy_values(model, sample_batch, device):
    """Collects energy values with separate tracking for same/different pairs."""
    model.eval()
    
    with torch.no_grad():
        # Unpack batch
        (img1, img2), labels = sample_batch
        img1, img2 = img1.to(device), img2.to(device)
        labels = labels.to(device)
        
        # Forward pass to get energy values
        _, oscillator_data = model(img1, img2)
        energy_values = oscillator_data[4]  # Extract energy values
        
        # Reorganize energy values for visualization
        formatted_energy = {
            # Overall average energy across all samples
            "all": {step: energy.mean().item() for step, energy in enumerate(energy_values)}
        }
        
        # Separate energy for same/different pairs
        same_mask = (labels == 1)
        diff_mask = (labels == 0)
        
        if same_mask.any():
            formatted_energy["same"] = {
                step: energy[same_mask].mean().item() 
                for step, energy in enumerate(energy_values)
            }
            
        if diff_mask.any():
            formatted_energy["different"] = {
                step: energy[diff_mask].mean().item() 
                for step, energy in enumerate(energy_values)
            }
    
    return formatted_energy
plt.style.use("seaborn-v0_8-paper")

def display_energy_plot(energy_values, epoch, ax=None, show_details=False):
    """
    Plot energy evolution for one epoch with an academic paper style.
    
    Parameters
    ----------
    energy_values : dict[str, dict[int, float]]
        Outer keys: "all", "same", "different".
        Inner dict: {step: energy}.
    epoch : int
        Zero-based epoch index (will be printed 1-based).
    ax : matplotlib.axes.Axes | None
        If given, plot there; otherwise create a new figure/axes.
    show_details : bool, default=False
        Whether to show additional details like start/end values and zero line.
    """
    own_ax = ax is None
    if own_ax:
        fig, ax = plt.subplots(figsize=(10, 5))
    
    # Use consistent colors that match your other plots
    RED = "#c86460"     # CSS name: "darkred"
    TEAL = "#2a9da3"   # CSS name: "darkgreen"
    BLACK = "#000000"    # Third color for the third category
    
    colors = {
        "all": BLACK,
        "same": TEAL,
        "different": RED
    }
    
    # Plot each category with consistent line styling (no markers)
    for cat, data in energy_values.items():
        steps, energies = zip(*sorted(data.items()))
        ax.plot(
            steps,
            energies,
            label=cat.lower() + " pairs",
            color=colors.get(cat, "gray"),
            lw=2.0,  # Consistent line width
        )
    
    # Optional detailed information that can be turned off for publication
    if show_details:
        # Subtle zero reference line
        ax.axhline(0, lw=0.5, color="0.8", alpha=0.3)
        
        # Add convergence information if "all" category exists
        if "all" in energy_values:
            steps, energies = zip(*sorted(energy_values["all"].items()))
            drop = energies[0] - energies[-1]
            rate = drop / (steps[-1] - steps[0]) if steps[-1] != steps[0] else 0
            
            # Add subtle annotations in matching colors
            color = colors["all"]
            ax.annotate(f"{energies[0]:.2f}",
                        (steps[0], energies[0]),
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=8, color=color, alpha=0.7)
            
            ax.annotate(f"{energies[-1]:.2f}",
                        (steps[-1], energies[-1]),
                        xytext=(-5, -10), textcoords="offset points",
                        fontsize=8, color=color, alpha=0.7, ha="right")
            
            # Add convergence information to title
            title_extra = f"  –  Δ = {drop:.2f} (rate {rate:.3f}/step)"
        else:
            title_extra = ""
    else:
        title_extra = ""
    
    # Set labels and title
    ax.set_xlabel("kuramoto step")
    ax.set_ylabel("energy")

    label = "init" if epoch < 0 else f"epoch {epoch+1}"
    ax.set_title(f"{label}{title_extra}")
    
    # Legend placement consistent with your other plots
    ax.legend(loc="lower left", frameon=True, edgecolor='lightgray', framealpha=0.8)
    
    # No grid
    ax.grid(False)
    
    if own_ax:
        plt.tight_layout()
        plt.close()
        return fig   # to log to wandb etc.
def build_energy_animation(all_epoch_energies,
                           categories=("all", "same", "different"),
                           fname="energy_evolution.gif",
                           fps=2):
    """
    Create and save a GIF that re-draws the energy curve for every epoch
    (including the pre-training “init” frame if present).

    Parameters
    ----------
    all_epoch_energies : dict[int → dict[str → dict[int → float]]]
        Epoch index → { category → { step → energy } }
        If the dict contains key -1, that frame is treated as the initial
        un-trained network.
    categories : iterable[str]
        Which energy traces to draw.
    """
    # ------------------------------------------------------------------ #
    # 1. static figure scaffolding                                       #
    # ------------------------------------------------------------------ #
    epochs = sorted(all_epoch_energies.keys())          # e.g. [-1, 0, 1, …]
    steps  = list(next(iter(all_epoch_energies.values()))["all"].keys())

    plt.style.use("seaborn-v0_8-paper")
    fig, ax = plt.subplots(figsize=(20, 5))

    # match colours used in display_energy_plot
    COLORS = {"all": "#000000", "same": "#2a9da3", "different": "#c86460"}
    lines  = {}
    for cat in categories:
        (ln,) = ax.plot([], [], lw=2.0,
                        color=COLORS.get(cat, "gray"),
                        label=f"{cat} pairs")
        lines[cat] = ln

    # fixed axes limits (so the frame doesn’t jump around)
    y_vals = []
    for ep in epochs:
        for cat in categories:
            if cat in all_epoch_energies[ep]:
                y_vals.extend(all_epoch_energies[ep][cat].values())
    ax.set_xlim(min(steps), max(steps))
    ax.set_ylim(min(y_vals) * 1.05, max(y_vals) * 1.05)

    ax.set_xlabel("kuramoto step")
    ax.set_ylabel("energy")
    ax.grid(alpha=.3)
    ax.legend(loc="lower left", frameon=True,
              edgecolor="lightgray", framealpha=0.8)
    ax.grid(False)

    # ------------------------------------------------------------------ #
    # 2. animation callbacks                                             #
    # ------------------------------------------------------------------ #
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
                ys = list(all_epoch_energies[epoch][cat].values())
                ln.set_data(steps, ys)
            else:                     # that category absent this epoch
                ln.set_data([], [])
        return tuple(lines.values())

    # ------------------------------------------------------------------ #
    # 3. make & save GIF                                                 #
    # ------------------------------------------------------------------ #
    ani = FuncAnimation(fig, update, frames=len(epochs),
                        init_func=init, blit=False,
                        interval=1000 / fps)

    Path(fname).parent.mkdir(parents=True, exist_ok=True)
    ani.save(fname, writer="pillow", fps=fps)
    plt.close(fig)

    return fname
## Visualization Helpers: Oscillators
def _best_grid(n: int) -> tuple[int, int]:
    """
    Return (rows, cols) giving the most square-ish grid ≥ n.
    Used to lay out per‑feature sub‑plots when N=2.          
    """
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols

def visualize_oscillators_2d(
    trajectory1,
    trajectory2,
    c_terms1,                 # shape [1,D,2]
    c_terms2,
    max_features: int = 16,
    trail_length: int = 5,
    animate: bool = True,
):
    """
    2‑D visualizer for oscillator trajectories in the complex plane.
    
    Parameters
    ----------
    trajectory1 : list
        First trajectory of oscillators.
    trajectory2 : list
        Second trajectory of oscillators.
    c_terms1 : torch.Tensor
        Coupling terms for first trajectory.
    c_terms2 : torch.Tensor
        Coupling terms for second trajectory.
    max_features : int, default=16
        Maximum number of features to display.
    trail_length : int, default=5
        Length of the trajectory trail to display.
    animate : bool, default=True
        Whether to create an animation or just the final frame.
        
    Returns
    -------
    matplotlib.animation.FuncAnimation or matplotlib.figure.Figure
        Animation or static figure of oscillator trajectories.
    """
    plt.style.use("seaborn-v0_8-paper")
    
    # Consistent colors with other plots
    DARK_BLUE = "#2a9da3"  
    DARK_RED = "#c86460" 
    BLACK = "#000000"
    
    num_steps = len(trajectory1)
    D = trajectory1[0].shape[1]
    features = min(D, max_features)
    rows, cols = _best_grid(features)
    
    # Pre‑convert to numpy (shape = [steps, D, 2])
    traj1_np = np.array([t[0].cpu().numpy()[:, :2] for t in trajectory1])
    traj2_np = np.array([t[0].cpu().numpy()[:, :2] for t in trajectory2])

    coh_np = 0.5 * (traj1_np + traj2_np) # [steps, D, 2]
    
    # Apply radius scaling to separate overlapping trajectories
    traj1_np_scaled = traj1_np * 0.97  # Inner orbit
    traj2_np_scaled = traj2_np * 1.03  # Outer orbit
    
    # ---   c‑terms (constant over time)   -------------------------------
    c1_np = c_terms1[0].cpu().numpy()[:, :2]          # [D,2]
    c2_np = c_terms2[0].cpu().numpy()[:, :2]
    
    # Create figure with proper spacing
    fig = plt.figure(figsize=(3*cols, 3*rows + 0.8), dpi=400)  # Add extra height for title/legend
    
    # Create a gridspec with extra space at the top
    #gs = fig.add_gridspec(rows, cols, top=0.85, wspace=0.1, hspace=0.1)
    gs = fig.add_gridspec(
        rows, cols,
        top=0.85,    # pull title up
        bottom=0.04, # push subplots down
        left=0.04,   # tighten left
        right=0.96,  # tighten right
        wspace=0.03,
        hspace=0.03
    )
    
    # Create axes from gridspec
    axes = []
    for i in range(rows):
        for j in range(cols):
            axes.append(fig.add_subplot(gs[i, j]))
    
    axes = np.array(axes)  # Convert to numpy array for convenience
    
    # Add a single legend to the figure with a box around it
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color=DARK_BLUE, label=r'$osc_{1}$',
                  markerfacecolor=DARK_BLUE, markersize=20, linewidth=0),
        plt.Line2D([0], [0], marker='o', color=DARK_RED, label=r'$osc_{2}$',
                  markerfacecolor=DARK_RED, markersize=20, linewidth=0),
        
        # c-term directions  (hollow triangles to distinguish them)
        plt.Line2D([0], [0], marker='^', color=DARK_BLUE, label=r'$c_{1}$',
                   markerfacecolor=DARK_BLUE, markersize=20, linewidth=0),
        plt.Line2D([0], [0], marker='^', color=DARK_RED,  label=r'$c_{2}$',
                   markerfacecolor=DARK_RED, markersize=20, linewidth=0),

        # coherence vector
        plt.Line2D([0], [0], marker='^', color=BLACK, label=r'$\rho_{d}$',
                   markerfacecolor=BLACK, markersize=20, linewidth=0),
    ]
    fig.legend(handles=legend_elements,
               loc="upper center",            # anchor legend to top-left
               bbox_to_anchor=(0.5, 0.98),
               ncol=5,                      # 3 columns  →  2 rows for 6 handles
               frameon=True,
               edgecolor="lightgray",
               framealpha=0.85,
               fontsize=28,
               borderaxespad=0.2)           # small padding around the box
    
    # Set a single title for the entire figure with time step
    time_title = fig.suptitle(f"t = 0", fontsize=28, y=0.89)
    
    # Hide unused panels cleanly
    for a in axes[features:]:
        a.set_visible(False)
    
    def _init():
        for i in range(features):
            a = axes[i]
            a.clear()
            a.set_aspect("equal", "box")
            a.set_xlim([-1.08, 1.08]) 
            a.set_ylim([-1.08, 1.08])
            
            # Remove all borders, axes and ticks
            a.axis('off')
            
            a.plot([], [])
        return axes
    
    def _update(frame):
        # Update the global time title (lowercase)
        time_title.set_text(f"t = {frame}")
        
        start = max(0, frame - trail_length + 1)
        for i in range(features):
            a = axes[i]
            a.clear()
            a.set_aspect("equal", "box")
            a.set_xlim([-1.08, 1.08]) 
            a.set_ylim([-1.08, 1.08])
            
            # Remove all borders, axes and ticks
            a.axis('off')
            
            # Draw unit circle with darker outline
            circ = plt.Circle((0, 0), 1, edgecolor="#888888", facecolor="none", 
                             alpha=0.7, linewidth=0.8)
            a.add_patch(circ)
            
            # # Optional: Add subtle circle indicators for the scaled radii
            # inner_circ = plt.Circle((0, 0), 0.9, edgecolor=DARK_BLUE, facecolor="none", 
            #                       alpha=0.2, linewidth=0.5, linestyle="--")
            # outer_circ = plt.Circle((0, 0), 1.1, edgecolor=DARK_RED, facecolor="none", 
            #                       alpha=0.2, linewidth=0.5, linestyle="--")
            # a.add_patch(inner_circ)
            # a.add_patch(outer_circ)

            coh_vec = coh_np[frame, i]                  # (x,y) for that feature
            a.quiver(0, 0,
                     coh_vec[0], coh_vec[1],
                     color=BLACK,      
                     angles="xy", scale_units="xy", scale=1,
                     width=0.007, alpha=0.5, zorder=4)
            
            # ------  c‑term arrows with darker styling  ------------------
            a.quiver(0, 0, 
                    c1_np[i, 0], c1_np[i, 1], 
                    color=DARK_BLUE, angles="xy", scale_units="xy", scale=1,
                    width=0.006, alpha=0.9)  # Darker vectors
            a.quiver(0, 0, 
                    c2_np[i, 0], c2_np[i, 1], 
                    color=DARK_RED, angles="xy", scale_units="xy", scale=1,
                    width=0.006, alpha=0.9)  # Darker vectors
            
            # Improved trajectory trails with thicker lines
            if frame > start:
                # Draw continuous trails with consistent styling and scaled radii
                a.plot(
                    traj1_np_scaled[start:frame+1, i, 0],
                    traj1_np_scaled[start:frame+1, i, 1], 
                    "-", color=DARK_BLUE, 
                    alpha=0.8, linewidth=2.0  # Thicker, more visible trails
                )
                a.plot(
                    traj2_np_scaled[start:frame+1, i, 0],
                    traj2_np_scaled[start:frame+1, i, 1], 
                    "-", color=DARK_RED, 
                    alpha=0.8, linewidth=2.0  # Thicker, more visible trails
                )
            
            # Current positions with translucent markers to see overlaps (also scaled)
            a.scatter(*traj1_np_scaled[frame, i], c=DARK_BLUE, marker="o", s=60, 
                     edgecolor="white", linewidth=0.5, zorder=5, alpha=0.8)
            a.scatter(*traj2_np_scaled[frame, i], c=DARK_RED, marker="o", s=60, 
                     edgecolor="white", linewidth=0.5, zorder=5, alpha=0.8)
        
        return axes

    ani=None
    if args.animate:
        ani = FuncAnimation(fig, _update,
                          frames=num_steps,
                          init_func=_init,
                          interval=400,
                          blit=False)
    _update(num_steps-1)
    return ani, fig
def visualize_oscillators_2d_overlay(
    trajectory1,
    trajectory2,
    c_terms1,                 # shape [1,D,2]
    c_terms2,
    max_features: int = 16,
    trail_length: int = 5,
    animate: bool = True,
):
    """
    2‑D visualizer for oscillator trajectories in the complex plane.
    All features are plotted on the same unit sphere with different radii.
    
    Parameters
    ----------
    trajectory1 : list
        First trajectory of oscillators.
    trajectory2 : list
        Second trajectory of oscillators.
    c_terms1 : torch.Tensor
        Coupling terms for first trajectory.
    c_terms2 : torch.Tensor
        Coupling terms for second trajectory.
    max_features : int, default=16
        Maximum number of features to display.
    trail_length : int, default=5
        Length of the trajectory trail to display.
    animate : bool, default=True
        Whether to create an animation or just the final frame.
        
    Returns
    -------
    matplotlib.animation.FuncAnimation or matplotlib.figure.Figure
        Animation or static figure of oscillator trajectories.
    """
    plt.style.use("seaborn-v0_8-paper")  # Academic paper style
    
    # Consistent colors with other plots
    DARK_BLUE = "#2a9da3"  
    DARK_RED = "#c86460" 
    BLACK = "#000000"
    
    num_steps = len(trajectory1)
    D = trajectory1[0].shape[1]
    features = min(D, max_features)
    
    # Pre‑convert to numpy (shape = [steps, D, 2])
    traj1_np = np.array([t[0].cpu().numpy()[:, :2] for t in trajectory1])
    traj2_np = np.array([t[0].cpu().numpy()[:, :2] for t in trajectory2])
    coh_np = 0.5 * (traj1_np + traj2_np) # [steps, D, 2]
    
    # Create radius scaling factors for each feature dimension
    # Start at 1.02 and increment by 0.03 for each feature
    radius_scales = np.array([1.03 + 0.03 * i for i in range(features)])
    
    # Create figure with proper spacing
    fig = plt.figure(figsize=(8, 8 + 0.8), dpi=400)
    fig.subplots_adjust(
        top=0.86,    # push title up
        bottom=0.00, # pull plot down
        left=0.04,
        right=0.96
    )
    
    # Create a single axis for all features
    ax = fig.add_subplot(111)
    
    # Add a legend to the figure with a box around it
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color=DARK_BLUE, label=r'$osc_{1}$',
                  markerfacecolor=DARK_BLUE, markersize=20, linewidth=0),
        plt.Line2D([0], [0], marker='o', color=DARK_RED, label=r'$osc_{2}$',
                  markerfacecolor=DARK_RED, markersize=20, linewidth=0),
        # coherence vector
        plt.Line2D([0], [0], marker='^', color=BLACK, label=r'$\rho_{d}$',
                   markerfacecolor=BLACK, markersize=20, linewidth=0),
    ]
    
    fig.legend(handles=legend_elements,
               loc="upper center",            # anchor legend to top-left
               bbox_to_anchor=(0.5, 0.98),    # slight inset from the corner
               ncol=3,                         # 3 columns
               frameon=True,
               edgecolor="lightgray",
               framealpha=0.85,
               fontsize=25,
               borderaxespad=0.2)             # small padding around the box
    
    # Set a single title for the entire figure with time step (in lowercase)
    time_title = fig.suptitle(f"t = 0", fontsize=25, y=0.88)
    def _init():
        ax.clear()
        ax.set_aspect("equal", "box")
        ax.set_xlim([-1.55, 1.55])
        ax.set_ylim([-1.55, 1.55])
        
        # Remove all borders, axes and ticks
        ax.axis('off')
        
        # Draw unit circle with darker outline
        circ = plt.Circle((0, 0), 1, edgecolor="#888888", facecolor="none", 
                         alpha=0.7, linewidth=0.8)
        ax.add_patch(circ)
        
        # Draw subtle circles for each feature radius
        for i in range(features):
            feature_circ = plt.Circle((0, 0), radius_scales[i], edgecolor="gray", 
                                    facecolor="none", alpha=0.15, linewidth=0.3, 
                                    linestyle="--")
            ax.add_patch(feature_circ)
        
        ax.plot([], [])
        return [ax]
    
    def _update(frame):
        # Update the global time title (lowercase)
        time_title.set_text(f"t = {frame}")
        
        ax.clear()
        ax.set_aspect("equal", "box")
        ax.set_xlim([-1.55, 1.55])
        ax.set_ylim([-1.55, 1.55])
        
        # Remove all borders, axes and ticks
        ax.axis('off')
        
        # Draw unit circle with darker outline
        circ = plt.Circle((0, 0), 1, edgecolor="#888888", facecolor="none", 
                         alpha=0.7, linewidth=0.8)
        ax.add_patch(circ)
        
        # Draw subtle circles for each feature radius
        for i in range(features):
            feature_circ = plt.Circle((0, 0), radius_scales[i], edgecolor="gray", 
                                     facecolor="none", alpha=0.15, linewidth=0.3, 
                                     linestyle="--")
            ax.add_patch(feature_circ)
        
        start = max(0, frame - trail_length + 1)
        
        # ------- Apply scaling and plot for each feature -----------
        for i in range(features):
            # Get radius for this feature
            radius = radius_scales[i]
            
            # Extract trajectories for this feature and apply scaling
            # traj1_np has shape [steps, D, 2] and we want feature i
            traj1_feature = traj1_np[:, i, :]  # Shape: [steps, 2]
            traj2_feature = traj2_np[:, i, :]  # Shape: [steps, 2]
            
            # Scale trajectories
            traj1_scaled = traj1_feature * radius
            traj2_scaled = traj2_feature * radius
            
            # Get coherence vector for this feature (calculated on pre-scaled trajectories)
            coh_vec = coh_np[frame, i, :]  # Shape: [2]
            # Scale coherence vector
            coh_vec_scaled = coh_vec * radius
            
            # Draw coherence vector
            ax.quiver(0, 0,
                     coh_vec_scaled[0], coh_vec_scaled[1],
                     color=BLACK,      
                     angles="xy", scale_units="xy", scale=1,
                     width=0.002, alpha=0.5, zorder=4)
            
            # # Get c-terms for this feature and scale them
            # c1_vec = c_terms1[0].cpu().numpy()[i, :2]  # Shape: [2]
            # c2_vec = c_terms2[0].cpu().numpy()[i, :2]  # Shape: [2]
            # #c1_scaled = c1_vec * radius
            # #c2_scaled = c2_vec * radius
            
            # # Draw c-term arrows
            # ax.quiver(0, 0, 
            #         c2_vec[0], c2_vec[1], 
            #         color=DARK_RED, angles="xy", scale_units="xy", scale=1,
            #         width=0.002, alpha=0.8)
            # ax.quiver(0, 0, 
            #         c1_vec[0], c1_vec[1], 
            #         color=DARK_BLUE, angles="xy", scale_units="xy", scale=1,
            #         width=0.002, alpha=0.8)
            
            # Draw trajectory trails
            if frame > start:
                ax.plot(
                    traj1_scaled[start:frame+1, 0],
                    traj1_scaled[start:frame+1, 1], 
                    "-", color=DARK_BLUE, 
                    alpha=0.8, linewidth=2.0  # Thicker, more visible trails
                )
                ax.plot(
                    traj2_scaled[start:frame+1, 0],
                    traj2_scaled[start:frame+1, 1], 
                    "-", color=DARK_RED, 
                    alpha=0.8, linewidth=2.0  # Thicker, more visible trails
                )
            
            # Draw current positions
            ax.scatter(traj1_scaled[frame, 0], traj1_scaled[frame, 1], 
                      c=DARK_BLUE, marker="o", s=60, 
                      edgecolor="white", linewidth=0.5, zorder=5, alpha=0.8)
            ax.scatter(traj2_scaled[frame, 0], traj2_scaled[frame, 1], 
                      c=DARK_RED, marker="o", s=60, 
                      edgecolor="white", linewidth=0.5, zorder=5, alpha=0.8)

        return [ax]

    ani=None
    if args.animate:
        ani = FuncAnimation(fig, _update,
                          frames=num_steps,
                          init_func=_init,
                          interval=400,
                          blit=False)
    _update(num_steps-1)
    return ani, fig
## Visualization Helpers: Examples
def visualize_image_pair(img1, img2, prob, label=None):
    """
    Visualize a pair of images with consistent academic paper styling.
    
    Parameters
    ----------
    img1, img2 : torch.Tensor
        Input image tensors.
    label : int, optional
        Ground truth label (1=same, 0=different).
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure with the input image pair.
    """
    # Use consistent academic paper style
    plt.style.use("seaborn-v0_8-paper")
    
    # Create figure for image pair
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Display first image
    img1_np = img1[0].cpu() if img1.dim() == 4 else img1.cpu()
    if img1_np.shape[0] == 1:  # grayscale
        img1_np = img1_np.squeeze()
        ax1.imshow(img1_np, cmap="gray")
    else:  # RGB or other multi-channel
        img1_np = img1_np.permute(1, 2, 0)
        ax1.imshow(img1_np)
    
    # Display second image
    img2_np = img2[0].cpu() if img2.dim() == 4 else img2.cpu()
    if img2_np.shape[0] == 1:  # grayscale
        img2_np = img2_np.squeeze()
        ax2.imshow(img2_np, cmap="gray")
    else:  # RGB or other multi-channel
        img2_np = img2_np.permute(1, 2, 0)
        ax2.imshow(img2_np)
    
    # Set titles and remove axes
    ax1.set_title("Image 1", fontsize=12)
    ax2.set_title("Image 2", fontsize=12)
    ax1.axis("off")
    ax2.axis("off")
    
    # # Add overall title if label is provided
    # if label is not None:
    #     relation = "Same" if label == 1 else "Different"
    #     fig.suptitle(f"{relation} Pair", fontsize=14)

    # Title with prediction and ground truth if provided
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
        Energy values from model output.
    prob : float, optional
        Model prediction probability.
    label : int, optional
        Ground truth label (1=same, 0=different).
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure with the energy curve plot.
    """
    # Use consistent academic paper style
    plt.style.use("seaborn-v0_8-paper")
    
    # Use consistent colors
    DARK_BLUE = "#2a9da3"  # Teal
    DARK_RED = "#c86460"   # Red
    BLACK = "#000000"      # Black
    
    # Create figure for energy curve
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Extract energy values
    steps = list(range(len(energy_vals)))
    energies = [e[0].item() for e in energy_vals]
    
    # Plot energy curve
    ax.plot(steps, energies, "-", color=BLACK, lw=2.0, alpha=0.8)
    
    
    # Axis labels
    ax.set_xlabel("kuramoto step", fontsize=11)
    ax.set_ylabel("energy", fontsize=11)
    
    # # Title with prediction and ground truth if provided
    # title = "Energy curve"
    # if label is not None or prob is not None:
    #     title += "\n"
    #     if label is not None:
    #         title += f"ground truth: {'same' if label == 1 else 'different'}"
    #     if label is not None and prob is not None:
    #         title += "  |  "
    #     if prob is not None:
    #         title += f"prediction: {'same' if prob > 0.5 else 'different'} ({prob:.2f})"
    
    #ax.set_title(title, fontsize=12)
    
    # # Legend
    # ax.legend(loc="lower right", frameon=True, 
    #           edgecolor="lightgray", framealpha=0.8, fontsize=10)
    
    # No grid
    ax.grid(False)
    
    plt.tight_layout()
    return fig

def visualize_oscillator_features(traj1, traj2, c1, c2, max_features=16):
    """
    Visualize oscillator trajectories for multiple features.
    
    Parameters
    ----------
    traj1, traj2 : list of torch.Tensor
        Trajectories of the oscillators.
    c1, c2 : torch.Tensor
        Coupling terms.
    max_features : int, optional
        Maximum number of features to display.
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure with the oscillator trajectories grid.
    """
    # Use consistent academic paper style
    plt.style.use("seaborn-v0_8-paper")
    
    # Use consistent colors
    DARK_BLUE = "#2a9da3"  # Teal
    DARK_RED = "#c86460"   # Red
    BLACK = "#000000"      # Black
    
    # Extract dimensionality information
    D = traj1[0].shape[1]  # embedding dimension
    features = min(D, max_features)
    
    # Calculate grid dimensions
    cols = math.ceil(math.sqrt(features))
    rows = math.ceil(features / cols)
    
    # Create figure with appropriate size
    fig = plt.figure(figsize=(3*cols, 3*rows + 0.8))
    
    # Add a legend at the top
    legend_elements = [

        # oscillators
        plt.Line2D([0], [0], marker='o', color=DARK_BLUE, label=r'$osc_1$',
                  markerfacecolor=DARK_BLUE, markersize=8, linewidth=0),
        plt.Line2D([0], [0], marker='o', color=DARK_RED, label=r'$osc_2$',
                  markerfacecolor=DARK_RED, markersize=8, linewidth=0),

        # c-term directions  (hollow triangles to distinguish them)
        plt.Line2D([0], [0], marker='^', color=DARK_BLUE, label=r'$c_{1}$',
                   markerfacecolor=DARK_BLUE, markersize=9, linewidth=0),
        plt.Line2D([0], [0], marker='^', color=DARK_RED,  label=r'$c_{2}$',
                   markerfacecolor=DARK_RED, markersize=9, linewidth=0),

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
    
    # Create grid of subplots
    gs = fig.add_gridspec(rows, cols, top=0.9, wspace=0.1, hspace=0.1)
    
    # Pre-convert trajectories to numpy
    traj1_np = np.array([t[0].cpu().numpy()[:, :2] for t in traj1])  # [steps, D, 2]
    traj2_np = np.array([t[0].cpu().numpy()[:, :2] for t in traj2])
    
    # Scale trajectories to separate them
    traj1_np_scaled = traj1_np * 0.97  # Inner orbit
    traj2_np_scaled = traj2_np * 1.03  # Outer orbit
    
    # Get stimulus vectors
    c1_np = c1[0].cpu().numpy()[:, :2]  # [D, 2]
    c2_np = c2[0].cpu().numpy()[:, :2]  # [D, 2]
    
    # Compute coherence vectors
    coh_np = 0.5 * (traj1_np + traj2_np)  # [steps, D, 2]
    
    # Final frame index
    final_idx = len(traj1) - 1
    
    # Plot each feature
    for i in range(features):
        row = i // cols
        col = i % cols
        ax = fig.add_subplot(gs[row, col])
        
        # Draw unit circle
        circle = plt.Circle((0, 0), 1.0, edgecolor="#888888", facecolor="none", 
                           alpha=0.7, linewidth=0.8)
        ax.add_patch(circle)
        
        # Draw coherence vector
        coh_vec = coh_np[final_idx, i]
        ax.quiver(0, 0,
                 coh_vec[0], coh_vec[1],
                 color=BLACK,
                 angles="xy", scale_units="xy", scale=1,
                 width=0.007, alpha=0.9, zorder=4)
                 
        # Draw c-term arrows
        ax.quiver(0, 0, 
                 c1_np[i, 0], c1_np[i, 1], 
                 color=DARK_BLUE, angles="xy", scale_units="xy", scale=1,
                 width=0.006, alpha=0.9)
        ax.quiver(0, 0, 
                 c2_np[i, 0], c2_np[i, 1], 
                 color=DARK_RED, angles="xy", scale_units="xy", scale=1,
                 width=0.006, alpha=0.9)
        
        # Draw trajectories
        ax.plot(traj1_np_scaled[:, i, 0], traj1_np_scaled[:, i, 1], 
               "-", color=DARK_BLUE, alpha=0.8, linewidth=1.5)
        ax.plot(traj2_np_scaled[:, i, 0], traj2_np_scaled[:, i, 1], 
               "-", color=DARK_RED, alpha=0.8, linewidth=1.5)
        
        # Mark ending positions
        ax.scatter(traj1_np_scaled[final_idx, i, 0], traj1_np_scaled[final_idx, i, 1], 
                  c=DARK_BLUE, s=40, edgecolor="white", linewidth=0.5, zorder=5, alpha=0.8)
        ax.scatter(traj2_np_scaled[final_idx, i, 0], traj2_np_scaled[final_idx, i, 1], 
                  c=DARK_RED, s=40, edgecolor="white", linewidth=0.5, zorder=5, alpha=0.8)
        
        # Configure axis
        ax.set_aspect("equal", "box")
        ax.set_xlim([-1.3, 1.3])
        ax.set_ylim([-1.3, 1.3])
        ax.axis('off')  # Remove borders, axes and ticks
    
    plt.subplots_adjust(top=0.85, wspace=0.1, hspace=0.1)
    return fig

def visualize_single_example(model, img1, img2, label=None, device=None, max_features=16):
    """
    Generate all visualization components for a single example, preserving
    the existing interface while allowing for separate plotting.
    
    Parameters
    ----------
    model : nn.Module
        The KuramotoRelationalModel to use for visualization.
    img1, img2 : torch.Tensor
        Input image tensors.
    label : int, optional
        Ground truth label (1=same, 0=different).
    device : torch.device, optional
        Device to run the model on.
    max_features : int,
        Maximum number of features to display.
    
    Returns
    -------
    tuple
        Contains (combined_fig, img_fig, energy_fig, osc_fig, model_outputs)
    """
    # Setup
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # Ensure batch dimension
    if img1.dim() == 3: img1 = img1.unsqueeze(0)
    if img2.dim() == 3: img2 = img2.unsqueeze(0)
    img1, img2 = img1.to(device), img2.to(device)
    
    # Forward pass
    with torch.no_grad():
        logits, osc_data = model(img1, img2)
        prob = torch.sigmoid(logits)[0, 0].item()
        final_osc1, final_osc2, traj1, traj2, energy_vals, c1, c2 = osc_data
    
    # Generate individual visualizations
    img_fig = visualize_image_pair(img1, img2, prob, label)
    energy_fig = visualize_energy_curve(energy_vals, prob, label)
    osc_fig = visualize_oscillator_features(traj1, traj2, c1, c2, max_features)
    
    # Return both individual figures and the traditional single figure
    # to maintain compatibility with existing code
    return (img_fig, energy_fig, osc_fig, (logits, osc_data))
def _log_example_visuals(model, sample_batch, epoch, device):
    """
    Logs one 'same' pair and one 'different' pair to WandB, plus 2‑D oscillator
    animations (first epoch and last epoch only).

    Parameters
    ----------
    model       : the KuramotoRelationalModel (already .eval()‑ed outside)
    sample_batch  : Mini-batch that yields ((img1,img2), label)
    epoch       : current epoch index     (int)
    device      : torch.device (accelerator.device)
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

        # -------------- SAME example -----------------------------------
        same_img1, same_img2 = imgs1[idx_same], imgs2[idx_same]
        img_fig, energy_fig, osc_fig, (logits, osc_data) = visualize_single_example(
            model,
            same_img1,
            same_img2,
            label=1,
            device=device,
            max_features=args.embedding_dim,
        )
        
        # Log each component separately for the "same" example
        wandb.log({
            f"examples_same/epoch_{epoch}_images": wandb.Image(img_fig),
            f"examples_same/epoch_{epoch}_energy": wandb.Image(energy_fig),
            f"examples_same/epoch_{epoch}_oscillators": wandb.Image(osc_fig)
        })
        
        # Close figures to prevent memory leaks
        plt.close(img_fig)
        plt.close(energy_fig)
        plt.close(osc_fig)

        # -------------- DIFFERENT example ------------------------------
        diff_img1, diff_img2 = imgs1[idx_diff], imgs2[idx_diff]
        img_fig, energy_fig, osc_fig, (logits, osc_data) = visualize_single_example(
            model,
            diff_img1,
            diff_img2,
            label=0,
            device=device,
            max_features=args.embedding_dim,
        )
        
        # Log each component separately for the "different" example
        wandb.log({
            f"examples_diff/epoch_{epoch}_images": wandb.Image(img_fig),
            f"examples_diff/epoch_{epoch}_energy": wandb.Image(energy_fig),
            f"examples_diff/epoch_{epoch}_oscillators": wandb.Image(osc_fig)
        })
        
        # Close figures to prevent memory leaks
        plt.close(img_fig)
        plt.close(energy_fig)
        plt.close(osc_fig)
        

        # -------------- oscillator animations (first & last epoch) -----
        if epoch in (-1, args.epochs - 1):
            # reuse the “same” and “different” cases above
            for tag, (i1, i2) in {
                "osc_same": (imgs1[idx_same], imgs2[idx_same]),
                "osc_diff": (imgs1[idx_diff], imgs2[idx_diff]),
            }.items():
                i1 = i1.unsqueeze(0).to(device)
                i2 = i2.unsqueeze(0).to(device)
                _, osc_data = model(i1, i2)
                _, _, traj1, traj2, _, c1, c2 = osc_data

                anim, fig = visualize_oscillators_2d(
                    traj1, traj2, c1, c2,
                    max_features=args.embedding_dim,
                    trail_length=25
                )
                if args.animate:
                    _wandb_log_animation(anim, tag=f"{tag}/epoch_{epoch}")
                wandb.log({f"{tag}/epoch_{epoch}_still": wandb.Image(fig)})
                plt.close(fig)
                
                ani, fig = visualize_oscillators_2d_overlay(
                    traj1, traj2, c1, c2,
                    max_features=args.embedding_dim, 
                    trail_length=25
                )
                if args.animate:
                    _wandb_log_animation(ani, tag=f"{tag}/epoch_{epoch}_overlay")
                wandb.log({f"{tag}/epoch_{epoch}_overlay_still": wandb.Image(fig)})
                plt.close(fig)

def _plot_history(train_h, title, save_dir, tag):
    """
    Draw a single‑axis figure that shows batch‑level *training* accuracy.
    """
    if not train_h["train_acc"]:          # nothing to plot
        return

    plt.style.use("seaborn-v0_8-paper")
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(train_h["train_acc"],
            label="train accuracy",
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

    #— optional: still push to Weights & Biases ————————————————
    wandb.log({f"history/{tag}": wandb.Image(str(out_path))})
## Main
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
):
    is_train = mode == "train"
    model.train() if is_train else model.eval()

    tot_loss = tot_correct = tot_samples = 0
    loader   = tqdm(dataloader, desc=f"E{epoch_idx+1} [{mode}]", leave=False, disable=True)

    grad_ctx  = nullcontext() if is_train else torch.no_grad()

    for step, (imgs, labels) in enumerate(loader, 1):
        img1, img2 = imgs                              # tuple unpack
        img1, img2, labels = (
            t.to(accelerator.device, non_blocking=True) for t in (img1, img2, labels)
        )
        labels = labels.float().unsqueeze(1)

        with grad_ctx, accelerator.autocast():
            logits, *_ = model(img1, img2)
            loss = criterion(logits, labels)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()                 # one LR update per *batch*

        # -------- stats ---------------------------------------------------
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = probs > 0.5
            batch_acc = (preds == labels.bool()).float().mean().item()

        if mode == "train":              # same / different (training)
            history_sd["train_acc"].append(batch_acc)

        bs = labels.size(0)
        tot_loss    += loss.item() * bs
        tot_correct += (preds == labels.bool()).sum().item()
        tot_samples += bs

        # -------- WandB batch logging ------------------------------------
        if step % log_every == 0:
            log_dict = {
                f"{mode}/batch_loss": loss.item(),
                f"{mode}/batch_acc" : batch_acc,
            }
            if is_train:
                log_dict["lr"] = optimizer.param_groups[0]["lr"]
            wandb.log(log_dict)

        # optional gradient histograms
        if is_train and step % grad_log_every == 0:
            log_gradient_norms(model, "feature_extractor")
            if args.model_type == "kuramoto":
                log_gradient_norms(model, "oscillator_network")
                log_gradient_norms(model, "coherence_measurement")
            log_gradient_norms(model, "classifier")

        loader.set_postfix(loss=f"{loss.item():.3f}", acc=f"{batch_acc:.3f}")

    avg_loss = tot_loss / tot_samples
    avg_acc  = tot_correct / tot_samples
    return avg_loss, avg_acc
def main():
    # Accelerator setup
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device
    accelerator.print(f"Running on {accelerator.device}")
    set_seed(args.seed)
    
    # --------------------------
    # SET SEEDS FOR REPRODUCIBILITY
    # --------------------------
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # Create job directory
    jobdir = os.path.join("runs", args.exp_name)
    os.makedirs(jobdir, exist_ok=True)
    
    # Set up logging
    logger = create_logger(jobdir)
    logger.info(f"Experiment directory created at {jobdir}")
    
    # --------------------------
    # DATASET CREATION
    # --------------------------
    logger.info("Generating training/validation datasets...")
    logger.info(range(args.n_train_glyphs))
    full_dataset = IconSameDiffDataset(
        image_dir="./imgs/",
        num_samples=args.n_train + args.n_val,
        transform=COMMON_TRANSFORM,
        icon_indices=range(args.n_train_glyphs),
        seed=args.seed
    )
    train_dataset, val_dataset = torch.utils.data.random_split(
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
    
    # DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers>0)
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers>0)
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers>0)
    )
    
    # --------------------------
    # INITIALIZE LOGGING
    # --------------------------
    wandb_config = {
        "model_type": "KuramotoRelationalModel",
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
        "model_type": args.model_type,
        "use_omega": args.use_omega,
        "omega_kappa": args.omega_kappa,
        "symmetric_j": args.symmetric_j,
    }
    wandb.init(project="kuramoto-relational-model", name=args.exp_name, config=wandb_config)
    
    # --------------------------
    # MODEL & TRAINER PREPARATION
    # --------------------------
    logger.info("Initializing model and optimizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
        )
    else:   # CNN‑only baseline
        model = BaselineSameDiffModel(
            input_channels=1,
            embedding_dim=args.embedding_dim,
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

    # ---------- LR schedule --------------------------------------------------
    total_batches = args.epochs * len(train_loader)      # step every batch
    warmup_steps  = int(0.05 * total_batches)            # 5 % of the run

    warmup_sched  = LinearLR(
        optimizer,
        start_factor=1/25,        # lr rises from args.lr/25  → args.lr
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
    
    # Model parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {total_params}")
    wandb.config.update({"total_params": total_params})
    
    # Watch model gradients
    wandb.watch(model, log="all", log_freq=100)

    # track best val performance
    best_val_acc = 0.0
    best_epoch = -1
    
    # -------- training -----------------------------------------------------
    logger.info(f"Start training for {args.epochs} epochs")
    sample_batch = next(iter(train_loader))              # for energy plot
    all_epoch_energies = {}

    # ---- baseline energy *before* training -------------------------------
    if args.model_type == "kuramoto":
        init_energy = collect_energy_values(model, sample_batch, accelerator.device)
        all_epoch_energies[-1] = init_energy
        wandb.log({"energy/plot":
                   wandb.Image(display_energy_plot(init_energy, epoch=-1))})

    # ---- baseline visuals *before* training -------------------------------
    # use epoch=-1 to mark “initialization”
    if args.model_type == "kuramoto":
        _log_example_visuals(model, sample_batch, epoch=-1, device=accelerator.device)

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
            scheduler=scheduler
        )

        val_loss, val_acc = run_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            accelerator=accelerator,
            epoch_idx=epoch,
            mode="val"
        )
        elapsed = time.perf_counter() - start

        # Check if this is the best validation accuracy so far
        if val_acc > best_val_acc and epoch > 24:
            best_val_acc = val_acc
            best_epoch = epoch
            
            # Save the best model
            accelerator.save(accelerator.unwrap_model(model).state_dict(),
                           os.path.join(jobdir, "model_best_val.pt"))
            logger.info(f"\nNew best model saved at epoch {epoch+1} with validation accuracy {val_acc:.4f}")

        # ---- epoch‑level summary -------------------------------------------
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,  "train/acc": train_acc,
            "val/loss"  : val_loss,    "val/acc"  : val_acc,
        })

        logger.info(f"\nEpoch {epoch+1}/{args.epochs} | {elapsed:6.1f}s\n"
                    f"train loss:{train_loss:.4f} | acc:{train_acc:.4f}\n"
                    f"val   loss:{val_loss:.4f} | acc:{val_acc:.4f}")

        if args.model_type == "kuramoto":
            # ---- energy visualisation ----------------------------------------
            energy_vals = collect_energy_values(model, sample_batch, accelerator.device)
            all_epoch_energies[epoch] = energy_vals
            wandb.log({"energy/plot": wandb.Image(display_energy_plot(energy_vals, epoch))})
    
            # ---- sample visualisations every 5 epochs ------------------------
            if epoch % 5 == 0 or epoch == args.epochs - 1:
                _log_example_visuals(model, sample_batch, epoch, accelerator.device)

    # ---------- end training  ---------------------------------------------
    accelerator.save(accelerator.unwrap_model(model).state_dict(),
                     os.path.join(jobdir, "model_final.pt"))

    # energy GIF once at the end
    if args.model_type == "kuramoto":
        gif = build_energy_animation(all_epoch_energies,
                                     fname=os.path.join(jobdir, "energy_evolution.gif"))
        wandb.log({"energy/gif": wandb.Video(gif, format="gif")})

    _plot_history(
        history_sd,
        title="Same/Different",
        save_dir=jobdir,
        tag="same_diff",
    )
    
    # test set evaluation ---------------
    test_loss, test_acc = run_epoch(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        accelerator=accelerator,
        epoch_idx=args.epochs,
        mode="test"
    )
    wandb.log({"test/loss": test_loss, "test/acc": test_acc})
    logger.info(f"\n S/D test loss={test_loss:.4f} | acc={test_acc:.4f}")
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
## Main RMTS
def main_rmts():
    """
    This function loads the trained same/different model, freezes it,
    and trains a new classifier on the relational match-to-sample (R-MTS) task
    using the IconRelMatchToSampleDataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create job directory
    jobdir = os.path.join("runs", args.exp_name)
    os.makedirs(jobdir, exist_ok=True)
    
    # 1) load the frozen encoder
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
        ).to(device)
    else:
        elder_model = BaselineSameDiffModel(
            input_channels=1,
            embedding_dim=args.embedding_dim,
        ).to(device)

    # Load trained weights (the path from same/diff run)
    model_ckpt = "model_final" if args.load_model == "final" else "model_best_val"
    ckpt_path = f"runs/{args.exp_name}/{model_ckpt}.pt"
    state_dict = torch.load(ckpt_path, map_location=device)
    elder_model.load_state_dict(state_dict)
    elder_model.to(device)

    # Freeze it
    elder_model.eval()
    for param in elder_model.parameters():
        param.requires_grad = False

    # 2) Create R-MTS dataset

    # "TRAIN" on first m icons
    rmts_train_dataset = IconRelMatchToSampleDataset(
        image_dir="./imgs/",
        num_samples=args.n_train,
        transform=COMMON_TRANSFORM,
        icon_indices=range(args.n_train_glyphs),  # only icons [0..79]
        seed=args.seed
    )
    # "VAL" also on first m icons (could use fewer samples, but let's do 500 for example)
    rmts_val_dataset = IconRelMatchToSampleDataset(
        image_dir="./imgs/",
        num_samples=args.n_val,
        transform=COMMON_TRANSFORM,
        icon_indices=range(args.n_train_glyphs), # also [0..79]
        seed=args.seed
    )
    # "TEST" on last 100-m icons => out-of-distribution
    rmts_test_dataset = IconRelMatchToSampleDataset(
        image_dir="./imgs/",
        num_samples=args.n_test,
        transform=COMMON_TRANSFORM,
        icon_indices=(range(args.n_train_glyphs, 100) if args.n_train_glyphs != 100 else range(100)),
        seed=args.seed
    )

    rmts_train_loader = torch.utils.data.DataLoader(rmts_train_dataset, batch_size=args.batch_size, shuffle=True)
    rmts_val_loader   = torch.utils.data.DataLoader(rmts_val_dataset,   batch_size=args.batch_size, shuffle=False)
    rmts_test_loader  = torch.utils.data.DataLoader(rmts_test_dataset,  batch_size=args.batch_size, shuffle=False)
    

    # 3) Create an RMTSClassifier
    if args.model_type == "kuramoto":
        rmts_classifier = RMTSClassifier(
            coherence_dim=args.embedding_dim, 
            hidden_dim=64
        ).to(device)
    else:
        rmts_classifier = BaselineRMTSClassifier(
            coherence_dim=args.embedding_dim, 
            hidden_dim=64
        ).to(device)
    get_vec = elder_model.get_coherence_vector

    # 4) Training Setup
    optimizer = optim.Adam(rmts_classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    num_epochs = args.rmts_epochs

    # 5) Train R-MTS Classifier
    for epoch in range(num_epochs):
        rmts_classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for (s1, s2, t1a, t1b, t2a, t2b), label in rmts_train_loader:
            # Move everything to device
            s1, s2 = s1.to(device), s2.to(device)
            t1a, t1b = t1a.to(device), t1b.to(device)
            t2a, t2b = t2a.to(device), t2b.to(device)
            label = label.long().to(device)  # shape [batch_size]

            # Zero out old grads
            optimizer.zero_grad()

            # Extract coherence vectors from the frozen model
            source_vec = get_vec(s1, s2)   # [batch_size, D=64]
            t1_vec = get_vec(t1a, t1b)     # [batch_size, 64]
            t2_vec = get_vec(t2a, t2b)     # [batch_size, 64]

            # Forward pass in R-MTS classifier => 2 logits
            logits = rmts_classifier(source_vec, t1_vec, t2_vec)    # shape [batch_size, 2]

            # Compute cross-entropy
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * label.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == label).sum().item()
            train_total += label.size(0)

            history_rmts["train_acc"].append(
                (preds == label).float().mean().item()
            )

        avg_train_loss = train_loss / train_total
        avg_train_acc = train_correct / train_total

        # Validation
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
        np.save(Path(jobdir) / f"{args.model_type}_Jout{args.disable_between}_Omega{args.use_omega}_Jsymm{args.symmetric_j}_rmts_train_batch_acc.npy", 
                np.asarray(history_rmts["train_acc"]))
    else:
        np.save(Path(jobdir) / f"{args.model_type}_rmts_train_batch_acc.npy", 
                np.asarray(history_rmts["train_acc"]))



# %%


def parse_args() -> argparse.Namespace:
    """Return hyper‑parameters from the command line (or sbatch env)."""
    parser = argparse.ArgumentParser(
        description="Train Kuramoto relational model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- experiment & bookkeeping ------------------------------------- #
    parser.add_argument("--exp_name",          default="4-23-noJout")
    parser.add_argument("--seed",      type=int, default=123)
    parser.add_argument("--checkpoint_every",   type=int, default=10)
    parser.add_argument("--load_model", choices=["final", "best"], default="final")
    parser.add_argument("--model_type", choices=["kuramoto", "baseline"], default="kuramoto")
    parser.add_argument("--animate", action='store_true')

    # --- optimisation -------------------------------------------------- #
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--rmts_epochs",type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--finetune",   type=str,   default=None,
                        help="Path to checkpoint for fine‑tuning")

    # --- model hyper‑params ------------------------------------------- #
    parser.add_argument("--embedding_dim",   type=int, default=16)
    parser.add_argument("--oscillator_dim",  type=int, default=2)
    parser.add_argument("--num_steps",       type=int, default=25)
    parser.add_argument("--step_size",       type=float, default=0.1)
    parser.add_argument("--use_omega",       action='store_true')
    parser.add_argument("--omega_kappa",     type=float, default=2.0)
    parser.add_argument("--disable_between", action='store_true')
    parser.add_argument("--symmetric_j",     action='store_true')

    # --- dataset sizes ------------------------------------------------- #
    parser.add_argument("--n_train_glyphs", type=int, default=100)
    parser.add_argument("--n_train",        type=int, default=4000)
    parser.add_argument("--n_val",          type=int, default=100)
    parser.add_argument("--n_test",         type=int, default=10000)
    parser.add_argument("--num_workers",    type=int, default=2)

    args = parser.parse_args()


    return args


if __name__ == "__main__":
    # Parse CLI flags ➜ global `args` so existing code keeps working
    args = parse_args()

    # Echo configuration to stdout / log file (handy inside sbatch logs)
    print(">>> Hyper‑parameters")
    for k, v in vars(args).items():
        print(f"{k:20}: {v}")
    print(">>> Starting training …")

    history_sd   = {"train_acc": []}           # same / different
    history_rmts = {"train_acc": []}    # RMTS

    main()
    main_rmts()

# %%



