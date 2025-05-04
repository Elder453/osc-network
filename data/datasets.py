"""
Dataset classes for experiments.

Includes datasets for same/different and
relational match-to-sample tasks.
"""

import os
import random
from typing import Optional, List, Sequence

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from data.transforms import COMMON_TRANSFORM


def load_icons(image_dir: str, icon_indices: Sequence[int]):
    """
    Load and return a list of PIL.Image from the specified directory for the given indices.
    
    Parameters
    ----------
    image_dir : str
        Directory containing icon images
    icon_indices : Sequence[int]
        Indices of icons to load
        
    Returns
    -------
    list
        List of loaded PIL images
        
    Raises
    ------
    ValueError
        If no images were found at the specified locations
    """
    icons = []
    for i in icon_indices:
        path = os.path.join(image_dir, f"{i}.png")
        if os.path.exists(path):
            icons.append(Image.open(path).convert("L"))
    if not icons:
        raise ValueError(f"No images found in {image_dir} with indices: {icon_indices}")
    return icons


class IconSameDiffDataset(Dataset):
    """
    Dataset generating pairs of icons (same or different) from a folder of icons.
    
    Each sample consists of (icon_tensor1, icon_tensor2), label where label is 1 for
    same pairs and 0 for different pairs.

    Parameters
    ----------
    image_dir : str
        Folder containing icon images
    num_samples : int
        Number of pairs to generate
    transform : callable, optional
        Transforms to apply to images
    icon_indices : list, optional
        Which icon indices to load (default: range(100))
    seed : int, optional
        Random seed for reproducibility
    """
    def __init__(
        self,
        image_dir: str,
        num_samples: int = 10000,
        transform = None,
        icon_indices: Optional[List[int]] = None,
        seed: int = 0
    ):
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
            if rng.random() < 0.5:                       # "same" case
                j = rng.randrange(len(self.icons))
                self._pairs.append((j, j, 1))
            else:                                        # "different" case
                j, k = rng.sample(range(len(self.icons)), 2)
                self._pairs.append((j, k, 0))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        j, k, label = self._pairs[idx]
        iconA, iconB = self.icons[j], self.icons[k]
        return (self.transform(iconA), self.transform(iconB)), label


class IconRelMatchToSampleDataset(Dataset):
    """
    Relational Match-to-Sample Dataset using icons.
    
    Each item yields (S1, S2, T1a, T1b, T2a, T2b), label in {0,1}
    indicating which target matches the source's relation (0=first target, 1=second target).

    Parameters
    ----------
    image_dir : str
        Folder containing icon images
    num_samples : int
        Number of samples to generate
    transform : callable, optional
        Transforms to apply to images
    icon_indices : list, optional
        Which icon indices to load (default: range(100))
    seed : int, optional
        Random seed for reproducibility
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
            src_same = rng.random() < 0.5
            source = self._make_pair(rng, src_same)
            pair_same = self._make_pair(rng, True)
            pair_diff = self._make_pair(rng, False)

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
        Get a sample from the dataset.
        
        Returns
        -------
        tuple
            (s1, s2, t1a, t1b, t2a, t2b), label
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
        Create a pair of icons that are either the same or different.
        
        Parameters
        ----------
        rng : random.Random
            Random number generator
        is_same : bool
            Whether to create a same (True) or different (False) pair
            
        Returns
        -------
        tuple
            (iconA, iconB)
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


def visualize_icon_samediff_sample(dataset, idx=None):
    """
    Visualize a sample from IconSameDiffDataset.
    
    Parameters
    ----------
    dataset : IconSameDiffDataset
        Dataset instance
    idx : int, optional
        Sample index (if None, a random sample is chosen)
    """
    import matplotlib.pyplot as plt
    
    if idx is None:
        idx = random.randint(0, len(dataset) - 1)
    
    # retrieve sample: ((img1_tensor, img2_tensor), label)
    (img1, img2), label = dataset[idx]
    
    # convert tensors to numpy arrays for visualization
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


def visualize_icon_rmts_sample(dataset, idx=None):
    """
    Visualize a sample from IconRelMatchToSampleDataset.
    
    Parameters
    ----------
    dataset : IconRelMatchToSampleDataset
        Dataset instance
    idx : int, optional
        Sample index (if None, a random sample is chosen)
    """
    import matplotlib.pyplot as plt
    
    if idx is None:
        idx = random.randint(0, len(dataset) - 1)
        
    # retrieve sample: (s1, s2, t1a, t1b, t2a, t2b), label 
    (s1, s2, t1a, t1b, t2a, t2b), label = dataset[idx]
    
    # convert tensors to numpy arrays
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
    
    # row 1: source pair
    plt.subplot(3, 2, 1)
    plt.imshow(s1_np, cmap=cmap_s1)
    plt.title('Source Image 1')
    plt.axis('off')
    
    plt.subplot(3, 2, 2)
    plt.imshow(s2_np, cmap=cmap_s1)
    plt.title('Source Image 2')
    plt.axis('off')
    
    # row 2: first target pair
    plt.subplot(3, 2, 3)
    plt.imshow(t1a_np, cmap=cmap_s1)
    plt.title('Target 1, Image 1')
    plt.axis('off')
    
    plt.subplot(3, 2, 4)
    plt.imshow(t1b_np, cmap=cmap_s1)
    plt.title('Target 1, Image 2')
    plt.axis('off')
    
    # row 3: second target pair
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