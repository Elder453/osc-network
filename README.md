# Kuramoto Relational Learning

A differentiable, oscillator-based neural architecture for abstract relational reasoning. This repository implements biologically-inspired binding-by-synchrony mechanisms using Kuramoto dynamics to solve relational reasoning tasks such as same/different discrimination and relational match-to-sample (RMTS).

## Overview

Relational reasoning requires a learning system to represent *how* entities are related, not merely *what* they are. This project implements a neural architecture that leverages synchronous oscillations to dynamically bind distributed features, inspired by neuroscientific evidence. Each feature dimension of two input objects is realized as a pair of coupled phase oscillators whose interactions are governed by learned within- and between-object coupling tensors.

After a short roll-out of vector-valued Kuramoto dynamics, cross-object *phase coherence* forms a compact *relational bottleneck* that is fed to a lightweight classifier, enabling efficient relational abstraction.

## Key Features

- **Oscillatory Binding Mechanism**: Implementation of binding-by-synchrony through Kuramoto oscillator networks
- **Relational Bottleneck**: Feature coherence measurements that abstract relationships while suppressing object-specific details
- **Transfer Learning Architecture**: Two-stage learning framework from same/different to RMTS tasks
- **Visualization Tools**: Energy landscape analysis and phase-space visualization of oscillator dynamics
- **Multiple Model Variants**: Baseline CNN+MLP and ablated oscillator models for comparison

## System Requirements

- **CUDA-capable GPU**: This codebase requires CUDA support for GPU acceleration. Training and inference have been optimized for NVIDIA GPUs.
- **Python 3.8+**: Compatible with recent Python versions.
- **CUDA Toolkit**: Compatible with CUDA 11.0 or higher.
- **Sufficient GPU Memory**: Recommend at least 8GB of VRAM for training with default parameters.

## Installation

We provide an `environment.yml` file for easy setup with Conda:

```bash
# Create environment and install dependencies
conda env create -f environment.yml

# Activate the environment
conda activate kuramoto-rel
```

Please verify your CUDA installation is properly configured by running:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Directory Structure

```
kuramoto_relational/
├── README.md                  # Project documentation and usage instructions
├── __init__.py                # Root package initialization
├── main.py                    # Entry point script
├── config.py                  # Configuration and hyperparameters
├── environment.yml            # Conda environment specification
├── requirements.txt           # Package dependencies for pip
├── data/                      # Dataset handling
│   ├── __init__.py          
│   ├── datasets.py            # IconSameDiffDataset, IconRelMatchToSampleDataset
│   └── transforms.py          # RandomResizeAndPad, COMMON_TRANSFORM
├── models/                    # Neural network models
│   ├── __init__.py
│   ├── feature_extractor.py   # CNN encoder
│   ├── kuramoto.py            # Kuramoto oscillator network components
│   ├── coherence.py           # Coherence measurement
│   ├── classification.py      # Classifiers (MLP, RMTS)
│   └── baseline.py            # Baseline models
├── training/                  # Training pipelines
│   ├── __init__.py
│   ├── trainer.py             # Same/different training
│   └── rmts_trainer.py        # RMTS training
├── utils/                     # Utility functions
│   ├── __init__.py
│   ├── logging.py             # Logging utilities
│   └── visualization.py       # Plotting functions
├── imgs/                      # Image data directory
│   ├── 0.png                  # Icon image 0
│   ├── 1.png                  # Icon image 1
│   └── ...                    # Icon images 2-99
└── results/                   # To re-create results in paper
```

## Usage

### Training a Same/Different Model

```bash
python main.py --exp_name kuramoto_sd --model_type kuramoto --embedding_dim 64 \
               --oscillator_dim 2 --num_steps 25 --step_size 0.1 \
               --n_train_glyphs 95 --epochs 50
```

### Training Options

- `--model_type`: Choose between `kuramoto` (oscillator-based) or `baseline` (CNN+MLP)
- `--embedding_dim`: Dimension of feature embeddings (D)
- `--oscillator_dim`: Dimension of each oscillator (N)
- `--num_steps`: Number of Kuramoto update steps (T)
- `--n_train_glyphs`: Number of characters used in training (generalization regime)
- `--disable_between`: Flag to ablate between-object coupling
- `--use_omega`: Flag to enable natural frequency matrices
- `--symmetric_j`: Flag to enforce symmetry in coupling matrices

### Evaluating Models

The training process automatically evaluates models on test data from the specified generalization regime after training. Results are logged to the console and to Weights & Biases (if enabled).

## Model Architecture

The architecture consists of four main components:

1. **Feature Extraction**: A CNN encodes input images into feature embeddings
2. **Kuramoto Oscillator Network**: Implements binding through synchronization dynamics
3. **Coherence Measurement**: Computes oscillator phase alignment as a relational bottleneck
4. **Classification**: Task-specific MLP classifiers

### Oscillator Dynamics

Each feature dimension is represented by an oscillator evolving on an N-dimensional unit sphere. Oscillators are influenced by:

- **Natural Frequency**: Intrinsic rotation specified by a skew-symmetric matrix
- **Within-Object Coupling**: Forces driving synchronization between different features of the same object
- **Between-Object Coupling**: Forces driving synchronization between corresponding features across objects

The update rule implements first-order Euler integration with normalization to maintain the unit sphere constraint:

```
F_total = c_terms + F_within + F_between
Δx = F_natural + Proj_x(F_total)
x(t+1) = normalize(x(t) + γ·Δx)
```

## Tasks

### Same/Different Discrimination

Given a pair of images, determine whether they depict the same character or different characters.

### Relational Match-to-Sample (RMTS)

Given a source pair of images that establishes a relation (either "same" or "different"), identify which of two target pairs exhibits the same relation as the source.

### Generalization Regimes

To evaluate the model's ability to generalize abstract rules to novel stimuli, we consider three training scenarios:

- **Extensive Training (m=95)**: 95 characters used in training, 5 reserved for testing
- **Moderate Generalization (m=50)**: 50 characters for training, 50 for testing
- **Limited Training (m=15)**: Only 15 characters available during training, 85 for testing

## Experimental Results

The oscillator-based model exhibits several advantages over the baseline CNN+MLP architecture:

1. **Faster Convergence**: Especially in the RMTS transfer learning task
2. **Better Generalization**: Consistently higher test accuracy across generalization regimes
3. **Energy Separation**: The trained model develops an energy landscape that separates "same" from "different" inputs
4. **Interpretable Dynamics**: Phase portraits reveal global synchronization for "same" pairs and partial synchronization for "different" pairs

## Citation

If you use this code or model in your research, please cite:

```
@article{veliz2023binding,
  title={Binding by Oscillatory Dynamics in Neural Architectures for Relational Reasoning},
  author={Veliz, Elder G.},
  journal={[Yale University] Undergraduate Thesis},
  year={2025}
}
```

## Acknowledgments

This work builds upon insights from several areas:

- The binding-by-synchrony hypothesis in neuroscience
- Kuramoto oscillator models in physics
- Relational bottleneck principles in machine learning

Special thanks to Professor John D. Lafferty and Awni Altabaa for their contributions and support.