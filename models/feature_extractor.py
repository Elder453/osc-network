"""
CNN feature extractor for converting images to embeddings.

Extracts feature embeddings from input images using a simple
convolutional neural network architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    """
    CNN feature extractor to convert images to feature embeddings.
    
    Parameters
    ----------
    input_channels : int
        Number of input image channels
    embedding_dim : int
        Dimension of output embeddings
        
    Attributes
    ----------
    conv1 : nn.Conv2d
        First convolutional layer
    conv2 : nn.Conv2d
        Second convolutional layer
    fc : nn.Linear
        Fully connected output layer
    flatten : nn.Flatten
        Flattening layer
    """
    def __init__(self, input_channels: int = 1, embedding_dim: int = 64):
        super(FeatureExtractor, self).__init__()
        self.embedding_dim = embedding_dim

        # cnn architecture
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.fc = nn.Linear(32 * 8 * 8, embedding_dim)
        self.flatten = nn.Flatten()

        # parameter initialization
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.conv1.bias)
        
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.conv2.bias)
        
        nn.init.kaiming_uniform_(self.fc.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of feature extractor.
        
        Parameters
        ----------
        x : torch.Tensor
            Input image tensor [batch_size, channels, height, width]
        
        Returns
        -------
        torch.Tensor
            Feature embedding [batch_size, embedding_dim]
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.fc(x)
        return x