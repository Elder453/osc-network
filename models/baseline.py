"""
Baseline models that don't use Kuramoto dynamics.

Contains CNN-based baseline models for comparison with
Kuramoto oscillator networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.feature_extractor import FeatureExtractor
from models.coherence import CoherenceMeasurement


class BaselineSameDiffModel(nn.Module):
    """
    Baseline model for same/different classification without Kuramoto dynamics.
    
    Uses a shared CNN encoder followed by a 2-layer MLP classifier.
    
    Parameters
    ----------
    input_channels : int
        Number of input image channels
    embedding_dim : int
        Dimension of feature embeddings
    oscillator_dim : int
        Dimension of oscillator space (used for compatibility with Kuramoto model)
    """
    def __init__(self, 
                 input_channels: int = 1,
                 embedding_dim: int = 64,
                 oscillator_dim: int = 2):
        super().__init__()
        self.feature_extractor = FeatureExtractor(
            input_channels=input_channels,
            embedding_dim=embedding_dim,
        )

        # project to R^{D×N} space (similar to c_terms and implicitly osc states)
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

    def forward(self, img1: torch.Tensor, img2: torch.Tensor, alpha: float = 1.0):
        """
        Forward pass through the model.
        
        Parameters
        ----------
        img1 : torch.Tensor
            First input image
        img2 : torch.Tensor
            Second input image
        alpha : float
            Not used in baseline model, included for API compatibility
            
        Returns
        -------
        tuple
            (logits, auxiliary_data)
        """
        e1 = self.feature_extractor(img1)          # [B,D]
        e2 = self.feature_extractor(img2)          # [B,D]
    
        # project to R^{D×N}
        p1 = self.projector(e1).view(-1, self.embedding_dim, self.oscillator_dim)
        p2 = self.projector(e2).view(-1, self.embedding_dim, self.oscillator_dim)
        
        # normalize (like oscillators)
        p1 = F.normalize(p1, p=2, dim=2) 
        p2 = F.normalize(p2, p=2, dim=2)
        
        # compute coherence directly
        coherence_vector = self.coherence_measurement(p1, p2)
        
        # classify
        logit = self.classifier(coherence_vector)
        
        # return a structure compatible with the kuramoto model's output
        # - p1, p2 as final oscillator states
        # - [p1], [p2] as trajectories (just single-step here)
        # - None for energy values
        # - None for c_terms
        return logit, (p1, p2, [p1], [p2], None, None, None)

    @torch.no_grad()
    def get_coherence_vector(self, imgA: torch.Tensor, imgB: torch.Tensor):
        """
        Return D-dim coherence vector for an (A,B) pair.
        
        Parameters
        ----------
        imgA : torch.Tensor
            First input image
        imgB : torch.Tensor
            Second input image
            
        Returns
        -------
        torch.Tensor
            Coherence vector
        """
        self.eval()
        eA = self.feature_extractor(imgA)
        eB = self.feature_extractor(imgB)
        
        # project to R^{D×N}
        pA = self.projector(eA).view(-1, self.embedding_dim, self.oscillator_dim)
        pB = self.projector(eB).view(-1, self.embedding_dim, self.oscillator_dim)
        
        # normalize
        pA = F.normalize(pA, p=2, dim=2)
        pB = F.normalize(pB, p=2, dim=2)
        
        # compute coherence
        coherence_vector = self.coherence_measurement(pA, pB)
        
        return coherence_vector
    
class BaselineRMTSClassifier(nn.Module):
    """
    Classifier for the relational match-to-sample task using baseline model.
    
    Takes three coherence vectors and classifies which target matches the source relation.
    
    Parameters
    ----------
    coherence_dim : int
        Dimension of coherence vectors
    hidden_dim : int
        Dimension of hidden layer
    """
    def __init__(self, coherence_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(3 * coherence_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, src_vec, t1_vec, t2_vec):
        """
        Forward pass through the classifier.
        
        Parameters
        ----------
        src_vec : torch.Tensor
            Source coherence vector
        t1_vec : torch.Tensor
            First target coherence vector
        t2_vec : torch.Tensor
            Second target coherence vector
            
        Returns
        -------
        torch.Tensor
            Logits for classification
        """
        x = torch.cat([src_vec, t1_vec, t2_vec], dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)