"""
Classification models for Kuramoto Relational Learning.

Contains MLP classifiers for both same/different task and
relational match-to-sample task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationMLP(nn.Module):
    """
    MLP for same/different classification.
    
    Takes coherence feature vector and outputs probability of "same" relation.
    
    Parameters
    ----------
    input_dim : int
        Dimension of the coherence feature vector
    hidden_dim : int
        Dimension of hidden layer
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super(ClassificationMLP, self).__init__()
        
        # mlp layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.
        
        Parameters
        ----------
        x : torch.Tensor
            Coherence feature vector [batch_size, input_dim]
        
        Returns
        -------
        torch.Tensor
            Logits for "same" relation [batch_size, 1]
        """
        x = F.relu(self.fc1(x))
        logit = self.fc2(x)
        return logit


class RMTSClassifier(nn.Module):
    """
    Classifier for the relational match-to-sample task.
    
    Takes three coherence vectors and outputs logits for which target matches the source relation.
    
    Parameters
    ----------
    coherence_dim : int
        Dimension of coherence vectors
    hidden_dim : int
        Dimension of hidden layer
    """
    def __init__(self, coherence_dim=64, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(3 * coherence_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)  # 2 classes => T1=correct or T2=correct

    def forward(self, source_vec, t1_vec, t2_vec):
        """
        Forward pass through the classifier.
        
        Parameters
        ----------
        source_vec : torch.Tensor
            Source coherence vector
        t1_vec : torch.Tensor
            First target coherence vector
        t2_vec : torch.Tensor
            Second target coherence vector
            
        Returns
        -------
        torch.Tensor
            Logits for classification [batch_size, 2]
        """
        # shape [batch_size, 3D]
        x = torch.cat([source_vec, t1_vec, t2_vec], dim=1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)  # shape [batch_size, 2]
        return logits
