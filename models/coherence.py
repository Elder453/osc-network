"""
Coherence measurement for oscillator states.

Implements methods to measure synchronization between oscillators.
"""

import torch
import torch.nn as nn


class CoherenceMeasurement(nn.Module):
    """
    Takes final oscillator states and computes coherence measures across multiple objects.
    
    Uses the norm of the sum for each feature dimension to quantify synchronization.
    
    Parameters
    ----------
    embedding_dim : int
        Dimension D of feature embeddings
    oscillator_dim : int
        Dimension N of each oscillator
    """
    def __init__(
        self, 
        embedding_dim: int, 
        oscillator_dim: int
    ):
        super(CoherenceMeasurement, self).__init__()
        self.embedding_dim = embedding_dim    # D
        self.oscillator_dim = oscillator_dim  # N
    
    def compute_coherence(self, *oscillators_list) -> torch.Tensor:
        """
        Compute coherence features.
        
        Uses the formula: 
            ρ = (ρ_1, ..., ρ_D), where ρ_d = ‖∑_{i=1}^n x_{i,d}(T)‖_2
            
        Parameters
        ----------
        *oscillators_list : torch.Tensor
            Variable number of oscillator tensors, each [batch_size, D, N]
        
        Returns
        -------
        torch.Tensor
            Coherence vector [batch_size, D]
        """
        # sum all oscillators
        summed_oscillators = sum(oscillators_list)  # shape: [batch_size, D, N]
        
        # compute L2 norm along the oscillator dimension (N)
        coherence = torch.norm(summed_oscillators, p=2, dim=2)
        
        return coherence
    
    def forward(self, *oscillators_list) -> torch.Tensor:
        """
        Forward pass to compute coherence.
        
        Parameters
        ----------
        *oscillators_list : torch.Tensor
            Variable number of oscillator tensors, each [batch_size, D, N]
        
        Returns
        -------
        torch.Tensor
            Coherence vector [batch_size, D]
        """
        return self.compute_coherence(*oscillators_list)