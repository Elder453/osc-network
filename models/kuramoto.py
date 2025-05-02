"""
Kuramoto oscillator network for relational learning.

Implements coupled oscillator dynamics for learning relationships
between features extracted from pairs of images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

from models.feature_extractor import FeatureExtractor
from models.coherence import CoherenceMeasurement
from models.classification import ClassificationMLP


@torch.jit.script
def compute_within_coupling_optimized(oscillators: torch.Tensor, J_in: torch.Tensor) -> torch.Tensor:
    """
    Compute within-object coupling forces with optimized implementation.
    
    Mathematical formula:
    F^{IN}_{i,d}(t) = \\sum_{d'=1}^D J^{IN}_{d,d'} x_{i,d'}(t)
    
    Parameters
    ----------
    oscillators : torch.Tensor
        Oscillator states [batch_size, D, N]
    J_in : torch.Tensor
        Within-object coupling matrix [D, D, N, N]
    
    Returns
    -------
    torch.Tensor
        Within-object coupling forces [batch_size, D, N]
    """
    # using einsum for efficient computation
    return torch.einsum("ijkl,bjl->bik", J_in, oscillators)


@torch.jit.script
def compute_between_coupling_optimized(oscillators_other: torch.Tensor, J_out: torch.Tensor) -> torch.Tensor:
    """
    Compute between-object coupling forces with optimized implementation.
    
    Mathematical formula:
    F^{OUT}_{i,d}(t) = \\sum_{d'=1}^D J^{OUT}_{d,d'} x_{3-i,d'}(t)
    
    Parameters
    ----------
    oscillators_other : torch.Tensor
        Other object's oscillator states [batch_size, D, N]
    J_out : torch.Tensor
        Between-object coupling matrix [D, D, N, N]
    
    Returns
    -------
    torch.Tensor
        Between-object coupling forces [batch_size, D, N]
    """
    return torch.einsum("ijkl,bjl->bik", J_out, oscillators_other)


class KuramotoOscillatorNetwork(nn.Module):
    """
    Implements Kuramoto dynamics on feature embeddings for relational learning.
    
    Takes feature embeddings from two objects and applies Kuramoto dynamics to create
    oscillator synchronization patterns that reflect the relationship between the objects.
    
    Parameters
    ----------
    embedding_dim : int
        Dimension D of feature embeddings
    oscillator_dim : int
        Dimension N of each oscillator
    num_steps : int
        Number of Kuramoto update steps (T)
    step_size : float
        Step size for discrete updates (gamma)
    use_omega : bool
        Whether to use natural frequency matrices
    omega_kappa : float
        Concentration parameter for sampling natural frequencies
    disable_between : bool
        Whether to disable between-object coupling
    symmetric_j : bool
        Whether to enforce symmetry in coupling matrices
    """
    def __init__(
        self, 
        embedding_dim: int, 
        oscillator_dim: int, 
        num_steps: int = 25, 
        step_size: float = 0.1,
        use_omega=True, 
        omega_kappa=2.0,
        disable_between: bool = False,
        symmetric_j: bool = False,
    ):
        super(KuramotoOscillatorNetwork, self).__init__()
        self.embedding_dim = embedding_dim    # D
        self.oscillator_dim = oscillator_dim  # N
        self.num_steps = num_steps            # T
        self.step_size = step_size            # gamma
        self.symmetric_j = symmetric_j
        
        # natural frequency matrices
        self.use_omega = use_omega
        if self.use_omega:
            Omega = self.sample_fixed_omegas(embedding_dim,
                                    oscillator_dim,
                                    kappa=omega_kappa,
                                    max_freq=1.0,
                                    device='cpu')
            # register as buffer (no grad, moves with .to(device))
            self.register_buffer("Omega", Omega)
        else:
            self.register_buffer("Omega", torch.zeros(
                embedding_dim, oscillator_dim, oscillator_dim))

        # initialize parameters
        self.W_d = nn.Parameter(torch.empty(embedding_dim, oscillator_dim))
        self.b_d = nn.Parameter(torch.empty(embedding_dim, oscillator_dim))
        nn.init.xavier_uniform_(self.W_d)
        nn.init.zeros_(self.b_d)

        # dense coupling parameters
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

    @staticmethod
    def _symmetrize(J: torch.Tensor) -> torch.Tensor:
        """
        Symmetrize a coupling matrix across feature and oscillator dimensions.
        
        Parameters
        ----------
        J : torch.Tensor
            Coupling matrix to symmetrize
            
        Returns
        -------
        torch.Tensor
            Symmetrized coupling matrix
        """
        # symmetrize across feature dims (d,d′) then oscillator dims (n,n′)
        J = 0.5 * (J + J.permute(1, 0, 2, 3))
        return 0.5 * (J + J.permute(0, 1, 3, 2))

    @property
    def J_in_sym(self) -> torch.Tensor:
        """
        Return symmetrized version of within-object coupling matrix if needed.
        """
        return self._symmetrize(self.J_in) if self.symmetric_j else self.J_in

    @property
    def J_out_sym(self) -> torch.Tensor:
        """
        Return symmetrized version of between-object coupling matrix if needed.
        """
        if self.disable_between:
            return self.J_out          # zero buffer
        return self._symmetrize(self.J_out) if self.symmetric_j else self.J_out
                
    def compute_conditional_stimulus(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute conditional stimulus (data terms) from embeddings.
        
        Parameters
        ----------
        embeddings : torch.Tensor
            Feature embeddings [batch_size, D]
        
        Returns
        -------
        torch.Tensor
            Conditional stimulus terms [batch_size, D, N]
        """
        batch_size, D = embeddings.shape
        
        # expand embeddings to [batch_size, D, 1]
        feature_values = embeddings.unsqueeze(2)
        
        # multiply by weights to get [batch_size, D, N]
        # c_{i,d} = W_d * E_i[d] + b_d
        c_terms = feature_values * self.W_d.unsqueeze(0) + self.b_d.unsqueeze(0)
        return c_terms   
    
    def initialize_oscillators(self, c_terms: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Initialize oscillators with a small, feature-aligned phase vector.
        
        Parameters
        ----------
        c_terms : torch.Tensor
            Conditional stimulus terms
        alpha : float
            Strength of feature alignment
            
        Returns
        -------
        torch.Tensor
            Initialized oscillators
        """
        sigma = 0.05  # noise std-dev
        noise = torch.randn_like(c_terms) * sigma
        oscillators = noise + alpha * c_terms  # feature-aligned start
        oscillators = F.normalize(oscillators, p=2, dim=2)
        return oscillators

    @staticmethod
    def sample_fixed_omegas(D: int, N: int,
                            kappa: float = 5.0,
                            max_freq: float = 1.0,
                            device=None) -> torch.Tensor:
        """
        Sample fixed natural frequencies for oscillators.
        
        Parameters
        ----------
        D : int
            Feature dimension
        N : int
            Oscillator dimension
        kappa : float
            Concentration parameter for von Mises distribution
        max_freq : float
            Maximum frequency magnitude
        device : torch.device, optional
            Device to create tensor on
            
        Returns
        -------
        torch.Tensor
            Natural frequency matrix Ω ∈ ℝ[D, N, N]
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
        # skew-symm: Ω_d = ω_d (u_d ⊗ û_d – û_d ⊗ u_d)
        Omega = omega.view(-1, 1, 1) * (u.unsqueeze(2) - u.unsqueeze(1)) / 2
        return Omega
    
    def project_to_tangent_space(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Project vector y onto the tangent space at x.
        
        Parameters
        ----------
        x : torch.Tensor
            Base point on the sphere [batch_size, D, N]
        y : torch.Tensor
            Vector to project [batch_size, D, N]
        
        Returns
        -------
        torch.Tensor
            Projection of y onto tangent space at x [batch_size, D, N]
        """
        # calculate the dot product <x, y> along the last dimension
        dot_product = torch.sum(x * y, dim=2, keepdim=True)
        
        # projection formula: y - <x, y>x
        return y - dot_product * x
    
    def compute_within_coupling(self, oscillators: torch.Tensor) -> torch.Tensor:
        """
        Compute within-object coupling forces.
        
        Parameters
        ----------
        oscillators : torch.Tensor
            Oscillator states [batch_size, D, N]
        
        Returns
        -------
        torch.Tensor
            Within-object coupling forces [batch_size, D, N]
        """
        J_in = self.J_in_sym
        return compute_within_coupling_optimized(oscillators, J_in)
    
    def compute_between_coupling(self, oscillators_other: torch.Tensor) -> torch.Tensor:
        """
        Compute between-object coupling forces.
        
        Parameters
        ----------
        oscillators_other : torch.Tensor
            OTHER object's oscillators [batch_size, D, N]
        
        Returns
        -------
        torch.Tensor
            Between-object coupling forces for first object [batch_size, D, N]
        """
        if self.disable_between:
            # same shape as "within" result, but all zeros
            return torch.zeros_like(oscillators_other) 
        J_out = self.J_out_sym
        return compute_between_coupling_optimized(oscillators_other, J_out)

    def calculate_energy(self, 
                        x1: torch.Tensor, 
                        x2: torch.Tensor,
                        c1: torch.Tensor, 
                        c2: torch.Tensor) -> torch.Tensor:
        """
        Calculate Lyapunov energy of the Kuramoto system.
        
        Uses the formula: E = -1/2 * sum_{i,j} x_i^T J_{ij} x_j - sum_i c_i^T x_i
        
        Parameters
        ----------
        x1, x2 : torch.Tensor
            Oscillator states [batch_size, D, N]
        c1, c2 : torch.Tensor
            Conditional stimulus terms [batch_size, D, N]
        
        Returns
        -------
        torch.Tensor
            Energy values for batch [batch_size]
        """
        # get the appropriate J matrices
        J_in = self.J_in_sym

        # coupling energy
        # within-object terms
        E_w1 = torch.einsum('bdn,ijkl,bjl->b',
                             x1, J_in, x1)          # x1ᵀ J_in x1
        E_w2 = torch.einsum('bdn,ijkl,bjl->b',
                             x2, J_in, x2)
        # between-object (double-count avoided by *no* factor ½ here)
        if self.disable_between:
            E_bt = 0.0
        else:
            # x1ᵀ J_out x2 + x2ᵀ J_out x1
            J_out = self.J_out_sym
            E_bt_1to2 = torch.einsum('bdn,ijkl,bjl->b', x1, J_out, x2)
            E_bt_2to1 = torch.einsum('bdn,ijkl,bjl->b', x2, J_out, x1)
            E_bt = E_bt_1to2 + E_bt_2to1
    
        E_cpl = - 0.5 * (E_w1 + E_w2 + E_bt)              # final coupling energy
    
        # stimulus alignment
        E_stim = -(x1 * c1).sum((1, 2)) - (x2 * c2).sum((1, 2))
    
        # total
        return E_cpl + E_stim        # shape [batch]
    
    def kuramoto_step(self, 
                     oscillators1: torch.Tensor, 
                     oscillators2: torch.Tensor,
                     c_terms1: torch.Tensor, 
                     c_terms2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one step of Kuramoto dynamics update.
        
        Parameters
        ----------
        oscillators1 : torch.Tensor
            First object's oscillators [batch_size, D, N]
        oscillators2 : torch.Tensor
            Second object's oscillators [batch_size, D, N]
        c_terms1 : torch.Tensor
            First object's conditional stimulus terms [batch_size, D, N]
        c_terms2 : torch.Tensor
            Second object's conditional stimulus terms [batch_size, D, N]
        
        Returns
        -------
        tuple
            Updated oscillators for both objects
        """
        # natural frequency Ω x
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
                    c_terms2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Run Kuramoto dynamics for T steps.
        
        Parameters
        ----------
        oscillators1 : torch.Tensor
            Initial oscillators for first object [batch_size, D, N]
        oscillators2 : torch.Tensor
            Initial oscillators for second object [batch_size, D, N]
        c_terms1 : torch.Tensor
            Conditional stimulus terms for first object [batch_size, D, N]
        c_terms2 : torch.Tensor
            Conditional stimulus terms for second object [batch_size, D, N]
        
        Returns
        -------
        tuple
            Final oscillator states, trajectories, and energy values
        """
        current_oscillators1 = oscillators1
        current_oscillators2 = oscillators2
        
        # store all states for visualization
        osc_trajectory1 = [current_oscillators1]
        osc_trajectory2 = [current_oscillators2]

        # initialize energy tracking
        energy_values = []
        
        # calculate initial energy
        initial_energy = self.calculate_energy(
            current_oscillators1, current_oscillators2, 
            c_terms1, c_terms2
        )
        energy_values.append(initial_energy)
        
        # run for num_steps
        for t in range(self.num_steps):
            current_oscillators1, current_oscillators2 = self.kuramoto_step(
                current_oscillators1, current_oscillators2, 
                c_terms1, c_terms2
            )
            
            osc_trajectory1.append(current_oscillators1)
            osc_trajectory2.append(current_oscillators2)

            # calculate energy after this step
            step_energy = self.calculate_energy(
                current_oscillators1, current_oscillators2, 
                c_terms1, c_terms2
            )
            energy_values.append(step_energy)
        
        # return final states and all intermediate states
        return current_oscillators1, current_oscillators2, osc_trajectory1, osc_trajectory2, energy_values
    
    def forward(self, 
               embeddings1: torch.Tensor, 
               embeddings2: torch.Tensor,
               alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Kuramoto Oscillator Network.
        
        Parameters
        ----------
        embeddings1 : torch.Tensor
            Feature embedding of first object [batch_size, D]
        embeddings2 : torch.Tensor
            Feature embedding of second object [batch_size, D]
        alpha : float
            Strength of feature alignment in initialization
        
        Returns
        -------
        tuple
            Final oscillator states, trajectories, energy values, and coupling terms
        """
        # compute data terms
        c_terms1 = self.compute_conditional_stimulus(embeddings1)
        c_terms2 = self.compute_conditional_stimulus(embeddings2)
        
        # initialize oscillators
        oscillators1 = self.initialize_oscillators(c_terms1, alpha)
        oscillators2 = self.initialize_oscillators(c_terms2, alpha)
        
        # run kuramoto dynamics
        final_osc1, final_osc2, osc_trajectory1, osc_trajectory2, energy_values = self.run_dynamics(
            oscillators1, oscillators2, 
            c_terms1, c_terms2
        )
        
        return (final_osc1, final_osc2, 
                osc_trajectory1, osc_trajectory2, 
                energy_values, 
                c_terms1, c_terms2)


class KuramotoRelationalModel(nn.Module):
    """
    Complete Kuramoto-based relational learning model.
    
    Combines feature extraction, oscillator network, coherence measurement, and classification.
    
    Parameters
    ----------
    input_channels : int
        Number of input image channels
    embedding_dim : int
        Dimension D of feature embeddings
    oscillator_dim : int
        Dimension N of each oscillator
    num_steps : int
        Number of Kuramoto update steps T
    step_size : float
        Step size for Kuramoto updates (gamma)
    mlp_hidden_dim : int
        Hidden layer dimension of classification MLP
    use_omega : bool
        Whether to use natural frequency matrices
    omega_kappa : float
        Concentration parameter for sampling natural frequencies
    disable_between : bool
        Whether to disable between-object coupling
    symmetric_j : bool
        Whether to enforce symmetry in coupling matrices
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
    
    def forward(self, image1: torch.Tensor, image2: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, tuple]:
        """
        Forward pass through the complete model.
        
        Parameters
        ----------
        image1 : torch.Tensor
            First image tensor [batch_size, channels, height, width]
        image2 : torch.Tensor
            Second image tensor [batch_size, channels, height, width]
        alpha : float
            Strength of feature alignment in oscillator initialization
        
        Returns
        -------
        tuple
            (logits, oscillator_states)
            where oscillator_states contains final states, trajectories, energy values, etc.
        """
        # extract feature embeddings
        embedding1 = self.feature_extractor(image1)
        embedding2 = self.feature_extractor(image2)
        
        # run kuramoto dynamics
        final_osc1, final_osc2, osc_trajectory1, osc_trajectory2, energy_values, c1, c2 = self.oscillator_network(
            embedding1, embedding2, alpha)
        
        # compute coherence measures
        coherence_vector = self.coherence_measurement(final_osc1, final_osc2)
        
        # classify based on coherence
        relation_prob = self.classifier(coherence_vector)
        
        return relation_prob, (final_osc1, final_osc2, 
                               osc_trajectory1, osc_trajectory2, 
                               energy_values,
                               c1, c2)

    @torch.no_grad()
    def get_coherence_vector(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Get the coherence vector for a pair of images.
        
        Parameters
        ----------
        img1 : torch.Tensor
            First image tensor
        img2 : torch.Tensor
            Second image tensor
            
        Returns
        -------
        torch.Tensor
            Coherence vector [batch_size, D]
        """
        self.eval()
        # extract embeddings
        emb1 = self.feature_extractor(img1)   # [batch_size, D]
        emb2 = self.feature_extractor(img2)   # [batch_size, D]
        
        # run oscillator network
        final_osc1, final_osc2, *_ = self.oscillator_network(emb1, emb2)
        
        # measure coherence
        coherence_vec = self.coherence_measurement(final_osc1, final_osc2)  # [batch_size, D]
                
        return coherence_vec