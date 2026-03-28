import torch
import torch.nn as nn
import geoopt


class EntropicDissonanceLoss(nn.Module):
    """
    V3 Loss Function for Resonance Mapping Networks.
    
    Fix 3: Entropic Dissonance DOES NOT REPLACE Cross-Entropy. 
    It augments standard language modeling to prevent paradox collapse.
    
    L_Total = L_CE + alpha * (L_dissonance + lambda * L_coherence)
    
    This module implements the constraint penalty:
    - L_dissonance: rewards the network for sustaining productive tension (r > 0)
      when encountering paradoxical data (Isfet)
    - L_coherence: penalizes embeddings whose hyperbolic distance exceeds a 
      maximum threshold, preventing gradient exploitation where the optimizer
      pushes all embeddings to the Poincaré disk boundary
    """
    def __init__(self, d_max=5.0, lambda_coherence=0.1):
        """
        Args:
            d_max: Maximum allowed hyperbolic distance before coherence penalty applies.
                   Controls the radius of the "allowed zone" on the Poincaré disk.
            lambda_coherence: Weight of the coherence regularizer. Higher values
                              enforce tighter clustering; lower values allow more spread.
        """
        super().__init__()
        self.d_max = d_max
        self.lambda_coherence = lambda_coherence
        self.manifold = geoopt.PoincareBall()
    
    def forward(self, tension, q_hyp, k_hyp):
        """
        Args:
            tension: (batch, seq, 1) - the computed tension scalar r
            q_hyp: (batch, seq, dim) - query embeddings on the Poincaré disk
            k_hyp: (batch, seq, dim) - key embeddings on the Poincaré disk
        
        Returns:
            loss: scalar - the combined Entropic Dissonance Regularization Penalty.
                  This should be added to standard Cross-Entropy Loss.
        """
        # L_dissonance: reward productive tension
        # We want r > 0 when paradox is present, so we penalize negative tension
        # and reward positive tension via a soft hinge
        l_dissonance = -torch.mean(torch.clamp(tension, min=-1.0, max=5.0))
        
        # L_coherence: penalize excessive hyperbolic distance
        # This prevents the optimizer from farming gradient rewards by pushing
        # all embeddings to the disk boundary
        hyp_dist = self.manifold.dist(q_hyp, k_hyp)  # (batch, seq, dim)
        mean_dist = hyp_dist.mean(dim=-1)  # (batch, seq)
        
        # Hinge penalty: only activates when distance exceeds d_max
        l_coherence = torch.mean(torch.clamp(mean_dist - self.d_max, min=0.0))
        
        # Combined regularization penalty
        loss = l_dissonance + self.lambda_coherence * l_coherence
        
        return loss
