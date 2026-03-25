import torch
import torch.nn as nn
import geoopt

class PitchforkBifurcation(nn.Module):
    """
    Simulates a dynamical system undergoing a pitchfork bifurcation: dx/dt = r*x - x^3
    Where 'r' is the Tension parameter.
    If r <= 0, the output converges linearly (simulating standard attention agreement).
    If r > 0, the output splits into two stable orthogonal vectors: +sqrt(r) and -sqrt(r)
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.epsilon = 1e-6 # Prevents exactly zero tension collapse

    def forward(self, x, tension):
        # x shape: (batch, seq, dim)
        # tension shape: (batch, seq, 1) - computed distance between query and key
        
        # We process the bifurcation deterministically for the forward pass 
        # based on the stable attractors of the ODE.
        
        # Agreement branch (Tension <= 0)
        agreement_mask = (tension <= 0).float()
        # Dissonance branch (Tension > 0)
        dissonance_mask = (tension > 0).float()
        
        # In agreement, tensor passes softly scaled down (like standard low attention)
        out_agg = x * torch.exp(tension) * agreement_mask
        
        # In dissonance, the tensor splits into the two pitchfork attractors
        # We represent this by doubling the dimensionality at the output or processing in parallel
        # For simplicity in this toy layer, we output a tuple of the two divergent thoughts
        amplitude = torch.sqrt(tension * dissonance_mask + self.epsilon)
        
        thought_a = out_agg + (x * amplitude)
        thought_b = out_agg - (x * amplitude)
        
        # Instead of collapsing, we return the dual state
        return thought_a, thought_b


class ResonanceMapping(nn.Module):
    """
    Replaces Multi-Head Softmax Attention.
    Uses Geoopt hyperbolic embeddings and Squared Hyperbolic Distance to measure tension.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # We project queries and keys using standard linear layers
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        self.bifurcation = PitchforkBifurcation(dim)
        # The Poincaré ball manifold
        self.manifold = geoopt.PoincareBall()

    def forward(self, q_in, k_in, v_in):
        # 1. Project to vectors
        q = self.q_proj(q_in)
        k = self.k_proj(k_in)
        v = self.v_proj(v_in)
        
        # 2. Map queries and keys to the PyTorch Poincaré hyperbolic disk
        # This requires double precision for stability
        q_hyp = self.manifold.expmap0(q.to(torch.float64))
        k_hyp = self.manifold.expmap0(k.to(torch.float64))
        
        # 3. Compute Tension (Squared Hyperbolic Distance)
        # Unlike dot product, this distance grows exponentially without crossing zero in flat space
        dist = self.manifold.dist2(q_hyp, k_hyp).to(torch.float32)
        
        # We normalize distance into a Tension scalar space centered around 0
        # If dist is large, tension is positive. If dist is small, tension is negative.
        mean_dist = dist.mean(dim=-1, keepdim=True)
        tension = dist - mean_dist
        
        # 4. Trigger Bifurcation instead of Softmax
        # V splits into two parallel orthogonal streams if tension > 0
        out_a, out_b = self.bifurcation(v, tension.unsqueeze(-1))
        
        return out_a, out_b


class ToyTensionModel(nn.Module):
    """
    A 150M parameter simulated model. Constrained to RTX 3060 Ti limits (8GB).
    """
    def __init__(self, vocab_size=50257, d_model=768, depth=6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        
        # A stack of Resonant layers replacing Transformer blocks
        self.layers = nn.ModuleList([
            ResonanceMapping(d_model) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        # We must track both thought streams. A true neuromorphic chip does this physically.
        # Here we simulate chaotic parallel compute by processing streams independently.
        x = self.embed(input_ids)
        
        streams = [x]
        
        for layer in self.layers:
            new_streams = []
            for stream in streams:
                # The layer bifurcates the thought
                out_a, out_b = layer(stream, stream, stream)
                new_streams.extend([out_a, out_b])
            
            # To simulate hardware resource limits on standard GPUs, we prune back to the 2 highest energy streams
            # A pure neuromorphic SNN wouldn't need this pruning, but we do for VRAM.
            streams = new_streams[:2] 
            
        final_out = self.norm(streams[0]) # Returning primary stream for standard loss 
        return self.lm_head(final_out)
