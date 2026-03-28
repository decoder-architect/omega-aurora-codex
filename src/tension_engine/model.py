import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt

class PitchforkBifurcation(nn.Module):
    """
    Implements the Pitchfork Bifurcation ODE: dx/dt = r*x - x^3
    
    If r <= 0: attractor at x=0 (agreement)
    If r > 0: instantaneous stochastic jump exploring both branches ±c*tanh(r/c) (dissonance)
    """
    def __init__(self, dim, manifold):
        super().__init__()
        self.dim = dim
        self.epsilon = 1e-6
        self.manifold = manifold
        self.c = 1.0
        self.tangent_scale = nn.Parameter(torch.tensor(0.1))  # Keeps tangents in tanh linear regime

    def forward(self, h_comp, tension, w_proj):
        # 1. Project: extract scalar magnitude using isomorphic unclipped vectors
        # v16: h_comp represents the safe asinh-compressed sequence magnitude
        x = torch.sum(h_comp * w_proj, dim=-1, keepdim=True)  # (batch, seq, 1)
        
        # 2. Compute stable attractor x* via Differentiable stochastic analytical jump
        agreement_mask = (tension <= 0).float()
        dissonance_mask = (tension > 0).float()
        
        x_star_agree = torch.zeros_like(x) * agreement_mask
        
        # The attractor amplitude dynamically scales precisely with the Compressed Tangent Envelope
        h_magnitude = torch.norm(h_comp, dim=-1, keepdim=True)
        amplitude = h_magnitude * self.c * torch.tanh(tension / self.c) * dissonance_mask
        
        # V21 FIX: Explicit Gumbel Vector Dot Product Formalization
        # Formally dynamically dot-product the orthogonal limits to safely parameterize the continuous map natively.
        logits = torch.zeros(x.size(0), x.size(1), 2, device=x.device, dtype=x.dtype)
        gumbel_prob = F.gumbel_softmax(logits, tau=0.5, hard=True, dim=-1)
        gumbel_vector = torch.tensor([1.0, -1.0], device=x.device, dtype=x.dtype)
        epsilon_gumbel = torch.matmul(gumbel_prob, gumbel_vector).unsqueeze(-1)
        
        x_star_dissonance = epsilon_gumbel * amplitude
        
        x_star = x_star_agree + x_star_dissonance
        
        # V23: Scale tangent into tanh linear regime before Exp map
        delta = x_star - x
        h_prime = h_comp + delta * w_proj
        h_prime_scaled = h_prime * self.tangent_scale
        h_prime_hyp = self.manifold.expmap0(h_prime_scaled.to(torch.float64))
        
        return h_prime_hyp


class ResonanceMapping(nn.Module):
    """
    Replaces Multi-Head Softmax Attention.
    V16 Fixes (The Crucible):
    - Geodesic Temperature float32 Stabilization (subtract min distance)
    - O(N*d) Matrix-Free Power Iteration for TPCA
    - Inner-Tangent Asinh Compression for displacement survivability
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        # Möbius-legal output weight (float64 for manifold ops)
        self.o_weight = nn.Parameter(torch.randn(dim, dim, dtype=torch.float64) * 0.02)
        
        self.manifold = geoopt.PoincareBall()
        self.bifurcation = PitchforkBifurcation(dim, self.manifold)
        
        self.tau = nn.Parameter(torch.tensor(1.5))
        
    def einstein_midpoint(self, points_hyp, weights):
        """
        The O(1) Analytical Fréchet Extrapolation
        """
        tangent_vectors = self.manifold.logmap0(points_hyp) # (batch, 1, seq_context, dim)
        weights_norm = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8) # (batch, seq, seq_context)
        mean_tangent = torch.bmm(weights_norm, tangent_vectors.squeeze(1)) # (batch, seq, dim)
        
        centroid_unclipped = self.manifold.expmap0(mean_tangent)
        centroid_norm = torch.norm(centroid_unclipped, dim=-1, keepdim=True)
        centroid = centroid_unclipped * torch.clamp( (1.0 - 1e-5) / (centroid_norm + 1e-8), max=1.0 )
        
        return centroid

    def forward(self, q_in, k_in, v_in):
        batch_size, seq_len, dim = q_in.size()
        
        q = self.q_proj(q_in)
        k = self.k_proj(k_in)
        v = self.v_proj(v_in)
        
        q_hyp = self.manifold.expmap0(q.to(torch.float64))
        k_hyp = self.manifold.expmap0(k.to(torch.float64))
        
        q_hyp_expand = q_hyp.unsqueeze(2)  
        k_hyp_expand = k_hyp.unsqueeze(1)  
        dist_matrix = self.manifold.dist(q_hyp_expand, k_hyp_expand).to(torch.float32)
        
        # V16 FIX: Geodesic Temperature Float32 Stabilization
        # Prevents exponential underflow to 0.0 by structurally shifting the frame of reference
        min_dist = torch.min(dist_matrix, dim=-1, keepdim=True)[0]
        dist_matrix_shifted = dist_matrix - min_dist
        weights_unnormalized = torch.exp(-dist_matrix_shifted) # (batch, seq, seq_context)
        
        # Base Residual Tangent Accumulation (Unbounded Velocity)
        h = torch.bmm(weights_unnormalized, v)  # (batch, seq, dim)
        
        k_centroid_hyp = self.einstein_midpoint(k_hyp_expand, weights_unnormalized.to(torch.float64))
        
        # Tension parameter calibration via explicit Fréchet Average Token Variance
        k_centroid_hyp_expand = k_centroid_hyp.unsqueeze(2) # (batch, seq, 1, dim)
        centroid_to_keys_dist = self.manifold.dist(k_centroid_hyp_expand, k_hyp_expand).to(torch.float32) # (batch, seq, seq_context)
        
        # V23: Dynamic tau — scales threshold with effective context mass
        effective_mass = torch.sum(weights_unnormalized, dim=-1, keepdim=True) + 1e-8
        variance = torch.sum(weights_unnormalized * (centroid_to_keys_dist ** 2), dim=-1, keepdim=True)
        tension = variance - self.tau * effective_mass
        
        # Pure Topological Centering
        k_local_tangent = self.manifold.logmap(k_centroid_hyp_expand, k_hyp_expand).to(torch.float32)
        weighted_k = k_local_tangent * torch.sqrt(weights_unnormalized.unsqueeze(-1) + 1e-8)
        
        # V16 FIX: Matrix-Free O(d) TPCA Extraction
        # Never constructs the memory-fatal (d x d) covariance matrix. Extracts vector natively via O(N*d) projections.
        v_rand = torch.randn(batch_size, seq_len, dim, 1, device=weighted_k.device, dtype=weighted_k.dtype)
        v_rand = F.normalize(v_rand, dim=2)
        
        for _ in range(3):
            # proj = (batch, seq, seq_context, 1)
            proj = torch.matmul(weighted_k, v_rand)
            # v_rand = (batch, seq, dim, 1)
            v_rand = torch.matmul(weighted_k.transpose(2, 3), proj)
            v_rand = F.normalize(v_rand, dim=2)
            
        w_proj_local = v_rand.squeeze(-1) # (batch, seq, dim)
        w_proj_global = self.manifold.transp0back(k_centroid_hyp, w_proj_local.to(torch.float64)).to(torch.float32)
        
        # If variance is identically 0, fallback to query-centroid direction to prevent NaN
        q_tangent = self.manifold.logmap0(q_hyp).to(torch.float32)
        k_centroid_tangent = self.manifold.logmap0(k_centroid_hyp).to(torch.float32)
        fallback_w_proj = F.normalize(q_tangent - k_centroid_tangent, dim=-1, eps=1e-8)
        
        variance_mask = (variance > 1e-5).expand_as(w_proj_global)
        w_proj = torch.where(variance_mask, w_proj_global, fallback_w_proj)
        
        # V16 FIX: Inner-Tangent Asinh Compression (Tangent Erasure Fix)
        # Prevents massive exponential tensor sums from being irreversibly clamped to 0.9999 by tanh.
        h_norm = torch.norm(h, dim=-1, keepdim=True)
        h_comp = torch.asinh(h_norm) * (h / (h_norm + 1e-8))
        
        # V23: Möbius matrix-vector multiplication + logmap0 return to Euclidean
        h_prime_hyp = self.bifurcation(h_comp, tension, w_proj)
        h_transformed = self.manifold.mobius_matvec(self.o_weight, h_prime_hyp)
        out = self.manifold.logmap0(h_transformed).to(torch.float32)
        
        # Spread Parameter (Phase volume metric)
        origin_dist = self.manifold.dist0(k_centroid_hyp).to(torch.float32)
        
        state = {
            'variance_matrix': centroid_to_keys_dist ** 2,
            'tension': tension,
            'tau': self.tau.item(),
            'w_proj': w_proj,
            'origin_dist': origin_dist,
            'x_in': torch.sum(h_comp * w_proj, dim=-1, keepdim=True),
            'envelope': h_norm,  # Raw unscaled exponential envelope
            'envelope_comp': torch.norm(h_comp, dim=-1, keepdim=True) # Successfully compressed envelope
        }
        
        return out, state


class ToyTensionModel(nn.Module):
    """
    A 150M parameter simulated model. Constrained to RTX 3060 Ti limits (8GB).
    Maintains constant memory layer-to-layer.
    """
    def __init__(self, vocab_size=50257, d_model=768, depth=6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        
        self.layers = nn.ModuleList([
            ResonanceMapping(d_model) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        
        for layer in self.layers:
            x, _ = layer(x, x, x)
        
        x = self.norm(x)
        return self.lm_head(x)
