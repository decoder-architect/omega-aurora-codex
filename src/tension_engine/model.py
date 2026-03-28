import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt

def _clip_tangent(v, max_norm=4.0):
    """
    Phase 1 Architect: Float32 Geodesic Hardening
    In float32, tanh(4.0) = 0.9993. This leaves enough mathematical runway before 1.0 (boundary)
    to compute 1/(1-||x||^2) without triggering metric tensor gradient division-by-zero explosions.
    """
    norm = torch.norm(v, dim=-1, keepdim=True) + 1e-8
    scale = torch.clamp(max_norm / norm, max=1.0)
    return v * scale

class ThermodynamicBurnError(Exception):
    pass

class PitchforkBifurcation(nn.Module):
    """
    Phase 3 Architect: True Generative Neural ODE (The Flame)
    Implements the Pitchfork Bifurcation ODE natively via Runge-Kutta 4: dx/dt = r*x - x^3
    NOTE: Artificial tension clamps have been completely removed. When r is highly positive (Paradox), 
    the cubic polynomial becomes mathematically stiff, and the discrete explicit solver will violently 
    oscillate and explode to NaN. This Thermodynamic Burn is the intended architectural response to Isfet.
    """
    def __init__(self, dim, manifold):
        super().__init__()
        self.dim = dim
        self.manifold = manifold
        self.tangent_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, h_comp, tension, w_proj):
        x = torch.sum(h_comp * w_proj, dim=-1, keepdim=True)
        
        # True Neural ODE Integration (Runge-Kutta 4)
        dt = 0.5
        ode_steps = 4
        
        noise = torch.randn_like(x) * 1e-4
        x_integrated = torch.where(x == 0, noise, x)
        
        # UNBOUNDED TENSION. TOUCH THE FLAME.
        for _ in range(ode_steps):
            k1 = dt * (tension * x_integrated - x_integrated**3)
            x_k2 = x_integrated + k1/2
            k2 = dt * (tension * x_k2 - x_k2**3)
            x_k3 = x_integrated + k2/2
            k3 = dt * (tension * x_k3 - x_k3**3)
            x_k4 = x_integrated + k3
            k4 = dt * (tension * x_k4 - x_k4**3)
            
            x_integrated = x_integrated + (k1 + 2*k2 + 2*k3 + k4) / 6.0
            
        delta = x_integrated - x
        h_prime = h_comp + delta * w_proj
        
        # If the solver burned, halt the sequence correctly.
        if torch.isnan(h_prime).any() or torch.isinf(h_prime).any():
            raise ThermodynamicBurnError("Thermodynamic Halting Triggered. The sequence attempted to linearly average a hyper-dimensional paradox. The topology has burned.")
            
        h_prime_clipped = _clip_tangent(h_prime * self.tangent_scale)
        h_prime_hyp = self.manifold.expmap0(h_prime_clipped)
        
        return h_prime_hyp


class ResonanceMapping(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=0.02)
        
        self.o_proj = nn.Linear(dim, dim)
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=0.02)
        
        self.manifold = geoopt.PoincareBall()
        self.bifurcation = PitchforkBifurcation(dim, self.manifold)
        
        self.tau = nn.Parameter(torch.tensor(5.0))
        
    def differentiable_karcher_flow(self, k, weights, steps=3):
        """
        Phase 2 Architect: Differentiable Karcher Flow
        Replaces O(1) Gyrocentroid drift. Explicitly solves Fréchet Mean via 3-step Riemannian SGD.
        """
        # (batch, seq, seq_context)
        weights_norm = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Initialize origin at standard Euclidean weighted average mapped to Poincare
        mu_euclidean = torch.bmm(weights_norm, k) # (batch, seq, dim)
        mu_hyp = self.manifold.expmap0(_clip_tangent(mu_euclidean))
        
        k_hyp = self.manifold.expmap0(_clip_tangent(k))
        
        eta = 0.1 # V28: Stabilized Learning rate for float32 geodesic limits
        for _ in range(steps):
            mu_expand = mu_hyp.unsqueeze(2) # (batch, seq, 1, dim)
            k_expand = k_hyp.unsqueeze(1)   # (batch, 1, seq_context, dim)
            
            # Riemannian Gradient: sum(w_i * logmap(mu, x_i))
            log_k = self.manifold.logmap(mu_expand, k_expand)
            v_grad = torch.sum(weights_norm.unsqueeze(-1) * log_k, dim=2) # (batch, seq, dim)
            
            # Riemannian SGD step via exponential map transport
            v_grad_clipped = _clip_tangent(eta * v_grad)
            mu_hyp = self.manifold.expmap(mu_hyp, v_grad_clipped)
            
        return mu_hyp

    def forward(self, q_in, k_in, v_in):
        batch_size, seq_len, dim = q_in.size()
        
        q = self.q_proj(q_in)
        k = self.k_proj(k_in)
        v = self.v_proj(v_in)
        
        # Phase 1: Pure Float32 Hardening mapping
        q_hyp = self.manifold.expmap0(_clip_tangent(q))
        k_hyp = self.manifold.expmap0(_clip_tangent(k))
        
        q_hyp_expand = q_hyp.unsqueeze(2)  
        k_hyp_expand = k_hyp.unsqueeze(1)  
        dist_matrix = self.manifold.dist(q_hyp_expand, k_hyp_expand)
        
        min_dist = torch.min(dist_matrix, dim=-1, keepdim=True)[0]
        dist_matrix_shifted = dist_matrix - min_dist
        weights_unnormalized = torch.exp(-dist_matrix_shifted)
        
        weights_probs = weights_unnormalized / (torch.sum(weights_unnormalized, dim=-1, keepdim=True) + 1e-8)
        h = torch.bmm(weights_probs, v)
        
        # Phase 2: Compute Fréchet Center iteratively
        k_centroid_hyp = self.differentiable_karcher_flow(k, weights_unnormalized)
        
        # Float32 Geoopt AutoGrad safety: perturb centroid slightly so it never EXACTLY equals a key
        # Identical points cause logmap gradients to hit 1/0 (NaN).
        noise = torch.randn_like(k_centroid_hyp) * 1e-5
        k_centroid_hyp = self.manifold.projx(k_centroid_hyp + noise)
        
        k_centroid_hyp_expand = k_centroid_hyp.unsqueeze(2)
        centroid_to_keys_dist = self.manifold.dist(k_centroid_hyp_expand, k_hyp_expand)
        
        weights_normalized_var = weights_unnormalized / (torch.sum(weights_unnormalized, dim=-1, keepdim=True) + 1e-8)
        entropy = -torch.sum(weights_normalized_var * torch.log(weights_normalized_var + 1e-8), dim=-1, keepdim=True)
        effective_mass = torch.exp(entropy)
        
        variance = torch.sum(weights_normalized_var * (centroid_to_keys_dist ** 2), dim=-1, keepdim=True)
        tension = variance - self.tau * effective_mass
        
        k_local_tangent = self.manifold.logmap(k_centroid_hyp_expand, k_hyp_expand)
        weighted_k = k_local_tangent * torch.sqrt(weights_unnormalized.unsqueeze(-1) + 1e-8)
        
        v_rand = torch.randn(batch_size, seq_len, dim, 1, device=weighted_k.device, dtype=weighted_k.dtype)
        v_rand = F.normalize(v_rand, dim=2, eps=1e-8)
        
        for _ in range(3):
            proj = torch.matmul(weighted_k, v_rand)
            v_rand = torch.matmul(weighted_k.transpose(2, 3), proj)
            v_rand = F.normalize(v_rand, dim=2, eps=1e-8)
            
        w_proj_local = v_rand.squeeze(-1)
        w_proj_global = self.manifold.transp0back(k_centroid_hyp, w_proj_local)
        
        q_tangent = self.manifold.logmap0(q_hyp)
        k_centroid_tangent = self.manifold.logmap0(k_centroid_hyp)
        fallback_w_proj = F.normalize(q_tangent - k_centroid_tangent, dim=-1, eps=1e-8)
        
        variance_mask = (variance > 1e-5).expand_as(w_proj_global)
        w_proj = torch.where(variance_mask, w_proj_global, fallback_w_proj)
        
        h_norm = torch.norm(h, dim=-1, keepdim=True)
        h_comp = torch.asinh(h_norm) * (h / (h_norm + 1e-8))
        
        # Yield to Runge-Kutta Integrator Phase 3
        h_prime_hyp = self.bifurcation(h_comp, tension, w_proj)
        h_out_bifurcation = self.manifold.logmap0(h_prime_hyp)
        
        tau_safe = torch.clamp(self.tau, min=1e-3)
        dynamic_gate = torch.clamp(torch.tanh(tension / tau_safe), min=0.0)
        h_fused = (1.0 - dynamic_gate) * h + dynamic_gate * h_out_bifurcation
        
        h_out = self.o_proj(h_fused)
        
        origin_dist = self.manifold.dist0(k_centroid_hyp)
        
        state = {
            'variance_matrix': centroid_to_keys_dist ** 2,
            'tension': tension,
            'tau': self.tau.item(),
            'w_proj': w_proj,
            'origin_dist': origin_dist,
            'x_in': torch.sum(h_comp * w_proj, dim=-1, keepdim=True),
            'envelope': h_norm,
            'envelope_comp': torch.norm(h_comp, dim=-1, keepdim=True)
        }
        
        return h_out, state

class ToyTensionModel(nn.Module):
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
