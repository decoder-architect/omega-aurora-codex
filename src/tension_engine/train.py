"""
train.py — Empirical Training for Resonance Mapping Networks
RTX 3060 Ti (8GB VRAM) — dim=128, seq_len=128, batch_size=4
Trains next-token prediction (Cross-Entropy) + Entropic Dissonance Regularizer.
Logs loss curves to JSON for paper figures.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer
from model import ResonanceMapping
from loss import EntropicDissonanceLoss

# ── Config ──────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DIM = 128          # Embedding dimension (fits in 8GB VRAM)
SEQ_LEN = 128      # Max sequence length
BATCH_SIZE = 4     # Micro-batch size
EPOCHS = 30        # Training epochs
LR = 3e-4          # Learning rate
ALPHA_REG = 0.01   # Weight for entropic dissonance regularizer
DATASET_PATH = r"d:\Antigravity\subconscious\architect\projects\tension-math\data\isfet_dataset.jsonl"
OUTPUT_DIR = r"d:\Antigravity\omega-aurora-codex\results"

print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ── Dataset ─────────────────────────────────────────
class IsfetDataset(torch.utils.data.Dataset):
    """Loads thesis-antithesis pairs as concatenated token sequences."""
    def __init__(self, path, tokenizer, max_len=SEQ_LEN):
        with open(path, 'r', encoding='utf-8') as f:
            self.pairs = [json.loads(line) for line in f]
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        for p in self.pairs:
            text = p['thesis'] + " " + p['antithesis']
            tokens = tokenizer.encode(text, truncation=True, max_length=max_len)
            if len(tokens) >= 4:  # Minimum viable sequence
                self.samples.append(torch.tensor(tokens, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]
        # Pad to max_len
        if len(tokens) < self.max_len:
            pad = torch.full((self.max_len - len(tokens),), self.tokenizer.pad_token_id, dtype=torch.long)
            tokens = torch.cat([tokens, pad])
        return tokens


# ── Model ───────────────────────────────────────────
class RMNLanguageModel(nn.Module):
    """
    Minimal language model with RMN attention for empirical proof.
    Embedding -> RMN Attention -> LM Head
    """
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.rmn = ResonanceMapping(dim)
        self.ln = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        # Tie weights
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids):
        x = self.embed(input_ids)  # (B, S, D)
        out, state = self.rmn(x, x, x)
        out = self.ln(x + out)  # Residual + LayerNorm
        logits = self.lm_head(out)  # (B, S, vocab)
        return logits, state


# ── Baseline Model ──────────────────────────────────
class BaselineAttentionModel(nn.Module):
    """Standard Euclidean attention baseline for comparison."""
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.ln = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids):
        x = self.embed(input_ids)
        out, _ = self.attn(x, x, x)
        out = self.ln(x + out)
        logits = self.lm_head(out)
        return logits, {}


# ── Training Loop ───────────────────────────────────
def train_model(model, dataset, model_name, use_dissonance=False):
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}")

    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    dissonance_loss_fn = EntropicDissonanceLoss() if use_dissonance else None

    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    history = {
        'epoch': [], 'loss_total': [], 'loss_ce': [], 'loss_dissonance': [],
        'perplexity': [], 'bifurcation_rate': [], 'mean_tension': [],
        'time_per_epoch': [], 'act_cycles': []
    }

    for epoch in range(EPOCHS):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_dis = 0.0
        epoch_bif = 0.0
        epoch_tension = 0.0
        epoch_cycles = 0.0
        n_batches = 0

        for batch in loader:
            batch = batch.to(DEVICE)
            # Shifted targets for next-token prediction
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]

            logits, state = model(input_ids)
            # Cross-Entropy Loss
            ce_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=dataset.tokenizer.pad_token_id
            )

            # Dissonance Regularizer (RMN only)
            dis_loss = torch.tensor(0.0, device=DEVICE)
            if use_dissonance and 'tension' in state:
                tension = state['tension']
                # We don't have q_hyp/k_hyp directly from forward, so use tension-only loss
                dis_loss = -torch.mean(torch.clamp(tension, min=-1.0, max=5.0))

                # Track bifurcation rate
                bif_rate = (tension > 0).float().mean().item()
                epoch_bif += bif_rate
                epoch_tension += tension.mean().item()
                if 'hearth_cycles' in state:
                    epoch_cycles += state['hearth_cycles'].mean().item()

            total_loss = ce_loss + ALPHA_REG * dis_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_ce += ce_loss.item()
            epoch_dis += dis_loss.item()
            n_batches += 1

        scheduler.step()
        epoch_time = time.time() - t0

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_ce = epoch_ce / max(n_batches, 1)
        avg_dis = epoch_dis / max(n_batches, 1)
        avg_bif = epoch_bif / max(n_batches, 1)
        avg_tension = epoch_tension / max(n_batches, 1)
        avg_cycles = epoch_cycles / max(n_batches, 1)
        ppl = min(torch.exp(torch.tensor(avg_ce)).item(), 1e6)

        history['epoch'].append(epoch)
        history['loss_total'].append(avg_loss)
        history['loss_ce'].append(avg_ce)
        history['loss_dissonance'].append(avg_dis)
        history['perplexity'].append(ppl)
        history['bifurcation_rate'].append(avg_bif)
        history['mean_tension'].append(avg_tension)
        history['act_cycles'].append(avg_cycles)
        history['time_per_epoch'].append(epoch_time)

        print(f"  Epoch {epoch+1:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | CE: {avg_ce:.4f} | PPL: {ppl:.1f} | Bif%: {avg_bif*100:.1f}% | τ̄: {avg_tension:.3f} | ACT: {avg_cycles:.2f} | Time: {epoch_time:.1f}s")

    return history


# ── Main ────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Tokenizer
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    print("Loading Isfet dataset...")
    dataset = IsfetDataset(DATASET_PATH, tokenizer, max_len=SEQ_LEN)
    print(f"Loaded {len(dataset)} samples")

    # ── Train RMN ───────────────────────────────────
    rmn_model = RMNLanguageModel(tokenizer.vocab_size, DIM)
    rmn_history = train_model(rmn_model, dataset, "RMN (Resonance Mapping Network)", use_dissonance=True)

    # ── Train Baseline ──────────────────────────────
    baseline_model = BaselineAttentionModel(tokenizer.vocab_size, DIM)
    baseline_history = train_model(baseline_model, dataset, "Baseline (Euclidean Attention)", use_dissonance=False)

    # ── Save Results ────────────────────────────────
    results = {
        'config': {'dim': DIM, 'seq_len': SEQ_LEN, 'batch_size': BATCH_SIZE, 'epochs': EPOCHS, 'lr': LR},
        'rmn': rmn_history,
        'baseline': baseline_history
    }
    results_path = os.path.join(OUTPUT_DIR, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ── Generate Figures ────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Tension Is What You Need — Empirical Training Results', fontsize=14, fontweight='bold')

    # Loss Curves
    ax = axes[0, 0]
    ax.plot(rmn_history['epoch'], rmn_history['loss_ce'], 'b-', label='RMN (CE Loss)', linewidth=2)
    ax.plot(baseline_history['epoch'], baseline_history['loss_ce'], 'r--', label='Baseline (CE Loss)', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('Convergence: Cross-Entropy Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Perplexity
    ax = axes[0, 1]
    ax.plot(rmn_history['epoch'], rmn_history['perplexity'], 'b-', label='RMN', linewidth=2)
    ax.plot(baseline_history['epoch'], baseline_history['perplexity'], 'r--', label='Baseline', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Perplexity')
    ax.set_title('Convergence: Perplexity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Bifurcation Rate (RMN only)
    ax = axes[1, 0]
    ax.plot(rmn_history['epoch'], [r*100 for r in rmn_history['bifurcation_rate']], 'g-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Bifurcation Rate (%)')
    ax.set_title('RMN: Paradox Detection Rate')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    # Mean Tension
    ax = axes[1, 1]
    ax.plot(rmn_history['epoch'], rmn_history['mean_tension'], 'm-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle=':', alpha=0.5, label='τ threshold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Tension (r̄)')
    ax.set_title('RMN: Mean Tension Scalar')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'training_curves.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {fig_path}")

    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"RMN      — Final CE: {rmn_history['loss_ce'][-1]:.4f}, PPL: {rmn_history['perplexity'][-1]:.1f}")
    print(f"Baseline — Final CE: {baseline_history['loss_ce'][-1]:.4f}, PPL: {baseline_history['perplexity'][-1]:.1f}")
    print(f"RMN Bifurcation Rate: {rmn_history['bifurcation_rate'][-1]*100:.1f}%")
    print(f"RMN Mean Tension: {rmn_history['mean_tension'][-1]:.4f}")


if __name__ == '__main__':
    main()
