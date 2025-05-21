"""
Signal-Based Language Model Architecture - Proof of Concept
==========================================================

This file implements a proof-of-concept for a signal-based language model architecture
that represents language in the frequency domain rather than as discrete tokens only.

The key components:
1. Spectral Token Embeddings - Represent tokens as combinations of frequency components
2. Fourier Attention - Attention mechanism operating in the frequency domain
3. Spectral FFN - Feed-forward network with frequency-selective processing
4. Simple training and evaluation loop to demonstrate functionality

This is a minimal implementation to demonstrate the concept. A full implementation would
include additional components and optimizations.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SpectralEmbedding(nn.Module):
    """
    Embedding layer that represents tokens as superpositions of frequency components.
    Instead of learning a direct embedding vector, learns amplitudes and phases for 
    different frequency components.
    """
    def __init__(self, vocab_size, embed_dim, harmonic_bases=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.harmonic_bases = harmonic_bases
        
        # Each token gets frequency amplitudes and phases
        self.frequency_amplitudes = nn.Parameter(torch.randn(vocab_size, harmonic_bases))
        self.frequency_phases = nn.Parameter(torch.randn(vocab_size, harmonic_bases))
        
        # Generate frequency bases (fixed)
        self.register_buffer('frequencies', 
                           torch.linspace(0.1, math.pi, harmonic_bases))
    
    def forward(self, x):
        # x: batch_size x seq_length
        batch_size, seq_length = x.shape
        
        # Get amplitudes and phases for each token
        amplitudes = self.frequency_amplitudes[x]  # batch_size x seq_length x harmonic_bases
        phases = self.frequency_phases[x]  # batch_size x seq_length x harmonic_bases
        
        # Generate time points for embedding dimension
        t = torch.linspace(0, 1, self.embed_dim, device=x.device)
        t = t.view(1, 1, 1, self.embed_dim)  # 1 x 1 x 1 x embed_dim
        
        # Each frequency contributes across the embedding dimension
        frequencies = self.frequencies.view(1, 1, self.harmonic_bases, 1)  # 1 x 1 x harmonic_bases x 1
        amplitudes = amplitudes.unsqueeze(-1)  # batch x seq_len x harmonic_bases x 1
        phases = phases.unsqueeze(-1)  # batch x seq_len x harmonic_bases x 1
        
        # Generate embeddings as superposition of harmonics
        signal = amplitudes * torch.sin(2 * math.pi * frequencies * t + phases)
        embeddings = signal.sum(dim=2)  # batch x seq_len x embed_dim
        
        return embeddings


class SignalPositionalEncoding(nn.Module):
    """
    Positional encoding using basis of sinusoidal signals at different frequencies.
    Extends the standard sinusoidal encoding with learnable parameters.
    """
    def __init__(self, max_seq_length, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Create standard sinusoidal encoding as starting point
        pe = torch.zeros(max_seq_length, embed_dim)
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                            -(math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Make parameters learnable with initial sinusoidal values
        self.pe = nn.Parameter(pe.unsqueeze(0))
        
        # Frequency modulation parameters
        self.freq_mod = nn.Parameter(torch.ones(1, 1, embed_dim))
        self.phase_shift = nn.Parameter(torch.zeros(1, 1, embed_dim))
    
    def forward(self, x):
        # x: batch_size x seq_length x embed_dim
        seq_length = x.size(1)
        
        # Apply frequency modulation and phase shift
        pos_encoding = self.pe[:, :seq_length] * self.freq_mod + self.phase_shift
        
        return x + pos_encoding


class FourierAttention(nn.Module):
    """
    Attention mechanism that operates partially in the frequency domain.
    Uses Fourier transforms to process queries and keys before computing attention.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Projection matrices
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Learnable frequency filters for each head
        self.freq_filter = nn.Parameter(torch.randn(num_heads, self.head_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.scale = (self.head_dim) ** -0.5
    
    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.shape
        
        # Project queries, keys, values
        q = self.q_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        
        # Reshape for batch-wise operations
        q = q.transpose(1, 2)  # batch x heads x seq x dim
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Transform to frequency domain for filtering
        q_fft = torch.fft.rfft(q, dim=2)  # Batch x Heads x Freq x Dim
        k_fft = torch.fft.rfft(k, dim=2)  # Batch x Heads x Freq x Dim
        
        # Apply learnable frequency filtering
        filter_shaped = self.freq_filter.unsqueeze(0).unsqueeze(2)  # 1 x heads x 1 x dim
        q_fft = q_fft * filter_shaped
        k_fft = k_fft * filter_shaped
        
        # Transform back to time domain
        q_filtered = torch.fft.irfft(q_fft, n=seq_length, dim=2)
        k_filtered = torch.fft.irfft(k_fft, n=seq_length, dim=2)
        
        # Standard attention calculation
        attn_weights = torch.matmul(q_filtered, k_filtered.transpose(2, 3)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1)  # Add head dimension
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)  # batch x heads x seq x dim
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)
        
        return self.out_proj(output)


class SpectralFFN(nn.Module):
    """
    Feed-forward network that operates partially in the frequency domain.
    Applies different transformations to different frequency components.
    """
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Traditional projection matrices
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        
        # Frequency-selective processing
        fft_size = hidden_dim // 2 + 1  # Size of rfft output for hidden_dim
        self.low_pass = nn.Parameter(torch.ones(1, 1, fft_size) * 0.8)
        self.high_pass = nn.Parameter(torch.zeros(1, 1, fft_size) + 0.2)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Traditional feedforward part
        h = F.gelu(self.fc1(x))
        h = self.dropout(h)
        
        # Apply frequency-domain processing
        h_fft = torch.fft.rfft(h, dim=-1)
        
        # Frequency-selective filtering
        h_fft_filtered = h_fft * (self.low_pass + self.high_pass)
        
        # Back to time domain
        h_filtered = torch.fft.irfft(h_fft_filtered, n=self.hidden_dim, dim=-1)
        
        # Final projection
        return self.fc2(h_filtered)


class SignalTransformerBlock(nn.Module):
    """A single transformer block with signal-based components"""
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.attn = FourierAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = SpectralFFN(embed_dim, hidden_dim, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Attention with residual connection
        attn_output = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # FFN with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x


class SignalBasedLLM(nn.Module):
    """
    Complete signal-based language model architecture.
    Uses spectral embeddings, Fourier attention, and spectral feed-forward networks.
    """
    def __init__(self, vocab_size, max_seq_length, embed_dim, num_heads, 
                 num_layers, hidden_dim, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Token embeddings - spectral version
        self.token_embedding = SpectralEmbedding(vocab_size, embed_dim, harmonic_bases=16)
        
        # Positional encoding
        self.pos_encoding = SignalPositionalEncoding(max_seq_length, embed_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            SignalTransformerBlock(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output head
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x, mask=None):
        # Get embeddings
        x = self.token_embedding(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Process through transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final normalization and projection
        x = self.norm(x)
        return self.output_proj(x)


# Simple dummy dataset for demonstration
class DummyTextDataset(Dataset):
    def __init__(self, vocab_size=1000, seq_length=32, size=1000):
        self.data = torch.randint(1, vocab_size, (size, seq_length))
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        # Use next token prediction as a simple task
        y = torch.roll(x, -1)
        y[-1] = 0  # padding token
        return x, y


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        
        # Calculate loss
        loss = F.cross_entropy(output.view(-1, output.size(-1)), target.view(-1), 
                             ignore_index=0)  # ignore padding
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            
            # Calculate loss
            loss = F.cross_entropy(output.view(-1, output.size(-1)), target.view(-1),
                                 ignore_index=0)  # ignore padding
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    """Main function to demonstrate the signal-based LLM"""
    # Model parameters
    vocab_size = 1000
    max_seq_length = 32
    embed_dim = 64
    num_heads = 2
    num_layers = 2
    hidden_dim = 128
    batch_size = 32
    
    # Create datasets
    train_dataset = DummyTextDataset(vocab_size=vocab_size, seq_length=max_seq_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    model = SignalBasedLLM(
        vocab_size=vocab_size,
        max_seq_length=max_seq_length,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_dim=hidden_dim
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print("Starting training...")
    
    for epoch in range(3):
        print(f"Epoch {epoch+1}")
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch+1}, Average loss: {train_loss:.4f}")
    
    print("Training complete!")
    
    # Generate a simple example
    print("\nGenerating sample text:")
    
    # Start with a random sequence
    seed_seq = torch.randint(1, vocab_size, (1, 5)).to(device)
    
    with torch.no_grad():
        # Simple auto-regressive generation
        for _ in range(10):
            # Get model prediction
            logits = model(seed_seq)
            next_token_logits = logits[0, -1, :]
            
            # Sample from the output distribution
            next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), 1)
            
            # Add to sequence
            seed_seq = torch.cat([seed_seq, next_token.unsqueeze(0)], dim=1)
        
        print(f"Generated token IDs: {seed_seq[0].cpu().numpy()}")
    
    # Signal-based analysis
    print("\nAnalyzing spectral characteristics:")
    
    # Extract token embeddings
    with torch.no_grad():
        # Get embeddings for a range of tokens
        token_ids = torch.arange(1, 10).to(device)
        embeddings = model.token_embedding(token_ids.unsqueeze(0)).squeeze(0)
        
        # Analyze frequency components
        embeddings_fft = torch.fft.rfft(embeddings, dim=1)
        power_spectra = torch.abs(embeddings_fft) ** 2
        
        # Print average power at different frequencies
        print("Average power spectrum across tokens:")
        avg_power = power_spectra.mean(dim=0)
        for i, power in enumerate(avg_power[:10]):  # First 10 frequencies
            print(f"Frequency {i}: {power.item():.4f}")
    
    print("\nSignal-Based LLM demonstration complete!")


if __name__ == "__main__":
    main()
