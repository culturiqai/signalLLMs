#!/usr/bin/env python3

"""
SignalLLM Visualization Tool
============================
This script provides visualizations of how text is represented in the frequency domain
using the SignalLLM approach. It demonstrates spectrograms of different text types and
visualizes the attention mechanism in frequency space.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.nn.functional as F

# Import components from the main POC file
from signalllm_hrfevo_poc import (
    Config, SpectralEmbedding, WaveletTransform, FrequencyDomainAttention,
    SimpleTokenizer, BasisFunction, VisualizationTools
)

def create_custom_colormap():
    """Create a custom colormap suitable for language visualization"""
    colors = [(0.1, 0.1, 0.6), (0.2, 0.8, 0.8), (1.0, 0.8, 0.2)]
    return LinearSegmentedColormap.from_list("signal_llm", colors, N=256)

def visualize_token_embeddings(text, embed_dim=128, harmonic_bases=16, output_dir="./visualizations"):
    """
    Visualize how tokens from a text are represented in the spectral domain
    
    Args:
        text: Input text to visualize
        embed_dim: Embedding dimension
        harmonic_bases: Number of harmonic bases
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a simple character-level tokenizer
    tokenizer = SimpleTokenizer(mode='char')
    tokens = tokenizer.tokenize(text)
    tokenizer.build_vocab([text], max_vocab_size=256)
    token_ids = [tokenizer.word_to_id.get(t, 0) for t in tokens]
    
    # Create spectral embedding
    spectral_embed = SpectralEmbedding(tokenizer.get_vocab_size(), embed_dim, harmonic_bases=harmonic_bases).to(device)
    
    # Get embeddings for the tokens
    token_tensor = torch.tensor(token_ids, dtype=torch.long).to(device)
    embeddings = spectral_embed(token_tensor).detach().cpu().numpy()
    
    # Create visualizations
    fig, axes = plt.subplots(3, 1, figsize=(12, 16))
    
    # 1. Visualize token embeddings as spectrograms
    cmap = create_custom_colormap()
    im = axes[0].imshow(embeddings, aspect='auto', cmap=cmap)
    axes[0].set_title(f"Spectral Embeddings of Text Tokens (dim={embed_dim}, bases={harmonic_bases})")
    axes[0].set_xlabel("Embedding Dimension")
    axes[0].set_ylabel("Token Position")
    plt.colorbar(im, ax=axes[0], label="Activation")
    
    # Add token labels
    max_display = 40  # Limit number of tokens to display for readability
    tokens_to_show = tokens[:max_display] if len(tokens) > max_display else tokens
    tick_positions = np.arange(len(tokens_to_show))
    if len(tokens) > max_display:
        axes[0].set_yticks(tick_positions)
        axes[0].set_yticklabels(tokens_to_show)
        axes[0].set_ylabel(f"Token Position (showing first {max_display} of {len(tokens)})")
    else:
        axes[0].set_yticks(tick_positions)
        axes[0].set_yticklabels(tokens_to_show)
    
    # 2. Visualize frequency components
    # Apply FFT to get frequency representation
    freq_repr = np.abs(np.fft.fft2(embeddings))
    freq_repr = np.log1p(freq_repr)  # Log scale for better visualization
    
    im2 = axes[1].imshow(freq_repr, aspect='auto', cmap='viridis')
    axes[1].set_title("Frequency Domain Representation (2D FFT)")
    axes[1].set_xlabel("Frequency Component")
    axes[1].set_ylabel("Token Position")
    plt.colorbar(im2, ax=axes[1], label="Magnitude (log scale)")
    
    # 3. Visualize bases contribution
    # Extract coefficient weights from the embedding layer
    basis_weights = spectral_embed.harmonic_bases.weight.detach().cpu().numpy()
    token_coeffs = spectral_embed.token_coefficients.weight[token_ids].detach().cpu().numpy()
    
    # Plot contribution of each basis to each token
    im3 = axes[2].imshow(token_coeffs, aspect='auto', cmap='plasma')
    axes[2].set_title("Basis Function Contributions to Tokens")
    axes[2].set_xlabel("Basis Function Index")
    axes[2].set_ylabel("Token Position")
    plt.colorbar(im3, ax=axes[2], label="Coefficient Value")
    
    if len(tokens) > max_display:
        axes[2].set_yticks(tick_positions)
        axes[2].set_yticklabels(tokens_to_show)
    else:
        axes[2].set_yticks(tick_positions)
        axes[2].set_yticklabels(tokens_to_show)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "token_spectral_visualization.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Visualization saved to {output_path}")
    return output_path

def visualize_attention_mechanism(text, embed_dim=128, num_heads=4, output_dir="./visualizations"):
    """
    Visualize how the frequency domain attention works
    
    Args:
        text: Input text to visualize
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a simple character-level tokenizer
    tokenizer = SimpleTokenizer(mode='char')
    tokens = tokenizer.tokenize(text)
    tokenizer.build_vocab([text], max_vocab_size=256)
    token_ids = [tokenizer.word_to_id.get(t, 0) for t in tokens]
    
    # Create spectral embedding and attention
    spectral_embed = SpectralEmbedding(tokenizer.get_vocab_size(), embed_dim).to(device)
    freq_attn = FrequencyDomainAttention(embed_dim, num_heads).to(device)
    
    # Get embeddings for the tokens
    token_tensor = torch.tensor(token_ids, dtype=torch.long).to(device)
    embeddings = spectral_embed(token_tensor).unsqueeze(0)  # Add batch dimension
    
    # Register a hook to capture attention values
    attention_output = {}
    
    def hook_fn(module, input, output):
        attention_output['attention_weights'] = module.last_attn_weights.detach().cpu()
        attention_output['fft_values'] = module.fft_debug_values
    
    freq_attn.register_forward_hook(hook_fn)
    
    # Forward pass
    with torch.no_grad():
        output = freq_attn(embeddings)
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # 1. Original attention weights (across heads)
    if 'attention_weights' in attention_output:
        attn_weights = attention_output['attention_weights'][0]  # First batch
        for h in range(min(num_heads, 4)):
            if h < len(axes):
                im = axes[h].imshow(attn_weights[h], cmap='viridis')
                axes[h].set_title(f"Attention Head {h+1}")
                axes[h].set_xlabel("Key Position")
                axes[h].set_ylabel("Query Position")
                plt.colorbar(im, ax=axes[h])
                
                # Add token labels (limit for readability)
                max_display = 20
                if len(tokens) <= max_display:
                    axes[h].set_xticks(np.arange(len(tokens)))
                    axes[h].set_xticklabels(tokens, rotation=90)
                    axes[h].set_yticks(np.arange(len(tokens)))
                    axes[h].set_yticklabels(tokens)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "attention_visualization.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    # Additional visualization for the frequency domain transformation
    if 'fft_values' in attention_output and attention_output['fft_values']:
        fft_q = attention_output['fft_values'].get('fft_q', None)
        fft_k = attention_output['fft_values'].get('fft_k', None)
        
        if fft_q is not None and fft_k is not None:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Convert to numpy and take first batch, first head
            fft_q_np = fft_q[0, 0].abs().cpu().numpy()
            fft_k_np = fft_k[0, 0].abs().cpu().numpy()
            
            # Visualize real and imaginary components
            axes[0, 0].imshow(fft_q_np, cmap='plasma')
            axes[0, 0].set_title("FFT Magnitude (Q)")
            axes[0, 0].set_xlabel("Frequency Component")
            axes[0, 0].set_ylabel("Position")
            
            axes[0, 1].imshow(fft_k_np, cmap='plasma')
            axes[0, 1].set_title("FFT Magnitude (K)")
            axes[0, 1].set_xlabel("Frequency Component")
            axes[0, 1].set_ylabel("Position")
            
            # Compute and visualize the convolution in frequency domain
            fft_prod = fft_q[0, 0] * fft_k[0, 0].conj()
            fft_prod_np = fft_prod.abs().cpu().numpy()
            
            axes[1, 0].imshow(fft_prod_np, cmap='viridis')
            axes[1, 0].set_title("Q·K* in Frequency Domain")
            axes[1, 0].set_xlabel("Frequency Component")
            axes[1, 0].set_ylabel("Position")
            
            # Inverse FFT to get attention
            ifft_result = torch.fft.ifft2(fft_prod).real.cpu().numpy()
            axes[1, 1].imshow(ifft_result, cmap='viridis')
            axes[1, 1].set_title("Attention after IFFT")
            axes[1, 1].set_xlabel("Key Position")
            axes[1, 1].set_ylabel("Query Position")
            
            plt.tight_layout()
            fft_output_path = os.path.join(output_dir, "frequency_domain_attention.png")
            plt.savefig(fft_output_path, dpi=300)
            plt.close()
            print(f"Frequency domain visualization saved to {fft_output_path}")
    
    print(f"Attention visualization saved to {output_path}")
    return output_path

def visualize_wavelet_decomposition(text, wavelet_type='db4', levels=3, output_dir="./visualizations"):
    """
    Visualize how the wavelet transform decomposes text representations
    
    Args:
        text: Input text to visualize
        wavelet_type: Type of wavelet to use
        levels: Number of decomposition levels
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create tokenizer and get embeddings
    tokenizer = SimpleTokenizer(mode='char')
    tokens = tokenizer.tokenize(text)
    tokenizer.build_vocab([text], max_vocab_size=256)
    token_ids = [tokenizer.word_to_id.get(t, 0) for t in tokens]
    
    # Create spectral embedding and wavelet transform
    embed_dim = 128
    spectral_embed = SpectralEmbedding(tokenizer.get_vocab_size(), embed_dim).to(device)
    wavelet_transform = WaveletTransform(wavelet_type=wavelet_type, levels=levels).to(device)
    
    # Get embeddings
    token_tensor = torch.tensor(token_ids, dtype=torch.long).to(device)
    embeddings = spectral_embed(token_tensor).unsqueeze(0)  # Add batch dimension
    
    # Apply wavelet transform
    with torch.no_grad():
        approx, details = wavelet_transform(embeddings)
    
    # Visualize
    fig, axes = plt.subplots(levels + 1, 1, figsize=(12, 3 * (levels + 1)))
    
    # Plot approximation coefficients
    approx_np = approx[0].cpu().numpy()  # First batch
    im = axes[0].imshow(approx_np, aspect='auto', cmap='viridis')
    axes[0].set_title(f"Approximation Coefficients (Level {levels})")
    axes[0].set_xlabel("Feature Dimension")
    axes[0].set_ylabel("Token Position")
    plt.colorbar(im, ax=axes[0])
    
    # Add token labels
    max_display = 30
    tokens_to_show = tokens[:max_display] if len(tokens) > max_display else tokens
    tick_positions = np.arange(len(tokens_to_show))
    if len(tokens) > max_display:
        axes[0].set_yticks(tick_positions)
        axes[0].set_yticklabels(tokens_to_show)
    else:
        axes[0].set_yticks(tick_positions)
        axes[0].set_yticklabels(tokens_to_show)
    
    # Plot detail coefficients for each level
    for i, detail in enumerate(details):
        detail_np = detail[0].cpu().numpy()  # First batch
        level = levels - i
        im = axes[i + 1].imshow(detail_np, aspect='auto', cmap='plasma')
        axes[i + 1].set_title(f"Detail Coefficients (Level {level})")
        axes[i + 1].set_xlabel("Feature Dimension")
        axes[i + 1].set_ylabel("Token Position")
        plt.colorbar(im, ax=axes[i + 1])
        
        if len(tokens) > max_display:
            axes[i + 1].set_yticks(tick_positions)
            axes[i + 1].set_yticklabels(tokens_to_show)
        else:
            axes[i + 1].set_yticks(tick_positions)
            axes[i + 1].set_yticklabels(tokens_to_show)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "wavelet_decomposition.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Wavelet decomposition visualization saved to {output_path}")
    return output_path

def main():
    """Main function to run visualizations"""
    parser = argparse.ArgumentParser(description="SignalLLM Visualization Tools")
    parser.add_argument("--text", type=str, default="SignalLLM represents language in the frequency domain using spectral embeddings and achieves O(n log n) attention complexity.", 
                        help="Text to visualize")
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--harmonic_bases", type=int, default=16, help="Number of harmonic bases")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--wavelet_type", type=str, default="db4", help="Wavelet type")
    parser.add_argument("--wavelet_levels", type=int, default=3, help="Wavelet decomposition levels")
    parser.add_argument("--output_dir", type=str, default="./visualizations", help="Output directory")
    parser.add_argument("--mode", type=str, default="all", 
                        choices=["embeddings", "attention", "wavelet", "all"],
                        help="Visualization mode")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run visualizations based on mode
    if args.mode in ["embeddings", "all"]:
        visualize_token_embeddings(
            args.text, 
            embed_dim=args.embed_dim, 
            harmonic_bases=args.harmonic_bases,
            output_dir=args.output_dir
        )
    
    if args.mode in ["attention", "all"]:
        visualize_attention_mechanism(
            args.text, 
            embed_dim=args.embed_dim, 
            num_heads=args.num_heads,
            output_dir=args.output_dir
        )
    
    if args.mode in ["wavelet", "all"]:
        visualize_wavelet_decomposition(
            args.text, 
            wavelet_type=args.wavelet_type, 
            levels=args.wavelet_levels,
            output_dir=args.output_dir
        )
    
    print(f"All visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main() 