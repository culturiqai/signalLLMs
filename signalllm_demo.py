#!/usr/bin/env python3

"""
SignalLLM Complexity Demo
=========================
This script demonstrates the theoretical complexity advantages of SignalLLM 
by directly accessing the components in the main POC file.
"""

import os
import time
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

# Import components from the main POC file
from signalllm_hrfevo_poc import (
    Config, SignalLLM, FrequencyDomainAttention, FourierConvolutionAttention,
    SpectralEmbedding, BasisFunction, HRFEvoController, SpectralGapAnalyzer
)

def measure_attention_complexity(seq_lengths, embed_dim=64, num_heads=2, device=None):
    """
    Directly measure and compare the complexity of different attention mechanisms
    
    Args:
        seq_lengths: List of sequence lengths to test
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        device: Computing device
    
    Returns:
        Dictionary with results
    """
    if device is None:
        device = torch.device("mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() 
                            else "cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Create attention mechanisms to compare
    standard_attn = torch.nn.MultiheadAttention(embed_dim, num_heads).to(device)
    freq_attn = FrequencyDomainAttention(embed_dim, num_heads).to(device)
    
    # Results storage
    results = {
        'seq_lengths': seq_lengths,
        'standard_times': [],
        'frequency_times': [],
        'standard_flops': [],
        'frequency_flops': []
    }
    
    # Use larger batch sizes for better measurement
    batch_size = 8
    num_runs = 10  # Number of runs for averaging
    print(f"Using batch size: {batch_size}, averaging over {num_runs} runs")
    
    for seq_len in seq_lengths:
        print(f"\nTesting sequence length: {seq_len}")
        
        # Create test data
        x_standard = torch.randn(seq_len, batch_size, embed_dim).to(device)  # [seq_len, batch, embed_dim]
        x_freq = x_standard.transpose(0, 1)  # [batch, seq, embed_dim]
        
        # Warmup to ensure GPU is at steady state
        for _ in range(3):
            _ = standard_attn(x_standard, x_standard, x_standard)
            _ = freq_attn(x_freq)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        
        # Measure standard attention
        standard_times = []
        for _ in range(num_runs):
            start = time.time()
            with torch.no_grad():
                _ = standard_attn(x_standard, x_standard, x_standard)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            standard_times.append(time.time() - start)
        
        avg_standard_time = sum(standard_times) / len(standard_times)
        results['standard_times'].append(avg_standard_time)
        results['standard_flops'].append(seq_len * seq_len * embed_dim)  # O(n²d) complexity
        print(f"Standard attention: {avg_standard_time:.6f}s")
        
        # Measure frequency domain attention
        try:
            frequency_times = []
            for _ in range(num_runs):
                start = time.time()
                with torch.no_grad():
                    _ = freq_attn(x_freq)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                frequency_times.append(time.time() - start)
            
            avg_frequency_time = sum(frequency_times) / len(frequency_times)
            results['frequency_times'].append(avg_frequency_time)
            # O(n log n * d) complexity
            results['frequency_flops'].append(seq_len * np.log2(seq_len) * embed_dim)
            print(f"Frequency domain attention: {avg_frequency_time:.6f}s")
        except Exception as e:
            print(f"Error in frequency domain attention: {e}")
            results['frequency_times'].append(None)
            results['frequency_flops'].append(None)
    
    return results

def measure_embedding_efficiency(vocab_sizes, embed_dim=64, harmonic_bases=16, device=None):
    """
    Compare standard embeddings with spectral embeddings for efficiency
    
    Args:
        vocab_sizes: List of vocabulary sizes to test
        embed_dim: Embedding dimension
        harmonic_bases: Number of harmonic bases for spectral embedding
        device: Computing device
    
    Returns:
        Dictionary with results
    """
    if device is None:
        device = torch.device("mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() 
                            else "cuda" if torch.cuda.is_available() else "cpu")
    
    results = {
        'vocab_sizes': vocab_sizes,
        'standard_params': [],
        'spectral_params': [],
        'parameter_ratios': []
    }
    
    for vocab_size in vocab_sizes:
        print(f"\nTesting vocabulary size: {vocab_size}")
        
        # Create standard embedding
        standard_embed = torch.nn.Embedding(vocab_size, embed_dim)
        standard_params = sum(p.numel() for p in standard_embed.parameters())
        results['standard_params'].append(standard_params)
        print(f"Standard embedding parameters: {standard_params:,}")
        
        # Create spectral embedding with harmonic bases
        spectral_embed = SpectralEmbedding(vocab_size, embed_dim, harmonic_bases=harmonic_bases)
        spectral_params = sum(p.numel() for p in spectral_embed.parameters())
        results['spectral_params'].append(spectral_params)
        print(f"Spectral embedding parameters: {spectral_params:,}")
        
        # Calculate parameter ratio
        ratio = standard_params / spectral_params
        results['parameter_ratios'].append(ratio)
        print(f"Parameter reduction ratio: {ratio:.2f}x")
        
        # Calculate theoretical parameter counts
        print(f"Standard embedding theory: vocab_size * embed_dim = {vocab_size} * {embed_dim} = {vocab_size * embed_dim}")
        print(f"Spectral embedding theory: harmonic_bases * embed_dim + vocab_size * harmonic_bases = {harmonic_bases} * {embed_dim} + {vocab_size} * {harmonic_bases}")
        print(f"Theoretical reduction for very large vocab: ~{embed_dim / harmonic_bases:.2f}x")
    
    return results

def test_evolution_performance(config, device=None):
    """
    Demonstrate how HRFEvo adapts basis functions for better performance
    
    Args:
        config: Configuration object
        device: Computing device
    
    Returns:
        Evolution results
    """
    if device is None:
        device = torch.device("mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() 
                            else "cuda" if torch.cuda.is_available() else "cpu")
    
    print("\nDemonstrating HRFEvo basis function evolution")
    
    # Create a simple model
    model = SignalLLM(config).to(device)
    
    # Create HRFEvo controller with small population
    hrfevo = HRFEvoController(config)
    
    # Define a simple evaluation function
    def evaluate_basis(basis):
        # Track original basis
        original_basis = model.current_basis
        
        # Apply this basis to the model
        model.update_basis_function(basis)
        
        # Create random input
        batch_size = 2
        seq_length = 16
        dummy_input = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device)
        
        # Measure performance
        start_time = time.time()
        with torch.no_grad():
            output = model(dummy_input)
        inference_time = time.time() - start_time
        
        # Calculate efficiency score
        computation_efficiency = 1.0 / inference_time
        
        # Restore original basis
        model.update_basis_function(original_basis)
        
        return {
            'computation_efficiency': computation_efficiency,
            'perplexity': 50.0  # Dummy value, would be real in actual training
        }
    
    # Run a few generations
    results = []
    
    for gen in range(config.evolution_generations):
        print(f"\nGeneration {gen+1}/{config.evolution_generations}")
        stats = hrfevo.evolve_generation(evaluate_basis)
        results.append(stats)
        print(f"Best fitness: {stats['best_fitness']:.4f}")
    
    # Show best basis
    best_basis = hrfevo.get_best_basis()
    print(f"\nBest evolved basis: {best_basis}")
    
    return results

def plot_complexity_results(results, output_dir):
    """Plot complexity comparison results"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot execution time vs. sequence length if we have that data
    if 'seq_lengths' in results:
        plt.figure(figsize=(10, 6))
        
        seq_lengths = results['seq_lengths']
        standard_times = results['standard_times']
        
        # Plot standard attention
        plt.plot(seq_lengths, standard_times, 'bo-', label='Standard Attention')
        
        # Plot frequency domain attention if available
        if 'frequency_times' in results:
            frequency_times = results['frequency_times']
            valid_indices = [i for i, t in enumerate(frequency_times) if t is not None]
            if valid_indices:
                valid_seq_lengths = [seq_lengths[i] for i in valid_indices]
                valid_freq_times = [frequency_times[i] for i in valid_indices]
                plt.plot(valid_seq_lengths, valid_freq_times, 'ro-', label='Frequency Domain Attention')
        
        # Reference scaling curves
        if len(seq_lengths) > 1:
            # O(n²) reference
            n2_factor = standard_times[0] / (seq_lengths[0] ** 2)
            n2_reference = [n2_factor * (n ** 2) for n in seq_lengths]
            plt.plot(seq_lengths, n2_reference, 'b--', alpha=0.5, label='O(n²) scaling')
            
            # O(n log n) reference
            if 'frequency_times' in results and any(t is not None for t in results['frequency_times']):
                valid_idx = next(i for i, t in enumerate(results['frequency_times']) if t is not None)
                nlogn_factor = results['frequency_times'][valid_idx] / (seq_lengths[valid_idx] * np.log2(seq_lengths[valid_idx]))
                nlogn_reference = [nlogn_factor * (n * np.log2(n)) for n in seq_lengths]
                plt.plot(seq_lengths, nlogn_reference, 'r--', alpha=0.5, label='O(n log n) scaling')
        
        plt.xlabel('Sequence Length')
        plt.ylabel('Execution Time (s)')
        plt.title('Attention Mechanism Execution Time vs. Sequence Length')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        complexity_plot_path = os.path.join(output_dir, 'complexity_comparison.png')
        plt.savefig(complexity_plot_path)
        print(f"Saved complexity plot to {complexity_plot_path}")
    
    # Plot parameter efficiency for embeddings
    if 'vocab_sizes' in results:
        plt.figure(figsize=(10, 6))
        plt.plot(results['vocab_sizes'], results['standard_params'], 'bo-', label='Standard Embedding')
        plt.plot(results['vocab_sizes'], results['spectral_params'], 'ro-', label='Spectral Embedding')
        plt.xlabel('Vocabulary Size')
        plt.ylabel('Number of Parameters')
        plt.title('Embedding Parameter Efficiency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        embedding_plot_path = os.path.join(output_dir, 'embedding_efficiency.png')
        plt.savefig(embedding_plot_path)
        print(f"Saved embedding efficiency plot to {embedding_plot_path}")
        
        # Additional plot showing parameter ratio
        plt.figure(figsize=(10, 6))
        plt.plot(results['vocab_sizes'], results['parameter_ratios'], 'go-')
        plt.xlabel('Vocabulary Size')
        plt.ylabel('Parameter Reduction Ratio')
        plt.title('Embedding Parameter Reduction Ratio (Higher = Better)')
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        ratio_plot_path = os.path.join(output_dir, 'parameter_ratio.png')
        plt.savefig(ratio_plot_path)
        print(f"Saved parameter ratio plot to {ratio_plot_path}")

def main():
    """Run the SignalLLM complexity demonstration"""
    # Create a simple argument parser
    parser = argparse.ArgumentParser(description="SignalLLM Complexity Demo")
    
    # Add arguments
    parser.add_argument("--output_dir", type=str, default="./outputs/demo",
                      help="Directory to save outputs")
    parser.add_argument("--demo_mode", type=str, default="embedding",
                      choices=["attention", "embedding", "evolution", "all"],
                      help="Which demo to run")
    parser.add_argument("--seq_lengths", type=str, default="64,256,512,1024,2048,4096",
                      help="Comma-separated list of sequence lengths to test")
    parser.add_argument("--vocab_sizes", type=str, default="1000,5000,10000,50000,100000",
                      help="Comma-separated list of vocabulary sizes to test")
    parser.add_argument("--embed_dim", type=int, default=768,
                      help="Embedding dimension")
    parser.add_argument("--harmonic_bases", type=int, default=64,
                      help="Number of harmonic bases for spectral embedding")
    parser.add_argument("--num_heads", type=int, default=2,
                      help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=1,
                      help="Number of transformer layers")
    parser.add_argument("--hidden_dim", type=int, default=128,
                      help="Hidden dimension size")
    parser.add_argument("--wavelet_type", type=str, default="haar",
                      help="Wavelet type (e.g., haar, db2)")
    parser.add_argument("--wavelet_levels", type=int, default=1,
                      help="Number of wavelet decomposition levels")
    parser.add_argument("--evolution_population", type=int, default=5,
                      help="Population size for HRFEvo")
    parser.add_argument("--evolution_generations", type=int, default=3,
                      help="Number of generations for HRFEvo")
    
    args = parser.parse_args()
    
    # Parse sequence lengths and vocabulary sizes
    seq_lengths = [int(x) for x in args.seq_lengths.split(',')]
    vocab_sizes = [int(x) for x in args.vocab_sizes.split(',')]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine device
    device = torch.device("mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() 
                        else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Start with a clean results dictionary
    results = {}
    
    # Run attention complexity demo
    if args.demo_mode in ["attention", "all"]:
        print("\n===== RUNNING ATTENTION COMPLEXITY DEMO =====")
        try:
            attention_results = measure_attention_complexity(
                seq_lengths, 
                embed_dim=args.embed_dim, 
                num_heads=args.num_heads,
                device=device
            )
            results.update(attention_results)
        except Exception as e:
            print(f"Error in attention complexity demo: {e}")
            import traceback
            traceback.print_exc()
    
    # Run embedding efficiency demo
    if args.demo_mode in ["embedding", "all"]:
        print("\n===== RUNNING EMBEDDING EFFICIENCY DEMO =====")
        try:
            embedding_results = measure_embedding_efficiency(
                vocab_sizes,
                embed_dim=args.embed_dim,
                harmonic_bases=args.harmonic_bases,
                device=device
            )
            results.update(embedding_results)
        except Exception as e:
            print(f"Error in embedding efficiency demo: {e}")
            import traceback
            traceback.print_exc()
    
    # Run evolution demo
    if args.demo_mode in ["evolution", "all"]:
        print("\n===== RUNNING HRFEVO EVOLUTION DEMO =====")
        try:
            # Create a configuration for the demonstration
            config = Config(
                vocab_size=1000,
                max_seq_length=128,
                embed_dim=args.embed_dim,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                hidden_dim=args.hidden_dim,
                harmonic_bases=args.harmonic_bases,
                wavelet_type=args.wavelet_type,
                wavelet_levels=args.wavelet_levels,
                evolution_population=args.evolution_population,
                evolution_generations=args.evolution_generations,
                output_dir=args.output_dir
            )
            
            evolution_results = test_evolution_performance(config, device)
            results['evolution'] = evolution_results
        except Exception as e:
            print(f"Error in evolution demo: {e}")
            import traceback
            traceback.print_exc()
    
    # Plot and save results
    try:
        plot_complexity_results(results, args.output_dir)
    except Exception as e:
        print(f"Error plotting results: {e}")
        import traceback
        traceback.print_exc()
    
    # Save raw results
    results_path = os.path.join(args.output_dir, 'demo_results.json')
    try:
        # Convert numpy types and other non-serializable objects
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            else:
                return obj
        
        serializable_results = convert_to_serializable(results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"Saved raw results to {results_path}")
    except Exception as e:
        print(f"Error saving results: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nSignalLLM complexity demonstration completed successfully!")

if __name__ == "__main__":
    main() 