#!/usr/bin/env python3
# test_benchmark.py - Test benchmark for optimized SignalLLM

import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import os

from signalllm_hrfevo_poc import SignalLLM, Config, SimpleTokenizer
from mps_optimizations import MPSWaveletTransform, MPSOptimizedAttention, optimize_for_mps, setup_mps_optimizations

# Check for MPS
if torch.backends.mps.is_available():
    setup_mps_optimizations()
    device = torch.device("mps")
    print(f"Using MPS: {device}")
else:
    device = torch.device("cpu")
    print(f"Using CPU: {device}")

# Create benchmark function
def benchmark_model(model_type, seq_lengths, embed_dim=128, batch_size=4, num_runs=5):
    """Benchmark model performance across different sequence lengths"""
    results = {
        'model_type': model_type,
        'seq_lengths': seq_lengths,
        'forward_times': [],
        'memory_usage': []
    }
    
    # Create config
    config = Config(
        vocab_size=1000,
        max_seq_length=max(seq_lengths),
        embed_dim=embed_dim,
        num_heads=4,
        num_layers=2,
        use_signal_embed=model_type != "standard",
        use_wavelet_attention=model_type != "standard"
    )
    
    # Create model
    model = SignalLLM(config).to(device)
    
    # Apply optimizations for optimized model
    if model_type == "optimized":
        print("Applying MPS optimizations...")
        model = optimize_for_mps(model)
    
    # Print model parameters
    param_count = sum(p.numel() for p in model.parameters())
    print(f"{model_type} model has {param_count:,} parameters")
    
    # Benchmark each sequence length
    for seq_len in seq_lengths:
        print(f"Benchmarking {model_type} with seq_length={seq_len}...")
        
        # Create test data
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        
        # Warmup
        with torch.no_grad():
            _ = model(input_ids)
        
        # Benchmark
        total_time = 0
        for i in range(num_runs):
            torch.mps.synchronize() if device.type == 'mps' else torch.cuda.synchronize() if device.type == 'cuda' else None
            start = time.time()
            
            with torch.no_grad():
                _ = model(input_ids)
            
            torch.mps.synchronize() if device.type == 'mps' else torch.cuda.synchronize() if device.type == 'cuda' else None
            end = time.time()
            
            run_time = end - start
            total_time += run_time
            
            print(f"  Run {i+1}: {run_time:.4f}s")
        
        # Calculate average
        avg_time = total_time / num_runs
        results['forward_times'].append(avg_time)
        
        # Memory usage
        if device.type == 'cuda':
            memory_used = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
        else:
            memory_used = 0  # Not available for CPU/MPS
        
        results['memory_usage'].append(memory_used)
        
        print(f"  Average time: {avg_time:.4f}s")
    
    return results

# Main benchmark
def main():
    # Create output directory
    os.makedirs("./benchmarks", exist_ok=True)
    
    # Configure benchmark parameters
    seq_lengths = [32, 64, 128, 256]
    embed_dim = 256
    batch_size = 4
    
    # Run benchmarks
    standard_results = benchmark_model("standard", seq_lengths, embed_dim, batch_size)
    optimized_results = benchmark_model("optimized", seq_lengths, embed_dim, batch_size)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lengths, standard_results['forward_times'], 'bo-', label='Standard')
    plt.plot(seq_lengths, optimized_results['forward_times'], 'ro-', label='MPS Optimized')
    plt.title('Forward Pass Time vs Sequence Length')
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.savefig('./benchmarks/mps_optimization_benchmark.png')
    
    # Calculate speedup
    speedups = [std / opt for std, opt in zip(standard_results['forward_times'], 
                                           optimized_results['forward_times'])]
    
    # Plot speedup
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lengths, speedups, 'go-')
    plt.title('MPS Optimization Speedup vs Sequence Length')
    plt.xlabel('Sequence Length')
    plt.ylabel('Speedup Factor (higher is better)')
    plt.axhline(y=1.0, color='black', linestyle='--')
    plt.grid(True)
    
    # Annotate speedup values
    for i, (x, y) in enumerate(zip(seq_lengths, speedups)):
        plt.annotate(f"{y:.2f}x", (x, y), textcoords="offset points", 
                    xytext=(0, 10), ha='center')
    
    plt.savefig('./benchmarks/mps_speedup.png')
    
    # Print summary
    print("\nBenchmark Results:")
    print(f"{'Sequence Length':<15} {'Standard (s)':<15} {'Optimized (s)':<15} {'Speedup':<10}")
    print("-" * 55)
    for i, seq_len in enumerate(seq_lengths):
        print(f"{seq_len:<15} {standard_results['forward_times'][i]:<15.4f} "
              f"{optimized_results['forward_times'][i]:<15.4f} {speedups[i]:<10.2f}x")
    
    # Save results to file
    with open('./benchmarks/mps_benchmark_results.txt', 'w') as f:
        f.write("MPS Optimization Benchmark Results\n")
        f.write("=================================\n\n")
        f.write(f"Device: {device}\n\n")
        f.write(f"{'Sequence Length':<15} {'Standard (s)':<15} {'Optimized (s)':<15} {'Speedup':<10}\n")
        f.write("-" * 55 + "\n")
        for i, seq_len in enumerate(seq_lengths):
            f.write(f"{seq_len:<15} {standard_results['forward_times'][i]:<15.4f} "
                  f"{optimized_results['forward_times'][i]:<15.4f} {speedups[i]:<10.2f}x\n")

if __name__ == "__main__":
    main() 