#!/usr/bin/env python3
# benchmark_implementation_comparison.py - Compare different wavelet transform implementations

import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Check for MPS 
MPS_AVAILABLE = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

# Setup MPS optimizations if available
if MPS_AVAILABLE:
    from mps_optimizations import MPSWaveletTransform, setup_mps_optimizations
    setup_mps_optimizations()
    print("Using Apple Silicon MPS (Metal Performance Shaders)")
    print("MPS optimizations enabled.")
else:
    print("Apple Silicon MPS is not available. Running on CPU.")
    MPSWaveletTransform = None

def compare_wavelet_implementations(seq_lengths=[32, 64, 128, 256, 512], 
                                   embed_dim=128, batch_size=8,
                                   output_dir="./benchmarks"):
    """Compare different wavelet transform implementations"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare device
    device = torch.device("mps" if MPS_AVAILABLE else "cpu")
    print(f"Using device: {device}")
    
    # Implementations to test
    standard_times = []
    mps_times = []
    
    # Test each sequence length
    for seq_length in seq_lengths:
        print(f"\nTesting sequence length: {seq_length}")
        
        # Create test tensor
        x = torch.randn(batch_size, seq_length, embed_dim, device=device)
        
        # Time standard implementation
        standard_transform = None
        try:
            from signalllm_hrfevo_poc import WaveletTransform
            standard_transform = WaveletTransform(
                wavelet_type='db4', 
                levels=3, 
                mode='reflect'
            ).to(device)
        except Exception as e:
            print(f"Error creating standard WaveletTransform: {e}")
            standard_times.append(float('nan'))
            continue
        
        # Warmup
        try:
            with torch.no_grad():
                approx, details = standard_transform.forward(x)
                output = standard_transform.inverse(approx, details)
        except Exception as e:
            print(f"Error in standard transform warmup: {e}")
            standard_times.append(float('nan'))
            continue
        
        # Measure time
        standard_time = 0
        num_runs = 10
        
        try:
            start_time = time.time()
            with torch.no_grad():
                for _ in range(num_runs):
                    approx, details = standard_transform.forward(x)
                    output = standard_transform.inverse(approx, details)
            end_time = time.time()
            standard_time = (end_time - start_time) * 1000 / num_runs  # ms per run
        except Exception as e:
            print(f"Error timing standard transform: {e}")
            standard_time = float('nan')
        
        standard_times.append(standard_time)
        
        # Time MPS-optimized implementation
        mps_transform = MPSWaveletTransform(
            wavelet_type='db4', 
            levels=3, 
            mode='reflect'
        ).to(device)
        
        # Warmup
        try:
            with torch.no_grad():
                approx, details = mps_transform.forward(x)
                output = mps_transform.inverse(approx, details)
        except Exception as e:
            print(f"Error in MPS transform warmup: {e}")
            mps_times.append(float('nan'))
            continue
        
        # Measure time
        mps_time = 0
        
        try:
            start_time = time.time()
            with torch.no_grad():
                for _ in range(num_runs):
                    approx, details = mps_transform.forward(x)
                    output = mps_transform.inverse(approx, details)
            end_time = time.time()
            mps_time = (end_time - start_time) * 1000 / num_runs  # ms per run
        except Exception as e:
            print(f"Error timing MPS transform: {e}")
            mps_time = float('nan')
        
        mps_times.append(mps_time)
        
        # Calculate reconstruction error and shape match
        try:
            with torch.no_grad():
                # Standard transform
                std_approx, std_details = standard_transform.forward(x)
                std_output = standard_transform.inverse(std_approx, std_details)
                std_error = torch.mean((x - std_output) ** 2).item()
                std_shape_match = x.shape == std_output.shape
                
                # MPS transform
                mps_approx, mps_details = mps_transform.forward(x)
                mps_output = mps_transform.inverse(mps_approx, mps_details)
                mps_error = torch.mean((x - mps_output) ** 2).item()
                mps_shape_match = x.shape == mps_output.shape
                
            print(f"Sequence length {seq_length}:")
            print(f"  Standard: {standard_time:.2f} ms, error: {std_error:.4f}, shape match: {std_shape_match}")
            print(f"  MPS Optimized: {mps_time:.2f} ms, error: {mps_error:.4f}, shape match: {mps_shape_match}")
            if standard_time > 0 and mps_time > 0:
                speedup = standard_time / mps_time
                print(f"  Speedup: {speedup:.2f}x")
        except Exception as e:
            print(f"Error calculating reconstruction stats: {e}")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    
    valid_indices = [i for i, (s, m) in enumerate(zip(standard_times, mps_times)) 
                     if not (np.isnan(s) or np.isnan(m))]
    valid_seq_lengths = [seq_lengths[i] for i in valid_indices]
    valid_standard_times = [standard_times[i] for i in valid_indices]
    valid_mps_times = [mps_times[i] for i in valid_indices]
    
    if len(valid_indices) > 0:
        plt.plot(valid_seq_lengths, valid_standard_times, 'o-', label='Standard Implementation')
        plt.plot(valid_seq_lengths, valid_mps_times, 'o-', label='MPS Optimized')
        
        plt.title('Wavelet Transform Implementation Comparison')
        plt.xlabel('Sequence Length')
        plt.ylabel('Time (ms)')
        plt.legend()
        plt.grid(True)
        plt.xscale('log', base=2)
        plt.yscale('log')
        
        plt.savefig(f"{output_dir}/mps_implementation_comparison.png")
        
        # Calculate and plot speedup
        if len(valid_indices) > 1:
            speedups = [s / m for s, m in zip(valid_standard_times, valid_mps_times)]
            
            plt.figure(figsize=(10, 6))
            plt.plot(valid_seq_lengths, speedups, 'go-')
            plt.title('MPS Optimization Speedup')
            plt.xlabel('Sequence Length')
            plt.ylabel('Speedup Factor (higher is better)')
            plt.axhline(y=1.0, color='black', linestyle='--')
            plt.grid(True)
            
            # Annotate speedup values
            for i, (x, y) in enumerate(zip(valid_seq_lengths, speedups)):
                plt.annotate(f"{y:.2f}x", (x, y), textcoords="offset points", 
                            xytext=(0, 10), ha='center')
            
            plt.savefig(f"{output_dir}/mps_wavelet_speedup.png")
    
    print("\nResults summary:")
    print(f"{'Sequence Length':<15} {'Standard (ms)':<15} {'MPS (ms)':<15} {'Speedup':<10}")
    print("-" * 60)
    
    for i, seq_len in enumerate(seq_lengths):
        std_time = standard_times[i]
        mps_time = mps_times[i]
        
        if not (np.isnan(std_time) or np.isnan(mps_time)):
            speedup = std_time / mps_time
            print(f"{seq_len:<15} {std_time:<15.2f} {mps_time:<15.2f} {speedup:<10.2f}x")
        else:
            print(f"{seq_len:<15} {'N/A':<15} {'N/A':<15} {'N/A':<10}")
    
    # Return results
    return {
        "seq_lengths": seq_lengths,
        "standard_times": standard_times,
        "mps_times": mps_times
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Compare wavelet transform implementations")
    
    parser.add_argument("--seq_lengths", nargs="+", type=int, default=[32, 64, 128, 256, 512],
                       help="Sequence lengths to test")
    parser.add_argument("--embed_dim", type=int, default=128,
                       help="Embedding dimension")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--output_dir", type=str, default="./benchmarks",
                       help="Output directory for benchmark results")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Running implementation comparison with:")
    print(f"  Sequence lengths: {args.seq_lengths}")
    print(f"  Embedding dimension: {args.embed_dim}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Output directory: {args.output_dir}")
    
    results = compare_wavelet_implementations(
        seq_lengths=args.seq_lengths,
        embed_dim=args.embed_dim,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )
    
    print("\nBenchmark completed.")
    print(f"Plots saved to {args.output_dir}")

if __name__ == "__main__":
    main() 