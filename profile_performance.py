#!/usr/bin/env python3
# profile_performance.py - Profile performance of SignalLLM components

import cProfile
import pstats
import io
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from signalllm_hrfevo_poc import WaveletTransform, FourierConvolutionAttention, WaveletAttention, SignalLLM, Config

def profile_wavelet_transform():
    """Profile the wavelet transform implementation."""
    print("\n=== Profiling WaveletTransform ===")
    
    # Create test tensor
    batch_size = 8
    seq_length = 64
    embed_dim = 128
    x = torch.randn(batch_size, seq_length, embed_dim)
    
    # Create wavelet transform
    wavelet_transform = WaveletTransform(wavelet_type='db4', levels=3, mode='reflect')
    
    # Profile forward transform
    print("Profiling forward transform...")
    pr = cProfile.Profile()
    pr.enable()
    
    # Run multiple times to get better stats
    for _ in range(10):
        approx, details = wavelet_transform.forward(x)
    
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(15)  # Show top 15 functions by time
    print(s.getvalue())
    
    # Profile inverse transform
    print("\nProfiling inverse transform...")
    pr = cProfile.Profile()
    pr.enable()
    
    # Run multiple times
    for _ in range(10):
        output = wavelet_transform.inverse(approx, details)
    
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(15)
    print(s.getvalue())
    
    return approx, details, output

def profile_attention_mechanisms():
    """Profile different attention mechanisms."""
    print("\n=== Profiling Attention Mechanisms ===")
    
    # Create test tensor
    batch_size = 8
    seq_length = 64
    embed_dim = 128
    x = torch.randn(batch_size, seq_length, embed_dim)
    
    # Create attention modules
    fourier_attn = FourierConvolutionAttention(embed_dim=embed_dim, num_heads=4)
    wavelet_attn = WaveletAttention(embed_dim=embed_dim, num_heads=4)
    
    # Profile Fourier attention
    print("Profiling Fourier attention...")
    pr = cProfile.Profile()
    pr.enable()
    
    for _ in range(5):
        output = fourier_attn(x)
    
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(15)
    print(s.getvalue())
    
    # Profile Wavelet attention
    print("\nProfiling Wavelet attention...")
    pr = cProfile.Profile()
    pr.enable()
    
    for _ in range(5):
        output = wavelet_attn(x)
    
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(15)
    print(s.getvalue())

def track_memory_usage(fn, *args, **kwargs):
    """Track memory usage before and after function execution."""
    # Use psutil for cross-platform memory tracking
    import psutil
    
    # Record memory before
    process = psutil.Process()
    mem_before = process.memory_info().rss
    
    # Run function
    result = fn(*args, **kwargs)
    
    # Record memory after
    mem_after = process.memory_info().rss
    
    print(f"Memory change: {(mem_after - mem_before) / 1024**2:.2f} MB")
    print(f"Current memory usage: {mem_after / 1024**2:.2f} MB")
    
    return result

def compare_implementations():
    """Compare different implementations for performance."""
    print("\n=== Comparing Implementations ===")
    
    # Test data
    batch_size = 8
    seq_lengths = [32, 64, 128, 256, 512]
    embed_dim = 128
    
    # Measure time for different sequence lengths
    times_pywt = []
    times_fft = []
    
    for seq_length in seq_lengths:
        x = torch.randn(batch_size, seq_length, embed_dim)
        
        # Test PyWavelets implementation
        wavelet_pywt = WaveletTransform(wavelet_type='db4', levels=3, mode='reflect', use_learned=True)
        wavelet_pywt.using_pywt = True
        
        # Apple Silicon doesn't support cuda events, use time.time() instead
        import time
        
        # Warmup
        try:
            approx, details = wavelet_pywt.forward(x)
            output = wavelet_pywt.inverse(approx, details)
        except Exception as e:
            print(f"PyWavelets warmup error: {e}")
        
        # Actual timing
        start_time = time.time()
        success = True
        for _ in range(5):
            try:
                approx, details = wavelet_pywt.forward(x)
                output = wavelet_pywt.inverse(approx, details)
            except Exception as e:
                print(f"PyWavelets error: {e}")
                success = False
                break
        end_time = time.time()
        
        if success:
            elapsed_pywt = (end_time - start_time) * 1000 / 5  # Average time per iteration in ms
        else:
            elapsed_pywt = float('nan')  # Not a number
        times_pywt.append(elapsed_pywt)
        
        # Test FFT implementation
        wavelet_fft = WaveletTransform(wavelet_type='db4', levels=3, mode='reflect', use_learned=True)
        wavelet_fft.using_pywt = False
        
        # Warmup
        approx, details = wavelet_fft.forward(x)
        output = wavelet_fft.inverse(approx, details)
        
        # Actual timing
        start_time = time.time()
        for _ in range(5):
            approx, details = wavelet_fft.forward(x)
            output = wavelet_fft.inverse(approx, details)
        end_time = time.time()
        
        elapsed_fft = (end_time - start_time) * 1000 / 5  # Average time per iteration in ms
        times_fft.append(elapsed_fft)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lengths, times_pywt, 'o-', label='PyWavelets')
    plt.plot(seq_lengths, times_fft, 'o-', label='FFT')
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (ms)')
    plt.title('Performance Comparison: PyWavelets vs FFT')
    plt.legend()
    plt.grid(True)
    plt.savefig('./benchmarks/implementation_comparison.png')
    plt.close()
    
    print(f"PyWavelets timings: {times_pywt}")
    print(f"FFT timings: {times_fft}")
    print("Comparison plot saved to ./benchmarks/implementation_comparison.png")

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else 
                         "cpu")
    print(f"Using device: {device}")
    
    # Profile components
    approx, details, output = profile_wavelet_transform()
    profile_attention_mechanisms()
    
    # Compare implementations
    compare_implementations()
    
    print("\nProfiling completed!")

if __name__ == "__main__":
    main() 