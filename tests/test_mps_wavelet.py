#!/usr/bin/env python3
# test_mps_wavelet.py - Test the fixed MPS wavelet transform

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from mps_optimizations import MPSWaveletTransform, setup_mps_optimizations

# Create output directory
os.makedirs("./benchmarks", exist_ok=True)

# Set up device
if torch.backends.mps.is_available():
    setup_mps_optimizations()
    device = torch.device("mps")
    print(f"Using MPS device: {device}")
else:
    device = torch.device("cpu")
    print(f"Using CPU device: {device}")

# Test parameters
batch_size = 2
embed_dim = 64
seq_lengths = [32, 64, 128, 256]

def test_wavelet_transform(seq_length):
    """Test wavelet transform for a specific sequence length"""
    print(f"\nTesting wavelet transform with seq_length={seq_length}")
    
    # Create input tensor
    x = torch.randn(batch_size, seq_length, embed_dim, device=device)
    print(f"Input shape: {x.shape}, device: {x.device}")
    
    # Create wavelet transform
    wavelet = MPSWaveletTransform(wavelet_type='db4', levels=3, mode='reflect').to(device)
    
    # Forward transform
    try:
        print("Performing forward transform...")
        approx, details = wavelet.forward(x)
        
        print(f"Approximation shape: {approx.shape}, device: {approx.device}")
        print(f"Number of detail coefficients: {len(details)}")
        for i, d in enumerate(details):
            print(f"Detail {i} shape: {d.shape}, device: {d.device}")
        
        # Inverse transform
        print("Performing inverse transform...")
        output = wavelet.inverse(approx, details)
        
        print(f"Output shape: {output.shape}, device: {output.device}")
        
        # Calculate reconstruction error
        error = torch.mean((x - output) ** 2).item()
        print(f"Mean squared reconstruction error: {error:.6f}")
        
        return True, {"error": error, "input_shape": x.shape, "output_shape": output.shape}
    
    except Exception as e:
        print(f"Error during wavelet transform: {e}")
        import traceback
        traceback.print_exc()
        return False, {"error": str(e)}

# Test all sequence lengths
results = {}
for seq_len in seq_lengths:
    success, result = test_wavelet_transform(seq_len)
    results[seq_len] = {"success": success, **result}

# Print summary
print("\n=== Summary ===")
print(f"{'Sequence Length':<15} {'Success':<10} {'Error':<15} {'Shape Match':<15}")
print("-" * 55)

for seq_len, result in results.items():
    if result["success"]:
        shape_match = "Yes" if result["input_shape"] == result["output_shape"] else "No"
        print(f"{seq_len:<15} {'✓':<10} {result['error']:<15.6f} {shape_match:<15}")
    else:
        print(f"{seq_len:<15} {'✗':<10} {'N/A':<15} {'N/A':<15}")

# Plot reconstruction errors if available
successful_lengths = [sl for sl in seq_lengths if results[sl]["success"]]
if successful_lengths:
    errors = [results[sl]["error"] for sl in successful_lengths]
    
    plt.figure(figsize=(10, 6))
    plt.plot(successful_lengths, errors, 'bo-')
    plt.title('Wavelet Transform Reconstruction Error')
    plt.xlabel('Sequence Length')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    plt.savefig('./benchmarks/mps_wavelet_errors.png')
    
    print(f"\nPlot saved to ./benchmarks/mps_wavelet_errors.png") 