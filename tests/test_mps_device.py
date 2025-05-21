#!/usr/bin/env python3
# test_mps_device.py - Diagnose MPS device issues

import torch
import numpy as np
from mps_optimizations import MPSWaveletTransform, setup_mps_optimizations

# Check if MPS is available
print(f"MPS available: {torch.backends.mps.is_available()}")
if torch.backends.mps.is_available():
    setup_mps_optimizations()
    device = torch.device("mps")
    print("Using MPS device")
else:
    device = torch.device("cpu")
    print("Using CPU device")

# Create a test tensor
batch_size = 4
seq_length = 64
embed_dim = 128
x = torch.randn(batch_size, seq_length, embed_dim).to(device)
print(f"Input tensor device: {x.device}")

# Create MPS wavelet transform
wavelet = MPSWaveletTransform(wavelet_type='db4', levels=3, mode='reflect').to(device)
print(f"Wavelet transform device: {next(wavelet.parameters()).device}")

# Check filter device
print(f"Wavelet dec_lo device: {wavelet.dec_lo.device}")
print(f"Wavelet dec_hi device: {wavelet.dec_hi.device}")

# Try forward pass
try:
    print("Attempting forward pass...")
    with torch.no_grad():
        approx, details = wavelet.forward(x)
    print(f"Forward pass successful. Output device: {approx.device}")
    
    # Check details
    print(f"Number of detail coefficients: {len(details)}")
    for i, d in enumerate(details):
        print(f"Detail {i} shape: {d.shape}, device: {d.device}")
    
    # Try inverse transform
    print("Attempting inverse transform...")
    output = wavelet.inverse(approx, details)
    print(f"Inverse transform successful. Output shape: {output.shape}, device: {output.device}")
    
    # Verify data flow remains on MPS
    print("\nChecking for any CPU transfers during computation...")
    output2 = wavelet.forward_optimized(x)[0]
    print(f"Output device after optimized forward: {output2.device}")
    
except Exception as e:
    print(f"Error during transform: {e}")
    
    # Try diagnosing the issue
    import traceback
    traceback.print_exc()
    
    print("\nAttempting diagnostic operations...")
    
    # Check if operations are falling back to CPU
    try:
        # Test basic MPS operations to ensure they're working
        a = torch.randn(10, 10, device=device)
        b = torch.randn(10, 10, device=device)
        c = a @ b
        print(f"Basic matrix multiplication device: {c.device}")
        
        # Try convolution (used in wavelet transform)
        a = torch.randn(1, 1, 10, device=device)
        b = torch.randn(1, 1, 3, device=device)
        c = torch.nn.functional.conv1d(a, b)
        print(f"Conv1d output device: {c.device}")
        
        # Check if numpy conversion causes device issues
        try:
            a_np = a.detach().cpu().numpy()
            print("CPU->numpy conversion works")
        except Exception as e2:
            print(f"CPU->numpy conversion failed: {e2}")
    
    except Exception as e3:
        print(f"Diagnostic operations failed: {e3}")

print("\nTest complete.") 