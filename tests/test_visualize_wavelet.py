#!/usr/bin/env python3
# test_visualize_wavelet.py - Visual verification of MPS wavelet transform

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

# Create a synthetic signal with known patterns to test transform visually
def create_test_signal(seq_length=256):
    """Create a test signal with multiple frequency components"""
    # Create time base
    t = np.linspace(0, 1, seq_length)
    
    # Create signal with multiple frequency components
    sig1 = np.sin(2 * np.pi * 5 * t)  # 5 Hz
    sig2 = 0.5 * np.sin(2 * np.pi * 20 * t)  # 20 Hz
    sig3 = 0.25 * np.sin(2 * np.pi * 50 * t)  # 50 Hz
    
    # Combine signals
    signal = sig1 + sig2 + sig3
    
    # Convert to tensor [batch=1, seq_length, embed_dim=1]
    tensor = torch.tensor(signal, dtype=torch.float32).view(1, seq_length, 1).to(device)
    
    return tensor, [sig1, sig2, sig3]

# Test wavelet transform and visualization
def visualize_transform(seq_length=256):
    """Visualize the wavelet transform and reconstruction"""
    print(f"\nVisualizing wavelet transform with seq_length={seq_length}")
    
    # Create test signal
    x, components = create_test_signal(seq_length)
    
    # Create wavelet transform
    wavelet = MPSWaveletTransform(wavelet_type='db4', levels=3, mode='reflect').to(device)
    
    # Forward transform
    with torch.no_grad():  # Disable gradient tracking for visualization
        approx, details = wavelet.forward(x)
        
        # Inverse transform
        output = wavelet.inverse(approx, details)
        
        # Calculate reconstruction error
        error = torch.mean((x - output) ** 2).item()
        print(f"Mean squared reconstruction error: {error:.6f}")
        
        # Visualize original signal, components and reconstruction
        plt.figure(figsize=(12, 8))
        
        # Plot original signal
        plt.subplot(3, 1, 1)
        plt.plot(x[0, :, 0].detach().cpu().numpy())
        plt.title('Original Signal')
        plt.grid(True)
        
        # Plot approximation and details
        plt.subplot(3, 1, 2)
        plt.plot(approx[0, :, 0].detach().cpu().numpy(), label='Approximation')
        for i, d in enumerate(details):
            # Pad details to make them the same length as the original signal for visualization
            detail_padded = torch.nn.functional.interpolate(
                d.transpose(1, 2),
                size=seq_length,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            plt.plot(detail_padded[0, :, 0].detach().cpu().numpy(), label=f'Detail {i+1}', alpha=0.7)
        plt.legend()
        plt.title('Wavelet Decomposition')
        plt.grid(True)
        
        # Plot reconstruction
        plt.subplot(3, 1, 3)
        plt.plot(x[0, :, 0].detach().cpu().numpy(), label='Original', alpha=0.7)
        plt.plot(output[0, :, 0].detach().cpu().numpy(), label='Reconstructed', linestyle='--')
        plt.legend()
        plt.title(f'Signal Reconstruction (MSE: {error:.6f})')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'./benchmarks/wavelet_visualization_{seq_length}.png')
        print(f"Visualization saved to ./benchmarks/wavelet_visualization_{seq_length}.png")
    
    return error

# Run visualization for different sequence lengths
seq_lengths = [64, 128, 256, 512]
errors = []

for seq_len in seq_lengths:
    error = visualize_transform(seq_len)
    errors.append(error)

# Plot reconstruction errors
plt.figure(figsize=(10, 6))
plt.plot(seq_lengths, errors, 'bo-')
plt.title('Wavelet Transform Reconstruction Error')
plt.xlabel('Sequence Length')
plt.ylabel('Mean Squared Error')
plt.grid(True)
plt.savefig('./benchmarks/wavelet_reconstruction_errors.png')
print(f"Error plot saved to ./benchmarks/wavelet_reconstruction_errors.png")

# Create 2D visualization with a more complex pattern
def visualize_2d_wavelet():
    """Visualize 2D pattern using wavelet transform"""
    # Create a 2D pattern (resembling a simple image)
    size = 64
    x = torch.zeros(1, size, size).to(device)
    
    # Add some patterns
    for i in range(size):
        for j in range(size):
            # Add horizontal stripes
            if (i // 8) % 2 == 0:
                x[0, i, j] += 0.5
            
            # Add vertical stripes
            if (j // 16) % 2 == 0:
                x[0, i, j] += 0.25
            
            # Add a circle
            if ((i - size/2)**2 + (j - size/2)**2) < (size/6)**2:
                x[0, i, j] += 0.5
    
    # Create wavelet transform
    wavelet = MPSWaveletTransform(wavelet_type='db4', levels=3, mode='reflect').to(device)
    
    # Forward transform
    with torch.no_grad():  # Disable gradient tracking for visualization
        approx, details = wavelet.forward(x)
        
        # Inverse transform
        output = wavelet.inverse(approx, details)
        
        # Calculate reconstruction error
        error = torch.mean((x - output) ** 2).item()
        print(f"\n2D Pattern - Mean squared reconstruction error: {error:.6f}")
        
        # Visualize original, approximation and reconstruction
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(x[0].detach().cpu().numpy(), cmap='viridis')
        plt.title('Original Pattern')
        plt.colorbar()
        
        # Show approximation (resized for visualization)
        approx_resized = torch.nn.functional.interpolate(
            approx.transpose(1, 2),
            size=size,
            mode='nearest'
        ).transpose(1, 2)
        plt.subplot(1, 3, 2)
        plt.imshow(approx_resized[0].detach().cpu().numpy(), cmap='viridis')
        plt.title('Approximation')
        plt.colorbar()
        
        plt.subplot(1, 3, 3)
        plt.imshow(output[0].detach().cpu().numpy(), cmap='viridis')
        plt.title(f'Reconstructed (MSE: {error:.6f})')
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig('./benchmarks/wavelet_2d_visualization.png')
        print(f"2D visualization saved to ./benchmarks/wavelet_2d_visualization.png")

# Run 2D visualization
visualize_2d_wavelet() 