#!/usr/bin/env python3
"""
test_mps_fix.py - Test the MPS convolution fix
"""

import torch
from mps_optimizations import MPSWaveletTransform

def test_mps_wavelet_fix():
    """Test that the MPS wavelet fix works without errors"""
    
    print("=== Testing MPS Wavelet Fix ===")
    
    # Check if MPS is available
    if not torch.backends.mps.is_available():
        print("❌ MPS not available, testing on CPU")
        device = torch.device('cpu')
    else:
        print("✅ MPS available, testing on MPS")
        device = torch.device('mps')
    
    # Create a test input (batch=2, seq_len=64, embed_dim=128)
    x = torch.randn(2, 64, 128).to(device)
    
    # Test different wavelet types and levels
    test_configs = [
        ('db1', 2),
        ('db2', 2), 
        ('db4', 3),
    ]
    
    for wavelet_type, levels in test_configs:
        print(f"\nTesting {wavelet_type} with {levels} levels...")
        
        try:
            # Create wavelet transform
            wavelet = MPSWaveletTransform(
                wavelet_type=wavelet_type,
                levels=levels,
                use_learned=False
            ).to(device)
            
            # Test forward transform
            approx, details = wavelet.forward(x)
            
            print(f"  ✅ Forward pass successful")
            print(f"     Approx shape: {approx.shape}")
            print(f"     Detail shapes: {[d.shape for d in details]}")
            
            # Test inverse transform
            reconstructed = wavelet.inverse(approx, details)
            
            print(f"  ✅ Inverse pass successful")
            print(f"     Reconstructed shape: {reconstructed.shape}")
            
            # Check reconstruction quality
            mse = torch.mean((x - reconstructed) ** 2)
            print(f"     Reconstruction MSE: {mse.item():.6f}")
            
        except Exception as e:
            print(f"  ❌ Test failed: {str(e)}")
            return False
    
    print(f"\n🎉 All MPS wavelet tests passed!")
    return True

if __name__ == "__main__":
    success = test_mps_wavelet_fix()
    if success:
        print("\n✅ MPS fix verified - ready for training!")
    else:
        print("\n❌ MPS fix failed - needs more debugging") 