#!/usr/bin/env python3
# mps_optimizations.py - MPS-specific optimizations for SignalLLM

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Tuple, List, Dict, Optional, Any, Union
import os

class MPSWaveletTransform(nn.Module):
    """
    Optimized wavelet transform implementation for Apple Silicon MPS
    """
    def __init__(self, wavelet_type: str = 'db4', levels: int = 3, 
                 mode: str = 'reflect', use_learned: bool = True):
        super().__init__()
        self.wavelet_type = wavelet_type
        self.levels = levels
        self.mode = mode
        self.use_learned = use_learned
        
        # Create optimized filter parameters
        self.initialize_filters()
        
        # Set flag for using optimized implementation
        self.using_optimized = True
    
    def initialize_filters(self):
        """Initialize wavelet filter coefficients based on wavelet type"""
        # Pre-defined coefficients for common wavelets
        wavelet_coeffs = {
            'db1': {'dec_lo': [0.7071067811865476, 0.7071067811865476],
                    'dec_hi': [0.7071067811865476, -0.7071067811865476]},
            'db2': {'dec_lo': [-0.1294095225512604, 0.2241438680420134, 0.8365163037378079, 0.4829629131445341],
                    'dec_hi': [-0.4829629131445341, 0.8365163037378079, -0.2241438680420134, -0.1294095225512604]},
            'db4': {'dec_lo': [0.0107772507, 0.0328830116, 0.0308413818, -0.1870348117, 
                             -0.0279837694, 0.6308807679, 0.7148465706, 0.2303778133],
                    'dec_hi': [-0.2303778133, 0.7148465706, -0.6308807679, -0.0279837694, 
                              0.1870348117, 0.0308413818, -0.0328830116, 0.0107772507]},
        }
        
        # Use db4 as default if wavelet type not found
        coeffs = wavelet_coeffs.get(self.wavelet_type, wavelet_coeffs['db4'])
        
        # Create tensor filter banks with proper shape for group convolution
        # Shape should be [1, 1, filter_length] - will be expanded to [embed_dim, 1, filter_length] during convolution
        if self.use_learned:
            # Learnable filters initialized with standard coefficients
            self.dec_lo = nn.Parameter(torch.tensor(coeffs['dec_lo'], dtype=torch.float32).view(1, 1, -1))
            self.dec_hi = nn.Parameter(torch.tensor(coeffs['dec_hi'], dtype=torch.float32).view(1, 1, -1))
        else:
            # Fixed filters
            self.register_buffer('dec_lo', torch.tensor(coeffs['dec_lo'], dtype=torch.float32).view(1, 1, -1))
            self.register_buffer('dec_hi', torch.tensor(coeffs['dec_hi'], dtype=torch.float32).view(1, 1, -1))
        
        # Pre-calculated filter lengths for performance
        self.filter_len = len(coeffs['dec_lo'])
        
        # FFT filters for frequency domain operations
        if self.use_learned:
            self.low_pass = nn.Parameter(torch.ones(1, 1, 32) * 0.7)
            self.high_pass = nn.Parameter(torch.ones(1, 1, 32) * 0.3)
        else:
            self.register_buffer('low_pass', torch.ones(1, 1, 32) * 0.7)
            self.register_buffer('high_pass', torch.ones(1, 1, 32) * 0.3)
    
    def _pad_signal(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pad signal for convolution with proper padding based on filter length
        
        Args:
            x: Input tensor [batch_size, seq_length, embed_dim]
            
        Returns:
            Padded tensor [batch_size, seq_length+pad, embed_dim]
        """
        # Calculate padding based on filter length to ensure proper coefficient sizes
        # For wavelet transform, we need padding of (filter_len - 1)
        pad_len = self.filter_len - 1
        
        # Apply symmetric padding (better for wavelets) or zero padding
        if self.mode == 'reflect':
            # Use symmetric padding which better preserves frequency characteristics
            padded = torch.nn.functional.pad(x, (0, 0, pad_len//2, pad_len//2), mode='reflect')
        elif self.mode == 'symmetric':
            # Symmetric padding is similar to reflect but handles boundaries differently
            padded = torch.nn.functional.pad(x, (0, 0, pad_len//2, pad_len//2), mode='replicate')
        else:
            # Zero padding
            padded = torch.nn.functional.pad(x, (0, 0, pad_len//2, pad_len//2), mode='constant')
        
        return padded
    
    def _downsample(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample signal by taking every other element"""
        return x[:, ::2, :]
    
    def _conv1d_mps(self, x: torch.Tensor, filter_bank: torch.Tensor) -> torch.Tensor:
        """
        Optimized 1D convolution using MPS (Metal Performance Shaders)
        
        Args:
            x: Input tensor [batch_size, seq_length, embed_dim]
            filter_bank: Filter coefficients [filter_length] or [1, filter_length]
            
        Returns:
            Convolution result [batch_size, seq_length, embed_dim]
        """
        # Ensure inputs are on the correct device
        if filter_bank.device != x.device:
            filter_bank = filter_bank.to(x.device)
        
        # Reshape for conv1d (batch, channels, length)
        batch_size, seq_length, embed_dim = x.shape
        x_reshaped = x.transpose(1, 2)  # [batch, embed, seq]
        
        # Fix for dimension error: ensure filter_bank has correct dimensions before expanding
        # The correct shape should be [1, 1, filter_length] before expansion
        if filter_bank.dim() == 1:
            # If 1D tensor, add two dimensions
            filter_bank = filter_bank.unsqueeze(0).unsqueeze(0)
        elif filter_bank.dim() == 2:
            # If 2D tensor, add one dimension
            filter_bank = filter_bank.unsqueeze(0)
        
        # Now safely expand to [embed_dim, 1, filter_length]
        filter_expanded = filter_bank.expand(embed_dim, 1, -1)
        
        # Apply convolution across sequence dimension using MPS-optimized conv1d
        # Use groups=embed_dim to apply the same filter to each channel independently
        output = torch.nn.functional.conv1d(
            x_reshaped,
            filter_expanded,
            groups=embed_dim,
            padding='same'
        )
        
        # Reshape back to original format
        output = output.transpose(1, 2)  # [batch, seq, embed]
        
        return output
    
    def _ensure_coefficient_shapes(self, approx: torch.Tensor, details: List[torch.Tensor], expected_sizes: List[int] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Ensure coefficients have consistent shapes to prevent mismatches
        
        Args:
            approx: Approximation coefficients [batch_size, seq_length, embed_dim]
            details: List of detail coefficients
            expected_sizes: Expected sequence lengths for each level
            
        Returns:
            Corrected approx and details with consistent shapes
        """
        # If no expected sizes provided, maintain current sizes
        if expected_sizes is None:
            if not details:
                return approx, details
                
            # Use the size of the first detail coefficient as reference
            expected_sizes = [d.size(1) for d in details]
            expected_sizes.append(approx.size(1))
        
        # Ensure details have expected sizes
        corrected_details = []
        for idx, detail in enumerate(details):
            if idx >= len(expected_sizes):
                # More details than expected, just append as is
                corrected_details.append(detail)
                continue
                
            expected_size = expected_sizes[idx]
            actual_size = detail.size(1)
            
            if actual_size != expected_size:
                # Resize using interpolation for better numerical stability
                resized_detail = torch.nn.functional.interpolate(
                    detail.transpose(1, 2),  # [batch, embed, seq]
                    size=expected_size,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)  # back to [batch, seq, embed]
                corrected_details.append(resized_detail)
            else:
                corrected_details.append(detail)
        
        # Ensure approx has expected size
        if expected_sizes and approx.size(1) != expected_sizes[-1]:
            approx = torch.nn.functional.interpolate(
                approx.transpose(1, 2),  # [batch, embed, seq]
                size=expected_sizes[-1],
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # back to [batch, seq, embed]
        
        return approx, corrected_details
    
    def forward_optimized(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward wavelet transform using optimized MPS implementation
        
        Args:
            x: Input signal [batch_size, seq_length, embed_dim]
            
        Returns:
            approximation, detail_coefficients
        """
        batch_size, seq_length, embed_dim = x.shape
        
        # Store original sequence length for reconstruction
        self.original_seq_length = seq_length
        
        # Check if signal length is sufficient for the requested level
        min_length = 2 ** self.levels
        if seq_length < min_length:
            actual_levels = max(1, int(math.log2(seq_length)))
            print(f"Warning: Sequence length {seq_length} is too short for {self.levels} levels. Using {actual_levels} levels instead.")
            levels = actual_levels
        else:
            levels = self.levels
        
        # For very short sequences, use FFT-based method which is more robust for small sizes
        if seq_length < 16:
            return self.fft_forward(x)
            
        # Validate filter dimensions early to prevent later errors
        if self.dec_lo.dim() == 1:
            self.dec_lo = self.dec_lo.unsqueeze(0).unsqueeze(0)
            self.dec_hi = self.dec_hi.unsqueeze(0).unsqueeze(0)
        elif self.dec_lo.dim() == 2:
            self.dec_lo = self.dec_lo.unsqueeze(0)
            self.dec_hi = self.dec_hi.unsqueeze(0)
            
        # If filters are not on the correct device, move them
        if self.dec_lo.device != x.device:
            self.dec_lo = self.dec_lo.to(x.device)
            self.dec_hi = self.dec_hi.to(x.device)
        
        # Calculate expected output sizes for each level
        # This makes the transform more precise and matches what PyWavelets expects
        expected_sizes = []
        current_size = seq_length
        for _ in range(levels):
            # Each level reduces size approximately by half, with filter length adjustment
            # The exact formula depends on the decimation factor (2) and the filter size
            next_size = (current_size + self.filter_len - 1) // 2
            expected_sizes.append(next_size)
            current_size = next_size
        
        # Create lists for coefficients
        detail_coeffs = []
        approx = x
        
        # Apply wavelet transform for each level
        for level in range(levels):
            # Pad signal for convolution
            padded = self._pad_signal(approx)
            
            try:
                # Apply low-pass and high-pass filters with error handling
                approx_coeffs = self._conv1d_mps(padded, self.dec_lo)
                detail = self._conv1d_mps(padded, self.dec_hi)
                
                # Downsample - adjust to use expected sizes
                approx_coeffs = self._downsample(approx_coeffs)
                detail = self._downsample(detail)
            except RuntimeError as e:
                print(f"Warning: MPS wavelet transform failed at level {level}: {str(e)}")
                print(f"Falling back to FFT method")
                return self.fft_forward(x)
            
            # If actual size differs from expected, resize
            expected_size = expected_sizes[level]
            if approx_coeffs.size(1) != expected_size:
                # Adjust size to match expected dimensions
                if approx_coeffs.size(1) > expected_size:
                    # Trim if larger
                    approx_coeffs = approx_coeffs[:, :expected_size, :]
                else:
                    # Pad if smaller
                    pad_size = expected_size - approx_coeffs.size(1)
                    approx_coeffs = torch.nn.functional.pad(approx_coeffs, (0, 0, 0, pad_size), mode='constant')
            
            # Also adjust detail coefficients
            if detail.size(1) != expected_size:
                if detail.size(1) > expected_size:
                    detail = detail[:, :expected_size, :]
                else:
                    pad_size = expected_size - detail.size(1)
                    detail = torch.nn.functional.pad(detail, (0, 0, 0, pad_size), mode='constant')
            
            # Save detail coefficients
            detail_coeffs.append(detail)
            
            # Use approximation for next level
            approx = approx_coeffs
            
            # Check if we can continue
            if approx.size(1) < 2:
                break
        
        # Ensure consistent coefficient shapes before returning
        approx, detail_coeffs = self._ensure_coefficient_shapes(approx, detail_coeffs, expected_sizes[-1:])
        
        # Store expected sizes for reconstruction
        self.expected_sizes = expected_sizes
        
        return approx, detail_coeffs
    
    def inverse_optimized(self, approx: torch.Tensor, details: List[torch.Tensor]) -> torch.Tensor:
        """
        Inverse wavelet transform using optimized MPS implementation
        
        Args:
            approx: Approximation coefficients [batch_size, seq_length, embed_dim]
            details: List of detail coefficients
            
        Returns:
            Reconstructed signal [batch_size, seq_length, embed_dim]
        """
        # Safety check for empty details or invalid approx
        if not details or approx.size(1) == 0:
            return self.fft_inverse(approx, details)
        
        # For short sequences or if details don't match levels, use FFT method
        if approx.size(1) < 8 or len(details) != self.levels:
            return self.fft_inverse(approx, details)
        
        # Ensure consistent coefficient shapes
        if hasattr(self, 'expected_sizes'):
            # Use stored expected sizes from forward pass
            approx, details = self._ensure_coefficient_shapes(approx, details, self.expected_sizes[::-1])
        else:
            # Just ensure they're consistent with each other
            approx, details = self._ensure_coefficient_shapes(approx, details)
        
        # Create reconstruction filters (time-reversed)
        # Flip along the filter dimension (dim=2 for our [1, 1, filter_len] shape)
        rec_lo = self.dec_lo.flip(2)
        rec_hi = self.dec_hi.flip(2)
        
        # Start with approximation coefficients
        x = approx
        
        # Calculate the expected final sequence length
        # Use the stored original sequence length if available
        if hasattr(self, 'original_seq_length'):
            final_seq_len = self.original_seq_length
        else:
            # Fallback calculation - this is an approximation
            final_seq_len = approx.size(1) * (2 ** len(details))
        
        # Process from coarsest to finest level (reverse order of details)
        for level in range(len(details)-1, -1, -1):
            detail = details[level]
            
            # Ensure compatible sizes
            if x.size(1) != detail.size(1):
                # For proper reconstruction, detail and approx must have the same length
                # We'll use interpolation to match sizes
                x = torch.nn.functional.interpolate(
                    x.transpose(1, 2),  # [batch, embed, seq]
                    size=detail.size(1),
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)  # back to [batch, seq, embed]
            
            # Calculate expected size after upsampling
            expected_up_size = x.size(1) * 2
            
            # Upsample by inserting zeros
            x_up = torch.zeros(
                (x.size(0), expected_up_size, x.size(2)), 
                device=x.device, 
                dtype=x.dtype
            )
            x_up[:, ::2, :] = x
            
            # Upsample detail coefficients
            detail_up = torch.zeros(
                (detail.size(0), expected_up_size, detail.size(2)), 
                device=detail.device, 
                dtype=detail.dtype
            )
            detail_up[:, ::2, :] = detail
            
            # Apply reconstruction filters with proper padding
            approx_part = self._conv1d_mps(self._pad_signal(x_up), rec_lo)
            detail_part = self._conv1d_mps(self._pad_signal(detail_up), rec_hi)
            
            # Ensure both parts have the same size before combining
            if approx_part.size(1) != detail_part.size(1):
                min_size = min(approx_part.size(1), detail_part.size(1))
                approx_part = approx_part[:, :min_size, :]
                detail_part = detail_part[:, :min_size, :]
            
            # Combine
            x = approx_part + detail_part
        
        # Final adjustment to match original input size
        if x.size(1) != final_seq_len:
            x = torch.nn.functional.interpolate(
                x.transpose(1, 2),  # [batch, embed, seq]
                size=final_seq_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # back to [batch, seq, embed]
        
        return x
    
    def fft_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Fallback FFT-based forward transform implementation"""
        batch_size, seq_length, embed_dim = x.shape
        
        # Ensure we have filter parameters initialized
        if not hasattr(self, 'low_pass') or not hasattr(self, 'high_pass'):
            if self.use_learned:
                self.low_pass = nn.Parameter(torch.ones(1, 1, 32) * 0.7)
                self.high_pass = nn.Parameter(torch.ones(1, 1, 32) * 0.3)
            else:
                self.register_buffer('low_pass', torch.ones(1, 1, 32) * 0.7)
                self.register_buffer('high_pass', torch.ones(1, 1, 32) * 0.3)
        
        # Apply FFT - use efficient MPS implementation
        x_fft = torch.fft.rfft(x, dim=1)
        fft_size = x_fft.size(1)
        
        # Create results based on levels
        detail_tensors = []
        approx = x.clone()
        
        # Apply filters for each level
        for level in range(self.levels):
            # Calculate frequency bands for this level
            band_size = fft_size // (2 ** (level + 1))
            if band_size < 1:
                break
                
            # Apply low-pass and high-pass filters
            x_fft_lowpass = x_fft.clone()
            x_fft_highpass = x_fft.clone()
            
            # Shape filters for proper broadcasting
            # Use modulo to cycle through available filters if we have more levels than filter parameters
            filter_idx = level % self.low_pass.size(2)
            low_filter = torch.sigmoid(self.low_pass[:, :, filter_idx])
            high_filter = torch.sigmoid(self.high_pass[:, :, filter_idx])
            
            # Apply filters (simplified version)
            x_fft_lowpass[:, band_size:] = 0
            x_fft_highpass[:, :band_size] = 0
            
            # Apply learnable filter shapes
            x_fft_lowpass = x_fft_lowpass * low_filter.unsqueeze(-1)
            x_fft_highpass = x_fft_highpass * high_filter.unsqueeze(-1)
            
            # Inverse FFT to get coefficients - use efficient MPS implementation
            approx = torch.fft.irfft(x_fft_lowpass, n=seq_length, dim=1)
            detail = torch.fft.irfft(x_fft_highpass, n=seq_length, dim=1)
            
            # Save detail coefficients
            detail_tensors.append(detail)
            
            # Update FFT for next level
            x_fft = x_fft_lowpass
        
        return approx, detail_tensors
    
    def fft_inverse(self, approx: torch.Tensor, details: List[torch.Tensor]) -> torch.Tensor:
        """Fallback FFT-based inverse transform implementation"""
        batch_size, seq_length, embed_dim = approx.shape
        
        # Convert approximation to frequency domain - use efficient MPS implementation
        x_fft = torch.fft.rfft(approx, dim=1)
        
        # Add detail coefficients in frequency domain, handling different sizes
        for detail in details:
            # Ensure detail has the same sequence length as approx using interpolation if needed
            if detail.size(1) != seq_length:
                detail = torch.nn.functional.interpolate(
                    detail.transpose(1, 2),  # [batch, embed, seq]
                    size=seq_length,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)  # back to [batch, seq, embed]
            
            detail_fft = torch.fft.rfft(detail, dim=1)
            
            # Ensure FFT outputs have the same size
            min_freq_size = min(x_fft.size(1), detail_fft.size(1))
            x_fft_trim = x_fft[:, :min_freq_size, :]
            detail_fft_trim = detail_fft[:, :min_freq_size, :]
            
            # Add the frequency components
            x_fft = x_fft_trim + detail_fft_trim
        
        # Inverse FFT to get reconstructed signal - use efficient MPS implementation
        output = torch.fft.irfft(x_fft, n=seq_length, dim=1)
        
        return output
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass of the wavelet transform"""
        try:
            # Disable debug output if environment variable is set
            if os.environ.get('SIGNALLLM_DISABLE_DEBUG', '0') == '1':
                return self.forward_optimized(x)
                
            # First try optimized version
            return self.forward_optimized(x)
        except Exception as e:
            print(f"MPS wavelet transform failed, falling back to FFT: {str(e)}")
            # Fallback to FFT method without saving debug info
            return self.fft_forward(x)
    
    def inverse(self, approx: torch.Tensor, details: List[torch.Tensor]) -> torch.Tensor:
        """Apply inverse wavelet transform using the most efficient implementation"""
        return self.inverse_optimized(approx, details)


class MPSOptimizedAttention(nn.Module):
    """
    Optimized attention implementation for MPS leveraging sparse operations
    and batch convolutions to reduce computational complexity.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.head_dim = embed_dim // num_heads
        
        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Initialize weights
        self._reset_parameters()
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # For frequency domain attention
        self.filter_length = 32
        self.freq_filters = nn.Parameter(torch.randn(num_heads, self.filter_length))
        
        # Add wavelet transform module - needed for compatibility with WaveletAttention
        self.wavelet = MPSWaveletTransform(
            wavelet_type='db4',
            levels=3, 
            mode='reflect',
            use_learned=True
        )
        
        # Performance metrics
        self.register_buffer('compute_time', torch.zeros(1))
        self.register_buffer('complexity_ratio', torch.zeros(1))
    
    def _reset_parameters(self):
        """Initialize projection weights"""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.out_proj.bias)
    
    def _sparse_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                         mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute attention using sparse operations for better MPS performance"""
        # q, k, v shape: [batch, heads, seq, dim]
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Scaled dot-product
        scale = math.sqrt(head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # [batch, heads, seq, seq]
        
        # Apply mask if provided
        if mask is not None:
            # Adapt mask to attention shape
            if mask.dim() == 2:  # [batch, seq]
                mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq]
            
            # Apply mask (-inf for masked positions, will become 0 after softmax)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply attention via efficient sparse softmax
        # First convert to half precision for faster MPS computation
        scores_half = scores.half()
        
        # Apply sparsification with dynamic threshold
        # Keep only top 25% of the values, set rest to -inf
        top_k = max(4, int(seq_len * 0.25))  # At least 4 values or 25%
        top_values, _ = torch.topk(scores_half, top_k, dim=-1)
        threshold = top_values[:, :, :, -1].unsqueeze(-1)
        sparse_scores = torch.where(scores_half >= threshold, scores_half, torch.tensor(float('-inf'), 
                                                                                        device=scores.device, 
                                                                                        dtype=torch.half))
        
        # Convert back to float for softmax
        attn_weights = torch.softmax(sparse_scores.float(), dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights
        output = torch.matmul(attn_weights, v)  # [batch, heads, seq, dim]
        
        return output
    
    def _frequency_domain_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                  mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute attention in frequency domain for O(n log n) complexity"""
        # q, k, v shape: [batch, heads, seq, dim]
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Process each head
        outputs = []
        for h in range(num_heads):
            # Get head-specific components
            q_h = q[:, h]  # [batch, seq, dim]
            k_h = k[:, h]  # [batch, seq, dim]
            v_h = v[:, h]  # [batch, seq, dim]
            
            # Convert to frequency domain
            q_fft = torch.fft.rfft(q_h, dim=1)  # [batch, fft_size, dim]
            k_fft = torch.fft.rfft(k_h, dim=1)  # [batch, fft_size, dim]
            v_fft = torch.fft.rfft(v_h, dim=1)  # [batch, fft_size, dim]
            
            # Get frequency domain filter for this head
            filter_h = torch.sigmoid(self.freq_filters[h])  # [filter_length]
            
            # Ensure filter matches FFT size
            fft_size = q_fft.size(1)
            if fft_size != self.filter_length:
                # Interpolate to match FFT size
                filter_h = torch.nn.functional.interpolate(
                    filter_h.unsqueeze(0).unsqueeze(0), 
                    size=fft_size,
                    mode='linear',
                    align_corners=False
                ).squeeze(0).squeeze(0)
            
            # Apply filter
            filter_h = filter_h.to(q_fft.device)
            
            # Compute attention in frequency domain (element-wise multiplications)
            # This is equivalent to convolution in time domain
            attn_fft = q_fft * k_fft * filter_h.unsqueeze(0).unsqueeze(-1)
            
            # Apply to values
            out_fft = attn_fft * v_fft
            
            # Convert back to time domain
            out = torch.fft.irfft(out_fft, n=seq_len, dim=1)  # [batch, seq, dim]
            outputs.append(out)
        
        # Combine all heads [batch, heads, seq, dim]
        output = torch.stack(outputs, dim=1)
        
        return output
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, 
               use_sparse: bool = True, use_wavelet: bool = False) -> torch.Tensor:
        """
        Apply efficient attention mechanism
        
        Args:
            x: Input tensor [batch_size, seq_length, embed_dim]
            mask: Attention mask [batch_size, seq_length]
            use_sparse: Whether to use sparse attention (True) or frequency domain (False)
            use_wavelet: Whether to use wavelet-based attention processing
            
        Returns:
            Output tensor [batch_size, seq_length, embed_dim]
        """
        batch_size, seq_length, _ = x.shape
        
        # Check if we should use wavelet-based processing (for compatibility with WaveletAttention)
        if use_wavelet and hasattr(self, 'wavelet'):
            # Apply wavelet transform
            approx, details = self.wavelet(x)
            
            # Project approximation coefficients
            approx_q = self.q_proj(approx)
            approx_k = self.k_proj(approx)
            approx_v = self.v_proj(approx)
            
            # Apply attention to approximation
            approx_attn = self._sparse_attention(
                approx_q.view(batch_size, approx.size(1), self.num_heads, self.head_dim).transpose(1, 2),
                approx_k.view(batch_size, approx.size(1), self.num_heads, self.head_dim).transpose(1, 2),
                approx_v.view(batch_size, approx.size(1), self.num_heads, self.head_dim).transpose(1, 2),
                None
            ).transpose(1, 2).contiguous().view(batch_size, approx.size(1), self.embed_dim)
            
            # Project back
            approx_out = self.out_proj(approx_attn)
            
            # Reconstruct with processed approximation
            output = self.wavelet.inverse(approx_out, details)
            
            return output
        
        # For very short sequences, use standard attention
        if seq_length <= 16:
            use_sparse = True
        
        # Track complexity for analytical purposes
        if self.training:
            # Standard attention: O(n²)
            standard_ops = seq_length ** 2
            # Frequency domain: O(n log n)
            freq_ops = seq_length * math.log2(seq_length) if seq_length > 1 else 1
            # Update complexity ratio
            self.complexity_ratio[0] = freq_ops / standard_ops
        
        # Start timing
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time is not None:
            start_time.record()
        
        # Project to queries, keys, values
        q = self.q_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        
        # Transpose for attention
        q = q.transpose(1, 2)  # [batch, heads, seq, dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply attention mechanism
        if use_sparse:
            output = self._sparse_attention(q, k, v, mask)
        else:
            output = self._frequency_domain_attention(q, k, v, mask)
        
        # Transpose back and reshape
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)
        
        # Final projection
        output = self.out_proj(output)
        
        # End timing
        if end_time is not None:
            end_time.record()
            torch.cuda.synchronize()
            self.compute_time[0] = start_time.elapsed_time(end_time)
        
        return output


# Helper function to replace standard modules with MPS-optimized versions
def optimize_for_mps(model, device=None):
    """
    Replace standard modules with MPS-optimized versions
    
    Args:
        model: The model to optimize
        device: Device to use (will infer from model if None)
        
    Returns:
        Optimized model
    """
    from signalllm_hrfevo_poc import WaveletTransform, FourierConvolutionAttention, WaveletAttention
    
    # Dictionary to track replacements
    replacements = {
        'WaveletTransform': 0,
        'Attention': 0
    }
    
    # Get the device 
    if device is None:
        # Try to get device from model parameters, or use default MPS
        try:
            # Find the first parameter with a device
            for param in model.parameters():
                if param.device:
                    device = param.device
                    break
        except (StopIteration, RuntimeError):
            # If no parameters or other error, use MPS if available
            if torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
    
    # Recursively replace modules
    for name, module in list(model.named_children()):
        # Skip modules that are None
        if module is None:
            continue
            
        # Optimize children modules first (recursive)
        try:
            model._modules[name] = optimize_for_mps(module, device)
        except Exception as e:
            print(f"Warning: Could not optimize submodule {name}: {e}")
        
        # Replace WaveletTransform with optimized version
        if isinstance(module, WaveletTransform):
            try:
                # Create optimized replacement with same parameters
                optimized = MPSWaveletTransform(
                    wavelet_type=module.wavelet_type,
                    levels=module.levels,
                    mode=module.mode,
                    use_learned=module.use_learned
                ).to(device)
                
                # Copy trained parameters if they exist
                if hasattr(module, 'low_pass') and hasattr(optimized, 'low_pass'):
                    if isinstance(module.low_pass, torch.Tensor):
                        optimized.low_pass.data.copy_(module.low_pass.data.to(device))
                if hasattr(module, 'high_pass') and hasattr(optimized, 'high_pass'):
                    if isinstance(module.high_pass, torch.Tensor):
                        optimized.high_pass.data.copy_(module.high_pass.data.to(device))
                
                # Replace module
                model._modules[name] = optimized
                replacements['WaveletTransform'] += 1
                print(f"Replaced WaveletTransform module {name} with MPS-optimized version")
            except Exception as e:
                print(f"Warning: Could not replace WaveletTransform {name}: {e}")
        
        # Replace attention mechanisms
        elif isinstance(module, (FourierConvolutionAttention, WaveletAttention)):
            try:
                # Create optimized attention
                optimized = MPSOptimizedAttention(
                    embed_dim=module.embed_dim,
                    num_heads=module.num_heads,
                    dropout=0.1  # Default
                ).to(device)
                
                # Copy weights
                if hasattr(module, 'q_proj') and hasattr(optimized, 'q_proj'):
                    optimized.q_proj.weight.data.copy_(module.q_proj.weight.data.to(device))
                    optimized.q_proj.bias.data.copy_(module.q_proj.bias.data.to(device))
                    
                    optimized.k_proj.weight.data.copy_(module.k_proj.weight.data.to(device))
                    optimized.k_proj.bias.data.copy_(module.k_proj.bias.data.to(device))
                    
                    optimized.v_proj.weight.data.copy_(module.v_proj.weight.data.to(device))
                    optimized.v_proj.bias.data.copy_(module.v_proj.bias.data.to(device))
                    
                    optimized.out_proj.weight.data.copy_(module.out_proj.weight.data.to(device))
                    optimized.out_proj.bias.data.copy_(module.out_proj.bias.data.to(device))
                
                # Copy wavelet parameters for WaveletAttention
                if isinstance(module, WaveletAttention) and hasattr(module, 'wavelet'):
                    # Configure optimized wavelet with same parameters
                    if hasattr(module.wavelet, 'wavelet_type'):
                        optimized.wavelet = MPSWaveletTransform(
                            wavelet_type=module.wavelet.wavelet_type,
                            levels=getattr(module.wavelet, 'levels', 3),
                            mode=getattr(module.wavelet, 'mode', 'reflect'),
                            use_learned=getattr(module.wavelet, 'use_learned', True)
                        ).to(device)
                    
                    # Copy wavelet weights if available
                    if hasattr(module.wavelet, 'dec_lo') and hasattr(optimized.wavelet, 'dec_lo'):
                        if isinstance(module.wavelet.dec_lo, torch.Tensor):
                            optimized.wavelet.dec_lo.data.copy_(module.wavelet.dec_lo.data.to(device))
                    if hasattr(module.wavelet, 'dec_hi') and hasattr(optimized.wavelet, 'dec_hi'):
                        if isinstance(module.wavelet.dec_hi, torch.Tensor):
                            optimized.wavelet.dec_hi.data.copy_(module.wavelet.dec_hi.data.to(device))
                
                # Replace module
                model._modules[name] = optimized
                replacements['Attention'] += 1
                print(f"Replaced Attention module {name} with MPS-optimized version")
            except Exception as e:
                print(f"Warning: Could not replace Attention {name}: {e}")
    
    # Ensure the entire model is on the correct device
    for param in model.parameters():
        if param.device != device:
            param.data = param.data.to(device)
    
    # Check buffers too (non-parameter tensors)
    for buffer_name, buffer in model.named_buffers():
        if buffer.device != device:
            # Sanitize buffer names - replace dots with underscores to avoid KeyError
            safe_buffer_name = buffer_name.split(".")[-1]  # Get just the last part of the name
            try:
                model.register_buffer(safe_buffer_name, buffer.to(device))
            except Exception as e:
                print(f"Warning: Could not register buffer {buffer_name}: {e}")
    
    return model

# Function to set up MPS-specific optimizations
def setup_mps_optimizations():
    """Configure MPS-specific optimizations"""
    if not torch.backends.mps.is_available():
        print("MPS not available. Cannot apply MPS-specific optimizations.")
        return False
    
    # Set MPS-specific flags
    torch.backends.mps.enable_cache = True
    
    # Set threading optimizations
    torch.set_num_threads(6)  # Adjust based on core count
    
    print("MPS optimizations enabled.")
    return True 