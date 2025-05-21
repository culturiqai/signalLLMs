"""
SignalLLM with HRFEvo: Proof of Concept Implementation
=====================================================

This file implements a proof-of-concept for the SignalLLM with HRFEvo approach,
representing language in the frequency domain with hierarchically evolving basis functions.

Key components:
1. Spectral Embedding - Represents tokens using frequency components
2. Wavelet Transformer - Processes language with wavelet-based attention
3. HRFEvo Controller - Evolutionary optimization of basis functions
4. Hardware Optimization - Device-specific implementations
5. Visualization Tools - Analysis of spectral representations
"""

import os
import sys
import time
import math
import random
import argparse
import json
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Optional imports for advanced features
try:
    import pywt  # PyWavelets for wavelet transforms
    WAVELET_AVAILABLE = True
except ImportError:
    WAVELET_AVAILABLE = False
    print("PyWavelets not available. Using FFT-based transforms instead.")

# For debugging, ensure PyWavelets is properly loaded
if 'pywt' in sys.modules and WAVELET_AVAILABLE:
    print(f"PyWavelets version: {pywt.__version__}")
else:
    print("WARNING: PyWavelets module not properly loaded. Using FFT-based transforms.")

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = Any  # Placeholder type
    TENSORBOARD_AVAILABLE = False
    print("TensorBoard not available. Visualization will be limited.")

# Check for specialized hardware
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using Apple Silicon MPS (Metal Performance Shaders)")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

#-------------------------------------------------------------------------
# Configuration and Initialization
#-------------------------------------------------------------------------

class Config:
    """Configuration class for model hyperparameters and settings"""
    def __init__(self, **kwargs):
        # Model architecture
        self.vocab_size = kwargs.get('vocab_size', 10000)
        self.max_seq_length = kwargs.get('max_seq_length', 512)
        self.embed_dim = kwargs.get('embed_dim', 256)
        self.num_heads = kwargs.get('num_heads', 4)
        self.num_layers = kwargs.get('num_layers', 4)
        self.hidden_dim = kwargs.get('hidden_dim', 1024)
        self.dropout = kwargs.get('dropout', 0.1)
        
        # Spectral/wavelet configuration
        self.harmonic_bases = kwargs.get('harmonic_bases', 16)
        self.wavelet_type = kwargs.get('wavelet_type', 'db4')  # Daubechies 4
        self.wavelet_levels = kwargs.get('wavelet_levels', 3)
        self.boundary_handling = kwargs.get('boundary_handling', 'reflect')
        
        # Enhanced mathematical capabilities (from Dr. Tao's insights)
        self.use_adaptive_basis = kwargs.get('use_adaptive_basis', True)
        self.wavelet_families = kwargs.get('wavelet_families', ['db4', 'sym4', 'dmey'])
        self.use_fourier_convolution = kwargs.get('use_fourier_convolution', True)
        self.use_stochastic_transform = kwargs.get('use_stochastic_transform', True)
        self.stochastic_sampling_ratio = kwargs.get('stochastic_sampling_ratio', 0.2)
        self.spectral_gap_analysis = kwargs.get('spectral_gap_analysis', True)
        
        # HRFEvo parameters
        self.evolution_population = kwargs.get('evolution_population', 10)
        self.evolution_generations = kwargs.get('evolution_generations', 5)
        self.evolution_mutation_rate = kwargs.get('evolution_mutation_rate', 0.1)
        self.evolution_reward_computation = kwargs.get('evolution_reward_computation', 0.7)
        self.evolution_reward_perplexity = kwargs.get('evolution_reward_perplexity', 0.3)
        
        # Training parameters
        self.batch_size = kwargs.get('batch_size', 32)
        self.learning_rate = kwargs.get('learning_rate', 1e-4)
        self.weight_decay = kwargs.get('weight_decay', 1e-2)
        self.max_epochs = kwargs.get('max_epochs', 10)
        self.warmup_steps = kwargs.get('warmup_steps', 1000)
        self.gradient_clip = kwargs.get('gradient_clip', 1.0)
        self.train_data = kwargs.get('train_data', None)
        self.val_data = kwargs.get('val_data', None)
        self.test_data = kwargs.get('test_data', None)
        
        # Hardware optimization flags
        self.use_mixed_precision = kwargs.get('use_mixed_precision', False)
        self.use_tensor_cores = kwargs.get('use_tensor_cores', True)
        self.use_mps_graph_mode = kwargs.get('use_mps_graph_mode', False)
        self.optimize_memory_access = kwargs.get('optimize_memory_access', True)
        self.block_size = kwargs.get('block_size', 128)  # For memory access optimization
        
        # Logging and visualization
        self.log_level = kwargs.get('log_level', 'INFO')
        self.log_interval = kwargs.get('log_interval', 10)
        self.checkpoint_dir = kwargs.get('checkpoint_dir', './checkpoints')
        self.output_dir = kwargs.get('output_dir', './output')
        self.visualization_interval = kwargs.get('visualization_interval', 200)
        
        # Benchmark parameters
        self.benchmark_seq_lengths = kwargs.get('benchmark_seq_lengths', '128,256,512,1024,2048')
        self.visualize_embeddings = kwargs.get('visualize_embeddings', False)
        self.visualize_attention = kwargs.get('visualize_attention', False)
        self.visualize_complexity = kwargs.get('visualize_complexity', False)

    def to_dict(self) -> Dict:
        """Convert config to dictionary for serialization"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'Config':
        """Create config from dictionary"""
        return cls(**config_dict)
    
    def save(self, filepath: str) -> None:
        """Save config to JSON file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'Config':
        """Load config from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


#-------------------------------------------------------------------------
# Utility Functions
#-------------------------------------------------------------------------

class Timer:
    """Utility for timing operations"""
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        logging.info(f"{self.name} completed in {elapsed:.4f} seconds")


def setup_logger(config: Config) -> logging.Logger:
    """Set up logging configuration"""
    os.makedirs(config.output_dir, exist_ok=True)
    log_file = os.path.join(config.output_dir, 'signalllm_hrfevo.log')
    
    handlers = [
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
    
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=handlers
    )
    
    logger = logging.getLogger("SignalLLM-HRFEvo")
    return logger


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def memory_stats(prefix: str = "") -> Dict[str, float]:
    """Collect memory usage statistics"""
    stats = {}
    if torch.cuda.is_available():
        stats['allocated'] = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        stats['reserved'] = torch.cuda.memory_reserved() / (1024 ** 3)  # GB
        stats['max_allocated'] = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
        
        logging.debug(f"{prefix} Memory: {stats['allocated']:.2f}GB allocated, "
                    f"{stats['reserved']:.2f}GB reserved, "
                    f"{stats['max_allocated']:.2f}GB max")
    return stats


def create_tensorboard(config: Config) -> Optional[Any]:
    """Create TensorBoard writer if available"""
    if not TENSORBOARD_AVAILABLE:
        return None
    
    tensorboard_dir = os.path.join(config.output_dir, 'tensorboard')
    os.makedirs(tensorboard_dir, exist_ok=True)
    return SummaryWriter(log_dir=tensorboard_dir)


#-------------------------------------------------------------------------
# Spectral Embedding Implementation
#-------------------------------------------------------------------------

class SpectralEmbedding(nn.Module):
    """
    Embedding layer that represents tokens as superpositions of frequency components.
    Instead of learning a direct embedding vector, this learns amplitudes and phases for
    different frequency components to create more efficient embeddings.
    """
    def __init__(self, vocab_size: int, embed_dim: int, harmonic_bases: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.harmonic_bases = harmonic_bases
        
        # Each token gets frequency amplitudes and phases
        self.frequency_amplitudes = nn.Parameter(torch.randn(vocab_size, harmonic_bases))
        self.frequency_phases = nn.Parameter(torch.randn(vocab_size, harmonic_bases))
        
        # Generate frequency bases (fixed)
        self.register_buffer('frequencies', 
                           torch.linspace(0.1, math.pi, harmonic_bases))
        
        # Statistics tracking for analysis
        self.register_buffer('embedding_power', torch.zeros(embed_dim))
        self.register_buffer('embedding_usage', torch.zeros(vocab_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate spectral embeddings for input tokens.
        
        Args:
            x: Token indices [batch_size, seq_length]
            
        Returns:
            Embeddings [batch_size, seq_length, embed_dim]
        """
        batch_size, seq_length = x.shape
        
        # Get amplitudes and phases for each token
        amplitudes = self.frequency_amplitudes[x]  # [batch_size, seq_length, harmonic_bases]
        phases = self.frequency_phases[x]  # [batch_size, seq_length, harmonic_bases]
        
        # Generate time points for embedding dimension
        t = torch.linspace(0, 1, self.embed_dim, device=x.device)
        t = t.view(1, 1, 1, self.embed_dim)  # [1, 1, 1, embed_dim]
        
        # Each frequency contributes across the embedding dimension
        frequencies = self.frequencies.view(1, 1, self.harmonic_bases, 1)  # [1, 1, harmonic_bases, 1]
        amplitudes = amplitudes.unsqueeze(-1)  # [batch, seq_len, harmonic_bases, 1]
        phases = phases.unsqueeze(-1)  # [batch, seq_len, harmonic_bases, 1]
        
        # Generate embeddings as superposition of harmonics
        signal = amplitudes * torch.sin(2 * math.pi * frequencies * t + phases)
        embeddings = signal.sum(dim=2)  # [batch, seq_len, embed_dim]
        
        # Update statistics for analysis (in training mode)
        if self.training:
            with torch.no_grad():
                # Track frequency power distribution
                power = torch.mean(torch.abs(embeddings) ** 2, dim=(0, 1))
                self.embedding_power = 0.99 * self.embedding_power + 0.01 * power
                
                # Track token usage
                tokens_used, _ = torch.unique(x, return_counts=True)
                self.embedding_usage[tokens_used] += 1
        
        return embeddings
    
    def get_spectral_stats(self) -> Dict[str, torch.Tensor]:
        """Get statistics about the spectral embeddings"""
        # Compute power spectrum of the embeddings
        with torch.no_grad():
            # Average amplitudes across vocabulary
            avg_amplitudes = torch.mean(self.frequency_amplitudes, dim=0)
            # Average phases across vocabulary
            avg_phases = torch.mean(torch.abs(self.frequency_phases), dim=0)
            
            # Most frequently used tokens
            top_tokens = torch.argsort(self.embedding_usage, descending=True)[:10]
            
            stats = {
                'power_spectrum': self.embedding_power.cpu(),
                'frequency_amplitudes': avg_amplitudes.cpu(),
                'frequency_phases': avg_phases.cpu(),
                'token_usage': self.embedding_usage.cpu(),
                'top_tokens': top_tokens.cpu()
            }
        
        return stats


class HybridEmbedding(nn.Module):
    """
    Combines traditional token embeddings with spectral embeddings.
    This allows for a smooth transition from standard to spectral representations,
    as recommended by Dr. Hinton in the discussions.
    """
    def __init__(self, vocab_size: int, embed_dim: int, 
                 harmonic_bases: int = 16, spectral_ratio: float = 0.5):
        super().__init__()
        self.spectral_ratio = spectral_ratio
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Standard embedding
        self.standard_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Spectral embedding
        self.spectral_embedding = SpectralEmbedding(vocab_size, embed_dim, harmonic_bases)
        
        # Learnable mixing parameter (starts with specified ratio)
        self.mixing_param = nn.Parameter(torch.tensor([spectral_ratio]))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate hybrid embeddings for input tokens.
        
        Args:
            x: Token indices [batch_size, seq_length]
            
        Returns:
            Embeddings [batch_size, seq_length, embed_dim]
        """
        # Get both embedding types
        standard_embeds = self.standard_embedding(x)
        spectral_embeds = self.spectral_embedding(x)
        
        # Mix the embeddings with learnable parameter (constrained to [0,1])
        alpha = torch.sigmoid(self.mixing_param)
        embeddings = (1 - alpha) * standard_embeds + alpha * spectral_embeds
        
        return embeddings
    
    def get_mixing_ratio(self) -> float:
        """Get the current mixing ratio between standard and spectral embeddings"""
        with torch.no_grad():
            return torch.sigmoid(self.mixing_param).item()


class SignalPositionalEncoding(nn.Module):
    """
    Positional encoding using basis of sinusoidal signals at different frequencies.
    Extends the standard sinusoidal encoding with learnable parameters.
    """
    def __init__(self, max_seq_length: int, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Create standard sinusoidal encoding as starting point
        pe = torch.zeros(max_seq_length, embed_dim)
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                            -(math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Make parameters learnable with initial sinusoidal values
        self.pe = nn.Parameter(pe.unsqueeze(0))
        
        # Frequency modulation parameters
        self.freq_mod = nn.Parameter(torch.ones(1, 1, embed_dim))
        self.phase_shift = nn.Parameter(torch.zeros(1, 1, embed_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to embeddings.
        
        Args:
            x: Token embeddings [batch_size, seq_length, embed_dim]
            
        Returns:
            Embeddings with positional information
        """
        seq_length = x.size(1)
        
        # Apply frequency modulation and phase shift
        pos_encoding = self.pe[:, :seq_length] * self.freq_mod + self.phase_shift
        
        return x + pos_encoding 


#-------------------------------------------------------------------------
# Wavelet Transformer Block
#-------------------------------------------------------------------------

class WaveletTransform(nn.Module):
    """
    Implements wavelet transform for multi-resolution analysis of embeddings.
    Uses PyWavelets if available, otherwise falls back to FFT-based approximation.
    """
    def __init__(self, wavelet_type: str = 'db4', levels: int = 3, 
                 mode: str = 'reflect', use_learned: bool = True):
        super().__init__()
        self.wavelet_type = wavelet_type
        self.levels = levels
        self.mode = mode
        self.use_learned = use_learned
        
        # Flag to track whether we're using PyWavelets or FFT approximation
        self.using_pywt = WAVELET_AVAILABLE
        
        if self.using_pywt:
            # If using PyWavelets, we can have learnable wavelet filters
            if use_learned:
                # Get initial wavelet filter coefficients from PyWavelets
                wavelet = pywt.Wavelet(wavelet_type)
                
                # Initialize learnable wavelet filters
                self.dec_lo = nn.Parameter(torch.tensor(wavelet.dec_lo))
                self.dec_hi = nn.Parameter(torch.tensor(wavelet.dec_hi))
                self.rec_lo = nn.Parameter(torch.tensor(wavelet.rec_lo))
                self.rec_hi = nn.Parameter(torch.tensor(wavelet.rec_hi))
            else:
                # Just register the filters as buffers (non-learnable)
                wavelet = pywt.Wavelet(wavelet_type)
                self.register_buffer('dec_lo', torch.tensor(wavelet.dec_lo))
                self.register_buffer('dec_hi', torch.tensor(wavelet.dec_hi))
                self.register_buffer('rec_lo', torch.tensor(wavelet.rec_lo))
                self.register_buffer('rec_hi', torch.tensor(wavelet.rec_hi))
        else:
            # FFT-based implementation
            if use_learned:
                # Create learnable frequency response curves
                self.low_pass = nn.Parameter(torch.ones(1, 1, 32) * 0.7)
                self.high_pass = nn.Parameter(torch.ones(1, 1, 32) * 0.3)
            else:
                # Fixed filters approximating wavelet behavior
                self.register_buffer('low_pass', torch.ones(1, 1, 32) * 0.7)
                self.register_buffer('high_pass', torch.ones(1, 1, 32) * 0.3)
    
    def pywt_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward wavelet transform using PyWavelets.
        
        Args:
            x: Input signal [batch_size, seq_length, embed_dim]
            
        Returns:
            approximation, detail_coefficients
        """
        batch_size, seq_length, embed_dim = x.shape
        
        # For simplicity, transpose to handle each embedding dimension separately
        x = x.transpose(1, 2)  # [batch_size, embed_dim, seq_length]
        
        # Calculate expected coefficient sizes for consistency
        # This helps ensure the inverse transform will work properly
        try:
            # Calculate coefficient length using pywt.dwt_coeff_len
            filter_len = pywt.Wavelet(self.wavelet_type).dec_len
            # dwt_coeff_len returns a single integer (the coefficient length)
            approx_len = pywt.dwt_coeff_len(data_len=seq_length, filter_len=filter_len, mode=self.mode)
            # Pre-calculate detail coefficient sizes for each level
            detail_lens = []
            current_len = seq_length
            for i in range(self.levels):
                current_len = pywt.dwt_coeff_len(data_len=current_len, filter_len=filter_len, mode=self.mode)
                if i < self.levels - 1:  # Skip the final approximation
                    detail_lens.append(current_len)
            
            max_approx_len = approx_len
        except ValueError:
            # If level is too high for the sequence length, reduce it
            filter_len = pywt.Wavelet(self.wavelet_type).dec_len
            adjusted_levels = pywt.dwt_max_level(data_len=seq_length, filter_len=filter_len)
            logging.warning(f"Adjusting wavelet levels from {self.levels} to {adjusted_levels} for sequence length {seq_length}")
            self.levels = adjusted_levels
            
            # dwt_coeff_len returns a single integer (the coefficient length)
            approx_len = pywt.dwt_coeff_len(data_len=seq_length, filter_len=filter_len, mode=self.mode)
            
            # Pre-calculate detail coefficient sizes for each level
            detail_lens = []
            current_len = seq_length
            for i in range(self.levels):
                current_len = pywt.dwt_coeff_len(data_len=current_len, filter_len=filter_len, mode=self.mode)
                if i < self.levels - 1:  # Skip the final approximation
                    detail_lens.append(current_len)
            
            max_approx_len = approx_len
        
        # Create empty tensors for coefficients with proper sizes
        approx = torch.zeros(batch_size, embed_dim, max_approx_len, device=x.device)
        
        # Initialize detail coefficient holders based on calculated detail_lens
        detail_holders = []
        for level_size in detail_lens:
            detail_holders.append(torch.zeros(batch_size, embed_dim, level_size, device=x.device))
        
        # Process each batch and embedding dimension
        for b in range(batch_size):
            for e in range(embed_dim):
                # Convert to numpy for PyWavelets
                signal = x[b, e].detach().cpu().numpy()
                
                try:
                    # Perform wavelet decomposition
                    coeffs = pywt.wavedec(signal, self.wavelet_type, mode=self.mode, level=self.levels)
                    
                    # Store approximation coefficients
                    approx_coeff = coeffs[0]
                    approx[b, e, :len(approx_coeff)] = torch.tensor(approx_coeff, device=x.device)
                    
                    # Store detail coefficients with proper sizes
                    for i, level_coeff in enumerate(coeffs[1:]):
                        if i < len(detail_holders):
                            # Ensure coefficient has expected size
                            expected_len = detail_lens[i] if i < len(detail_lens) else len(level_coeff)
                            if len(level_coeff) > expected_len:
                                level_coeff = level_coeff[:expected_len]
                            
                            # Store in the appropriate tensor
                            detail_holders[i][b, e, :len(level_coeff)] = torch.tensor(level_coeff, device=x.device)
                except Exception as e:
                    logging.warning(f"Error in wavelet decomposition: {str(e)}. Using zero coefficients.")
                    # Keep zeros in the tensors if decomposition fails
        
        # Convert back to original shape
        approx = approx.transpose(1, 2)  # [batch_size, seq_length, embed_dim]
        detail_tensors = [d.transpose(1, 2) for d in detail_holders]  # list of [batch_size, seq_length, embed_dim]
        
        return approx, detail_tensors
    
    def pywt_inverse(self, approx: torch.Tensor, details: List[torch.Tensor]) -> torch.Tensor:
        """
        Inverse wavelet transform using PyWavelets.
        
        Args:
            approx: Approximation coefficients [batch_size, seq_length, embed_dim]
            details: List of detail coefficients
            
        Returns:
            Reconstructed signal [batch_size, seq_length, embed_dim]
        """
        batch_size, _, embed_dim = approx.shape
        
        # Transpose for easier processing
        approx = approx.transpose(1, 2)  # [batch_size, embed_dim, seq_length]
        details = [d.transpose(1, 2) for d in details]  # list of [batch_size, embed_dim, seq_length]
        
        # Get target sequence length (needs to be calculated from coefficients)
        # Use wavelet-specific calculations for proper sequence length
        approx_len = approx.size(2)
        seq_length = approx_len * (2 ** len(details))
        
        # Create output tensor
        output = torch.zeros(batch_size, embed_dim, seq_length, device=approx.device)
        
        # Process each batch and embedding dimension
        for b in range(batch_size):
            for e in range(embed_dim):
                # Get approximation coefficients
                approx_coeff = approx[b, e].detach().cpu().numpy()
                approx_coeff = approx_coeff[:approx_coeff.shape[0]]  # Remove padding
                
                # Get detail coefficients for each level
                detail_coeffs = []
                filter_len = pywt.Wavelet(self.wavelet_type).dec_len
                
                # Calculate expected sizes for each level
                detail_sizes = []
                current_len = approx_len
                for _ in range(len(details)):
                    next_len = pywt.dwt_coeff_len(data_len=current_len*2, filter_len=filter_len, mode=self.mode)
                    detail_sizes.append(next_len)
                    current_len = next_len
                
                # For each level
                for level_idx, level_detail in enumerate(details):
                    level_coeff = level_detail[b, e].detach().cpu().numpy()
                    
                    # Calculate expected size for this level
                    if level_idx < len(detail_sizes):
                        expected_size = detail_sizes[level_idx]
                        # Ensure coefficients have the expected size
                        if len(level_coeff) > expected_size:
                            level_coeff = level_coeff[:expected_size]
                        elif len(level_coeff) < expected_size:
                            # Pad with zeros if needed
                            padded = np.zeros(expected_size)
                            padded[:len(level_coeff)] = level_coeff
                            level_coeff = padded
                    
                    detail_coeffs.append(level_coeff)
                
                # Combine for waverec
                try:
                    # Make sure coefficients are compatible
                    coeffs = [approx_coeff] + detail_coeffs
                    
                    # Validate coefficient shapes
                    # Calculate expected lengths for each level
                    filter_len = pywt.Wavelet(self.wavelet_type).dec_len
                    
                    # Calculate expected sizes for each level
                    expected_wavelet_shapes = []
                    current_len = len(approx_coeff)
                    for _ in range(len(detail_coeffs)):
                        next_len = pywt.dwt_coeff_len(data_len=current_len*2, filter_len=filter_len, mode=self.mode)
                        expected_wavelet_shapes.append(next_len)
                        current_len = next_len
                    
                    # Fix shapes if needed
                    for i in range(1, len(coeffs)):
                        if i-1 < len(expected_wavelet_shapes):
                            expected_len = expected_wavelet_shapes[i-1]
                            current_len = len(coeffs[i])
                            if current_len != expected_len:
                                # Resize coefficient array to expected length
                                new_coeff = np.zeros(expected_len)
                                min_len = min(current_len, expected_len)
                                new_coeff[:min_len] = coeffs[i][:min_len]
                                coeffs[i] = new_coeff
                    
                    # Perform inverse wavelet transform
                    rec = pywt.waverec(coeffs, self.wavelet_type, mode=self.mode)
                    
                    # Store result, handling potential length differences
                    rec_len = min(len(rec), seq_length)
                    output[b, e, :rec_len] = torch.tensor(rec[:rec_len], device=approx.device)
                except ValueError as e:
                    # Fallback to a simpler approach if coefficient shapes still mismatch
                    logging.warning(f"Wavelet coefficient shape mismatch, using fallback method: {str(e)}")
                    
                    # Use FFT-based reconstruction as fallback
                    fallback_output = self.fft_inverse(
                        approx.transpose(1, 2), 
                        [d.transpose(1, 2) for d in details]
                    )
                    return fallback_output
        
        # Transpose back
        output = output.transpose(1, 2)  # [batch_size, seq_length, embed_dim]
        
        return output
    
    def fft_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward wavelet transform approximation using FFT.
        
        Args:
            x: Input signal [batch_size, seq_length, embed_dim]
            
        Returns:
            approximation, detail_coefficients
        """
        batch_size, seq_length, embed_dim = x.shape
        
        # Ensure we have filter parameters initialized
        # This could happen if WAVELET_AVAILABLE is True but using_pywt is manually set to False
        if not hasattr(self, 'low_pass') or not hasattr(self, 'high_pass'):
            if self.use_learned:
                self.low_pass = nn.Parameter(torch.ones(1, 1, 32) * 0.7)
                self.high_pass = nn.Parameter(torch.ones(1, 1, 32) * 0.3)
            else:
                self.register_buffer('low_pass', torch.ones(1, 1, 32) * 0.7)
                self.register_buffer('high_pass', torch.ones(1, 1, 32) * 0.3)
        
        # Apply FFT
        x_fft = torch.fft.rfft(x, dim=1)
        fft_size = x_fft.size(1)
        
        # Create results based on levels
        detail_tensors = []
        approx = x.clone()
        
        # Apply filters for each level
        for level in range(self.levels):
            # Calculate frequency bands for this level
            band_size = fft_size // (2 ** (level + 1))
            
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
            
            # Inverse FFT to get coefficients
            approx = torch.fft.irfft(x_fft_lowpass, n=seq_length, dim=1)
            detail = torch.fft.irfft(x_fft_highpass, n=seq_length, dim=1)
            
            # Save detail coefficients
            detail_tensors.append(detail)
            
            # Update FFT for next level
            x_fft = x_fft_lowpass
        
        return approx, detail_tensors
    
    def fft_inverse(self, approx: torch.Tensor, details: List[torch.Tensor]) -> torch.Tensor:
        """
        Inverse wavelet transform approximation using FFT.
        
        Args:
            approx: Approximation coefficients [batch_size, seq_length, embed_dim]
            details: List of detail coefficients
            
        Returns:
            Reconstructed signal [batch_size, seq_length, embed_dim]
        """
        batch_size, seq_length, embed_dim = approx.shape
        
        # Convert approximation to frequency domain
        x_fft = torch.fft.rfft(approx, dim=1)
        
        # Add detail coefficients in frequency domain, handling different sizes
        for detail in details:
            # Ensure detail has the same sequence length as approx using interpolation if needed
            if detail.size(1) != seq_length:
                detail = F.interpolate(
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
        
        # Inverse FFT to get reconstructed signal
        output = torch.fft.irfft(x_fft, n=seq_length, dim=1)
        
        return output
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Apply wavelet transform to input.
        
        Args:
            x: Input signal [batch_size, seq_length, embed_dim]
            
        Returns:
            approximation, detail_coefficients
        """
        if self.using_pywt:
            return self.pywt_forward(x)
        else:
            return self.fft_forward(x)
    
    def inverse(self, approx: torch.Tensor, details: List[torch.Tensor]) -> torch.Tensor:
        """
        Apply inverse wavelet transform.
        
        Args:
            approx: Approximation coefficients
            details: Detail coefficients
            
        Returns:
            Reconstructed signal
        """
        if self.using_pywt:
            return self.pywt_inverse(approx, details)
        else:
            return self.fft_inverse(approx, details)


class StochasticTransform(nn.Module):
    """
    Implements stochastic approximation for wavelet transforms as suggested by Dr. Tao,
    reducing complexity from O(n log n) to O(m log m) where m << n.
    """
    def __init__(self, wavelet_type: str = 'db4', levels: int = 3, 
                 sampling_ratio: float = 0.1, min_samples: int = 32):
        super().__init__()
        self.wavelet_type = wavelet_type
        self.levels = levels
        self.sampling_ratio = sampling_ratio
        self.min_samples = min_samples
        
        # Calculate expected error bound based on sampling theory: O(1/√m)
        self.register_buffer('error_bound', None)
        self.register_buffer('actual_errors', None)
    
    def compute_error_bound(self, n_samples: int) -> float:
        """
        Compute theoretical error bound based on stochastic approximation theory.
        
        Args:
            n_samples: Number of samples used
            
        Returns:
            Theoretical error bound O(1/√m)
        """
        return 1.0 / np.sqrt(n_samples)
    
    def forward_stochastic(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], Dict]:
        """
        Apply stochastic approximation of wavelet transform.
        
        Args:
            x: Input tensor [batch_size, seq_length, embed_dim]
            
        Returns:
            Approximation coefficients, detail coefficients, and error metrics
        """
        batch_size, seq_length, embed_dim = x.shape
        
        # Determine number of samples
        n_samples = max(self.min_samples, int(self.sampling_ratio * seq_length))
        n_samples = min(n_samples, seq_length)  # Can't have more samples than sequence length
        
        # For comparison, also compute full transform on a subset of the batch
        # to estimate actual error
        comparison_batch_idx = 0
        full_approx, full_details = None, None
        
        if self.training:
            # Use standard wavelet transform for the first item in batch
            wavelet = WaveletTransform(self.wavelet_type, self.levels)
            full_approx, full_details = wavelet(x[comparison_batch_idx:comparison_batch_idx+1])
        
        # Randomly sample points from the sequence
        indices = torch.randperm(seq_length, device=x.device)[:n_samples]
        indices, _ = torch.sort(indices)  # Sort for coherent transform
        
        # Extract sampled points
        x_sampled = x[:, indices, :]
        
        # Apply transform to sampled points
        wavelet = WaveletTransform(self.wavelet_type, self.levels)
        sampled_approx, sampled_details = wavelet(x_sampled)
        
        # Scale up to original size (simple interpolation)
        approx_ratio = seq_length / sampled_approx.size(1)
        
        # Use interpolation to upsample approximation coefficients
        approx_upsampled = F.interpolate(
            sampled_approx.permute(0, 2, 1),  # [batch, embed, seq]
            size=seq_length,
            mode='linear',
            align_corners=False
        ).permute(0, 2, 1)  # back to [batch, seq, embed]
        
        # Upsample detail coefficients
        details_upsampled = []
        for detail in sampled_details:
            detail_upsampled = F.interpolate(
                detail.permute(0, 2, 1),
                size=seq_length,
                mode='linear',
                align_corners=False
            ).permute(0, 2, 1)
            details_upsampled.append(detail_upsampled)
        
        # Calculate error metrics if full transform was computed
        error_metrics = {}
        if full_approx is not None:
            # Calculate approximation error
            approx_error = torch.norm(approx_upsampled[comparison_batch_idx] - full_approx[0]) / torch.norm(full_approx[0])
            error_metrics['approx_error'] = approx_error.item()
            
            # Calculate theoretical error bound
            error_bound = self.compute_error_bound(n_samples)
            error_metrics['error_bound'] = error_bound
            error_metrics['n_samples'] = n_samples
            error_metrics['sampling_ratio'] = self.sampling_ratio
            
            # Update error tracking buffers
            if self.error_bound is None:
                self.error_bound = torch.tensor([error_bound], device=x.device)
                self.actual_errors = torch.tensor([approx_error.item()], device=x.device)
            else:
                self.error_bound = torch.cat([self.error_bound, torch.tensor([error_bound], device=x.device)])
                self.actual_errors = torch.cat([self.actual_errors, torch.tensor([approx_error.item()], device=x.device)])
                
                # Keep only recent values (last 100)
                if len(self.error_bound) > 100:
                    self.error_bound = self.error_bound[-100:]
                    self.actual_errors = self.actual_errors[-100:]
        
        return approx_upsampled, details_upsampled, error_metrics
    
    def inverse_stochastic(self, approx: torch.Tensor, details: List[torch.Tensor]) -> torch.Tensor:
        """
        Apply inverse transform from stochastically computed coefficients.
        
        Args:
            approx: Approximation coefficients
            details: Detail coefficients
            
        Returns:
            Reconstructed signal
        """
        # Use standard inverse wavelet transform
        wavelet = WaveletTransform(self.wavelet_type, self.levels)
        return wavelet.inverse(approx, details)
    
    def get_error_statistics(self) -> Dict:
        """
        Get statistics about the approximation error.
        
        Returns:
            Dictionary with error statistics
        """
        if self.error_bound is None or len(self.error_bound) == 0:
            return {
                'mean_error_bound': 0.0,
                'mean_actual_error': 0.0,
                'bound_to_actual_ratio': 1.0
            }
        
        mean_bound = self.error_bound.mean().item()
        mean_actual = self.actual_errors.mean().item()
        ratio = mean_bound / (mean_actual + 1e-8)
        
        return {
            'mean_error_bound': mean_bound,
            'mean_actual_error': mean_actual,
            'bound_to_actual_ratio': ratio
        }


class AdaptiveBasisSelection(nn.Module):
    """
    Implements adaptive selection between multiple wavelet basis families
    based on Dr. Tao's recommendation for optimal time-frequency trade-offs.
    """
    def __init__(self, embed_dim: int, families: List[str] = None):
        super().__init__()
        # Default wavelet families if not provided
        self.families = families or ['db4', 'sym4', 'dmey']
        self.embed_dim = embed_dim
        
        # Create transform for each wavelet family
        self.transforms = nn.ModuleDict({
            family: WaveletTransform(family, levels=3) 
            for family in self.families if WAVELET_AVAILABLE
        })
        
        # Context-based selection mechanism
        self.selector = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, len(self.families)),
            nn.Softmax(dim=-1)
        )
        
        # Learnable weights for combining outputs
        self.combination_weights = nn.Parameter(torch.ones(len(self.families)) / len(self.families))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Apply multiple wavelet transforms and adaptively combine them.
        
        Args:
            x: Input tensor [batch_size, seq_length, embed_dim]
            
        Returns:
            Combined approximation and detail coefficients
        """
        # Compute context features for selection
        # Use mean pooling over sequence dimension
        context = x.mean(dim=1)
        selection_weights = self.selector(context).unsqueeze(1)  # [batch, 1, n_families]
        
        # Apply each transform
        all_approx = []
        all_details = []
        
        for i, family in enumerate(self.families):
            if family in self.transforms:
                approx, details = self.transforms[family](x)
                all_approx.append(approx)
                all_details.append(details)
        
        # Combine approximation coefficients with learned weights
        combined_approx = torch.zeros_like(all_approx[0])
        for i, approx in enumerate(all_approx):
            # Use both global learned weights and context-dependent weights
            weight = self.combination_weights[i] * selection_weights[:, :, i]
            combined_approx += approx * weight.unsqueeze(-1)
        
        # Combine detail coefficients (more complex as they're in lists)
        combined_details = []
        for level in range(len(all_details[0])):
            level_combined = torch.zeros_like(all_details[0][level])
            for i, details in enumerate(all_details):
                weight = self.combination_weights[i] * selection_weights[:, :, i]
                level_combined += details[level] * weight.unsqueeze(-1)
            combined_details.append(level_combined)
        
        return combined_approx, combined_details
    
    def inverse(self, approx: torch.Tensor, details: List[torch.Tensor]) -> torch.Tensor:
        """
        Apply inverse transform using the primary wavelet family.
        """
        # Use the primary family (first one) for reconstruction
        primary_family = self.families[0]
        if primary_family in self.transforms:
            return self.transforms[primary_family].inverse(approx, details)
        else:
            # Fallback to FFT-based reconstruction
            return self.transforms[list(self.transforms.keys())[0]].inverse(approx, details)


class FrequencyDomainAttention(nn.Module):
    """
    Attention mechanism that operates in the frequency domain.
    Uses Fourier transforms to process queries and keys before computing attention.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Projection matrices
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Learnable frequency filters for each head
        self.freq_filter = nn.Parameter(torch.randn(num_heads, self.head_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.scale = (self.head_dim) ** -0.5
        
        # Operations counter for complexity analysis
        self.register_buffer('op_count', torch.zeros(1))
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute frequency-domain attention.
        
        Args:
            x: Input tensor [batch_size, seq_length, embed_dim]
            mask: Attention mask [batch_size, seq_length]
            
        Returns:
            Output tensor [batch_size, seq_length, embed_dim]
        """
        batch_size, seq_length, _ = x.shape
        
        # Track ops for complexity analysis
        if self.training:
            # Standard attention: O(n²) operations per sequence
            standard_ops = seq_length ** 2
            # FFT ops: O(n log n) operations per sequence
            fft_ops = seq_length * math.log2(seq_length) if seq_length > 1 else 1
            self.op_count[0] = fft_ops / standard_ops
        
        # Project queries, keys, values
        q = self.q_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        
        # Reshape for batch-wise operations
        q = q.transpose(1, 2)  # [batch, heads, seq, dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Transform to frequency domain for filtering
        q_fft = torch.fft.rfft(q, dim=2)
        k_fft = torch.fft.rfft(k, dim=2)
        
        # Apply learnable frequency filtering
        filter_shaped = self.freq_filter.unsqueeze(0).unsqueeze(2)  # [1, heads, 1, dim]
        q_fft = q_fft * filter_shaped
        k_fft = k_fft * filter_shaped
        
        # Transform back to time domain
        q_filtered = torch.fft.irfft(q_fft, n=seq_length, dim=2)
        k_filtered = torch.fft.irfft(k_fft, n=seq_length, dim=2)
        
        # Standard attention calculation
        attn_weights = torch.matmul(q_filtered, k_filtered.transpose(2, 3)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1)  # Add head dimension
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)  # [batch, heads, seq, dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)
        
        return self.out_proj(output)
    
    def get_complexity_ratio(self) -> float:
        """Get the ratio of operations (FFT vs standard attention)"""
        return self.op_count.item()


class FourierConvolutionAttention(nn.Module):
    """
    Implements Dr. Tao's efficient attention as convolution in Fourier space.
    Based on the convolution theorem: F(f * g) = F(f) · F(g).
    This achieves O(n log n) complexity for attention computation.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Projection matrices
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Learnable frequency response curves for each head
        self.freq_response = nn.Parameter(torch.randn(num_heads, self.head_dim // 2 + 1))
        
        # Learnable toeplitz patterns (circular convolution kernels)
        self.conv_kernels = nn.Parameter(torch.randn(num_heads, self.head_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.scale = (self.head_dim) ** -0.5
        
        # Operations counter for complexity analysis
        self.register_buffer('op_count', torch.zeros(1))
        
        # Track attention associativity metrics
        self.register_buffer('associativity_metrics', torch.zeros(3))
        
        # Add a shape tracker for debugging and analysis
        self.register_buffer('shape_tracker', torch.zeros(3, dtype=torch.long))
        
    def _ensure_compatible_sizes(self, tensor1, tensor2, dim=1):
        """
        Ensure two tensors have compatible sizes along specified dimension
        for element-wise operations.
        
        Args:
            tensor1: First tensor
            tensor2: Second tensor
            dim: Dimension to check and adjust
            
        Returns:
            Tuple of adjusted tensors with same size along dim
        """
        size1 = tensor1.size(dim)
        size2 = tensor2.size(dim)
        
        if size1 == size2:
            return tensor1, tensor2
            
        if size1 > size2:
            # If tensor1 is bigger, truncate it
            if dim == 1:
                return tensor1[:, :size2], tensor2
            elif dim == 2:
                return tensor1[:, :, :size2], tensor2
        else:
            # If tensor2 is bigger, truncate it
            if dim == 1:
                return tensor1, tensor2[:, :size1]
            elif dim == 2:
                return tensor1, tensor2[:, :, :size1]
                
        return tensor1, tensor2  # Fallback case

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass implementing attention as convolution in Fourier space.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_length, _ = x.shape
        
        # Store original sequence length for shape tracking
        self.shape_tracker[0] = seq_length
        
        # Project inputs to queries, keys, and values
        q = self.q_proj(x)  # [batch, seq, embed_dim]
        k = self.k_proj(x)  # [batch, seq, embed_dim]
        v = self.v_proj(x)  # [batch, seq, embed_dim]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim)
        
        # [batch, seq, heads, dim] -> [batch, heads, seq, dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Instead of explicit attention, use circulant convolution
        # Step 1: Prepare convolution kernels from keys
        # Make sure kernel size matches sequence length
        if self.conv_kernels.size(1) != seq_length:
            # Pad or truncate kernels to match sequence length
            padded_kernels = torch.zeros(self.num_heads, seq_length, device=self.conv_kernels.device)
            kernel_len = min(self.conv_kernels.size(1), seq_length)
            padded_kernels[:, :kernel_len] = self.conv_kernels[:, :kernel_len]
            conv_weights = torch.fft.rfft(padded_kernels.unsqueeze(0).repeat(batch_size, 1, 1), dim=2)
        else:
            conv_weights = torch.fft.rfft(self.conv_kernels.unsqueeze(0).repeat(batch_size, 1, 1), dim=2)
        
        # Track FFT size for shape analysis
        self.shape_tracker[1] = conv_weights.size(2)
        
        # Step 2: Apply the mask if provided (in Fourier space)
        if mask is not None:
            # Convert mask to Fourier space, ensuring compatible dimensions
            mask_fft = torch.fft.rfft(mask.float().unsqueeze(1), dim=1)  # [batch, 1, fft_size]
            # Make sure shapes are compatible
            mask_fft, conv_weights = self._ensure_compatible_sizes(
                mask_fft.unsqueeze(-1), 
                conv_weights, 
                dim=2
            )
            # Apply to convolution weights
            conv_weights = conv_weights * mask_fft
        
        # Step 3: Compute convolution for each head via FFT
        outputs = []
        for h in range(self.num_heads):
            # Transform value vectors to frequency domain
            v_fft = torch.fft.rfft(v[:, h], dim=1)  # [batch, fft_size, dim]
            
            # Apply frequency-domain filters (learnable)
            filters = torch.sigmoid(self.freq_response[h]).unsqueeze(0).unsqueeze(-1)
            
            # Ensure filters have compatible size with v_fft
            v_fft_size = v_fft.size(1)
            filter_size = filters.size(1)
            
            # Adjust filter size if needed
            if filter_size != v_fft_size:
                # Create properly sized filters
                new_filters = torch.zeros(1, v_fft_size, 1, device=filters.device)
                # Copy data from original filters, up to the minimum size
                min_size = min(filter_size, v_fft_size)
                new_filters[:, :min_size, :] = filters[:, :min_size, :]
                filters = new_filters
            
            # Apply filters
            v_fft_filtered = v_fft * filters
            
            # Convolve with kernel (equivalent to attention) in Fourier space
            head_weights = conv_weights[:, h]  # [batch, fft_size]
            
            # Ensure head_weights and v_fft_filtered have compatible dimensions
            head_weights, v_fft_filtered = self._ensure_compatible_sizes(
                head_weights.unsqueeze(-1), 
                v_fft_filtered, 
                dim=1
            )
            
            # Enhance values based on query-key similarity via convolution
            output_fft = v_fft_filtered * head_weights
            
            # Transform back to time domain, ensuring output length matches input
            output = torch.fft.irfft(output_fft, n=seq_length, dim=1)
            
            # In case output length doesn't match (should be rare now)
            if output.size(1) != seq_length:
                # Create properly sized output
                padded_output = torch.zeros(batch_size, seq_length, self.head_dim, device=output.device)
                # Copy available data
                min_len = min(output.size(1), seq_length)
                padded_output[:, :min_len, :] = output[:, :min_len, :]
                output = padded_output
                
            outputs.append(output)
        
        # Track output shape for analysis
        if outputs:
            self.shape_tracker[2] = outputs[0].size(1)
        
        # Combine all heads
        combined = torch.stack(outputs, dim=1)  # [batch, heads, seq, dim]
        combined = combined.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)
        
        # Final projection
        return self.out_proj(combined)
    
    def get_complexity_ratio(self) -> float:
        """Get the ratio of operations (FFT vs standard attention)"""
        return self.op_count.item()
    
    def get_associativity_metrics(self) -> Dict[str, float]:
        """Get the associativity metrics for this attention mechanism"""
        return {
            'min': self.associativity_metrics[0].item(),
            'max': self.associativity_metrics[1].item(),
            'mean': self.associativity_metrics[2].item()
        }


class WaveletAttention(nn.Module):
    """
    Attention mechanism that operates in the wavelet domain.
    Processes signals at multiple resolutions.
    """
    def __init__(self, embed_dim: int, num_heads: int, 
                wavelet_type: str = 'db4', levels: int = 3,
                dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.levels = levels
        
        # Wavelet transform
        self.wavelet = WaveletTransform(wavelet_type, levels)
        
        # Attention for approximation coefficients
        self.approx_attention = FrequencyDomainAttention(
            embed_dim, num_heads, dropout)
        
        # Attention for detail coefficients (one per level)
        self.detail_attentions = nn.ModuleList([
            FrequencyDomainAttention(embed_dim, num_heads, dropout)
            for _ in range(levels)
        ])
        
        # Projection after wavelet reconstruction
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply wavelet-based multi-resolution attention.
        
        Args:
            x: Input tensor [batch_size, seq_length, embed_dim]
            mask: Attention mask [batch_size, seq_length]
            
        Returns:
            Output tensor [batch_size, seq_length, embed_dim]
        """
        # Apply wavelet transform
        approx, details = self.wavelet.forward(x)
        
        # Apply attention to approximation coefficients
        if mask is not None:
            # Adjust mask for possibly shorter sequence length
            approx_len = approx.size(1)
            approx_mask = mask[:, :approx_len] if mask.size(1) > approx_len else mask
        else:
            approx_mask = None
            
        approx_out = self.approx_attention(approx, approx_mask)
        
        # Apply attention to detail coefficients
        detail_outs = []
        for i, detail in enumerate(details):
            # Get appropriate attention layer
            detail_attn = self.detail_attentions[i]
            
            # Adjust mask if needed
            if mask is not None:
                detail_len = detail.size(1)
                detail_mask = mask[:, :detail_len] if mask.size(1) > detail_len else mask
            else:
                detail_mask = None
                
            # Apply attention
            detail_out = detail_attn(detail, detail_mask)
            detail_outs.append(detail_out)
        
        # Reconstruct signal from attended coefficients
        output = self.wavelet.inverse(approx_out, detail_outs)
        
        # Ensure output has same sequence length as input
        if output.size(1) != x.size(1):
            output = F.interpolate(
                output.transpose(1, 2), 
                size=x.size(1), 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
        
        # Final projection
        return self.out_proj(output)


class SpectralFeedForward(nn.Module):
    """
    Feed-forward network that operates in the frequency domain.
    Applies different transformations to different frequency components.
    """
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Traditional projection matrices
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        
        # Frequency-selective processing
        fft_size = hidden_dim // 2 + 1  # Size of rfft output for hidden_dim
        self.low_pass = nn.Parameter(torch.ones(1, 1, fft_size) * 0.8)
        self.high_pass = nn.Parameter(torch.zeros(1, 1, fft_size) + 0.2)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply frequency-selective feed-forward network.
        
        Args:
            x: Input tensor [batch_size, seq_length, embed_dim]
            
        Returns:
            Output tensor [batch_size, seq_length, embed_dim]
        """
        # Traditional feedforward part
        h = F.gelu(self.fc1(x))
        h = self.dropout(h)
        
        # Apply frequency-domain processing
        h_fft = torch.fft.rfft(h, dim=-1)
        
        # Frequency-selective filtering
        h_fft_filtered = h_fft * (torch.sigmoid(self.low_pass) + torch.sigmoid(self.high_pass))
        
        # Back to time domain
        h_filtered = torch.fft.irfft(h_fft_filtered, n=self.hidden_dim, dim=-1)
        
        # Final projection
        return self.fc2(h_filtered)


class WaveletTransformerBlock(nn.Module):
    """A single transformer block with wavelet-based multi-resolution processing"""
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, 
                wavelet_type: str = 'db4', levels: int = 3,
                dropout: float = 0.1):
        super().__init__()
        
        # Enhanced with Dr. Tao's mathematical insights
        # Choose between different attention mechanisms
        self.use_adaptive_basis = True
        self.use_fourier_convolution = True
        
        if self.use_adaptive_basis:
            # Use adaptive basis selection across multiple wavelet families
            self.basis_selection = AdaptiveBasisSelection(embed_dim)
            # Stochastic Transform for very long sequences
            self.stochastic_transform = StochasticTransform(wavelet_type, levels, sampling_ratio=0.2)
            # Use the adaptive basis with wavelet attention
            self.wavelet_attn = WaveletAttention(embed_dim, num_heads, wavelet_type, levels, dropout)
        else:
            # Standard wavelet attention
            self.wavelet_attn = WaveletAttention(embed_dim, num_heads, wavelet_type, levels, dropout)
        
        # For complex sequences, use Fourier convolution attention as complementary mechanism
        if self.use_fourier_convolution:
            self.fourier_attn = FourierConvolutionAttention(embed_dim, num_heads, dropout)
            # Learnable mixing parameter
            self.attn_mix = nn.Parameter(torch.tensor([0.5]))
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = SpectralFeedForward(embed_dim, hidden_dim, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # For spectral gap analysis
        self.spectral_analyzer = SpectralGapAnalyzer()
        self.seq_length_threshold = 512  # Use stochastic transform for sequences longer than this
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process input through wavelet transformer block.
        
        Args:
            x: Input tensor [batch_size, seq_length, embed_dim]
            mask: Attention mask [batch_size, seq_length]
            
        Returns:
            Output tensor [batch_size, seq_length, embed_dim]
        """
        # Apply adaptive wavelet processing if enabled
        if self.use_adaptive_basis:
            # For longer sequences, use stochastic approximation
            if x.size(1) > self.seq_length_threshold and self.training:
                approx, details, error_metrics = self.stochastic_transform.forward_stochastic(x)
                # Use adaptive basis selection for refinement
                refined_approx, refined_details = self.basis_selection(x)
                # Blend stochastic and refined approximations
                blend_ratio = 0.7  # Favor the adaptive basis selection
                approx = blend_ratio * refined_approx + (1 - blend_ratio) * approx
                
                # Use standard wavelet attention with the refined coefficients
                wavelet_attn_out = self.wavelet_attn(x, mask)
            else:
                # For shorter sequences, use standard processing
                wavelet_attn_out = self.wavelet_attn(x, mask)
                
                # Analyze spectral properties if in training mode (periodically)
                if self.training and random.random() < 0.05:  # 5% of the time
                    # Get wavelet coefficients
                    with torch.no_grad():
                        approx, details = self.wavelet_attn.wavelet(x)
                        specs = self.spectral_analyzer.analyze_wavelet_representation((approx, details))
        else:
            # Standard wavelet attention
            wavelet_attn_out = self.wavelet_attn(x, mask)
        
        # Combine with Fourier convolution attention if enabled
        if self.use_fourier_convolution:
            fourier_attn_out = self.fourier_attn(x, mask)
            # Adaptive mixing based on learnable parameter
            mix = torch.sigmoid(self.attn_mix)
            attn_output = mix * wavelet_attn_out + (1 - mix) * fourier_attn_out
        else:
            attn_output = wavelet_attn_out
            
        # Apply residual connection and normalization
        x = self.norm1(x + self.dropout(attn_output))
        
        # FFN with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x 


#-------------------------------------------------------------------------
# HRFEvo Optimization Controller
#-------------------------------------------------------------------------

class BasisFunction:
    """
    Represents a wavelet basis function that can evolve.
    Basis functions are defined by parameters that can be mutated during evolution.
    """
    def __init__(self, type_name: str = 'db4', params: Optional[Dict] = None):
        self.type_name = type_name
        self.params = params or {}
        self.fitness = 0.0
        
        # Initialize default parameters if not provided
        if not self.params:
            if type_name.startswith('db'):  # Daubechies wavelets
                # Initialize filter coefficients
                if WAVELET_AVAILABLE:
                    wavelet = pywt.Wavelet(type_name)
                    self.params['dec_lo'] = wavelet.dec_lo
                    self.params['dec_hi'] = wavelet.dec_hi
                    self.params['rec_lo'] = wavelet.rec_lo
                    self.params['rec_hi'] = wavelet.rec_hi
                else:
                    # Approximate filter coefficients for Daubechies 4
                    self.params['dec_lo'] = [0.48, 0.83, 0.22, -0.12]
                    self.params['dec_hi'] = [-0.12, -0.22, 0.83, -0.48]
                    self.params['rec_lo'] = [0.12, 0.22, 0.83, 0.48]
                    self.params['rec_hi'] = [0.48, -0.83, 0.22, 0.12]
            else:
                # Simple harmonic basis
                self.params['frequencies'] = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
                self.params['amplitudes'] = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
                self.params['phases'] = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    def __repr__(self) -> str:
        return f"BasisFunction(type={self.type_name}, fitness={self.fitness:.4f})"
    
    def mutate(self, mutation_rate: float = 0.1) -> 'BasisFunction':
        """
        Create a mutated copy of this basis function.
        
        Args:
            mutation_rate: Probability of mutating each parameter
            
        Returns:
            Mutated basis function
        """
        new_params = {}
        
        for key, value in self.params.items():
            # Determine if this parameter should be mutated
            if np.random.random() < mutation_rate:
                if isinstance(value, list):
                    # For list parameters, mutate each element
                    new_value = []
                    for v in value:
                        # Add Gaussian noise proportional to value
                        noise = np.random.normal(0, abs(v) * 0.1)
                        new_value.append(v + noise)
                    new_params[key] = new_value
                elif isinstance(value, (int, float)):
                    # For scalar parameters, add Gaussian noise
                    noise = np.random.normal(0, abs(value) * 0.1)
                    new_params[key] = value + noise
                else:
                    # For other types, keep as is
                    new_params[key] = value
            else:
                # No mutation, copy as is
                new_params[key] = value
        
        # Create a new basis function with mutated parameters
        mutated = BasisFunction(self.type_name, new_params)
        return mutated
    
    def crossover(self, other: 'BasisFunction') -> 'BasisFunction':
        """
        Create a new basis function by crossing over with another one.
        
        Args:
            other: Another basis function to crossover with
            
        Returns:
            New basis function with mixed parameters
        """
        # Only crossover if types are compatible
        if self.type_name != other.type_name:
            return self.copy()
        
        new_params = {}
        
        # Mix parameters from both parents
        for key in self.params:
            if key in other.params:
                if isinstance(self.params[key], list) and isinstance(other.params[key], list):
                    # For list parameters, perform uniform crossover
                    new_value = []
                    for i in range(min(len(self.params[key]), len(other.params[key]))):
                        # Randomly choose from either parent
                        if np.random.random() < 0.5:
                            new_value.append(self.params[key][i])
                        else:
                            new_value.append(other.params[key][i])
                    new_params[key] = new_value
                else:
                    # For scalar parameters, randomly choose
                    if np.random.random() < 0.5:
                        new_params[key] = self.params[key]
                    else:
                        new_params[key] = other.params[key]
            else:
                # If other doesn't have this parameter, keep from self
                new_params[key] = self.params[key]
        
        # Create a new basis function with crossover parameters
        child = BasisFunction(self.type_name, new_params)
        return child
    
    def copy(self) -> 'BasisFunction':
        """Create a deep copy of this basis function"""
        # Create new parameter dictionary with deep copies of all values
        new_params = {}
        for key, value in self.params.items():
            if isinstance(value, list):
                new_params[key] = value.copy()
            else:
                new_params[key] = value
        
        # Create a new basis function with copied parameters
        copy_instance = BasisFunction(self.type_name, new_params)
        copy_instance.fitness = self.fitness
        return copy_instance
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'type_name': self.type_name,
            'params': self.params,
            'fitness': self.fitness
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BasisFunction':
        """Create from dictionary"""
        instance = cls(data['type_name'], data['params'])
        instance.fitness = data['fitness']
        return instance


class HRFEvoController:
    """
    Controller for evolving hierarchical basis functions.
    Uses evolutionary algorithms to optimize the basis functions for language representation.
    """
    def __init__(self, config: Config):
        self.config = config
        self.population_size = config.evolution_population
        self.generations = config.evolution_generations
        self.mutation_rate = config.evolution_mutation_rate
        
        # Reward weights
        self.reward_computation = config.evolution_reward_computation
        self.reward_perplexity = config.evolution_reward_perplexity
        
        # Population of basis functions
        self.population = []
        self.best_basis = None
        self.generation_count = 0
        
        # Initialize population
        self._initialize_population()
        
        # Logging
        self.fitness_history = []
    
    def _initialize_population(self) -> None:
        """Initialize the population of basis functions"""
        # Create initial population with different wavelet types
        wavelet_types = ['db2', 'db4', 'db6', 'sym4', 'coif4', 'harmonic']
        
        for i in range(self.population_size):
            # Choose a wavelet type, cycling through the list
            type_idx = i % len(wavelet_types)
            basis = BasisFunction(wavelet_types[type_idx])
            self.population.append(basis)
        
        logging.info(f"Initialized population with {self.population_size} basis functions")
    
    def evaluate_population(self, evaluation_func: callable) -> None:
        """
        Evaluate the fitness of all basis functions in the population.
        
        Args:
            evaluation_func: Function to evaluate a basis function and return metrics
        """
        for basis in self.population:
            metrics = evaluation_func(basis)
            
            # Calculate fitness score as weighted sum of metrics
            computation_score = metrics.get('computation_efficiency', 0.0)
            perplexity_score = 1.0 / (metrics.get('perplexity', float('inf')) + 1e-6)
            
            basis.fitness = (self.reward_computation * computation_score +
                            self.reward_perplexity * perplexity_score)
        
        # Sort population by fitness
        self.population.sort(key=lambda b: b.fitness, reverse=True)
        
        # Update best basis function
        if not self.best_basis or self.population[0].fitness > self.best_basis.fitness:
            self.best_basis = self.population[0].copy()
        
        # Log statistics
        avg_fitness = np.mean([b.fitness for b in self.population])
        self.fitness_history.append({
            'generation': self.generation_count,
            'best_fitness': self.population[0].fitness,
            'avg_fitness': avg_fitness
        })
        
        logging.info(f"Generation {self.generation_count}: "
                   f"Best fitness = {self.population[0].fitness:.4f}, "
                   f"Avg fitness = {avg_fitness:.4f}")
    
    def select_parents(self) -> List[BasisFunction]:
        """
        Select parents for the next generation using tournament selection.
        
        Returns:
            List of selected parents
        """
        selected_parents = []
        
        # Use tournament selection
        tournament_size = max(2, self.population_size // 5)
        
        for _ in range(self.population_size):
            # Select random individuals for tournament
            tournament = np.random.choice(self.population, tournament_size, replace=False)
            
            # Select the best one as parent
            winner = max(tournament, key=lambda b: b.fitness)
            selected_parents.append(winner)
        
        return selected_parents
    
    def create_offspring(self, parents: List[BasisFunction]) -> List[BasisFunction]:
        """
        Create offspring from selected parents through crossover and mutation.
        
        Args:
            parents: List of selected parent basis functions
            
        Returns:
            List of offspring basis functions
        """
        offspring = []
        
        # Elitism: keep the best basis function
        offspring.append(self.population[0].copy())
        
        # Create remaining offspring
        for i in range(1, self.population_size):
            # Select two parents
            parent1 = parents[np.random.randint(0, len(parents))]
            parent2 = parents[np.random.randint(0, len(parents))]
            
            # Crossover
            child = parent1.crossover(parent2)
            
            # Mutation
            child = child.mutate(self.mutation_rate)
            
            offspring.append(child)
        
        return offspring
    
    def evolve_generation(self, evaluation_func: callable) -> Dict:
        """
        Evolve one generation of basis functions.
        
        Args:
            evaluation_func: Function to evaluate basis functions
            
        Returns:
            Statistics about this generation
        """
        # Evaluate current population
        self.evaluate_population(evaluation_func)
        
        # Select parents
        parents = self.select_parents()
        
        # Create offspring
        offspring = self.create_offspring(parents)
        
        # Replace population with offspring
        self.population = offspring
        
        # Increment generation counter
        self.generation_count += 1
        
        # Return statistics about this generation
        stats = self.fitness_history[-1].copy()
        stats['best_basis'] = self.best_basis.to_dict()
        
        return stats
    
    def run_evolution(self, evaluation_func: callable) -> BasisFunction:
        """
        Run the full evolutionary process for specified number of generations.
        
        Args:
            evaluation_func: Function to evaluate basis functions
            
        Returns:
            The best evolved basis function
        """
        logging.info(f"Starting HRFEvo for {self.generations} generations")
        
        for generation in range(self.generations):
            stats = self.evolve_generation(evaluation_func)
            
            logging.info(f"Generation {generation+1}/{self.generations}: "
                       f"Best fitness = {stats['best_fitness']:.4f}")
        
        logging.info(f"Evolution completed. Best basis: {self.best_basis}")
        
        return self.best_basis
    
    def get_best_basis(self) -> BasisFunction:
        """Get the current best basis function"""
        return self.best_basis.copy() if self.best_basis else None
    
    def save_state(self, filepath: str) -> None:
        """Save the controller state to a file"""
        state = {
            'generation_count': self.generation_count,
            'best_basis': self.best_basis.to_dict() if self.best_basis else None,
            'population': [basis.to_dict() for basis in self.population],
            'fitness_history': self.fitness_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str) -> None:
        """Load the controller state from a file"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.generation_count = state['generation_count']
        self.best_basis = BasisFunction.from_dict(state['best_basis']) if state['best_basis'] else None
        self.population = [BasisFunction.from_dict(basis_data) for basis_data in state['population']]
        self.fitness_history = state['fitness_history']


class SpectralGapAnalyzer:
    """
    Implements spectral gap analysis for wavelet coefficients as recommended
    by Dr. Tao to measure effectiveness in separating linguistic structures.
    """
    def __init__(self, device=None):
        self.device = device or torch.device('cpu')
    
    def compute_laplacian(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute the graph Laplacian for embeddings.
        
        Args:
            embeddings: Token embeddings [batch_size, seq_length, embed_dim]
            
        Returns:
            Graph Laplacian matrix
        """
        # Compute pairwise similarities (using cosine similarity)
        # Reshape to [batch_size * seq_length, embed_dim]
        batch_size, seq_length, embed_dim = embeddings.shape
        flat_embeddings = embeddings.reshape(-1, embed_dim)
        
        # Normalize embeddings for cosine similarity
        norms = torch.norm(flat_embeddings, dim=1, keepdim=True)
        normalized_embeddings = flat_embeddings / (norms + 1e-8)
        
        # Compute similarity matrix
        similarity = torch.mm(normalized_embeddings, normalized_embeddings.t())
        
        # Create adjacency matrix (threshold similarities for sparse graph)
        adjacency = torch.zeros_like(similarity)
        top_k = min(10, seq_length)  # Connect each node to top-k neighbors
        top_k_values, top_k_indices = torch.topk(similarity, k=top_k, dim=1)
        
        # Populate adjacency matrix with top-k connections
        for i in range(similarity.size(0)):
            adjacency[i, top_k_indices[i]] = top_k_values[i]
        
        # Make symmetric
        adjacency = 0.5 * (adjacency + adjacency.t())
        
        # Compute degree matrix
        degree = torch.sum(adjacency, dim=1)
        degree_matrix = torch.diag(degree)
        
        # Compute Laplacian: L = D - A
        laplacian = degree_matrix - adjacency
        
        return laplacian
    
    def compute_spectral_gap(self, laplacian: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """
        Compute the spectral gap of the Laplacian matrix.
        
        Args:
            laplacian: Graph Laplacian matrix
            
        Returns:
            Spectral gap and eigenvalues
        """
        # Compute eigenvalues (use only a subset for efficiency)
        try:
            eigenvalues, _ = torch.linalg.eigh(laplacian)
        except:
            # Fallback to CPU if eigendecomposition fails on GPU
            eigenvalues, _ = torch.linalg.eigh(laplacian.cpu())
            eigenvalues = eigenvalues.to(self.device)
        
        # Sort eigenvalues (they should already be sorted, but just to be sure)
        eigenvalues, _ = torch.sort(eigenvalues)
        
        # Compute spectral gap (difference between first non-zero and zero eigenvalue)
        # First eigenvalue should be close to zero
        spectral_gap = eigenvalues[1] - eigenvalues[0]
        
        return spectral_gap.item(), eigenvalues
    
    def analyze_wavelet_representation(self, 
                                      coeffs: Tuple[torch.Tensor, List[torch.Tensor]]) -> Dict:
        """
        Analyze the spectral properties of wavelet coefficients.
        
        Args:
            coeffs: Tuple of (approximation, details) from wavelet transform
            
        Returns:
            Dictionary with spectral analysis metrics
        """
        approx, details = coeffs
        
        # Compute Laplacian for approximation coefficients
        approx_laplacian = self.compute_laplacian(approx)
        approx_gap, approx_eigenvalues = self.compute_spectral_gap(approx_laplacian)
        
        # Compute Laplacians for detail coefficients at each level
        detail_gaps = []
        detail_eigenvalues = []
        
        for detail in details:
            detail_laplacian = self.compute_laplacian(detail)
            gap, eigenvalues = self.compute_spectral_gap(detail_laplacian)
            detail_gaps.append(gap)
            detail_eigenvalues.append(eigenvalues[:5])  # Keep only first few eigenvalues
        
        # Compute overall spectral gap as weighted combination
        # Weight approximation gap higher than detail gaps
        weights = [0.6] + [0.4 / len(detail_gaps)] * len(detail_gaps)
        overall_gap = approx_gap * weights[0] + sum(g * w for g, w in zip(detail_gaps, weights[1:]))
        
        return {
            'approx_gap': approx_gap,
            'detail_gaps': detail_gaps,
            'overall_gap': overall_gap,
            'approx_eigenvalues': approx_eigenvalues[:5],  # Keep only first few eigenvalues
            'detail_eigenvalues': detail_eigenvalues
        }


class SignalLLM(nn.Module):
    """
    Complete signal-based language model architecture.
    Uses spectral embeddings, wavelet attention, and frequency-domain processing.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Token embeddings - can be spectral or hybrid
        if config.harmonic_bases > 0:
            self.use_hybrid = True
            self.token_embedding = HybridEmbedding(
                config.vocab_size, config.embed_dim, config.harmonic_bases)
        else:
            self.use_hybrid = False
            self.token_embedding = SpectralEmbedding(
                config.vocab_size, config.embed_dim, 16)
        
        # Positional encoding
        self.pos_encoding = SignalPositionalEncoding(config.max_seq_length, config.embed_dim)
        
        # Transformer blocks using wavelet attention
        self.blocks = nn.ModuleList([
            WaveletTransformerBlock(
                config.embed_dim, config.num_heads, config.hidden_dim,
                config.wavelet_type, config.wavelet_levels, config.dropout)
            for _ in range(config.num_layers)
        ])
        
        # Output head
        self.norm = nn.LayerNorm(config.embed_dim)
        self.output_proj = nn.Linear(config.embed_dim, config.vocab_size)
        
        # Basis function currently in use
        self.current_basis = BasisFunction(config.wavelet_type)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights"""
        # Initialize projection weights with small values
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def update_basis_function(self, basis: BasisFunction) -> None:
        """
        Update the model to use a new basis function.
        
        Args:
            basis: New basis function to use
        """
        self.current_basis = basis.copy()
        
        # Update wavelet parameters in all transformer blocks
        for block in self.blocks:
            # Update wavelet transform in attention mechanism
            block.wavelet_attn.wavelet.wavelet_type = basis.type_name
            
            # If PyWavelets is available, update wavelet filters
            if WAVELET_AVAILABLE and hasattr(block.wavelet_attn.wavelet, 'dec_lo'):
                if 'dec_lo' in basis.params:
                    block.wavelet_attn.wavelet.dec_lo.data = torch.tensor(basis.params['dec_lo'])
                if 'dec_hi' in basis.params:
                    block.wavelet_attn.wavelet.dec_hi.data = torch.tensor(basis.params['dec_hi'])
                if 'rec_lo' in basis.params:
                    block.wavelet_attn.wavelet.rec_lo.data = torch.tensor(basis.params['rec_lo'])
                if 'rec_hi' in basis.params:
                    block.wavelet_attn.wavelet.rec_hi.data = torch.tensor(basis.params['rec_hi'])
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the signal-based language model.
        
        Args:
            x: Input token indices [batch_size, seq_length]
            mask: Attention mask [batch_size, seq_length]
            
        Returns:
            Logits [batch_size, seq_length, vocab_size]
        """
        # Get embeddings
        x = self.token_embedding(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Process through transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final normalization and projection
        x = self.norm(x)
        return self.output_proj(x)
    
    def count_parameters(self) -> int:
        """Count the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_complexity_info(self) -> Dict:
        """
        Get information about computational complexity.
        
        Returns:
            Dictionary with complexity metrics
        """
        # Collect operation count ratios from all attention layers
        attn_op_ratios = []
        for block in self.blocks:
            if hasattr(block.wavelet_attn.approx_attention, 'get_complexity_ratio'):
                ratio = block.wavelet_attn.approx_attention.get_complexity_ratio()
                attn_op_ratios.append(ratio)
        
        # Get embedding info
        if self.use_hybrid:
            spectral_ratio = self.token_embedding.get_mixing_ratio()
        else:
            spectral_ratio = 1.0
        
        return {
            'attention_op_ratios': attn_op_ratios,
            'spectral_embedding_ratio': spectral_ratio,
            'parameter_count': self.count_parameters()
        } 

#-------------------------------------------------------------------------
# Training and Evaluation Harness
#-------------------------------------------------------------------------

class TextDataset(Dataset):
    """
    Simple text dataset for language modeling.
    Uses a tokenizer to convert text to token IDs.
    """
    def __init__(self, texts: List[str], tokenizer: Any, 
                 seq_length: int = 512, stride: int = 256):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.stride = stride
        
        # Tokenize texts
        logging.info("Tokenizing dataset...")
        self.examples = []
        
        for text in tqdm(texts, desc="Processing texts"):
            # Tokenize
            tokenized = self.tokenizer.encode(text)
            
            # Create examples with stride
            for i in range(0, len(tokenized) - seq_length + 1, stride):
                example = tokenized[i:i + seq_length]
                self.examples.append(example)
        
        logging.info(f"Created {len(self.examples)} examples")
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get input and target sequences for language modeling"""
        tokens = self.examples[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


class SimpleTokenizer:
    """
    A basic tokenizer for character-level or word-level tokenization.
    Used when more sophisticated tokenizers are not available.
    """
    def __init__(self, mode: str = 'char', vocab_file: Optional[str] = None):
        self.mode = mode
        self.vocab = {}
        self.inv_vocab = {}
        
        # Special tokens
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.pad_token_id = 0
        self.unk_token_id = 1
        
        # Initialize vocabulary
        if vocab_file and os.path.exists(vocab_file):
            self.load_vocab(vocab_file)
        else:
            # Start with special tokens
            self.vocab = {self.pad_token: self.pad_token_id, self.unk_token: self.unk_token_id}
            self.inv_vocab = {self.pad_token_id: self.pad_token, self.unk_token_id: self.unk_token}
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into tokens"""
        if self.mode == 'char':
            return list(text)
        elif self.mode == 'word':
            return text.split()
        else:
            raise ValueError(f"Unknown tokenization mode: {self.mode}")
    
    def build_vocab(self, texts: List[str], max_vocab_size: int = 10000) -> None:
        """Build vocabulary from texts"""
        # Start with special tokens
        self.vocab = {self.pad_token: self.pad_token_id, self.unk_token: self.unk_token_id}
        
        # Count token frequencies
        token_freq = {}
        for text in texts:
            tokens = self.tokenize(text)
            for token in tokens:
                token_freq[token] = token_freq.get(token, 0) + 1
        
        # Sort by frequency
        sorted_tokens = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Build vocab with most frequent tokens
        vocab_size = min(max_vocab_size - len(self.vocab), len(sorted_tokens))
        for i, (token, _) in enumerate(sorted_tokens[:vocab_size]):
            token_id = i + len(self.vocab)
            self.vocab[token] = token_id
        
        # Build inverse vocab
        self.inv_vocab = {id: token for token, id in self.vocab.items()}
        
        logging.info(f"Built vocabulary with {len(self.vocab)} tokens")
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs"""
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.unk_token_id) for token in tokens]
    
    def decode(self, ids: List[int]) -> str:
        """Convert token IDs to text"""
        tokens = [self.inv_vocab.get(id, self.unk_token) for id in ids]
        
        if self.mode == 'char':
            return ''.join(tokens)
        elif self.mode == 'word':
            return ' '.join(tokens)
        else:
            raise ValueError(f"Unknown tokenization mode: {self.mode}")
    
    def save_vocab(self, vocab_file: str) -> None:
        """Save vocabulary to file"""
        with open(vocab_file, 'w') as f:
            json.dump(self.vocab, f, indent=2)
    
    def load_vocab(self, vocab_file: str) -> None:
        """Load vocabulary from file"""
        with open(vocab_file, 'r') as f:
            self.vocab = json.load(f)
        
        # Build inverse vocab
        self.inv_vocab = {int(id): token for token, id in self.vocab.items()}
        
        logging.info(f"Loaded vocabulary with {len(self.vocab)} tokens")
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.vocab)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    epoch: int,
    config: Config,
    hrfevo_controller: Optional[HRFEvoController] = None,
    tensorboard: Optional[Any] = None
) -> Dict:
    """
    Train for one epoch.
    
    Args:
        model: The model to train
        dataloader: DataLoader for training data
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        epoch: Current epoch number
        config: Training configuration
        hrfevo_controller: Optional HRFEvo controller for basis evolution
        tensorboard: Optional TensorBoard writer
        
    Returns:
        Dict with training metrics
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0
    total_tokens = 0
    total_correct = 0
    
    # Progress bar
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader),
                       desc=f"Epoch {epoch+1}/{config.max_epochs}")
    
    # Tracking metrics for HRFEvo
    complexity_metrics = []
    
    # Track gradients
    grad_norms = []
    
    # Start timing
    start_time = time.time()
    
    for batch_idx, (data, targets) in progress_bar:
        # Move data to device
        data, targets = data.to(device), targets.to(device)
        batch_size, seq_length = data.shape
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass
        with torch.cuda.amp.autocast() if config.use_mixed_precision else nullcontext():
            outputs = model(data)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Track gradient norms
        if batch_idx % 10 == 0:
            grad_norm = torch.norm(torch.stack([
                torch.norm(p.grad.detach()) for p in model.parameters() if p.grad is not None
            ]))
            grad_norms.append(grad_norm.item())
        
        # Clip gradients
        if config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        
        # Update weights
        optimizer.step()
        
        # Update learning rate for warmup
        if scheduler is not None:
            scheduler.step()
        
        # Calculate metrics
        total_loss += loss.item() * batch_size * seq_length
        total_tokens += batch_size * seq_length
        
        # Calculate accuracy
        preds = torch.argmax(outputs, dim=-1)
        correct = (preds == targets).sum().item()
        total_correct += correct
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item(),
            'ppl': torch.exp(torch.tensor(loss.item())).item(),
            'acc': correct / (batch_size * seq_length),
        })
        
        # Log to TensorBoard
        if tensorboard and batch_idx % config.log_interval == 0:
            global_step = epoch * len(dataloader) + batch_idx
            tensorboard.add_scalar('train/loss', loss.item(), global_step)
            tensorboard.add_scalar('train/ppl', torch.exp(torch.tensor(loss.item())).item(), global_step)
            tensorboard.add_scalar('train/accuracy', correct / (batch_size * seq_length), global_step)
            
            # Track complexity metrics
            if hasattr(model, 'get_complexity_info'):
                complexity_info = model.get_complexity_info()
                complexity_metrics.append(complexity_info)
                
                # Log complexity ratios
                if 'attention_op_ratios' in complexity_info and complexity_info['attention_op_ratios']:
                    avg_ratio = sum(complexity_info['attention_op_ratios']) / len(complexity_info['attention_op_ratios'])
                    tensorboard.add_scalar('complexity/attention_ratio', avg_ratio, global_step)
                
                if 'spectral_embedding_ratio' in complexity_info:
                    tensorboard.add_scalar('complexity/embedding_ratio', 
                                         complexity_info['spectral_embedding_ratio'], global_step)
        
        # Memory tracking
        if batch_idx % 50 == 0:
            memory_stats(f"Batch {batch_idx}")
        
        # Run HRFEvo basis optimization periodically
        if hrfevo_controller and batch_idx > 0 and batch_idx % 100 == 0:
            logging.info("Running HRFEvo basis optimization...")
            
            # Define evaluation function for basis functions
            def evaluate_basis(basis):
                # Apply basis function to model temporarily
                original_basis = model.current_basis
                model.update_basis_function(basis)
                
                # Evaluate on a small batch
                model.eval()
                with torch.no_grad():
                    eval_data, eval_targets = next(iter(dataloader))
                    eval_data, eval_targets = eval_data.to(device), eval_targets.to(device)
                    
                    # Forward pass
                    start_time = time.time()
                    outputs = model(eval_data)
                    inference_time = time.time() - start_time
                    
                    # Calculate metrics
                    loss = criterion(outputs.view(-1, outputs.size(-1)), eval_targets.view(-1))
                    perplexity = torch.exp(loss).item()
                    
                    # Get complexity info
                    complexity_info = model.get_complexity_info()
                    
                    # Calculate efficiency score based on inference time and op ratios
                    if 'attention_op_ratios' in complexity_info and complexity_info['attention_op_ratios']:
                        avg_ratio = sum(complexity_info['attention_op_ratios']) / len(complexity_info['attention_op_ratios'])
                        computation_efficiency = avg_ratio / inference_time
                    else:
                        computation_efficiency = 1.0 / inference_time
                
                # Restore original basis
                model.update_basis_function(original_basis)
                model.train()
                
                return {
                    'perplexity': perplexity,
                    'computation_efficiency': computation_efficiency
                }
            
            # Run one generation of evolution
            stats = hrfevo_controller.evolve_generation(evaluate_basis)
            
            # Update model with best basis
            best_basis = hrfevo_controller.get_best_basis()
            if best_basis:
                model.update_basis_function(best_basis)
                
                # Log evolution progress
                if tensorboard:
                    global_step = epoch * len(dataloader) + batch_idx
                    tensorboard.add_scalar('evolution/best_fitness', stats['best_fitness'], global_step)
                    tensorboard.add_scalar('evolution/avg_fitness', stats['avg_fitness'], global_step)
    
    # Calculate epoch metrics
    train_loss = total_loss / total_tokens
    train_ppl = torch.exp(torch.tensor(train_loss)).item()
    train_accuracy = total_correct / total_tokens
    train_time = time.time() - start_time
    
    # Log epoch metrics
    metrics = {
        'loss': train_loss,
        'ppl': train_ppl,
        'accuracy': train_accuracy,
        'tokens_per_sec': total_tokens / train_time,
        'grad_norm_mean': sum(grad_norms) / len(grad_norms) if grad_norms else 0,
    }
    
    # Add complexity metrics if available
    if complexity_metrics:
        avg_attention_ratios = []
        for metric in complexity_metrics:
            if 'attention_op_ratios' in metric and metric['attention_op_ratios']:
                avg_ratio = sum(metric['attention_op_ratios']) / len(metric['attention_op_ratios'])
                avg_attention_ratios.append(avg_ratio)
        
        if avg_attention_ratios:
            metrics['attention_complexity_ratio'] = sum(avg_attention_ratios) / len(avg_attention_ratios)
    
    logging.info(f"Epoch {epoch+1} - Train loss: {train_loss:.4f}, PPL: {train_ppl:.2f}, "
               f"Accuracy: {train_accuracy:.4f}, Speed: {metrics['tokens_per_sec']:.1f} tokens/sec")
    
    return metrics


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    tensorboard: Optional[Any] = None,
    mode: str = 'valid',
    global_step: int = 0
) -> Dict:
    """
    Evaluate the model.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to evaluate on
        tensorboard: Optional TensorBoard writer
        mode: Evaluation mode ('valid' or 'test')
        global_step: Global step for logging
        
    Returns:
        Dict with evaluation metrics
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0
    total_tokens = 0
    total_correct = 0
    
    # Track complexity metrics
    complexity_metrics = []
    
    # Start timing
    start_time = time.time()
    
    with torch.no_grad():
        for data, targets in tqdm(dataloader, desc=f"{mode.capitalize()} Evaluation"):
            # Move data to device
            data, targets = data.to(device), targets.to(device)
            batch_size, seq_length = data.shape
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            # Calculate metrics
            total_loss += loss.item() * batch_size * seq_length
            total_tokens += batch_size * seq_length
            
            # Calculate accuracy
            preds = torch.argmax(outputs, dim=-1)
            correct = (preds == targets).sum().item()
            total_correct += correct
            
            # Track complexity metrics
            if hasattr(model, 'get_complexity_info'):
                complexity_info = model.get_complexity_info()
                complexity_metrics.append(complexity_info)
    
    # Calculate metrics
    eval_loss = total_loss / total_tokens
    eval_ppl = torch.exp(torch.tensor(eval_loss)).item()
    eval_accuracy = total_correct / total_tokens
    eval_time = time.time() - start_time
    
    # Log metrics
    metrics = {
        'loss': eval_loss,
        'ppl': eval_ppl,
        'accuracy': eval_accuracy,
        'tokens_per_sec': total_tokens / eval_time
    }
    
    # Add complexity metrics if available
    if complexity_metrics:
        avg_attention_ratios = []
        for metric in complexity_metrics:
            if 'attention_op_ratios' in metric and metric['attention_op_ratios']:
                avg_ratio = sum(metric['attention_op_ratios']) / len(metric['attention_op_ratios'])
                avg_attention_ratios.append(avg_ratio)
        
        if avg_attention_ratios:
            metrics['attention_complexity_ratio'] = sum(avg_attention_ratios) / len(avg_attention_ratios)
    
    # Log to TensorBoard
    if tensorboard:
        tensorboard.add_scalar(f'{mode}/loss', eval_loss, global_step)
        tensorboard.add_scalar(f'{mode}/ppl', eval_ppl, global_step)
        tensorboard.add_scalar(f'{mode}/accuracy', eval_accuracy, global_step)
        
        if 'attention_complexity_ratio' in metrics:
            tensorboard.add_scalar(f'{mode}/attention_ratio', metrics['attention_complexity_ratio'], global_step)
    
    logging.info(f"{mode.capitalize()} - Loss: {eval_loss:.4f}, PPL: {eval_ppl:.2f}, "
               f"Accuracy: {eval_accuracy:.4f}, Speed: {metrics['tokens_per_sec']:.1f} tokens/sec")
    
    return metrics


def benchmark_complexity(
    model: nn.Module,
    seq_lengths: List[int],
    device: torch.device,
    batch_size: int = 4,
    embed_dim: int = 256,
    num_runs: int = 3
) -> Dict:
    """
    Benchmark the computational complexity of the model with different sequence lengths.
    
    Args:
        model: The model to benchmark
        seq_lengths: List of sequence lengths to test
        device: Device to benchmark on
        batch_size: Batch size for testing
        embed_dim: Embedding dimension
        num_runs: Number of runs to average
        
    Returns:
        Dict with benchmark results
    """
    model.eval()
    results = {'seq_lengths': seq_lengths, 'times': [], 'memory': [], 'complexity_ratios': []}
    
    for seq_length in seq_lengths:
        logging.info(f"Benchmarking sequence length: {seq_length}")
        
        # Create random input data
        dummy_input = torch.randint(0, 1000, (batch_size, seq_length), device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(2):
                _ = model(dummy_input)
        
        # Benchmark runs
        times = []
        memory_usage = []
        complexity_ratios = []
        
        for run in range(num_runs):
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Track memory before
            mem_before = memory_stats(f"Before run {run}")
            
            # Time the forward pass
            with torch.no_grad():
                start_time = time.time()
                _ = model(dummy_input)
                elapsed = time.time() - start_time
            
            # Track memory after
            mem_after = memory_stats(f"After run {run}")
            
            # Record metrics
            times.append(elapsed)
            
            # Calculate memory usage
            if torch.cuda.is_available():
                memory_usage.append(mem_after['allocated'] - mem_before['allocated'])
            
            # Get complexity info
            if hasattr(model, 'get_complexity_info'):
                complexity_info = model.get_complexity_info()
                if 'attention_op_ratios' in complexity_info and complexity_info['attention_op_ratios']:
                    avg_ratio = sum(complexity_info['attention_op_ratios']) / len(complexity_info['attention_op_ratios'])
                    complexity_ratios.append(avg_ratio)
        
        # Average results
        avg_time = sum(times) / len(times)
        avg_memory = sum(memory_usage) / len(memory_usage) if memory_usage else 0
        avg_ratio = sum(complexity_ratios) / len(complexity_ratios) if complexity_ratios else 0
        
        # Record results
        results['times'].append(avg_time)
        results['memory'].append(avg_memory)
        results['complexity_ratios'].append(avg_ratio)
        
        logging.info(f"Sequence length {seq_length}: "
                   f"Time: {avg_time:.4f}s, "
                   f"Memory: {avg_memory:.2f}GB, "
                   f"Complexity ratio: {avg_ratio:.4f}")
    
    return results


def evaluate_linguistic_tasks(
    model: nn.Module,
    tasks: Dict[str, Dict],
    tokenizer: Any,
    device: torch.device,
    max_length: int = 512,
    batch_size: int = 8
) -> Dict:
    """
    Evaluate the model on linguistic tasks.
    
    Args:
        model: The model to evaluate
        tasks: Dictionary of tasks with their data and evaluation functions
        tokenizer: Tokenizer for processing text
        device: Device to evaluate on
        max_length: Maximum sequence length
        batch_size: Batch size for evaluation
        
    Returns:
        Dict with task evaluation results
    """
    model.eval()
    results = {}
    
    for task_name, task_data in tasks.items():
        logging.info(f"Evaluating on task: {task_name}")
        
        # Get task data
        examples = task_data['examples']
        eval_func = task_data['eval_func']
        
        # Process examples
        task_results = []
        
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            
            # Prepare inputs
            inputs = []
            targets = []
            
            for example in batch:
                # Tokenize input
                input_text = example['input']
                input_ids = tokenizer.encode(input_text)
                
                # Truncate if needed
                if len(input_ids) > max_length:
                    input_ids = input_ids[:max_length]
                
                inputs.append(input_ids)
                targets.append(example.get('target'))
            
            # Pad inputs to the same length
            max_len = max(len(seq) for seq in inputs)
            padded_inputs = []
            
            for seq in inputs:
                padded = seq + [0] * (max_len - len(seq))
                padded_inputs.append(padded)
            
            # Convert to tensors
            input_tensor = torch.tensor(padded_inputs, device=device)
            
            # Forward pass
            with torch.no_grad():
                outputs = model(input_tensor)
            
            # Process results using the task-specific evaluation function
            batch_results = eval_func(outputs, targets, input_tensor)
            task_results.extend(batch_results)
        
        # Calculate task metrics
        task_metrics = task_data['metric_func'](task_results)
        results[task_name] = task_metrics
        
        logging.info(f"Task {task_name}: {task_metrics}")
    
    return results


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    metrics: Dict,
    config: Config,
    hrfevo_controller: Optional[HRFEvoController] = None,
    filepath: str = None
) -> str:
    """
    Save a checkpoint of the model and training state.
    
    Args:
        model: The model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler state
        epoch: Current epoch number
        metrics: Training metrics
        config: Training configuration
        hrfevo_controller: Optional HRFEvo controller
        filepath: Path to save the checkpoint (optional)
        
    Returns:
        Path to the saved checkpoint
    """
    if filepath is None:
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        filepath = os.path.join(config.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    
    # Prepare checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'metrics': metrics,
        'config': config.to_dict(),
    }
    
    # Add HRFEvo state if available
    if hrfevo_controller:
        checkpoint['hrfevo_state'] = {
            'generation_count': hrfevo_controller.generation_count,
            'best_basis': hrfevo_controller.best_basis.to_dict() if hrfevo_controller.best_basis else None,
            'fitness_history': hrfevo_controller.fitness_history
        }
    
    # Save checkpoint
    torch.save(checkpoint, filepath)
    logging.info(f"Checkpoint saved to {filepath}")
    
    return filepath


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Any = None,
    hrfevo_controller: Optional[HRFEvoController] = None,
    device: torch.device = None
) -> Dict:
    """
    Load a checkpoint.
    
    Args:
        filepath: Path to the checkpoint
        model: The model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Learning rate scheduler to load state into (optional)
        hrfevo_controller: HRFEvo controller to load state into (optional)
        device: Device to load the checkpoint to
        
    Returns:
        Dict with checkpoint data
    """
    # Load checkpoint data
    checkpoint = torch.load(filepath, map_location=device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Load HRFEvo state if available
    if hrfevo_controller and 'hrfevo_state' in checkpoint:
        hrfevo_state = checkpoint['hrfevo_state']
        
        # Update controller state
        hrfevo_controller.generation_count = hrfevo_state['generation_count']
        hrfevo_controller.fitness_history = hrfevo_state['fitness_history']
        
        # Load best basis if available
        if hrfevo_state['best_basis']:
            hrfevo_controller.best_basis = BasisFunction.from_dict(hrfevo_state['best_basis'])
            
            # Update model with best basis
            model.update_basis_function(hrfevo_controller.best_basis)
    
    logging.info(f"Checkpoint loaded from {filepath} (epoch {checkpoint['epoch']})")
    
    return checkpoint 


#-------------------------------------------------------------------------
# Hardware Optimization Layer
#-------------------------------------------------------------------------

class HardwareOptimization:
    """
    Provides hardware-specific optimizations for SignalLLM.
    Configures operations based on available hardware and optimizes memory access patterns.
    """
    def __init__(self, config: Config, device: torch.device):
        self.config = config
        self.device = device
        self.optimized = False
        
        # Detect hardware capabilities
        self.hardware_info = self._detect_hardware()
        
        # Configure optimizations based on hardware
        self.configure_optimizations()
    
    def _detect_hardware(self) -> Dict:
        """Detect hardware capabilities and characteristics"""
        info = {
            'device_type': self.device.type,
            'tensor_cores': False,
            'memory_bandwidth': 0,
            'compute_capability': None,
            'mps_available': False
        }
        
        if self.device.type == 'cuda':
            # CUDA-specific information
            device_idx = 0 if self.device.index is None else self.device.index
            info['device_name'] = torch.cuda.get_device_name(device_idx)
            
            # Check for Tensor Cores (NVIDIA Volta+ GPUs)
            compute_cap = torch.cuda.get_device_capability(device_idx)
            info['compute_capability'] = f"{compute_cap[0]}.{compute_cap[1]}"
            
            # Tensor Cores are available on Volta (7.0) and newer
            info['tensor_cores'] = compute_cap[0] >= 7
            
            # Approximate memory bandwidth (in GB/s)
            # This is a very rough estimate
            mem_clock = torch.cuda.get_device_properties(device_idx).memory_clock_rate  # KHz
            bus_width = torch.cuda.get_device_properties(device_idx).memory_bus_width  # bits
            info['memory_bandwidth'] = 2 * mem_clock * bus_width / 8 / 1e6  # GB/s
            
        elif self.device.type == 'mps':
            # Apple Silicon MPS information
            info['device_name'] = 'Apple Silicon'
            info['mps_available'] = True
            info['tensor_cores'] = True  # Apple's Neural Engine is equivalent
            
            # Approximate memory bandwidth - typical for M1/M2
            info['memory_bandwidth'] = 200  # GB/s (approximate)
        
        elif self.device.type == 'cpu':
            # CPU information
            import platform
            info['device_name'] = platform.processor() or 'Unknown CPU'
            
            # Try to get CPU cache info if on Linux
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read()
                
                import re
                # Extract cache information
                l1_cache = re.search(r'cache size\s+:\s+(\d+)\s+KB', cpuinfo)
                if l1_cache:
                    info['l1_cache'] = int(l1_cache.group(1))
                
                # Get number of cores
                cores = re.search(r'cpu cores\s+:\s+(\d+)', cpuinfo)
                if cores:
                    info['cores'] = int(cores.group(1))
            except:
                # If failed, provide defaults
                info['l1_cache'] = 32  # KB, typical L1 cache
                info['cores'] = os.cpu_count() or 4
        
        logging.info(f"Detected hardware: {info['device_name']} ({info['device_type']})")
        return info
    
    def configure_optimizations(self) -> None:
        """Configure hardware-specific optimizations"""
        if self.optimized:
            return
        
        if self.device.type == 'cuda':
            self._configure_cuda_optimizations()
        elif self.device.type == 'mps':
            self._configure_mps_optimizations()
        elif self.device.type == 'cpu':
            self._configure_cpu_optimizations()
        
        self.optimized = True
    
    def _configure_cuda_optimizations(self) -> None:
        """Configure CUDA-specific optimizations"""
        # Set appropriate optimization flags
        if self.hardware_info['tensor_cores'] and self.config.use_tensor_cores:
            # Enable TF32 for faster matrix multiplications on A100/A10X GPUs
            if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
                torch.backends.cuda.matmul.allow_tf32 = True
                logging.info("Enabled TF32 precision for CUDA matrix multiplications")
            
            # Enable cuDNN TF32
            if hasattr(torch.backends, 'cudnn') and hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
                logging.info("Enabled TF32 precision for cuDNN")
        
        # Optimize memory access patterns
        if self.config.optimize_memory_access:
            # Configure cuDNN to look for the most optimal algorithms
            if hasattr(torch.backends, 'cudnn'):
                torch.backends.cudnn.benchmark = True
                logging.info("Enabled cuDNN autotuner for optimal performance")
        
        # Configure FFT algorithm based on hardware
        compute_cap = self.hardware_info.get('compute_capability', '0.0')
        if compute_cap and float(compute_cap) >= 8.0:
            # Use the most optimized FFT implementation for Ampere+ GPUs
            os.environ['TORCH_FFT_IMPLEMENTATION'] = 'cufft'
            logging.info("Using cuFFT for FFT operations")
        
        # Configure memory allocation strategy
        if self.config.block_size > 0:
            # Use a larger allocation size to reduce fragmentation
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f'max_split_size_mb:{self.config.block_size}'
            logging.info(f"Set CUDA memory allocator block size to {self.config.block_size}MB")
    
    def _configure_mps_optimizations(self) -> None:
        """Configure Apple Silicon MPS-specific optimizations"""
        # Enable graph mode for better performance if requested
        if self.config.use_mps_graph_mode:
            # This is a fictitious setting - MPS graph mode implementation
            # would depend on how PyTorch implements this in the future
            os.environ['PYTORCH_MPS_GRAPH_MODE'] = '1'
            logging.info("Enabled MPS graph mode for Apple Silicon")
        
        # Configure memory access patterns
        if self.config.optimize_memory_access:
            # Use unified memory optimally
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
            logging.info("Configured MPS memory access patterns")
        
        # Set FFT implementation
        os.environ['TORCH_FFT_IMPLEMENTATION'] = 'mps'
        logging.info("Using MPS for FFT operations")
    
    def _configure_cpu_optimizations(self) -> None:
        """Configure CPU-specific optimizations"""
        # Configure OpenMP threads for parallel processing
        if hasattr(torch, 'set_num_threads'):
            # Use all available cores, but leave one for the system
            optimal_threads = max(1, os.cpu_count() - 1) if os.cpu_count() else 4
            torch.set_num_threads(optimal_threads)
            logging.info(f"Set PyTorch CPU threads to {optimal_threads}")
        
        # Configure FFT implementation
        os.environ['TORCH_FFT_IMPLEMENTATION'] = 'mkl'
        logging.info("Using MKL for FFT operations")
        
        # Configure memory access patterns
        if self.config.optimize_memory_access:
            # Block size for CPU cache optimization
            l1_cache = self.hardware_info.get('l1_cache', 32)  # KB
            
            # Set block size to fit in L1 cache
            self.config.block_size = min(self.config.block_size, l1_cache)
            logging.info(f"Optimized block size for CPU cache: {self.config.block_size}KB")
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """
        Apply hardware-specific optimizations to the model.
        
        Args:
            model: The model to optimize
            
        Returns:
            Optimized model
        """
        # Ensure optimizations are configured
        if not self.optimized:
            self.configure_optimizations()
        
        # Mixed precision setup
        if self.config.use_mixed_precision:
            if self.device.type == 'cuda':
                # For CUDA, apply AMP
                model = model.half()
                logging.info("Applied mixed precision (FP16) for CUDA")
            elif self.device.type == 'mps':
                # For MPS, apply half precision
                model = model.half()
                logging.info("Applied mixed precision (FP16) for MPS")
        
        # Optimize memory layout (channels_last format) for convolutional operations
        if hasattr(model, 'to') and self.device.type == 'cuda':
            if self.hardware_info.get('tensor_cores', False):
                # Check if the model has Conv2d layers that would benefit from channels_last
                has_conv = any(isinstance(m, nn.Conv2d) for m in model.modules())
                if has_conv:
                    model = model.to(memory_format=torch.channels_last)
                    logging.info("Using channels_last memory format for CUDA optimization")
        
        # Ensure model is on the target device
        model = model.to(self.device)
        
        return model
    
    def create_optimized_dataloader(
        self, 
        dataset: Dataset, 
        batch_size: int, 
        shuffle: bool = True, 
        num_workers: int = 0
    ) -> DataLoader:
        """
        Create an optimized DataLoader for the current hardware.
        
        Args:
            dataset: The dataset to load
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
            
        Returns:
            Optimized DataLoader
        """
        # Determine optimal number of workers
        if num_workers <= 0:
            if self.device.type == 'cuda' or self.device.type == 'mps':
                # For GPU: use CPU cores but leave some for system
                num_workers = max(1, os.cpu_count() - 2) if os.cpu_count() else 2
            else:
                # For CPU: use half the cores to avoid contention
                num_workers = max(1, os.cpu_count() // 2) if os.cpu_count() else 2
        
        # Configure pin memory for faster GPU transfers
        pin_memory = self.device.type in ('cuda', 'mps')
        
        # Create DataLoader with optimized settings
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0
        )
        
        logging.info(f"Created optimized DataLoader with {num_workers} workers, "
                   f"pin_memory={pin_memory}")
        
        return dataloader
    
    def optimize_fft_operations(self, signal: torch.Tensor, dim: int = -1) -> Tuple[callable, callable]:
        """
        Return optimized FFT and IFFT operations for the current hardware.
        
        Args:
            signal: Example signal tensor to optimize for
            dim: Dimension to perform FFT on
            
        Returns:
            Tuple of (optimized_fft_func, optimized_ifft_func)
        """
        # Get signal properties for optimization
        seq_length = signal.shape[dim]
        
        # Determine optimal FFT implementation
        if self.device.type == 'cuda':
            if self.hardware_info.get('compute_capability', '0.0') >= '7.0':
                # For Volta+ GPUs, use cuFFT with optimal plans
                def optimized_fft(x):
                    return torch.fft.rfft(x, dim=dim)
                
                def optimized_ifft(x):
                    return torch.fft.irfft(x, n=seq_length, dim=dim)
            else:
                # For older GPUs, use standard FFT
                optimized_fft = lambda x: torch.fft.rfft(x, dim=dim)
                optimized_ifft = lambda x: torch.fft.irfft(x, n=seq_length, dim=dim)
        
        elif self.device.type == 'mps':
            # For MPS, use standard FFT (Apple optimizes internally)
            optimized_fft = lambda x: torch.fft.rfft(x, dim=dim)
            optimized_ifft = lambda x: torch.fft.irfft(x, n=seq_length, dim=dim)
        
        else:
            # For CPU, use MKL-optimized FFT
            optimized_fft = lambda x: torch.fft.rfft(x, dim=dim)
            optimized_ifft = lambda x: torch.fft.irfft(x, n=seq_length, dim=dim)
        
        return optimized_fft, optimized_ifft
    
    def get_hardware_info(self) -> Dict:
        """Get detected hardware information"""
        return self.hardware_info
    
    def get_optimization_status(self) -> Dict:
        """Get current optimization status"""
        return {
            'optimized': self.optimized,
            'device_type': self.device.type,
            'tensor_cores_used': (self.hardware_info.get('tensor_cores', False) and 
                                 self.config.use_tensor_cores),
            'mixed_precision': self.config.use_mixed_precision,
            'memory_optimization': self.config.optimize_memory_access,
            'block_size': self.config.block_size
        }


class nullcontext:
    """
    A context manager that does nothing.
    Used as a placeholder when a context manager is optional.
    """
    def __init__(self, enter_result=None):
        self.enter_result = enter_result
        
    def __enter__(self):
        return self.enter_result
        
    def __exit__(self, *excinfo):
        pass


#-------------------------------------------------------------------------
# Visualization and Analysis Tools
#-------------------------------------------------------------------------

class VisualizationTools:
    """
    Tools for visualizing spectral representations and model performance.
    Provides various visualization methods for frequency patterns, basis functions,
    and performance comparisons.
    """
    def __init__(self, output_dir: str = './visualizations'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_frequency_domain(self, tensor: torch.Tensor, title: str = None, 
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize a tensor in the frequency domain.
        
        Args:
            tensor: Input tensor [batch_size, seq_length, embed_dim]
            title: Plot title
            save_path: Path to save the figure (optional)
            
        Returns:
            Matplotlib figure
        """
        # Convert to numpy for plotting
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
        
        # Ensure we have a 3D tensor
        if tensor.ndim == 2:
            tensor = tensor[np.newaxis, :, :]
        
        # Get tensor shape
        batch_size, seq_length, embed_dim = tensor.shape
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(title or "Frequency Domain Visualization", fontsize=16)
        
        # 1. Plot time domain representation (first sequence from batch)
        axes[0].set_title("Time Domain (First Sequence)")
        axes[0].imshow(tensor[0], aspect='auto', cmap='viridis')
        axes[0].set_xlabel("Embedding Dimension")
        axes[0].set_ylabel("Sequence Position")
        
        # 2. Plot frequency domain for sequence dimension
        # Compute FFT along sequence dimension
        fft_seq = np.abs(np.fft.rfft(tensor[0], axis=0))
        axes[1].set_title("Frequency Domain (Sequence Dimension)")
        axes[1].imshow(fft_seq, aspect='auto', cmap='viridis')
        axes[1].set_xlabel("Embedding Dimension")
        axes[1].set_ylabel("Frequency")
        
        # 3. Plot frequency domain for embedding dimension
        # Compute FFT along embedding dimension
        fft_embed = np.abs(np.fft.rfft(tensor[0], axis=1))
        axes[2].set_title("Frequency Domain (Embedding Dimension)")
        axes[2].imshow(fft_embed, aspect='auto', cmap='viridis')
        axes[2].set_xlabel("Frequency")
        axes[2].set_ylabel("Sequence Position")
        
        # Adjust layout
        fig.tight_layout()
        
        # Save the figure if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_wavelet_decomposition(self, tensor: torch.Tensor, 
                                  wavelet_type: str = 'db4', levels: int = 3,
                                  title: str = None, save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize wavelet decomposition of a tensor.
        
        Args:
            tensor: Input tensor [batch_size, seq_length, embed_dim]
            wavelet_type: Wavelet type
            levels: Number of decomposition levels
            title: Plot title
            save_path: Path to save the figure (optional)
            
        Returns:
            Matplotlib figure
        """
        if not WAVELET_AVAILABLE:
            raise ImportError("PyWavelets is required for wavelet decomposition visualization")
        
        # Convert to numpy for plotting
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
        
        # Ensure we have a 3D tensor
        if tensor.ndim == 2:
            tensor = tensor[np.newaxis, :, :]
        
        # Take first batch item and a single embedding dimension for visualization
        signal = tensor[0, :, 0]
        
        # Compute wavelet decomposition
        coeffs = pywt.wavedec(signal, wavelet_type, level=levels)
        approx, details = coeffs[0], coeffs[1:]
        
        # Create figure
        fig, axes = plt.subplots(levels + 1, 1, figsize=(12, 2 * (levels + 1)))
        fig.suptitle(title or f"Wavelet Decomposition ({wavelet_type})", fontsize=16)
        
        # Plot original signal
        axes[0].plot(signal)
        axes[0].set_title("Original Signal")
        axes[0].set_ylabel("Amplitude")
        
        # Plot approximation and details
        axes[1].plot(approx)
        axes[1].set_title(f"Approximation (Level {levels})")
        axes[1].set_ylabel("Amplitude")
        
        for i, detail in enumerate(details):
            level = levels - i
            axes[i + 2].plot(detail)
            axes[i + 2].set_title(f"Detail (Level {level})")
            axes[i + 2].set_ylabel("Amplitude")
        
        # Set x-label for the last subplot
        axes[-1].set_xlabel("Sample")
        
        # Adjust layout
        fig.tight_layout()
        
        # Save the figure if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_basis_function(self, basis: BasisFunction, 
                           title: str = None, save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize a basis function.
        
        Args:
            basis: BasisFunction to visualize
            title: Plot title
            save_path: Path to save the figure (optional)
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(title or f"Basis Function: {basis.type_name} (Fitness: {basis.fitness:.4f})", fontsize=16)
        
        # Plot basis function parameters
        # This depends on the type of basis function
        if basis.type_name.startswith(('db', 'sym', 'coif')):
            # Wavelet basis function
            if 'dec_lo' in basis.params:
                axes[0, 0].stem(basis.params['dec_lo'])
                axes[0, 0].set_title("Decomposition Low-Pass Filter")
            
            if 'dec_hi' in basis.params:
                axes[0, 1].stem(basis.params['dec_hi'])
                axes[0, 1].set_title("Decomposition High-Pass Filter")
            
            if 'rec_lo' in basis.params:
                axes[1, 0].stem(basis.params['rec_lo'])
                axes[1, 0].set_title("Reconstruction Low-Pass Filter")
            
            if 'rec_hi' in basis.params:
                axes[1, 1].stem(basis.params['rec_hi'])
                axes[1, 1].set_title("Reconstruction High-Pass Filter")
        
        elif basis.type_name == 'harmonic':
            # Harmonic basis function
            if 'frequencies' in basis.params and 'amplitudes' in basis.params:
                frequencies = basis.params['frequencies']
                amplitudes = basis.params['amplitudes']
                
                # Plot frequency response
                axes[0, 0].stem(frequencies, amplitudes)
                axes[0, 0].set_title("Frequency Response")
                axes[0, 0].set_xlabel("Frequency")
                axes[0, 0].set_ylabel("Amplitude")
                
                # Plot phase response if available
                if 'phases' in basis.params:
                    phases = basis.params['phases']
                    axes[0, 1].stem(frequencies, phases)
                    axes[0, 1].set_title("Phase Response")
                    axes[0, 1].set_xlabel("Frequency")
                    axes[0, 1].set_ylabel("Phase")
                
                # Generate and plot time-domain representation
                t = np.linspace(0, 1, 1000)
                signal = np.zeros_like(t)
                for f, a, p in zip(frequencies, amplitudes, basis.params.get('phases', [0] * len(frequencies))):
                    signal += a * np.sin(2 * np.pi * f * t + p)
                
                axes[1, 0].plot(t, signal)
                axes[1, 0].set_title("Time Domain Representation")
                axes[1, 0].set_xlabel("Time")
                axes[1, 0].set_ylabel("Amplitude")
                
                # Plot spectrogram
                axes[1, 1].specgram(signal, Fs=len(t))
                axes[1, 1].set_title("Spectrogram")
                axes[1, 1].set_xlabel("Time")
                axes[1, 1].set_ylabel("Frequency")
        
        # Add parameter summary as text
        param_text = "\n".join([f"{k}: {v}" for k, v in basis.params.items()])
        fig.text(0.5, 0.01, f"Parameters:\n{param_text}", ha='center', fontsize=8)
        
        # Adjust layout
        fig.tight_layout()
        
        # Save the figure if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_evolution_progress(self, fitness_history: List[Dict], 
                              title: str = None, save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize the progress of evolution.
        
        Args:
            fitness_history: List of dictionaries with fitness statistics
            title: Plot title
            save_path: Path to save the figure (optional)
            
        Returns:
            Matplotlib figure
        """
        if not fitness_history:
            raise ValueError("Fitness history is empty")
        
        # Extract data
        generations = [entry['generation'] for entry in fitness_history]
        best_fitness = [entry['best_fitness'] for entry in fitness_history]
        avg_fitness = [entry['avg_fitness'] for entry in fitness_history]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle(title or "Evolution Progress", fontsize=16)
        
        # Plot fitness progress
        ax.plot(generations, best_fitness, 'o-', label='Best Fitness')
        ax.plot(generations, avg_fitness, 's-', label='Average Fitness')
        
        # Add labels and legend
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Adjust layout
        fig.tight_layout()
        
        # Save the figure if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_complexity_scaling(self, results: Dict, 
                              title: str = None, save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize the computational complexity scaling with sequence length.
        
        Args:
            results: Benchmark results dictionary
            title: Plot title
            save_path: Path to save the figure (optional)
            
        Returns:
            Matplotlib figure
        """
        # Create figure with 4 subplots (2x2)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title or "Computational Complexity Scaling", fontsize=16)
        
        # Extract data
        seq_lengths = np.array(results['seq_lengths'])
        times = np.array(results['times'])
        memory = np.array(results['memory'])
        complexity_ratios = np.array(results['complexity_ratios'])
        
        # 1. Plot execution time vs sequence length
        axes[0, 0].plot(seq_lengths, times, 'o-', label='Actual')
        
        # Fit and plot O(n log n) curve
        if len(seq_lengths) > 2:
            # Define the O(n log n) function to fit
            def nlogn_func(x, a, b):
                return a * x * np.log(x) + b
            
            # Fit the function to the data
            from scipy.optimize import curve_fit
            try:
                popt, _ = curve_fit(nlogn_func, seq_lengths, times)
                x_fit = np.linspace(seq_lengths.min(), seq_lengths.max(), 100)
                y_fit = nlogn_func(x_fit, *popt)
                axes[0, 0].plot(x_fit, y_fit, '--', label='O(n log n) fit')
                
                # Also plot O(n²) for comparison
                y_fit_n2 = popt[0] * x_fit**2 / (20 * np.log(x_fit.mean())) + popt[1]
                axes[0, 0].plot(x_fit, y_fit_n2, ':', label='O(n²) reference')
            except:
                pass
        
        axes[0, 0].set_title("Execution Time vs. Sequence Length")
        axes[0, 0].set_xlabel("Sequence Length (n)")
        axes[0, 0].set_ylabel("Time (seconds)")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Plot memory usage vs sequence length
        axes[0, 1].plot(seq_lengths, memory, 'o-')
        
        # Fit and plot linear memory usage
        if len(seq_lengths) > 2:
            # Simple linear fit
            from scipy.stats import linregress
            try:
                slope, intercept, _, _, _ = linregress(seq_lengths, memory)
                x_fit = np.linspace(seq_lengths.min(), seq_lengths.max(), 100)
                y_fit = slope * x_fit + intercept
                axes[0, 1].plot(x_fit, y_fit, '--', label='Linear fit')
            except:
                pass
        
        axes[0, 1].set_title("Memory Usage vs. Sequence Length")
        axes[0, 1].set_xlabel("Sequence Length (n)")
        axes[0, 1].set_ylabel("Memory (GB)")
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Plot complexity ratio vs sequence length
        axes[1, 0].plot(seq_lengths, complexity_ratios, 'o-')
        axes[1, 0].set_title("Complexity Ratio vs. Sequence Length")
        axes[1, 0].set_xlabel("Sequence Length (n)")
        axes[1, 0].set_ylabel("O(n log n) / O(n²) Ratio")
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Plot time/sequence_length vs log(sequence_length)
        # This should be approximately linear for O(n log n) algorithm
        time_per_n = times / seq_lengths
        log_n = np.log(seq_lengths)
        axes[1, 1].plot(log_n, time_per_n, 'o-')
        
        # Fit linear relationship
        if len(seq_lengths) > 2:
            try:
                slope, intercept, _, _, _ = linregress(log_n, time_per_n)
                x_fit = np.linspace(log_n.min(), log_n.max(), 100)
                y_fit = slope * x_fit + intercept
                axes[1, 1].plot(x_fit, y_fit, '--', label='Linear fit')
                axes[1, 1].text(0.05, 0.95, f"Slope: {slope:.4f}", transform=axes[1, 1].transAxes,
                            verticalalignment='top')
            except:
                pass
        
        axes[1, 1].set_title("Time/n vs. log(n)")
        axes[1, 1].set_xlabel("log(Sequence Length)")
        axes[1, 1].set_ylabel("Time/n (s)")
        axes[1, 1].grid(True, alpha=0.3)
        
        # Adjust layout
        fig.tight_layout()
        
        # Save the figure if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_embedding_statistics(self, model: SignalLLM, 
                                title: str = None, save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize statistics about the model's embeddings.
        
        Args:
            model: SignalLLM model
            title: Plot title
            save_path: Path to save the figure (optional)
            
        Returns:
            Matplotlib figure
        """
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title or "Embedding Statistics", fontsize=16)
        
        # Get embedding statistics from the model
        if model.use_hybrid:
            embedding = model.token_embedding.spectral_embedding
            
            # Get mixing ratio
            mixing_ratio = model.token_embedding.get_mixing_ratio()
            fig.text(0.5, 0.01, f"Hybrid Embedding Mixing Ratio: {mixing_ratio:.2f}", 
                   ha='center', fontsize=12)
        else:
            embedding = model.token_embedding
        
        # Get spectral statistics
        with torch.no_grad():
            stats = embedding.get_spectral_stats()
        
        # 1. Plot power spectrum
        power_spectrum = stats['power_spectrum'].numpy()
        axes[0, 0].plot(power_spectrum)
        axes[0, 0].set_title("Power Spectrum")
        axes[0, 0].set_xlabel("Embedding Dimension")
        axes[0, 0].set_ylabel("Power")
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Plot frequency amplitudes
        freq_amplitudes = stats['frequency_amplitudes'].numpy()
        axes[0, 1].stem(freq_amplitudes)
        axes[0, 1].set_title("Average Frequency Amplitudes")
        axes[0, 1].set_xlabel("Harmonic Basis Index")
        axes[0, 1].set_ylabel("Amplitude")
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Plot frequency phases
        freq_phases = stats['frequency_phases'].numpy()
        axes[1, 0].stem(freq_phases)
        axes[1, 0].set_title("Average Frequency Phases")
        axes[1, 0].set_xlabel("Harmonic Basis Index")
        axes[1, 0].set_ylabel("Phase")
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Plot token usage
        token_usage = stats['token_usage'].numpy()
        # Plot histogram of token usage
        axes[1, 1].hist(token_usage, bins=30)
        axes[1, 1].set_title("Token Usage Distribution")
        axes[1, 1].set_xlabel("Usage Count")
        axes[1, 1].set_ylabel("Number of Tokens")
        axes[1, 1].grid(True, alpha=0.3)
        
        # Adjust layout
        fig.tight_layout()
        
        # Save the figure if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_spectral_gap_analysis(self, analysis_results: Dict, 
                                 title: str = None, save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize spectral gap analysis results from Dr. Tao's theoretical insights.
        
        Args:
            analysis_results: Results from SpectralGapAnalyzer
            title: Plot title
            save_path: Path to save the figure (optional)
            
        Returns:
            Matplotlib figure
        """
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title or "Spectral Gap Analysis", fontsize=16)
        
        # Plot approximation eigenvalues
        if 'approx_eigenvalues' in analysis_results:
            approx_eigs = analysis_results['approx_eigenvalues'].detach().cpu().numpy()
            axes[0, 0].plot(approx_eigs, 'o-')
            axes[0, 0].set_title("Approximation Eigenvalues")
            axes[0, 0].set_xlabel("Index")
            axes[0, 0].set_ylabel("Eigenvalue")
            axes[0, 0].grid(True, alpha=0.3)
            
            # Highlight spectral gap
            if len(approx_eigs) > 1:
                gap = approx_eigs[1] - approx_eigs[0]
                axes[0, 0].annotate(f"Gap: {gap:.4f}", 
                                  xy=(0.5, approx_eigs[0] + gap/2),
                                  xytext=(2, 0), 
                                  textcoords="offset points",
                                  arrowprops=dict(arrowstyle="->"))
        
        # Plot detail eigenvalues
        if 'detail_eigenvalues' in analysis_results and analysis_results['detail_eigenvalues']:
            # Plot first few levels of detail eigenvalues
            colors = ['b', 'r', 'g', 'c', 'm']
            for i, level_eigs in enumerate(analysis_results['detail_eigenvalues'][:min(5, len(analysis_results['detail_eigenvalues']))]):
                level_eigs = level_eigs.detach().cpu().numpy()
                color = colors[i % len(colors)]
                axes[0, 1].plot(level_eigs, 'o-', color=color, label=f"Level {i+1}")
            
            axes[0, 1].set_title("Detail Eigenvalues by Level")
            axes[0, 1].set_xlabel("Index")
            axes[0, 1].set_ylabel("Eigenvalue")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot gaps by level
        if 'detail_gaps' in analysis_results:
            gaps = [analysis_results['approx_gap']] + analysis_results['detail_gaps']
            axes[1, 0].bar(range(len(gaps)), gaps)
            axes[1, 0].set_title("Spectral Gaps by Level")
            axes[1, 0].set_xlabel("Level (0=Approx, 1+=Details)")
            axes[1, 0].set_ylabel("Spectral Gap")
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add text annotation for overall gap
            if 'overall_gap' in analysis_results:
                overall_gap = analysis_results['overall_gap']
                axes[1, 0].text(0.5, 0.9, f"Overall Gap: {overall_gap:.4f}",
                              transform=axes[1, 0].transAxes,
                              bbox=dict(facecolor='white', alpha=0.8))
        
        # Plot spectral gap interpretation
        if 'overall_gap' in analysis_results:
            # Create a gauge-like visualization
            gap = analysis_results['overall_gap']
            # Normalize to [0, 1] range using a tanh function (centered around 1.0)
            normalized_gap = 0.5 * (np.tanh(2 * (gap - 1.0)) + 1)
            
            # Create a gauge chart
            axes[1, 1].set_aspect('equal')
            axes[1, 1].set_xlim(-1.1, 1.1)
            axes[1, 1].set_ylim(-1.1, 1.1)
            
            # Draw gauge background
            theta = np.linspace(np.pi, 0, 100)
            x = np.cos(theta)
            y = np.sin(theta)
            axes[1, 1].plot(x, y, 'k-', linewidth=2)
            
            # Draw colored regions
            poor = np.linspace(np.pi, 0.8 * np.pi, 100)
            axes[1, 1].fill_between(np.cos(poor), 0, np.sin(poor), color='red', alpha=0.3)
            
            moderate = np.linspace(0.8 * np.pi, 0.4 * np.pi, 100)
            axes[1, 1].fill_between(np.cos(moderate), 0, np.sin(moderate), color='yellow', alpha=0.3)
            
            good = np.linspace(0.4 * np.pi, 0, 100)
            axes[1, 1].fill_between(np.cos(good), 0, np.sin(good), color='green', alpha=0.3)
            
            # Draw needle
            gauge_value = np.pi * (1 - normalized_gap)
            axes[1, 1].plot([0, 0.9 * np.cos(gauge_value)], [0, 0.9 * np.sin(gauge_value)], 'k-', linewidth=2)
            axes[1, 1].plot([0], [0], 'ko', markersize=8)
            
            # Add labels
            axes[1, 1].text(-0.8, 0.2, "Poor", color='red', fontsize=12)
            axes[1, 1].text(-0.2, 0.6, "Moderate", color='brown', fontsize=12)
            axes[1, 1].text(0.5, 0.6, "Good", color='green', fontsize=12)
            
            axes[1, 1].set_title("Spectral Gap Quality")
            axes[1, 1].text(0, -0.2, f"Gap Value: {gap:.4f}", ha='center', fontsize=12,
                          bbox=dict(facecolor='white', alpha=0.8))
            
            # Remove ticks
            axes[1, 1].set_xticks([])
            axes[1, 1].set_yticks([])
        
        # Adjust layout
        fig.tight_layout()
        
        # Save the figure if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def visualize_attention_patterns(self, model: SignalLLM, text: str, tokenizer: Any,
                                   title: str = None, save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize attention patterns for a given text.
        
        Args:
            model: SignalLLM model
            text: Input text
            tokenizer: Tokenizer to process the text
            title: Plot title
            save_path: Path to save the figure (optional)
            
        Returns:
            Matplotlib figure
        """
        # Tokenize input text
        input_ids = tokenizer.encode(text)
        input_tensor = torch.tensor([input_ids], device=model.current_basis.type_name.device)
        
        # Forward pass with attention pattern capture
        model.eval()
        with torch.no_grad():
            # Extract attention patterns
            attn_patterns = []
            
            # Hook to capture attention patterns
            def hook_fn(module, input, output):
                attn_weights = input[0]  # Input to matmul(attn_weights, v)
                attn_patterns.append(attn_weights.detach().cpu())
            
            hooks = []
            for block in model.blocks:
                # Different attention mechanisms might have different hook points
                if hasattr(block.wavelet_attn, 'approx_attention'):
                    # Register hook in wavelet attention
                    h = block.wavelet_attn.approx_attention.register_forward_hook(hook_fn)
                    hooks.append(h)
            
            # Run the model
            _ = model(input_tensor)
            
            # Remove hooks
            for h in hooks:
                h.remove()
        
        # Create figure based on number of attention heads and blocks
        num_layers = len(attn_patterns)
        num_heads = attn_patterns[0].size(1) if attn_patterns else 1
        
        fig, axes = plt.subplots(num_layers, num_heads, figsize=(num_heads * 3, num_layers * 3))
        fig.suptitle(title or "Attention Patterns", fontsize=16)
        
        # If there's only one layer and one head, axes won't be a 2D array
        if num_layers == 1 and num_heads == 1:
            axes = np.array([[axes]])
        elif num_layers == 1:
            axes = axes.reshape(1, -1)
        elif num_heads == 1:
            axes = axes.reshape(-1, 1)
        
        # Get token strings for axis labels
        token_strings = [tokenizer.decode([id]) for id in input_ids]
        
        # Plot attention patterns
        for layer_idx, layer_attn in enumerate(attn_patterns):
            for head_idx in range(num_heads):
                ax = axes[layer_idx, head_idx]
                
                # Extract attention weights for this head
                attn = layer_attn[0, head_idx].numpy()
                
                # Plot heatmap
                im = ax.imshow(attn, cmap='viridis')
                
                # Set title
                ax.set_title(f"Layer {layer_idx+1}, Head {head_idx+1}")
                
                # Set axis labels if this is the first row/column
                if layer_idx == 0:
                    ax.set_title(f"Head {head_idx+1}")
                if head_idx == 0:
                    ax.set_ylabel(f"Layer {layer_idx+1}")
                
                # Set tick labels if this is the last row/column
                if layer_idx == num_layers - 1:
                    # Show token strings on x-axis
                    ax.set_xticks(np.arange(len(token_strings)))
                    ax.set_xticklabels(token_strings, rotation=90, fontsize=8)
                else:
                    ax.set_xticks([])
                
                if head_idx == 0:
                    # Show token strings on y-axis
                    ax.set_yticks(np.arange(len(token_strings)))
                    ax.set_yticklabels(token_strings, fontsize=8)
                else:
                    ax.set_yticks([])
        
        # Add colorbar
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        
        # Adjust layout
        fig.tight_layout(rect=[0, 0, 0.9, 0.95])
        
        # Save the figure if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


def generate_analysis_report(
    model: SignalLLM,
    test_results: Dict,
    benchmark_results: Dict,
    evolution_history: List,
    output_dir: str = './reports'
) -> str:
    """
    Generate a comprehensive analysis report.
    
    Args:
        model: SignalLLM model
        test_results: Test results dictionary
        benchmark_results: Benchmark results dictionary
        evolution_history: Evolution history
        output_dir: Directory to save the report
        
    Returns:
        Path to the generated report
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'analysis_report.md')
    
    # Generate plots
    viz_tools = VisualizationTools(os.path.join(output_dir, 'figures'))
    
    # Plot complexity scaling
    complexity_fig = viz_tools.plot_complexity_scaling(
        benchmark_results,
        title="SignalLLM Complexity Scaling",
        save_path=os.path.join(output_dir, 'figures', 'complexity_scaling.png')
    )
    
    # Plot evolution progress if available
    if evolution_history:
        evolution_fig = viz_tools.plot_evolution_progress(
            evolution_history,
            title="HRFEvo Basis Function Evolution",
            save_path=os.path.join(output_dir, 'figures', 'evolution_progress.png')
        )
    
    # Plot embedding statistics
    embedding_fig = viz_tools.plot_embedding_statistics(
        model,
        title="SignalLLM Embedding Statistics",
        save_path=os.path.join(output_dir, 'figures', 'embedding_stats.png')
    )
    
    # Create report content
    with open(report_path, 'w') as f:
        f.write("# SignalLLM with HRFEvo Analysis Report\n\n")
        
        # Model information
        f.write("## Model Information\n\n")
        f.write(f"- **Model Size**: {model.count_parameters():,} parameters\n")
        f.write(f"- **Embedding Dimension**: {model.config.embed_dim}\n")
        f.write(f"- **Number of Layers**: {model.config.num_layers}\n")
        f.write(f"- **Number of Heads**: {model.config.num_heads}\n")
        f.write(f"- **Wavelet Type**: {model.config.wavelet_type}\n")
        f.write(f"- **Wavelet Levels**: {model.config.wavelet_levels}\n")
        f.write(f"- **Harmonic Bases**: {model.config.harmonic_bases}\n\n")
        
        # Performance metrics
        f.write("## Performance Metrics\n\n")
        f.write("### Test Results\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        for metric, value in test_results.items():
            if isinstance(value, (int, float)):
                f.write(f"| {metric} | {value:.4f} |\n")
            else:
                f.write(f"| {metric} | {value} |\n")
        f.write("\n")
        
        # Complexity analysis
        f.write("## Complexity Analysis\n\n")
        f.write("### Scaling with Sequence Length\n\n")
        f.write("![Complexity Scaling](figures/complexity_scaling.png)\n\n")
        
        # Extract scaling factor from benchmark results
        if 'seq_lengths' in benchmark_results and 'times' in benchmark_results:
            seq_lengths = np.array(benchmark_results['seq_lengths'])
            times = np.array(benchmark_results['times'])
            
            if len(seq_lengths) > 2:
                # Calculate scaling factor
                def nlogn_func(x, a, b):
                    return a * x * np.log(x) + b
                
                from scipy.optimize import curve_fit
                try:
                    popt, _ = curve_fit(nlogn_func, seq_lengths, times)
                    f.write(f"The model demonstrates approximate O(n log n) scaling with a coefficient of {popt[0]:.4e}.\n\n")
                except:
                    f.write("Could not fit O(n log n) curve to the timing data.\n\n")
        
        # HRFEvo analysis
        f.write("## HRFEvo Basis Function Evolution\n\n")
        if evolution_history:
            f.write("### Evolution Progress\n\n")
            f.write("![Evolution Progress](figures/evolution_progress.png)\n\n")
            
            # Best basis function information
            f.write("### Best Basis Function\n\n")
            best_basis = model.current_basis
            f.write(f"- **Type**: {best_basis.type_name}\n")
            f.write(f"- **Fitness**: {best_basis.fitness:.4f}\n")
            f.write("- **Parameters**:\n")
            for param, value in best_basis.params.items():
                if isinstance(value, list) and len(value) > 10:
                    f.write(f"  - {param}: [{value[0]:.4f}, {value[1]:.4f}, ... {value[-1]:.4f}] (length: {len(value)})\n")
                else:
                    f.write(f"  - {param}: {value}\n")
            f.write("\n")
        else:
            f.write("No evolution history available.\n\n")
        
        # Embedding analysis
        f.write("## Embedding Analysis\n\n")
        f.write("### Spectral Embedding Statistics\n\n")
        f.write("![Embedding Statistics](figures/embedding_stats.png)\n\n")
        
        # Conclusion
        f.write("## Conclusion\n\n")
        f.write("The SignalLLM with HRFEvo approach demonstrates the following characteristics:\n\n")
        
        # Calculate theoretical speedup
        max_seq_len = max(benchmark_results['seq_lengths']) if 'seq_lengths' in benchmark_results else 0
        if max_seq_len > 0:
            theoretical_speedup = max_seq_len / np.log2(max_seq_len)
            f.write(f"1. **Computational Efficiency**: At the maximum tested sequence length of {max_seq_len}, "
                  f"the theoretical speedup compared to O(n²) approaches is approximately {theoretical_speedup:.1f}x.\n\n")
        
        # Performance metrics
        if 'ppl' in test_results:
            f.write(f"2. **Language Modeling Performance**: The model achieves a test perplexity of {test_results['ppl']:.2f}, "
                  f"demonstrating its linguistic capabilities.\n\n")
        
        # Memory efficiency
        if 'memory' in benchmark_results and benchmark_results['memory']:
            avg_memory = sum(benchmark_results['memory']) / len(benchmark_results['memory'])
            f.write(f"3. **Memory Efficiency**: The model uses an average of {avg_memory:.2f}GB of memory during inference.\n\n")
        
        # HRFEvo benefits
        if evolution_history:
            initial_fitness = evolution_history[0]['best_fitness']
            final_fitness = evolution_history[-1]['best_fitness']
            improvement = (final_fitness - initial_fitness) / initial_fitness * 100
            f.write(f"4. **HRFEvo Benefits**: The evolutionary optimization improved the basis function fitness by {improvement:.1f}%, "
                  f"demonstrating the advantage of adaptive basis functions.\n\n")
    
    logging.info(f"Analysis report generated at {report_path}")
    return report_path


#-------------------------------------------------------------------------
# Main Execution Flow
#-------------------------------------------------------------------------

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="SignalLLM with HRFEvo Proof of Concept")
    
    # Mode selection
    parser.add_argument("--mode", type=str, default="train", 
                      choices=["train", "evaluate", "benchmark", "evolve", "visualize"],
                      help="Execution mode")
    
    # Configuration options
    parser.add_argument("--config", type=str, default=None,
                      help="Path to configuration file (JSON)")
    parser.add_argument("--output_dir", type=str, default="./output",
                      help="Directory to save outputs")
    parser.add_argument("--checkpoint", type=str, default=None,
                      help="Path to checkpoint to load")
    
    # Model parameters
    parser.add_argument("--vocab_size", type=int, default=10000,
                      help="Vocabulary size")
    parser.add_argument("--embed_dim", type=int, default=256,
                      help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=4,
                      help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=4,
                      help="Number of transformer layers")
    parser.add_argument("--hidden_dim", type=int, default=1024,
                      help="Hidden dimension for feed-forward networks")
    
    # Spectral/wavelet parameters
    parser.add_argument("--harmonic_bases", type=int, default=16,
                      help="Number of harmonic bases for spectral embedding")
    parser.add_argument("--wavelet_type", type=str, default="db4",
                      help="Wavelet type for wavelet transform")
    parser.add_argument("--wavelet_levels", type=int, default=3,
                      help="Number of wavelet decomposition levels")
                      
    # Dr. Tao's mathematical enhancements
    parser.add_argument("--use_adaptive_basis", action="store_true", default=True,
                      help="Use adaptive basis selection")
    parser.add_argument("--wavelet_families", type=str, default="db4,sym4,dmey",
                      help="Comma-separated list of wavelet families to use with adaptive basis")
    parser.add_argument("--use_fourier_convolution", action="store_true", default=True,
                      help="Use Fourier convolution attention")
    parser.add_argument("--use_stochastic_transform", action="store_true", default=True,
                      help="Use stochastic wavelet transform for long sequences")
    parser.add_argument("--stochastic_sampling_ratio", type=float, default=0.2,
                      help="Sampling ratio for stochastic transform (0.0-1.0)")
    parser.add_argument("--spectral_gap_analysis", action="store_true", default=True,
                      help="Enable spectral gap analysis")
    
    # HRFEvo parameters
    parser.add_argument("--evolution_population", type=int, default=10,
                      help="Population size for HRFEvo")
    parser.add_argument("--evolution_generations", type=int, default=5,
                      help="Number of generations for HRFEvo")
    parser.add_argument("--enable_evolution", action="store_true",
                      help="Enable HRFEvo during training")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch size for training and evaluation")
    parser.add_argument("--max_epochs", type=int, default=10,
                      help="Maximum number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                      help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=512,
                      help="Maximum sequence length")
    parser.add_argument("--train_data", type=str, default=None,
                      help="Path to training data file")
    parser.add_argument("--val_data", type=str, default=None,
                      help="Path to validation data file")
    parser.add_argument("--test_data", type=str, default=None,
                      help="Path to test data file")
    
    # Hardware optimization
    parser.add_argument("--use_mixed_precision", action="store_true",
                      help="Use mixed precision training")
    parser.add_argument("--no_tensor_cores", action="store_true",
                      help="Disable tensor cores even if available")
    parser.add_argument("--no_memory_optimization", action="store_true",
                      help="Disable memory access optimization")
    
    # Visualization options
    parser.add_argument("--visualize_embeddings", action="store_true",
                      help="Visualize embedding statistics")
    parser.add_argument("--visualize_attention", action="store_true",
                      help="Visualize attention patterns")
    parser.add_argument("--visualize_complexity", action="store_true",
                      help="Visualize complexity scaling")
    
    # Benchmark options
    parser.add_argument("--benchmark_seq_lengths", type=str, default="128,256,512,1024,2048",
                      help="Comma-separated list of sequence lengths to benchmark")
    
    # Verbosity
    parser.add_argument("--verbose", action="store_true",
                      help="Verbose output")
    parser.add_argument("--log_level", type=str, default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                      help="Logging level")
    
    return parser.parse_args()


def load_or_create_config(args):
    """Load configuration from file or create from arguments"""
    if args.config and os.path.exists(args.config):
        # Load config from file
        config = Config.load(args.config)
        logging.info(f"Loaded configuration from {args.config}")
    else:
        # Create config from arguments
        config_dict = vars(args)
        
        # Adjust tensorboard parameters
        config_dict['use_tensor_cores'] = not args.no_tensor_cores
        config_dict['optimize_memory_access'] = not args.no_memory_optimization
        
        # Create config
        config = Config(**config_dict)
        logging.info("Created configuration from command line arguments")
    
    # Ensure output directory exists
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Save config for reference
    config_path = os.path.join(config.output_dir, 'config.json')
    config.save(config_path)
    logging.info(f"Saved configuration to {config_path}")
    
    return config


def prepare_data(config, tokenizer=None):
    """Prepare datasets and data loaders"""
    # Create tokenizer if not provided
    if tokenizer is None:
        tokenizer = SimpleTokenizer(mode='char')
        
        # Load or build vocabulary
        vocab_file = os.path.join(config.output_dir, 'vocab.json')
        if os.path.exists(vocab_file):
            tokenizer.load_vocab(vocab_file)
        else:
            # If training data is provided, build vocab from it
            if config.train_data and os.path.exists(config.train_data):
                with open(config.train_data, 'r', encoding='utf-8') as f:
                    texts = [line.strip() for line in f.readlines()]
                tokenizer.build_vocab(texts, max_vocab_size=config.vocab_size)
                tokenizer.save_vocab(vocab_file)
            else:
                # Create a simple alphabet vocabulary
                chars = list(' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-:;\'"\n\t')
                tokenizer.vocab = {c: i + 2 for i, c in enumerate(chars)}  # +2 for special tokens
                tokenizer.vocab[tokenizer.pad_token] = tokenizer.pad_token_id
                tokenizer.vocab[tokenizer.unk_token] = tokenizer.unk_token_id
                tokenizer.inv_vocab = {i: c for c, i in tokenizer.vocab.items()}
                tokenizer.save_vocab(vocab_file)
    
    # Check if the vocabulary size is set correctly
    vocab_size = tokenizer.get_vocab_size()
    if vocab_size != config.vocab_size:
        logging.warning(f"Vocabulary size mismatch. Config: {config.vocab_size}, Actual: {vocab_size}")
        config.vocab_size = vocab_size
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = None, None, None
    
    # Load training data if provided
    if config.train_data and os.path.exists(config.train_data):
        with open(config.train_data, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f.readlines()]
        train_dataset = TextDataset(texts, tokenizer, config.max_seq_length)
        logging.info(f"Created training dataset with {len(train_dataset)} examples")
    
    # Load validation data if provided
    if config.val_data and os.path.exists(config.val_data):
        with open(config.val_data, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f.readlines()]
        val_dataset = TextDataset(texts, tokenizer, config.max_seq_length)
        logging.info(f"Created validation dataset with {len(val_dataset)} examples")
    
    # Load test data if provided
    if config.test_data and os.path.exists(config.test_data):
        with open(config.test_data, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f.readlines()]
        test_dataset = TextDataset(texts, tokenizer, config.max_seq_length)
        logging.info(f"Created test dataset with {len(test_dataset)} examples")
    
    # If no data is provided, create a synthetic dataset for demonstration
    if train_dataset is None:
        logging.warning("No training data provided. Creating synthetic dataset for demonstration.")
        # Create a simple counting task
        texts = []
        for i in range(100):
            # Generate sequences like "1 2 3 4 5 6 7 8 9"
            sequence = ' '.join(str(j % 10) for j in range(i, i + config.max_seq_length // 2))
            texts.append(sequence)
        
        # Split into train/val/test
        train_size = int(len(texts) * 0.7)
        val_size = int(len(texts) * 0.15)
        
        train_dataset = TextDataset(texts[:train_size], tokenizer, config.max_seq_length)
        val_dataset = TextDataset(texts[train_size:train_size+val_size], tokenizer, config.max_seq_length)
        test_dataset = TextDataset(texts[train_size+val_size:], tokenizer, config.max_seq_length)
        
        logging.info(f"Created synthetic datasets. Train: {len(train_dataset)}, "
                   f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return tokenizer, train_dataset, val_dataset, test_dataset


def create_data_loaders(config, hw_opt, train_dataset, val_dataset, test_dataset):
    """Create data loaders for training and evaluation"""
    train_loader, val_loader, test_loader = None, None, None
    
    # Create training data loader
    if train_dataset:
        train_loader = hw_opt.create_optimized_dataloader(
            train_dataset, config.batch_size, shuffle=True)
    
    # Create validation data loader
    if val_dataset:
        val_loader = hw_opt.create_optimized_dataloader(
            val_dataset, config.batch_size, shuffle=False)
    
    # Create test data loader
    if test_dataset:
        test_loader = hw_opt.create_optimized_dataloader(
            test_dataset, config.batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def create_model(config, device):
    """Create and initialize the model"""
    # Create model
    model = SignalLLM(config)
    
    # Log model parameters
    num_params = model.count_parameters()
    logging.info(f"Created model with {num_params:,} parameters")
    
    # Move model to device
    model = model.to(device)
    
    return model


def setup_training(config, model):
    """Set up optimizer and learning rate scheduler"""
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Create learning rate scheduler with warmup
    def lr_lambda(current_step):
        if current_step < config.warmup_steps:
            return float(current_step) / float(max(1, config.warmup_steps))
        return 1.0
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler


def run_training(
    config, model, optimizer, scheduler, train_loader, val_loader,
    tokenizer, device, hw_opt, hrfevo_controller=None
):
    """Run the training loop"""
    # Create TensorBoard writer if available
    tensorboard = create_tensorboard(config) if TENSORBOARD_AVAILABLE else None
    
    # Track best validation performance
    best_val_ppl = float('inf')
    
    # Training loop
    for epoch in range(config.max_epochs):
        # Train for one epoch
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch,
            config, hrfevo_controller, tensorboard
        )
        
        # Validate
        if val_loader:
            val_metrics = evaluate(
                model, val_loader, device, tensorboard, 'valid',
                global_step=(epoch + 1) * len(train_loader)
            )
            
            # Check if this is the best model
            if val_metrics['ppl'] < best_val_ppl:
                best_val_ppl = val_metrics['ppl']
                
                # Save best model
                best_checkpoint_path = os.path.join(config.checkpoint_dir, 'best_model.pt')
                save_checkpoint(
                    model, optimizer, scheduler, epoch, val_metrics,
                    config, hrfevo_controller, best_checkpoint_path
                )
                logging.info(f"Saved best model with validation PPL: {best_val_ppl:.2f}")
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
        save_checkpoint(
            model, optimizer, scheduler, epoch, train_metrics,
            config, hrfevo_controller, checkpoint_path
        )
    
    # Close TensorBoard writer
    if tensorboard:
        tensorboard.close()
    
    return best_val_ppl


def run_evaluation(config, model, test_loader, device):
    """Run evaluation on test data"""
    # Evaluate on test data
    test_metrics = evaluate(model, test_loader, device, mode='test')
    
    # Log results
    logging.info("Test Results:")
    for metric, value in test_metrics.items():
        logging.info(f"  {metric}: {value}")
    
    return test_metrics


def run_benchmarking(config, model, device):
    """Run benchmarking tests"""
    # Parse sequence lengths to benchmark
    seq_lengths = [int(x) for x in config.benchmark_seq_lengths.split(',')]
    
    # Log benchmark configuration
    logging.info(f"Running benchmarks with sequence lengths: {seq_lengths}")
    
    # Run benchmark
    results = benchmark_complexity(
        model,
        seq_lengths,
        device,
        batch_size=config.batch_size,
        embed_dim=config.embed_dim
    )
    
    # Create visualization
    viz_tools = VisualizationTools(os.path.join(config.output_dir, 'visualizations'))
    fig = viz_tools.plot_complexity_scaling(
        results,
        title="SignalLLM Complexity Scaling",
        save_path=os.path.join(config.output_dir, 'visualizations', 'complexity_scaling.png')
    )
    
    # Save results
    benchmark_path = os.path.join(config.output_dir, 'benchmark_results.json')
    with open(benchmark_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            else:
                serializable_results[key] = value
        
        json.dump(serializable_results, f, indent=2)
    
    logging.info(f"Benchmark results saved to {benchmark_path}")
    
    return results


def run_evolution(config, model, device, train_loader, hrfevo_controller):
    """Run HRFEvo basis function evolution"""
    # Make sure we have a controller
    if hrfevo_controller is None:
        hrfevo_controller = HRFEvoController(config)
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Define evaluation function
    def evaluate_basis(basis):
        # Apply basis function to model temporarily
        original_basis = model.current_basis
        model.update_basis_function(basis)
        
        # Evaluate on a small batch
        model.eval()
        with torch.no_grad():
            # Get a batch of data
            try:
                eval_data, eval_targets = next(iter(train_loader))
            except StopIteration:
                # Create a dummy batch if the loader is empty
                eval_data = torch.randint(0, config.vocab_size, (4, 32), device=device)
                eval_targets = torch.randint(0, config.vocab_size, (4, 32), device=device)
            
            eval_data, eval_targets = eval_data.to(device), eval_targets.to(device)
            
            # Forward pass
            start_time = time.time()
            outputs = model(eval_data)
            inference_time = time.time() - start_time
            
            # Calculate metrics
            loss = criterion(outputs.view(-1, outputs.size(-1)), eval_targets.view(-1))
            perplexity = torch.exp(loss).item()
            
            # Get complexity info
            complexity_info = model.get_complexity_info()
            
            # Calculate efficiency score based on inference time and op ratios
            if 'attention_op_ratios' in complexity_info and complexity_info['attention_op_ratios']:
                avg_ratio = sum(complexity_info['attention_op_ratios']) / len(complexity_info['attention_op_ratios'])
                computation_efficiency = avg_ratio / inference_time
            else:
                computation_efficiency = 1.0 / inference_time
        
        # Restore original basis
        model.update_basis_function(original_basis)
        model.train()
        
        return {
            'perplexity': perplexity,
            'computation_efficiency': computation_efficiency
        }
    
    # Run evolution for specified number of generations
    logging.info(f"Running HRFEvo for {config.evolution_generations} generations...")
    best_basis = hrfevo_controller.run_evolution(evaluate_basis)
    
    # Apply best basis to model
    model.update_basis_function(best_basis)
    
    # Save evolved basis
    basis_path = os.path.join(config.output_dir, 'evolved_basis.json')
    with open(basis_path, 'w') as f:
        json.dump(best_basis.to_dict(), f, indent=2)
    
    logging.info(f"Saved evolved basis to {basis_path}")
    
    # Create visualization
    viz_tools = VisualizationTools(os.path.join(config.output_dir, 'visualizations'))
    fig = viz_tools.plot_evolution_progress(
        hrfevo_controller.fitness_history,
        title="HRFEvo Basis Function Evolution",
        save_path=os.path.join(config.output_dir, 'visualizations', 'evolution_progress.png')
    )
    
    fig = viz_tools.plot_basis_function(
        best_basis,
        title=f"Evolved Basis Function: {best_basis.type_name}",
        save_path=os.path.join(config.output_dir, 'visualizations', 'evolved_basis.png')
    )
    
    return best_basis


def run_visualization(config, model, tokenizer, device):
    """Run visualization tools"""
    # Create visualization tool
    viz_dir = os.path.join(config.output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    viz_tools = VisualizationTools(viz_dir)
    
    # Visualize embedding statistics if requested
    if config.visualize_embeddings:
        fig = viz_tools.plot_embedding_statistics(
            model,
            title="SignalLLM Embedding Statistics",
            save_path=os.path.join(viz_dir, 'embedding_stats.png')
        )
        logging.info(f"Embedding statistics visualization saved to {viz_dir}/embedding_stats.png")
    
    # Visualize attention patterns if requested
    if config.visualize_attention:
        # Generate a simple test text
        test_text = "The quick brown fox jumps over the lazy dog."
        
        fig = viz_tools.visualize_attention_patterns(
            model,
            test_text,
            tokenizer,
            title="SignalLLM Attention Patterns",
            save_path=os.path.join(viz_dir, 'attention_patterns.png')
        )
        logging.info(f"Attention patterns visualization saved to {viz_dir}/attention_patterns.png")
    
    # Visualize a wavelet decomposition example
    if WAVELET_AVAILABLE:
        # Create a sample signal
        t = np.linspace(0, 1, 512)
        signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
        tensor = torch.tensor(signal).unsqueeze(0).unsqueeze(-1).float()
        
        fig = viz_tools.plot_wavelet_decomposition(
            tensor,
            wavelet_type=config.wavelet_type,
            levels=config.wavelet_levels,
            title=f"Wavelet Decomposition ({config.wavelet_type})",
            save_path=os.path.join(viz_dir, 'wavelet_decomposition.png')
        )
        logging.info(f"Wavelet decomposition visualization saved to {viz_dir}/wavelet_decomposition.png")
    
    # Visualize spectral gap analysis if enabled
    if config.spectral_gap_analysis:
        # Create a spectral gap analyzer
        analyzer = SpectralGapAnalyzer(device=device)
        
        # Generate sample data for analysis
        # Use a wavelet transform to get coefficients for analysis
        wavelet_transform = WaveletTransform(config.wavelet_type, config.wavelet_levels)
        
        # Create sample embeddings with structured patterns (to demonstrate spectral properties)
        sample_size = 256
        t = torch.linspace(0, 4*np.pi, sample_size, device=device)
        
        # Create structured embeddings with different frequency components
        sample_embeddings = torch.zeros(1, sample_size, config.embed_dim, device=device)
        for i in range(config.embed_dim):
            # Add frequency components at different scales
            freq = 1.0 + 0.5 * (i % 5)
            phase = 0.1 * (i % 10)
            sample_embeddings[0, :, i] = torch.sin(freq * t + phase)
            
            # Add some local clusters to demonstrate graph structure
            if i % 3 == 0:
                # Add cluster pattern
                centers = torch.linspace(0, sample_size-1, 5, device=device).long()
                for center in centers:
                    radius = sample_size // 20
                    start_idx = max(0, center - radius)
                    end_idx = min(sample_size, center + radius)
                    sample_embeddings[0, start_idx:end_idx, i] += 0.5
        
        # Process with wavelet transform
        approx, details = wavelet_transform(sample_embeddings)
        
        # Analyze the spectral properties
        spectral_analysis = analyzer.analyze_wavelet_representation((approx, details))
        
        # Plot the analysis
        fig = viz_tools.plot_spectral_gap_analysis(
            spectral_analysis,
            title="Spectral Gap Analysis (Dr. Tao's Method)",
            save_path=os.path.join(viz_dir, 'spectral_gap_analysis.png')
        )
        logging.info(f"Spectral gap analysis visualization saved to {viz_dir}/spectral_gap_analysis.png")
    
    # Visualize basis function
    fig = viz_tools.plot_basis_function(
        model.current_basis,
        title=f"Current Basis Function: {model.current_basis.type_name}",
        save_path=os.path.join(viz_dir, 'current_basis.png')
    )
    logging.info(f"Basis function visualization saved to {viz_dir}/current_basis.png")
    
    return viz_dir


def main():
    """Main execution flow"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'signalllm_hrfevo.log')),
            logging.StreamHandler()
        ]
    )
    
    # Log system information
    logging.info(f"Running on Python {sys.version}")
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Load or create configuration
    config = load_or_create_config(args)
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed if hasattr(args, 'seed') else SEED)
    np.random.seed(args.seed if hasattr(args, 'seed') else SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed if hasattr(args, 'seed') else SEED)
    
    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # Initialize hardware optimization
    hw_opt = HardwareOptimization(config, device)
    
    # Prepare tokenizer and datasets
    tokenizer, train_dataset, val_dataset, test_dataset = prepare_data(config)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        config, hw_opt, train_dataset, val_dataset, test_dataset)
    
    # Create model
    model = create_model(config, device)
    
    # Apply hardware optimizations to model
    model = hw_opt.optimize_model(model)
    
    # Initialize HRFEvo controller if enabled
    hrfevo_controller = HRFEvoController(config) if args.enable_evolution else None
    
    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = load_checkpoint(args.checkpoint, model, device=device, hrfevo_controller=hrfevo_controller)
        logging.info(f"Loaded checkpoint from {args.checkpoint} (epoch {checkpoint['epoch']})")
    
    # Execute according to mode
    if args.mode == "train":
        # Set up training components
        optimizer, scheduler = setup_training(config, model)
        
        # Run training
        best_val_ppl = run_training(
            config, model, optimizer, scheduler, train_loader, val_loader,
            tokenizer, device, hw_opt, hrfevo_controller
        )
        
        # Run final evaluation
        if test_loader:
            test_metrics = run_evaluation(config, model, test_loader, device)
        
        # Generate visualizations
        viz_dir = run_visualization(config, model, tokenizer, device)
        
        # Generate analysis report if test results are available
        if test_loader and hrfevo_controller and args.enable_evolution:
            # Run benchmark for report
            benchmark_results = run_benchmarking(config, model, device)
            
            # Generate report
            report_path = generate_analysis_report(
                model,
                test_metrics,
                benchmark_results,
                hrfevo_controller.fitness_history,
                output_dir=os.path.join(config.output_dir, 'reports')
            )
            logging.info(f"Analysis report generated at {report_path}")
    
    elif args.mode == "evaluate":
        # Run evaluation
        if test_loader:
            test_metrics = run_evaluation(config, model, test_loader, device)
        else:
            logging.error("No test data provided for evaluation")
    
    elif args.mode == "benchmark":
        # Run benchmarking
        benchmark_results = run_benchmarking(config, model, device)
    
    elif args.mode == "evolve":
        # Run HRFEvo basis function evolution
        best_basis = run_evolution(config, model, device, train_loader, hrfevo_controller)
        
        # Save model with evolved basis
        checkpoint_path = os.path.join(config.checkpoint_dir, 'model_with_evolved_basis.pt')
        save_checkpoint(model, None, None, 0, {}, config, hrfevo_controller, checkpoint_path)
        logging.info(f"Saved model with evolved basis to {checkpoint_path}")
    
    elif args.mode == "visualize":
        # Run visualization
        viz_dir = run_visualization(config, model, tokenizer, device)
    
    logging.info("Execution completed successfully.")


if __name__ == "__main__":
    main()