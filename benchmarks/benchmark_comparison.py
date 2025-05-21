#!/usr/bin/env python3

"""
SignalLLM Benchmarking Tool
===========================
This script provides comprehensive benchmarks comparing SignalLLM against
traditional transformer architectures across various dimensions:

1. Computational complexity (time and FLOPS)
2. Memory usage
3. Parameter efficiency
4. Performance on NLP tasks
"""

import os
import time
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# Import SignalLLM components
from signalllm_hrfevo_poc import (
    Config, SignalLLM, FrequencyDomainAttention, FourierConvolutionAttention,
    SpectralEmbedding, SpectralGapAnalyzer, SimpleTokenizer, count_parameters
)

# Memory profiling
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil not available. Memory usage benchmarks will be limited.")

try:
    import torch.cuda as cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("CUDA not available. Using CPU only.")

# Basic implementation of a standard Transformer for comparison
class StandardTransformer(nn.Module):
    """Standard transformer implementation for benchmarking comparison"""
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, hidden_dim, max_seq_length, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_length, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_layer = nn.Linear(embed_dim, vocab_size)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        # Initialize position encodings
        position = torch.arange(0, self.pos_encoding.size(1), dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() * (-np.log(10000.0) / self.embed_dim))
        
        # Create a temporary tensor and then assign to pos_encoding to avoid in-place operation on leaf variable
        pos_encoding = torch.zeros_like(self.pos_encoding.data)
        pos_encoding[0, :, 0::2] = torch.sin(position * div_term)
        pos_encoding[0, :, 1::2] = torch.cos(position * div_term)
        self.pos_encoding.data.copy_(pos_encoding)
        
        # Alternatively, make pos_encoding not require gradients
        # self.pos_encoding = nn.Parameter(self.pos_encoding, requires_grad=False)
    
    def forward(self, x):
        # Create a mask to prevent attention to padding tokens
        mask = None
        if isinstance(x, tuple) and len(x) == 2:
            x, mask = x
        
        # Embedding and positional encoding
        embedded = self.embedding(x)
        seq_length = embedded.size(1)
        embedded = embedded + self.pos_encoding[:, :seq_length, :]
        
        # Process through transformer
        if mask is not None:
            output = self.transformer_encoder(embedded, src_key_padding_mask=mask)
        else:
            output = self.transformer_encoder(embedded)
        
        # Final linear layer
        logits = self.output_layer(output)
        return logits

def benchmark_computational_complexity(model_type, seq_lengths, vocab_size=1000, embed_dim=256, num_heads=4, 
                                     num_layers=4, hidden_dim=1024, batch_size=32, device=None, num_runs=5):
    """
    Benchmark computational complexity by measuring execution time across different sequence lengths
    
    Args:
        model_type: Either "signal" (for SignalLLM) or "standard" (for traditional Transformer)
        seq_lengths: List of sequence lengths to test
        vocab_size: Vocabulary size
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        hidden_dim: Hidden dimension for feed-forward layers
        batch_size: Batch size
        device: Computing device
        num_runs: Number of runs for averaging
        
    Returns:
        Dictionary with benchmark results
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else 
                            "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else
                            "cpu")
    print(f"Benchmarking on device: {device}")
    
    # Results storage
    results = {
        "model_type": model_type,
        "seq_lengths": seq_lengths,
        "execution_times": [],
        "theoretical_flops": []
    }
    
    for seq_len in seq_lengths:
        print(f"\nTesting sequence length: {seq_len}")
        
        # Create model based on type
        if model_type == "signal":
            config = Config(
                vocab_size=vocab_size,
                max_seq_length=seq_len,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                harmonic_bases=16
            )
            model = SignalLLM(config).to(device)
        else:  # standard
            model = StandardTransformer(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                max_seq_length=seq_len
            ).to(device)
        
        # Create input data
        inputs = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model(inputs)
            torch.cuda.synchronize() if device.type == 'cuda' else None
        
        # Measure execution time
        execution_times = []
        for _ in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                _ = model(inputs)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            execution_times.append(time.time() - start_time)
        
        avg_time = sum(execution_times) / len(execution_times)
        results["execution_times"].append(avg_time)
        
        # Calculate theoretical FLOPS
        if model_type == "signal":
            # O(n log n) complexity for SignalLLM
            theoretical_flops = num_layers * batch_size * seq_len * np.log2(seq_len) * embed_dim
        else:
            # O(n²) complexity for standard transformer
            theoretical_flops = num_layers * batch_size * seq_len * seq_len * embed_dim
        
        results["theoretical_flops"].append(theoretical_flops)
        
        print(f"Average execution time: {avg_time:.4f}s")
        print(f"Theoretical FLOPS: {theoretical_flops:,}")
        
        # Clean up to free memory
        del model
        torch.cuda.empty_cache() if device.type == 'cuda' else None
    
    return results

def benchmark_memory_usage(model_type, seq_lengths, vocab_size=1000, embed_dim=256, num_heads=4, 
                         num_layers=4, hidden_dim=1024, batch_size=32, device=None):
    """
    Benchmark memory usage across different sequence lengths
    
    Args:
        model_type: Either "signal" or "standard"
        seq_lengths: List of sequence lengths to test
        vocab_size: Vocabulary size
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        hidden_dim: Hidden dimension for feed-forward layers
        batch_size: Batch size
        device: Computing device
        
    Returns:
        Dictionary with benchmark results
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else 
                            "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else
                            "cpu")
    print(f"Benchmarking memory usage on device: {device}")
    
    # Results storage
    results = {
        "model_type": model_type,
        "seq_lengths": seq_lengths,
        "peak_memory_mb": [],
        "model_size_mb": []
    }
    
    for seq_len in seq_lengths:
        print(f"\nTesting sequence length: {seq_len}")
        
        # Clear cache before testing
        torch.cuda.empty_cache() if device.type == 'cuda' else None
        
        # Create model based on type
        if model_type == "signal":
            config = Config(
                vocab_size=vocab_size,
                max_seq_length=seq_len,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                harmonic_bases=16
            )
            model = SignalLLM(config).to(device)
        else:  # standard
            model = StandardTransformer(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                max_seq_length=seq_len
            ).to(device)
        
        # Measure model size
        model_size_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
        model_size_mb = model_size_bytes / (1024 * 1024)
        results["model_size_mb"].append(model_size_mb)
        
        # Create input data
        inputs = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        # Measure peak memory usage
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            _ = model(inputs)
            peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # Convert to MB
            results["peak_memory_mb"].append(peak_memory)
            print(f"Model size: {model_size_mb:.2f} MB")
            print(f"Peak memory usage: {peak_memory:.2f} MB")
        else:
            # For non-CUDA devices, just record model size
            results["peak_memory_mb"].append(None)
            print(f"Model size: {model_size_mb:.2f} MB")
            print("Peak memory tracking only available on CUDA devices")
        
        # Clean up
        del model
        torch.cuda.empty_cache() if device.type == 'cuda' else None
    
    return results

def benchmark_parameter_efficiency(vocab_sizes, embed_dim=256, harmonic_bases=16):
    """
    Benchmark parameter efficiency for embeddings across different vocabulary sizes
    
    Args:
        vocab_sizes: List of vocabulary sizes to test
        embed_dim: Embedding dimension
        harmonic_bases: Number of harmonic bases for spectral embedding
        
    Returns:
        Dictionary with benchmark results
    """
    print(f"Benchmarking parameter efficiency for embeddings")
    
    # Results storage
    results = {
        "vocab_sizes": vocab_sizes,
        "standard_params": [],
        "spectral_params": [],
        "reduction_ratio": []
    }
    
    for vocab_size in vocab_sizes:
        print(f"\nTesting vocabulary size: {vocab_size}")
        
        # Standard embedding
        standard_embed = nn.Embedding(vocab_size, embed_dim)
        standard_params = count_parameters(standard_embed)
        results["standard_params"].append(standard_params)
        
        # Spectral embedding
        spectral_embed = SpectralEmbedding(vocab_size, embed_dim, harmonic_bases=harmonic_bases)
        spectral_params = count_parameters(spectral_embed)
        results["spectral_params"].append(spectral_params)
        
        # Calculate reduction ratio
        ratio = standard_params / spectral_params
        results["reduction_ratio"].append(ratio)
        
        print(f"Standard embedding parameters: {standard_params:,}")
        print(f"Spectral embedding parameters: {spectral_params:,}")
        print(f"Parameter reduction ratio: {ratio:.2f}x")
    
    return results

def load_text_dataset(filepath, max_samples=1000):
    """Load a text dataset from a file"""
    if filepath is None or not os.path.exists(filepath):
        # Create a synthetic dataset if no file is provided
        logging.info("No dataset file provided. Creating synthetic dataset...")
        synthetic_texts = []
        
        # Generate simple patterns for the model to learn
        patterns = [
            "The quick brown fox jumps over the lazy dog.",
            "A journey of a thousand miles begins with a single step.",
            "To be or not to be, that is the question.",
            "All that glitters is not gold.",
            "The early bird catches the worm.",
            "Actions speak louder than words.",
            "You can't teach an old dog new tricks.",
            "Don't count your chickens before they hatch.",
            "The pen is mightier than the sword.",
            "When in Rome, do as the Romans do."
        ]
        
        # Repeat patterns with variations to create a larger dataset
        for i in range(max_samples):
            pattern = patterns[i % len(patterns)]
            
            # Add some variations
            if i % 3 == 0:
                synthetic_texts.append(pattern)
            elif i % 3 == 1:
                # Split and rejoin with slight modification
                words = pattern.split()
                if len(words) > 5:
                    mid = len(words) // 2
                    synthetic_texts.append(' '.join(words[:mid]) + ' really ' + ' '.join(words[mid:]))
                else:
                    synthetic_texts.append(pattern)
            else:
                # Reverse the pattern
                synthetic_texts.append(pattern[::-1])
        
        return synthetic_texts
    
    # Load from file if provided
    texts = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                texts.append(line.strip())
                if len(texts) >= max_samples:
                    break
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return []
    
    return texts

def train_and_evaluate(model_type, dataset, epochs=1, batch_size=32, seq_length=64, 
                      vocab_size=1000, embed_dim=256, num_heads=4, num_layers=2, 
                      hidden_dim=512, device=None):
    """
    Train and evaluate models on a text dataset
    
    Args:
        model_type: Either "signal" or "standard"
        dataset: List of text samples
        epochs: Number of training epochs
        batch_size: Batch size
        seq_length: Sequence length
        vocab_size: Vocabulary size (max)
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        hidden_dim: Hidden dimension for feed-forward layers
        device: Computing device
        
    Returns:
        Dictionary with evaluation results
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else 
                            "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else
                            "cpu")
    print(f"Training and evaluating on device: {device}")
    
    # Create tokenizer
    tokenizer = SimpleTokenizer(mode='char')
    tokenizer.build_vocab(dataset, max_vocab_size=vocab_size)
    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"Actual vocabulary size: {actual_vocab_size}")
    
    # Create model based on type
    if model_type == "signal":
        config = Config(
            vocab_size=actual_vocab_size,
            max_seq_length=seq_length,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            harmonic_bases=16
        )
        model = SignalLLM(config).to(device)
    else:  # standard
        model = StandardTransformer(
            vocab_size=actual_vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            max_seq_length=seq_length
        ).to(device)
    
    # Create simple dataset
    tokenized_data = []
    for text in dataset:
        tokens = tokenizer.encode(text[:seq_length*2])  # Ensure enough context
        if len(tokens) >= seq_length + 1:  # Need at least seq_length + 1 tokens
            tokenized_data.append(tokens[:seq_length + 1])  # Input + target
    
    # Create data batches
    num_batches = len(tokenized_data) // batch_size
    
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Results storage
    results = {
        "model_type": model_type,
        "train_loss": [],
        "eval_loss": [],
        "perplexity": [],
        "training_time": 0
    }
    
    # Training loop
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        for i in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}"):
            batch_data = tokenized_data[i * batch_size:(i + 1) * batch_size]
            
            # Create input-target pairs
            inputs = [data[:-1] for data in batch_data]
            targets = [data[1:] for data in batch_data]
            
            # Pad sequences
            max_len = max(len(seq) for seq in inputs)
            padded_inputs = [seq + [0] * (max_len - len(seq)) for seq in inputs]
            padded_targets = [seq + [0] * (max_len - len(seq)) for seq in targets]
            
            # Convert to tensors
            input_tensor = torch.tensor(padded_inputs, device=device)
            target_tensor = torch.tensor(padded_targets, device=device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(input_tensor)
            
            # Calculate loss
            loss = criterion(output.view(-1, actual_vocab_size), target_tensor.view(-1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Evaluation
        model.eval()
        eval_losses = []
        
        with torch.no_grad():
            for i in range(num_batches):
                batch_data = tokenized_data[i * batch_size:(i + 1) * batch_size]
                
                # Create input-target pairs
                inputs = [data[:-1] for data in batch_data]
                targets = [data[1:] for data in batch_data]
                
                # Pad sequences
                max_len = max(len(seq) for seq in inputs)
                padded_inputs = [seq + [0] * (max_len - len(seq)) for seq in inputs]
                padded_targets = [seq + [0] * (max_len - len(seq)) for seq in targets]
                
                # Convert to tensors
                input_tensor = torch.tensor(padded_inputs, device=device)
                target_tensor = torch.tensor(padded_targets, device=device)
                
                # Forward pass
                output = model(input_tensor)
                
                # Calculate loss
                loss = criterion(output.view(-1, actual_vocab_size), target_tensor.view(-1))
                eval_losses.append(loss.item())
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_eval_loss = sum(eval_losses) / len(eval_losses)
        perplexity = torch.exp(torch.tensor(avg_eval_loss)).item()
        
        results["train_loss"].append(avg_train_loss)
        results["eval_loss"].append(avg_eval_loss)
        results["perplexity"].append(perplexity)
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Eval Loss: {avg_eval_loss:.4f}")
        print(f"  Perplexity: {perplexity:.4f}")
    
    results["training_time"] = time.time() - start_time
    print(f"Total training time: {results['training_time']:.2f}s")
    
    # Clean up
    del model
    torch.cuda.empty_cache() if device.type == 'cuda' else None
    
    return results

def plot_benchmark_results(computational_results, memory_results, parameter_results, nlp_results, output_dir="./benchmarks"):
    """Plot benchmark results and save figures"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Plot computational complexity
    if computational_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Extract data for standard and signal models
        standard_data = next((data for data in computational_results if data["model_type"] == "standard"), None)
        signal_data = next((data for data in computational_results if data["model_type"] == "signal"), None)
        
        if standard_data and signal_data:
            seq_lengths = standard_data["seq_lengths"]
            
            # Plot execution times
            ax1.plot(seq_lengths, standard_data["execution_times"], 'o-', label="Standard Transformer")
            ax1.plot(seq_lengths, signal_data["execution_times"], 's-', label="SignalLLM")
            ax1.set_xlabel("Sequence Length")
            ax1.set_ylabel("Execution Time (s)")
            ax1.set_title("Execution Time vs. Sequence Length")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Calculate theoretical lines for reference
            x = np.array(seq_lengths)
            n2_line = x**2 / x[0]**2 * standard_data["execution_times"][0]
            nlogn_line = x * np.log2(x) / (x[0] * np.log2(x[0])) * signal_data["execution_times"][0]
            
            ax2.plot(seq_lengths, standard_data["execution_times"], 'o-', label="Standard Transformer")
            ax2.plot(seq_lengths, signal_data["execution_times"], 's-', label="SignalLLM")
            ax2.plot(seq_lengths, n2_line, '--', label="O(n²) reference", color='gray')
            ax2.plot(seq_lengths, nlogn_line, ':', label="O(n log n) reference", color='gray')
            ax2.set_xlabel("Sequence Length")
            ax2.set_ylabel("Execution Time (s)")
            ax2.set_title("Execution Time vs. Sequence Length (with Theoretical Curves)")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "computational_complexity.png"), dpi=300)
            plt.close()
    
    # 2. Plot memory usage
    if memory_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Extract data for standard and signal models
        standard_data = next((data for data in memory_results if data["model_type"] == "standard"), None)
        signal_data = next((data for data in memory_results if data["model_type"] == "signal"), None)
        
        if standard_data and signal_data:
            seq_lengths = standard_data["seq_lengths"]
            
            # Plot model size
            ax1.plot(seq_lengths, standard_data["model_size_mb"], 'o-', label="Standard Transformer")
            ax1.plot(seq_lengths, signal_data["model_size_mb"], 's-', label="SignalLLM")
            ax1.set_xlabel("Sequence Length")
            ax1.set_ylabel("Model Size (MB)")
            ax1.set_title("Model Size vs. Sequence Length")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot peak memory if available
            if all(p is not None for p in standard_data["peak_memory_mb"]) and all(p is not None for p in signal_data["peak_memory_mb"]):
                ax2.plot(seq_lengths, standard_data["peak_memory_mb"], 'o-', label="Standard Transformer")
                ax2.plot(seq_lengths, signal_data["peak_memory_mb"], 's-', label="SignalLLM")
                ax2.set_xlabel("Sequence Length")
                ax2.set_ylabel("Peak Memory Usage (MB)")
                ax2.set_title("Peak Memory Usage vs. Sequence Length")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "memory_usage.png"), dpi=300)
            plt.close()
    
    # 3. Plot parameter efficiency
    if parameter_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        vocab_sizes = parameter_results["vocab_sizes"]
        standard_params = parameter_results["standard_params"]
        spectral_params = parameter_results["spectral_params"]
        reduction_ratio = parameter_results["reduction_ratio"]
        
        # Plot parameter counts
        ax1.plot(vocab_sizes, standard_params, 'o-', label="Standard Embedding")
        ax1.plot(vocab_sizes, spectral_params, 's-', label="Spectral Embedding")
        ax1.set_xlabel("Vocabulary Size")
        ax1.set_ylabel("Parameter Count")
        ax1.set_title("Parameter Count vs. Vocabulary Size")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        
        # Plot reduction ratio
        ax2.bar(range(len(vocab_sizes)), reduction_ratio)
        ax2.set_xlabel("Vocabulary Size")
        ax2.set_ylabel("Parameter Reduction Ratio")
        ax2.set_title("Parameter Reduction Ratio")
        ax2.set_xticks(range(len(vocab_sizes)))
        ax2.set_xticklabels([f"{v:,}" for v in vocab_sizes])
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "parameter_efficiency.png"), dpi=300)
        plt.close()
    
    # 4. Plot NLP task performance
    if nlp_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Extract data for standard and signal models
        standard_data = next((data for data in nlp_results if data["model_type"] == "standard"), None)
        signal_data = next((data for data in nlp_results if data["model_type"] == "signal"), None)
        
        if standard_data and signal_data:
            epochs = range(1, len(standard_data["train_loss"]) + 1)
            
            # Plot training loss
            ax1.plot(epochs, standard_data["train_loss"], 'o-', label="Standard Transformer")
            ax1.plot(epochs, signal_data["train_loss"], 's-', label="SignalLLM")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Training Loss")
            ax1.set_title("Training Loss vs. Epoch")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot perplexity
            ax2.plot(epochs, standard_data["perplexity"], 'o-', label="Standard Transformer")
            ax2.plot(epochs, signal_data["perplexity"], 's-', label="SignalLLM")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Perplexity")
            ax2.set_title("Perplexity vs. Epoch")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "nlp_performance.png"), dpi=300)
            plt.close()
            
            # Create bar chart for training time
            fig, ax = plt.subplots(figsize=(10, 6))
            models = ["Standard Transformer", "SignalLLM"]
            times = [standard_data["training_time"], signal_data["training_time"]]
            
            ax.bar(models, times)
            ax.set_ylabel("Training Time (s)")
            ax.set_title("Total Training Time Comparison")
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "training_time.png"), dpi=300)
            plt.close()
    
    # Save consolidated results to a single HTML file with embedded plots
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SignalLLM Benchmark Results</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #2c3e50; }
            h2 { color: #3498db; }
            .result-section { margin-bottom: 30px; }
            .plot-container { margin: 20px 0; text-align: center; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f2f2f2; }
            tr:hover { background-color: #f5f5f5; }
        </style>
    </head>
    <body>
        <h1>SignalLLM Benchmark Results</h1>
        
        <div class="result-section">
            <h2>1. Computational Complexity</h2>
            <p>This section compares the computational complexity of SignalLLM (O(n log n)) with standard Transformers (O(n²)) across various sequence lengths.</p>
            <div class="plot-container">
                <img src="computational_complexity.png" alt="Computational Complexity" style="max-width: 100%;">
            </div>
        </div>
        
        <div class="result-section">
            <h2>2. Memory Usage</h2>
            <p>This section compares the memory efficiency of SignalLLM with standard Transformers.</p>
            <div class="plot-container">
                <img src="memory_usage.png" alt="Memory Usage" style="max-width: 100%;">
            </div>
        </div>
        
        <div class="result-section">
            <h2>3. Parameter Efficiency</h2>
            <p>This section demonstrates the parameter reduction achieved by spectral embeddings compared to standard embeddings.</p>
            <div class="plot-container">
                <img src="parameter_efficiency.png" alt="Parameter Efficiency" style="max-width: 100%;">
            </div>
        </div>
        
        <div class="result-section">
            <h2>4. NLP Task Performance</h2>
            <p>This section compares the language modeling performance of both architectures.</p>
            <div class="plot-container">
                <img src="nlp_performance.png" alt="NLP Task Performance" style="max-width: 100%;">
            </div>
            <div class="plot-container">
                <img src="training_time.png" alt="Training Time Comparison" style="max-width: 100%;">
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, "benchmark_results.html"), "w") as f:
        f.write(html_content)
    
    print(f"All benchmark results saved to {output_dir}")

def main():
    """Main function to run benchmarks"""
    parser = argparse.ArgumentParser(description="SignalLLM Benchmarking Tool")
    parser.add_argument("--seq_lengths", type=str, default="64,128,256,512,1024", 
                        help="Comma-separated list of sequence lengths to test")
    parser.add_argument("--vocab_sizes", type=str, default="1000,10000,50000,100000", 
                        help="Comma-separated list of vocabulary sizes to test")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--output_dir", type=str, default="./benchmarks", help="Output directory")
    parser.add_argument("--dataset", type=str, default=None, help="Path to text dataset for NLP evaluation")
    parser.add_argument("--mode", type=str, default="all", 
                        choices=["computational", "memory", "parameter", "nlp", "all"],
                        help="Benchmark mode")
    
    args = parser.parse_args()
    
    # Parse arguments
    seq_lengths = [int(s) for s in args.seq_lengths.split(",")]
    vocab_sizes = [int(v) for v in args.vocab_sizes.split(",")]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run benchmarks based on mode
    computational_results = []
    memory_results = []
    parameter_results = None
    nlp_results = []
    
    if args.mode in ["computational", "all"]:
        print("\n=== Running Computational Complexity Benchmark ===")
        standard_results = benchmark_computational_complexity(
            model_type="standard",
            seq_lengths=seq_lengths,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            batch_size=args.batch_size
        )
        computational_results.append(standard_results)
        
        signal_results = benchmark_computational_complexity(
            model_type="signal",
            seq_lengths=seq_lengths,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            batch_size=args.batch_size
        )
        computational_results.append(signal_results)
    
    if args.mode in ["memory", "all"]:
        print("\n=== Running Memory Usage Benchmark ===")
        standard_results = benchmark_memory_usage(
            model_type="standard",
            seq_lengths=seq_lengths,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            batch_size=args.batch_size
        )
        memory_results.append(standard_results)
        
        signal_results = benchmark_memory_usage(
            model_type="signal",
            seq_lengths=seq_lengths,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            batch_size=args.batch_size
        )
        memory_results.append(signal_results)
    
    if args.mode in ["parameter", "all"]:
        print("\n=== Running Parameter Efficiency Benchmark ===")
        parameter_results = benchmark_parameter_efficiency(
            vocab_sizes=vocab_sizes,
            embed_dim=args.embed_dim
        )
    
    if args.mode in ["nlp", "all"] and args.dataset:
        print("\n=== Running NLP Task Benchmark ===")
        dataset = load_text_dataset(args.dataset)
        
        standard_results = train_and_evaluate(
            model_type="standard",
            dataset=dataset,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            batch_size=args.batch_size
        )
        nlp_results.append(standard_results)
        
        signal_results = train_and_evaluate(
            model_type="signal",
            dataset=dataset,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            batch_size=args.batch_size
        )
        nlp_results.append(signal_results)
    
    # Plot results
    plot_benchmark_results(
        computational_results=computational_results,
        memory_results=memory_results,
        parameter_results=parameter_results,
        nlp_results=nlp_results,
        output_dir=args.output_dir
    )
    
    # Save raw results to JSON
    results = {
        "computational": computational_results,
        "memory": memory_results,
        "parameter": parameter_results,
        "nlp": nlp_results
    }
    
    with open(os.path.join(args.output_dir, "benchmark_results.json"), "w") as f:
        # Convert NumPy values to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                                np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            return obj
        
        json.dump(results, f, default=convert_to_serializable, indent=2)
    
    print(f"All benchmark results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 