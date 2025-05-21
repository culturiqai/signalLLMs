#!/usr/bin/env python3
# benchmark_comparison_mps.py - MPS-optimized benchmark for SignalLLM

import os
import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from tqdm import tqdm

# Import SignalLLM components
from signalllm_hrfevo_poc import SignalLLM, Config, SimpleTokenizer, count_parameters

# Import MPS optimizations
from mps_optimizations import optimize_for_mps, setup_mps_optimizations, MPSWaveletTransform

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Memory profiling
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil not available. Memory usage benchmarks will be limited.")

# Check MPS availability
MPS_AVAILABLE = torch.backends.mps.is_available()
if MPS_AVAILABLE:
    print("Apple Silicon MPS is available and will be used.")
    setup_mps_optimizations()
else:
    print("Apple Silicon MPS not available. Using CPU fallback.")

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
            "Don't judge a book by its cover.",
            "The pen is mightier than the sword.",
            "When in Rome, do as the Romans do."
        ]
        
        # Create variations of these patterns
        for i in range(max_samples):
            pattern = patterns[i % len(patterns)]
            if i >= len(patterns):
                # Add variations after first set
                words = pattern.split()
                if len(words) > 3:
                    # Swap some words
                    idx1 = np.random.randint(0, len(words))
                    idx2 = np.random.randint(0, len(words))
                    words[idx1], words[idx2] = words[idx2], words[idx1]
                
                pattern = " ".join(words)
            
            synthetic_texts.append(pattern)
        
        return synthetic_texts
    else:
        # Load from file
        with open(filepath, 'r', encoding='utf-8') as f:
            texts = f.readlines()
        
        # Clean and limit
        texts = [text.strip() for text in texts if text.strip()]
        texts = texts[:max_samples]
        
        return texts

def train_and_evaluate(model_type, texts, tokenizer=None, device=None, 
                      epochs=3, batch_size=16, seq_length=64, 
                      embed_dim=128, num_heads=2, num_layers=1,
                      optimizer_cls=torch.optim.Adam, lr=0.001,
                      verbose=False, use_mps_optimizations=False):
    """Train and evaluate a model on the given texts"""
    if device is None:
        device = torch.device("mps" if MPS_AVAILABLE else "cpu")
    
    # Create tokenizer if not provided
    if tokenizer is None:
        tokenizer = SimpleTokenizer(mode='char')
        tokenizer.build_vocab(texts)
    
    vocab_size = tokenizer.get_vocab_size()
    
    # Create model
    if model_type == "standard":
        # Standard transformer model
        config = Config(
            vocab_size=vocab_size,
            max_seq_length=seq_length,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            use_signal_embed=False,  # Use standard embedding
            use_wavelet_attention=False  # Use standard attention
        )
    else:
        # SignalLLM with wavelet transform
        config = Config(
            vocab_size=vocab_size,
            max_seq_length=seq_length,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            use_signal_embed=True,  # Use signal embedding
            use_wavelet_attention=True  # Use wavelet attention
        )
    
    # Create model
    model = SignalLLM(config).to(device)
    
    # Apply MPS optimizations if requested
    if use_mps_optimizations and MPS_AVAILABLE:
        print(f"Applying MPS optimizations to {model_type} model...")
        model = optimize_for_mps(model, device=device)
        print(f"MPS optimizations applied successfully.")
    
    # Prepare data
    # Convert texts to token IDs
    token_ids = []
    for text in texts:
        ids = tokenizer.encode(text)
        if len(ids) > seq_length:
            ids = ids[:seq_length]
        elif len(ids) < seq_length:
            # Pad with zeros
            ids = ids + [0] * (seq_length - len(ids))
        token_ids.append(ids)
    
    # Create dataset
    data_size = len(token_ids)
    train_size = int(0.8 * data_size)
    train_data = token_ids[:train_size]
    val_data = token_ids[train_size:]
    
    # Print model parameters
    if verbose:
        print(f"Training on {len(train_data)} examples, validating on {len(val_data)} examples")
        print(f"Vocabulary size: {vocab_size}")
        print(f"Model parameters: {count_parameters(model)}")
    
    # Create optimizer
    optimizer = optimizer_cls(model.parameters(), lr=lr)
    
    # Training loop
    train_losses = []
    val_losses = []
    perplexities = []
    
    # Measure training time
    start_time = time.time()
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Create batches
        np.random.shuffle(train_data)
        num_batches = len(train_data) // batch_size
        
        # Train loop with progress bar
        train_iter = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}") if verbose else range(num_batches)
        
        for batch_idx in train_iter:
            # Get batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(train_data))
            batch_data = train_data[start_idx:end_idx]
            
            # Convert to tensors
            batch_tensor = torch.tensor(batch_data, dtype=torch.long, device=device)
            
            # Create input and target
            input_ids = batch_tensor[:, :-1]  # All but last token
            target_ids = batch_tensor[:, 1:]  # All but first token
            
            # Forward pass
            outputs = model(input_ids)
            
            # Compute loss
            loss = F.cross_entropy(
                outputs.view(-1, vocab_size),
                target_ids.reshape(-1)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update stats
            total_loss += loss.item()
        
        # Compute average loss
        avg_train_loss = total_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Evaluate
        model.eval()
        val_loss = 0
        num_val_batches = len(val_data) // batch_size
        
        with torch.no_grad():
            for batch_idx in range(num_val_batches):
                # Get batch
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(val_data))
                batch_data = val_data[start_idx:end_idx]
                
                # Convert to tensors
                batch_tensor = torch.tensor(batch_data, dtype=torch.long, device=device)
                
                # Create input and target
                input_ids = batch_tensor[:, :-1]  # All but last token
                target_ids = batch_tensor[:, 1:]  # All but first token
                
                # Forward pass
                outputs = model(input_ids)
                
                # Compute loss
                loss = F.cross_entropy(
                    outputs.view(-1, vocab_size),
                    target_ids.reshape(-1)
                )
                
                # Update stats
                val_loss += loss.item()
        
        # Compute average loss and perplexity
        avg_val_loss = val_loss / num_val_batches
        val_losses.append(avg_val_loss)
        perplexity = np.exp(avg_val_loss)
        perplexities.append(perplexity)
        
        if verbose:
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Eval Loss: {avg_val_loss:.4f}")
            print(f"  Perplexity: {perplexity:.4f}")
    
    # Total training time
    total_time = time.time() - start_time
    
    # Return results
    return {
        "model_type": model_type,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "perplexities": perplexities,
        "final_perplexity": perplexities[-1],
        "training_time": total_time,
        "model_params": count_parameters(model)
    }

def plot_nlp_results(nlp_results, output_dir="./benchmarks"):
    """Plot NLP benchmark results and save figures"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for standard and signal models
    standard_data = next((data for data in nlp_results if data["model_type"] == "standard"), None)
    signal_data = next((data for data in nlp_results if data["model_type"] == "signalllm"), None)
    signal_mps_data = next((data for data in nlp_results if data["model_type"] == "signalllm_mps"), None)
    
    if standard_data and signal_data:
        # Plot loss curves
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(standard_data["val_losses"])+1)
        
        plt.plot(epochs, standard_data["val_losses"], 'b-', label='Standard Transformer')
        plt.plot(epochs, signal_data["val_losses"], 'r-', label='SignalLLM')
        if signal_mps_data:
            plt.plot(epochs, signal_mps_data["val_losses"], 'g-', label='SignalLLM (MPS Optimized)')
        
        plt.title('Validation Loss Comparison')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/nlp_performance.png")
        plt.close()
        
        # Plot perplexity
        plt.figure(figsize=(10, 6))
        
        plt.plot(epochs, standard_data["perplexities"], 'b-', label='Standard Transformer')
        plt.plot(epochs, signal_data["perplexities"], 'r-', label='SignalLLM')
        if signal_mps_data:
            plt.plot(epochs, signal_mps_data["perplexities"], 'g-', label='SignalLLM (MPS Optimized)')
        
        plt.title('Perplexity Comparison')
        plt.xlabel('Epochs')
        plt.ylabel('Perplexity')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/convergence_rate.png")
        plt.close()
        
        # Plot training time
        plt.figure(figsize=(8, 6))
        models = ['Standard', 'SignalLLM']
        times = [standard_data["training_time"], signal_data["training_time"]]
        
        if signal_mps_data:
            models.append('SignalLLM (MPS)')
            times.append(signal_mps_data["training_time"])
        
        plt.bar(models, times, color=['blue', 'red', 'green'][:len(models)])
        plt.title('Training Time Comparison')
        plt.ylabel('Time (seconds)')
        plt.grid(axis='y')
        
        # Add values on top of bars
        for i, v in enumerate(times):
            plt.text(i, v + 0.1, f"{v:.2f}s", ha='center')
        
        plt.savefig(f"{output_dir}/training_time.png")
        plt.close()
        
        # Plot training speedup (relative to standard transformer)
        plt.figure(figsize=(8, 6))
        
        models = ['Standard', 'SignalLLM']
        standard_time = standard_data["training_time"]
        speedups = [1.0, standard_time / signal_data["training_time"]]
        
        if signal_mps_data:
            models.append('SignalLLM (MPS)')
            speedups.append(standard_time / signal_mps_data["training_time"])
        
        plt.bar(models, speedups, color=['blue', 'red', 'green'][:len(models)])
        plt.title('Training Speedup Ratio')
        plt.ylabel('Speedup Factor (higher is better)')
        plt.grid(axis='y')
        plt.axhline(y=1.0, color='black', linestyle='--')
        
        # Add values on top of bars
        for i, v in enumerate(speedups):
            plt.text(i, v + 0.05, f"{v:.2f}x", ha='center')
        
        plt.savefig(f"{output_dir}/training_speedup.png")
        plt.close()

def compare_wavelet_implementations(seq_lengths=[32, 64, 128, 256, 512], 
                                   embed_dim=128, batch_size=8,
                                   output_dir="./benchmarks"):
    """Compare different wavelet transform implementations"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare device
    device = torch.device("mps" if MPS_AVAILABLE else "cpu")
    
    # Implementations to test
    standard_times = []
    mps_times = []
    
    # Test each sequence length
    for seq_length in seq_lengths:
        # Create test tensor
        x = torch.randn(batch_size, seq_length, embed_dim, device=device)
        
        # Time standard implementation
        standard_transform = (torch.nn.Module()).to(device)
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
    
    # Return results
    return {
        "seq_lengths": seq_lengths,
        "standard_times": standard_times,
        "mps_times": mps_times
    }

def run_nlp_benchmark(args):
    """Run NLP performance benchmark with MPS optimizations"""
    # Set up device
    device = torch.device("mps" if MPS_AVAILABLE else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    texts = load_text_dataset(args.dataset_path, max_samples=args.max_samples)
    
    # Create tokenizer
    tokenizer = SimpleTokenizer(mode='char')
    tokenizer.build_vocab(texts)
    
    # Run standard model
    print("Training standard transformer model...")
    standard_results = train_and_evaluate(
        model_type="standard",
        texts=texts,
        tokenizer=tokenizer,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        verbose=args.verbose
    )
    
    # Run SignalLLM model
    print("Training SignalLLM model...")
    signal_results = train_and_evaluate(
        model_type="signalllm",
        texts=texts,
        tokenizer=tokenizer,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        verbose=args.verbose
    )
    
    # Run SignalLLM with MPS optimizations
    if MPS_AVAILABLE:
        print("Training SignalLLM model with MPS optimizations...")
        signal_mps_results = train_and_evaluate(
            model_type="signalllm_mps",
            texts=texts,
            tokenizer=tokenizer,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            verbose=args.verbose,
            use_mps_optimizations=True
        )
        all_results = [standard_results, signal_results, signal_mps_results]
    else:
        all_results = [standard_results, signal_results]
    
    # Compare wavelet implementations
    wavelet_comparison = compare_wavelet_implementations(
        seq_lengths=[32, 64, 128, 256, 512],
        embed_dim=args.embed_dim,
        batch_size=args.batch_size
    )
    
    # Plot results
    print("Plotting benchmark results...")
    plot_nlp_results(all_results, output_dir=args.output_dir)
    
    # Save results to JSON
    results_file = os.path.join(args.output_dir, "mps_benchmark_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            "nlp_results": all_results,
            "wavelet_comparison": {
                "seq_lengths": wavelet_comparison["seq_lengths"],
                "standard_times": wavelet_comparison["standard_times"],
                "mps_times": wavelet_comparison["mps_times"]
            }
        }, f, indent=2)
    
    # Print summary
    print("\n=== NLP Benchmark Results ===")
    print(f"Standard Transformer final perplexity: {standard_results['final_perplexity']:.4f}")
    print(f"SignalLLM final perplexity: {signal_results['final_perplexity']:.4f}")
    
    print(f"Standard Transformer training time: {standard_results['training_time']:.2f}s")
    print(f"SignalLLM training time: {signal_results['training_time']:.2f}s")
    
    speedup = standard_results['training_time'] / signal_results['training_time']
    print(f"Training speedup: {speedup:.2f}x")
    
    if MPS_AVAILABLE:
        print(f"SignalLLM (MPS) final perplexity: {signal_mps_results['final_perplexity']:.4f}")
        print(f"SignalLLM (MPS) training time: {signal_mps_results['training_time']:.2f}s")
        mps_speedup = standard_results['training_time'] / signal_mps_results['training_time']
        print(f"MPS training speedup: {mps_speedup:.2f}x")
    
    print(f"All MPS benchmark results saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MPS-optimized benchmark for SignalLLM")
    
    parser.add_argument("--dataset_path", type=str, default=None,
                       help="Path to text dataset file (will create synthetic if not provided)")
    parser.add_argument("--max_samples", type=int, default=2000,
                       help="Maximum number of samples to use from dataset")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--seq_length", type=int, default=64,
                       help="Sequence length for training")
    parser.add_argument("--embed_dim", type=int, default=256,
                       help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=4,
                       help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=1,
                       help="Number of transformer layers")
    parser.add_argument("--output_dir", type=str, default="./benchmarks",
                       help="Directory to save benchmark results")
    parser.add_argument("--verbose", action="store_true",
                       help="Print verbose training progress")
    
    args = parser.parse_args()
    
    # Run benchmark
    run_nlp_benchmark(args) 