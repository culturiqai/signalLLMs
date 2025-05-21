#!/usr/bin/env python3
# run_nlp_benchmark.py - Run NLP task benchmark for SignalLLM vs standard Transformer

import os
import argparse
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from benchmark_comparison import train_and_evaluate, load_text_dataset

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NLP-Benchmark")

def plot_nlp_results(nlp_results, output_dir="./benchmarks"):
    """Plot NLP benchmark results and save figures"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for standard and signal models
    standard_data = next((data for data in nlp_results if data["model_type"] == "standard"), None)
    signal_data = next((data for data in nlp_results if data["model_type"] == "signal"), None)
    
    if standard_data and signal_data:
        # 1. Training and evaluation metrics
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        epochs = range(1, len(standard_data["train_loss"]) + 1)
        
        # Plot training loss
        ax1.plot(epochs, standard_data["train_loss"], 'o-', color='blue', label="Standard Transformer")
        ax1.plot(epochs, signal_data["train_loss"], 's-', color='red', label="SignalLLM")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Training Loss")
        ax1.set_title("Training Loss vs. Epoch")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot perplexity
        ax2.plot(epochs, standard_data["perplexity"], 'o-', color='blue', label="Standard Transformer")
        ax2.plot(epochs, signal_data["perplexity"], 's-', color='red', label="SignalLLM")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Perplexity")
        ax2.set_title("Perplexity vs. Epoch")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "nlp_performance.png"), dpi=300)
        plt.close()
        
        # 2. Training time comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        models = ["Standard Transformer", "SignalLLM"]
        times = [standard_data["training_time"], signal_data["training_time"]]
        colors = ['blue', 'red']
        
        ax.bar(models, times, color=colors)
        ax.set_ylabel("Training Time (s)")
        ax.set_title("Total Training Time Comparison")
        
        # Add text labels on top of bars
        for i, v in enumerate(times):
            ax.text(i, v + 0.1, f"{v:.2f}s", ha='center')
        
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_time.png"), dpi=300)
        plt.close()
        
        # 3. Calculate and plot speedup ratio
        if signal_data["training_time"] > 0 and standard_data["training_time"] > 0:
            speedup = standard_data["training_time"] / signal_data["training_time"]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(["Speedup Ratio"], [speedup], color='green')
            ax.set_ylabel("Ratio (higher = better)")
            ax.set_title("SignalLLM Training Speedup Ratio")
            ax.text(0, speedup + 0.05, f"{speedup:.2f}x", ha='center')
            ax.set_ylim(0, max(2.0, speedup * 1.2))  # Set y limit with some margin
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "training_speedup.png"), dpi=300)
            plt.close()
        
        # 4. Create convergence plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        standard_min = min(standard_data["perplexity"])
        signal_min = min(signal_data["perplexity"])
        
        # Normalize to show convergence to best value
        standard_norm = [(p - standard_min) / (standard_data["perplexity"][0] - standard_min) 
                        if (standard_data["perplexity"][0] - standard_min) > 0 else 0 
                        for p in standard_data["perplexity"]]
        signal_norm = [(p - signal_min) / (signal_data["perplexity"][0] - signal_min) 
                      if (signal_data["perplexity"][0] - signal_min) > 0 else 0 
                      for p in signal_data["perplexity"]]
        
        ax.plot(epochs, standard_norm, 'o-', color='blue', label="Standard Transformer")
        ax.plot(epochs, signal_norm, 's-', color='red', label="SignalLLM")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Normalized Convergence (0 = best)")
        ax.set_title("Convergence Rate Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "convergence_rate.png"), dpi=300)
        plt.close()
        
        logger.info(f"All NLP benchmark plots saved to {output_dir}")
        
        # Return metrics for comparison
        return {
            "standard_final_perplexity": standard_data["perplexity"][-1],
            "signal_final_perplexity": signal_data["perplexity"][-1],
            "standard_training_time": standard_data["training_time"],
            "signal_training_time": signal_data["training_time"],
            "speedup_ratio": speedup if signal_data["training_time"] > 0 else 0
        }
    else:
        logger.error("Missing data for one or both models")
        return None

def update_benchmark_html(output_dir="./benchmarks"):
    """Update the benchmark HTML file to include the NLP Task Performance section"""
    html_path = os.path.join(output_dir, "benchmark_results.html")
    
    if not os.path.exists(html_path):
        logger.error(f"Benchmark HTML file not found at {html_path}")
        return
    
    with open(html_path, 'r') as f:
        html_content = f.read()
    
    # Check if the "Pending Benchmarks" section includes NLP tasks
    if "Pending Benchmarks" in html_content and "NLP Task Performance" in html_content:
        # Replace pending section with actual results
        new_section = """
    <div class="result-section">
        <h2>4. NLP Task Performance</h2>
        <p>This section compares the language modeling performance of both architectures.</p>
        <div class="plot-container">
            <img src="nlp_performance.png" alt="NLP Task Performance" style="max-width: 100%;">
            <p><em>Figure: Training loss and perplexity comparison between standard transformer and SignalLLM.</em></p>
        </div>
        <div class="plot-container">
            <img src="training_time.png" alt="Training Time Comparison" style="max-width: 100%;">
            <p><em>Figure: Total training time comparison showing computational efficiency.</em></p>
        </div>
        <div class="plot-container">
            <img src="convergence_rate.png" alt="Convergence Rate" style="max-width: 100%;">
            <p><em>Figure: Convergence rate comparison showing how quickly each model approaches its best performance.</em></p>
        </div>
        <div class="plot-container">
            <img src="training_speedup.png" alt="Training Speedup" style="max-width: 100%;">
            <p><em>Figure: Training speedup ratio of SignalLLM compared to standard transformer.</em></p>
        </div>
    </div>
        """
        
        # Remove the pending benchmarks section
        import re
        html_content = re.sub(
            r'<div class="result-section">\s*<h2>Pending Benchmarks</h2>.*?</div>\s*</body>',
            f"{new_section}\n</body>",
            html_content, 
            flags=re.DOTALL
        )
        
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Updated benchmark HTML with NLP task performance results")

def main():
    """Main function to run NLP benchmarks"""
    parser = argparse.ArgumentParser(description="SignalLLM NLP Benchmarking Tool")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--seq_length", type=int, default=64, help="Sequence length")
    parser.add_argument("--vocab_size", type=int, default=1000, help="Vocabulary size")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--output_dir", type=str, default="./benchmarks", help="Output directory")
    parser.add_argument("--dataset", type=str, default=None, help="Path to text dataset (optional)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else
                         "cpu")
    logger.info(f"Using device: {device}")
    
    # Load or create dataset
    logger.info("Loading dataset...")
    dataset = load_text_dataset(args.dataset, max_samples=1000)
    logger.info(f"Dataset loaded with {len(dataset)} samples")
    
    # Run NLP benchmarks
    nlp_results = []
    
    logger.info("\n=== Training Standard Transformer ===")
    standard_results = train_and_evaluate(
        model_type="standard",
        dataset=dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        device=device
    )
    nlp_results.append(standard_results)
    
    logger.info("\n=== Training SignalLLM ===")
    signal_results = train_and_evaluate(
        model_type="signal",
        dataset=dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        device=device
    )
    nlp_results.append(signal_results)
    
    # Plot and compare results
    logger.info("Plotting benchmark results...")
    metrics = plot_nlp_results(nlp_results, output_dir=args.output_dir)
    
    if metrics:
        logger.info("\n=== NLP Benchmark Results ===")
        logger.info(f"Standard Transformer final perplexity: {metrics['standard_final_perplexity']:.4f}")
        logger.info(f"SignalLLM final perplexity: {metrics['signal_final_perplexity']:.4f}")
        logger.info(f"Standard Transformer training time: {metrics['standard_training_time']:.2f}s")
        logger.info(f"SignalLLM training time: {metrics['signal_training_time']:.2f}s")
        logger.info(f"Training speedup: {metrics['speedup_ratio']:.2f}x")
    
    # Update the benchmark HTML file
    update_benchmark_html(args.output_dir)
    
    logger.info(f"All NLP benchmark results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 