#!/usr/bin/env python3

"""
Visualize Model Outputs
=======================
This script analyzes the outputs, checkpoints, and performance of trained models
without actually loading them.
"""

import os
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pprint

def analyze_checkpoint_structure(checkpoint_path):
    """
    Analyze the structure of a model checkpoint without loading the actual model
    
    Args:
        checkpoint_path: Path to the checkpoint file
    """
    print(f"Analyzing checkpoint: {checkpoint_path}")
    
    # Load checkpoint metadata only
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract general information
    epoch = checkpoint.get('epoch', 'Unknown')
    step = checkpoint.get('step', 'Unknown')
    
    print(f"Checkpoint Info:")
    print(f"  Epoch: {epoch}")
    print(f"  Step: {step}")
    
    # Analyze the model state dict structure
    state_dict = checkpoint.get('model_state_dict', {})
    
    # Count parameters and layers
    total_params = 0
    param_stats = defaultdict(int)
    layer_types = defaultdict(int)
    
    # Get shape information
    param_shapes = {}
    
    for key, param in state_dict.items():
        # Count parameters
        param_count = np.prod(param.shape)
        total_params += param_count
        
        # Save shape
        param_shapes[key] = tuple(param.shape)
        
        # Extract layer type
        parts = key.split('.')
        if len(parts) > 1:
            layer_type = parts[0]
            layer_types[layer_type] += 1
        
        # Categorize parameters by size
        param_stats[f"{param_count:,}"] += 1
    
    print(f"\nModel Structure:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Number of layers/modules: {len(layer_types)}")
    
    print(f"\nLayer types:")
    for layer_type, count in layer_types.items():
        print(f"  {layer_type}: {count} parameters")
    
    # Print shape information for key components
    print("\nKey parameter shapes:")
    embed_shape = None
    layer_shapes = {}
    
    # Extract embedding shape
    for key, shape in param_shapes.items():
        if 'embedding' in key and 'weight' in key:
            embed_shape = shape
            print(f"  Embedding shape: {shape}")
            break
    
    # Extract block layer shapes
    for i in range(10):  # Check up to 10 layers
        layer_key = f"blocks.{i}"
        layer_params = {k: v for k, v in param_shapes.items() if k.startswith(layer_key)}
        if layer_params:
            layer_shapes[i] = layer_params
            print(f"\n  Layer {i} shapes:")
            for k, v in layer_params.items():
                if 'attention' in k or 'wavelet' in k or 'fourier' in k:
                    print(f"    {k}: {v}")
    
    return {
        'epoch': epoch,
        'step': step,
        'total_params': total_params,
        'layer_types': dict(layer_types),
        'param_shapes': param_shapes,
        'embed_shape': embed_shape,
        'layer_shapes': layer_shapes
    }

def analyze_results_file(results_path):
    """
    Analyze the training results file
    
    Args:
        results_path: Path to the results JSON file
    """
    print(f"\nAnalyzing results file: {results_path}")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Extract key metrics
    test_perplexity = results.get('test_perplexity', 'N/A')
    best_perplexity = results.get('best_perplexity', 'N/A')
    num_parameters = results.get('num_parameters', 'N/A')
    vocab_size = results.get('vocab_size', 'N/A')
    
    print(f"Results Summary:")
    print(f"  Test perplexity: {test_perplexity}")
    print(f"  Best validation perplexity: {best_perplexity}")
    print(f"  Number of parameters: {num_parameters:,}")
    print(f"  Vocabulary size: {vocab_size}")
    
    # Extract training history
    train_losses = results.get('train_losses', [])
    val_losses = results.get('val_losses', [])
    perplexities = results.get('perplexities', [])
    
    if train_losses and val_losses:
        # Plot training and validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(os.path.dirname(results_path), 'loss_analysis.png'))
        
        # Plot perplexity
        if perplexities:
            plt.figure(figsize=(10, 6))
            plt.plot(perplexities, 'g-')
            plt.xlabel('Epoch')
            plt.ylabel('Perplexity')
            plt.title('Validation Perplexity')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(os.path.dirname(results_path), 'perplexity_analysis.png'))
    
    return results

def analyze_tokenizer(tokenizer_path):
    """
    Analyze the tokenizer vocabulary
    
    Args:
        tokenizer_path: Path to the tokenizer JSON file
    """
    print(f"\nAnalyzing tokenizer: {tokenizer_path}")
    
    with open(tokenizer_path, 'r') as f:
        tokenizer_data = json.load(f)
    
    vocab = tokenizer_data.get('vocab', {})
    
    print(f"Tokenizer Summary:")
    print(f"  Vocabulary size: {len(vocab)}")
    
    # Show some example tokens
    print(f"  Sample tokens:")
    items = list(vocab.items())
    for i, (token, idx) in enumerate(items[:10]):
        print(f"    '{token}': {idx}")
    
    if len(items) > 10:
        print(f"    ... plus {len(items) - 10} more tokens")
    
    return tokenizer_data

def main():
    parser = argparse.ArgumentParser(description="Visualize model outputs")
    parser.add_argument("--model_dir", type=str, default="./wikitext_signalllm_mps",
                        help="Directory containing model checkpoints and results")
    
    args = parser.parse_args()
    
    model_dir = args.model_dir
    
    # Find checkpoint files
    checkpoint_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
    
    for ckpt_file in checkpoint_files:
        ckpt_path = os.path.join(model_dir, ckpt_file)
        analyze_checkpoint_structure(ckpt_path)
        print("\n" + "="*80)
    
    # Find results file
    results_path = os.path.join(model_dir, 'results.json')
    if os.path.exists(results_path):
        analyze_results_file(results_path)
    
    # Find tokenizer file
    tokenizer_path = os.path.join(model_dir, 'tokenizer.json')
    if os.path.exists(tokenizer_path):
        analyze_tokenizer(tokenizer_path)

if __name__ == "__main__":
    main() 