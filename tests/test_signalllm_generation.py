#!/usr/bin/env python3

"""
Test SignalLLM Text Generation
==============================
This script loads a trained SignalLLM model and generates text from it.
"""

import os
import argparse
import logging
import torch
import json
import numpy as np
from tqdm import tqdm

# Import from main SignalLLM POC file and train_wikitext
from signalllm_hrfevo_poc import Config, SignalLLM
from train_wikitext import SimpleTokenizer, load_checkpoint
from mps_optimizations import optimize_for_mps, setup_mps_optimizations

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_path, tokenizer_path, device):
    """
    Load a trained model and tokenizer
    
    Args:
        model_path: Path to the saved model checkpoint
        tokenizer_path: Path to the saved tokenizer
        device: Device to load the model on
    
    Returns:
        model, tokenizer
    """
    # First set up MPS if available
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        setup_mps_optimizations()
    
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = SimpleTokenizer(mode='char')
    tokenizer.load_vocab(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    logger.info(f"Loaded tokenizer with vocabulary size: {vocab_size}")
    
    # Use the exact same parameters as in the training script
    config = Config(
        vocab_size=vocab_size,
        max_seq_length=512,
        embed_dim=256,        # From the training script
        num_heads=4,          # From the training script
        num_layers=2,         # From the training script
        use_signal_embed=True,
        use_wavelet_attention=True
    )
    
    # Create model
    logger.info(f"Creating model with training configuration")
    model = SignalLLM(config)
    
    # Load checkpoint with our own implementation
    logger.info(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Use a strict=False loading to handle any differences
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    # Log any issues
    if missing_keys:
        logger.warning(f"Missing keys when loading model: {len(missing_keys)} keys")
    if unexpected_keys:
        logger.warning(f"Unexpected keys when loading model: {len(unexpected_keys)} keys")
    
    logger.info(f"Loaded model from epoch {checkpoint['epoch']}")
    
    # Apply MPS optimizations if using Apple Silicon
    if device.type == 'mps':
        logger.info("Applying MPS optimizations...")
        model = optimize_for_mps(model, device=device)
    
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0, top_k=0, top_p=0.9, device='cpu'):
    """
    Generate text from the model using a prompted start
    
    Args:
        model: The SignalLLM model
        tokenizer: The SimpleTokenizer
        prompt: Text prompt to start generation
        max_length: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Number of highest probability tokens to keep (0 = disabled)
        top_p: Cumulative probability threshold for nucleus sampling
        device: Device to run generation on
    
    Returns:
        Generated text
    """
    logger.info(f"Generating text with prompt: '{prompt}'")
    
    # Tokenize the prompt
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    
    # Generate tokens
    generated = list(tokens)
    
    with torch.no_grad():
        for _ in tqdm(range(max_length), desc="Generating"):
            # Get the last context window (up to max_seq_length)
            inputs = torch.tensor([generated[-model.config.max_seq_length:]], dtype=torch.long).to(device)
            
            # Forward pass
            outputs = model(inputs)
            next_token_logits = outputs[:, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool).scatter_(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze().item()
            
            generated.append(next_token)
    
    # Decode the generated tokens
    generated_text = tokenizer.decode(generated)
    return generated_text

def main():
    parser = argparse.ArgumentParser(description="Test SignalLLM Text Generation")
    
    # Model parameters
    parser.add_argument("--model_dir", type=str, default="./wikitext_signalllm_mps",
                       help="Directory containing the saved model and tokenizer")
    parser.add_argument("--model_file", type=str, default="best_model.pt",
                       help="Model file name in the model directory")
    
    # Generation parameters
    parser.add_argument("--prompt", type=str, default="The",
                       help="Text prompt to start generation")
    parser.add_argument("--max_length", type=int, default=200,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=0,
                       help="Top-k sampling parameter (0 to disable)")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Determine device
    device = torch.device("mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() 
                        else "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model and tokenizer
    model_path = os.path.join(args.model_dir, args.model_file)
    tokenizer_path = os.path.join(args.model_dir, "tokenizer.json")
    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path, device)
    
    # Generate text
    generated_text = generate_text(
        model,
        tokenizer,
        args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device
    )
    
    # Print generated text
    print("\n" + "="*50)
    print("GENERATED TEXT:")
    print("="*50)
    print(generated_text)
    print("="*50)
    
    # Save generated text
    output_dir = args.model_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "generated_text.txt")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(generated_text)
    
    logger.info(f"Generated text saved to {output_path}")

if __name__ == "__main__":
    main() 