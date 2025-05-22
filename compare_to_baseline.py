#!/usr/bin/env python
# Compare SignalLLM to standard baselines on WikiText-103

import os
import torch
import argparse
import logging
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    GPT2Config, 
    GPT2LMHeadModel
)
from torch.utils.data import Dataset, DataLoader

# Import our model
from train_wikitext103 import WikiText103Dataset
from signalllm_hrfevo_poc import SignalLLM, Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(model, eval_loader, device, model_name):
    """Evaluate a model on WikiText-103 validation set"""
    logger.info(f"Evaluating {model_name}...")
    model.eval()
    
    total_loss = 0
    total_tokens = 0
    
    # Configure loss function
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(eval_loader, desc=f"Evaluating {model_name}")):
            # Move to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Get model outputs
            if model_name == "SignalLLM":
                outputs = model(inputs)
            else:
                # Transformer models like GPT-2
                outputs = model(inputs).logits
            
            # Calculate loss
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
            
            # Sum loss and count tokens
            total_loss += loss.sum().item()
            total_tokens += targets.numel()
            
            # Log intermediate results every 10 batches
            if batch_idx % 10 == 0:
                current_ppl = torch.exp(torch.tensor(total_loss / max(total_tokens, 1))).item()
                logger.info(f"Batch {batch_idx}/{len(eval_loader)}: Current PPL = {current_ppl:.4f}")
    
    # Calculate perplexity
    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    logger.info(f"{model_name} - Average Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
    return perplexity

def load_signalllm(model_path, device):
    """Load a trained SignalLLM model"""
    logger.info(f"Loading SignalLLM from {model_path}")
    
    # First load config
    config_path = os.path.join(os.path.dirname(model_path), "config.json")
    if os.path.exists(config_path):
        config = Config.load(config_path)
    else:
        # Create default config
        logger.warning(f"Config file not found at {config_path}, using default values")
        config = Config(
            vocab_size=50257,  # GPT-2 vocab size
            max_seq_length=512,
            embed_dim=256,
            num_heads=8,
            num_layers=4,
            hidden_dim=1024,
            dropout=0.1,
            use_signal_embed=True,
            use_wavelet_attention=True
        )
    
    # Create model and load weights
    model = SignalLLM(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    return model

def load_gpt2(model_size='small', device='cpu'):
    """Load a GPT-2 model of specified size"""
    if model_size == 'small':
        name = 'gpt2'  # 124M params
    elif model_size == 'medium':
        name = 'gpt2-medium'  # 355M params
    elif model_size == 'large':
        name = 'gpt2-large'  # 774M params
    else:
        name = 'gpt2-xl'  # 1.5B params
    
    logger.info(f"Loading {name} model...")
    model = AutoModelForCausalLM.from_pretrained(name)
    model = model.to(device)
    return model

def load_tiny_gpt2(vocab_size, device):
    """Load a tiny GPT-2 model with comparable parameters to SignalLLM"""
    logger.info("Creating tiny GPT-2 model with comparable parameters...")
    
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=512,
        n_embd=256,
        n_layer=4,
        n_head=8,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
    
    model = GPT2LMHeadModel(config)
    model = model.to(device)
    return model

def main(args):
    # Set device
    if torch.cuda.is_available() and not args.force_cpu:
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and not args.force_cpu:
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    logger.info(f"Using device: {device}")
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Create validation dataset and dataloader
    val_dataset = WikiText103Dataset(
        split='validation',
        seq_length=args.seq_length,
        tokenizer=tokenizer,
        cache_dir=args.data_dir
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    logger.info(f"Validation dataset size: {len(val_dataset)} samples")
    
    # Dictionary to store results
    results = {}
    
    # Evaluate SignalLLM if specified
    if args.signalllm_path:
        signalllm = load_signalllm(args.signalllm_path, device)
        signalllm_ppl = evaluate_model(signalllm, val_loader, device, "SignalLLM")
        results["SignalLLM"] = signalllm_ppl
    
    # Evaluate tiny GPT-2 (comparable size)
    if args.eval_tiny_gpt2:
        tiny_gpt2 = load_tiny_gpt2(len(tokenizer), device)
        tiny_gpt2_ppl = evaluate_model(tiny_gpt2, val_loader, device, "Tiny GPT-2")
        results["Tiny GPT-2"] = tiny_gpt2_ppl
    
    # Evaluate standard GPT-2 models
    if args.eval_gpt2:
        for size in ['small', 'medium']:
            if size == 'medium' and device.type == 'cpu':
                logger.warning("Skipping GPT-2 Medium on CPU as it would be too slow")
                continue
                
            gpt2 = load_gpt2(size, device)
            gpt2_ppl = evaluate_model(gpt2, val_loader, device, f"GPT-2 {size}")
            results[f"GPT-2 {size}"] = gpt2_ppl
    
    # Print summary
    logger.info("=" * 50)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 50)
    for model_name, ppl in results.items():
        logger.info(f"{model_name:20s}: Perplexity = {ppl:.4f}")
    
    # Print expected values from literature
    logger.info("\nFor reference, published perplexity values on WikiText-103:")
    logger.info("GPT-2 small:     ~30-35")
    logger.info("GPT-2 medium:    ~25-30")
    logger.info("GPT-2 large:     ~20-25")
    logger.info("GPT-2 xl:        ~18-20")
    logger.info("SOTA models:     ~10-15")
    logger.info("Perplexity = 1:  Perfect prediction (impossible for general text)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare SignalLLM to baseline models")
    
    parser.add_argument("--signalllm_path", type=str, default=None, 
                       help="Path to SignalLLM checkpoint to evaluate")
    parser.add_argument("--eval_gpt2", action="store_true", 
                       help="Evaluate GPT-2 models")
    parser.add_argument("--eval_tiny_gpt2", action="store_true", 
                       help="Evaluate tiny GPT-2 with comparable parameters")
    parser.add_argument("--data_dir", type=str, default="./data",
                       help="Directory to store dataset cache")
    parser.add_argument("--seq_length", type=int, default=512,
                       help="Sequence length for evaluation")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for evaluation")
    parser.add_argument("--force_cpu", action="store_true", 
                       help="Force using CPU even if GPU is available")
    
    args = parser.parse_args()
    
    # Default to evaluating at least something
    if not (args.signalllm_path or args.eval_gpt2 or args.eval_tiny_gpt2):
        args.eval_tiny_gpt2 = True
    
    main(args) 