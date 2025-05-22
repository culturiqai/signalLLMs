#!/usr/bin/env python
# Test script to validate metrics and check for data leakage

import os
import torch
import logging
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the WikiText103Dataset class definition
from train_wikitext103 import WikiText103Dataset

def check_token_overlap():
    """Check if there's overlap between train and validation token sequences"""
    logger.info("Checking for token overlap between train and validation sets...")
    
    # Load datasets with small sequence length for faster processing
    seq_length = 128
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    train_dataset = WikiText103Dataset(
        split='train', 
        seq_length=seq_length, 
        tokenizer=tokenizer,
        cache_dir='./data'
    )
    
    val_dataset = WikiText103Dataset(
        split='validation', 
        seq_length=seq_length, 
        tokenizer=tokenizer,
        cache_dir='./data'
    )
    
    # Print dataset sizes
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Validation dataset: {len(val_dataset)} samples")
    
    # Check step counts - should be significantly less with non-overlapping sequences
    steps_with_batch_size_8 = len(train_dataset) // 8
    logger.info(f"Steps per epoch with batch size 8: {steps_with_batch_size_8}")
    
    # Check for exact sequence matches between train and validation
    logger.info("Checking for exact sequence matches...")
    
    # Sample 100 validation sequences
    val_samples = [val_dataset[i][0].tolist() for i in range(min(100, len(val_dataset)))]
    
    # Sample 1000 train sequences
    train_samples = [train_dataset[i][0].tolist() for i in range(min(1000, len(train_dataset)))]
    
    # Check for exact matches
    exact_matches = 0
    for val_seq in val_samples:
        for train_seq in train_samples:
            if val_seq == train_seq:
                exact_matches += 1
                break
    
    logger.info(f"Found {exact_matches} exact sequence matches out of {len(val_samples)} validation samples")
    
    # Check for partial overlaps (50% or more)
    logger.info("Checking for partial sequence overlaps...")
    partial_matches = 0
    high_overlap_matches = 0
    
    for val_seq in tqdm(val_samples):
        for train_seq in train_samples:
            # Convert to sets for faster overlap computation
            val_set = set(val_seq)
            train_set = set(train_seq)
            
            # Calculate overlap ratio
            overlap = len(val_set.intersection(train_set)) / len(val_set)
            
            if overlap >= 0.5:
                partial_matches += 1
                if overlap >= 0.9:
                    high_overlap_matches += 1
                break
    
    logger.info(f"Found {partial_matches} partial matches (≥50% overlap) out of {len(val_samples)} validation samples")
    logger.info(f"Found {high_overlap_matches} high overlap matches (≥90% overlap)")

def test_perplexity_calculation():
    """Test if perplexity is calculated correctly"""
    logger.info("Testing perplexity calculation...")
    
    # Create mock data
    batch_size = 4
    seq_length = 10
    vocab_size = 100
    
    # Case 1: Perfect predictions (should give PPL close to 1)
    logger.info("Case 1: Perfect predictions")
    targets = torch.randint(0, vocab_size, (batch_size, seq_length))
    perfect_logits = torch.zeros(batch_size, seq_length, vocab_size)
    
    # Set the logit for the correct class very high
    for b in range(batch_size):
        for s in range(seq_length):
            perfect_logits[b, s, targets[b, s]] = 100.0
    
    # Calculate loss
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    loss_per_token = criterion(perfect_logits.reshape(-1, vocab_size), targets.reshape(-1))
    
    # Method 1: Average loss then exp
    avg_loss = loss_per_token.mean().item()
    correct_ppl = torch.exp(torch.tensor(avg_loss)).item()
    
    # Method 2: Exp per token then average (incorrect)
    incorrect_ppl = torch.exp(loss_per_token).mean().item()
    
    logger.info(f"Perfect predictions - Correct PPL: {correct_ppl:.4f}, Incorrect PPL: {incorrect_ppl:.4f}")
    
    # Case 2: Random predictions (should give PPL close to vocab size)
    logger.info("Case 2: Random predictions")
    random_logits = torch.randn(batch_size, seq_length, vocab_size) * 0.1
    
    # Calculate loss
    loss_per_token = criterion(random_logits.reshape(-1, vocab_size), targets.reshape(-1))
    
    # Method 1: Average loss then exp
    avg_loss = loss_per_token.mean().item()
    correct_ppl = torch.exp(torch.tensor(avg_loss)).item()
    
    # Method 2: Exp per token then average (incorrect)
    incorrect_ppl = torch.exp(loss_per_token).mean().item()
    
    logger.info(f"Random predictions - Correct PPL: {correct_ppl:.4f}, Incorrect PPL: {incorrect_ppl:.4f}")
    logger.info(f"Expected PPL for random predictions: ~{vocab_size} (uniform distribution over vocab)")
    
    # Case 3: Realistic language model predictions
    logger.info("Case 3: Simulated realistic predictions")
    
    # Simulate a distribution where the correct token has ~30% probability, and rest is distributed among other tokens
    realistic_logits = torch.randn(batch_size, seq_length, vocab_size) * 0.1
    
    # Boost the correct token and a few likely alternatives
    for b in range(batch_size):
        for s in range(seq_length):
            correct_idx = targets[b, s].item()
            realistic_logits[b, s, correct_idx] = 2.0  # Correct token gets higher logit
            
            # A few "likely" alternatives also get higher logits
            for k in range(5):
                alt_idx = (correct_idx + k + 1) % vocab_size
                realistic_logits[b, s, alt_idx] = 1.0
    
    # Calculate loss
    loss_per_token = criterion(realistic_logits.reshape(-1, vocab_size), targets.reshape(-1))
    
    # Method 1: Average loss then exp
    avg_loss = loss_per_token.mean().item()
    correct_ppl = torch.exp(torch.tensor(avg_loss)).item()
    
    # Method 2: Exp per token then average (incorrect)
    incorrect_ppl = torch.exp(loss_per_token).mean().item()
    
    logger.info(f"Realistic predictions - Correct PPL: {correct_ppl:.4f}, Incorrect PPL: {incorrect_ppl:.4f}")

if __name__ == "__main__":
    logger.info("Starting validation tests...")
    
    # Test perplexity calculation
    test_perplexity_calculation()
    
    # Check for data leakage between train and validation
    check_token_overlap()
    
    logger.info("Tests completed!") 