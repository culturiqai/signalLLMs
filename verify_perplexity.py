import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
import numpy as np
import logging
import sys
from tqdm import tqdm
from datasets import load_dataset
import json
import os
import time

# Import our dataset class
from train_wikitext103 import WikiText103Dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def test_perplexity_calculation():
    """Verify perplexity is calculated correctly"""
    # Test case with known values
    logger.info("Testing perplexity calculation with known values...")
    
    # Case 1: Perfect prediction (loss should be 0, perplexity should be 1)
    perfect_loss = torch.tensor(0.0)
    perfect_ppl = torch.exp(perfect_loss)
    logger.info(f"Perfect prediction - Loss: {perfect_loss:.4f}, PPL: {perfect_ppl:.4f}")
    assert perfect_ppl.item() == 1.0, "Perfect prediction should have PPL of 1.0"
    
    # Case 2: Random prediction with 10,000 vocab (loss should be ~9.21, perplexity should be ~10,000)
    random_loss = torch.tensor(np.log(10000))
    random_ppl = torch.exp(random_loss)
    logger.info(f"Random prediction (10K vocab) - Loss: {random_loss:.4f}, PPL: {random_ppl:.2f}")
    assert abs(random_ppl.item() - 10000) < 10, "Random prediction should have PPL close to vocabulary size"
    
    # Case 3: Multiple token case with average loss
    multi_token_losses = torch.tensor([1.0, 2.0, 3.0])
    avg_loss = multi_token_losses.mean()
    multi_token_ppl = torch.exp(avg_loss)
    logger.info(f"Multi-token average - Loss: {avg_loss:.4f}, PPL: {multi_token_ppl:.4f}")
    assert abs(multi_token_ppl.item() - torch.exp(torch.tensor(2.0)).item()) < 0.001, "Multi-token PPL should be exp of average loss"
    
    # Case 4: Weighted loss with varying token counts
    batch_losses = [
        {"loss": 2.0, "tokens": 100},  # Batch 1
        {"loss": 3.0, "tokens": 50},   # Batch 2
        {"loss": 1.0, "tokens": 150}   # Batch 3
    ]
    
    total_loss = sum(batch["loss"] * batch["tokens"] for batch in batch_losses)
    total_tokens = sum(batch["tokens"] for batch in batch_losses)
    weighted_avg_loss = total_loss / total_tokens
    weighted_ppl = torch.exp(torch.tensor(weighted_avg_loss))
    
    logger.info(f"Weighted average - Loss: {weighted_avg_loss:.4f}, PPL: {weighted_ppl:.4f}")
    expected_weighted_loss = (2.0*100 + 3.0*50 + 1.0*150) / 300
    assert abs(weighted_avg_loss - expected_weighted_loss) < 0.001, "Weighted average loss calculation is incorrect"
    
    logger.info("✅ Perplexity calculation verification PASSED")
    
def evaluate_gpt2_baseline(batch_size=1, seq_length=512, device='cpu', model_config=None):
    """Evaluate a baseline GPT-2 model on WikiText-103 validation set"""
    model_size = model_config.n_embd if model_config else 256
    layer_count = model_config.n_layer if model_config else 4
    logger.info(f"Evaluating GPT-2 baseline (dim={model_size}, layers={layer_count}) on WikiText-103...")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Create model configuration if not provided
    if not model_config:
        model_config = GPT2Config(
            vocab_size=50257,
            n_positions=1024,
            n_embd=256,    # Small embedding dimension
            n_layer=4,     # Fewer layers
            n_head=8,      # Standard head count
        )
    
    model = GPT2LMHeadModel(model_config)
    model.to(device)
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")
    
    # Load validation dataset 
    val_dataset = WikiText103Dataset(
        split='validation', 
        seq_length=seq_length, 
        tokenizer=tokenizer
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Evaluate
    val_loss = 0.0
    val_tokens = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # GPT-2 expects input_ids only
            outputs = model(inputs, labels=targets)
            loss = outputs.loss
            
            # Each element in the batch contributes seq_length tokens
            batch_tokens = targets.numel()
            val_loss += loss.item() * batch_tokens
            val_tokens += batch_tokens
    
    # Calculate perplexity
    avg_loss = val_loss / val_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    logger.info(f"GPT-2 Baseline - Val Loss: {avg_loss:.4f}, Val PPL: {perplexity:.2f}")
    return perplexity

def check_dataset_overlap(seq_length=1024):
    """Check for dataset overlap between training and validation sets"""
    logger.info("Checking for dataset overlap...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Load datasets
    train_dataset = WikiText103Dataset(
        split='train', 
        seq_length=seq_length, 
        tokenizer=tokenizer
    )
    
    val_dataset = WikiText103Dataset(
        split='validation', 
        seq_length=seq_length, 
        tokenizer=tokenizer
    )
    
    # Sample a subset of validation sequences to check for overlap
    val_samples = min(500, len(val_dataset))
    train_samples = min(5000, len(train_dataset))
    
    logger.info(f"Checking {val_samples} validation samples against {train_samples} training samples...")
    
    overlap_count = 0
    high_overlap_samples = []
    
    # For each validation sample, check for matches in training set
    for i in tqdm(range(val_samples), desc="Checking overlap"):
        val_x, _ = val_dataset[i]
        val_ids = val_x.tolist()
        
        max_overlap = 0
        for j in range(train_samples):
            train_x, _ = train_dataset[j]
            train_ids = train_x.tolist()
            
            # Count exact subsequence matches of at least 10 tokens
            for k in range(len(val_ids) - 10):
                subseq = val_ids[k:k+10]
                # Convert to string for substring search
                if str(subseq) in str(train_ids):  # Quick string match check
                    # Compute more precise overlap percentage
                    overlap_size = sum(1 for x in val_ids if x in train_ids) / len(val_ids)
                    max_overlap = max(max_overlap, overlap_size)
        
        if max_overlap > 0.5:  # Consider high overlap if more than 50% tokens match
            overlap_count += 1
            high_overlap_samples.append({"sample_id": i, "overlap": max_overlap})
    
    logger.info(f"Found {overlap_count} validation samples with >50% overlap with training data ({overlap_count/val_samples:.2%})")
    
    if high_overlap_samples:
        logger.info("Top 5 highest overlap samples:")
        for sample in sorted(high_overlap_samples, key=lambda x: x["overlap"], reverse=True)[:5]:
            logger.info(f"Sample {sample['sample_id']}: {sample['overlap']:.2%} overlap")
    
    return overlap_count/val_samples

def verify_training_steps(seq_length=512, batch_size=8):
    """Verify the expected number of training steps per epoch"""
    logger.info("Verifying training steps with non-overlapping sequences...")
    
    # Load train dataset
    logger.info("Loading training dataset...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    train_dataset = WikiText103Dataset(
        split='train', 
        seq_length=seq_length, 
        tokenizer=tokenizer
    )
    
    # Calculate expected steps with given batch size
    total_samples = len(train_dataset)
    steps_per_epoch = (total_samples + batch_size - 1) // batch_size  # Ceiling division
    
    logger.info(f"Dataset configuration: seq_length={seq_length}, batch_size={batch_size}")
    logger.info(f"Total samples in train dataset: {total_samples}")
    logger.info(f"Expected steps per epoch: {steps_per_epoch}")
    
    # Save results for comparison
    results_dir = os.path.join(".", "verification_results")
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, "steps_verification.json")
    
    # Load previous results if they exist
    previous_results = {}
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                previous_results = json.load(f)
                logger.info(f"Loaded previous verification results for comparison")
        except Exception as e:
            logger.warning(f"Could not load previous results: {str(e)}")
    
    # Get today's result
    today = time.strftime("%Y-%m-%d")
    
    # Update results
    current_results = {
        "date": today,
        "seq_length": seq_length,
        "batch_size": batch_size,
        "total_samples": total_samples,
        "steps_per_epoch": steps_per_epoch
    }
    
    # Compare with previous results if available
    if "before_fix" in previous_results:
        before = previous_results["before_fix"]
        if "steps_per_epoch" in before:
            before_steps = before["steps_per_epoch"]
            ratio = steps_per_epoch / before_steps if before_steps > 0 else 0
            
            logger.info(f"Step count comparison:")
            logger.info(f"  Before fix: {before_steps} steps/epoch")
            logger.info(f"  After fix:  {steps_per_epoch} steps/epoch")
            logger.info(f"  Ratio:      {ratio:.2f} (should be ~0.5 if fixed correctly)")
            
            if 0.45 <= ratio <= 0.55:
                logger.info("✅ VALIDATION PASSED: Step count is approximately half after the fix")
            else:
                logger.warning("⚠️ VALIDATION FAILED: Step count ratio is not ~0.5 as expected")
    
    # Save or update results
    if "after_fix" not in previous_results:
        previous_results["after_fix"] = current_results
        with open(results_file, 'w') as f:
            json.dump(previous_results, f, indent=2)
            logger.info(f"Saved verification results to {results_file}")
    
    return steps_per_epoch

def evaluate_multiple_baselines(seq_length=512, device='cpu'):
    """Evaluate multiple baseline models for comparison"""
    logger.info("Evaluating multiple baseline models for comparison...")
    
    results = {}
    
    # 1. GPT-2 Tiny (comparable to SignalLLM)
    logger.info("1. Evaluating GPT-2 Tiny baseline:")
    gpt2_tiny_ppl = evaluate_gpt2_baseline(
        batch_size=1, 
        seq_length=seq_length, 
        device=device, 
        model_config=GPT2Config(
            vocab_size=50257,
            n_positions=1024,
            n_embd=256,    # Small embedding dim
            n_layer=4,     # Fewer layers
            n_head=8       # Standard head count
        )
    )
    results["gpt2_tiny"] = gpt2_tiny_ppl
    
    # 2. Slightly larger GPT-2 Small variant
    logger.info("2. Evaluating GPT-2 Small-like baseline:")
    gpt2_small_ppl = evaluate_gpt2_baseline(
        batch_size=1, 
        seq_length=seq_length, 
        device=device,
        model_config=GPT2Config(
            vocab_size=50257,
            n_positions=1024,
            n_embd=384,    # Medium embedding dim
            n_layer=6,     # More layers
            n_head=12      # More heads
        )
    )
    results["gpt2_small"] = gpt2_small_ppl
    
    # Log comparison table
    logger.info("\nBaseline perplexity comparison:")
    logger.info(f"{'Model':<20} {'Parameters':<15} {'Perplexity':<10}")
    logger.info(f"{'-'*45}")
    logger.info(f"{'GPT-2 Tiny':<20} {'~17M':<15} {gpt2_tiny_ppl:<10.2f}")
    logger.info(f"{'GPT-2 Small-like':<20} {'~60M':<15} {gpt2_small_ppl:<10.2f}")
    
    # Save results
    results_dir = os.path.join(".", "verification_results")
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, "baseline_comparison.json")
    
    # Add timestamp
    results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    results["seq_length"] = seq_length
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
        logger.info(f"\nSaved baseline results to {results_file}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify perplexity calculation and dataset integrity")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--perplexity", action="store_true", help="Test perplexity calculation")
    parser.add_argument("--baseline", action="store_true", help="Evaluate baseline model")
    parser.add_argument("--multiple-baselines", action="store_true", help="Evaluate multiple baseline models")
    parser.add_argument("--overlap", action="store_true", help="Check dataset overlap")
    parser.add_argument("--verify-steps", action="store_true", help="Verify training step counts")
    parser.add_argument("--before-fix", action="store_true", help="Record training steps before fix was applied")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu, cuda, mps)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--seq-length", type=int, default=512, help="Sequence length for evaluation")
    
    args = parser.parse_args()
    
    # Default to all tests if none specified
    run_all = args.all or not (args.perplexity or args.baseline or args.overlap or 
                              args.multiple_baselines or args.verify_steps)
    
    if run_all or args.perplexity:
        test_perplexity_calculation()
    
    if run_all or args.overlap:
        overlap_percentage = check_dataset_overlap(args.seq_length)
        if overlap_percentage > 0.05:  # More than 5% overlap is concerning
            logger.warning("⚠️ WARNING: Significant dataset overlap detected!")
    
    if run_all or args.verify_steps:
        steps = verify_training_steps(args.seq_length, args.batch_size)
        
        # Record as "before fix" if requested
        if args.before_fix:
            results_dir = os.path.join(".", "verification_results")
            os.makedirs(results_dir, exist_ok=True)
            results_file = os.path.join(results_dir, "steps_verification.json")
            
            # Create or update results file
            results = {}
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results = json.load(f)
            
            # Add before fix results
            results["before_fix"] = {
                "date": time.strftime("%Y-%m-%d"),
                "seq_length": args.seq_length,
                "batch_size": args.batch_size,
                "steps_per_epoch": steps
            }
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
                logger.info(f"Recorded pre-fix step count to {results_file}")
    
    # Determine device
    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS requested but not available. Falling back to CPU.")
        device = "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    
    if run_all or args.baseline:
        baseline_ppl = evaluate_gpt2_baseline(args.batch_size, args.seq_length, device)
        
        # Add perspective on expected perplexity for this size model
        if baseline_ppl < 10:
            logger.warning("⚠️ WARNING: Baseline perplexity is suspiciously low!")
        elif baseline_ppl < 20:
            logger.warning("⚠️ WARNING: Baseline perplexity is better than state-of-the-art!")
        elif baseline_ppl < 100:
            logger.info("Baseline perplexity is in the range of well-tuned large language models")
        else:
            logger.info("Baseline perplexity is in the expected range for a small model")
    
    if run_all or args.multiple_baselines:
        evaluate_multiple_baselines(args.seq_length, device) 