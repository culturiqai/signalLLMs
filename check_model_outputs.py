#!/usr/bin/env python
# Script to analyze the model's output distributions

import os
import torch
import logging
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

# Configure matplotlib for headless environment
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Import our model classes
from train_wikitext103 import WikiText103Dataset
from signalllm_hrfevo_poc import SignalLLM, Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_path, device):
    """Load the trained SignalLLM model"""
    logger.info(f"Loading model from {model_path}")
    
    # First load config
    config_path = os.path.join(os.path.dirname(model_path), "config.json")
    if os.path.exists(config_path):
        logger.info(f"Loading config from {config_path}")
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
    
    # Create model
    model = SignalLLM(config)
    
    # Load weights from checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check if this is a full checkpoint dictionary (includes optimizer, scheduler etc.)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        logger.info("Loading from full training checkpoint")
        model.load_state_dict(checkpoint['model'])
    else:
        logger.info("Loading from model-only checkpoint")
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    
    return model

def analyze_output_distribution(model, data_loader, device, tokenizer, num_samples=10):
    """Analyze the model's output probability distributions"""
    model.eval()
    
    all_entropies = []
    all_top_probs = []
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            if i >= num_samples:
                break
                
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Get model outputs
            outputs = model(inputs)  # [batch_size, seq_len, vocab_size]
            
            # Convert to probabilities
            probs = torch.softmax(outputs, dim=-1)
            
            # Get top-k probabilities and indices for each position
            top_k = 5
            top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)
            
            # Calculate entropy of the distribution at each position
            # H(p) = -sum(p_i * log(p_i))
            distribution_entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            
            # Store values for analysis
            all_entropies.append(distribution_entropy.cpu().numpy())
            all_top_probs.append(top_probs.cpu().numpy())
            
            # Analyze a few positions in detail
            batch_idx = 0  # First item in batch
            for pos_idx in [0, inputs.size(1)//2, inputs.size(1)-1]:  # Start, middle, end
                logger.info(f"\nSample {i}, Position {pos_idx}")
                
                # Get input context
                context_start = max(0, pos_idx - 5)
                context_tokens = inputs[batch_idx, context_start:pos_idx+1].cpu().tolist()
                context_text = tokenizer.decode(context_tokens)
                logger.info(f"Context: '{context_text}'")
                
                # Get actual next token
                next_token = targets[batch_idx, pos_idx].item()
                next_token_text = tokenizer.decode([next_token])
                logger.info(f"Actual next token: '{next_token_text}' (id: {next_token})")
                
                # Get model's predictions
                top_5_probs = top_probs[batch_idx, pos_idx].tolist()
                top_5_tokens = top_indices[batch_idx, pos_idx].cpu().tolist()
                top_5_texts = [tokenizer.decode([idx]) for idx in top_5_tokens]
                
                logger.info("Model's top predictions:")
                for j, (token_text, token_id, prob) in enumerate(zip(top_5_texts, top_5_tokens, top_5_probs)):
                    logger.info(f"  {j+1}. '{token_text}' (id: {token_id}) - probability: {prob:.4f}")
                
                # Get rank and probability of the actual next token
                actual_prob = probs[batch_idx, pos_idx, next_token].item()
                logger.info(f"Probability of actual token: {actual_prob:.6f}")
                
                # Calculate entropy at this position
                entropy = distribution_entropy[batch_idx, pos_idx].item()
                logger.info(f"Entropy of distribution: {entropy:.4f}")
                
                # Check if the distribution is suspiciously concentrated
                max_prob = top_probs[batch_idx, pos_idx, 0].item()
                if max_prob > 0.9:
                    logger.warning(f"SUSPICIOUS: Top probability is very high ({max_prob:.4f})")
    
    # Calculate and display statistics
    all_entropies = np.concatenate([e.flatten() for e in all_entropies])
    all_top1_probs = np.concatenate([p[:,:,0].flatten() for p in all_top_probs])
    
    logger.info("\nOutput Distribution Statistics:")
    logger.info(f"Average entropy: {np.mean(all_entropies):.4f} ± {np.std(all_entropies):.4f}")
    logger.info(f"Average top-1 probability: {np.mean(all_top1_probs):.4f} ± {np.std(all_top1_probs):.4f}")
    logger.info(f"Median top-1 probability: {np.median(all_top1_probs):.4f}")
    
    # Create histograms
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(all_entropies, bins=50)
    plt.title('Distribution of Entropy')
    plt.xlabel('Entropy')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    plt.hist(all_top1_probs, bins=50)
    plt.title('Distribution of Top-1 Probability')
    plt.xlabel('Probability')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('output_distribution_analysis.png')
    logger.info("Saved distribution plots to output_distribution_analysis.png")
    
    return {
        'mean_entropy': np.mean(all_entropies),
        'std_entropy': np.std(all_entropies),
        'mean_top1_prob': np.mean(all_top1_probs),
        'median_top1_prob': np.median(all_top1_probs),
        'std_top1_prob': np.std(all_top1_probs)
    }

def check_memorization(model, train_loader, val_loader, device, num_samples=10):
    """Check if the model is memorizing training data by comparing train/val loss distributions"""
    model.eval()
    
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    
    train_losses = []
    val_losses = []
    
    logger.info("Checking for memorization (comparing train vs validation loss distributions)...")
    
    # Get training losses
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(train_loader):
            if i >= num_samples:
                break
                
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
            train_losses.append(loss.cpu().numpy())
    
    # Get validation losses
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            if i >= num_samples:
                break
                
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
            val_losses.append(loss.cpu().numpy())
    
    # Flatten arrays
    train_losses = np.concatenate(train_losses)
    val_losses = np.concatenate(val_losses)
    
    # Calculate statistics
    logger.info("\nLoss Distribution Statistics:")
    logger.info(f"Train loss: mean={np.mean(train_losses):.4f}, median={np.median(train_losses):.4f}, std={np.std(train_losses):.4f}")
    logger.info(f"Val loss: mean={np.mean(val_losses):.4f}, median={np.median(val_losses):.4f}, std={np.std(val_losses):.4f}")
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(train_losses, bins=50, alpha=0.5, label='Training')
    plt.hist(val_losses, bins=50, alpha=0.5, label='Validation')
    plt.title('Loss Distribution: Training vs Validation')
    plt.xlabel('Loss')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig('memorization_analysis.png')
    logger.info("Saved memorization analysis plot to memorization_analysis.png")
    
    # Perform statistical test
    from scipy.stats import ttest_ind
    t_stat, p_value = ttest_ind(train_losses, val_losses, equal_var=False)
    
    logger.info(f"Statistical test: t={t_stat:.4f}, p-value={p_value:.8f}")
    if p_value < 0.001:
        logger.info("CONCLUSION: Significant difference between train and val losses (p<0.001)")
        if np.mean(train_losses) < np.mean(val_losses):
            logger.info("The model performs BETTER on training data than validation data, suggesting MEMORIZATION.")
        else:
            logger.info("Unexpectedly, the model performs BETTER on validation data than training data.")
    else:
        logger.info("CONCLUSION: No significant difference between train and val losses")
    
    return {
        'train_mean_loss': np.mean(train_losses),
        'train_median_loss': np.median(train_losses),
        'val_mean_loss': np.mean(val_losses),
        'val_median_loss': np.median(val_losses),
        'p_value': p_value
    }

def main():
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    logger.info(f"Using device: {device}")
    
    # Get latest model from wikitext103_signalllm_fixed directory
    model_dir = "./wikitext103_signalllm_fixed"
    
    # Find checkpoint files
    checkpoint_files = [f for f in os.listdir(model_dir) if f.endswith('.pt') and f != 'best_model.pt']
    
    if 'best_model.pt' in os.listdir(model_dir):
        model_path = os.path.join(model_dir, 'best_model.pt')
    elif checkpoint_files:
        # Sort by modification time (newest first)
        checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
        model_path = os.path.join(model_dir, checkpoint_files[0])
    else:
        logger.error(f"No model checkpoints found in {model_dir}")
        return
    
    logger.info(f"Using model: {model_path}")
    
    # Load model
    model = load_model(model_path, device)
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Create datasets with smaller sequence length for faster processing
    seq_length = 128
    batch_size = 2
    
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
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Analyze model's output distributions
    analyze_output_distribution(model, val_loader, device, tokenizer, num_samples=5)
    
    # Check for memorization
    check_memorization(model, train_loader, val_loader, device, num_samples=10)

if __name__ == "__main__":
    main() 