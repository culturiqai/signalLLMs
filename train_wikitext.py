#!/usr/bin/env python3
# train_wikitext.py - Train SignalLLM on WikiText dataset with MPS optimizations

import os
import time
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Import SignalLLM components
from signalllm_hrfevo_poc import SignalLLM, Config, SimpleTokenizer, count_parameters
from mps_optimizations import setup_mps_optimizations, optimize_for_mps

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check MPS availability
MPS_AVAILABLE = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
if MPS_AVAILABLE:
    logger.info("Apple Silicon MPS is available and will be used.")
    setup_mps_optimizations()
else:
    logger.info("Apple Silicon MPS not available. Using CPU fallback.")

class WikiTextDataset(Dataset):
    """WikiText dataset for language modeling"""
    def __init__(self, text_path, tokenizer, seq_length=512, stride=256):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.stride = stride
        
        # Load and tokenize text
        logger.info(f"Loading text from {text_path}")
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tokenize full text
        logger.info("Tokenizing text...")
        self.tokens = tokenizer.encode(text)
        
        # Create samples with stride
        self.samples = []
        for i in range(0, len(self.tokens) - seq_length, stride):
            self.samples.append(self.tokens[i:i + seq_length])
        
        logger.info(f"Created {len(self.samples)} samples with length {seq_length}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Get token sequence
        tokens = self.samples[idx]
        
        # Create input and target tensors
        x = torch.tensor(tokens[:-1], dtype=torch.long)  # All but last token
        y = torch.tensor(tokens[1:], dtype=torch.long)   # All but first token
        
        return x, y

def download_wikitext(output_dir="./data"):
    """Download WikiText dataset if not already present"""
    import urllib.request
    import zipfile
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if dataset already exists
    extracted_dir = os.path.join(output_dir, "wikitext-2")
    if os.path.exists(extracted_dir):
        logger.info(f"WikiText dataset already exists at {extracted_dir}")
        return extracted_dir
    
    # WikiText-2 dataset (using HuggingFace as alternative source)
    wikitext_url = "https://huggingface.co/datasets/wikitext/resolve/main/wikitext-2-raw-v1.zip"
    zip_path = os.path.join(output_dir, "wikitext-2-v1.zip")
    
    try:
        # Download dataset
        logger.info(f"Downloading WikiText dataset from {wikitext_url}")
        urllib.request.urlretrieve(wikitext_url, zip_path)
        
        # Extract dataset
        logger.info(f"Extracting dataset to {output_dir}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
            
        # Check if directory structure is as expected (may be different from original)
        if not os.path.exists(extracted_dir):
            # Create the expected directory structure
            os.makedirs(extracted_dir, exist_ok=True)
            
            # Find extracted files and move/rename them as needed
            raw_dir = os.path.join(output_dir, "wikitext-2-raw-v1")
            if os.path.exists(raw_dir):
                logger.info(f"Reorganizing files from {raw_dir} to {extracted_dir}")
                
                # Map files to expected names
                file_mapping = {
                    "wiki.train.raw": "wiki.train.tokens",
                    "wiki.valid.raw": "wiki.valid.tokens",
                    "wiki.test.raw": "wiki.test.tokens"
                }
                
                import shutil
                for src, dst in file_mapping.items():
                    src_path = os.path.join(raw_dir, src)
                    dst_path = os.path.join(extracted_dir, dst)
                    if os.path.exists(src_path):
                        shutil.copy2(src_path, dst_path)
        
        logger.info(f"WikiText dataset downloaded and extracted to {extracted_dir}")
        return extracted_dir
        
    except Exception as e:
        logger.error(f"Error downloading WikiText dataset: {e}")
        
        # Fallback - try alternate URL
        try:
            alt_url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"
            logger.info(f"Trying alternate URL: {alt_url}")
            urllib.request.urlretrieve(alt_url, zip_path)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
                
            logger.info(f"WikiText dataset downloaded and extracted to {extracted_dir}")
            return extracted_dir
            
        except Exception as e2:
            logger.error(f"Error with fallback download: {e2}")
            
            # Create dummy data for testing if everything fails
            logger.warning("Creating minimal dummy WikiText data for testing")
            os.makedirs(extracted_dir, exist_ok=True)
            
            dummy_text = "= WikiText Test =\n\nThis is a test article.\nIt contains several sentences that can be used for testing language models.\n"
            for split in ["train", "valid", "test"]:
                with open(os.path.join(extracted_dir, f"wiki.{split}.tokens"), "w") as f:
                    # Create slightly larger dummy data for training
                    if split == "train":
                        f.write(dummy_text * 100)
                    else:
                        f.write(dummy_text * 10)
            
            return extracted_dir

def train_epoch(model, dataloader, optimizer, device, epoch, args):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    start_time = time.time()
    
    # Training loop with progress bar
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{args.epochs}")
    
    for step, (x, y) in pbar:
        # Move data to device
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        logits = model(x)
        
        # Compute loss
        loss = F.cross_entropy(logits.view(-1, model.config.vocab_size), y.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        
        # Update weights
        optimizer.step()
        
        # Update progress bar
        total_loss += loss.item()
        avg_loss = total_loss / (step + 1)
        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
        
        # Save checkpoint every N steps
        if args.save_steps > 0 and (step + 1) % args.save_steps == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{epoch+1}-{step+1}.pt")
            save_checkpoint(model, optimizer, epoch, step, checkpoint_path)
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    elapsed = time.time() - start_time
    logger.info(f"Epoch {epoch+1} completed in {elapsed:.2f}s - Avg loss: {avg_loss:.4f}")
    
    return avg_loss

def evaluate(model, dataloader, device):
    """Evaluate model on dataloader"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Evaluating"):
            # Move data to device
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits = model(x)
            
            # Compute loss
            loss = F.cross_entropy(logits.view(-1, model.config.vocab_size), y.view(-1))
            total_loss += loss.item()
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    perplexity = np.exp(avg_loss)
    
    logger.info(f"Evaluation - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
    
    return avg_loss, perplexity

def save_checkpoint(model, optimizer, epoch, step, path):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Prepare checkpoint data
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": model.config.to_dict() if hasattr(model, "config") else None
    }
    
    # Save checkpoint
    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved to {path}")

def load_checkpoint(path, model, optimizer=None):
    """Load model checkpoint"""
    logger.info(f"Loading checkpoint from {path}")
    
    # Load checkpoint
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    
    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint["epoch"], checkpoint.get("step", 0)

def plot_training_results(train_losses, val_losses, perplexities, output_dir):
    """Plot training results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot loss
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "training_loss.png"))
    plt.close()
    
    # Plot perplexity
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, perplexities, 'g-')
    plt.title('Validation Perplexity')
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "validation_perplexity.png"))
    plt.close()

def train_wikitext(args):
    """Train SignalLLM on WikiText dataset"""
    # Set device
    device = torch.device("mps" if MPS_AVAILABLE else "cpu")
    logger.info(f"Using device: {device}")
    
    # Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download dataset if needed
    dataset_dir = download_wikitext()
    
    # Prepare data paths
    train_path = os.path.join(dataset_dir, "wiki.train.tokens")
    val_path = os.path.join(dataset_dir, "wiki.valid.tokens")
    test_path = os.path.join(dataset_dir, "wiki.test.tokens")
    
    # Create tokenizer
    tokenizer = SimpleTokenizer(mode='char')
    
    # Build vocabulary from training data
    logger.info("Building vocabulary from training data...")
    with open(train_path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokenizer.build_vocab([text], max_vocab_size=args.vocab_size)
    vocab_size = tokenizer.get_vocab_size()
    logger.info(f"Vocabulary size: {vocab_size}")
    
    # Save tokenizer for later use
    tokenizer_path = os.path.join(args.output_dir, "tokenizer.json")
    tokenizer.save_vocab(tokenizer_path)
    
    # Create datasets
    train_dataset = WikiTextDataset(train_path, tokenizer, seq_length=args.seq_length)
    val_dataset = WikiTextDataset(val_path, tokenizer, seq_length=args.seq_length)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Create model configuration
    config = Config(
        vocab_size=vocab_size,
        max_seq_length=args.seq_length,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        use_signal_embed=args.use_signal_embed,
        use_wavelet_attention=args.use_wavelet_attention
    )
    
    # Create model
    model = SignalLLM(config).to(device)
    
    # Apply MPS optimizations if available
    if MPS_AVAILABLE and args.use_mps:
        logger.info("Applying MPS optimizations...")
        model = optimize_for_mps(model, device=device)
    
    # Print model parameters
    num_params = count_parameters(model)
    logger.info(f"Model parameters: {num_params:,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * len(train_loader),
        eta_min=args.learning_rate * 0.1
    )
    
    # Load checkpoint if specified
    start_epoch = 0
    if args.resume_from:
        start_epoch, _ = load_checkpoint(args.resume_from, model, optimizer)
    
    # Training loop
    train_losses = []
    val_losses = []
    perplexities = []
    best_perplexity = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        # Train epoch
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, args)
        train_losses.append(train_loss)
        
        # Evaluate
        val_loss, perplexity = evaluate(model, val_loader, device)
        val_losses.append(val_loss)
        perplexities.append(perplexity)
        
        # Save best model
        if perplexity < best_perplexity:
            best_perplexity = perplexity
            best_model_path = os.path.join(args.output_dir, "best_model.pt")
            save_checkpoint(model, optimizer, epoch, len(train_loader), best_model_path)
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{epoch+1}.pt")
        save_checkpoint(model, optimizer, epoch, len(train_loader), checkpoint_path)
        
        # Plot results
        plot_training_results(train_losses, val_losses, perplexities, args.output_dir)
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model.pt")
    save_checkpoint(model, optimizer, args.epochs, 0, final_model_path)
    
    # Evaluate on test set
    test_dataset = WikiTextDataset(test_path, tokenizer, seq_length=args.seq_length)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    test_loss, test_perplexity = evaluate(model, test_loader, device)
    
    # Log final results
    logger.info(f"Training completed. Final test perplexity: {test_perplexity:.4f}")
    
    # Save results summary
    results = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "perplexities": perplexities,
        "test_loss": test_loss,
        "test_perplexity": test_perplexity,
        "best_perplexity": best_perplexity,
        "num_parameters": num_params,
        "vocab_size": vocab_size
    }
    
    import json
    with open(os.path.join(args.output_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {args.output_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train SignalLLM on WikiText dataset")
    
    # Data parameters
    parser.add_argument("--vocab_size", type=int, default=10000,
                       help="Maximum vocabulary size")
    parser.add_argument("--seq_length", type=int, default=512,
                       help="Sequence length for training")
    
    # Model parameters
    parser.add_argument("--embed_dim", type=int, default=512,
                       help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=8,
                       help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=4,
                       help="Number of transformer layers")
    parser.add_argument("--use_signal_embed", action="store_true",
                       help="Use signal embedding instead of standard embedding")
    parser.add_argument("--use_wavelet_attention", action="store_true",
                       help="Use wavelet-based attention")
    parser.add_argument("--use_mps", action="store_true",
                       help="Use MPS optimizations if available")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Maximum gradient norm for clipping")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of worker processes for data loading")
    parser.add_argument("--save_steps", type=int, default=0,
                       help="Save checkpoint every N steps (0 to disable)")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Resume training from checkpoint")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./wikitext_output",
                       help="Output directory for checkpoints and results")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_wikitext(args) 