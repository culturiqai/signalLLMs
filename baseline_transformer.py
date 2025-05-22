#!/usr/bin/env python3
"""
Baseline Transformer for WikiText-103 Comparison

This script trains a standard transformer on WikiText-103 with identical
parameters to SignalLLM for fair comparison.
"""

import os
import sys
import math
import logging
import argparse
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

# Import the same dataset class for fair comparison
from train_wikitext103 import WikiText103Dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class StandardTransformerConfig:
    def __init__(self, vocab_size=50257, embed_dim=256, hidden_dim=512, 
                 num_heads=8, num_layers=4, seq_length=512, dropout=0.1):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.dropout = dropout

class StandardTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.seq_length, config.embed_dim)
        
        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(config.embed_dim, config.vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, x):
        # Get embeddings
        seq_len = x.size(1)
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand_as(x)
        
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(pos_ids)
        
        # Combine embeddings
        embeddings = self.dropout(token_emb + pos_emb)
        
        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        
        # Apply transformer
        output = self.transformer(embeddings, mask=mask, is_causal=True)
        
        # Project to vocabulary
        logits = self.output_projection(output)
        
        return logits

def train_baseline_transformer(args):
    """Train baseline transformer"""
    logger.info("Starting baseline transformer training")
    
    # Set device
    if args.use_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = WikiText103Dataset(
        split='train', 
        seq_length=args.seq_length, 
        tokenizer=tokenizer,
        cache_dir=args.data_dir
    )
    
    val_dataset = WikiText103Dataset(
        split='validation', 
        seq_length=args.seq_length, 
        tokenizer=tokenizer,
        cache_dir=args.data_dir
    )
    
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
    
    # Create model
    config = StandardTransformerConfig(
        vocab_size=len(tokenizer),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        seq_length=args.seq_length,
        dropout=args.dropout
    )
    
    model = StandardTransformer(config).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    
    # Create optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    
    def lr_lambda(current_step):
        if current_step < args.warmup_steps:
            return float(current_step) / float(max(1, args.warmup_steps))
        return max(0.1, float(total_steps - current_step) / float(max(1, total_steps - args.warmup_steps)))
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    history = []
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        epoch_train_loss = 0.0
        epoch_train_accuracy = 0.0
        train_batches = 0
        train_total_tokens = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            # Calculate metrics
            preds = torch.argmax(outputs, dim=-1)
            correct = (preds == targets).float().sum().item()
            total = targets.numel()
            accuracy = correct / total
            
            epoch_train_loss += loss.item() * total
            epoch_train_accuracy += accuracy * total
            train_batches += 1
            train_total_tokens += total
            
            # Update progress bar
            current_ppl = torch.exp(torch.tensor(loss.item())).item()
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'ppl': f'{current_ppl:.2f}',
                'acc': f'{accuracy:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
        
        # Calculate epoch training metrics
        epoch_train_loss = epoch_train_loss / train_total_tokens
        epoch_train_ppl = torch.exp(torch.tensor(epoch_train_loss)).item()
        epoch_train_accuracy = epoch_train_accuracy / train_total_tokens
        
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_accuracy = 0.0
        val_batches = 0
        val_total_tokens = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validating"):
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
                
                preds = torch.argmax(outputs, dim=-1)
                correct = (preds == targets).float().sum().item()
                total = targets.numel()
                accuracy = correct / total
                
                epoch_val_loss += loss.item() * total
                epoch_val_accuracy += accuracy * total
                val_batches += 1
                val_total_tokens += total
        
        # Calculate epoch validation metrics
        epoch_val_loss = epoch_val_loss / val_total_tokens
        epoch_val_ppl = torch.exp(torch.tensor(epoch_val_loss)).item()
        epoch_val_accuracy = epoch_val_accuracy / val_total_tokens
        
        # Print epoch summary
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        logger.info(f"  Train - Loss: {epoch_train_loss:.4f}, PPL: {epoch_train_ppl:.2f}, Acc: {epoch_train_accuracy:.4f}")
        logger.info(f"  Val   - Loss: {epoch_val_loss:.4f}, PPL: {epoch_val_ppl:.2f}, Acc: {epoch_val_accuracy:.4f}")
        
        # Save metrics
        history.append({
            'epoch': epoch + 1,
            'train_loss': epoch_train_loss,
            'train_ppl': epoch_train_ppl,
            'train_accuracy': epoch_train_accuracy,
            'val_loss': epoch_val_loss,
            'val_ppl': epoch_val_ppl,
            'val_accuracy': epoch_val_accuracy,
            'learning_rate': scheduler.get_last_lr()[0]
        })
        
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_path = os.path.join(args.output_dir, "baseline_best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best baseline model with val loss: {best_val_loss:.4f}")
    
    # Save results
    results = {
        'model_type': 'baseline_transformer',
        'config': vars(config),
        'training_args': vars(args),
        'history': history,
        'best_val_loss': best_val_loss
    }
    
    results_path = os.path.join(args.output_dir, "baseline_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Baseline results saved to {results_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline transformer on WikiText-103")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--seq_length", type=int, default=512)
    
    # Model parameters
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_mps", action="store_true")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./wikitext103_baseline")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_baseline_transformer(args) 