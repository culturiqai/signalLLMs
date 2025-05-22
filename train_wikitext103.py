#!/usr/bin/env python3
"""
Train SignalLLM on WikiText-103 Dataset

This script trains the SignalLLM model on the WikiText-103 dataset,
which is a much larger dataset than WikiText-2 used in the initial proof-of-concept.
It includes proper support for HRFEvo tensor optimization and checkpointing
for long training runs.
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
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import random
import traceback  # Add traceback for better error reporting
from typing import Dict, Any, Optional, List

# Configure debug mode
DEBUG = False  # Reduced to minimize log spam

# Notes on metrics calculation:
# - Perplexity (PPL) is correctly calculated as exp(average_loss) not average(exp(loss))
# - The dataset now uses non-overlapping sequences to prevent data leakage
# - Loss and accuracy are weighted by token count for proper averaging
# - Accuracy is less important than perplexity for language modeling tasks

# Import SignalLLM implementation
from signalllm_hrfevo_poc import Config, SignalLLM, HRFEvoController

# Check if MPS is available (Apple Silicon)
try:
    MPS_AVAILABLE = torch.backends.mps.is_available() 
    if MPS_AVAILABLE:
        from mps_optimizations import optimize_for_mps, setup_mps_optimizations
        import pywt
        print(f"PyWavelets version: {pywt.__version__}")
        print("Using Apple Silicon MPS (Metal Performance Shaders)")
except Exception as e:
    MPS_AVAILABLE = False
    print(f"MPS initialization error: {e}")

# Setup logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,  # Set to DEBUG for more information
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("wikitext103_training.log")
    ]
)
logger = logging.getLogger(__name__)

# Add a custom exception handler
def log_exception(e, context=""):
    error_msg = f"ERROR in {context}: {str(e)}\n{traceback.format_exc()}"
    logger.error(error_msg)
    with open("wikitext103_training_error.log", "a") as f:
        f.write(f"\n{'='*50}\n{time.strftime('%Y-%m-%d %H:%M:%S')}\n{error_msg}\n")
    return error_msg

class WikiText103Dataset(Dataset):
    """Dataset class for WikiText-103"""
    
    def __init__(self, split='train', seq_length=1024, tokenizer=None, cache_dir='./data'):
        """Initialize dataset
        
        Args:
            split (str): Dataset split ('train', 'validation', or 'test')
            seq_length (int): Sequence length for training
            tokenizer: Tokenizer to use
            cache_dir (str): Directory to cache dataset
        """
        logger.info(f"Initializing WikiText-103 dataset ({split} split)")
        
        # Load dataset from huggingface datasets
        dataset = load_dataset('wikitext', 'wikitext-103-v1', split=split, cache_dir=cache_dir)
        
        # Store tokenizer
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        # Tokenize all texts
        logger.info("Tokenizing texts...")
        self.token_ids = []
        
        for item in tqdm(dataset, desc="Tokenizing"):
            text = item['text']
            if text.strip():  # Skip empty lines
                tokens = torch.tensor(tokenizer(text).input_ids, dtype=torch.long)
                if len(tokens) > 0:
                    self.token_ids.append(tokens)
        
        # Concatenate all chunks
        logger.info("Concatenating chunks...")
        self.all_tokens = torch.cat(self.token_ids)
        
        # Get vocabulary size
        self.vocab_size = len(self.tokenizer)
        logger.info(f"Vocabulary size: {self.vocab_size}")
        logger.info(f"Total tokens: {len(self.all_tokens)}")
        
        # Create samples of sequence_length
        self.samples = []
        
        # CRITICAL FIX: Always use non-overlapping sequences (stride = seq_length)
        # This ensures we get ~230K steps per epoch instead of ~460K
        stride = seq_length
        logger.info(f"Using non-overlapping sequences (stride = {stride})")
        
        # Create non-overlapping samples
        for i in range(0, len(self.all_tokens) - seq_length - 1, stride):
            self.samples.append(i)
        
        # Enforce validation: First basic check on sample count
        expected_samples = (len(self.all_tokens) - seq_length - 1) // stride
        actual_samples = len(self.samples)
        
        # Allow at most 1% difference due to edge effects
        margin = expected_samples * 0.01
        assert abs(expected_samples - actual_samples) <= margin, \
            f"Sample count validation failed: expected ~{expected_samples}, got {actual_samples}"
        
        # Second validation: Check stride between samples
        if len(self.samples) >= 2:
            actual_stride = self.samples[1] - self.samples[0]
            assert actual_stride == stride, \
                f"Stride validation failed: expected {stride}, got {actual_stride}"
        
        logger.info(f"Created {len(self.samples)} samples from {split} split (stride={stride})")
        logger.info(f"Validated dataset construction: ✓ non-overlapping sequences (stride={stride})")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        start_idx = self.samples[idx]
        end_idx = start_idx + self.seq_length + 1
        tokens = self.all_tokens[start_idx:end_idx]
        
        # Create input and target tensors
        x = tokens[:-1]  # All but last token
        y = tokens[1:]   # All but first token
        
        return x, y

def create_optimizer_and_scheduler(model, args):
    """Create optimizer and learning rate scheduler"""
    # Define optimizer
    if args.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Learning rate scheduler with warmup
    def lr_lambda(current_step):
        if current_step < args.warmup_steps:
            return float(current_step) / float(max(1, args.warmup_steps))
        return max(
            1e-5,
            float(args.epochs * args.steps_per_epoch - current_step) / 
            float(max(1, args.epochs * args.steps_per_epoch - args.warmup_steps))
        )
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler

def load_checkpoint(model, optimizer, scheduler, hrfevo_controller, checkpoint_path):
    """Load model and training state from checkpoint"""
    if not os.path.exists(checkpoint_path):
        return 0, 0, []
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    # Load model state
    model.load_state_dict(checkpoint['model'])
    
    # Load optimizer state
    if 'optimizer' in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Load scheduler state
    if 'scheduler' in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    # Load HRFEvo controller state
    if 'hrfevo' in checkpoint and hrfevo_controller is not None:
        hrfevo_controller.load_state(checkpoint['hrfevo'])
    
    # Load training history
    history = checkpoint.get('history', [])
    epoch = checkpoint.get('epoch', 0)
    global_step = checkpoint.get('global_step', 0)
    
    logger.info(f"Resuming from epoch {epoch}, step {global_step}")
    return epoch, global_step, history

def save_checkpoint(model, optimizer, scheduler, hrfevo_controller, 
                   epoch, global_step, history, checkpoint_path):
    """Save model and training state to checkpoint"""
    logger.info(f"Saving checkpoint to {checkpoint_path}")
    
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'epoch': epoch,
        'global_step': global_step,
        'history': history
    }
    
    # Save HRFEvo controller state if available
    if hrfevo_controller is not None:
        checkpoint['hrfevo'] = hrfevo_controller.save_state()
    
    # Save model checkpoint
    torch.save(checkpoint, checkpoint_path)
    
    # Save training configuration
    config_path = os.path.join(os.path.dirname(checkpoint_path), "config.json")
    with open(config_path, 'w') as f:
        json.dump(model.get_config(), f, indent=2)

def train_wikitext103(args):
    """Train SignalLLM on WikiText-103 dataset"""
    try:
        # Setup logging
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=getattr(logging, args.log_level),
            format=log_format,
            handlers=[
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)
        
        # Log configuration summary
        logger.info(f"Training SignalLLM with config: vocab_size={args.vocab_size}, embed_dim={args.embed_dim}, layers={args.num_layers}")
        
        # Set device
        MPS_AVAILABLE = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        if args.use_mps and MPS_AVAILABLE:
            logger.info("Apple Silicon MPS is available and will be used.")
            device = torch.device('mps')
        else:
            if args.use_mps and not MPS_AVAILABLE:
                logger.warning("MPS requested but not available. Falling back to CUDA if available.")
            
            if torch.cuda.is_available():
                logger.info("CUDA is available and will be used.")
                device = torch.device('cuda')
            else:
                logger.info("No GPU available, using CPU.")
                device = torch.device('cpu')
        
        logger.info(f"Using device: {device}")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Get tokenizer (using GPT-2 tokenizer)
        logger.debug("Initializing tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        vocab_size = len(tokenizer)
        logger.info(f"Vocabulary size: {vocab_size}")
        
        # Create datasets
        logger.debug("Creating train dataset...")
        train_dataset = WikiText103Dataset(
            split='train', 
            seq_length=args.seq_length, 
            tokenizer=tokenizer,
            cache_dir=args.data_dir
        )
        
        logger.debug("Creating validation dataset...")
        val_dataset = WikiText103Dataset(
            split='validation', 
            seq_length=args.seq_length, 
            tokenizer=tokenizer,
            cache_dir=args.data_dir
        )
        
        # Create data loaders
        logger.debug("Creating data loaders...")
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
        
        # Calculate steps per epoch for scheduler
        args.steps_per_epoch = len(train_loader)
        total_steps = args.epochs * args.steps_per_epoch
        logger.info(f"Training for {args.epochs} epochs, {args.steps_per_epoch} steps per epoch, {total_steps} total steps")
        
        # Create model configuration
        logger.debug("Creating model configuration...")
        config = Config(
            vocab_size=vocab_size,
            max_seq_length=args.seq_length,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            use_signal_embed=args.use_signal_embed,
            use_wavelet_attention=args.use_wavelet_attention,
            harmonic_bases=args.harmonic_bases,
            wavelet_type=args.wavelet_type,
            wavelet_levels=args.wavelet_levels,
            use_adaptive_basis=args.use_adaptive_basis,
            wavelet_families=args.wavelet_families.split(',') if args.wavelet_families else ['db4', 'sym4', 'dmey'],
            use_fourier_convolution=args.use_fourier_convolution,
            use_stochastic_transform=args.use_stochastic_transform,
            spectral_gap_analysis=args.spectral_gap_analysis,
            evolution_population=args.evolution_population,
            evolution_generations=args.evolution_generations
        )
        
        # Create model
        logger.debug("Creating model...")
        model = SignalLLM(config)
        
        # Log model parameters before moving to device
        params_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Model created with {params_count:,} parameters")
        
        # Verify tensor shapes before optimization
        logger.debug("Verifying initial tensor shapes...")
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.debug(f"Parameter {name}: shape={param.shape}, device={param.device}")
        
        logger.debug(f"Moving model to device: {device}")
        try:
            model = model.to(device)
            logger.debug("Model successfully moved to device")
        except Exception as e:
            error = log_exception(e, "moving model to device")
            logger.warning(f"Failed to move model to {device}, falling back to CPU. Error: {error}")
            device = torch.device('cpu')
            model = model.to(device)
        
        # Apply MPS optimizations if requested
        if args.use_mps and MPS_AVAILABLE:
            logger.info("Applying MPS optimizations...")
            try:
                model = optimize_for_mps(model)
                logger.debug("MPS optimization applied successfully")
            except Exception as e:
                error = log_exception(e, "applying MPS optimizations")
                logger.warning(f"Failed to apply MPS optimizations: {error}")
        
        # Initialize HRFEvo controller if evolution is enabled
        hrfevo_controller = None
        if args.enable_evolution:
            logger.info("Initializing HRFEvo controller...")
            try:
                hrfevo_controller = HRFEvoController(config)
                logger.debug("HRFEvo controller initialized successfully")
            except Exception as e:
                error = log_exception(e, "initializing HRFEvo controller")
                logger.warning(f"Failed to initialize HRFEvo controller: {error}")
                args.enable_evolution = False
                
        # Create optimizer and scheduler
        logger.debug("Creating optimizer and scheduler...")
        optimizer, scheduler = create_optimizer_and_scheduler(model, args)
        
        # Setup loss function
        criterion = nn.CrossEntropyLoss()
        
        # Load checkpoint if it exists
        checkpoint_path = os.path.join(args.output_dir, "latest_checkpoint.pt")
        start_epoch, global_step, history = load_checkpoint(
            model, optimizer, scheduler, hrfevo_controller, checkpoint_path
        )
        
        # Training loop
        best_val_loss = float('inf')
        
        # Try a small forward pass to verify the model works
        logger.debug("Attempting test forward pass...")
        try:
            model.eval()
            with torch.no_grad():
                # Create a small test input
                test_input = torch.randint(0, vocab_size, (2, args.seq_length), device=device)
                logger.debug(f"Test input shape: {test_input.shape}, device: {test_input.device}")
                
                # Forward pass
                logger.debug("Starting forward pass")
                test_output = model(test_input)
                logger.debug(f"Test output shape: {test_output.shape}")
                logger.info("Test forward pass successful")
        except Exception as e:
            error = log_exception(e, "test forward pass")
            logger.error(f"Test forward pass failed: {error}")
            raise RuntimeError(f"Model forward pass failed: {error}")
        
        # Check if input loader will actually work by fetching a batch
        logger.debug("Testing data loader...")
        try:
            test_batch = next(iter(train_loader))
            logger.debug(f"Sample batch shapes: inputs={test_batch[0].shape}, targets={test_batch[1].shape}")
        except Exception as e:
            error = log_exception(e, "testing data loader")
            logger.error(f"Failed to get a batch from data loader: {error}")
            raise RuntimeError(f"Data loader failed: {error}")
            
        # Now proceed with training
        for epoch in range(start_epoch, args.epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_accuracy = 0.0
            train_batches = 0
            train_total_tokens = 0
            
            # Progress bar
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                            desc=f"Epoch {epoch+1}/{args.epochs}")
            
            epoch_start_time = time.time()
            
            # Add flag file checking function
            def check_for_flag_files():
                """Check for flag files requesting model save and/or training stop"""
                flag_dir = os.path.join(args.output_dir, "..", "saved_models")
                os.makedirs(flag_dir, exist_ok=True)
                
                for filename in os.listdir(flag_dir):
                    if filename.startswith("save_and_stop_") and filename.endswith(".flag"):
                        flag_path = os.path.join(flag_dir, filename)
                        
                        # Read the flag file
                        try:
                            with open(flag_path, "r") as f:
                                flag_data = json.load(f)
                                
                            # Check if this flag is for our process
                            pid_to_stop = flag_data.get("pid_to_stop")
                            if pid_to_stop is not None and str(os.getpid()) != str(pid_to_stop):
                                continue
                                
                            # Save the model
                            timestamp = flag_data.get("timestamp", time.strftime("%Y%m%d_%H%M%S"))
                            save_path = os.path.join(flag_dir, f"model_save_{timestamp}.pt")
                            
                            logger.info(f"Flag file detected: {filename}. Saving model to {save_path}")
                            
                            # Save model and training state
                            save_checkpoint(
                                model, optimizer, scheduler, hrfevo_controller,
                                epoch, global_step, history, save_path
                            )
                            
                            # Create completion indicator
                            completion_path = f"{flag_path}.completed"
                            with open(completion_path, "w") as f:
                                json.dump({
                                    "model_path": save_path,
                                    "timestamp": time.strftime("%Y%m%d_%H%M%S"),
                                    "stats": {
                                        "epoch": epoch,
                                        "global_step": global_step,
                                        "loss": epoch_train_loss if 'epoch_train_loss' in locals() else None,
                                        "perplexity": epoch_train_ppl if 'epoch_train_ppl' in locals() else None,
                                        "accuracy": epoch_train_accuracy if 'epoch_train_accuracy' in locals() else None
                                    }
                                }, f)
                                
                            # Check if we should stop training
                            if flag_data.get("command") == "save_and_stop":
                                logger.info("Stopping training as requested by flag file")
                                return True  # Signal to stop training
                                
                            # Remove the flag file to prevent reprocessing
                            try:
                                os.remove(flag_path)
                            except:
                                logger.warning(f"Could not remove flag file {flag_path}")
                        except Exception as e:
                            error = log_exception(e, f"processing flag file {filename}")
                            logger.error(f"Error processing flag file: {error}")
                
                return False  # Continue training
            
            for batch_idx, (inputs, targets) in progress_bar:
                # Check for save/stop flags every 10 batches
                if batch_idx % 10 == 0 and check_for_flag_files():
                    logger.info("Training stopped by flag file")
                    return  # Exit training function
                
                try:
                    # Move to device
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # Zero gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    logger.debug(f"Batch {batch_idx}: Forward pass starting")
                    outputs = model(inputs)
                    logger.debug(f"Forward pass complete: output shape={outputs.shape}")
                    
                    # Calculate loss
                    loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
                    logger.debug(f"Loss calculated: {loss.item()}")
                    
                    # Backward pass
                    loss.backward()
                    logger.debug("Backward pass complete")
                    
                    # Clip gradients
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    
                    # Update weights
                    optimizer.step()
                    scheduler.step()
                    logger.debug(f"Batch {batch_idx} update complete")
                    
                    # Calculate metrics
                    with torch.no_grad():
                        preds = torch.argmax(outputs, dim=-1)
                        correct = (preds == targets).float().sum().item()
                        total = targets.numel()
                        accuracy = correct / total
                        
                        train_loss += loss.item() * total  # Weight by tokens
                        train_accuracy += accuracy * total  # Weight by tokens
                        train_batches += 1
                        train_total_tokens += total
                        
                        current_loss = loss.item()
                        current_ppl = torch.exp(torch.tensor(current_loss)).item()
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{current_loss:.4f}",
                        'ppl': f"{current_ppl:.2f}",
                        'acc': f"{accuracy:.4f}",
                        'lr': scheduler.get_last_lr()[0]
                    })
                    
                    # Increment global step
                    global_step += 1
                    
                    # Save memory usage information
                    if DEBUG and batch_idx % 100 == 0:
                        if device.type == 'cuda':
                            memory_allocated = torch.cuda.memory_allocated(device) / 1024**2
                            memory_reserved = torch.cuda.memory_reserved(device) / 1024**2
                            logger.debug(f"CUDA Memory: Allocated={memory_allocated:.2f}MB, Reserved={memory_reserved:.2f}MB")
                    
                    # Run HRFEvo basis optimization if enabled
                    if hrfevo_controller and global_step % args.evolution_steps == 0:
                        logger.info(f"Running HRFEvo optimization at step {global_step}...")
                        
                        # Define evaluation function for basis functions
                        def evaluate_basis(basis):
                            # Apply basis function to model temporarily
                            original_basis = model.current_basis
                            model.update_basis_function(basis)
                            
                            # Evaluate on a small batch
                            model.eval()
                            with torch.no_grad():
                                eval_data, eval_targets = next(iter(val_loader))
                                eval_data, eval_targets = eval_data.to(device), eval_targets.to(device)
                                
                                # Forward pass
                                start_time = time.time()
                                outputs = model(eval_data)
                                inference_time = time.time() - start_time
                                
                                # Calculate metrics
                                loss = criterion(outputs.reshape(-1, outputs.size(-1)), 
                                            eval_targets.reshape(-1))
                                perplexity = torch.exp(loss).item()
                                
                                # Get complexity info
                                complexity_info = model.get_complexity_info()
                                
                                # Calculate efficiency score
                                if ('attention_op_ratios' in complexity_info and 
                                    complexity_info['attention_op_ratios']):
                                    avg_ratio = sum(complexity_info['attention_op_ratios']) / len(complexity_info['attention_op_ratios'])
                                    computation_efficiency = avg_ratio / inference_time
                                else:
                                    computation_efficiency = 1.0 / inference_time
                            
                            # Restore original basis
                            model.update_basis_function(original_basis)
                            model.train()
                            
                            return {
                                'perplexity': perplexity,
                                'computation_efficiency': computation_efficiency
                            }
                        
                        try:
                            # Run one generation of evolution
                            stats = hrfevo_controller.evolve_generation(evaluate_basis)
                            
                            # Update model with best basis
                            best_basis = hrfevo_controller.get_best_basis()
                            if best_basis:
                                model.update_basis_function(best_basis)
                                logger.info(f"Updated model with evolved basis. Fitness: {stats['best_fitness']:.4f}")
                        except Exception as e:
                            error = log_exception(e, "HRFEvo evolution")
                            logger.warning(f"HRFEvo evolution failed: {error}")
                            # Continue training without evolution
                    
                    # Save checkpoint periodically
                    if args.checkpoint_every > 0 and global_step % args.checkpoint_every == 0:
                        step_checkpoint_path = os.path.join(args.output_dir, f"checkpoint_step_{global_step}.pt")
                        save_checkpoint(
                            model, optimizer, scheduler, hrfevo_controller,
                            epoch, global_step, history, step_checkpoint_path
                        )
                
                except Exception as e:
                    error = log_exception(e, f"training batch {batch_idx}")
                    logger.error(f"Error processing batch {batch_idx}: {error}")
                    if DEBUG:
                        # Save problematic tensors for analysis
                        debug_dir = os.path.join(args.output_dir, "debug")
                        os.makedirs(debug_dir, exist_ok=True)
                        try:
                            torch.save({
                                'inputs': inputs.cpu(),
                                'targets': targets.cpu(),
                                'model_state': model.state_dict()
                            }, os.path.join(debug_dir, f"error_batch_{batch_idx}.pt"))
                            logger.debug(f"Saved error debug info to {debug_dir}")
                        except:
                            logger.debug("Could not save debug tensors")
                    # Continue with next batch
                    continue
            
            # Calculate epoch metrics
            epoch_train_loss = train_loss / max(train_total_tokens, 1)
            epoch_train_ppl = torch.exp(torch.tensor(epoch_train_loss)).item()
            epoch_train_accuracy = train_accuracy / max(train_total_tokens, 1)
            epoch_time = time.time() - epoch_start_time
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_accuracy = 0.0
            val_batches = 0
            val_total_tokens = 0  # Track total number of tokens for proper weighting
            
            with torch.no_grad():
                for inputs, targets in tqdm(val_loader, desc="Validation"):
                    try:
                        # Move to device
                        inputs, targets = inputs.to(device), targets.to(device)
                        
                        # Forward pass
                        outputs = model(inputs)
                        
                        # Calculate loss
                        loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
                        
                        # Calculate metrics
                        preds = torch.argmax(outputs, dim=-1)
                        correct = (preds == targets).float().sum().item()
                        total = targets.numel()
                        accuracy = correct / total
                        
                        # Note: This accuracy metric only measures exact matches of the highest probability token
                        # It doesn't fully capture the quality of the probabilistic distribution, which is better
                        # reflected by perplexity. For language modeling, perplexity is the primary metric.
                        val_loss += loss.item() * total  # Weight loss by number of tokens
                        val_accuracy += accuracy * total  # Weight accuracy by number of tokens
                        val_batches += 1
                        val_total_tokens += total
                    except Exception as e:
                        error = log_exception(e, "validation batch")
                        logger.error(f"Error in validation: {error}")
                        # Continue with next batch
                        continue
            
            # Calculate validation metrics
            epoch_val_loss = val_loss / max(val_total_tokens, 1)  # Properly weighted average loss
            epoch_val_ppl = torch.exp(torch.tensor(epoch_val_loss)).item()  # Compute perplexity from average loss
            epoch_val_accuracy = val_accuracy / max(val_total_tokens, 1)  # Properly weighted average accuracy
            
            # Print epoch summary
            logger.info(f"Epoch {epoch+1}/{args.epochs} - Time: {epoch_time:.2f}s")
            logger.info(f"  Train - Loss: {epoch_train_loss:.4f}, PPL: {epoch_train_ppl:.2f}, Acc: {epoch_train_accuracy:.4f}")
            logger.info(f"  Val   - Loss: {epoch_val_loss:.4f}, PPL: {epoch_val_ppl:.2f}, Acc: {epoch_val_accuracy:.4f}")
            
            # Save metrics to history
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
            
            # Save checkpoint
            save_checkpoint(
                model, optimizer, scheduler, hrfevo_controller,
                epoch + 1, global_step, history, checkpoint_path
            )
            
            # Save best model
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_model_path = os.path.join(args.output_dir, "best_model.pt")
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"Saved best model with val loss: {best_val_loss:.4f}")
        
        # Final evaluation
        logger.info("Training complete. Final evaluation on test set...")
        
        # Create test dataset
        test_dataset = WikiText103Dataset(
            split='test', 
            seq_length=args.seq_length, 
            tokenizer=tokenizer,
            cache_dir=args.data_dir
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        
        # Load best model
        best_model_path = os.path.join(args.output_dir, "best_model.pt")
        model.load_state_dict(torch.load(best_model_path))
        
        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        test_accuracy = 0.0
        test_batches = 0
        test_total_tokens = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc="Testing"):
                try:
                    # Move to device
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # Forward pass
                    outputs = model(inputs)
                    
                    # Calculate loss
                    loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
                    
                    # Calculate metrics
                    preds = torch.argmax(outputs, dim=-1)
                    correct = (preds == targets).float().sum().item()
                    total = targets.numel()
                    accuracy = correct / total
                    
                    test_loss += loss.item() * total  # Weight by tokens
                    test_accuracy += accuracy * total  # Weight by tokens
                    test_batches += 1
                    test_total_tokens += total
                except Exception as e:
                    error = log_exception(e, "test batch")
                    logger.error(f"Error in test evaluation: {error}")
                    # Continue with next batch
                    continue
        
        # Calculate test metrics
        epoch_test_loss = test_loss / max(test_total_tokens, 1)
        epoch_test_ppl = torch.exp(torch.tensor(epoch_test_loss)).item()
        epoch_test_accuracy = test_accuracy / max(test_total_tokens, 1)
        
        # Print test summary
        logger.info(f"Test - Loss: {epoch_test_loss:.4f}, PPL: {epoch_test_ppl:.2f}, Acc: {epoch_test_accuracy:.4f}")
        
        # Save test results
        results = {
            'test_loss': epoch_test_loss,
            'test_ppl': epoch_test_ppl,
            'test_accuracy': epoch_test_accuracy,
            'history': history
        }
        
        results_path = os.path.join(args.output_dir, "results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
    except Exception as e:
        error = log_exception(e, "train_wikitext103 main function")
        logger.critical(f"Training aborted due to error: {error}")
        sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Train SignalLLM on WikiText-103 dataset")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="./data",
                       help="Directory to store dataset cache")
    parser.add_argument("--vocab_size", type=int, default=50000,
                       help="Maximum vocabulary size")
    parser.add_argument("--seq_length", type=int, default=1024,
                       help="Sequence length for training")
    
    # Model parameters
    parser.add_argument("--embed_dim", type=int, default=512,
                       help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=2048,
                       help="Hidden layer dimension")
    parser.add_argument("--num_heads", type=int, default=8,
                       help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=8,
                       help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate")
    parser.add_argument("--use_signal_embed", action="store_true",
                       help="Use signal embedding instead of standard embedding")
    parser.add_argument("--use_wavelet_attention", action="store_true",
                       help="Use wavelet-based attention")
    parser.add_argument("--use_mps", action="store_true",
                       help="Use MPS optimizations if available")
    
    # Spectral configuration
    parser.add_argument("--harmonic_bases", type=int, default=32,
                       help="Number of harmonic bases for spectral embedding")
    parser.add_argument("--wavelet_type", type=str, default="db4",
                       help="Wavelet type for wavelet-based attention")
    parser.add_argument("--wavelet_levels", type=int, default=3,
                       help="Number of wavelet decomposition levels")
    parser.add_argument("--use_adaptive_basis", action="store_true",
                       help="Use adaptive basis selection")
    parser.add_argument("--wavelet_families", type=str, default="db4,sym4,dmey",
                       help="Comma-separated list of wavelet families")
    parser.add_argument("--use_fourier_convolution", action="store_true",
                       help="Use Fourier convolution attention")
    parser.add_argument("--use_stochastic_transform", action="store_true",
                       help="Use stochastic wavelet transform for long sequences")
    parser.add_argument("--spectral_gap_analysis", action="store_true",
                       help="Enable spectral gap analysis")
    
    # HRFEvo parameters
    parser.add_argument("--enable_evolution", action="store_true",
                       help="Enable HRFEvo during training")
    parser.add_argument("--evolution_population", type=int, default=10,
                       help="Population size for HRFEvo")
    parser.add_argument("--evolution_generations", type=int, default=5,
                       help="Number of generations for HRFEvo")
    parser.add_argument("--evolution_steps", type=int, default=1000,
                       help="Run evolution every N steps")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Maximum gradient norm for clipping")
    parser.add_argument("--warmup_steps", type=int, default=2000,
                       help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--optimizer", type=str, default="adamw",
                       help="Optimizer to use (adam or adamw)")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of worker processes for data loading")
    parser.add_argument("--checkpoint_every", type=int, default=1000,
                       help="Save checkpoint every N steps (0 to disable)")
    parser.add_argument("--output_dir", type=str, default="./wikitext103_signalllm",
                       help="Output directory for model checkpoints and results")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Logging level")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_wikitext103(args) 