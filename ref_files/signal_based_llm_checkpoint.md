"""
Enhanced Signal-Based Language Model Architecture with Checkpoints
=================================================================

This implementation scales up the signal-based LLM concept with:
1. Training on actual text data (WikiText-103)
2. Checkpointing for interrupted training
3. Prompt testing functionality
4. Benchmarking against standard transformer architectures
5. Visualization of learned spectral representations

The model represents language in the frequency domain rather than as discrete tokens only.
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import argparse
import json
from torch.utils.tensorboard import SummaryWriter

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device setup with MPS support for Apple Silicon
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon MPS (Metal Performance Shaders)")
else:
    device = torch.device("cpu")
    print("Using CPU")


class SpectralEmbedding(nn.Module):
    """
    Embedding layer that represents tokens as superpositions of frequency components.
    Instead of learning a direct embedding vector, learns amplitudes and phases for 
    different frequency components.
    """
    def __init__(self, vocab_size, embed_dim, harmonic_bases=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.harmonic_bases = harmonic_bases
        
        # Each token gets frequency amplitudes and phases
        self.frequency_amplitudes = nn.Parameter(torch.randn(vocab_size, harmonic_bases))
        self.frequency_phases = nn.Parameter(torch.randn(vocab_size, harmonic_bases))
        
        # Generate frequency bases (fixed)
        self.register_buffer('frequencies', 
                           torch.linspace(0.1, math.pi, harmonic_bases))
    
    def forward(self, x):
        # x: batch_size x seq_length
        batch_size, seq_length = x.shape
        
        # Get amplitudes and phases for each token
        amplitudes = self.frequency_amplitudes[x]  # batch_size x seq_length x harmonic_bases
        phases = self.frequency_phases[x]  # batch_size x seq_length x harmonic_bases
        
        # Generate time points for embedding dimension
        t = torch.linspace(0, 1, self.embed_dim, device=x.device)
        t = t.view(1, 1, 1, self.embed_dim)  # 1 x 1 x 1 x embed_dim
        
        # Each frequency contributes across the embedding dimension
        frequencies = self.frequencies.view(1, 1, self.harmonic_bases, 1)  # 1 x 1 x harmonic_bases x 1
        amplitudes = amplitudes.unsqueeze(-1)  # batch x seq_len x harmonic_bases x 1
        phases = phases.unsqueeze(-1)  # batch x seq_len x harmonic_bases x 1
        
        # Generate embeddings as superposition of harmonics
        signal = amplitudes * torch.sin(2 * math.pi * frequencies * t + phases)
        embeddings = signal.sum(dim=2)  # batch x seq_len x embed_dim
        
        return embeddings


class SignalPositionalEncoding(nn.Module):
    """
    Positional encoding using basis of sinusoidal signals at different frequencies.
    Extends the standard sinusoidal encoding with learnable parameters.
    """
    def __init__(self, max_seq_length, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Create standard sinusoidal encoding as starting point
        pe = torch.zeros(max_seq_length, embed_dim)
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                            -(math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Make parameters learnable with initial sinusoidal values
        self.pe = nn.Parameter(pe.unsqueeze(0))
        
        # Frequency modulation parameters
        self.freq_mod = nn.Parameter(torch.ones(1, 1, embed_dim))
        self.phase_shift = nn.Parameter(torch.zeros(1, 1, embed_dim))
    
    def forward(self, x):
        # x: batch_size x seq_length x embed_dim
        seq_length = x.size(1)
        
        # Apply frequency modulation and phase shift
        pos_encoding = self.pe[:, :seq_length] * self.freq_mod + self.phase_shift
        
        return x + pos_encoding


class FourierAttention(nn.Module):
    """
    Attention mechanism that operates partially in the frequency domain.
    Uses Fourier transforms to process queries and keys before computing attention.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Projection matrices
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Learnable frequency filters for each head
        self.freq_filter = nn.Parameter(torch.randn(num_heads, self.head_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.scale = (self.head_dim) ** -0.5
    
    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.shape
        
        # Project queries, keys, values
        q = self.q_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        
        # Reshape for batch-wise operations
        q = q.transpose(1, 2)  # batch x heads x seq x dim
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Transform to frequency domain for filtering
        q_fft = torch.fft.rfft(q, dim=2)  # Batch x Heads x Freq x Dim
        k_fft = torch.fft.rfft(k, dim=2)  # Batch x Heads x Freq x Dim
        
        # Apply learnable frequency filtering
        filter_shaped = self.freq_filter.unsqueeze(0).unsqueeze(2)  # 1 x heads x 1 x dim
        q_fft = q_fft * filter_shaped
        k_fft = k_fft * filter_shaped
        
        # Transform back to time domain
        q_filtered = torch.fft.irfft(q_fft, n=seq_length, dim=2)
        k_filtered = torch.fft.irfft(k_fft, n=seq_length, dim=2)
        
        # Standard attention calculation
        attn_weights = torch.matmul(q_filtered, k_filtered.transpose(2, 3)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1)  # Add head dimension
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)  # batch x heads x seq x dim
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)
        
        return self.out_proj(output)


class SpectralFFN(nn.Module):
    """
    Feed-forward network that operates partially in the frequency domain.
    Applies different transformations to different frequency components.
    """
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Traditional projection matrices
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        
        # Frequency-selective processing
        fft_size = hidden_dim // 2 + 1  # Size of rfft output for hidden_dim
        self.low_pass = nn.Parameter(torch.ones(1, 1, fft_size) * 0.8)
        self.high_pass = nn.Parameter(torch.zeros(1, 1, fft_size) + 0.2)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Traditional feedforward part
        h = F.gelu(self.fc1(x))
        h = self.dropout(h)
        
        # Apply frequency-domain processing
        h_fft = torch.fft.rfft(h, dim=-1)
        
        # Frequency-selective filtering
        h_fft_filtered = h_fft * (self.low_pass + self.high_pass)
        
        # Back to time domain
        h_filtered = torch.fft.irfft(h_fft_filtered, n=self.hidden_dim, dim=-1)
        
        # Final projection
        return self.fc2(h_filtered)


class SignalTransformerBlock(nn.Module):
    """A single transformer block with signal-based components"""
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.attn = FourierAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = SpectralFFN(embed_dim, hidden_dim, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Attention with residual connection
        attn_output = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # FFN with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x 


class SignalBasedLLM(nn.Module):
    """
    Complete signal-based language model architecture.
    Uses spectral embeddings, Fourier attention, and spectral feed-forward networks.
    """
    def __init__(self, vocab_size, max_seq_length, embed_dim, num_heads, 
                 num_layers, hidden_dim, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Token embeddings - spectral version
        self.token_embedding = SpectralEmbedding(vocab_size, embed_dim, harmonic_bases=16)
        
        # Positional encoding
        self.pos_encoding = SignalPositionalEncoding(max_seq_length, embed_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            SignalTransformerBlock(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output head
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x, mask=None):
        # Get embeddings
        x = self.token_embedding(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Process through transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final normalization and projection
        x = self.norm(x)
        return self.output_proj(x)
    
    def count_parameters(self):
        """Count the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def generate(self, prompt_ids, max_length=50, temperature=0.7, top_k=40, top_p=0.9, repetition_penalty=1.2):
        """Generate text from a prompt"""
        self.eval()
        with torch.no_grad():
            # Start with the prompt
            input_ids = prompt_ids.clone()
            
            # Generate tokens until max_length or EOS
            for _ in range(max_length):
                # Get model predictions
                logits = self(input_ids.unsqueeze(0))
                next_token_logits = logits[0, -1, :].clone()
                
                # Apply repetition penalty to discourage repetitions
                # This penalizes tokens that have already been generated
                if repetition_penalty > 1.0:
                    # For each token already generated, reduce its probability
                    for prev_token in input_ids:
                        if next_token_logits[prev_token] > 0:
                            next_token_logits[prev_token] /= repetition_penalty
                        else:
                            next_token_logits[prev_token] *= repetition_penalty
                
                # Apply temperature
                next_token_logits = next_token_logits / (temperature if temperature > 0 else 1.0)
                
                # Apply filtering
                if top_k > 0 or top_p > 0:
                    # Create logits distribution
                    if top_k > 0:
                        # Get indices of the top k probabilities
                        top_k_values, _ = torch.topk(next_token_logits, top_k)
                        # Remove all tokens with a probability less than the last token of the top-k
                        indices_to_remove = next_token_logits < top_k_values[-1]
                        next_token_logits.masked_fill_(indices_to_remove, float("-inf"))
                    
                    if top_p > 0:
                        # Sort logits in descending order
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        # Shift the indices to the right to keep also the first token above the threshold
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        # Create a boolean mask of the same shape as next_token_logits
                        indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool)
                        # Set the positions that need to be removed to True
                        indices_to_remove.scatter_(0, sorted_indices[sorted_indices_to_remove], True)
                        next_token_logits.masked_fill_(indices_to_remove, float("-inf"))
                
                # Sample from the filtered distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Add the next token to the sequence
                input_ids = torch.cat([input_ids, next_token])
                
                # Check for end of sequence (optional, depending on your tokenizer)
                # if next_token.item() == eos_token_id:
                #     break
            
            return input_ids


# Standard Transformer for comparison
class StandardTransformer(nn.Module):
    """Standard transformer model for benchmarking"""
    def __init__(self, vocab_size, max_seq_length, embed_dim, num_heads,
                num_layers, hidden_dim, dropout=0.1):
        super().__init__()
        
        # Standard token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Standard positional encoding
        self.pos_encoding = nn.Parameter(
            torch.zeros(1, max_seq_length, embed_dim)
        )
        nn.init.normal_(self.pos_encoding, mean=0, std=0.02)
        
        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # Output head
        self.output_proj = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x, mask=None):
        # Get embeddings
        x = self.token_embedding(x)
        
        # Add positional encoding
        seq_length = x.size(1)
        x = x + self.pos_encoding[:, :seq_length]
        
        # Process through transformer
        if mask is not None:
            # Convert boolean mask to float attention mask
            attn_mask = torch.zeros_like(mask, dtype=torch.float)
            attn_mask.masked_fill_(~mask, float('-inf'))
            x = self.transformer_encoder(x, src_key_padding_mask=~mask)
        else:
            x = self.transformer_encoder(x)
        
        # Final projection
        return self.output_proj(x)
    
    def count_parameters(self):
        """Count the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def generate(self, prompt_ids, max_length=50, temperature=0.7, top_k=40, top_p=0.9, repetition_penalty=1.2):
        """Generate text from a prompt"""
        self.eval()
        with torch.no_grad():
            # Start with the prompt
            input_ids = prompt_ids.clone()
            
            # Generate tokens until max_length or EOS
            for _ in range(max_length):
                # Get model predictions
                logits = self(input_ids.unsqueeze(0))
                next_token_logits = logits[0, -1, :].clone()
                
                # Apply repetition penalty to discourage repetitions
                # This penalizes tokens that have already been generated
                if repetition_penalty > 1.0:
                    # For each token already generated, reduce its probability
                    for prev_token in input_ids:
                        if next_token_logits[prev_token] > 0:
                            next_token_logits[prev_token] /= repetition_penalty
                        else:
                            next_token_logits[prev_token] *= repetition_penalty
                
                # Apply temperature
                next_token_logits = next_token_logits / (temperature if temperature > 0 else 1.0)
                
                # Apply filtering
                if top_k > 0 or top_p > 0:
                    # Create logits distribution
                    if top_k > 0:
                        # Get indices of the top k probabilities
                        top_k_values, _ = torch.topk(next_token_logits, top_k)
                        # Remove all tokens with a probability less than the last token of the top-k
                        indices_to_remove = next_token_logits < top_k_values[-1]
                        next_token_logits.masked_fill_(indices_to_remove, float("-inf"))
                    
                    if top_p > 0:
                        # Sort logits in descending order
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        # Shift the indices to the right to keep also the first token above the threshold
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        # Create a boolean mask of the same shape as next_token_logits
                        indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool)
                        # Set the positions that need to be removed to True
                        indices_to_remove.scatter_(0, sorted_indices[sorted_indices_to_remove], True)
                        next_token_logits.masked_fill_(indices_to_remove, float("-inf"))
                
                # Sample from the filtered distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Add the next token to the sequence
                input_ids = torch.cat([input_ids, next_token])
                
                # Check for end of sequence (optional, depending on your tokenizer)
                # if next_token.item() == eos_token_id:
                #     break
            
            return input_ids 


# Data processing for WikiText-103 using HuggingFace datasets
class WikiTextDataset(Dataset):
    """WikiText-103 dataset for language modeling using HuggingFace datasets"""
    def __init__(self, split='train', seq_length=128, data_dir='./data'):
        self.seq_length = seq_length
        self.data_dir = data_dir
        
        # Make sure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Setup tokenizer - using a simple tokenizer from HuggingFace
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Load dataset from HuggingFace
        print(f"Loading WikiText-103 {split} dataset...")
        dataset_split = "train" if split == "train" else "validation" if split == "valid" else "test"
        self.dataset = load_dataset("wikitext", "wikitext-103-v1", split=dataset_split)
        
        # Process the dataset
        print("Processing dataset...")
        token_ids_chunks = []
        
        # Process in chunks to avoid tokenizer length limits
        for item in tqdm(self.dataset):
            if len(item['text'].strip()) > 0:  # Skip empty lines
                # Tokenize each text separately to avoid length issues
                tokens = self.tokenizer(item['text'], return_tensors="pt").input_ids.squeeze(0)
                if len(tokens) > 0:
                    token_ids_chunks.append(tokens)
        
        # Concatenate all chunks
        print("Concatenating chunks...")
        self.token_ids = torch.cat(token_ids_chunks)
        
        # Create vocabulary size
        self.vocab_size = len(self.tokenizer)
        print(f"Vocabulary size: {self.vocab_size}")
        
        # Create samples of sequence_length + 1 (for targets)
        self.samples = []
        for i in range(0, len(self.token_ids) - seq_length, seq_length):
            self.samples.append(self.token_ids[i:i + seq_length + 1])
        
        print(f"Created {len(self.samples)} samples from {split} split")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Get sequence and target (next token prediction)
        sample = self.samples[idx]
        x = sample[:-1]
        y = sample[1:]
        return x, y
    
    def encode_prompt(self, text):
        """Encode a text prompt to token ids"""
        return self.tokenizer.encode(text, return_tensors="pt").squeeze(0)
    
    def decode_tokens(self, token_ids):
        """Decode token ids to text"""
        return self.tokenizer.decode(token_ids)


def create_padding_mask(batch, pad_idx=1):
    """Create mask for padding tokens"""
    return (batch != pad_idx)


def save_checkpoint(model, optimizer, scheduler, epoch, iteration, loss, args, model_name, output_dir):
    """Save a checkpoint for resuming training later"""
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_checkpoint_e{epoch}_i{iteration}.pt")
    
    checkpoint = {
        'epoch': epoch,
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'args': vars(args)
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save the latest checkpoint path in a metadata file
    metadata = {
        'latest_checkpoint': checkpoint_path,
        'epoch': epoch,
        'iteration': iteration,
        'loss': loss
    }
    
    with open(os.path.join(checkpoint_dir, f"{model_name}_latest.json"), 'w') as f:
        json.dump(metadata, f)
    
    return checkpoint_path


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load a checkpoint to resume training"""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    iteration = checkpoint['iteration']
    loss = checkpoint['loss']
    
    return model, optimizer, scheduler, epoch, iteration, loss


def find_latest_checkpoint(model_name, output_dir):
    """Find the latest checkpoint for a model"""
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    metadata_path = os.path.join(checkpoint_dir, f"{model_name}_latest.json")
    
    if not os.path.exists(metadata_path):
        return None, 0, 0
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    if 'latest_checkpoint' in metadata and os.path.exists(metadata['latest_checkpoint']):
        return metadata['latest_checkpoint'], metadata['epoch'], metadata['iteration']
    
    return None, 0, 0


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, start_iteration=0, 
               checkpoint_interval=1000, args=None, model_name="signal", output_dir="./output", clip_value=1.0):
    """Train for one epoch with checkpointing"""
    model.train()
    total_loss = 0
    total_tokens = 0
    start_time = time.time()
    iteration_in_epoch = 0
    
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    
    for batch_idx, (data, target) in progress_bar:
        # Skip iterations we've already processed if resuming
        if iteration_in_epoch < start_iteration:
            iteration_in_epoch += 1
            continue
        
        data, target = data.to(device), target.to(device)
        batch_size, seq_length = data.shape
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        
        # Calculate loss
        loss = criterion(output.reshape(-1, output.size(-1)), target.reshape(-1))
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item() * (batch_size * seq_length)
        total_tokens += batch_size * seq_length
        
        # Update progress bar
        progress_bar.set_description(
            f"Train Loss: {loss.item():.4f}, "
            f"Perplexity: {math.exp(loss.item()):.2f}"
        )
        
        # Save checkpoint at regular intervals
        iteration_in_epoch += 1
        global_iteration = (epoch * len(dataloader)) + iteration_in_epoch
        
        if iteration_in_epoch % checkpoint_interval == 0:
            save_checkpoint(
                model, optimizer, None, epoch, global_iteration, loss.item(), 
                args, model_name, output_dir
            )
    
    # Calculate average loss and perplexity
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    elapsed = time.time() - start_time
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'time': elapsed,
        'tokens_per_sec': total_tokens / elapsed
    }


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            batch_size, seq_length = data.shape
            
            # Forward pass
            output = model(data)
            
            # Calculate loss
            loss = criterion(output.reshape(-1, output.size(-1)), target.reshape(-1))
            
            # Update statistics
            total_loss += loss.item() * (batch_size * seq_length)
            total_tokens += batch_size * seq_length
    
    # Calculate average loss and perplexity
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity
    }


def test_generation(model, tokenizer, prompts, max_length=50, temperature=0.7, top_k=40, top_p=0.9, repetition_penalty=1.2):
    """Test the model's generation capabilities on a list of prompts"""
    model.eval()
    
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}: \"{prompt}\"")
        
        # Encode the prompt
        prompt_ids = tokenizer.encode(prompt, return_tensors="pt").squeeze(0).to(device)
        
        # Generate continuation
        output_ids = model.generate(
            prompt_ids, 
            max_length=max_length, 
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )
        
        # Decode the generated text
        generated_text = tokenizer.decode(output_ids)
        
        print(f"Generated: {generated_text}")


def visualize_token_frequencies(model, tokens, tokenizer, save_path=None):
    """Visualize the frequency structure of token embeddings"""
    model.eval()
    
    # Convert tokens to indices if they are strings
    if isinstance(tokens[0], str):
        token_ids = [tokenizer.convert_tokens_to_ids(token) for token in tokens]
    else:
        token_ids = tokens
    
    # Convert to tensor
    token_tensor = torch.tensor(token_ids, dtype=torch.long).to(device)
    
    # Get embeddings
    with torch.no_grad():
        # For signal-based model, use the spectral embedding
        if hasattr(model, 'token_embedding') and isinstance(model.token_embedding, SpectralEmbedding):
            embeddings = model.token_embedding(token_tensor.unsqueeze(0)).squeeze(0)
        else:
            # For standard transformer, we'll need to use the embedding layer
            embeddings = model.token_embedding(token_tensor)
        
        # Compute FFT to see frequency components
        embeddings_fft = torch.fft.rfft(embeddings, dim=1)
        power_spectra = torch.abs(embeddings_fft) ** 2
    
    # Plot the power spectra
    plt.figure(figsize=(12, 8))
    
    # Plot mean power spectrum
    mean_power = power_spectra.mean(dim=0).cpu().numpy()
    plt.subplot(2, 1, 1)
    plt.plot(mean_power)
    plt.title("Average Power Spectrum Across Tokens")
    plt.xlabel("Frequency Component")
    plt.ylabel("Power")
    
    # Plot spectrogram (tokens x frequencies)
    plt.subplot(2, 1, 2)
    plt.imshow(power_spectra.cpu().numpy(), aspect='auto', cmap='viridis')
    plt.colorbar(label="Power")
    plt.title("Power Spectrum Per Token")
    plt.xlabel("Frequency Component")
    plt.ylabel("Token Index")
    
    if isinstance(tokens[0], str):
        # Add token labels if tokens are provided as strings
        if len(tokens) <= 20:  # Only label if there aren't too many tokens
            plt.yticks(range(len(tokens)), tokens)
    else:
        # Try to decode token IDs to get readable labels
        try:
            token_texts = [tokenizer.decode(token_id) for token_id in token_ids]
            if len(token_texts) <= 20:
                plt.yticks(range(len(token_texts)), token_texts)
        except:
            pass
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def benchmark_models(signal_model, standard_model, test_loader, device):
    """Benchmark signal model vs standard transformer"""
    # Performance metrics
    signal_model.eval()
    standard_model.eval()
    
    criterion = nn.CrossEntropyLoss()
    
    results = {
        'signal': {},
        'standard': {}
    }
    
    # Parameter count
    results['signal']['params'] = signal_model.count_parameters()
    results['standard']['params'] = standard_model.count_parameters()
    
    # Evaluation metrics
    print("Evaluating Signal-Based Model...")
    signal_start = time.time()
    signal_eval = evaluate(signal_model, test_loader, criterion, device)
    signal_time = time.time() - signal_start
    
    print("Evaluating Standard Transformer...")
    standard_start = time.time()
    standard_eval = evaluate(standard_model, test_loader, criterion, device)
    standard_time = time.time() - standard_start
    
    # Store results
    results['signal'].update(signal_eval)
    results['signal']['eval_time'] = signal_time
    
    results['standard'].update(standard_eval)
    results['standard']['eval_time'] = standard_time
    
    # Inference speed (tokens per second)
    total_tokens = 0
    for data, _ in test_loader:
        batch_size, seq_length = data.shape
        total_tokens += batch_size * seq_length
    
    results['signal']['tokens_per_sec'] = total_tokens / signal_time
    results['standard']['tokens_per_sec'] = total_tokens / standard_time
    
    # Print results
    print("\n=== BENCHMARK RESULTS ===")
    print(f"Parameter count:")
    print(f"  Signal-Based: {results['signal']['params']:,}")
    print(f"  Standard: {results['standard']['params']:,}")
    print(f"  Ratio: {results['signal']['params'] / results['standard']['params']:.2f}x")
    
    print(f"\nPerplexity:")
    print(f"  Signal-Based: {results['signal']['perplexity']:.2f}")
    print(f"  Standard: {results['standard']['perplexity']:.2f}")
    
    print(f"\nInference Speed:")
    print(f"  Signal-Based: {results['signal']['tokens_per_sec']:.2f} tokens/sec")
    print(f"  Standard: {results['standard']['tokens_per_sec']:.2f} tokens/sec")
    print(f"  Ratio: {results['signal']['tokens_per_sec'] / results['standard']['tokens_per_sec']:.2f}x")
    
    return results 


def main():
    """Main function to demonstrate the enhanced Signal-Based LLM with checkpoints"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Enhanced Signal-Based LLM with Checkpoints")
    parser.add_argument("--model", type=str, default="signal", choices=["signal", "standard", "both"],
                       help="Which model architecture to train")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--visualize", action="store_true", help="Visualize embeddings")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    parser.add_argument("--checkpoint_interval", type=int, default=1000, 
                        help="Number of iterations between checkpoints")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default=None, 
                        help="Specific checkpoint to resume from (optional)")
    parser.add_argument("--prompt_test", action="store_true", 
                        help="Test model with prompts after training")
    parser.add_argument("--prompts", nargs="+", type=str, default=[
        "Once upon a time", 
        "The signal-based language model", 
        "In the frequency domain"
    ], help="Prompts to test the model with")
    parser.add_argument("--gen_max_length", type=int, default=50, 
                        help="Maximum length for text generation")
    parser.add_argument("--temperature", type=float, default=0.7, 
                        help="Temperature for text generation")
    parser.add_argument("--top_k", type=int, default=40,
                        help="Top-k value for text generation")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p (nucleus sampling) value for text generation")
    parser.add_argument("--repetition_penalty", type=float, default=1.2,
                        help="Penalty to apply to repeated tokens")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tensorboard"))
    
    # Load data
    print("Loading data...")
    train_dataset = WikiTextDataset(split='train', seq_length=args.seq_length, data_dir=args.data_dir)
    valid_dataset = WikiTextDataset(split='valid', seq_length=args.seq_length, data_dir=args.data_dir)
    test_dataset = WikiTextDataset(split='test', seq_length=args.seq_length, data_dir=args.data_dir)
    
    # Get vocabulary size
    vocab_size = train_dataset.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Model parameters
    model_args = {
        "vocab_size": vocab_size,
        "max_seq_length": args.seq_length,
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout
    }
    
    # Create models
    models = {}
    
    if args.model in ["signal", "both"]:
        print("Creating Signal-Based LLM...")
        signal_model = SignalBasedLLM(**model_args).to(device)
        models["signal"] = signal_model
        print(f"Signal-Based LLM parameters: {signal_model.count_parameters():,}")
    
    if args.model in ["standard", "both"]:
        print("Creating Standard Transformer...")
        standard_model = StandardTransformer(**model_args).to(device)
        models["standard"] = standard_model
        print(f"Standard Transformer parameters: {standard_model.count_parameters():,}")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Train each model
    for model_name, model in models.items():
        print(f"\n=== Training {model_name.capitalize()} Model ===")
        
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=1
        )
        
        # Check if resuming from checkpoint
        start_epoch = 0
        start_iteration = 0
        
        if args.resume:
            if args.checkpoint_path:
                checkpoint_path = args.checkpoint_path
                # Check if the checkpoint exists
                if not os.path.exists(checkpoint_path):
                    print(f"Checkpoint {checkpoint_path} not found. Starting from scratch.")
                else:
                    model, optimizer, scheduler, start_epoch, start_iteration, _ = load_checkpoint(
                        model, optimizer, scheduler, checkpoint_path
                    )
            else:
                # Find latest checkpoint
                checkpoint_path, start_epoch, start_iteration = find_latest_checkpoint(
                    model_name, args.output_dir
                )
                if checkpoint_path:
                    model, optimizer, scheduler, start_epoch, start_iteration, _ = load_checkpoint(
                        model, optimizer, scheduler, checkpoint_path
                    )
                    print(f"Resuming training from epoch {start_epoch}, iteration {start_iteration}")
                else:
                    print("No checkpoint found. Starting from scratch.")
        
        # Training loop
        best_valid_loss = float('inf')
        
        for epoch in range(start_epoch, args.epochs):
            print(f"Epoch {epoch+1}/{args.epochs}")
            
            # Train with checkpointing
            start_iter = start_iteration if epoch == start_epoch else 0
            train_metrics = train_epoch(
                model, train_loader, optimizer, criterion, device, 
                epoch, start_iter, args.checkpoint_interval,
                args, model_name, args.output_dir
            )
            
            # Validate
            valid_metrics = evaluate(model, valid_loader, criterion, device)
            
            # Update learning rate
            scheduler.step(valid_metrics['loss'])
            
            # Log metrics
            for k, v in train_metrics.items():
                writer.add_scalar(f"{model_name}/train/{k}", v, epoch)
            
            for k, v in valid_metrics.items():
                writer.add_scalar(f"{model_name}/valid/{k}", v, epoch)
            
            # Print metrics
            print(f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Train Perplexity: {train_metrics['perplexity']:.2f}")
            print(f"Valid Loss: {valid_metrics['loss']:.4f}, "
                  f"Valid Perplexity: {valid_metrics['perplexity']:.2f}")
            
            # Save best model
            if valid_metrics['loss'] < best_valid_loss:
                best_valid_loss = valid_metrics['loss']
                best_model_path = os.path.join(args.output_dir, f"{model_name}_model_best.pt")
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved to {best_model_path}!")
            
            # Save final checkpoint for this epoch
            save_checkpoint(
                model, optimizer, scheduler, epoch+1, 0, valid_metrics['loss'],
                args, model_name, args.output_dir
            )
        
        # Final evaluation
        print(f"\nFinal evaluation of {model_name.capitalize()} Model")
        test_metrics = evaluate(model, test_loader, criterion, device)
        print(f"Test Loss: {test_metrics['loss']:.4f}, "
              f"Test Perplexity: {test_metrics['perplexity']:.2f}")
        
        # Test generation if requested
        if args.prompt_test:
            print(f"\n=== Testing {model_name.capitalize()} Model Generation ===")
            test_generation(
                model, 
                train_dataset.tokenizer, 
                args.prompts, 
                max_length=args.gen_max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty
            )
    
    # Benchmarking
    if args.benchmark and "signal" in models and "standard" in models:
        print("\n=== Running Benchmarks ===")
        benchmark_results = benchmark_models(
            models["signal"], models["standard"], test_loader, device
        )
    
    # Visualization
    if args.visualize:
        print("\n=== Visualizing Token Representations ===")
        
        # Sample common words for visualization
        common_words = ['the', 'of', 'and', 'a', 'to', 'in', 'that', 'it', 
                      'is', 'was', 'for', 'with', 'as', 'he', 'on']
        
        for model_name, model in models.items():
            # Visualize common words
            save_path = os.path.join(args.output_dir, f"{model_name}_token_frequencies.png")
            visualize_token_frequencies(
                model, common_words, train_dataset.tokenizer, save_path
            )
    
    writer.close()
    print("Training and evaluation complete!")


if __name__ == "__main__":
    main() 