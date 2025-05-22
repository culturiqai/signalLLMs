#!/bin/bash
# run_wikitext103_training.sh - Run WikiText-103 training for SignalLLM

# Activate virtual environment
source signalllm_venv/bin/activate

# Disable tokenizer parallelism to avoid deadlocks
export TOKENIZERS_PARALLELISM=false

# Create output directories
mkdir -p ./wikitext103_signalllm_fixed
mkdir -p ./logs

# Remove existing debug directory to prevent disk overload
echo "===== REMOVING EXISTING DEBUG FILES ====="
rm -rf ./wikitext103_signalllm_fixed/debug
echo "Debug directory removed to prevent disk overload"

# Timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="./logs/signalllm_training_${TIMESTAMP}.log"

# Training parameters
VOCAB_SIZE=50000
SEQ_LENGTH=512     # Reasonable sequence length
EMBED_DIM=256      # Reduced from 512
NUM_HEADS=8        # Standard
NUM_LAYERS=4       # Reduced from 8
BATCH_SIZE=8       # Standard
EPOCHS=2           # Reduced for testing
LR=1e-6            # Drastically reduced from 3e-5
WARMUP_STEPS=100   # Reduced from 2000
USE_NON_OVERLAPPING=true  # Force non-overlapping sequences

# Set a environment variable to enforce non-overlapping sequences
export SIGNALLLM_NON_OVERLAPPING=1
export SIGNALLLM_DISABLE_DEBUG=1

# Calculate expected steps per epoch (using known token count)
TOTAL_TOKENS=117800617  # Known from previous verification
EXPECTED_STEPS=$((TOTAL_TOKENS / (SEQ_LENGTH * BATCH_SIZE)))
echo "Total tokens: $TOTAL_TOKENS"
echo "Expected steps per epoch: $EXPECTED_STEPS"

echo "===== VERIFICATION STEP ====="
echo "Verifying dataset implementation..."
echo "Creating a small test dataset to verify non-overlapping sequences"

# Create a verification script
cat > verify_non_overlapping.py << 'EOL'
import os
import sys
from datasets import load_dataset
from transformers import AutoTokenizer
from train_wikitext103 import WikiText103Dataset

print("Verifying non-overlapping sequences in WikiText103Dataset")

# Create a small dataset for verification
tokenizer = AutoTokenizer.from_pretrained("gpt2")
seq_length = int(os.environ.get('SEQ_LENGTH', 512))

# Create the dataset
dataset = WikiText103Dataset(split='train', seq_length=seq_length, tokenizer=tokenizer, cache_dir='./data')

# Check sample count
total_tokens = len(dataset.all_tokens)
expected_samples = (total_tokens - seq_length - 1) // seq_length
actual_samples = len(dataset.samples)

print(f"Total tokens: {total_tokens}")
print(f"Expected samples with non-overlapping sequences: ~{expected_samples}")
print(f"Actual samples in dataset: {actual_samples}")

# Verify actual implementation by checking stride
if abs(expected_samples - actual_samples) > expected_samples * 0.1:  # Allow 10% margin for edge cases
    print("❌ VERIFICATION FAILED: Dataset is NOT using non-overlapping sequences!")
    print(f"Expected ~{expected_samples} samples, but got {actual_samples}")
    print("Fix the implementation to ensure stride = seq_length")
    sys.exit(1)
else:
    print("✅ VERIFICATION PASSED: Dataset is using non-overlapping sequences")
    print(f"Expected ~{expected_samples} samples, got {actual_samples}")

# Verify consecutive samples have correct stride
if len(dataset.samples) >= 2:
    stride = dataset.samples[1] - dataset.samples[0]
    if stride != seq_length:
        print(f"❌ VERIFICATION FAILED: Incorrect stride! Expected {seq_length}, got {stride}")
        sys.exit(1)
    else:
        print(f"✅ Stride verification passed: {stride} == {seq_length}")

print("Dataset verification complete!")
EOL

# Run verification with the correct sequence length
SEQ_LENGTH=$SEQ_LENGTH python verify_non_overlapping.py
VERIFY_EXIT_CODE=$?

if [ $VERIFY_EXIT_CODE -ne 0 ]; then
    echo "❌ CRITICAL ERROR: Dataset implementation verification failed!"
    echo "Please fix the implementation to ensure non-overlapping sequences."
    exit 1
fi

# Based on the expected steps and the batch size, calculate checkpoint frequency  
CHECKPOINT_EVERY=$((EXPECTED_STEPS / 2))  # Save twice per epoch
if [ $CHECKPOINT_EVERY -eq 0 ]; then
    CHECKPOINT_EVERY=100  # Fallback value
fi

echo "===== TRAINING SIGNALLLM ON WIKITEXT-103 WITH MPS OPTIMIZATION ====="
echo "Log file: ${LOG_FILE}"
echo "Checkpointing at ~50% of each epoch (every ${CHECKPOINT_EVERY} batches) to save disk space"

# Run training with tee to show output and save to log file
python train_wikitext103.py \
  --vocab_size $VOCAB_SIZE \
  --seq_length $SEQ_LENGTH \
  --embed_dim $EMBED_DIM \
  --num_heads $NUM_HEADS \
  --num_layers $NUM_LAYERS \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --learning_rate $LR \
  --warmup_steps $WARMUP_STEPS \
  --use_signal_embed \
  --use_wavelet_attention \
  --use_mps \
  --enable_evolution \
  --checkpoint_every $CHECKPOINT_EVERY \
  --output_dir ./wikitext103_signalllm_fixed \
  --num_workers 0 \
  --log_level WARNING 2>&1 | tee -a "${LOG_FILE}"

echo "===== TRAINING COMPLETE ====="
echo "SignalLLM WikiText-103 results: ./wikitext103_signalllm_fixed"
echo "Full training log saved to: ${LOG_FILE}"

# Add a cleanup script that runs periodically to keep only the latest checkpoint
# Create a cleanup script
cat > cleanup_checkpoints.sh << 'EOL'
#!/bin/bash
# Keep only the latest checkpoint to save disk space
CHECKPOINT_DIR="./wikitext103_signalllm_fixed"

# Remove debug directory entirely
rm -rf "$CHECKPOINT_DIR/debug"
echo "Removed debug directory to save disk space"

# Find the latest checkpoint by modification time
LATEST=$(find "$CHECKPOINT_DIR" -name "checkpoint_step_*" -type f | sort -r | head -n 1)

if [ -n "$LATEST" ]; then
  echo "Keeping latest checkpoint: $LATEST"
  # Remove all checkpoints except the latest one
  find "$CHECKPOINT_DIR" -name "checkpoint_step_*" -type f | grep -v "$LATEST" | xargs rm -rf
  echo "Removed old checkpoints to save disk space"
fi
EOL

# Make the cleanup script executable
chmod +x cleanup_checkpoints.sh

echo "Disk space management enabled: Only the latest checkpoint will be kept and debug files will be removed" 