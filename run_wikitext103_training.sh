#!/bin/bash
# run_wikitext103_training.sh - Run WikiText-103 training for SignalLLM

# Activate virtual environment
source signalllm_venv/bin/activate

# Disable tokenizer parallelism to avoid deadlocks
export TOKENIZERS_PARALLELISM=false

# Create output directories
mkdir -p ./wikitext103_signalllm
mkdir -p ./logs

# Remove existing debug directory to prevent disk overload
echo "===== REMOVING EXISTING DEBUG FILES ====="
rm -rf ./wikitext103_signalllm/debug
echo "Debug directory removed to prevent disk overload"

# Timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="./logs/signalllm_training_${TIMESTAMP}.log"

# Smaller debug parameters
VOCAB_SIZE=50000
SEQ_LENGTH=256    # Reduced from 1024
EMBED_DIM=128     # Reduced from 512
NUM_HEADS=4       # Reduced from 8
NUM_LAYERS=2      # Reduced from 8
BATCH_SIZE=2      # Reduced from 8
EPOCHS=2          # Reduced from 50
LR=3e-5
WARMUP_STEPS=100  # Reduced from 2000

# Based on log data, there are approximately 460,158 batches per epoch
# Save only at 50% of epoch and at end of epoch (less frequent checkpointing)
CHECKPOINT_EVERY=230079  # ~50% of an epoch

echo "===== DOWNLOADING WIKITEXT-103 DATASET ====="
# Use huggingface datasets to download wikitext-103
python -c "from datasets import load_dataset; load_dataset('wikitext', 'wikitext-103-v1', cache_dir='./data')"

echo "===== TRAINING SIGNALLLM ON WIKITEXT-103 WITH MPS OPTIMIZATION ====="
echo "Log file: ${LOG_FILE}"
echo "Checkpointing at ~50% of each epoch (every ${CHECKPOINT_EVERY} batches) to save disk space"

# Set a environment variable to disable debug output
export SIGNALLLM_DISABLE_DEBUG=1

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
  --output_dir ./wikitext103_signalllm \
  --num_workers 0 \
  --log_level INFO 2>&1 | tee -a "${LOG_FILE}"

echo "===== TRAINING COMPLETE ====="
echo "SignalLLM WikiText-103 results: ./wikitext103_signalllm"
echo "Full training log saved to: ${LOG_FILE}"

# Add a cleanup script that runs periodically to keep only the latest checkpoint
# Create a cleanup script
cat > cleanup_checkpoints.sh << 'EOL'
#!/bin/bash
# Keep only the latest checkpoint to save disk space
CHECKPOINT_DIR="./wikitext103_signalllm"

# Remove debug directory entirely
rm -rf "$CHECKPOINT_DIR/debug"
echo "Removed debug directory to save disk space"

# Find the latest checkpoint by modification time
LATEST=$(find "$CHECKPOINT_DIR" -name "checkpoint-*" -type d | sort -r | head -n 1)

if [ -n "$LATEST" ]; then
  echo "Keeping latest checkpoint: $LATEST"
  # Remove all checkpoints except the latest one
  find "$CHECKPOINT_DIR" -name "checkpoint-*" -type d | grep -v "$LATEST" | xargs rm -rf
  echo "Removed old checkpoints to save disk space"
fi
EOL

# Make the cleanup script executable
chmod +x cleanup_checkpoints.sh

# Set up a periodic task to run the cleanup script every hour
echo "Setting up periodic cleanup job to run every hour"
(crontab -l 2>/dev/null; echo "0 * * * * cd $(pwd) && ./cleanup_checkpoints.sh") | crontab -

echo "Disk space management enabled: Only the latest checkpoint will be kept and debug files will be removed" 