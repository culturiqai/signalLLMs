#!/bin/bash
# run_wikitext103_training.sh - Run WikiText-103 training for SignalLLM

# Activate virtual environment
source signalllm_venv/bin/activate

# Disable tokenizer parallelism to avoid deadlocks
export TOKENIZERS_PARALLELISM=false

# Create output directories
mkdir -p ./wikitext103_signalllm
mkdir -p ./logs

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

echo "===== DOWNLOADING WIKITEXT-103 DATASET ====="
# Use huggingface datasets to download wikitext-103
python -c "from datasets import load_dataset; load_dataset('wikitext', 'wikitext-103-v1', cache_dir='./data')"

echo "===== TRAINING SIGNALLLM ON WIKITEXT-103 WITH MPS OPTIMIZATION ====="
echo "Log file: ${LOG_FILE}"

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
  --checkpoint_every 100 \
  --output_dir ./wikitext103_signalllm \
  --num_workers 0 \
  --log_level INFO 2>&1 | tee -a "${LOG_FILE}"

echo "===== TRAINING COMPLETE ====="
echo "SignalLLM WikiText-103 results: ./wikitext103_signalllm"
echo "Full training log saved to: ${LOG_FILE}" 