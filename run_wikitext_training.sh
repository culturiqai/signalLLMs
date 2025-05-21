#!/bin/bash
# run_wikitext_training.sh - Run WikiText training for both model types

# Activate virtual environment
source signalllm_env/bin/activate

# Create output directories
mkdir -p ./wikitext_standard
mkdir -p ./wikitext_signalllm

# Common parameters
VOCAB_SIZE=10000
SEQ_LENGTH=512
EMBED_DIM=256
NUM_HEADS=4
NUM_LAYERS=2
BATCH_SIZE=8
EPOCHS=2
LR=5e-5

echo "===== TRAINING STANDARD TRANSFORMER ====="
python3 train_wikitext.py \
  --vocab_size $VOCAB_SIZE \
  --seq_length $SEQ_LENGTH \
  --embed_dim $EMBED_DIM \
  --num_heads $NUM_HEADS \
  --num_layers $NUM_LAYERS \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --learning_rate $LR \
  --output_dir ./wikitext_standard

echo "===== TRAINING SIGNALLLM WITH MPS OPTIMIZATION ====="
python3 train_wikitext.py \
  --vocab_size $VOCAB_SIZE \
  --seq_length $SEQ_LENGTH \
  --embed_dim $EMBED_DIM \
  --num_heads $NUM_HEADS \
  --num_layers $NUM_LAYERS \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --learning_rate $LR \
  --use_signal_embed \
  --use_wavelet_attention \
  --use_mps \
  --output_dir ./wikitext_signalllm

echo "===== TRAINING COMPLETE ====="
echo "Standard model results: ./wikitext_standard"
echo "SignalLLM results: ./wikitext_signalllm" 